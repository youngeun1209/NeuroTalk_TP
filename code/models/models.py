import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding, get_mask_from_lengths
import math

LRELU_SLOPE = 0.1


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1,3,5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,
                       kernel_size, 1, 
                       dilation=dilation[0],                               
                       padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[1],                               
                       padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1,                                
                       dilation=dilation[2],                               
                       padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(channels, channels,                                
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1))),
            weight_norm(
                Conv1d(channels, channels, 
                       kernel_size, 1, 
                       dilation=1,
                       padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
            
            
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.i_mid = 0
        self.i_mid_gru = 1
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_upsample//2,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        
        self.GRU = nn.GRU(h.ch_init_upsample//2, 
                          h.ch_init_upsample//4, 
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        # self.ups = nn.ModuleList()
        # for i, (u, k) in enumerate(zip(h.upsample_rates, 
        #                                h.upsample_kernel_sizes)):
        #     self.ups.append(weight_norm(
        #         ConvTranspose1d(h.ch_init_upsample//(2**i), 
        #                         h.ch_init_upsample//(2**(i+1)),
        #                         k, u, padding=(k-u)//2)))
            
        # self.conv_mid1 = weight_norm(
        #     Conv1d(h.ch_init_upsample//(2**self.i_mid), 
        #            h.ch_init_upsample//(2**self.i_mid), 
        #            3, 1, 
        #            padding=0))
        
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = h.ch_init_upsample#//(2**(i+1))  
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))

        self.conv_post = weight_norm(
            Conv1d(ch, 
                   h.out_ch, 
                   9, 1, 
                   padding=get_padding(9,1)))
        
        self.conv_pre.apply(init_weights)
        # self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        # self.conv_mid1.apply(init_weights)

    def forward(self, x):
        # print(0)

        # in_masks = get_mask_from_lengths(in_len, max_in_len)
        # x = x.masked_fill(in_masks.unsqueeze(-1), 0)
        
        # print("Input: {}".format(x.shape))
        x = self.conv_pre(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)
        # print("GRU: {}".format(x.shape))
        # print(1)
        for i in range(self.num_upsamples):
            # to match the output size
            # if i == self.i_mid:
            #     x = self.conv_mid1(x)
            # x = F.leaky_relu(x, LRELU_SLOPE)
            # print(x.shape)
            # x = self.ups[i](x)
            # print("up: {}".format(x.shape))
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            # print("Resblocks: {}".format(x.shape))
        # print(2)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        # print(3)
        # print("Out: {}".format(x.shape))
        # print("End G")
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        # for l in self.ups:
        #     remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        # remove_weight_norm(self.conv_mid1)


class Discriminator(torch.nn.Module):
    def __init__(self, h, args):
        super(Discriminator, self).__init__()
        self.h = h
        self.ch_init_downsample = h.ch_init_downsample
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.downsample_rates)
        self.m = 1
        self.input_size = int(round(args.win_len * args.sampling_rate / args.hop_length))
        
        for j in range(len(h.downsample_rates)):
            self.m = self.m * h.downsample_rates[j]
        
        # model define
        self.conv_pre = weight_norm(
            Conv1d(h.in_ch, 
                   h.ch_init_downsample,
                   3, 1, 
                   padding=get_padding(3,1)))
        
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.downsample_rates, 
                                       h.downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                Conv1d(h.ch_init_downsample*(2**i), 
                       h.ch_init_downsample*(2**(i+1)),
                       k, u, padding=math.ceil((k-u)/2))))
            
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = h.ch_init_downsample*(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, 
                                           h.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(h, ch, k, d))
        
        self.GRU = nn.GRU(ch, ch//2,
                          num_layers=1, 
                          batch_first=True, 
                          bidirectional=True)
        
        self.conv_post = weight_norm(Conv1d(ch, ch, 9, 1, padding=get_padding(9,1)))
        
        self.ch = ch
        
        # FC Layer 
        self.flatten = nn.Flatten()
        self.adv_classifier = nn.Sequential(
            nn.Linear(h.ch_init_downsample*8*(self.input_size//self.m), 1),
            nn.Sigmoid())
        
        self.conv_pre.apply(init_weights)
        self.downs.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        
        # print(x.shape)
        # print("ch: {}".format(self.ch))
        x = self.conv_pre(x)

        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.downs[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            # print(x.shape)
        x = F.leaky_relu(x)
        x_temp = x
        x = x.transpose(1, 2)
        self.GRU.flatten_parameters()
        x, _ = self.GRU(x)
        x = x.transpose(1, 2)
        x = torch.cat([x, x_temp], dim=1)

        # FC Layer
        x = x.view(-1,
                   self.ch_init_downsample
                   *8*(self.input_size//self.m))
        validity = self.adv_classifier(x)
        
        return validity

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.downs:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
            
        

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

