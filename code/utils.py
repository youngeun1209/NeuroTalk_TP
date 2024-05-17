 
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import glob
from torch.nn.utils import weight_norm

def audio_denorm(data):
    max_audio = 32768.0
    
    data = np.array(data * max_audio).astype(np.float32)
       
    return data
    
def imgSave(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.tight_layout()
    plt.savefig(dir + file_name)
    plt.clf()


def data_denorm(data, avg, std):
 
    avg = torch.as_tensor(avg)
    std = torch.as_tensor(std)
    
    data = data * std + avg
    
    data = data.float()
    
    
    # data = data.detach().cpu().numpy()
    # data = data * std + avg
    # data = torch.from_numpy(data).float().cuda()

    return data



def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()

    return fig
    
def imgSave(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.tight_layout()
    plt.savefig(dir + file_name)
    plt.clf()


def word_index_bundle(word_label, bundle):
    labels_ = ''.join(list(bundle.get_labels()))
    word_indices = np.zeros((len(word_label), 100), dtype=np.int64)
    word_length = np.zeros((len(word_label), ), dtype=np.int64)
    for w in range(len(word_label)):
        word = word_label[w]
        label_idx = []
        for ww in range(len(word)):
            label_idx.append(labels_.find(word[ww]))
        word_indices[w,:len(label_idx)] = torch.tensor(label_idx)
        word_length[w] = len(label_idx)
        
    return word_indices, word_length

def word_index(word_label, labels):
    labels_ = ''.join(labels)
    word_indices = np.zeros((len(word_label), 100), dtype=np.int64)
    word_length = np.zeros((len(word_label), ), dtype=np.int64)
    for w in range(len(word_label)):
        word = word_label[w]
        label_idx = []
        for ww in range(len(word)):
            label_idx.append(labels_.find(word[ww]))
        word_indices[w,:len(label_idx)] = torch.tensor(label_idx)
        word_length[w] = len(label_idx)
        
    return word_indices, word_length

def data_regularization(data, avg=None, std=None):
    
    max_ = np.max(data).astype(np.float32)
    min_ = np.min(data).astype(np.float32)
    
    if avg == None:
        avg = (max_ + min_) / 2
    if std == None:
        std = (max_ - min_) / 2
    
    data   = np.array((data - avg) / std).astype(np.float32)
    
    return data, avg, std

def data_regularization_sen(data, avg=None, std=None):
    
    max_append, min_append = [], []
    for i in range(len(data)):
        max_ = np.max(data[i]).astype(np.float32)
        min_ = np.min(data[i]).astype(np.float32)
        max_append.append(max_)
        min_append.append(min_)
    
    max_ = np.max(max_append).astype(np.float32)
    min_ = np.min(min_append).astype(np.float32)
    
    if avg == None:
        avg = (max_ + min_) / 2
    if std == None:
        std = (max_ - min_) / 2
    
    data_new = []
    for i in range(len(data)):
        data_reg = np.array((data[i] - avg) / std).astype(np.float32)
        data_new.append(data_reg)
    
    return data_new, avg, std

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

######################################################################
############                  HiFiGAN                   ##############
######################################################################
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


# %% masking
# def get_mask_from_lengths(lengths, max_len=None):
#     batch_size = lengths.shape[0]
#     if max_len is None:
#         max_len = torch.max(lengths).item()

#     ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()
#     mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

#     return mask

def to_device(data):
    if len(data) == 4:
        (
            input, 
            target,  
            audio, 
            text, 
        ) = data

        input = input.cuda()
        target = target.cuda()
        audio = audio.cuda()

        return (
            input, 
            target,  
            audio,  
            text
        )
    
# def to_device(data):
#     if len(data) == 12:
#         (
#             input, 
#             in_len, 
#             max_in_len, 
#             target, 
#             mel_lens, 
#             max_mel_len, 
#             audio, 
#             aud_lens, 
#             max_aud_len, 
#             text, 
#             txt_len, 
#             max_txt_len
#         ) = data

#         input = input.cuda()
#         in_len = in_len.cuda()
#         target = target.cuda()
#         mel_lens = mel_lens.cuda()
#         audio = audio.cuda()
#         aud_lens = aud_lens.cuda()

#         return (
#             input, 
#             in_len, 
#             max_in_len, 
#             target, 
#             mel_lens, 
#             max_mel_len, 
#             audio, 
#             aud_lens, 
#             max_aud_len, 
#             text, 
#             txt_len, 
#             max_txt_len
#         )