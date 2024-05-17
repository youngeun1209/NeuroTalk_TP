import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import DTW_align, AttrDict, RMSELoss
from modules import GreedyCTCDecoder, perform_STT
from utils import data_denorm, imgSave, to_device
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate
import json
import argparse
import wavio
import logging
import matplotlib.pyplot as plt
import librosa.display
from loss import NeuroTalkLoss
from transformers import Wav2Vec2ForCTC
from train import train_G, train_D
from initialization import initialization

logger = logging.getLogger('inference.py')


def train(args, train_loader, data_info,
          models, Loss, optimizers, epoch, trainValid=True, save=True, mode = 'train'):
    
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    
    (optimizer_g, optimizer_d) = optimizers
    
    # switch to train mode
    assert type(models) == tuple, "More than two models should be inputed (generator and discriminator)"
    
    total_batches = len(train_loader)
    
    # Initialization
    epoch_loss_g = []
    epoch_loss_d = []
    
    epoch_acc_g = []
    epoch_acc_d = []
    
    # batchs = next(iter(train_loader))
    for i, batchs in enumerate(train_loader):    
        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        # start_time = time.time()
        batch = batchs[0]
        
        batch = to_device(batch)

        # train generator
        mel_out, e_loss_g, e_acc_g = train_G(args, 
                                             batch,
                                             models, Loss, optimizer_g, 
                                             data_info, 
                                             trainValid)
        epoch_loss_g.append(e_loss_g)
        epoch_acc_g.append(e_acc_g)
    
        # train discriminator
        e_loss_d, e_acc_d = train_D(args, 
                                    mel_out, batch,
                                    models, Loss, optimizer_d, 
                                    trainValid)
        epoch_loss_d.append(e_loss_d)
        epoch_acc_d.append(e_acc_d)
        
        
        if save==True:
            saveAllData(args, batch, data_info, models, epoch, mode)
        
        # time_taken = time.time() - start_time
        # args.logger.info("Time: %.2f\n"%time_taken)

    epoch_loss_g = np.array(epoch_loss_g)
    epoch_acc_g = np.array(epoch_acc_g)
    epoch_loss_d = np.array(epoch_loss_d)
    epoch_acc_d = np.array(epoch_acc_d)
    
    
    args.loss_g = sum(epoch_loss_g[:,0]) / len(epoch_loss_g[:,0])
    args.loss_g_recon = sum(epoch_loss_g[:,1]) / len(epoch_loss_g[:,1])
    args.loss_g_valid = sum(epoch_loss_g[:,2]) / len(epoch_loss_g[:,2])
    args.loss_g_ctc = sum(epoch_loss_g[:,3]) / len(epoch_loss_g[:,3])
    args.acc_g_valid = sum(epoch_acc_g[:,0]) / len(epoch_acc_g[:,0])
    args.cer_gt = sum(epoch_acc_g[:,1]) / len(epoch_acc_g[:,1])
    args.cer_recon = sum(epoch_acc_g[:,2]) / len(epoch_acc_g[:,2])
    
    args.loss_d = sum(epoch_loss_d[:,0]) / len(epoch_loss_d[:,0])
    args.loss_d_valid = sum(epoch_loss_d[:,1]) / len(epoch_loss_d[:,1])
    args.acc_d_real = sum(epoch_acc_d[:,0]) / len(epoch_acc_d[:,0])
    args.acc_d_fake = sum(epoch_acc_d[:,1]) / len(epoch_acc_d[:,1])
    
    

    args.logger.info('\n[%3d/%3d] CER-gt: %.4f CER-recon: %.4f / ACC_R: %.4f ACC_F: %.4f / g-RMSE: %.4f g-lossValid: %.4f g-lossCTC: %.4f' 
          % (i, total_batches, 
             args.cer_gt, args.cer_recon, 
             args.acc_d_real, args.acc_d_fake, 
             args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc))
        
        
    return (args.cer_recon, args.loss_g, args.loss_g_recon, args.loss_g_valid, args.loss_g_ctc, args.acc_g_valid, 
            args.loss_d, args.acc_d_real, args.acc_d_fake)


def saveAllData(args, batch, data_info, models, epoch, mode):
    
    model_g = models[0].eval()
    # model_d = models[1].eval()
    vocoder = models[2].eval()
    model_STT = models[3].eval()
    decoder_STT = models[4]

    (
        input, 
        target, 
        audio, 
        text
        
    ) = batch


    with torch.no_grad():
        # run the mdoel
        output = model_g(input)
        
        
    mel_out = DTW_align(output, target)
    output_denorm = data_denorm(mel_out, data_info[0], data_info[1])
    target_denorm = data_denorm(target, data_info[0], data_info[1])
    
    wav_recon = vocoder(output_denorm.cuda())
    wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
    
    wav_recon = torchaudio.functional.resample(wav_recon, args.sampling_rate, args.sample_rate_STT)  
    if wav_recon.shape[1] !=  audio.shape[1]:
        p = audio.shape[1] - wav_recon.shape[1]
        p_s = p//2
        p_e = p-p_s
        wav_recon = F.pad(wav_recon, (p_s,p_e))
        
    ##### STT Wav2Vec 2.0
    transcript_recon = perform_STT(wav_recon, model_STT, decoder_STT, None, 1)
    
    wav_recon = wav_recon.cpu().detach().numpy()
    
    # save
    
    for i in range(len(wav_recon)):
        str_tar = text[i].replace("|", ",")
        str_tar = str_tar.replace(" ", ",")
        
        str_pred = transcript_recon[i].replace("|", ",")
        str_pred = str_pred.replace(" ", ",")
        
        title = "Tar_{}-Pred_{}".format(str_tar, str_pred)
        wavio.write(args.savevoice + '/' + mode +'/e{}_{}.wav'.format(str(str(epoch)), title), np.squeeze(wav_recon[i]), args.sample_rate_STT, sampwidth=1)
    
    
        # save mel
        fig, (ax1, ax2) = plt.subplots(2,1)
        img = librosa.display.specshow(target_denorm[i].cpu().detach().numpy(), x_axis='time',
                                  y_axis='mel', sr=args.sampling_rate,
                                  hop_length=256,
                                  ax=ax1)
        fig.colorbar(img, ax=ax1, format='%+2.0f dB')
        ax1.set(title='Original Mel-spectogram')
        
        img = librosa.display.specshow(output_denorm[i].cpu().detach().numpy(), x_axis='time',
                                  y_axis='mel', sr=args.sampling_rate,
                                  hop_length=256,
                                  ax=ax2)
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        ax2.set(title='Reconstructed Mel-spectogram')
        
        # caption="Target: {} Pred: {}".format(args.classname[labels[0].item()],args.classname[preds[0].item()])
    
        imgSave(args.savefig, '/' + mode +'/e{}_{}.png'.format(str(str(epoch)),title))


def main(args):
    
    args.dataDir = os.path.join(args.dataDir, args.dataset, 'preprocess')
    args.logDir = os.path.join(args.logDir, args.dataset)
    
    args.gpuNum = list(range(torch.cuda.device_count()))

    initialization(args)
    
    # define generator
    config_file = os.path.join(args.model_config, 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_g = AttrDict(json_config)
    h_g.in_ch = args.in_ch
    model_g = networks.Generator(h_g).cuda()
    args.logger.info('Config of generator: {}'.format(h_g))
    
    # define discriminator
    config_file = os.path.join(args.model_config, 'config_d.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)
    model_d = networks.Discriminator(h_d,args).cuda()
    args.logger.info('Config of Discriminator: {}'.format(h_d))
    
    # vocoder HiFiGAN
    # LJ_FT_T2_V3/generator_v3,   
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = model_HiFi(h).cuda()
    state_dict_g = torch.load(args.vocoder_pre) #, map_location=args.device)
    vocoder.load_state_dict(state_dict_g['generator'])
    args.logger.info('Config of vocoder HiFiGAN: {}'.format(h))

    # STT Wav2Vec
    args.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE #VOXPOPULI_ASR_BASE_10K_DE
    model_STT = args.bundle.get_model().cuda()
    args.sample_rate_STT = args.bundle.sample_rate
    args.labels=list(args.bundle.get_labels())
    decoder_STT = GreedyCTCDecoder(labels=args.bundle.get_labels())
    args.logger.info('Config of STT: {}'.format(args.bundle._path))
    
    # phoneme
    modelForCTC = Wav2Vec2ForCTC.from_pretrained("bookbot/wav2vec2-ljspeech-gruut").cuda() # 45
    
    # args info
    args.logger.info('General Config: {}'.format(args))
    
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.gpuNum)
    model_d = nn.DataParallel(model_d, device_ids=args.gpuNum)
    vocoder = nn.DataParallel(vocoder, device_ids=args.gpuNum)
    model_STT = nn.DataParallel(model_STT, device_ids=args.gpuNum)
    modelForCTC = nn.DataParallel(modelForCTC, device_ids=args.gpuNum)

    # loss function
    criterion_recon = RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_ctc = nn.CTCLoss().cuda()
    CER = CharErrorRate().cuda()
    
    # Load trained model
    loc_g = os.path.join(args.dataDir, args.sub, 'savemodel', 'BEST_checkpoint_g.pt')
    loc_d = os.path.join(args.dataDir, args.sub, 'savemodel', 'BEST_checkpoint_d.pt')

    if os.path.isfile(loc_g):
        args.logger.info("=> loading checkpoint '{}'".format(loc_g))
        checkpoint_g = torch.load(loc_g, map_location='cpu')
        model_g.load_state_dict(checkpoint_g['state_dict'])
        start_epoch = checkpoint_g['epoch']
    else:
        args.logger.info("=> no checkpoint found at '{}'".format(loc_g))
        exit(0)

    if os.path.isfile(loc_d):
        args.logger.info("=> loading checkpoint '{}'".format(loc_d))
        checkpoint_d = torch.load(loc_d, map_location='cpu')
        model_d.load_state_dict(checkpoint_d['state_dict'])
    else:
        args.logger.info("=> no checkpoint found at '{}'".format(loc_d))
    
    # dataset
    if args.dataset == 'EEG_word':
        from Dataset_EEG_Word import myDataset
    elif args.dataset == 'EEG_sen':
        from Dataset_EEG_Sen import myDataset
    elif args.dataset == 'sEEG':
        from Dataset_sEEG import myDataset

    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)


    testset = myDataset(args,
                       mode=1, 
                       data=os.path.join(args.dataDir,args.sub), 
                       task=args.task, 
                       recon=args.recon)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, collate_fn=testset.collate_fn,
        generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)


    # data info
    file_name = os.path.join(args.dataDir, args.sub, 'test', 'data_info.npy')
    data_info = np.load(file_name)
    
    # Define
    models = (model_g, model_d, vocoder, model_STT, decoder_STT, modelForCTC)
    criterions = (criterion_recon, criterion_adv, criterion_ctc, CER)
    
    Loss = NeuroTalkLoss(args, models, criterions)
    
    epoch = start_epoch
            
    start_time = time.time()

    
    Test_losses = train(args, test_loader, data_info,
                        models, 
                        Loss, 
                        ([],[]), 
                        epoch,
                        False, save=False, mode='test')
    
    
    
    
    time_taken = time.time() - start_time
    args.logger.info("Time: %.2f\n"%time_taken)
        

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--trained_model', type=str, default=None, help='trained model for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--prefreeze', type=bool, default=False)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    
    main(args)        
    
    
    
