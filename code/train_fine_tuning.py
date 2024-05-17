import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import GreedyCTCDecoder, AttrDict, ConstrainedLoss, save_checkpoint
from modules import mel2wav_vocoder, perform_STT
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
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt
import librosa.display
from loss import NeuroTalkLoss
from transformers import Wav2Vec2ForCTC
from initialization import initialization
from train import train

logger = logging.getLogger('Fine_Tuning.py')


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
    criterion_recon = ConstrainedLoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    criterion_ctc = nn.CTCLoss().cuda()
    CER = CharErrorRate().cuda()
    
    # optimizer
    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)


    # Load trained model
    start_epoch = 0
    
    if args.pretrain:
        if args.trained_model == None:
            args.trained_model = os.path.join(args.logDir, args.sub, args.logName, 'pretrain')
            loc_g = os.path.join(args.trained_model, 'BEST_checkpoint_g.pt')
            loc_d = os.path.join(args.trained_model, 'BEST_checkpoint_d.pt')
        else:    
            loc_g = os.path.join(args.trained_model, 'checkpoint_g.pt')
            loc_d = os.path.join(args.trained_model, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            args.logger.info("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            optimizer_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            args.logger.info("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
            optimizer_d.load_state_dict(checkpoint_d['optimizer_state_dict'])
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(loc_d))


    if args.resume:
        loc_g = os.path.join(args.savemodel, 'checkpoint_g.pt')
        loc_d = os.path.join(args.savemodel, 'checkpoint_d.pt')

        if os.path.isfile(loc_g):
            args.logger.info("=> loading checkpoint '{}'".format(loc_g))
            checkpoint_g = torch.load(loc_g, map_location='cpu')
            model_g.load_state_dict(checkpoint_g['state_dict'])
            start_epoch = checkpoint_g['epoch'] + 1
            best_loss = checkpoint_g['best_loss']
            optimizer_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(loc_g))

        if os.path.isfile(loc_d):
            args.logger.info("=> loading checkpoint '{}'".format(loc_d))
            checkpoint_d = torch.load(loc_d, map_location='cpu')
            model_d.load_state_dict(checkpoint_d['state_dict'])
            optimizer_d.load_state_dict(checkpoint_d['optimizer_state_dict'])
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(loc_d))


    # Tensorboard setting
    args.writer = SummaryWriter(args.logs)

    # dataset
    if args.dataset == 'EEG_word':
        from Dataset_EEG_Word import myDataset
    elif args.dataset == 'EEG_sen':
        from Dataset_EEG_Sen import myDataset
    elif args.dataset == 'sEEG':
        from Dataset_sEEG import myDataset

    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)

    trainset = myDataset(args,
                         mode=0, 
                         data=os.path.join(args.dataDir,args.sub), 
                         task=args.task, 
                         recon=args.recon)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate_fn,
        generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)
    
    valset = myDataset(args,
                       mode=2, 
                       data=os.path.join(args.dataDir,args.sub), 
                       task=args.task, 
                       recon=args.recon)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, collate_fn=valset.collate_fn,
        generator=generator, num_workers=4*len(args.gpuNum), pin_memory=True)


    # data info
    file_name = os.path.join(args.dataDir, args.sub, 'train', 'data_info.npy')
    data_info = np.load(file_name)
    
    # Define
    models = (model_g, model_d, vocoder, model_STT, decoder_STT, modelForCTC)
    criterions = (criterion_recon, criterion_adv, criterion_ctc, CER)
    optimizers = (optimizer_g, optimizer_d)
    
    Loss = NeuroTalkLoss(args, models, criterions)
    
    epoch = start_epoch
    lr_g = 0
    lr_d = 0
    best_loss = 1000
    is_best = False
    epochs_since_improvement = 0
    trainValid = True
    for epoch in range(start_epoch, args.max_epochs):
        
        start_time = time.time()
        
        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']
        for param_group in optimizer_d.param_groups:
            lr_d = param_group['lr']

        scheduler_g.step(epoch)
        scheduler_d.step(epoch)

        args.logger.info("Epoch : %d/%d" %(epoch, args.max_epochs) )
        args.logger.info("Learning rate for G: %.9f" %lr_g)
        args.logger.info("Learning rate for D: %.9f" %lr_d)
        
        
        Tr_losses = train(args, train_loader, data_info,
                          models, 
                          Loss, 
                          optimizers, 
                          epoch,
                          True) 
        
        Val_losses = train(args, val_loader, data_info,
                            models, 
                            Loss, 
                            ([],[]), 
                            epoch,
                            False)
        
        # Did validation loss improve?
        loss_total =  Val_losses[0] # CER loss
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)
        
        # Save checkpoint
        state_g = {'arch': str(model_g),
                 'state_dict': model_g.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_g.state_dict(),
                 'best_loss': best_loss}
        
        state_d = {'arch': str(model_d),
                 'state_dict': model_d.state_dict(),
                 'epoch': epoch,
                 'optimizer_state_dict': optimizer_d.state_dict()}
        
        save_checkpoint(state_g, is_best, args.savemodel, 'checkpoint_g.pt')
        save_checkpoint(state_d, is_best, args.savemodel, 'checkpoint_d.pt')

        
        if not is_best:
            epochs_since_improvement += 1
            args.logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        time_taken = time.time() - start_time
        args.logger.info("Time: %.2f\n"%time_taken)
        
    args.writer.flush()

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
    
    
    
