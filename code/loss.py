#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:58:51 2024

@author: center
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import data_denorm
import torchaudio
from utils import word_index
from torchmetrics import PearsonCorrCoef

class NeuroTalkLoss(nn.Module):
    
    def __init__(self, args, models, criterions):
        super(NeuroTalkLoss, self).__init__()
        (
            _,
            _,
            vocoder,
            model_STT,
            decoder_STT,
            modelForCTC
        ) = models
        
        (
            criterion_recon,
            criterion_adv,
            criterion_ctc,
            CER
        ) = criterions
        
        
        # Define models
        vocoder.eval()
        model_STT.eval()
        
        for p in vocoder.parameters():
            p.requires_grad_(False)  # freeze vocoder
        for p in model_STT.parameters():
            p.requires_grad_(False)  # freeze model_STT
            
        # pretrained models
        self.vocoder = vocoder
        self.model_STT = model_STT
        self.decoder_STT = decoder_STT
        self.modelForCTC = modelForCTC
        
        # loss function
        self.criterion_recon = criterion_recon
        self.criterion_adv = criterion_adv
        self.criterion_ctc = criterion_ctc
        self.CER = CER
        
        self.sampling_rate = args.sampling_rate
        self.sample_rate_STT = args.sample_rate_STT
        self.l_g = args.l_g
        self.labels = args.labels
        self.phon = args.phon

    def forward(self, batch, predictions, data_info):
        
        (
            _, 
            target, 
            audio, 
            text
        ) = batch
        
        (
            mel_out, 
            g_valid,
        ) = predictions
        
        # generator loss
        loss_recon = self.criterion_recon(mel_out, target)
        
        # Adversarial ground truths 1:real, 0: fake
        valid = torch.ones((len(target), 1), dtype=torch.float32).cuda()
        
        # GAN loss
        loss_valid = self.criterion_adv(g_valid, valid)
        
        # accuracy    args.l_g = h_g.l_g
        acc_g_valid = (g_valid.round() == valid).float().mean()
                
        # out_DTW
        # target_denorm = data_denorm(target, data_info[0], data_info[1])
        output_denorm = data_denorm(mel_out, data_info[0], data_info[1])

        # recon
        ##### HiFi-GAN
        # print(output_denorm)
        wav_recon = self.vocoder(output_denorm)
        wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
        
        #### resampling
        wav_recon = torchaudio.functional.resample(wav_recon, self.sampling_rate, self.sample_rate_STT)   
        if wav_recon.shape[1] !=  audio.shape[1]:
            p = audio.shape[1] - wav_recon.shape[1]
            p_s = p//2
            p_e = p-p_s
            wav_recon = F.pad(wav_recon, (p_s,p_e))
        
        ##### STT Wav2Vec 2.0  CER
        emission_gt, _ = self.model_STT(audio)
        # emission_tar, _ = self.model_STT(wav_target)
        emission_recon, _ = self.model_STT(wav_recon)
        
        #### CTC LOSS Phoneme
        if self.phon:
            logits_gt = self.modelForCTC(audio).logits
            logits_recon = self.modelForCTC(wav_recon).logits
            
            # take argmax and decode
            ids_gt = torch.argmax(logits_gt, dim=-1)
            ids_gt = torch.unique_consecutive(ids_gt, dim=-1)
            len_gt = torch.sum(ids_gt.bool(),-1)
            
            indices_gt = torch.zeros((len(ids_gt),max(len_gt)), dtype=torch.int64)
            for i, indices in enumerate(ids_gt):
                non_zero_indices = indices[indices != 0]
                indices_gt[i,:len_gt[i]] = non_zero_indices
            recon_lengths = torch.full(size=(logits_recon.size(dim=0),), fill_value=logits_recon.size(dim=1), dtype=torch.long)
            logits_recon_ = logits_recon.log_softmax(2)
            
        else:
            # Character-based
            word_indexs, word_lengths = word_index(text, self.labels)
            word_indexs = torch.as_tensor(word_indexs, dtype=torch.int64)
            word_lengths = torch.as_tensor(word_lengths, dtype=torch.int64)
            
            recon_lengths = torch.full(size=(emission_recon.size(dim=0),), fill_value=emission_recon.size(dim=1), dtype=torch.long)
            logits_recon_ = emission_recon.log_softmax(2)
            indices_gt = word_indexs
            len_gt = word_lengths
            
        # CTC loss
        loss_ctc = self.criterion_ctc(logits_recon_.transpose(0, 1), indices_gt, recon_lengths, len_gt) 
                
        # total generator loss
        loss_g = self.l_g[0] * loss_recon + self.l_g[1] * loss_valid + self.l_g[2] * loss_ctc
        
        # decoder STT
        transcript_gt = []
        # transcript_tar = []
        transcript_recon = []

        for j in range(len(emission_gt)):
            transcript = self.decoder_STT(emission_gt[j])   
            transcript_gt.append(transcript)
                
            # transcript = self.decoder_STT(emission_tar[j])   
            # transcript_tar.append(transcript)
                
            transcript = self.decoder_STT(emission_recon[j])
            transcript_recon.append(transcript)

        cer_gt = self.CER(transcript_gt, text)
        # cer_recon = self.CER(transcript_recon, text)
        cer_recon = self.CER(transcript_recon, text)
        
        e_loss_g = (loss_g.item(), loss_recon.item(), loss_valid.item(), loss_ctc.item())
        e_acc_g = (acc_g_valid.item(), cer_gt.item(), cer_recon.item())
        
        return loss_g, e_loss_g, e_acc_g
    
        
    def forward_D(self, inputs, predictions):
        (
            real_valid,
            fake_valid,
        ) = inputs
                
        (
            valid,
            fake,
        ) = predictions
        
        loss_d_real_valid = self.criterion_adv(real_valid, valid)
        loss_d_fake_valid = self.criterion_adv(fake_valid, fake)
        
        loss_d = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
        
        # accuracy
        acc_d_real = (real_valid.round() == valid).float().mean()
        acc_d_fake = (fake_valid.round() == fake).float().mean()
        
        e_loss_d = (loss_d.item(), loss_d_real_valid.item(), loss_d_fake_valid.item())
        e_acc_d = (acc_d_real.item(), acc_d_fake.item())
        
        return loss_d, e_loss_d, e_acc_d


