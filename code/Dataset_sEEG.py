import csv
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math

epsilon = np.finfo(float).eps

class myDataset(Dataset):
    def __init__(self, args, mode, data="./", task = "SpokenEEG", recon="Y_mel"):
        self.sample_rate = 8000
        self.n_classes = 13
        self.mode = mode
        self.iter = iter
        self.savedata = data
        self.task = task
        self.recon = recon
        self.max_audio = 32768.0
        self.lenth = len(os.listdir(self.savedata + '/train/audio/')) #780 # the number data
        self.lenthtest = len(os.listdir(self.savedata + '/test/audio/')) #260
        self.lenthval = len(os.listdir(self.savedata + '/val/audio/')) #260
        self.drop_last = False
        self.batch_size = args.batch_size

    def __len__(self):
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''

        if self.mode == 2:
            forder_name = os.path.join(self.savedata, 'val')
        elif self.mode == 1:
            forder_name = os.path.join(self.savedata, 'test')
        else:
            forder_name = os.path.join(self.savedata, 'train')
            
        if (self.task[0] == 'S') | (self.task[0] == 's'): # Spoken
            forder_eeg = 'eeg_fea_sp'
        elif (self.task[0] == 'W') | (self.task[0] == 'w'): # Whisper
            forder_eeg = 'eeg_fea_wh'
        elif (self.task[0] == 'I') | (self.task[0] == 'i'): # Imagined
            forder_eeg = 'eeg_fea_im'
        else:
            raise NameError('Task name should be correct')
            
        # audio
        allFileList = os.listdir(os.path.join(forder_name, 'audio'))
        allFileList.sort()
        file_name = os.path.join(forder_name, 'audio', allFileList[idx])
        
        audio = np.load(file_name)
        
            
        # eeg_fea
        allFileList = os.listdir(os.path.join(forder_name, forder_eeg))
        allFileList.sort()
        file_name = os.path.join(forder_name, forder_eeg, allFileList[idx])
        eeg = np.load(file_name)
        
        # print("eeg shape: {}".format(eeg.shape))
        
        # mel
        allFileList = os.listdir(os.path.join(forder_name, 'mel'))
        allFileList.sort()
        file_name = os.path.join(forder_name, 'mel', allFileList[idx])
        mel = np.load(file_name)
        
        # text
        allFileList = os.listdir(os.path.join(forder_name, 'text'))
        allFileList.sort()
        file_name = os.path.join(forder_name, 'text', allFileList[idx])
        text = np.load(file_name)

        # to tensor
        audio = torch.as_tensor(audio, dtype=torch.float32)
        eeg = torch.as_tensor(eeg, dtype=torch.float32)
        mel = torch.as_tensor(mel, dtype=torch.float32)
        
        data = {
            "eeg": eeg,
            "mel": mel,
            "audio": audio,
            "text": text}

        return data

    
    
    def reprocess(self, data, idxs):
        
        eegs, mels, audios, texts = [],[],[],[]
        # eeg_lens, mel_lens, audio_lens, text_lens = [],[],[],[]
        
        for i in idxs:
            eeg = data[i]["eeg"]
            mel = data[i]["mel"]
            audio = data[i]["audio"]
            text = str(data[i]["text"])
            
            eegs.append(eeg)
            mels.append(mel)
            audios.append(audio)
            texts.append(text)
        
        eegs = np.stack(eegs) 
        mels = np.stack(mels) 
        audios = np.stack(audios) 
        texts = np.stack(texts)
        
        # eeg_lens = np.stack(eeg_lens) 
        # mel_lens = np.stack(mel_lens) 
        # audio_lens = np.stack(audio_lens) 
        # text_lens = np.stack(text_lens)
        
        
        # to tensor
        eegs = torch.as_tensor(eegs, dtype=torch.float32)
        mels = torch.as_tensor(mels, dtype=torch.float32)
        audios = torch.as_tensor(audios, dtype=torch.float32)
   
        return (
            eegs,
            mels,
            audios,
            texts
        )


    def collate_fn(self, data):
        data_size = len(data)

        idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

def select_len(x, win_size, rand_i):
    
    if len(x) > win_size:
        x = x[rand_i-win_size : rand_i]
        
    if len(x) < win_size:
        if len(x.shape) == 2:
            x = pad_data_2d(x, win_size)
        if len(x.shape) < 2 :
            x = pad_data_1d(x, win_size)
    return x


def pad_1D(inputs, PAD=0):

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data_1d(x, max_len, PAD) for x in inputs])

    return padded


def pad_data_1d(x, length, PAD=0):
    x_padded = np.pad(
        x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
    )
    return x_padded


def pad_2D(inputs, maxlen=None):

    if maxlen:
        output = np.stack([pad_data_2d(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad_data_2d(x, max_len) for x in inputs])

    return output


def pad_data_2d(x, max_len):
    PAD = 0
    if np.shape(x)[0] > max_len:
        raise ValueError("not max_len")
    
    # print("x length: {}".format(np.shape(x)[0]))
    # print("max length: {}".format(max_len))
    
    s = np.shape(x)[1]
    x_padded = np.pad(
        x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
    )
    
    # print("paded length: {}".format(np.shape(x_padded)[0]))
    return x_padded[:, :s]

