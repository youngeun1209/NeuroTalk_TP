import numpy as np
from librosa.filters import mel as librosa_mel_fn
import torch
import torchaudio
import torch.nn.functional as F
import math

def eeg_fea_extractor(eeg_data, eeg_sr, data_size,
                hop_size=256, win_size=1024, sampling_rate=22050, 
                prep=True, line_noise=50, skip_stacking=True, overlapping=True, segment_flag=True,
                new_seg=8192):

    windowLength = win_size / sampling_rate # ~0.46s
    frameshift = hop_size / sampling_rate # ~0.11s
    overlap = int(windowLength * eeg_sr - frameshift * eeg_sr) # 71 / 34
        

    # Initialize filling zeros (since online method uses a warm start)
    fill_zeros = np.zeros((math.ceil(overlap/2), eeg_data.shape[1]))
    fill_zeros_ = np.zeros((overlap - math.ceil(overlap/2), eeg_data.shape[1]))
    data = np.vstack([fill_zeros, eeg_data, fill_zeros_])
    
    # numWindows = int(np.floor((data.shape[0] - windowLength * eeg_sr) / (frameshift * eeg_sr)))
    # numWindows += 1
    
    numWindows = data_size
        
    # Compute logarithmic high gamma broadband features
    eeg_features = np.zeros((data.shape[1],numWindows,))
    for win in range(numWindows):
        start_eeg = int(round((win * frameshift) * eeg_sr))
        stop_eeg = int(round(start_eeg + windowLength * eeg_sr))
        # print('{} {}'.format(start_eeg, stop_eeg))
        for c in range(data.shape[1]):
            eeg_features[c, win] = np.log(np.sum(data[start_eeg:stop_eeg, c] ** 2) + 0.01)
                
    return eeg_features



def mel_stft(audio_data, audio_sr, data_length,
                        n_fft=1024,  num_mels=80,
                        hop_size=256, win_size=1024, sampling_rate=22050,  
                        fmin=0, fmax=8000):

    # resample
    audio_ = torch.from_numpy(audio_data)
    
    y_voice = torchaudio.functional.resample(audio_, 
                                             audio_sr, 
                                             sampling_rate)   


    # STFT
    hann_window = torch.hann_window(win_size)
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,n_mels=num_mels,
                                fmin=fmin,fmax=fmax)

    mel_basis = torch.from_numpy(mel_basis).float()

    p = (n_fft - hop_size) // 2 # voice: 256 , EEG:64 #(n_fft - hop_length) // 2
    y = F.pad(y_voice, (p, p))
    

    # shape: [25873, 8960]

    spec = torch.stft(y, 
                      n_fft, 
                      hop_length=hop_size,
                      win_length=win_size,
                      window=hann_window,
                      center=False)
            
    magnitude = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    
    spectrogram = torch.matmul(mel_basis, magnitude)

    spectrogram = torch.log(torch.clamp(spectrogram, min=1e-5))
    spectrogram = np.array(spectrogram).astype(np.float64)
    
    y_voice = np.array(y_voice).astype(np.float64)

    
    return y_voice, spectrogram



