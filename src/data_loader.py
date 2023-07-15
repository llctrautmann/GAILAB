import torch
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torchaudio
import torchshow as ts
import random


class AvianNatureSounds(Dataset):
    def __init__(self, annotation_file=None, root_dir='../',mel_spectrogram = None,mode='wav',max_ms=5000):
        self.annotation_file = pd.read_csv(annotation_file).sort_values('fileName')
        self.root_dir = root_dir
        self.mel_transformation = mel_spectrogram
        self.AmplitudeToDB = torchaudio.transforms.AmplitudeToDB()
        self.mode = mode
        self.max_ms = max_ms


    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        if self.mode == 'wav':
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index]['habitat']
            signal, sr = torchaudio.load(audio_sample_path)

            return (signal, sr), label , audio_sample_path
        
        elif self.mode == 'mel':
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index]['habitat']
            signal, sr = torchaudio.load(audio_sample_path)
            signal = self.mel_transformation(signal)

            return (signal, sr), label
        
        elif self.mode == 'stft': 
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index]['habitat']
            signal, sr = torchaudio.load(audio_sample_path)

            stft = torch.stft(signal, n_fft=1024, hop_length=512, normalized=True, return_complex=True)

            mag = self.AmplitudeToDB(torch.abs(stft))
            phase = torch.angle(stft)

            return torch.cat([mag,phase],dim=0), label
        
        elif self.mode == 'testing':
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index]['fileName']
            signal, sr = self.pad_trunc(torchaudio.load(audio_sample_path), max_ms=self.max_ms)

            stft = torch.stft(signal, n_fft=56, hop_length=16, normalized=True, return_complex=True)

            mag = self.AmplitudeToDB(torch.abs(stft))
            phase = torch.angle(stft)

            return torch.cat([mag,phase],dim=0), label



        else:
            raise NotImplementedError
    
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
            
        return (sig, sr)
