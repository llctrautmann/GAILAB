import torch
import librosa
import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torchshow as ts
import random


class AvianNatureSounds(Dataset):
    def __init__(self,
                # file args
                annotation_file_path=None,
                root_dir='../',
                key='habitat',

                # sound transformation args
                mode='stft',
                length=5,
                sampling_rate=44100,
                n_fft=1024,
                hop_length=512,
                mel_spectrogram = None,
                verbose=False,
                fixed_limit=False
                ):
        
        self.column = key
        self.annotation_file = pd.read_csv(annotation_file_path).sort_values(self.column)
        self.root_dir = root_dir
        self.mel_transformation = mel_spectrogram
        self.AmplitudeToDB = torchaudio.transforms.AmplitudeToDB()
        self.mode = mode
        self.length = length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        self.fixed_limit = fixed_limit
        self.return_signal_dims()

    def return_signal_dims(self):
        if self.verbose:
            print(f'Returning signal dimensions: H = {self.n_fft // 2 + 1} W = {((self.sampling_rate * self.length)//self.hop_length) + 1}')
        else:
            pass


    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, index):
        ####################################################################################
        # Get stft spectrograms
        ####################################################################################
        if self.mode == 'stft':
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index][self.column]
            signal, sr = torchaudio.load(audio_sample_path)

            signal = self.clip(signal, sr, self.length,fixed_limit=self.fixed_limit)
            stft = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length,win_length=self.n_fft, normalized=False, return_complex=True)

            # librosa_stft = librosa.stft(signal.numpy(), n_fft=self.n_fft, hop_length=self.hop_length)
            stft = stft[1:][:]
            mag = self.AmplitudeToDB(torch.abs(stft))
            phase = torch.angle(stft)

            mag, phase = mag.unsqueeze(0), phase.unsqueeze(0)


            return torch.cat([mag,phase],dim=0), mag, label



        ####################################################################################
        # Get wav files
        ####################################################################################
        elif self.mode == 'wav':
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index][self.column]
            signal, sr = torchaudio.load(audio_sample_path)

            return (signal, sr), label , audio_sample_path
        

        ####################################################################################
        # Get mel spectrograms
        ####################################################################################
        elif self.mode == 'mel':
            audio_sample_path = os.path.join(self.root_dir,os.listdir(self.root_dir)[index])
            label = self.annotation_file.iloc[index][self.column]
            signal, sr = torchaudio.load(audio_sample_path)
            signal = self.mel_transformation(signal)

            return (signal, sr), label     
          

        else:
            raise NotImplementedError
    

    @staticmethod
    def clip(audio_signal, sr, desired_length,fixed_limit=False):
        num_rows, sig_len = audio_signal.shape
        length = int(sr * desired_length)

        if fixed_limit:
            sig = audio_signal[0][:262100]
            return  sig

        elif sig_len > length:
            offset = random.randint(0, sig_len - length)
            sig = audio_signal[:, offset:(offset+length)]

            return sig
        else:
            return audio_signal




