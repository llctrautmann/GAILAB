import torch
import numpy as np
from IPython.display import Audio

def play_tensor(wave: torch.tensor,sr=44100):
    numpy_waveform = wave.numpy()
    return Audio(numpy_waveform, rate=sr)

play_tensor(wav,sr=16000)