import os
import numpy as np
import torch
import librosa
import pickle
from torch.utils.data import TensorDataset, DataLoader
from constants import DATA_FOLDER, HOP_SIZE, SAMPLE_RATE, FRAME_SIZE

# Retrieve the list of audio files in the data folder
audio_files = [file for file in os.listdir(DATA_FOLDER) if file.endswith('.wav')]

tensors = []

for i in range(0, 320):
    file = audio_files[i]
    file, _ = librosa.load(f'/Users/luca/Desktop/ML/AutoencoderML/data/AudioMNIST/{file}')

    stft = librosa.stft(file,n_fft=FRAME_SIZE,hop_length=HOP_SIZE, window='hann')

    if stft.shape[1] < 400:
        continue

    else:
        magnitude = librosa.power_to_db(np.abs(stft[:,0:400]))
        phase = np.angle(stft[:,0:400])

        mag_tensor = torch.tensor(magnitude).unsqueeze(0)
        phase_tensor = torch.tensor(phase).unsqueeze(0)
        
        t = torch.cat((mag_tensor, phase_tensor), dim=0)

        tensors.append(t.unsqueeze(0))
        
# Convert the list of tensors into a single tensor
images_tensor = torch.cat(tensors, dim=0)

# Create a TensorDataset from the image tensor
dataset = TensorDataset(images_tensor)

# Define batch size and other DataLoader parameters
batch_size = 4

# Create a data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
with open('data/ProcessedAudioMNIST/DataLoader.pkl', 'wb') as file:
    pickle.dump(data_loader, file)

print(len(data_loader))