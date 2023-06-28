# Import necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchshow as ts
import pickle
from model import VariationalAutoencoder
import os
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader
from constants import DATA_FOLDER, HOP_SIZE, SAMPLE_RATE, FRAME_SIZE
from torch.profiler import profile, record_function, ProfilerActivity

with open('data/ProcessedAudioMNIST/DataLoader.pkl', 'rb') as file:
    train_loader = pickle.load(file)

######################
### MODEL TRAINING ###
######################


# Dataloader & Device
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'mps'
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 1
LR_RATE = 3e-4 # 3e-5 kaparthy constant

model = VariationalAutoencoder()
model = model.to(device)
optimiser = optim.AdamW(model.parameters(),lr=LR_RATE)
loss_fn = nn.MSELoss(reduction="mean")


# Training Loop
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch} has started.')
    loop = tqdm(enumerate(train_loader))
    for i, x in loop:
        # Forward pass

        x = x[0].to(device)
        x_reconstructed, mu, sigma = model(x)

        mag = x[:, 0:1, :, :]

        # Compute Loss
        reconstruction_loss = loss_fn(x_reconstructed,mag)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2)-sigma.pow(2))

        # backprop
        loss = reconstruction_loss + kl_div
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loop.set_postfix(loss=loss.item())
