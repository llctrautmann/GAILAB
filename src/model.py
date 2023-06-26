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
from blocks import DenseBlock, TransitionDown, TransitionUp, TransitionFinal, LinearCompressor, TemporalBlock



# Alpha Model

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        # Encoder Blocks
        super(VariationalAutoencoder, self).__init__()
        self.block2 = DenseBlock(in_channels=2)
        self.block34 = DenseBlock(in_channels=34)
        self.block100 = DenseBlock(in_channels=100)
        self.block232 = DenseBlock(in_channels=232)

        self.transdown34_2 = TransitionDown(in_channels=34,stride=2)
        self.transdown34_4 = TransitionDown(in_channels=34,stride=4)
        self.transdown34_8 = TransitionDown(in_channels=34,stride=8)
        self.transdown66_2 = TransitionDown(in_channels=66,stride=2)
        self.transdown66_4 = TransitionDown(in_channels=66,stride=4)
        self.transdown132_2 = TransitionDown(in_channels=132,stride=2)

        self.transfinal = TransitionFinal(in_channels=264)


        self.linearCompressor = LinearCompressor(65,32)

        # Decoder Blocks
        ## Linear Block
        self.linearCompressorUp = LinearCompressor(32,65)

        ## Transition Up Blocks
        self.transup_4_8 = TransitionUp(4,16,8)
        self.transup_4_2 = TransitionUp(4,16,2)
        self.transup_4_4 = TransitionUp(4,16,4)
        self.transup_48_2 = TransitionUp(48,16,2)
        self.transup_48_4 = TransitionUp(48,16,4)
        self.transup_64_2 = TransitionUp(64,16,2)

        ## Dense Blocks
        self.block16 = DenseBlock(16)
        self.block32 = DenseBlock(32)
        self.block48 = DenseBlock(48)

        ## Transiton Final Blocks
        self.transfinal2 = TransitionFinal(in_channels=80)

        ## Temporal Block
        self.tempBlock = TemporalBlock(1,1)

    def encoder(self,x):
        x = self.block2(x)

        x1 = self.transdown34_8(x)
        x2 = self.transdown34_4(x) # required in cat
        x3 = self.transdown34_2(x)

        x3 = self.block34(x3)
        x3 = self.transdown66_2(x3)
        x4 = self.transdown66_2(x3)

        x2 = torch.cat([x2,x3],dim=1)
        x2 = self.block100(x2)
        x2 = self.transdown132_2(x2)

        x = torch.cat([x1,x2,x4],dim=1)

        mu = self.linearCompressor(self.transfinal(self.block232(x)))
        sigma = self.linearCompressor(self.transfinal(self.block232(x)))

        return mu, sigma

        
    def magnitude_decoder(self,z):
        z = self.tempBlock(z) # Temp Block
        z = self.linearCompressorUp(z) # Linear Layer

        z1 = self.transup_4_8(z)
        z2 = self.transup_4_4(z)
        z3 = self.transup_4_2(z)

        z3 = self.block16(z3)
        z3s = z3.clone()


        z3 = self.transup_48_2(z3)

        z2 = torch.cat([z2,z3],dim=1)
        z2 = self.block32(z2)
        z2 = self.transup_64_2(z2)
        z4 = self.transup_48_4(z3s)

        z1 = torch.cat([z1,z2,z4],dim=1)
        z2 = z1.clone()

        z1 = self.block48(z1)
        z2 = self.block48(z2)

        z1 = self.transfinal2(z1)
        z2 = self.transfinal2(z2)

        return z1, torch.sigmoid(z2)

    def phase_decoder(self,z,est_mag,est_var):
        pass

    def forward(self,x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrised = mu + sigma*epsilon
        var, x_reconstructed = self.magnitude_decoder(z_reparametrised)

        return x_reconstructed, mu, sigma