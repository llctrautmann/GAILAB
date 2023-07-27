import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64,kernel_size=5, stride=1,requires_sum=True,generator=True):
        super(ConvBlock, self).__init__()
        self.requires_sum = requires_sum
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride, padding='same' if generator else 2),
            nn.PReLU() if generator else nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='same' if generator else 2) if self.requires_sum else nn.Identity()) # changed in_channels to out_channels 15 Jul 2023 @ 15:39:28
        
    def forward(self, x):
        if self.requires_sum is True:
            out = self.block(x)
            return x + out
        else:
            out = self.block(x)
            return out
        


class ConvBlock1D(nn.Module): # Requires to be rewritten to incooperate the 1DConvolutions
    def __init__(self,in_channels=1, out_channels=64,kernel_size=32, stride=2,padding=0):
        super(ConvBlock1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.block(x)
    


class Generator(nn.Module):
    def __init__(self,in_channel=2, out_channel=64, blocks=4):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=11,stride=1,padding='same'),
            nn.PReLU()
        )

        self.blocks = nn.Sequential(*[ConvBlock() for _ in range(blocks)])
        self.conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=5,stride=1,padding='same')

        self.terminal = nn.Sequential(
            ConvBlock(in_channels=out_channel, out_channels=out_channel,requires_sum=False),
            ConvBlock(in_channels=out_channel, out_channels=out_channel,requires_sum=False),
            nn.Conv2d(in_channels=out_channel, out_channels=3,kernel_size=11,stride=1,padding='same')
        )

    def forward(self, z):
        # https://de.wikipedia.org/wiki/Arctan2
        z = self.initial(z)
        out = self.blocks(z)
        out = self.conv(out)
        out = out + z
        out = self.terminal(out)

        mag  = out[:,:1,:,:]
        phase = torch.arctan2(out[:,1:2,:,:], out[:,2:,:,:])
        out = torch.cat([mag, phase],dim=1)
        return out


class Discriminator(nn.Module):
    def __init__(self,in_channels=3,out_channels=32,in_features=2048,out_features=512,in_features_final=2048,blocks=2):
        super(Discriminator, self).__init__()        
        self.magnitude_path = nn.Sequential(
            ConvBlock(in_channels=1,out_channels=out_channels,stride=2,requires_sum=False,generator=False),
            ConvBlock(in_channels=out_channels,out_channels=out_channels,stride=2,requires_sum=False,generator=False),
            ConvBlock(in_channels=out_channels,out_channels=out_channels,stride=2,requires_sum=False,generator=False),
            ConvBlock(in_channels=out_channels,out_channels=out_channels,stride=2,requires_sum=False,generator=False),
            ConvBlock(in_channels=out_channels,out_channels=out_channels,stride=2,requires_sum=False,generator=False))
        
        self.magnitude_linear = nn.Sequential(
            nn.Linear(in_features=32*16*16,out_features=65385),
            nn.LeakyReLU(),
        )
        
        self.phase_path = ConvBlock1D(in_channels=1,out_channels=32)


        # [f(x) if condition else g(x) for x in sequence]
        self.blocks = nn.Sequential(*[ConvBlock1D(in_channels=33, out_channels=64) if block == 0 else ConvBlock1D(in_channels=64, out_channels=64) for block in range(blocks)])

        self.terminal = nn.Sequential(
            nn.Linear(in_features=14400,out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024,out_features=1)
        )


    def forward(self, phase, magnitude):
        # phase = self.istft_transform(comp)        
        phase = self.phase_path(phase)

        mag = self.magnitude_path(magnitude) # will be 32 * H_mag * W_mag
        mag = mag.view(mag.shape[0],-1)
        mag = self.magnitude_linear(mag).unsqueeze_(1)

        out = torch.cat([mag,phase],dim=1)
        out = self.blocks(out)
        out = out.view(out.shape[0],-1)
        out = self.terminal(out)

        return out.view(out.size(0), 1, 1, -1)


    

    @staticmethod
    def istft_transform(batch):
        def convert_to_complex(mag, angle):
        # https://dsp.stackexchange.com/a/72172 
            return mag*(torch.cos(angle)+1j* torch.sin(angle))
        
        batch_size = batch[0].shape[0]

        signals = []
        for i in range(batch_size):
            complex_spectrogram = convert_to_complex(batch[i][0], batch[i][1])
            new_row = torch.zeros(1, 512)
            complex_spectrogram = torch.cat((complex_spectrogram, new_row), dim=0)
            signal = torch.istft(complex_spectrogram,n_fft=1024,hop_length=512).unsqueeze_(0).unsqueeze_(0)
            signals.append(signal)

        signals = torch.cat(signals, dim=0)
        return signals 