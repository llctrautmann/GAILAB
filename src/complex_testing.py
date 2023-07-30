import torch

# def convert_to_complex(mag, angle):

#     return mag*(torch.cos(angle)+1j* torch.sin(angle))


# def istft_transform(batch):
#     batch_size = batch[0].shape[0]

#     signals = []
#     for i in range(batch_size):
#         complex_spectrogram = convert_to_complex(batch[i][0], batch[i][1])
#         new_row = torch.zeros(1, 512)
#         complex_spectrogram = torch.cat((complex_spectrogram, new_row), dim=0)
#         signal = torch.istft(complex_spectrogram,n_fft=1024,hop_length=512).unsqueeze_(0).unsqueeze_(0)
#         signals.append(signal)

#     signals = torch.cat(signals, dim=0)
#     return signals


device = 'mps'

mag = torch.randn(512,512).to(device)
phase = torch.randn(512,512).to(device)

mag*(torch.cos(mag)+1j* torch.sin(phase))

# convert_to_complex(mag,phase).shape

