import matplotlib.pyplot as plt
import librosa
import torch

def plot_spectrograms(batch: torch.Tensor,magnitude=True):
    plt.figure(figsize=(10,9))
    if magnitude:
        librosa.display.specshow(batch[0][0].numpy(),
                                sr=48000,
                                x_axis='time',
                                y_axis='linear',
                                # vmin=0,
                                # vmax=44100//2
                                )
        plt.colorbar(format="%+2.f")
        plt.show()
    else:
        librosa.display.specshow(batch[0][1].numpy(),
                                sr=48000,
                                x_axis='time',
                                y_axis='linear',
                                cmap='Blues',
                                )
        plt.colorbar(format="%+2.f")
        plt.show()