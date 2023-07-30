import matplotlib.pyplot as plt
import torchaudio

def display_waveform(waveform, sample_rate=44100):
    # requires a torch tensor waveform
    plt.figure(figsize=(10, 3))
    plt.plot(waveform.t().numpy())
    
    plt.title("Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
