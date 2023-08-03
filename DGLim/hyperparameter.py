from dataclasses import dataclass

@dataclass
class Hyperparameter:
    # dataloader args
    annotation_file_path: str = '../../data/AvianID_AcousticIndices/UK_AI.csv'
    root_dir: str = '../../data/UK_BIRD/'
    key: str = 'habitat'
    mode: str = 'stft'
    length: int = 1
    sampling_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 512
    downsample: bool = True
    mel_spectrogram: bool = None
    verbose: bool = False
    fixed_limit: bool = True
    batch_size: int = 4
    num_workers: int = 1

    # Model args
    weight_decay: float = 0.0001
    learning_rate: float = 5e-4
    epochs: int = 100

    # save / load model
    save_path: str = '../data/checkpoints/DGL.pth.tar'
    load_path: str = '../data/checkpoints/DGL.pth.tar'





hp = Hyperparameter()