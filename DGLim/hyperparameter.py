from dataclasses import dataclass

@dataclass
class Hyperparameter:
    # dataloader args
    annotation_file_path: str = '../../data/AvianID_AcousticIndices/UK_AI.csv'
    root_dir: str = '../../data/UK_BIRD/'
    key: str = 'habitat'
    mode: str = 'stft'
    length: int = 5
    sampling_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 512
    downsample: bool = True
    mel_spectrogram: bool = None
    verbose: bool = False
    fixed_limit: bool = True
    batch_size: int = 4
    num_workers: int = 4




hp = Hyperparameter()