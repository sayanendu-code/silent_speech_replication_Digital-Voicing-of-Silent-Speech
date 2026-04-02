"""
config.py — All hyperparameters from Gaddy & Klein 2020.

Every magic number in the paper lives here so you can sweep them
without touching model code.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class EMGFeatureConfig:
    """Section 3.1 — EMG feature extraction parameters."""
    sample_rate: int = 1000          # EMG recorded at 1000 Hz
    n_channels: int = 8              # 8 EMG electrodes
    frame_length_ms: float = 27.0    # 27 ms window
    frame_shift_ms: float = 10.0     # 10 ms stride → 100 Hz features
    highpass_cutoff: float = 2.0     # Butterworth high-pass at 2 Hz
    notch_freq: float = 60.0         # AC notch at 60 Hz + harmonics
    triangular_cutoff: float = 134.0 # Low/high split for time-domain feats
    stft_nfft: int = 16              # 16-point STFT → 9 magnitude bins
    n_time_domain_features: int = 5  # per channel: 2 low + 2 high + ZCR
    # Total per channel = 5 (time-domain) + 9 (STFT) = 14
    # Total features = 14 * 8 channels = 112
    n_features_per_channel: int = 14
    total_features: int = 112

    @property
    def frame_length_samples(self) -> int:
        return int(self.frame_length_ms * self.sample_rate / 1000)

    @property
    def frame_shift_samples(self) -> int:
        return int(self.frame_shift_ms * self.sample_rate / 1000)

    @property
    def feature_rate(self) -> int:
        return int(1000 / self.frame_shift_ms)   # 100 Hz


@dataclass
class AudioFeatureConfig:
    """Section 3.1 — Audio (MFCC) feature parameters."""
    sample_rate: int = 16000         # Audio at 16 kHz
    n_mfcc: int = 26                 # 26 MFCCs
    frame_length_ms: float = 27.0    # Matched to EMG frames
    frame_shift_ms: float = 10.0     # 10 ms stride → 100 Hz
    n_fft: int = 512                 # For MFCC computation

    @property
    def frame_length_samples(self) -> int:
        return int(self.frame_length_ms * self.sample_rate / 1000)

    @property
    def frame_shift_samples(self) -> int:
        return int(self.frame_shift_ms * self.sample_rate / 1000)


@dataclass
class TransducerConfig:
    """Section 3.1 — BiLSTM transducer architecture."""
    emg_input_dim: int = 112         # EMG features
    session_embed_dim: int = 32      # Appended to EMG features
    lstm_input_dim: int = 144        # 112 + 32
    lstm_hidden_dim: int = 1024      # "1024 hidden units"
    lstm_num_layers: int = 3         # "3 bidirectional LSTM layers"
    lstm_bidirectional: bool = True
    dropout: float = 0.5             # "Dropout 0.5 … between all layers"
    output_dim: int = 26             # MFCC dimension


@dataclass
class WaveNetConfig:
    """Appendix A — WaveNet hyperparameters."""
    n_in_channels: int = 256         # μ-law quantization levels
    n_layers: int = 16
    max_dilation: int = 128
    n_residual_channels: int = 64
    n_skip_channels: int = 256
    n_out_channels: int = 256
    n_cond_channels: int = 128       # Conditioning from BiLSTM
    upsamp_window: int = 432
    upsamp_stride: int = 160         # 16000 / 100 Hz feature rate
    # Pre-WaveNet conditioning network
    cond_lstm_hidden: int = 512      # "bidirectional LSTM of 512 dims"
    cond_proj_dim: int = 128         # "linear projection down to 128"


@dataclass
class AlignmentConfig:
    """Section 3.2 — DTW / CCA / predicted-audio alignment."""
    cca_n_components: int = 15       # "15 dimensions for all experiments"
    lambda_audio: float = 10.0       # λ = 10 in δ_full
    warmup_epochs: int = 4           # "first four epochs using only δ_CCA"
    realign_interval: int = 5        # "re-compute alignments every 5 epochs"


@dataclass
class TrainConfig:
    """Section 3.1 + Appendix D — Training hyperparameters."""
    # Optimizer
    optimizer: str = "adam"
    initial_lr: float = 1e-3         # "initial learning rate .001"
    lr_patience: int = 5             # "decayed by half after every 5 epochs"
    lr_factor: float = 0.5
    # Training
    batch_size: int = 32
    max_epochs: int = 200
    early_stop_patience: int = 15
    # Loss
    loss_fn: str = "mse"             # "mean squared error loss"
    # Data splits (closed-vocab)
    cv_train: int = 370
    cv_val: int = 30
    cv_test: int = 100
    # Data splits (open-vocab)
    ov_val: int = 30
    ov_test: int = 100
    # Misc
    num_workers: int = 4
    seed: int = 42
    gradient_clip: float = 1.0


@dataclass
class WaveNetTrainConfig:
    """Section 3.3 — WaveNet training specifics."""
    batch_size: int = 1              # "no batching during training"
    initial_lr: float = 1e-3
    lr_patience: int = 5
    lr_factor: float = 0.5
    max_epochs: int = 200


@dataclass
class PathConfig:
    """File paths — adjust to your machine."""
    data_root: str = "./data"
    raw_emg_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    wavenet_checkpoint_dir: str = "./checkpoints/wavenet"

    def __post_init__(self):
        for d in [self.data_root, self.processed_dir,
                  self.checkpoint_dir, self.output_dir,
                  self.wavenet_checkpoint_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class FullConfig:
    """Master config aggregating all sub-configs."""
    emg: EMGFeatureConfig = field(default_factory=EMGFeatureConfig)
    audio: AudioFeatureConfig = field(default_factory=AudioFeatureConfig)
    transducer: TransducerConfig = field(default_factory=TransducerConfig)
    wavenet: WaveNetConfig = field(default_factory=WaveNetConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wavenet_train: WaveNetTrainConfig = field(default_factory=WaveNetTrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Number of sessions (set after scanning data)
    n_sessions: int = 1

    def __repr__(self):
        return (f"FullConfig(\n"
                f"  emg={self.emg},\n"
                f"  audio={self.audio},\n"
                f"  transducer={self.transducer},\n"
                f"  wavenet={self.wavenet},\n"
                f"  alignment={self.alignment},\n"
                f"  train={self.train}\n)")
