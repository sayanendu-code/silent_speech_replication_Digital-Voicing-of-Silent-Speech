"""
features/emg_features.py — EMG signal preprocessing and feature extraction.

Implements the exact feature pipeline from Section 3.1:
  - High-pass Butterworth filter (2 Hz cutoff)
  - Notch filters at 60 Hz + harmonics
  - Triangular low/high split at 134 Hz
  - 5 time-domain features per channel per frame
  - 9 STFT magnitude features per channel per frame
  → 14 features × 8 channels = 112-dim feature vector at 100 Hz
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter
from scipy.fft import rfft
from typing import Tuple, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMGFeatureConfig


# ─── Filtering ────────────────────────────────────────────────────────────────

def highpass_filter(signal: np.ndarray, cutoff: float, fs: int,
                    order: int = 5) -> np.ndarray:
    """Forward-backward Butterworth high-pass (zero phase delay)."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal: np.ndarray, freq: float, fs: int,
                 quality: float = 30.0) -> np.ndarray:
    """Remove AC noise at `freq` and its harmonics."""
    filtered = signal.copy()
    harmonic = freq
    while harmonic < fs / 2:
        b, a = iirnotch(harmonic, quality, fs)
        filtered = filtfilt(b, a, filtered, axis=0)
        harmonic += freq
    return filtered


def preprocess_emg(raw: np.ndarray, cfg: EMGFeatureConfig) -> np.ndarray:
    """
    Apply the full filter chain from Section 2.3.

    Args:
        raw: (n_samples, n_channels) raw EMG at 1000 Hz
        cfg: EMGFeatureConfig

    Returns:
        Filtered EMG, same shape.
    """
    assert raw.ndim == 2 and raw.shape[1] == cfg.n_channels, \
        f"Expected (T, {cfg.n_channels}), got {raw.shape}"

    filtered = highpass_filter(raw, cfg.highpass_cutoff, cfg.sample_rate)
    filtered = notch_filter(filtered, cfg.notch_freq, cfg.sample_rate)
    return filtered


# ─── Triangular low/high frequency split ──────────────────────────────────────

def _triangular_lowpass(signal_1d: np.ndarray, cutoff: float,
                        fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a 1-D signal into low- and high-frequency components using a
    triangular filter kernel centred at DC.  Cutoff at `cutoff` Hz.

    Returns:
        (x_low, x_high) each of length len(signal_1d).
    """
    # Triangular kernel width in samples (one side)
    half_width = int(fs / cutoff)
    kernel_len = 2 * half_width + 1
    kernel = np.bartlett(kernel_len)
    kernel /= kernel.sum()

    x_low = np.convolve(signal_1d, kernel, mode='same')
    x_high = signal_1d - x_low
    return x_low, x_high


# ─── Per-frame feature computation ────────────────────────────────────────────

def _zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of consecutive sample-pairs that cross zero."""
    if len(x) < 2:
        return 0.0
    signs = np.sign(x)
    crossings = np.sum(signs[1:] != signs[:-1])
    return crossings / (len(x) - 1)


def _frame_time_domain(x_low: np.ndarray, x_high: np.ndarray) -> np.ndarray:
    """
    Compute 5 time-domain features for one frame of one channel:
        [mean(x_low²),  mean(x_low),
         mean(x_high²), mean(|x_high|),
         ZCR(x_high)]
    """
    n = max(len(x_low), 1)
    feats = np.array([
        np.mean(x_low ** 2),
        np.mean(x_low),
        np.mean(x_high ** 2),
        np.mean(np.abs(x_high)),
        _zero_crossing_rate(x_high),
    ], dtype=np.float32)
    return feats


def _frame_stft(frame: np.ndarray, nfft: int) -> np.ndarray:
    """
    Magnitude of `nfft`-point STFT for one frame.
    Returns nfft//2 + 1 = 9 values for nfft=16.
    """
    windowed = frame * np.hanning(len(frame))
    # Zero-pad to nfft if frame is longer
    if len(windowed) < nfft:
        windowed = np.pad(windowed, (0, nfft - len(windowed)))
    spectrum = rfft(windowed[:nfft])
    return np.abs(spectrum).astype(np.float32)


# ─── Main extraction ──────────────────────────────────────────────────────────

def extract_emg_features(emg: np.ndarray,
                         cfg: Optional[EMGFeatureConfig] = None
                         ) -> np.ndarray:
    """
    Full EMG → feature pipeline.

    Args:
        emg: (n_samples, n_channels) filtered EMG signal at 1000 Hz.
        cfg: EMGFeatureConfig (uses defaults if None).

    Returns:
        features: (n_frames, 112) feature matrix at 100 Hz.
    """
    if cfg is None:
        cfg = EMGFeatureConfig()

    n_samples, n_channels = emg.shape
    frame_len = cfg.frame_length_samples   # 27
    frame_shift = cfg.frame_shift_samples  # 10
    n_frames = (n_samples - frame_len) // frame_shift + 1

    all_features = []

    for ch in range(n_channels):
        ch_signal = emg[:, ch]

        # Split into low and high frequency components
        x_low, x_high = _triangular_lowpass(
            ch_signal, cfg.triangular_cutoff, cfg.sample_rate
        )

        ch_feats = []
        for f in range(n_frames):
            start = f * frame_shift
            end = start + frame_len

            # Time-domain features (5)
            td = _frame_time_domain(x_low[start:end], x_high[start:end])

            # STFT magnitude features (9)
            stft = _frame_stft(ch_signal[start:end], cfg.stft_nfft)

            ch_feats.append(np.concatenate([td, stft]))  # 14-dim

        all_features.append(np.stack(ch_feats))  # (n_frames, 14)

    # Stack channels: (n_frames, 14 * 8) = (n_frames, 112)
    features = np.concatenate(all_features, axis=1)
    return features


# ─── Normalization ────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """
    "All EMG and audio features are normalized to approximately zero mean
     and unit variance before processing."
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features: np.ndarray):
        """Compute stats from training data. features: (N, D) or list."""
        if isinstance(features, list):
            features = np.concatenate(features, axis=0)
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std < 1e-6] = 1.0   # prevent division by zero
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.std = d['std']
