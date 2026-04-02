"""
features/audio_features.py — Audio feature extraction (MFCCs).

Section 3.1: "Speech is represented with 26 Mel-frequency cepstral
coefficients (MFCCs) from 27 ms frames with 10 ms stride."

Also handles μ-law encoding for WaveNet targets.
"""

import numpy as np
import librosa
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AudioFeatureConfig


def extract_mfcc(audio: np.ndarray,
                 cfg: Optional[AudioFeatureConfig] = None) -> np.ndarray:
    """
    Extract 26 MFCCs from waveform.

    Args:
        audio: (n_samples,) mono waveform at 16 kHz.
        cfg: AudioFeatureConfig.

    Returns:
        mfcc: (n_frames, 26) at 100 Hz.
    """
    if cfg is None:
        cfg = AudioFeatureConfig()

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.frame_shift_samples,   # 160 samples → 10 ms
        win_length=cfg.frame_length_samples,   # 432 samples → 27 ms
        center=True,
    )
    return mfcc.T   # (n_frames, 26)


def mfcc_to_audio(mfcc: np.ndarray,
                  cfg: Optional[AudioFeatureConfig] = None) -> np.ndarray:
    """
    Rough Griffin-Lim inversion from MFCCs (for debugging only —
    the real pipeline uses WaveNet).
    """
    if cfg is None:
        cfg = AudioFeatureConfig()

    mel_spec = librosa.feature.inverse.mfcc_to_mel(
        mfcc.T, n_mels=128, norm=None
    )
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=cfg.sample_rate,
        hop_length=cfg.frame_shift_samples,
        win_length=cfg.frame_length_samples,
    )
    return audio


# ─── μ-law for WaveNet ──────────────────────────────────────────────────────

def mu_law_encode(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """μ-law companding + quantization to [0, mu]."""
    audio = np.clip(audio, -1.0, 1.0)
    encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    quantized = ((encoded + 1) / 2 * mu + 0.5).astype(np.int64)
    return quantized


def mu_law_decode(quantized: np.ndarray, mu: int = 255) -> np.ndarray:
    """Inverse μ-law: integer codes → float waveform in [-1, 1]."""
    signal = 2.0 * quantized.astype(np.float32) / mu - 1.0
    decoded = np.sign(signal) * (1.0 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return decoded


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load and normalize audio file."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    # Normalize to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio
