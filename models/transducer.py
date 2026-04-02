"""
models/transducer.py — EMG-to-Speech Feature Transducer.

Section 3.1: "The LSTM model itself consists of 3 bidirectional LSTM
layers with 1024 hidden units, followed by a linear projection to the
speech feature dimension. Dropout 0.5 is used between all layers, as
well as before the first LSTM and after the last LSTM."

Input:  E'[i] + session_embedding  →  (batch, T, 112 + 32)
Output: Â'[i]                      →  (batch, T, 26)
"""

import torch
import torch.nn as nn
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TransducerConfig


class SessionEmbedding(nn.Module):
    """
    "We represent each session with a 32-dimensional session embedding
     and append the session embedding to the EMG features across all
     timesteps of an example before feeding into the LSTM."
    """
    def __init__(self, n_sessions: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(n_sessions, embed_dim)

    def forward(self, emg_features: torch.Tensor,
                session_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emg_features: (B, T, 112)
            session_ids: (B,) integer session indices

        Returns:
            (B, T, 112 + 32) with session embedding broadcast across time.
        """
        embed = self.embedding(session_ids)          # (B, 32)
        embed = embed.unsqueeze(1).expand(-1, emg_features.size(1), -1)
        return torch.cat([emg_features, embed], dim=-1)


class EMGTransducer(nn.Module):
    """
    3-layer BiLSTM transducer: EMG features → MFCC predictions.

    Architecture (Section 3.1):
        Dropout(0.5)
        → BiLSTM(1024) × 3 with inter-layer dropout(0.5)
        → Dropout(0.5)
        → Linear → 26 MFCCs
    """

    def __init__(self, cfg: TransducerConfig, n_sessions: int = 1):
        super().__init__()
        self.cfg = cfg

        # Session embedding
        self.session_embed = SessionEmbedding(
            n_sessions, cfg.session_embed_dim
        )

        # Pre-LSTM dropout: "as well as before the first LSTM"
        self.input_dropout = nn.Dropout(cfg.dropout)

        # 3-layer BiLSTM
        # Note: PyTorch LSTM `dropout` applies between layers (not after last)
        self.lstm = nn.LSTM(
            input_size=cfg.lstm_input_dim,   # 112 + 32 = 144
            hidden_size=cfg.lstm_hidden_dim,  # 1024
            num_layers=cfg.lstm_num_layers,   # 3
            batch_first=True,
            bidirectional=cfg.lstm_bidirectional,
            dropout=cfg.dropout,              # between layers
        )

        # Post-LSTM dropout: "after the last LSTM"
        self.output_dropout = nn.Dropout(cfg.dropout)

        # Linear projection to MFCC dimension
        lstm_out_dim = cfg.lstm_hidden_dim * (2 if cfg.lstm_bidirectional else 1)
        self.projection = nn.Linear(lstm_out_dim, cfg.output_dim)

    def forward(self, emg_features: torch.Tensor,
                session_ids: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            emg_features: (B, T, 112) normalized EMG features.
            session_ids: (B,) session index per example.
            lengths: (B,) actual sequence lengths for packing.

        Returns:
            predicted_mfcc: (B, T, 26)
        """
        # Append session embedding
        x = self.session_embed(emg_features, session_ids)  # (B, T, 144)

        # Pre-LSTM dropout
        x = self.input_dropout(x)

        # BiLSTM
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        x, _ = self.lstm(x)

        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Post-LSTM dropout
        x = self.output_dropout(x)

        # Linear projection
        predicted = self.projection(x)  # (B, T, 26)
        return predicted

    def predict(self, emg_features: torch.Tensor,
                session_ids: torch.Tensor) -> torch.Tensor:
        """Inference mode — no dropout."""
        self.eval()
        with torch.no_grad():
            return self.forward(emg_features, session_ids)
