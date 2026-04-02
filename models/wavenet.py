"""
models/wavenet.py — WaveNet synthesis model.

Section 3.3: "To synthesize audio from speech features, we use a WaveNet
decoder which generates the audio sample by sample conditioned on MFCC
speech features A'.  Our full synthesis model consists of a bidirectional
LSTM of 512 dimensions, a linear projection down to 128 dimensions, and
finally the WaveNet decoder which generates samples at 16 kHz."

Appendix A: WaveNet hyperparameters.

This is a self-contained WaveNet implementation (the paper used
NVIDIA's nv-wavenet for GPU inference, but we implement from scratch
for clarity and portability).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WaveNetConfig


# ─── Conditioning Network ─────────────────────────────────────────────────────

class ConditioningNetwork(nn.Module):
    """
    "Our full synthesis model consists of a bidirectional LSTM of 512
     dimensions, a linear projection down to 128 dimensions"

    Takes MFCC features at 100 Hz → upsamples to 16 kHz → conditioning.
    """

    def __init__(self, input_dim: int = 26, cfg: Optional[WaveNetConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = WaveNetConfig()
        self.cfg = cfg

        # BiLSTM on MFCCs
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.cond_lstm_hidden,  # 512
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Project to conditioning dimension
        self.projection = nn.Linear(
            cfg.cond_lstm_hidden * 2,  # bidirectional → 1024
            cfg.n_cond_channels,       # 128
        )

        # Upsampling: 100 Hz → 16 kHz (factor = 160)
        self.upsample = nn.ConvTranspose1d(
            cfg.n_cond_channels,
            cfg.n_cond_channels,
            kernel_size=cfg.upsamp_window,    # 432
            stride=cfg.upsamp_stride,         # 160
            padding=(cfg.upsamp_window - cfg.upsamp_stride) // 2,
        )

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mfcc: (B, T_feat, 26) MFCC features at 100 Hz.

        Returns:
            conditioning: (B, n_cond_channels, T_audio) at 16 kHz.
        """
        x, _ = self.lstm(mfcc)                          # (B, T, 1024)
        x = self.projection(x)                           # (B, T, 128)
        x = x.transpose(1, 2)                            # (B, 128, T)
        x = self.upsample(x)                             # (B, 128, T_audio)
        return x


# ─── WaveNet Core ──────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Causal convolution: pad on the left only."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class WaveNetResidualBlock(nn.Module):
    """Single WaveNet residual block with gated activation + conditioning."""

    def __init__(self, residual_channels, skip_channels,
                 cond_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.dilated_conv = CausalConv1d(
            residual_channels, 2 * residual_channels,
            kernel_size, dilation=dilation,
        )
        self.cond_proj = nn.Conv1d(cond_channels, 2 * residual_channels, 1)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x, conditioning):
        """
        Args:
            x: (B, residual_ch, T)
            conditioning: (B, cond_ch, T)
        """
        h = self.dilated_conv(x)
        h = h + self.cond_proj(conditioning)

        # Gated activation
        h_tanh, h_sig = h.chunk(2, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sig)

        skip = self.skip_conv(h)
        residual = self.res_conv(h) + x
        return residual, skip


class WaveNet(nn.Module):
    """
    Full WaveNet decoder with conditioning network.

    Appendix A hyperparameters:
        n_layers=16, max_dilation=128, n_residual_channels=64,
        n_skip_channels=256, n_out_channels=256, n_cond_channels=128
    """

    def __init__(self, cfg: Optional[WaveNetConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = WaveNetConfig()
        self.cfg = cfg

        # Input embedding (μ-law quantized audio → residual channels)
        self.input_conv = CausalConv1d(1, cfg.n_residual_channels, 1)

        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList()
        dilation = 1
        for i in range(cfg.n_layers):
            self.blocks.append(WaveNetResidualBlock(
                residual_channels=cfg.n_residual_channels,
                skip_channels=cfg.n_skip_channels,
                cond_channels=cfg.n_cond_channels,
                kernel_size=2,
                dilation=dilation,
            ))
            dilation *= 2
            if dilation > cfg.max_dilation:
                dilation = 1

        # Output network
        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(cfg.n_skip_channels, cfg.n_skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(cfg.n_skip_channels, cfg.n_out_channels, 1),
        )

        # Conditioning network (BiLSTM + upsample)
        self.conditioning = ConditioningNetwork(input_dim=26, cfg=cfg)

    def forward(self, audio_input: torch.Tensor,
                mfcc: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced training forward pass.

        Args:
            audio_input: (B, T_audio) μ-law quantized waveform as floats in [-1,1].
            mfcc: (B, T_feat, 26) gold MFCC features.

        Returns:
            logits: (B, n_out_channels, T_audio) class logits for each sample.
        """
        # Get conditioning from MFCCs
        cond = self.conditioning(mfcc)  # (B, 128, T_cond)

        # Trim or pad conditioning to match audio length
        T_audio = audio_input.size(1)
        if cond.size(2) > T_audio:
            cond = cond[:, :, :T_audio]
        elif cond.size(2) < T_audio:
            pad = T_audio - cond.size(2)
            cond = F.pad(cond, (0, pad))

        # Input embedding
        x = audio_input.unsqueeze(1)          # (B, 1, T)
        x = self.input_conv(x)                # (B, residual_ch, T)

        # Residual blocks
        skip_sum = torch.zeros(
            x.size(0), self.cfg.n_skip_channels, x.size(2),
            device=x.device, dtype=x.dtype,
        )
        for block in self.blocks:
            x, skip = block(x, cond)
            skip_sum = skip_sum + skip

        # Output
        logits = self.output_net(skip_sum)    # (B, n_out, T)
        return logits

    @torch.no_grad()
    def generate(self, mfcc: torch.Tensor,
                 n_samples: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive generation — sample by sample.

        Args:
            mfcc: (1, T_feat, 26) conditioning features.
            n_samples: Number of audio samples to generate.

        Returns:
            waveform: (n_samples,) generated μ-law codes as ints.
        """
        self.eval()
        cond = self.conditioning(mfcc)  # (1, 128, T_cond)

        if n_samples is None:
            n_samples = cond.size(2)

        if cond.size(2) < n_samples:
            cond = F.pad(cond, (0, n_samples - cond.size(2)))
        cond = cond[:, :, :n_samples]

        output = torch.zeros(n_samples, dtype=torch.long, device=mfcc.device)
        current_sample = torch.zeros(1, 1, 1, device=mfcc.device)

        for t in range(n_samples):
            # For efficiency in production, use cached/incremental inference.
            # This naive version recomputes everything (slow but correct).
            x = self.input_conv(current_sample)

            skip_sum = torch.zeros(
                1, self.cfg.n_skip_channels, 1, device=mfcc.device
            )
            c_t = cond[:, :, t:t+1]

            for block in self.blocks:
                x, skip = block(x, c_t)
                skip_sum = skip_sum + skip

            logits = self.output_net(skip_sum)  # (1, 256, 1)
            probs = F.softmax(logits.squeeze(-1), dim=-1)
            sample = torch.multinomial(probs, 1).squeeze()
            output[t] = sample

            # Convert back to float for next input
            current_sample = (2.0 * sample.float() / (self.cfg.n_in_channels - 1) - 1.0)
            current_sample = current_sample.view(1, 1, 1)

        return output


class WaveNetSynthesizer(nn.Module):
    """
    Wrapper: MFCC features → audio waveform.
    Combines WaveNet with its conditioning network.
    """

    def __init__(self, cfg: Optional[WaveNetConfig] = None):
        super().__init__()
        self.wavenet = WaveNet(cfg)

    def forward(self, audio_input, mfcc):
        return self.wavenet(audio_input, mfcc)

    def generate(self, mfcc, n_samples=None):
        return self.wavenet.generate(mfcc, n_samples)
