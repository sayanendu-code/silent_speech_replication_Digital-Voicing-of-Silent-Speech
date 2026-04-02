"""
train_wavenet.py — Train the WaveNet synthesis model.

Section 3.3: "WaveNet generates audio sample by sample conditioned on
MFCC speech features A'. During training, the model is given gold speech
features as input. Due to memory constraints we do not use any batching."

Usage:
    python train_wavenet.py --data_dir ./data/processed --gpu 0
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import FullConfig
from models.wavenet import WaveNetSynthesizer
from features.audio_features import mu_law_encode, mu_law_decode
from data.preprocessing import load_processed, split_data


def train_wavenet(args):
    cfg = FullConfig()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────
    print("Loading data...")
    data = load_processed(os.path.join(args.data_dir, 'processed_data.pkl'))

    # Collect all vocalized audio + MFCCs for WaveNet training
    all_audio = []
    all_mfcc = []

    for item in data['parallel']:
        if item.get('raw_audio') is not None:
            all_audio.append(item['raw_audio'])
            all_mfcc.append(item['vocalized_audio'])  # normalized MFCCs

    for item in data['nonparallel']:
        if item.get('raw_audio') is not None:
            all_audio.append(item['raw_audio'])
            all_mfcc.append(item['vocalized_audio'])

    print(f"  Total utterances for WaveNet: {len(all_audio)}")

    # ── Model ─────────────────────────────────────────────────────
    model = WaveNetSynthesizer(cfg.wavenet).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.wavenet_train.initial_lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.wavenet_train.lr_factor,
        patience=cfg.wavenet_train.lr_patience, verbose=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"WaveNet parameters: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────
    # "Due to memory constraints we do not use any batching" → batch_size=1
    os.makedirs(cfg.paths.wavenet_checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    segment_length = 16000  # 1 second segments for memory

    # Simple train/val split (90/10)
    n_val = max(1, len(all_audio) // 10)
    indices = np.random.RandomState(cfg.train.seed).permutation(len(all_audio))
    val_indices = set(indices[:n_val])
    train_indices = [i for i in range(len(all_audio)) if i not in val_indices]

    for epoch in range(cfg.wavenet_train.max_epochs):
        t0 = time.time()

        # ── Train ──
        model.train()
        np.random.shuffle(train_indices)
        train_loss_sum = 0
        n_train = 0

        for idx in train_indices:
            audio = all_audio[idx]
            mfcc = all_mfcc[idx]

            # Random segment crop
            if len(audio) > segment_length:
                start = np.random.randint(0, len(audio) - segment_length)
                audio_seg = audio[start:start + segment_length]
                feat_start = start // 160
                feat_end = feat_start + segment_length // 160
                mfcc_seg = mfcc[feat_start:min(feat_end, len(mfcc))]
            else:
                audio_seg = audio
                mfcc_seg = mfcc[:len(audio) // 160]

            # Prepare tensors
            audio_float = torch.tensor(
                audio_seg, dtype=torch.float32
            ).unsqueeze(0).to(device)

            quantized = mu_law_encode(audio_seg, mu=255)
            target = torch.tensor(quantized, dtype=torch.long).unsqueeze(0).to(device)

            mfcc_t = torch.tensor(
                mfcc_seg, dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(audio_float, mfcc_t)  # (1, 256, T)

            # Cross-entropy loss
            T_min = min(logits.size(2), target.size(1))
            loss = F.cross_entropy(logits[:, :, :T_min], target[:, :T_min])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train += 1

        train_loss = train_loss_sum / max(n_train, 1)

        # ── Validate ──
        model.eval()
        val_loss_sum = 0
        n_val_batches = 0
        with torch.no_grad():
            for idx in val_indices:
                audio = all_audio[idx]
                mfcc = all_mfcc[idx]

                if len(audio) > segment_length:
                    audio = audio[:segment_length]
                    mfcc = mfcc[:segment_length // 160]

                audio_float = torch.tensor(
                    audio, dtype=torch.float32
                ).unsqueeze(0).to(device)
                quantized = mu_law_encode(audio, mu=255)
                target = torch.tensor(
                    quantized, dtype=torch.long
                ).unsqueeze(0).to(device)
                mfcc_t = torch.tensor(
                    mfcc, dtype=torch.float32
                ).unsqueeze(0).to(device)

                logits = model(audio_float, mfcc_t)
                T_min = min(logits.size(2), target.size(1))
                loss = F.cross_entropy(logits[:, :, :T_min], target[:, :T_min])
                val_loss_sum += loss.item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(n_val_batches, 1)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | Time: {elapsed:.1f}s")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(cfg.paths.wavenet_checkpoint_dir, 'best_wavenet.pt'))
            print(f"  ✓ New best WaveNet (val_loss={val_loss:.4f})")

    print(f"\nWaveNet training complete. Best val: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WaveNet Decoder')
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    train_wavenet(args)
