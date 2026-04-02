"""
train_transducer.py — Train the EMG-to-Speech BiLSTM transducer.

Implements the full training pipeline from Sections 3.1 and 3.2:
  1. Initialize with raw DTW alignment
  2. Fit CCA and refine alignment
  3. Train transducer with MSE loss
  4. Periodically re-align with predicted audio (Section 3.2.2)
  5. Early stopping on validation loss

Usage:
    python train_transducer.py --data_dir ./data/processed --gpu 0
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import FullConfig
from models.transducer import EMGTransducer
from alignment.target_transfer import TargetTransfer
from data.dataset import (
    EMGSpeechDataset, CombinedDataset, build_dataloader, collate_fn
)
from data.preprocessing import load_processed, split_data


def masked_mse_loss(predicted: torch.Tensor, target: torch.Tensor,
                    lengths: torch.Tensor) -> torch.Tensor:
    """
    MSE loss with masking for padded sequences.

    "The model is trained with a mean squared error loss against
     time-aligned speech features."
    """
    batch_size, max_len, feat_dim = predicted.shape
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(predicted)

    diff = (predicted - target) ** 2
    diff = diff * mask.float()

    # Mean over non-padded elements
    n_elements = mask.float().sum()
    if n_elements > 0:
        return diff.sum() / n_elements
    return diff.sum()


def prepare_alignment_targets(
    parallel_train, transfer: TargetTransfer, epoch: int,
    model: EMGTransducer = None, device: str = 'cpu'
):
    """
    Compute or update alignment targets based on training phase.

    Returns list of warped audio features (training targets for silent EMG).
    """
    silent_emg = [item['silent_emg'] for item in parallel_train]
    vocalized_emg = [item['vocalized_emg'] for item in parallel_train]
    vocalized_audio = [item['vocalized_audio'] for item in parallel_train]

    if epoch == 0:
        # Phase 1: Initial raw DTW
        print("  Computing initial DTW alignments...")
        alignments = transfer.initial_alignment(silent_emg, vocalized_emg)

        # Phase 2: Fit CCA and re-align
        print("  Fitting CCA and re-aligning...")
        alignments = transfer.fit_cca_and_realign(
            silent_emg, vocalized_emg, alignments
        )
    elif transfer.should_realign(epoch) and model is not None:
        # Phase 3: Realign with predicted audio
        print(f"  Re-aligning with predicted audio at epoch {epoch}...")
        model.eval()
        predicted_audio = []
        for item in parallel_train:
            emg_t = torch.tensor(item['silent_emg'], dtype=torch.float32).unsqueeze(0).to(device)
            sess_t = torch.tensor([item['session']], dtype=torch.long).to(device)
            with torch.no_grad():
                pred = model(emg_t, sess_t).squeeze(0).cpu().numpy()
            predicted_audio.append(pred)
        model.train()

        alignments = transfer.realign_with_audio(
            silent_emg, vocalized_emg, predicted_audio, vocalized_audio
        )
    else:
        return None  # No re-alignment needed; use existing targets

    # Generate warped targets
    targets = transfer.generate_targets(vocalized_audio, alignments)
    return targets


def train_epoch(model, dataloader, optimizer, device):
    """Single training epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        emg = batch['emg'].to(device)
        audio = batch['audio'].to(device)
        sessions = batch['session_ids'].to(device)
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()
        predicted = model(emg, sessions, lengths)
        loss = masked_mse_loss(predicted, audio, lengths)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validation loss."""
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        emg = batch['emg'].to(device)
        audio = batch['audio'].to(device)
        sessions = batch['session_ids'].to(device)
        lengths = batch['lengths'].to(device)

        predicted = model(emg, sessions, lengths)
        loss = masked_mse_loss(predicted, audio, lengths)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main(args):
    cfg = FullConfig()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load data ─────────────────────────────────────────────────
    print("Loading processed data...")
    data = load_processed(os.path.join(args.data_dir, 'processed_data.pkl'))
    parallel_data = data['parallel']
    nonparallel_data = data['nonparallel']

    # Split parallel data
    parallel_train, parallel_val, parallel_test = split_data(
        parallel_data, n_val=cfg.train.ov_val, n_test=cfg.train.ov_test,
        seed=cfg.train.seed
    )
    print(f"  Parallel: {len(parallel_train)} train, "
          f"{len(parallel_val)} val, {len(parallel_test)} test")
    print(f"  Non-parallel: {len(nonparallel_data)} vocalized")

    n_sessions = len(data['session_map'])
    cfg.n_sessions = n_sessions

    # ── Initialize target transfer ────────────────────────────────
    transfer = TargetTransfer(
        cca_n_components=cfg.alignment.cca_n_components,
        lambda_audio=cfg.alignment.lambda_audio,
        warmup_epochs=cfg.alignment.warmup_epochs,
        realign_interval=cfg.alignment.realign_interval,
    )

    # Initial alignment
    silent_targets = prepare_alignment_targets(
        parallel_train, transfer, epoch=0, device=device
    )

    # ── Build datasets ────────────────────────────────────────────

    def build_training_datasets(silent_targets):
        """
        Three data sources (Section 3.2):
          1. Silent EMG + warped audio targets
          2. Vocalized EMG from parallel utterances
          3. Non-parallel vocalized EMG
        """
        # Source 1: Silent parallel
        silent_ds = EMGSpeechDataset(
            emg_features=[item['silent_emg'] for item in parallel_train],
            audio_features=silent_targets,
            session_ids=[item['session'] for item in parallel_train],
            texts=[item['text'] for item in parallel_train],
        )

        # Source 2: Vocalized parallel
        vocalized_parallel_ds = EMGSpeechDataset(
            emg_features=[item['vocalized_emg'] for item in parallel_train],
            audio_features=[item['vocalized_audio'] for item in parallel_train],
            session_ids=[item['session'] for item in parallel_train],
            texts=[item['text'] for item in parallel_train],
        )

        # Source 3: Non-parallel vocalized
        nonparallel_ds = EMGSpeechDataset(
            emg_features=[item['vocalized_emg'] for item in nonparallel_data],
            audio_features=[item['vocalized_audio'] for item in nonparallel_data],
            session_ids=[item['session'] for item in nonparallel_data],
            texts=[item['text'] for item in nonparallel_data],
        )

        return CombinedDataset(silent_ds, vocalized_parallel_ds, nonparallel_ds)

    train_dataset = build_training_datasets(silent_targets)

    # Validation: test on silent EMG with warped targets
    val_targets = transfer.generate_targets(
        [item['vocalized_audio'] for item in parallel_val],
        transfer.initial_alignment(
            [item['silent_emg'] for item in parallel_val],
            [item['vocalized_emg'] for item in parallel_val],
        )
    )
    val_dataset = EMGSpeechDataset(
        emg_features=[item['silent_emg'] for item in parallel_val],
        audio_features=val_targets,
        session_ids=[item['session'] for item in parallel_val],
    )

    train_loader = build_dataloader(
        train_dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.train.num_workers
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────
    model = EMGTransducer(cfg.transducer, n_sessions=n_sessions).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.train.initial_lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.train.lr_factor,
        patience=cfg.train.lr_patience, verbose=True,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ─────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.train.max_epochs):
        t0 = time.time()

        # Check if we need to re-align
        if epoch > 0 and transfer.should_realign(epoch):
            new_targets = prepare_alignment_targets(
                parallel_train, transfer, epoch, model, device
            )
            if new_targets is not None:
                silent_targets = new_targets
                train_dataset = build_training_datasets(silent_targets)
                train_loader = build_dataloader(
                    train_dataset, batch_size=cfg.train.batch_size,
                    shuffle=True, num_workers=cfg.train.num_workers
                )

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {lr:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': cfg,
            }, os.path.join(cfg.paths.checkpoint_dir, 'best_transducer.pt'))
            print(f"  ✓ New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


def finetune_closed_vocab(args):
    """
    Section 4.1: "All three models were trained on open vocabulary data
    before being fine-tuned on the closed vocabulary training set."

    Loads the best open-vocab transducer checkpoint and fine-tunes on
    the 370-utterance closed-vocab training set.
    """
    cfg = FullConfig()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    print("  CLOSED-VOCAB FINE-TUNING")
    print("="*60)

    # Load processed data
    data = load_processed(os.path.join(args.data_dir, 'processed_data.pkl'))
    closed_vocab = data.get('closed_vocab', [])
    if not closed_vocab:
        print("ERROR: No closed-vocab data found in processed_data.pkl.")
        print("Re-run preprocessing with a dataset that includes closed_vocab/.")
        return

    n_sessions = data.get('n_sessions', len(data.get('session_map', {})))
    cfg.n_sessions = n_sessions

    # Split: 370 train, 30 val, 100 test (Section 2.1)
    from data.preprocessing import split_data
    cv_train, cv_val, cv_test = split_data(
        closed_vocab,
        n_val=cfg.train.cv_val,     # 30
        n_test=cfg.train.cv_test,   # 100
        seed=cfg.train.seed,
    )
    print(f"  Closed-vocab: {len(cv_train)} train, {len(cv_val)} val, {len(cv_test)} test")

    # Load open-vocab checkpoint
    ov_ckpt_path = os.path.join(cfg.paths.checkpoint_dir, 'best_transducer.pt')
    if not os.path.exists(ov_ckpt_path):
        print(f"ERROR: No open-vocab checkpoint at {ov_ckpt_path}")
        print("Train open-vocab first: python train_transducer.py --data_dir ./data/processed")
        return

    print(f"  Loading open-vocab checkpoint: {ov_ckpt_path}")
    ckpt = torch.load(ov_ckpt_path, map_location=device)
    model = EMGTransducer(cfg.transducer, n_sessions=n_sessions).to(device)
    model.load_state_dict(ckpt['model_state'])

    # Build alignment + targets for closed-vocab
    transfer = TargetTransfer(
        cca_n_components=cfg.alignment.cca_n_components,
        lambda_audio=cfg.alignment.lambda_audio,
    )
    silent_targets = prepare_alignment_targets(cv_train, transfer, epoch=0, device=device)

    # Build datasets
    from data.dataset import EMGSpeechDataset, build_dataloader
    train_ds = EMGSpeechDataset(
        emg_features=[item['silent_emg'] for item in cv_train],
        audio_features=silent_targets,
        session_ids=[item['session'] for item in cv_train],
        texts=[item['text'] for item in cv_train],
    )
    # Also include vocalized closed-vocab for training
    from data.dataset import CombinedDataset
    vocalized_ds = EMGSpeechDataset(
        emg_features=[item['vocalized_emg'] for item in cv_train],
        audio_features=[item['vocalized_audio'] for item in cv_train],
        session_ids=[item['session'] for item in cv_train],
        texts=[item['text'] for item in cv_train],
    )
    combined_ds = CombinedDataset(train_ds, vocalized_ds)

    train_loader = build_dataloader(combined_ds, batch_size=cfg.train.batch_size, shuffle=True)

    # Validation on silent
    val_targets = transfer.generate_targets(
        [item['vocalized_audio'] for item in cv_val],
        transfer.initial_alignment(
            [item['silent_emg'] for item in cv_val],
            [item['vocalized_emg'] for item in cv_val],
        )
    )
    val_ds = EMGSpeechDataset(
        emg_features=[item['silent_emg'] for item in cv_val],
        audio_features=val_targets,
        session_ids=[item['session'] for item in cv_val],
    )
    val_loader = build_dataloader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    # Fine-tune with lower LR
    optimizer = Adam(model.parameters(), lr=cfg.train.initial_lr * 0.1)  # 0.0001
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    max_epochs = 100

    for epoch in range(max_epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"CV Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': cfg,
            }, os.path.join(cfg.paths.checkpoint_dir, 'best_transducer_closed_vocab.pt'))
            print(f"  ✓ New best closed-vocab model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nClosed-vocab fine-tuning complete. Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EMG-to-Speech Transducer')
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--closed_vocab', action='store_true',
                        help='Fine-tune on closed-vocab data (requires open-vocab checkpoint)')
    args = parser.parse_args()

    if args.closed_vocab:
        finetune_closed_vocab(args)
    else:
        main(args)
