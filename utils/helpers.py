"""
utils/helpers.py — Shared utilities for reproducibility, logging, checkpointing.
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
import torch
from typing import Optional, Dict, Any
from datetime import datetime


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility (Appendix D)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int = 0) -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"  Memory: {torch.cuda.get_device_properties(device).total_mem / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU (training will be slow)")
    return device


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    """Create a logger that writes to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{name}_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


# ─── Checkpointing ────────────────────────────────────────────────────────────

class CheckpointManager:
    """Manages model checkpoints with best-model tracking."""

    def __init__(self, save_dir: str, max_keep: int = 3):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.history = []
        os.makedirs(save_dir, exist_ok=True)

    def save(self, state: Dict[str, Any], metric: float,
             epoch: int, is_best: bool = False) -> str:
        """Save a checkpoint. Returns the save path."""
        filename = f'checkpoint_epoch{epoch:04d}.pt'
        path = os.path.join(self.save_dir, filename)
        torch.save(state, path)
        self.history.append((path, metric, epoch))

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(state, best_path)

        # Prune old checkpoints (keep best + most recent max_keep)
        if len(self.history) > self.max_keep:
            # Sort by metric (ascending = lower is better)
            sorted_hist = sorted(self.history, key=lambda x: x[1])
            keep_paths = {sorted_hist[0][0]}  # always keep best
            for item in self.history[-self.max_keep:]:
                keep_paths.add(item[0])

            for item_path, _, _ in self.history:
                if item_path not in keep_paths and os.path.exists(item_path):
                    os.remove(item_path)

        return path

    def load_best(self, device: torch.device = torch.device('cpu')) -> Dict:
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best checkpoint at {best_path}")
        return torch.load(best_path, map_location=device)


# ─── Training Meter ───────────────────────────────────────────────────────────

class AverageMeter:
    """Track running averages during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        secs = self.elapsed()
        if secs < 60:
            return f"{secs:.1f}s"
        elif secs < 3600:
            return f"{secs/60:.1f}m"
        else:
            return f"{secs/3600:.1f}h"


# ─── Experiment tracking ──────────────────────────────────────────────────────

def save_experiment_config(config, save_dir: str):
    """Save full config as JSON for experiment tracking."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'experiment_config.json')

    # Convert dataclass to dict
    from dataclasses import asdict
    config_dict = asdict(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    return path


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }


def format_params(n: int) -> str:
    """Human-readable parameter count."""
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
