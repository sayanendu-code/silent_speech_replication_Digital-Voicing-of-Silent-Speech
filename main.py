#!/usr/bin/env python3
"""
main.py — Unified entry point for the full Silent Speech replication.

Stages:
    1. download   — Download dataset from Zenodo
    2. preprocess — Raw data → normalized features
    3. train      — Train transducer + WaveNet
    4. evaluate   — Generate audio from test set + compute WER
    5. demo       — Single-file inference

Usage:
    python main.py download
    python main.py preprocess
    python main.py train --gpu 0
    python main.py evaluate --asr whisper
    python main.py demo --emg_path test.npy
    python main.py all --gpu 0    # run everything
"""

import os
import sys
import argparse
import subprocess
import time

from config import FullConfig
from utils.helpers import set_seed, get_device, setup_logger, Timer


def stage_download(args, cfg):
    """Stage 1: Download dataset."""
    print("\n" + "=" * 70)
    print("  STAGE 1: DOWNLOAD DATASET")
    print("=" * 70)
    from download_data import main as download_main
    download_main()


def stage_preprocess(args, cfg):
    """Stage 2: Preprocess raw data into features."""
    print("\n" + "=" * 70)
    print("  STAGE 2: PREPROCESS DATA")
    print("=" * 70)
    from data.preprocessing import preprocess_all
    preprocess_all(
        raw_dir=cfg.paths.raw_emg_dir,
        output_dir=cfg.paths.processed_dir,
        cfg=cfg,
    )


def stage_train(args, cfg):
    """Stage 3: Train both models."""
    print("\n" + "=" * 70)
    print("  STAGE 3A: TRAIN EMG-TO-SPEECH TRANSDUCER")
    print("=" * 70)

    timer = Timer().start()

    # Train transducer
    from train_transducer import main as train_transducer_main

    class TransducerArgs:
        data_dir = cfg.paths.processed_dir
        gpu = args.gpu

    train_transducer_main(TransducerArgs())
    print(f"\nTransducer training completed in {timer.elapsed_str()}")

    print("\n" + "=" * 70)
    print("  STAGE 3B: TRAIN WAVENET DECODER")
    print("=" * 70)

    timer.start()
    from train_wavenet import train_wavenet

    class WaveNetArgs:
        data_dir = cfg.paths.processed_dir
        gpu = args.gpu

    train_wavenet(WaveNetArgs())
    print(f"\nWaveNet training completed in {timer.elapsed_str()}")


def stage_evaluate(args, cfg):
    """Stage 4: Generate audio and evaluate WER."""
    print("\n" + "=" * 70)
    print("  STAGE 4: EVALUATE")
    print("=" * 70)

    import pickle
    import numpy as np
    from inference import SilentSpeechPipeline, evaluate_batch
    from data.preprocessing import load_processed, split_data

    device = f'cuda:{args.gpu}' if __import__('torch').cuda.is_available() else 'cpu'

    # Load test data
    data = load_processed(os.path.join(cfg.paths.processed_dir, 'processed_data.pkl'))
    _, _, test_data = split_data(data['parallel'],
                                 n_val=cfg.train.ov_val,
                                 n_test=cfg.train.ov_test,
                                 seed=cfg.train.seed)

    # Prepare test items with raw EMG for the pipeline
    for item in test_data:
        # The pipeline expects raw EMG — but we only have features.
        # For evaluation, we feed normalized features directly.
        item['silent_emg_raw'] = item.get('silent_emg_raw', item['silent_emg'])

    # Build pipeline
    pipeline = SilentSpeechPipeline(
        transducer_ckpt=os.path.join(cfg.paths.checkpoint_dir, 'best_transducer.pt'),
        wavenet_ckpt=os.path.join(cfg.paths.wavenet_checkpoint_dir, 'best_wavenet.pt'),
        normalizer_path=os.path.join(cfg.paths.processed_dir, 'processed_data.pkl'),
        device=device,
    )

    # Generate and evaluate
    results = evaluate_batch(pipeline, test_data, cfg.paths.output_dir)

    # ASR evaluation
    if hasattr(args, 'asr') and args.asr:
        from utils.evaluation import evaluate_model
        ref_texts = {
            f'test_{i:04d}.wav': item['text']
            for i, item in enumerate(test_data)
        }
        evaluate_model(cfg.paths.output_dir, ref_texts, asr_backend=args.asr)


def stage_demo(args, cfg):
    """Stage 5: Single-file demo."""
    print("\n" + "=" * 70)
    print("  DEMO: SINGLE FILE INFERENCE")
    print("=" * 70)

    import numpy as np
    import soundfile as sf
    from inference import SilentSpeechPipeline

    device = f'cuda:{args.gpu}' if __import__('torch').cuda.is_available() else 'cpu'

    pipeline = SilentSpeechPipeline(
        transducer_ckpt=os.path.join(cfg.paths.checkpoint_dir, 'best_transducer.pt'),
        wavenet_ckpt=os.path.join(cfg.paths.wavenet_checkpoint_dir, 'best_wavenet.pt'),
        normalizer_path=os.path.join(cfg.paths.processed_dir, 'processed_data.pkl'),
        device=device,
    )

    emg = np.load(args.emg_path)
    print(f"Input EMG shape: {emg.shape}")

    audio = pipeline(emg, session_id=getattr(args, 'session_id', 0))

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.output_dir, 'demo_output.wav')
    sf.write(out_path, audio, 16000)
    print(f"Output audio saved to: {out_path}")
    print(f"Duration: {len(audio)/16000:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Digital Voicing of Silent Speech — Full Replication',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py download                    # download data
    python main.py preprocess                  # extract features
    python main.py train --gpu 0               # train models
    python main.py evaluate --asr whisper      # evaluate WER
    python main.py demo --emg_path test.npy    # single inference
    python main.py all --gpu 0                 # run everything
        """
    )
    parser.add_argument('stage', type=str,
                        choices=['download', 'preprocess', 'train',
                                 'evaluate', 'demo', 'all'],
                        help='Pipeline stage to run')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--emg_path', type=str, default=None,
                        help='Path to EMG .npy file (for demo)')
    parser.add_argument('--asr', type=str, default='whisper',
                        choices=['whisper', 'deepspeech', 'manual'],
                        help='ASR backend for evaluation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Config
    cfg = FullConfig()
    set_seed(args.seed)

    total_timer = Timer().start()

    stages = {
        'download': [stage_download],
        'preprocess': [stage_preprocess],
        'train': [stage_train],
        'evaluate': [stage_evaluate],
        'demo': [stage_demo],
        'all': [stage_download, stage_preprocess, stage_train, stage_evaluate],
    }

    for stage_fn in stages[args.stage]:
        stage_fn(args, cfg)

    print(f"\nTotal time: {total_timer.elapsed_str()}")
    print("Done!")


if __name__ == '__main__':
    main()
