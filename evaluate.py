#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation script.

Transcribe generated audio with ASR and compute WER against references.

Usage:
    python evaluate.py --audio_dir ./outputs --ref_file refs.json --asr whisper
"""

import os
import json
import argparse
from utils.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate Silent Speech Outputs')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory with generated .wav files')
    parser.add_argument('--ref_file', type=str, required=True,
                        help='JSON file: {filename: reference_text}')
    parser.add_argument('--asr', type=str, default='whisper',
                        choices=['whisper', 'deepspeech', 'manual'])
    parser.add_argument('--whisper_model', type=str, default='base.en')
    parser.add_argument('--deepspeech_model', type=str, default=None)
    parser.add_argument('--deepspeech_scorer', type=str, default=None)
    parser.add_argument('--manual_file', type=str, default=None)
    args = parser.parse_args()

    # Load reference texts
    with open(args.ref_file) as f:
        reference_texts = json.load(f)

    # Set up ASR kwargs
    asr_kwargs = {}
    if args.asr == 'whisper':
        asr_kwargs['model_size'] = args.whisper_model
    elif args.asr == 'deepspeech':
        asr_kwargs['model_path'] = args.deepspeech_model
        if args.deepspeech_scorer:
            asr_kwargs['scorer_path'] = args.deepspeech_scorer
    elif args.asr == 'manual':
        asr_kwargs['transcription_file'] = args.manual_file

    results = evaluate_model(
        args.audio_dir, reference_texts,
        asr_backend=args.asr, asr_kwargs=asr_kwargs,
    )

    print(f"\nFinal WER: {results['aggregate_wer']*100:.1f}%")


if __name__ == '__main__':
    main()
