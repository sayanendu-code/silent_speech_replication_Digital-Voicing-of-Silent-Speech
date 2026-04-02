#!/usr/bin/env python3
"""
download_data.py — Download the EMG dataset from Zenodo.

Dataset: https://doi.org/10.5281/zenodo.4064408
Code reference: https://github.com/dgaddy/silent_speech

This downloads the ~2GB dataset and extracts it into ./data/raw/

Usage:
    python download_data.py [--output_dir ./data/raw]
"""

import os
import sys
import argparse
import hashlib
import subprocess
from pathlib import Path


ZENODO_RECORD = "4064408"
ZENODO_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files"

# Files in the Zenodo release (update if the dataset changes)
DATASET_FILES = [
    "silent_speech_data.tar.gz",
]

GITHUB_REPO = "https://github.com/dgaddy/silent_speech.git"


def download_file(url: str, output_path: str):
    """Download a file with wget (shows progress)."""
    print(f"Downloading {url}...")
    subprocess.run(
        ["wget", "-c", "--progress=bar:force:noscroll",
         "-O", output_path, url],
        check=True,
    )


def extract_archive(archive_path: str, output_dir: str):
    """Extract tar.gz archive."""
    print(f"Extracting {archive_path}...")
    subprocess.run(
        ["tar", "-xzf", archive_path, "-C", output_dir],
        check=True,
    )


def clone_reference_code(output_dir: str):
    """Clone the original paper's code for reference."""
    ref_dir = os.path.join(output_dir, "reference_code")
    if os.path.exists(ref_dir):
        print(f"Reference code already exists at {ref_dir}")
        return

    print(f"Cloning reference code from {GITHUB_REPO}...")
    subprocess.run(
        ["git", "clone", GITHUB_REPO, ref_dir],
        check=True,
    )


def verify_data(data_dir: str) -> bool:
    """Quick sanity check that data was extracted properly."""
    expected_indicators = [
        "emg", "audio", ".pkl", ".npy", ".json"
    ]
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        all_files.extend(files)

    if len(all_files) == 0:
        print("WARNING: No files found in data directory!")
        return False

    found_types = set()
    for f in all_files:
        for indicator in expected_indicators:
            if indicator in f.lower():
                found_types.add(indicator)

    print(f"Found {len(all_files)} files")
    print(f"Detected data types: {found_types}")
    return len(all_files) > 0


def setup_from_github(output_dir: str):
    """
    Alternative: use the original repo's data loading code.
    The original repo has its own data format; this function
    sets up for that.
    """
    print("\n" + "=" * 60)
    print("ALTERNATIVE SETUP: Using original repo's data format")
    print("=" * 60)
    print(f"""
The original paper's code (github.com/dgaddy/silent_speech) uses
a specific data format. To use their exact data loader:

    1. Clone: git clone {GITHUB_REPO}
    2. Download data into their expected directory structure
    3. Use their read_emg.py for data loading

Our reimplementation (this repo) provides its own preprocessing
pipeline in data/preprocessing.py that handles the raw data.

To proceed with OUR pipeline:
    python data/preprocessing.py --raw_dir {output_dir} --output_dir ./data/processed
    """)


def main():
    parser = argparse.ArgumentParser(description='Download EMG-Speech Dataset')
    parser.add_argument('--output_dir', type=str, default='./data/raw',
                        help='Directory to save raw data')
    parser.add_argument('--clone_reference', action='store_true',
                        help='Also clone the original paper code')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download (just verify existing data)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Silent Speech EMG Dataset Downloader")
    print(f"  Zenodo DOI: 10.5281/zenodo.{ZENODO_RECORD}")
    print(f"  Output: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    if not args.skip_download:
        # Download from Zenodo
        for filename in DATASET_FILES:
            url = f"{ZENODO_URL}/{filename}"
            output_path = os.path.join(args.output_dir, filename)

            if os.path.exists(output_path):
                print(f"  {filename} already exists, skipping download")
            else:
                try:
                    download_file(url, output_path)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print(f"\nAutomatic download failed. Please download manually:")
                    print(f"  URL: {url}")
                    print(f"  Save to: {output_path}")
                    print(f"\nOr visit: https://doi.org/10.5281/zenodo.{ZENODO_RECORD}")
                    sys.exit(1)

            # Extract
            if filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                extract_archive(output_path, args.output_dir)

    # Verify
    print("\nVerifying data...")
    if verify_data(args.output_dir):
        print("Data verification passed!")
    else:
        print("Data verification failed — check the directory contents.")
        setup_from_github(args.output_dir)

    # Optionally clone reference code
    if args.clone_reference:
        clone_reference_code(args.output_dir)

    print(f"""
{'='*60}
NEXT STEPS:
{'='*60}

1. Preprocess the data:
   python -c "
   from data.preprocessing import preprocess_all
   preprocess_all('{args.output_dir}', './data/processed')
   "

2. Train the transducer:
   python train_transducer.py --data_dir ./data/processed --gpu 0

3. Train WaveNet:
   python train_wavenet.py --data_dir ./data/processed --gpu 0

4. Run inference:
   python inference.py --emg_path <test_emg.npy>

5. Evaluate:
   python evaluate.py --audio_dir ./outputs --asr whisper
    """)


if __name__ == '__main__':
    main()
