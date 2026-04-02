# Digital Voicing of Silent Speech — Full Replication

A complete, modular reimplementation of **Gaddy & Klein (2020)**: *"Digital Voicing of Silent Speech"* (EMNLP 2020).

> **Paper**: [https://nlp.cs.berkeley.edu/pubs/Gaddy-Klein_2020_DigitalVoicing_paper.pdf](https://nlp.cs.berkeley.edu/pubs/Gaddy-Klein_2020_DigitalVoicing_paper.pdf)  
> **Original code**: [https://github.com/dgaddy/silent_speech](https://github.com/dgaddy/silent_speech)  
> **Dataset**: [https://doi.org/10.5281/zenodo.4064408](https://doi.org/10.5281/zenodo.4064408)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
│                                                                 │
│  ┌──────────┐    ┌──────────────────────────────────┐          │
│  │ Silent   │    │   Audio Target Transfer (§3.2)    │          │
│  │ EMG  E_S │───▶│  DTW → CCA → Predicted Audio    │──▶ Ã'_S  │
│  └──────────┘    │  alignment refinement            │          │
│  ┌──────────┐    └──────────────────────────────────┘          │
│  │ Vocal    │───────────────────────────────────────────▶ A'_V  │
│  │ EMG  E_V │                                                   │
│  └──────────┘                                                   │
│  ┌──────────┐                                                   │
│  │ Vocal    │──▶ MFCCs ──▶ WaveNet Training                    │
│  │ Audio A_V│                                                   │
│  └──────────┘                                                   │
│                                                                 │
│  ┌─────────────────────────────────────────────┐               │
│  │        BiLSTM Transducer (§3.1)             │               │
│  │  E' + SessionEmbed → BiLSTM×3 → Linear → Â' │               │
│  │  Loss: MSE(Â', Ã'_S)  +  MSE(Â', A'_V)     │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
│                                                                 │
│  Silent EMG                                                     │
│      │                                                          │
│      ▼                                                          │
│  Filter + Extract Features (112-dim @ 100Hz)                    │
│      │                                                          │
│      ▼                                                          │
│  BiLSTM Transducer → Predicted MFCCs (26-dim @ 100Hz)          │
│      │                                                          │
│      ▼                                                          │
│  WaveNet Decoder → Audio Waveform (16 kHz)                      │
│      │                                                          │
│      ▼                                                          │
│  🔊 Synthesized Speech                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
silent_speech/
├── config.py                    # All hyperparameters (§3.1, §3.3, App A)
│
├── features/
│   ├── emg_features.py          # EMG filtering + 112-dim extraction (§2.3, §3.1)
│   └── audio_features.py        # 26 MFCCs + μ-law encoding (§3.1, §3.3)
│
├── alignment/
│   ├── dtw.py                   # Dynamic Time Warping (§3.2)
│   ├── cca.py                   # Canonical Correlation Analysis (§3.2.1)
│   └── target_transfer.py       # Full 3-phase alignment pipeline (§3.2)
│
├── models/
│   ├── transducer.py            # 3-layer BiLSTM + session embedding (§3.1)
│   └── wavenet.py               # WaveNet + conditioning network (§3.3, App A)
│
├── data/
│   ├── dataset.py               # PyTorch datasets + combined loader (§2)
│   └── preprocessing.py         # Raw → normalized features pipeline (§2.3)
│
├── utils/
│   ├── helpers.py               # Logging, checkpointing, reproducibility
│   └── evaluation.py            # WER computation + ASR backends (§4)
│
├── train_transducer.py          # Transducer training with re-alignment (§3.1-3.2)
├── train_wavenet.py             # WaveNet training, batch_size=1 (§3.3)
├── inference.py                 # End-to-end EMG → audio pipeline (§4)
├── evaluate.py                  # Standalone WER evaluation
├── download_data.py             # Dataset download from Zenodo
├── main.py                      # Unified CLI entry point
├── requirements.txt
└── tests/
    └── test_all.py              # Unit + integration tests
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
python download_data.py --output_dir ./data/raw
```

Or manually from [Zenodo](https://doi.org/10.5281/zenodo.4064408).

### 3. Preprocess

```bash
python main.py preprocess
```

This runs the full pipeline: raw EMG → filtered → 112-dim features at 100 Hz, raw audio → 26 MFCCs, then normalizes everything to zero mean / unit variance.

### 4. Train

```bash
# Train the BiLSTM transducer (with target transfer alignment)
python train_transducer.py --data_dir ./data/processed --gpu 0

# Train the WaveNet decoder
python train_wavenet.py --data_dir ./data/processed --gpu 0
```

Or run everything at once:

```bash
python main.py all --gpu 0
```

### 5. Inference

```bash
python inference.py \
    --emg_path path/to/silent_emg.npy \
    --transducer_ckpt ./checkpoints/best_transducer.pt \
    --wavenet_ckpt ./checkpoints/wavenet/best_wavenet.pt \
    --output_dir ./outputs
```

### 6. Evaluate

```bash
python evaluate.py \
    --audio_dir ./outputs \
    --ref_file references.json \
    --asr whisper
```

---

## Paper ↔ Code Mapping

| Paper Section | What | Code Location |
|---|---|---|
| §2.3 | EMG filtering (2Hz HP, 60Hz notch) | `features/emg_features.py :: preprocess_emg()` |
| §3.1 | Time-domain + STFT features (112-dim) | `features/emg_features.py :: extract_emg_features()` |
| §3.1 | 26 MFCCs at 100 Hz | `features/audio_features.py :: extract_mfcc()` |
| §3.1 | Session embedding (32-dim) | `models/transducer.py :: SessionEmbedding` |
| §3.1 | 3-layer BiLSTM (1024 hidden) | `models/transducer.py :: EMGTransducer` |
| §3.2 | DTW alignment | `alignment/dtw.py :: dtw_alignment()` |
| §3.2.1 | CCA refinement (15 components) | `alignment/cca.py :: CCAAligner` |
| §3.2.2 | Predicted audio refinement (λ=10) | `alignment/target_transfer.py :: realign_with_audio()` |
| §3.2.2 | Re-alignment schedule (warmup 4, every 5) | `alignment/target_transfer.py :: should_realign()` |
| §3.3 | WaveNet (16 layers, dilation 128) | `models/wavenet.py :: WaveNet` |
| §3.3 | Conditioning BiLSTM (512) + proj (128) | `models/wavenet.py :: ConditioningNetwork` |
| App A | All WaveNet hyperparameters | `config.py :: WaveNetConfig` |
| §4 | WER evaluation | `utils/evaluation.py :: word_error_rate()` |
| §4.2.2 | ASR-based automatic eval | `utils/evaluation.py :: evaluate_model()` |

---

## Key Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| EMG channels | 8 | Table 3 |
| EMG sample rate | 1000 Hz | §2.3 |
| Audio sample rate | 16 kHz | §2.3 |
| Feature frame length | 27 ms | §3.1 |
| Feature stride | 10 ms (→ 100 Hz) | §3.1 |
| EMG features per frame | 112 (14 × 8) | §3.1 |
| MFCC dimension | 26 | §3.1 |
| Session embedding dim | 32 | §3.1 |
| LSTM hidden | 1024 | §3.1 |
| LSTM layers | 3 (bidirectional) | §3.1 |
| Dropout | 0.5 | §3.1 |
| Learning rate | 0.001 | §3.1 |
| LR decay | ×0.5 after 5 epochs no improvement | §3.1 |
| CCA components | 15 | §3.2.1 |
| λ (audio alignment weight) | 10.0 | §3.2.2 |
| WaveNet layers | 16 | App A |
| WaveNet max dilation | 128 | App A |
| WaveNet residual channels | 64 | App A |
| WaveNet skip channels | 256 | App A |
| WaveNet conditioning dim | 128 | App A |
| μ-law quantization levels | 256 | App A |

---

## Expected Results

| Setting | Metric | Baseline | Our Model |
|---|---|---|---|
| Closed vocab (human) | WER | 64.6% | **3.6%** |
| Open vocab (human) | WER | 95.1% | **74.8%** |
| Open vocab (ASR) | WER | 88.0% | **68.0%** |
| Open vocab (no CCA) | WER | — | 69.8% |
| Open vocab (no audio align) | WER | — | 76.5% |

*From Tables 4 and 5 in the paper.*

---

## Testing

```bash
pytest tests/test_all.py -v
```

Tests cover: feature extraction shapes, DTW monotonicity, CCA projections, model forward passes, WER computation, and the full integration pipeline.

---

## Citation

```bibtex
@inproceedings{gaddy2020digital,
  title={Digital Voicing of Silent Speech},
  author={Gaddy, David and Klein, Dan},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```

---

## License

This is a research reimplementation. The original dataset is available under the terms specified at [Zenodo](https://doi.org/10.5281/zenodo.4064408).
