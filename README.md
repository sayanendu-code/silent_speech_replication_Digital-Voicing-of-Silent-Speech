# Digital Voicing of Silent Speech — Paper 1 Replication

> **Paper**: Gaddy & Klein, EMNLP 2020 (★ Best Paper Award)
> **PDF**: https://aclanthology.org/2020.emnlp-main.445.pdf
> **Dataset**: https://doi.org/10.5281/zenodo.4064408
> **Target WER**: 68.0% (open vocab) / 3.6% (closed vocab)

---

## The Task

```
Person silently mouths words → EMG electrodes capture muscle signals → Model → Audible speech
```

---

## Hardware Setup

```
                    ┌──────────────────────────────────┐
                    │         HUMAN FACE                │
                    │                                    │
                    │    [7]●          ●[8]              │
                    │        nose area   back cheek      │
                    │                                    │
                    │  [1]●    mouth    ●[6]             │
                    │   left cheek      right cheek      │
                    │                                    │
                    │       [2]●   chin                  │
                    │                                    │
                    │       [3]●   below chin            │
                    │                                    │
                    │   [5]●           jaw               │
                    │                                    │
                    │       [4]●   throat                │
                    │        (near Adam's apple)         │
                    │                                    │
                    │  [ref]●              ●[bias]       │
                    │  behind              behind        │
                    │  left ear            right ear     │
                    └──────────────────────────────────┘

  8 active electrodes + 1 reference + 1 bias
  Gold-plated electrodes with Ten20 conductive paste
  OpenBCI Cyton board, WiFi, 1000 Hz per channel
```

---

## The Dataset

```
  emg_data/   (18.6 hours total, single male English speaker)
  │
  ├── silent_parallel_data/          3.6h, 7 sessions, ~2188 utterances
  │   Speaker mouths words WITHOUT producing sound.
  │
  ├── voiced_parallel_data/          3.9h, 7 sessions, ~2068 utterances
  │   SAME utterances spoken aloud. EMG + audio simultaneously.
  │
  ├── nonparallel_data/              11.2h, 10 sessions, ~6833 utterances
  │   Vocalized speech only. No silent counterpart.
  │
  └── closed_vocab/                  ~1h, 500 date/time utterances
      Only 67 unique words. For the 3.6% WER experiment.

  Per utterance:
    {id}_emg.npy             (N, 8)    float64 @ 1000 Hz
    {id}_audio_clean.flac    (M,)      float64 @ 16,000 Hz
    {id}_info.json           {"text": "...", "book": "...", ...}
```

---

## Complete Architecture — End to End

```
═══════════════════════════════════════════════════════════════════════
 RAW INPUT                    PREPROCESSING               FEATURE SPACE
═══════════════════════════════════════════════════════════════════════

 8 electrodes                Butterworth HP 2Hz           112 dimensions
 1000 Hz each                Notch 60Hz+harmonics         100 Hz
 ~3 seconds                  Zero phase delay             ~300 frames

 ┌─────────────────┐         ┌──────────────┐            ┌──────────────┐
 │ ch1: ∿∿∿∿∿∿∿∿∿  │ filter  │ ch1: ∿∿∿∿∿∿  │  extract   │ frame 0:     │
 │ ch2: ∿∿∿∿∿∿∿∿∿  │───────► │ ch2: ∿∿∿∿∿∿  │──────────► │  [112 vals]  │
 │ ...              │ remove  │ ...            │  27ms win  │ frame 1:     │
 │ ch8: ∿∿∿∿∿∿∿∿∿  │ noise   │ ch8: ∿∿∿∿∿∿  │  10ms hop  │  [112 vals]  │
 └─────────────────┘         └──────────────┘            │ ...          │
  (3000, 8)                   (3000, 8)                   │ frame 299:   │
                                                          │  [112 vals]  │
                                                          └──────────────┘
                                                           (300, 112)
```

---

## Feature Extraction — What those 112 dimensions are

```
  For EACH of 8 channels, for EACH 27ms frame:

  Raw channel signal (27 samples)
       │
       ├───────────────────────────────────────┐
       │                                       │
       ▼                                       ▼
  Triangular lowpass at 134 Hz           16-point STFT
       │                                       │
       ├──► x_low  (< 134 Hz)                 ▼
       │       │                          magnitude of
       │       ├──► mean(x_low²)    ◄1    9 freq bins
       │       └──► mean(x_low)     ◄2         │
       │                                       ├──► bin 0  ◄6
       └──► x_high (> 134 Hz)                 ├──► bin 1  ◄7
               │                               ├──► bin 2  ◄8
               ├──► mean(x_high²)   ◄3         ├──► ...
               ├──► mean(|x_high|)  ◄4         └──► bin 8  ◄14
               └──► ZCR(x_high)     ◄5
                                          9 frequency features
               5 time-domain features

  Per channel:  5 + 9 = 14 features
  Per frame:    14 × 8 channels = 112 features

  ┌────────────────────────────────────────────────────────────────┐
  │ One frame (112-dim vector):                                    │
  │                                                                │
  │ ┌── ch1 (14) ──┐ ┌── ch2 (14) ──┐     ┌── ch8 (14) ──┐     │
  │ │ td td td td  │ │ td td td td  │ ... │ td td td td  │     │
  │ │ td stft × 9  │ │ td stft × 9  │     │ td stft × 9  │     │
  │ └──────────────┘ └──────────────┘     └──────────────┘     │
  │  ◄──── 14 ─────►                        ◄──── 14 ─────►     │
  │  ◄───────────────────── 112 total ──────────────────────►     │
  └────────────────────────────────────────────────────────────────┘
```

---

## Audio Target Features — 26 MFCCs

```
  Vocalized audio waveform (16,000 Hz)
       │
       ▼
  For each 27ms frame with 10ms stride:
       │
       432 samples → Hamming window → FFT → power spectrum
       │
       ▼
       128 Mel-scale triangular filter bank
       │
       ▼
       log(filter outputs) → DCT → keep first 26 coefficients
       │
       ▼
  One 26-dim MFCC vector per frame

  MFCC 1:  overall loudness
  MFCC 2:  spectral tilt (vowel identity)
  MFCC 3+: increasingly fine spectral detail
  MFCC 26: finest detail (speaker characteristics)

  Output: (T, 26) at 100 Hz — same rate as EMG features
```

---

## Session Embedding — 32 dimensions

```
  Problem: electrodes reattached between sessions → position shifts.

  ┌────────────────────────────────────────────────────────────┐
  │  session_id = 13                                           │
  │       │                                                    │
  │       ▼                                                    │
  │  nn.Embedding(24 sessions, 32 dim)                        │
  │       │                                                    │
  │       ▼                                                    │
  │  [0.12, -0.34, 0.56, ..., 0.78]   ← 32-dim vector       │
  │       │                                                    │
  │       ▼                                                    │
  │  BROADCAST + CONCATENATE with EMG features:               │
  │                                                            │
  │  ┌───────────────────────────────────────────┐            │
  │  │ Frame 0: [emg × 112 | session × 32] = 144│            │
  │  │ Frame 1: [emg × 112 | session × 32] = 144│ same 32   │
  │  │ ...                                       │ every     │
  │  │ Frame T: [emg × 112 | session × 32] = 144│ frame     │
  │  └───────────────────────────────────────────┘            │
  │                                                            │
  │  Model learns: "session 13 = electrode shifted 2mm up"    │
  └────────────────────────────────────────────────────────────┘
```

---

## The Transducer — BiLSTM Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Input: (batch, T, 144)  ← EMG (112) + session (32)            │
  │                                                                  │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │ Dropout(0.5)           "before the first LSTM"           │   │
  │  │      │                                                    │   │
  │  │      ▼                                                    │   │
  │  │ ┌────────────────────────────────────────────────────┐   │   │
  │  │ │ BiLSTM Layer 1                                     │   │   │
  │  │ │                                                     │   │   │
  │  │ │ Forward:  h₁=f(x₁,h₀) → h₂=f(x₂,h₁) → ... → hₜ │   │   │
  │  │ │           144 input → 1024 hidden                   │   │   │
  │  │ │                                                     │   │   │
  │  │ │ Backward: hₜ=f(xₜ,hₜ₊₁) → ... → h₁               │   │   │
  │  │ │           144 input → 1024 hidden                   │   │   │
  │  │ │                                                     │   │   │
  │  │ │ Concat: 1024 + 1024 = 2048 per frame               │   │   │
  │  │ └────────────────────────────────────────────────────┘   │   │
  │  │      │                                                    │   │
  │  │ Dropout(0.5)           "between all layers"              │   │
  │  │      │                                                    │   │
  │  │ ┌────────────────────────────────────────────────────┐   │   │
  │  │ │ BiLSTM Layer 2                                     │   │   │
  │  │ │ 2048 → 1024 fwd + 1024 bwd → 2048                 │   │   │
  │  │ └────────────────────────────────────────────────────┘   │   │
  │  │      │                                                    │   │
  │  │ Dropout(0.5)                                              │   │
  │  │      │                                                    │   │
  │  │ ┌────────────────────────────────────────────────────┐   │   │
  │  │ │ BiLSTM Layer 3                                     │   │   │
  │  │ │ 2048 → 1024 fwd + 1024 bwd → 2048                 │   │   │
  │  │ └────────────────────────────────────────────────────┘   │   │
  │  │      │                                                    │   │
  │  │ Dropout(0.5)           "after the last LSTM"             │   │
  │  └──────────────────────────────────────────────────────────┘   │
  │       │                                                          │
  │       ▼                                                          │
  │  Linear(2048 → 26)                                              │
  │       │                                                          │
  │       ▼                                                          │
  │  Output: (batch, T, 26)  ← predicted MFCCs                     │
  │                                                                  │
  │  ~60M parameters total                                           │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Audio Target Transfer — How silent EMG gets training labels

```
  THE PROBLEM: Silent EMG has NO audio. Need MFCC targets for MSE loss.

  SOLUTION: Borrow MFCCs from vocalized recording via alignment.

  ┌──────────────────────────────────────────────────────────────────┐
  │ PHASE 1 — Raw DTW                                               │
  │                                                                  │
  │ E'_S (313, 112)  ←→  E'_V (265, 112)                          │
  │ silent features       vocalized features                        │
  │                                                                  │
  │ Cost: δ[i,j] = ‖E'_S[i] − E'_V[j]‖  (Euclidean)             │
  │                                                                  │
  │ DTW finds cheapest monotonic path:                              │
  │                                                                  │
  │      E'_V →                                                     │
  │   E  ┌──────────────────────┐                                   │
  │   '  │●─●                   │                                   │
  │   S  │    ╲                 │  Path gives alignment[i] → j     │
  │   │  │     ●─●             │  "silent frame i matches          │
  │   ▼  │         ╲           │   vocalized frame j"              │
  │      │          ●─●─●     │                                   │
  │      │                ╲   │  Warp: Ã'_S[i] = A'_V[align[i]] │
  │      │                 ●─●│                                   │
  │      └──────────────────────┘                                   │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ PHASE 2 — CCA Refinement                                        │
  │                                                                  │
  │ Problem: silent EMG ≠ vocalized EMG (no voicing, different artic)│
  │                                                                  │
  │ CCA learns projections P_S, P_V (112 → 15 dim):               │
  │   maximize correlation between P_S·E'_S and P_V·E'_V           │
  │                                                                  │
  │ New cost: δ_CCA[i,j] = ‖P_S·E'_S[i] − P_V·E'_V[j]‖         │
  │                                                                  │
  │ Re-run DTW → tighter alignment → better targets                │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │ PHASE 3 — Predicted Audio Refinement (epoch 5+, every 5 epochs)│
  │                                                                  │
  │ Model now predicts rough MFCCs Â_S from silent EMG.             │
  │                                                                  │
  │ New cost: δ_full = δ_CCA + 10 × ‖Â_S[i] − A'_V[j]‖          │
  │                           λ=10                                   │
  │                                                                  │
  │ "Does predicted audio match real audio?" → much better align    │
  │                                                                  │
  │ Every 5 epochs: re-align → new targets → resume training        │
  │ Positive feedback: better model → better align → better model  │
  └──────────────────────────────────────────────────────────────────┘
```

---

## WaveNet Vocoder

```
  Trained SEPARATELY. Never sees EMG. Learns: MFCCs → natural speech.

  ┌──────────────────────────────────────────────────────────────────┐
  │ CONDITIONING:                                                    │
  │   Gold MFCCs (T, 26) @ 100 Hz                                  │
  │     → BiLSTM(512, bidirectional) → (T, 1024)                   │
  │     → Linear(1024 → 128)                                       │
  │     → ConvTranspose1d (upsample ×160: 100Hz → 16kHz)           │
  │                                                                  │
  │ WAVENET CORE:                                                    │
  │   16 dilated causal conv layers                                  │
  │   Dilations: 1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128        │
  │   Each layer: dilated_conv → tanh ⊗ sigmoid + conditioning     │
  │              → skip (256) + residual (64)                       │
  │   Output: softmax over 256 μ-law levels → next audio sample    │
  │                                                                  │
  │ AUTOREGRESSIVE: sample t depends on samples 0..t-1             │
  │ 3-second utterance = 48,000 sequential forward passes          │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Results

| Setting | Baseline | Paper 1 |
|---|---|---|
| Closed vocab (human) | 64.6% WER | **3.6%** WER |
| Open vocab (ASR) | 88.0% WER | **68.0%** WER |
| Open vocab (human) | 95.1% WER | **74.8%** WER |

---

## Quick Start

```bash
pip install -r requirements.txt
python download_data.py
python data/preprocessing.py --raw_dir ./data/raw/emg_data
python train_transducer.py --gpu 0
python train_transducer.py --gpu 0 --closed_vocab
python train_wavenet.py --gpu 0
python evaluate.py --audio_dir ./outputs --ref_file refs.json --asr whisper
```