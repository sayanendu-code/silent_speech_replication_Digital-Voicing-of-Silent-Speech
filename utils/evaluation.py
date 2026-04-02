"""
utils/evaluation.py — Automatic intelligibility evaluation.

Section 4.2.2: "We perform an automatic evaluation by transcribing
system outputs with a large-vocabulary automatic speech recognition
(ASR) system … we use the open source implementation of DeepSpeech
from Mozilla."

We support multiple ASR backends:
  1. Mozilla DeepSpeech (original paper)
  2. OpenAI Whisper (modern alternative, much better)
  3. Manual transcription file loading
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple


# ─── Word Error Rate ──────────────────────────────────────────────────────────

def word_error_rate(reference: str, hypothesis: str) -> Tuple[float, Dict]:
    """
    Section 4.1:
        WER = (substitutions + insertions + deletions) / reference_length

    Returns:
        wer: Float WER value.
        details: Dict with S, I, D counts.
    """
    ref = reference.lower().strip().split()
    hyp = hypothesis.lower().strip().split()

    n, m = len(ref), len(hyp)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = 'match'
            else:
                candidates = [
                    (dp[i - 1][j] + 1, 'deletion'),
                    (dp[i][j - 1] + 1, 'insertion'),
                    (dp[i - 1][j - 1] + 1, 'substitution'),
                ]
                dp[i][j], ops[i][j] = min(candidates, key=lambda x: x[0])

    # Count operations
    substitutions, insertions, deletions = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ops[i][j] == 'match':
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and ops[i][j] == 'substitution':
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or ops[i][j] == 'insertion'):
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1

    wer = dp[n][m] / max(n, 1)
    details = {
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions,
        'ref_length': n,
        'hyp_length': m,
        'errors': dp[n][m],
    }
    return wer, details


def batch_wer(references: List[str], hypotheses: List[str]) -> Dict:
    """
    Compute WER across a batch of utterances.

    Returns aggregate WER (sum of errors / sum of ref lengths).
    """
    assert len(references) == len(hypotheses)

    total_errors = 0
    total_ref_len = 0
    per_utt = []

    for ref, hyp in zip(references, hypotheses):
        wer, details = word_error_rate(ref, hyp)
        total_errors += details['errors']
        total_ref_len += details['ref_length']
        per_utt.append({
            'reference': ref,
            'hypothesis': hyp,
            'wer': wer,
            **details,
        })

    aggregate_wer = total_errors / max(total_ref_len, 1)
    return {
        'aggregate_wer': aggregate_wer,
        'total_errors': total_errors,
        'total_ref_words': total_ref_len,
        'n_utterances': len(references),
        'per_utterance': per_utt,
    }


# ─── ASR Backends ─────────────────────────────────────────────────────────────

class ASRBackend:
    """Base class for ASR transcription."""

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError

    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        return [self.transcribe(p) for p in audio_paths]


class WhisperASR(ASRBackend):
    """
    OpenAI Whisper — modern, high-quality ASR.
    Much better than DeepSpeech; recommended for new experiments.

    Install: pip install openai-whisper
    """

    def __init__(self, model_size: str = "base.en"):
        try:
            import whisper
        except ImportError:
            raise ImportError("Install whisper: pip install openai-whisper")
        print(f"Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> str:
        import whisper
        result = self.model.transcribe(audio_path, language="en")
        return result["text"].strip()


class DeepSpeechASR(ASRBackend):
    """
    Mozilla DeepSpeech — used in the original paper (Section 4.2.2).

    "Running the recognizer on the original vocalized audio recordings
     from the test set results in a WER of 9.5%, which represents a
     lower bound for this evaluation."

    Install: pip install deepspeech
    Download model: https://github.com/mozilla/DeepSpeech/releases
    """

    def __init__(self, model_path: str, scorer_path: Optional[str] = None):
        try:
            import deepspeech
        except ImportError:
            raise ImportError("Install deepspeech: pip install deepspeech")
        self.model = deepspeech.Model(model_path)
        if scorer_path:
            self.model.enableExternalScorer(scorer_path)

    def transcribe(self, audio_path: str) -> str:
        import wave
        import deepspeech
        with wave.open(audio_path, 'rb') as w:
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
        return self.model.stt(audio)


class ManualTranscriptions(ASRBackend):
    """Load human transcriptions from a JSON/CSV file."""

    def __init__(self, transcription_file: str):
        with open(transcription_file) as f:
            if transcription_file.endswith('.json'):
                self.transcriptions = json.load(f)
            else:
                # CSV: filename,transcription
                self.transcriptions = {}
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        self.transcriptions[parts[0]] = parts[1]

    def transcribe(self, audio_path: str) -> str:
        key = os.path.basename(audio_path)
        return self.transcriptions.get(key, "")


# ─── Full Evaluation Pipeline ─────────────────────────────────────────────────

def evaluate_model(audio_dir: str, reference_texts: Dict[str, str],
                   asr_backend: str = "whisper",
                   asr_kwargs: Optional[Dict] = None) -> Dict:
    """
    Full evaluation pipeline:
      1. List generated audio files
      2. Transcribe with ASR
      3. Compute WER against references

    Args:
        audio_dir: Directory containing generated .wav files.
        reference_texts: {filename: reference_text} mapping.
        asr_backend: "whisper", "deepspeech", or "manual".
        asr_kwargs: Backend-specific arguments.

    Returns:
        Results dict with aggregate and per-utterance WER.
    """
    if asr_kwargs is None:
        asr_kwargs = {}

    # Initialize ASR
    if asr_backend == "whisper":
        asr = WhisperASR(**asr_kwargs)
    elif asr_backend == "deepspeech":
        asr = DeepSpeechASR(**asr_kwargs)
    elif asr_backend == "manual":
        asr = ManualTranscriptions(**asr_kwargs)
    else:
        raise ValueError(f"Unknown ASR backend: {asr_backend}")

    # Find audio files
    audio_files = sorted([
        f for f in os.listdir(audio_dir) if f.endswith('.wav')
    ])

    references = []
    hypotheses = []
    for fname in audio_files:
        if fname in reference_texts:
            audio_path = os.path.join(audio_dir, fname)
            hyp = asr.transcribe(audio_path)
            references.append(reference_texts[fname])
            hypotheses.append(hyp)
            print(f"  {fname}: REF='{reference_texts[fname][:50]}' | "
                  f"HYP='{hyp[:50]}'")

    # Compute WER
    results = batch_wer(references, hypotheses)
    print(f"\n{'='*60}")
    print(f"Aggregate WER: {results['aggregate_wer']*100:.1f}%")
    print(f"Total errors: {results['total_errors']} / "
          f"{results['total_ref_words']} words")
    print(f"Utterances evaluated: {results['n_utterances']}")

    # Save results
    results_path = os.path.join(audio_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results
