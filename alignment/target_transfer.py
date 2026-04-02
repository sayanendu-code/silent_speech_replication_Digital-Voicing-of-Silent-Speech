"""
alignment/target_transfer.py — Full audio target transfer pipeline.

Section 3.2: "Using a set of utterances recorded in both silent and
vocalized speaking modes, we find alignments between the two recordings
and use them to associate speech features from the vocalized instance
(A'_V) with the silent EMG E'_S."

Pipeline:
  1. Raw DTW with Euclidean EMG distance         (Section 3.2)
  2. CCA projection → refined DTW                (Section 3.2.1)
  3. Predicted audio + CCA → combined DTW         (Section 3.2.2)
"""

import numpy as np
from typing import List, Tuple, Optional

from alignment.dtw import (
    dtw_alignment, _euclidean_cost, combined_cost, warp_features
)
from alignment.cca import CCAAligner, collect_aligned_pairs


class TargetTransfer:
    """
    Manages the full alignment lifecycle for training on silent EMG.

    Training schedule from Section 3.2.2:
      - Epochs 1-4: align with δ_CCA only.
      - Epoch 5: use partially-trained transducer → δ_full.
      - Every 5 epochs thereafter: re-compute alignments with δ_full.
    """

    def __init__(self, cca_n_components: int = 15, lambda_audio: float = 10.0,
                 warmup_epochs: int = 4, realign_interval: int = 5):
        self.cca_aligner = CCAAligner(n_components=cca_n_components)
        self.lambda_audio = lambda_audio
        self.warmup_epochs = warmup_epochs
        self.realign_interval = realign_interval
        self.cca_fitted = False

    # ── Phase 1: Initial raw DTW alignment ─────────────────────────────

    def initial_alignment(self,
                          silent_emg_list: List[np.ndarray],
                          vocalized_emg_list: List[np.ndarray]
                          ) -> List[np.ndarray]:
        """
        "Initially, we use euclidean distance between the features
         of E'_S and E'_V for the alignment cost."

        Returns list of alignment arrays.
        """
        alignments = []
        for es, ev in zip(silent_emg_list, vocalized_emg_list):
            align = dtw_alignment(es, ev)
            alignments.append(align)
        return alignments

    # ── Phase 2: Fit CCA and re-align ──────────────────────────────────

    def fit_cca_and_realign(self,
                            silent_emg_list: List[np.ndarray],
                            vocalized_emg_list: List[np.ndarray],
                            initial_alignments: List[np.ndarray]
                            ) -> List[np.ndarray]:
        """
        Section 3.2.1:
          1. Collect aligned pairs from initial DTW.
          2. Fit CCA to learn projections P_S, P_V.
          3. Re-run DTW with δ_CCA cost.
        """
        # Collect pairs for CCA
        all_s, all_v = collect_aligned_pairs(
            silent_emg_list, vocalized_emg_list, initial_alignments
        )
        self.cca_aligner.fit(all_s, all_v)
        self.cca_fitted = True

        # Re-align with CCA cost
        alignments = []
        for es, ev in zip(silent_emg_list, vocalized_emg_list):
            cost = self.cca_aligner.cca_cost_matrix(es, ev)
            align = dtw_alignment(es, ev, cost_fn=lambda s1, s2: cost)
            alignments.append(align)
        return alignments

    # ── Phase 3: Refine with predicted audio ───────────────────────────

    def realign_with_audio(self,
                           silent_emg_list: List[np.ndarray],
                           vocalized_emg_list: List[np.ndarray],
                           predicted_audio_list: List[np.ndarray],
                           vocalized_audio_list: List[np.ndarray]
                           ) -> List[np.ndarray]:
        """
        Section 3.2.2:
          δ_full[i,j] = δ_CCA[i,j] + λ · ||Â'_S[i] - A'_V[j]||
        """
        assert self.cca_fitted, "Must fit CCA before audio refinement."

        alignments = []
        for es, ev, pred_a, true_a in zip(
            silent_emg_list, vocalized_emg_list,
            predicted_audio_list, vocalized_audio_list
        ):
            # CCA-based EMG cost
            emg_cost = self.cca_aligner.cca_cost_matrix(es, ev)

            # Audio-based cost: ||Â'_S[i] - A'_V[j]||
            audio_cost = _euclidean_cost(pred_a, true_a)

            # Combined cost
            full_cost = combined_cost(emg_cost, audio_cost, self.lambda_audio)

            align = dtw_alignment(es, ev, cost_fn=lambda s1, s2: full_cost)
            alignments.append(align)
        return alignments

    # ── Generate training targets ──────────────────────────────────────

    def generate_targets(self,
                         vocalized_audio_list: List[np.ndarray],
                         alignments: List[np.ndarray]
                         ) -> List[np.ndarray]:
        """
        Create warped audio targets: Ã'_S[i] = A'_V[alignment[i]]

        These serve as training targets for the transducer on silent EMG.
        """
        targets = []
        for audio_feats, align in zip(vocalized_audio_list, alignments):
            warped = warp_features(audio_feats, align)
            targets.append(warped)
        return targets

    # ── Epoch-level decision logic ─────────────────────────────────────

    def should_realign(self, epoch: int) -> bool:
        """Whether to recompute alignments at this epoch."""
        if epoch < self.warmup_epochs:
            return False
        if epoch == self.warmup_epochs:
            return True  # First audio-based realignment
        return (epoch - self.warmup_epochs) % self.realign_interval == 0
