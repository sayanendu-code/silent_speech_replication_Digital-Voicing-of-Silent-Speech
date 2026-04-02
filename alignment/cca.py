"""
alignment/cca.py — Canonical Correlation Analysis for alignment refinement.

Section 3.2.1: "CCA finds linear projections P_S and P_V that maximize
correlation between corresponding dimensions of P_S·v_S and P_V·v_V."

We use CCA-projected EMG features for a refined DTW cost:
    δ_CCA[i,j] = || P_S · E'_S[i]  -  P_V · E'_V[j] ||
"""

import numpy as np
from sklearn.cross_decomposition import CCA
from typing import Tuple, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AlignmentConfig


class CCAAligner:
    """
    Learns CCA projections from (silent EMG, vocalized EMG) pairs,
    then provides a DTW cost function in CCA space.
    """

    def __init__(self, n_components: int = 15):
        self.n_components = n_components
        self.cca = CCA(n_components=n_components, max_iter=1000)
        self.fitted = False

    def fit(self, silent_features: np.ndarray,
            vocalized_features: np.ndarray):
        """
        Fit CCA on paired (E'_S, E'_V) feature vectors.

        "We aggregate aligned E'_S and E'_V features over the entire
         dataset and feed these to a CCA algorithm."

        Args:
            silent_features: (N_total, D) concatenated aligned silent feats.
            vocalized_features: (N_total, D) concatenated aligned vocalized feats.
        """
        assert silent_features.shape == vocalized_features.shape
        self.cca.fit(silent_features, vocalized_features)
        self.fitted = True
        return self

    def project_silent(self, features: np.ndarray) -> np.ndarray:
        """
        Project silent EMG features: P_S · E'_S.
        Uses the X rotation learned by CCA.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        X_centered = features - self.cca._x_mean
        return X_centered @ self.cca.x_rotations_

    def project_vocalized(self, features: np.ndarray) -> np.ndarray:
        """
        Project vocalized EMG features: P_V · E'_V.
        Uses the Y rotation learned by CCA.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        Y_centered = features - self.cca._y_mean
        return Y_centered @ self.cca.y_rotations_

    def project_vocalized_y(self, features: np.ndarray) -> np.ndarray:
        """Alias for project_vocalized."""
        return self.project_vocalized(features)

    def cca_cost_matrix(self, silent_emg: np.ndarray,
                        vocalized_emg: np.ndarray) -> np.ndarray:
        """
        Compute δ_CCA[i,j] = || P_S·E'_S[i] - P_V·E'_V[j] ||

        Args:
            silent_emg: (M, D) silent EMG features.
            vocalized_emg: (N, D) vocalized EMG features.

        Returns:
            cost: (M, N) pairwise distance in CCA space.
        """
        assert self.fitted, "Call fit() first."

        proj_s = self.project_silent(silent_emg)        # (M, n_components)
        proj_v = self.project_vocalized_y(vocalized_emg)  # (N, n_components)

        # Euclidean distance in CCA space
        s_sq = np.sum(proj_s ** 2, axis=1, keepdims=True)
        v_sq = np.sum(proj_v ** 2, axis=1, keepdims=True)
        cross = proj_s @ proj_v.T
        dist_sq = np.maximum(s_sq + v_sq.T - 2 * cross, 0.0)
        return np.sqrt(dist_sq)

    def state_dict(self):
        """Serialize for checkpointing."""
        return {
            'n_components': self.n_components,
            'cca_params': {
                'x_weights_': self.cca.x_weights_,
                'y_weights_': self.cca.y_weights_,
                'x_rotations_': self.cca.x_rotations_,
                'y_rotations_': self.cca.y_rotations_,
                '_x_mean': self.cca._x_mean,
                '_y_mean': self.cca._y_mean,
            },
            'fitted': self.fitted,
        }

    def load_state_dict(self, d):
        self.n_components = d['n_components']
        for k, v in d['cca_params'].items():
            setattr(self.cca, k, v)
        self.fitted = d['fitted']


def collect_aligned_pairs(silent_emg_list, vocalized_emg_list,
                          alignments):
    """
    Aggregate aligned feature pairs from all parallel utterances.

    Args:
        silent_emg_list: List of (M_i, D) arrays.
        vocalized_emg_list: List of (N_i, D) arrays.
        alignments: List of (M_i,) alignment arrays.

    Returns:
        all_silent: (N_total, D) concatenated silent features.
        all_vocalized: (N_total, D) concatenated warped vocalized features.
    """
    all_s, all_v = [], []
    for es, ev, align in zip(silent_emg_list, vocalized_emg_list, alignments):
        all_s.append(es)
        all_v.append(ev[align])  # warp vocalized to silent timing
    return np.concatenate(all_s, axis=0), np.concatenate(all_v, axis=0)
