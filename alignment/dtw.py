"""
alignment/dtw.py — Dynamic Time Warping for silent ↔ vocalized alignment.

Section 3.2: "DTW builds a table d[i,j] of the minimum cost of alignment
between the first i items in s1 and the first j items in s2."

The mapping takes the first pair from any repeated-i sequence →
a function a_{SV}[i] → j for every position i in E'_S.
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit
def _dtw_cost_matrix(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the accumulated cost matrix and backpointer matrix.

    Args:
        cost: (M, N) local cost matrix δ[i, j].

    Returns:
        D: (M, N) accumulated cost.
        bp: (M, N, 2) backpointers — bp[i, j] = (prev_i, prev_j).
    """
    M, N = cost.shape
    D = np.full((M, N), np.inf, dtype=np.float64)
    bp = np.zeros((M, N, 2), dtype=np.int64)

    D[0, 0] = cost[0, 0]
    for j in range(1, N):
        D[0, j] = D[0, j - 1] + cost[0, j]
        bp[0, j, 0] = 0
        bp[0, j, 1] = j - 1
    for i in range(1, M):
        D[i, 0] = D[i - 1, 0] + cost[i, 0]
        bp[i, 0, 0] = i - 1
        bp[i, 0, 1] = 0

    for i in range(1, M):
        for j in range(1, N):
            candidates = np.array([
                D[i - 1, j],
                D[i, j - 1],
                D[i - 1, j - 1],
            ])
            idx = np.argmin(candidates)
            D[i, j] = candidates[idx] + cost[i, j]
            if idx == 0:
                bp[i, j, 0] = i - 1
                bp[i, j, 1] = j
            elif idx == 1:
                bp[i, j, 0] = i
                bp[i, j, 1] = j - 1
            else:
                bp[i, j, 0] = i - 1
                bp[i, j, 1] = j - 1

    return D, bp


def dtw_alignment(s1: np.ndarray, s2: np.ndarray,
                  cost_fn=None) -> np.ndarray:
    """
    Find the DTW alignment mapping every position i in s1 to a position j in s2.

    "We take the first pair from any such sequence to form a mapping
     a_{SV}[i] → j from every position i in s1 to a position j in s2."

    Args:
        s1: (M, D) — silent EMG features E'_S.
        s2: (N, D) — vocalized EMG features E'_V.
        cost_fn: Optional callable(s1, s2) → (M, N) cost matrix.
                 Defaults to Euclidean distance.

    Returns:
        alignment: (M,) integer array where alignment[i] = j.
    """
    M, N = len(s1), len(s2)

    if cost_fn is not None:
        cost = cost_fn(s1, s2)
    else:
        # Euclidean distance: δ_EMG[i,j] = ||E'_S[i] - E'_V[j]||
        cost = _euclidean_cost(s1, s2)

    D, bp = _dtw_cost_matrix(cost)

    # Backtrace to get the full path
    path = []
    i, j = M - 1, N - 1
    path.append((i, j))
    while i > 0 or j > 0:
        pi, pj = int(bp[i, j, 0]), int(bp[i, j, 1])
        path.append((pi, pj))
        i, j = pi, pj
    path.reverse()

    # Build the mapping: for each i, take the FIRST j
    alignment = np.zeros(M, dtype=np.int64)
    seen = set()
    for (pi, pj) in path:
        if pi not in seen:
            alignment[pi] = pj
            seen.add(pi)

    return alignment


def _euclidean_cost(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance cost matrix."""
    # Efficient: ||a-b||² = ||a||² + ||b||² - 2a·b
    s1_sq = np.sum(s1 ** 2, axis=1, keepdims=True)  # (M, 1)
    s2_sq = np.sum(s2 ** 2, axis=1, keepdims=True)  # (N, 1)
    cross = s1 @ s2.T                                # (M, N)
    dist_sq = s1_sq + s2_sq.T - 2 * cross
    dist_sq = np.maximum(dist_sq, 0.0)               # numerical safety
    return np.sqrt(dist_sq)


def combined_cost(emg_cost: np.ndarray, audio_cost: np.ndarray,
                  lam: float = 10.0) -> np.ndarray:
    """
    Section 3.2.2: δ_full[i,j] = δ_CCA[i,j] + λ · ||Â'_S[i] - A'_V[j]||

    Args:
        emg_cost: (M, N) CCA-space EMG distance.
        audio_cost: (M, N) predicted-audio distance.
        lam: Weight λ (default 10).

    Returns:
        (M, N) combined cost matrix.
    """
    return emg_cost + lam * audio_cost


def warp_features(features: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    """
    Create warped feature sequence:  Ã'_S[i] = A'_V[alignment[i]]

    Args:
        features: (N, D) vocalized audio features A'_V.
        alignment: (M,) mapping from silent to vocalized indices.

    Returns:
        warped: (M, D) time-aligned audio targets for silent EMG.
    """
    return features[alignment]
