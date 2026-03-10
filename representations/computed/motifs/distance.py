"""
Distance between motif soft membership vectors.

Input: two soft membership vectors over motif vocabulary.
Output: scalar distance.
DTW is for sequence-to-motif; cosine is for vector-to-vector.
"""

from collections.abc import Sequence

import numpy as np


def _dtw_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> float:
    """DTW distance with unit cost for mismatch."""
    if not seq_a or not seq_b:
        return max(len(seq_a), len(seq_b))
    n, m = len(seq_a), len(seq_b)
    d = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            d[i][j] = cost + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[n][m]


def dtw_similarity(sequence: Sequence[str], motif: Sequence[str]) -> float:
    """Similarity in [0, 1]: 1 = perfect match. For sequence-to-motif soft membership."""
    if not motif:
        return 1.0
    dist = _dtw_distance(sequence, motif)
    max_dist = max(len(sequence), len(motif))
    if max_dist == 0:
        return 1.0
    return max(0.0, 1.0 - dist / max_dist)


def _cosine_similarity(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def motif_distance(
    vec_a: dict[str, float] | np.ndarray | list[float],
    vec_b: dict[str, float] | np.ndarray | list[float],
    vocabulary: list[str] | None = None,
) -> float:
    """
    Cosine distance between soft membership vectors.
    Both vectors are over the same motif vocabulary.
    Returns 1 - cosine_similarity.
    """
    if isinstance(vec_a, dict) or isinstance(vec_b, dict):
        keys_a = set(vec_a.keys()) if isinstance(vec_a, dict) else set()
        keys_b = set(vec_b.keys()) if isinstance(vec_b, dict) else set()
        vocab = vocabulary or list(keys_a | keys_b)
        if not vocab:
            return 0.0
        a = np.array([(vec_a.get(k, 0.0) if isinstance(vec_a, dict) else 0.0) for k in vocab], dtype=np.float64)
        b = np.array([(vec_b.get(k, 0.0) if isinstance(vec_b, dict) else 0.0) for k in vocab], dtype=np.float64)
    else:
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)
    return 1.0 - _cosine_similarity(a, b)


def vocabulary_coverage(vec: dict[str, float] | np.ndarray | list[float]) -> float:
    """
    What fraction of the motif vocabulary has nonzero membership.
    Diagnostic: low coverage means the sequence is unusual or the vocabulary is too narrow.
    """
    if isinstance(vec, dict):
        total = len(vec)
        nonzero = sum(1 for v in vec.values() if v and abs(float(v)) > 1e-9)
    else:
        arr = np.asarray(vec, dtype=np.float64)
        total = arr.size
        nonzero = np.count_nonzero(np.abs(arr) > 1e-9)
    if total == 0:
        return 0.0
    return nonzero / total
