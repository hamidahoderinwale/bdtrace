"""Distance metrics for inferred representations (cosine, L2, etc.)."""

import numpy as np


def cosine_similarity(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Cosine similarity in [-1, 1]; 1 = identical direction."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Cosine distance = 1 - cosine_similarity. Range [0, 2]; 0 = identical."""
    return 1.0 - cosine_similarity(a, b)
