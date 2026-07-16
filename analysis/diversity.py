"""
Diversity analysis: operations on distance matrices.

Input: five distance matrices (edits, modules, motifs, behavioral, mechanistic)
       and stratum labels.
Output: rank correlation matrix, within/across stratum ratios, silhouette scores,
        unique variance per representation, per-instance representation correlation.

No raw annotations, embeddings, or certificates needed — all upstream.
"""

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score


REPR_NAMES = ["edits", "modules", "motifs", "behavioral", "mechanistic"]


def per_instance_rep_correlation(matrices: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Per-instance Spearman correlation between distance profiles across representation pairs.

    For each instance i: vec_a = D_a[i, :] (distances from i to all others).
    Low correlation = instance expressed differently in repr A vs B.

    Returns:
        mean_rho: (n,) mean correlation across all repr pairs per instance
        min_rho: (n,) minimum correlation (most variable pair) per instance
        pair_rho: dict (repr_i, repr_j) -> (n,) array
    """
    names = list(matrices)
    n_repr = len(names)
    n_instances = matrices[names[0]].shape[0]

    pair_rho = {}
    for i, a in enumerate(names):
        for j in range(i + 1, n_repr):
            b = names[j]
            Da, Db = matrices[a], matrices[b]
            rhos = np.full(n_instances, np.nan)
            for k in range(n_instances):
                vec_a = np.delete(Da[k, :], k)
                vec_b = np.delete(Db[k, :], k)
                if np.var(vec_a) > 0 and np.var(vec_b) > 0:
                    r, _ = spearmanr(vec_a, vec_b)
                    rhos[k] = r if not np.isnan(r) else 0.0
            pair_rho[(a, b)] = rhos

    mean_rho = np.full(n_instances, np.nan)
    min_rho = np.full(n_instances, np.nan)
    for k in range(n_instances):
        vals = [pair_rho[p][k] for p in pair_rho if not np.isnan(pair_rho[p][k])]
        if vals:
            mean_rho[k] = float(np.mean(vals))
            min_rho[k] = float(np.min(vals))

    return {
        "mean_rho": mean_rho,
        "min_rho": min_rho,
        "pair_rho": pair_rho,
        "pair_names": list(pair_rho.keys()),
    }


def rank_correlation_matrix(matrices: dict[str, np.ndarray]) -> np.ndarray:
    """
    Spearman rank correlation between all pairs of flattened distance vectors.
    Uses upper triangle only to avoid double counting.
    Returns 5x5 symmetric matrix of rho values.
    """
    names = list(matrices)
    n = len(names)
    n_instances = matrices[names[0]].shape[0]
    idx = np.triu_indices(n_instances, k=1)

    flats = {k: matrices[k][idx] for k in names}

    rho_matrix = np.eye(n)
    for i, a in enumerate(names):
        for j in range(i + 1, n):
            rho, _ = spearmanr(flats[a], flats[names[j]])
            rho_matrix[i, j] = rho if not np.isnan(rho) else 0.0
            rho_matrix[j, i] = rho_matrix[i, j]
    return rho_matrix


def stratum_ratio(D: np.ndarray, labels: np.ndarray | list) -> float:
    """
    Mean pairwise distance within stratum / mean across strata.
    Ratio < 1 means representation clusters by stratum.
    """
    labels = np.asarray(labels)
    within, across = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(D[i, j])
            else:
                across.append(D[i, j])
    if not within or not across:
        return float("nan")
    return np.mean(within) / np.mean(across)


def stratum_overlap(D: np.ndarray, labels: np.ndarray | list) -> float:
    """
    P(within < across): fraction of (within, across) pairs where within-distance < across-distance.
    Uses raw distances; no transformation. >0.5 = strata separable.
    """
    labels = np.asarray(labels)
    within, across = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(D[i, j])
            else:
                across.append(D[i, j])
    if not within or not across:
        return float("nan")
    n_pairs = len(within) * len(across)
    less = sum(1 for w in within for a in across if w < a)
    eq = sum(1 for w in within for a in across if w == a)
    return (less + 0.5 * eq) / n_pairs


def stratum_ratios(matrices: dict[str, np.ndarray], labels: np.ndarray | list) -> dict[str, float]:
    """Stratum ratio for each representation."""
    return {k: stratum_ratio(D, labels) for k, D in matrices.items()}


def stratum_overlaps(matrices: dict[str, np.ndarray], labels: np.ndarray | list) -> dict[str, float]:
    """P(within < across) for each representation."""
    return {k: stratum_overlap(D, labels) for k, D in matrices.items()}


def silhouette_scores(matrices: dict[str, np.ndarray], labels: np.ndarray | list) -> dict[str, float]:
    """Silhouette score per representation (precomputed distance)."""
    labels = np.asarray(labels)
    n_labels = len(set(labels))
    if n_labels < 2:
        return {k: float("nan") for k in matrices}
    return {
        k: silhouette_score(D, labels, metric="precomputed")
        for k, D in matrices.items()
    }


def unique_variance(target_flat: np.ndarray, other_flats: list[np.ndarray]) -> float:
    """
    Residual variance after regressing target on all others.
    High = representation captures info others do not.
    """
    if not other_flats:
        return 1.0
    X = np.column_stack(other_flats)
    reg = LinearRegression().fit(X, target_flat)
    residual = target_flat - reg.predict(X)
    var_target = np.var(target_flat)
    if var_target == 0:
        return 0.0
    return float(np.var(residual) / var_target)


def unique_variances(matrices: dict[str, np.ndarray]) -> dict[str, float]:
    """Unique variance for each representation."""
    names = list(matrices)
    n_instances = matrices[names[0]].shape[0]
    idx = np.triu_indices(n_instances, k=1)
    flats = {k: matrices[k][idx] for k in names}

    result = {}
    for i, name in enumerate(names):
        others = [flats[n] for j, n in enumerate(names) if j != i]
        result[name] = unique_variance(flats[name], others)
    return result


def run_diversity_analysis(
    matrices: dict[str, np.ndarray],
    labels: np.ndarray | list,
) -> dict[str, Any]:
    """
    Run all diversity analyses.
    Returns dict with rank_correlation, stratum_ratios, silhouette_scores,
    unique_variances, per_instance_rep_correlation.
    """
    labels = np.asarray(labels)
    out = {
        "rank_correlation": rank_correlation_matrix(matrices),
        "rank_correlation_names": list(matrices),
        "stratum_ratios": stratum_ratios(matrices, labels),
        "stratum_overlaps": stratum_overlaps(matrices, labels),
        "silhouette_scores": silhouette_scores(matrices, labels),
        "unique_variances": unique_variances(matrices),
    }
    out["per_instance_rep_correlation"] = per_instance_rep_correlation(matrices)
    return out
