"""
Eval saturation and structural transfer.

If an agent passes structurally similar tasks, new tasks in the same structural
region are likely to pass. Implements distance-to-centroid, kNN transfer, and
region pass rate.
"""

from typing import Any

import numpy as np


def _auc_distance_vs_pass(dist_to_centroid: np.ndarray, passed_mask: np.ndarray) -> float:
    """AUC for distance-to-passed-centroid vs pass. Higher distance = fail; use -dist as score."""
    from sklearn.metrics import roc_auc_score

    valid = ~np.isnan(dist_to_centroid)
    if np.sum(valid) < 2 or len(np.unique(passed_mask[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(passed_mask[valid], -dist_to_centroid[valid]))


def _saturation_knee(
    dist_to_centroid: np.ndarray,
    passed_mask: np.ndarray,
    window_pct: float = 0.1,
    slope_threshold: float = 0.01,
    min_rank_pct: float = 0.05,
) -> int | None:
    """
    Rank (by distance, closest first) where cumulative pass rate flattens.
    Uses rolling mean of marginal gain; knee = first rank (after min_rank_pct) where
    slope < threshold over a window.
    """
    order = np.argsort(dist_to_centroid)
    n = len(order)
    if n < 2:
        return None
    cum = np.cumsum(passed_mask[order].astype(np.float64)) / np.arange(1, n + 1)
    marginal = np.diff(cum, prepend=0)
    window = max(1, int(n * window_pct))
    min_k = max(0, int(n * min_rank_pct))
    for k in range(min_k, n - window):
        if np.mean(np.abs(marginal[k : k + window])) < slope_threshold:
            return k
    return n - 1


def distance_to_passed_centroid(
    D: np.ndarray,
    passed_mask: np.ndarray,
    instance_idx: int | None = None,
) -> np.ndarray | float:
    """
    Distance from each instance (or one instance) to the centroid of passed instances.

    Centroid = mean of distance profiles of passed instances. For instance i,
    profile = D[i, :]. Centroid profile = mean over passed rows.

    Args:
        D: (n, n) symmetric distance matrix
        passed_mask: (n,) boolean, True = passed
        instance_idx: If set, return scalar for that instance only

    Returns:
        (n,) array of distances to centroid, or scalar if instance_idx set
    """
    n = D.shape[0]
    passed_indices = np.where(passed_mask)[0]
    if len(passed_indices) == 0:
        return np.full(n, np.nan) if instance_idx is None else np.nan

    centroid_profile = np.mean(D[passed_indices, :], axis=0)

    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.mean(np.abs(D[i, :] - centroid_profile))

    if instance_idx is not None:
        return float(distances[instance_idx])
    return distances


def knn_transfer_predict(
    D: np.ndarray,
    passed_mask: np.ndarray,
    k: int = 5,
    exclude_self: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict pass from k-nearest-neighbor vote among evaluated instances.

    For each instance i: kNN = k closest (by distance). Pass rate = fraction of
    kNN that passed. Used for transfer: if kNN mostly passed, instance is saturated.

    Args:
        D: (n, n) distance matrix
        passed_mask: (n,) boolean
        k: number of neighbors
        exclude_self: exclude instance itself from kNN

    Returns:
        (predicted_pass_rate, kNN_indices_per_instance)
        predicted_pass_rate: (n,) fraction of kNN that passed
    """
    n = D.shape[0]
    passed_float = passed_mask.astype(np.float64)

    pred = np.full(n, np.nan)
    for i in range(n):
        row = D[i, :].copy()
        if exclude_self:
            row[i] = np.inf
        nn = np.argpartition(row, min(k, n - 1))[:k]
        nn = nn[row[nn] < np.inf]
        if len(nn) == 0:
            continue
        pred[i] = np.mean(passed_float[nn])

    return pred, np.array([])


def region_pass_rate(
    D: np.ndarray,
    passed_mask: np.ndarray,
    region_labels: np.ndarray,
) -> dict[Any, float]:
    """
    Pass rate per structural region (e.g. stratum or cluster).

    Args:
        D: distance matrix (unused if regions pre-defined)
        passed_mask: (n,) boolean
        region_labels: (n,) region id per instance

    Returns:
        {region_id: pass_rate}
    """
    regions = np.unique(region_labels)
    result = {}
    for r in regions:
        mask = region_labels == r
        n_in = np.sum(mask)
        n_passed = np.sum(passed_mask & mask)
        result[r] = n_passed / n_in if n_in > 0 else 0.0
    return result


def run_transfer_analysis(
    D: np.ndarray,
    passed_mask: np.ndarray,
    instance_ids: list[str] | None = None,
    region_labels: np.ndarray | None = None,
    repr_name: str = "edits",
    k_values: tuple[int, ...] = (3, 5, 10),
) -> dict[str, Any]:
    """
    Run full transfer analysis: centroid distance, kNN accuracy, region pass rates.

    Args:
        D: (n, n) distance matrix
        passed_mask: (n,) boolean
        instance_ids: optional, for per-instance output
        region_labels: optional, for region pass rate (default: all one region)
        repr_name: representation name for output
        k_values: k values for kNN evaluation

    Returns:
        Dict with transfer_metrics, per_instance (optional), region_pass_rates
    """
    n = D.shape[0]
    n_passed = int(np.sum(passed_mask))
    n_failed = n - n_passed

    if n_passed == 0 or n_failed == 0:
        return {
            "repr": repr_name,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "error": "Need both passed and failed instances",
        }

    dist_to_centroid = distance_to_passed_centroid(D, passed_mask)

    auc = _auc_distance_vs_pass(dist_to_centroid, passed_mask)
    knee = _saturation_knee(dist_to_centroid, passed_mask)
    overall_rate = n_passed / n
    coverage_summary = (
        f"saturates at ~{knee} instances (rank by distance)"
        if knee is not None
        else "saturation knee not found"
    )

    results: dict[str, Any] = {
        "repr": repr_name,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "mean_distance_passed": float(np.nanmean(dist_to_centroid[passed_mask])),
        "mean_distance_failed": float(np.nanmean(dist_to_centroid[~passed_mask])),
        "auc_distance_vs_pass": auc,
        "saturation_knee_rank": knee,
        "coverage_summary": coverage_summary,
        "overall_pass_rate": float(overall_rate),
        "knn": {},
    }

    for k in k_values:
        pred, _ = knn_transfer_predict(D, passed_mask, k=k)
        valid = ~np.isnan(pred)
        if np.sum(valid) == 0:
            continue
        pred_binary = (pred >= 0.5).astype(np.int32)
        acc = np.mean(pred_binary[valid] == passed_mask[valid])
        results["knn"][f"k={k}"] = {"accuracy": float(acc)}

    if region_labels is not None:
        results["region_pass_rates"] = region_pass_rate(D, passed_mask, region_labels)

    if instance_ids is not None:
        results["per_instance"] = [
            {
                "instance_id": instance_ids[i],
                "passed": bool(passed_mask[i]),
                "distance_to_passed_centroid": float(dist_to_centroid[i]),
            }
            for i in range(n)
        ]

    return results
