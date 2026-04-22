"""
Automated cluster validation battery for procedural-diff signatures.

Replaces human calibration with statistical cluster-validity metrics:
  - V-measure, ARI, NMI against reference partitions (external validity)
  - Bootstrap ARI (stability under resampling)
  - Silhouette, Davies-Bouldin (internal validity, given a distance matrix)

The point of V-measure against multiple reference partitions is interpretive:
if procedural clusters align with "which model solved it" but not with "which
repo the instance is from", procedures carry model-behavior information
independent of domain structure. That's the story — not any single score.

Metric definitions used:
  V-measure = 2 * (homogeneity * completeness) / (homogeneity + completeness)
  ARI = adjusted Rand index (chance-corrected, bounded roughly in [-0.5, 1])
  NMI = normalized mutual information (arithmetic-mean normalization)

Usage:
    from analysis.pdiff.vmeasure import run_vmeasure_battery
    table = run_vmeasure_battery(cluster_labels, references)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


@dataclass(frozen=True)
class ClusterMetrics:
    """External-validity metrics for a predicted partition vs. a reference."""

    reference_name: str
    n: int
    n_reference_classes: int
    n_predicted_clusters: int
    v_measure: float
    homogeneity: float
    completeness: float
    ari: float
    nmi: float
    ami: float


def _align_inputs(
    predicted: Sequence, reference: Sequence
) -> tuple[np.ndarray, np.ndarray]:
    pa = np.asarray(list(predicted))
    ra = np.asarray(list(reference))
    if len(pa) != len(ra):
        raise ValueError(
            f"predicted and reference lengths differ: {len(pa)} vs {len(ra)}"
        )
    mask = np.array([p is not None and r is not None for p, r in zip(pa, ra, strict=True)])
    return pa[mask], ra[mask]


def compute_metrics(
    predicted: Sequence,
    reference: Sequence,
    *,
    reference_name: str = "reference",
) -> ClusterMetrics:
    """Compute V-measure / ARI / NMI / AMI for predicted vs. reference partition.

    Pairs with `None` on either side are dropped. Both arrays may be string or
    int labels; sklearn handles encoding internally.
    """
    p, r = _align_inputs(predicted, reference)
    if len(p) == 0:
        return ClusterMetrics(
            reference_name=reference_name,
            n=0, n_reference_classes=0, n_predicted_clusters=0,
            v_measure=float("nan"), homogeneity=float("nan"),
            completeness=float("nan"), ari=float("nan"),
            nmi=float("nan"), ami=float("nan"),
        )
    return ClusterMetrics(
        reference_name=reference_name,
        n=len(p),
        n_reference_classes=len(set(r.tolist())),
        n_predicted_clusters=len(set(p.tolist())),
        v_measure=v_measure_score(r, p),
        homogeneity=homogeneity_score(r, p),
        completeness=completeness_score(r, p),
        ari=adjusted_rand_score(r, p),
        nmi=normalized_mutual_info_score(r, p),
        ami=adjusted_mutual_info_score(r, p),
    )


def run_vmeasure_battery(
    predicted: Sequence,
    references: dict[str, Sequence],
) -> pd.DataFrame:
    """Compute external-validity metrics against multiple reference partitions.

    Returns a DataFrame with one row per reference, sorted by V-measure desc.
    """
    rows = []
    for name, ref in references.items():
        m = compute_metrics(predicted, ref, reference_name=name)
        rows.append({
            "reference": m.reference_name,
            "n": m.n,
            "n_ref_classes": m.n_reference_classes,
            "n_clusters": m.n_predicted_clusters,
            "v_measure": m.v_measure,
            "homogeneity": m.homogeneity,
            "completeness": m.completeness,
            "ari": m.ari,
            "nmi": m.nmi,
            "ami": m.ami,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("v_measure", ascending=False).reset_index(drop=True)


def bootstrap_stability(
    predicted: Sequence,
    reference: Sequence,
    *,
    n_bootstrap: int = 50,
    sample_frac: float = 0.8,
    random_state: int = 0,
) -> dict[str, float]:
    """Bootstrap ARI — resample the joint (predicted, reference) and recompute.

    Measures how stable the external-validity score is to data perturbation.
    Low variance across bootstrap samples = stable partition alignment.
    """
    p, r = _align_inputs(predicted, reference)
    if len(p) < 2:
        return {"n_bootstrap": 0, "mean_ari": float("nan"), "std_ari": float("nan")}

    rng = np.random.default_rng(random_state)
    scores = []
    size = max(2, int(len(p) * sample_frac))
    for _ in range(n_bootstrap):
        idx = rng.choice(len(p), size=size, replace=True)
        if len(set(r[idx].tolist())) < 2 or len(set(p[idx].tolist())) < 2:
            continue
        scores.append(adjusted_rand_score(r[idx], p[idx]))
    if not scores:
        return {"n_bootstrap": 0, "mean_ari": float("nan"), "std_ari": float("nan")}
    return {
        "n_bootstrap": len(scores),
        "mean_ari": float(np.mean(scores)),
        "std_ari": float(np.std(scores)),
        "min_ari": float(np.min(scores)),
        "max_ari": float(np.max(scores)),
    }


def internal_validity(
    distance_matrix: np.ndarray,
    labels: Sequence,
) -> dict[str, float]:
    """Silhouette and Davies-Bouldin on a precomputed distance matrix.

    Note: both metrics require at least 2 distinct clusters and more points
    than clusters. Returns NaN when conditions aren't met.
    """
    labs = np.asarray(list(labels))
    n_clusters = len(set(labs.tolist()))
    if n_clusters < 2 or len(labs) <= n_clusters:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan")}

    try:
        sil = float(silhouette_score(distance_matrix, labs, metric="precomputed"))
    except ValueError:
        sil = float("nan")

    try:
        # Davies-Bouldin needs a feature matrix, not distances; skip if only distances given.
        db = float("nan")
        if distance_matrix.ndim == 2 and distance_matrix.shape[0] == distance_matrix.shape[1]:
            # Use distances-as-features as a crude fallback (documented limitation).
            db = float(davies_bouldin_score(distance_matrix, labs))
    except ValueError:
        db = float("nan")

    return {"silhouette": sil, "davies_bouldin": db}


def cluster_edits_by_vocab(
    trajectories: Iterable,
    *,
    k: int = 10,
    random_state: int = 0,
) -> list[int]:
    """Cheap k-means over edit-op indicator vectors — for experiments that
    need a baseline procedural clustering without building distance matrices.

    Returns a list of cluster labels aligned with input order.
    """
    from sklearn.cluster import KMeans

    trajs = list(trajectories)
    if not trajs:
        return []

    vocab: list[str] = sorted({op for t in trajs for op in getattr(t, "edits", set())})
    if not vocab:
        return [0] * len(trajs)

    idx = {op: i for i, op in enumerate(vocab)}
    X = np.zeros((len(trajs), len(vocab)), dtype=np.float32)
    for i, t in enumerate(trajs):
        for op in getattr(t, "edits", set()):
            j = idx.get(op)
            if j is not None:
                X[i, j] = 1.0

    actual_k = min(k, max(2, len(trajs) // 2 or 2))
    km = KMeans(n_clusters=actual_k, random_state=random_state, n_init=10)
    return km.fit_predict(X).tolist()
