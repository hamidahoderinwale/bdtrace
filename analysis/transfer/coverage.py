"""
Cross-dataset coverage and saturation analysis.

A. Coverage: for each instance in B, find nearest neighbor in A.
   Coverage(τ) = fraction of B within τ of some A instance.

B. Saturation curve: greedy farthest-first ordering of B relative to A.
   Selects B instances in order of maximum marginal distance from the current
   covered set (A ∪ previously selected B), then plots fraction of B covered
   vs instances added.

Representation choice for cross-dataset comparison:
  motifs_sequence  Jaccard on set of unique token types in the motif sequence.
                   Best for problem-space comparison: repo-agnostic, 184-dim vocabulary,
                   captures what kinds of operations appear. (Recommended.)
  edits_set_diff   Symmetric set-diff on edit operation types.
  edits            Jaccard on edit operation types.
  modules          Jaccard on module token sets.
  motifs           Cosine on soft_membership vectors (degenerate if vectors are all-zero).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


# ─── Representation extraction ────────────────────────────────────────────────


def _parse_motif_vec(raw: Any) -> dict[str, float]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        sm = raw.get("soft_membership", {})
        return sm if isinstance(sm, dict) else {}
    return {}


def _parse_motif_sequence_set(raw: Any) -> set[str]:
    """Set of unique token types from the motif sequence field."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return set()
    if isinstance(raw, dict):
        seq = raw.get("sequence", [])
        return set(seq) if isinstance(seq, list) else set()
    return set()


def _parse_optype_set(raw: Any) -> set[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return set()
    if not isinstance(raw, list):
        return set()
    types: set[str] = set()
    for cert in raw:
        if not isinstance(cert, dict):
            continue
        for op in cert.get("operations", []):
            if isinstance(op, dict) and op.get("type"):
                types.add(str(op["type"]))
    return types


def _parse_module_set(raw: Any) -> set[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return set()
    if isinstance(raw, dict):
        tokens = raw.get("tokens", [])
        return set(tokens) if isinstance(tokens, list) else set()
    return set()


def repr_sparsity(records: list[dict], repr_key: str) -> float:
    """Fraction of records with non-empty representation. Diagnostic."""
    if not records:
        return 0.0
    if repr_key == "motifs_sequence":
        non_empty = sum(1 for r in records if _parse_motif_sequence_set(r.get("motifs")))
    elif repr_key == "motifs":
        non_empty = sum(
            1 for r in records
            if any(v != 0.0 for v in _parse_motif_vec(r.get("motifs")).values())
        )
    elif repr_key in ("edits_set_diff", "edits"):
        non_empty = sum(1 for r in records if _parse_optype_set(r.get("edits")))
    elif repr_key == "modules":
        non_empty = sum(1 for r in records if _parse_module_set(r.get("modules")))
    else:
        return 0.0
    return non_empty / len(records)


# ─── Distance functions ────────────────────────────────────────────────────────


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return 1.0 - len(a & b) / union if union > 0 else 0.0


def _set_diff(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    total = len(a) + len(b)
    return len(a ^ b) / total if total > 0 else 0.0


# ─── Cross-dataset distance matrix ────────────────────────────────────────────


def cross_dataset_distances(
    records_a: list[dict],
    records_b: list[dict],
    repr_key: str = "motifs",
) -> np.ndarray:
    """
    Compute n_a × n_b distance matrix between two record sets.

    Args:
        records_a: base dataset records (each a dict with edits/modules/motifs)
        records_b: comparison dataset records
        repr_key: "motifs" | "edits_set_diff" | "edits" | "modules"

    Returns:
        D: (n_a, n_b) where D[i, j] = distance(records_a[i], records_b[j])
    """
    n_a, n_b = len(records_a), len(records_b)

    if repr_key == "motifs_sequence":
        sets_a = [_parse_motif_sequence_set(r.get("motifs")) for r in records_a]
        sets_b = [_parse_motif_sequence_set(r.get("motifs")) for r in records_b]
        D = np.empty((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                D[i, j] = _jaccard(sets_a[i], sets_b[j])
        return D

    if repr_key == "motifs":
        vecs_a = [_parse_motif_vec(r.get("motifs")) for r in records_a]
        vecs_b = [_parse_motif_vec(r.get("motifs")) for r in records_b]
        vocab = sorted(set().union(*[set(v) for v in vecs_a + vecs_b]))
        if not vocab:
            return np.zeros((n_a, n_b))
        arr_a = np.array([[v.get(k, 0.0) for k in vocab] for v in vecs_a], dtype=np.float64)
        arr_b = np.array([[v.get(k, 0.0) for k in vocab] for v in vecs_b], dtype=np.float64)
        D = np.empty((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                D[i, j] = _cosine_dist(arr_a[i], arr_b[j])
        return D

    if repr_key in ("edits_set_diff", "edits"):
        sets_a = [_parse_optype_set(r.get("edits")) for r in records_a]
        sets_b = [_parse_optype_set(r.get("edits")) for r in records_b]
        dist_fn = _set_diff if repr_key == "edits_set_diff" else _jaccard
        D = np.empty((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                D[i, j] = dist_fn(sets_a[i], sets_b[j])
        return D

    if repr_key == "modules":
        sets_a = [_parse_module_set(r.get("modules")) for r in records_a]
        sets_b = [_parse_module_set(r.get("modules")) for r in records_b]
        D = np.empty((n_a, n_b))
        for i in range(n_a):
            for j in range(n_b):
                D[i, j] = _jaccard(sets_a[i], sets_b[j])
        return D

    raise ValueError(
        f"Unknown repr_key: {repr_key!r}. Use: motifs, edits_set_diff, edits, modules"
    )


# ─── Coverage ─────────────────────────────────────────────────────────────────


def nn_coverage(D_ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each B instance (column), find nearest neighbor in A (rows).

    Returns:
        nn_dists: (n_b,) distance to nearest A neighbor
        nn_indices: (n_b,) index of nearest A neighbor
    """
    nn_indices = np.argmin(D_ab, axis=0)
    nn_dists = D_ab[nn_indices, np.arange(D_ab.shape[1])]
    return nn_dists, nn_indices


def coverage_curve(
    nn_dists: np.ndarray,
    thresholds: np.ndarray | None = None,
    n_thresholds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coverage fraction as a function of τ: fraction of B within τ of any A instance.

    Returns:
        thresholds: (n_thresholds,)
        fractions: (n_thresholds,) coverage at each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, float(np.percentile(nn_dists, 99)), n_thresholds)
    fractions = np.array([float(np.mean(nn_dists <= t)) for t in thresholds])
    return thresholds, fractions


# ─── Saturation curve ─────────────────────────────────────────────────────────


def _steps_to_threshold(coverage: list[float], target: float) -> int | None:
    for k, c in enumerate(coverage):
        if c >= target:
            return k
    return None


def saturation_curve(
    D_ab: np.ndarray,
    D_bb: np.ndarray,
    tau: float | None = None,
) -> dict[str, Any]:
    """
    Greedy farthest-first ordering of B instances relative to A.

    Starts with A as the covered set. At each step selects the B instance with
    the largest minimum distance to the current covered set (A ∪ selected B),
    then records what fraction of all B instances are within τ.

    Args:
        D_ab: (n_a, n_b) — column j gives distances from all A instances to B[j]
        D_bb: (n_b, n_b) — pairwise distances within B
        tau:  coverage threshold. If None, uses median of min(D_ab, axis=0),
              i.e. the median nearest-A-neighbor distance across all B.

    Returns dict with:
        order:            list of B indices in greedy selection order
        coverage:         list of length n_b+1; coverage[k] = fraction of B
                          covered after k additions (coverage[0] = A-only baseline)
        tau:              threshold used
        initial_coverage: coverage before any B instances added
        final_coverage:   coverage after all B instances added
        n_steps_to_50pct: steps needed to reach 50% coverage of B (None if never)
        n_steps_to_90pct: steps needed to reach 90% coverage of B (None if never)
    """
    n_b = D_ab.shape[1]

    # Distance from each B instance to its nearest A instance
    dist_to_a = np.min(D_ab, axis=0).copy()  # (n_b,)

    if tau is None:
        tau = float(np.median(dist_to_a))

    # min_dist_to_selected[c]: distance from c to the nearest *selected* B instance so far
    min_dist_to_selected = np.full(n_b, np.inf)

    order: list[int] = []
    remaining: set[int] = set(range(n_b))

    def _eff_dist(idx: int) -> float:
        """Effective distance from B[idx] to the current covered set."""
        return float(min(dist_to_a[idx], min_dist_to_selected[idx]))

    def _covered_fraction() -> float:
        eff = np.minimum(dist_to_a, min_dist_to_selected)
        return float(np.mean(eff <= tau))

    coverage = [_covered_fraction()]

    while remaining:
        # Select B instance farthest from covered set (most novel)
        best = max(remaining, key=_eff_dist)
        order.append(best)
        remaining.discard(best)

        # Update min distances for remaining instances
        for c in remaining:
            d = float(D_bb[best, c])
            if d < min_dist_to_selected[c]:
                min_dist_to_selected[c] = d

        coverage.append(_covered_fraction())

    return {
        "order": order,
        "coverage": coverage,
        "tau": tau,
        "initial_coverage": coverage[0],
        "final_coverage": coverage[-1],
        "n_steps_to_50pct": _steps_to_threshold(coverage, 0.5),
        "n_steps_to_90pct": _steps_to_threshold(coverage, 0.9),
    }
