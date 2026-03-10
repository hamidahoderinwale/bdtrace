"""
Distance between two behavioral annotations.

Input: two behavioral annotation objects (claim, before_behavior, after_behavior, testable, grounded).
Output: scalar distance per field + aggregate.
"""

from typing import Any

from ..utils import cosine_distance


def _embedding(ann: dict[str, Any], field: str) -> list[float]:
    """Get embedding for a field. Annotation may have 'embedding' for claim or per-field."""
    if field == "claim":
        emb = ann.get("embedding")
        if emb is not None:
            return emb if isinstance(emb, (list, tuple)) else list(emb)
    return ann.get(f"{field}_embedding") or []


def claim_distance(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> float:
    """Cosine distance between claim embeddings."""
    emb_a = _embedding(ann_a, "claim")
    emb_b = _embedding(ann_b, "claim")
    if not emb_a or not emb_b:
        return 1.0
    return cosine_distance(emb_a, emb_b)


def field_distances(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> dict[str, float]:
    """
    Cosine distance for each field independently.
    claim, before_behavior, after_behavior.
    Returns dict — useful for identifying which aspect of behavior diverged.
    """
    result = {}
    for key in ["claim", "before_behavior", "after_behavior"]:
        emb_a = _embedding(ann_a, key) or ann_a.get("embedding", [])
        emb_b = _embedding(ann_b, key) or ann_b.get("embedding", [])
        if emb_a and emb_b:
            result[key] = cosine_distance(emb_a, emb_b)
        else:
            result[key] = 1.0
    return result


def aggregate_distance(
    ann_a: dict[str, Any],
    ann_b: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Weighted average of field distances.
    Default equal weights. Weights tunable — claim probably most important for divergence matrix.
    """
    dists = field_distances(ann_a, ann_b)
    if not dists:
        return 0.0
    w = weights or {k: 1.0 / len(dists) for k in dists}
    total = sum(w.get(k, 0) for k in dists)
    if total == 0:
        return 0.0
    return sum(dists[k] * w.get(k, 0) for k in dists) / total
