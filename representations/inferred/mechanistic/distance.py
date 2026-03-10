"""
Distance between two mechanistic annotations.

Input: two mechanistic annotation objects (mechanism, pattern, locations, grounded).
Output: scalar distance per field + aggregate.
"""

from typing import Any

from ..utils import cosine_distance


def _embedding(ann: dict[str, Any], field: str) -> list[float]:
    """Get embedding for mechanism or pattern. embedding is for mechanism by default."""
    if field == "mechanism":
        emb = ann.get("embedding")
        if emb is not None:
            return emb if isinstance(emb, (list, tuple)) else list(emb)
    return ann.get(f"{field}_embedding") or ann.get("embedding") or []


def _location_set(ann: dict[str, Any]) -> set[str]:
    """Extract AST node type set from locations field."""
    loc = ann.get("locations", "")
    if isinstance(loc, (list, tuple, set)):
        return {str(x) for x in loc}
    if isinstance(loc, str):
        return {x.strip() for x in loc.replace(",", " ").split() if x.strip()}
    return set()


def pattern_distance(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> float:
    """Cosine distance between pattern embeddings. Pattern is the generalizable unit."""
    emb_a = _embedding(ann_a, "pattern")
    emb_b = _embedding(ann_b, "pattern")
    if not emb_a or not emb_b:
        return 1.0
    return cosine_distance(emb_a, emb_b)


def mechanism_distance(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> float:
    """Cosine distance between mechanism embeddings. More specific than pattern."""
    emb_a = _embedding(ann_a, "mechanism")
    emb_b = _embedding(ann_b, "mechanism")
    if not emb_a or not emb_b:
        return 1.0
    return cosine_distance(emb_a, emb_b)


def location_overlap(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> float:
    """
    Jaccard overlap between AST node type sets in locations field.
    Not embedding-based — locations are structured, overlap is the right metric.
    """
    s_a = _location_set(ann_a)
    s_b = _location_set(ann_b)
    if not s_a and not s_b:
        return 1.0
    if not s_a or not s_b:
        return 0.0
    return len(s_a & s_b) / len(s_a | s_b)


def aggregate_distance(
    ann_a: dict[str, Any],
    ann_b: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Weighted average of pattern_distance and mechanism_distance.
    location_overlap as diagnostic, not primary signal.
    """
    pattern_d = pattern_distance(ann_a, ann_b)
    mechanism_d = mechanism_distance(ann_a, ann_b)
    w = weights or {"pattern": 0.5, "mechanism": 0.5}
    total = w.get("pattern", 0) + w.get("mechanism", 0)
    if total == 0:
        return 0.0
    return (pattern_d * w.get("pattern", 0) + mechanism_d * w.get("mechanism", 0)) / total
