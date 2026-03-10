"""
Distance between two functional annotations.

Input: two functional annotation objects (role, role_change, impact, grounded).
Output: scalar distance per field + aggregate.
"""

from typing import Any

from ..utils import cosine_distance


def _embedding(ann: dict[str, Any], field: str) -> list[float]:
    """Get embedding for role or impact. embedding is for role by default."""
    if field == "role":
        emb = ann.get("embedding")
        if emb is not None:
            return emb if isinstance(emb, (list, tuple)) else list(emb)
    if field == "impact":
        emb = ann.get("impact_embedding") or ann.get("system_impact_embedding")
        if emb is not None:
            return emb if isinstance(emb, (list, tuple)) else list(emb)
    return []


def _grounded_edges(ann: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract cited module graph edges from grounded field."""
    g = ann.get("grounded_in") or ann.get("grounded")
    if g is None:
        return set()
    if isinstance(g, (list, tuple)):
        edges = set()
        for item in g:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                edges.add((str(item[0]), str(item[1])))
            elif isinstance(item, dict) and "from" in item and "to" in item:
                edges.add((str(item["from"]), str(item["to"])))
        return edges
    if isinstance(g, set):
        return g if all(isinstance(e, (list, tuple)) for e in g) else set()
    return set()


def role_distance(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> float:
    """Cosine distance between role embeddings. Role is the primary field for functional."""
    emb_a = _embedding(ann_a, "role")
    emb_b = _embedding(ann_b, "role")
    if not emb_a or not emb_b:
        return 1.0
    return cosine_distance(emb_a, emb_b)


def impact_distance(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> float:
    """Cosine distance between impact embeddings. Secondary — which downstream components are affected."""
    emb_a = _embedding(ann_a, "impact")
    emb_b = _embedding(ann_b, "impact")
    if not emb_a or not emb_b:
        return 1.0
    return cosine_distance(emb_a, emb_b)


def grounding_overlap(ann_a: dict[str, Any], ann_b: dict[str, Any]) -> float:
    """
    Jaccard overlap between cited module graph edges in grounded field.
    Are the two annotations drawing on the same part of the module graph?
    high impact_distance + high grounding_overlap = same graph, different interpretation
    high impact_distance + low grounding_overlap = different graph neighborhoods
    """
    e_a = _grounded_edges(ann_a)
    e_b = _grounded_edges(ann_b)
    if not e_a and not e_b:
        return 1.0
    if not e_a or not e_b:
        return 0.0
    return len(e_a & e_b) / len(e_a | e_b)


def aggregate_distance(
    ann_a: dict[str, Any],
    ann_b: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted average of role_distance and impact_distance."""
    role_d = role_distance(ann_a, ann_b)
    impact_d = impact_distance(ann_a, ann_b)
    w = weights or {"role": 0.5, "impact": 0.5}
    total = w.get("role", 0) + w.get("impact", 0)
    if total == 0:
        return 0.0
    return (role_d * w.get("role", 0) + impact_d * w.get("impact", 0)) / total
