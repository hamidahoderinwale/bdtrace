"""
Analysis: grounding lift and divergence from baseline.

Uses outputs from existing scripts. No new structural components.
"""

from typing import Any

from representations.inferred.utils import cosine_distance, embed_text


def _embed_repr(repr_val: Any, embed_model: str | None = None) -> list[float]:
    """Embed a representation for comparison. Handles tokens list or dict with embedding."""
    if isinstance(repr_val, list) and repr_val and isinstance(repr_val[0], str):
        text = " ".join(str(x) for x in repr_val[:200])
        return embed_text(text, embed_model) or []
    if isinstance(repr_val, dict):
        emb = repr_val.get("embedding")
        if emb is not None:
            return emb if isinstance(emb, (list, tuple)) else list(emb)
        # Fallback: embed primary text field
        for key in ("claim", "role", "mechanism"):
            if key in repr_val and repr_val[key]:
                return embed_text(str(repr_val[key]), embed_model) or []
    return []


def divergence_from_baseline(
    records: list[dict[str, Any]],
    baseline_key: str = "tokens",
    structured_keys: list[str] | None = None,
    instance_type_key: str | None = None,
    embed_model: str | None = None,
) -> dict[str, Any]:
    """
    Where do structured procedures diverge most from level 1 baseline?

    For each instance, compute cosine distance between baseline embedding and
    each structured procedure embedding. High divergence = structured view
    differs from raw token view.

    Returns:
        - per_instance: list of {instance_id, baseline_key, divergences: {proc: dist}}
        - per_procedure: mean divergence per structured procedure
        - per_instance_type: if instance_type_key provided, mean divergence by type
    """
    structured_keys = structured_keys or ["behavioral", "mechanistic", "functional"]

    per_instance = []
    procedure_sums: dict[str, list[float]] = {k: [] for k in structured_keys}
    type_sums: dict[str, dict[str, list[float]]] = {}

    for rec in records:
        baseline_val = rec.get(baseline_key)
        baseline_emb = _embed_repr(baseline_val, embed_model)
        if not baseline_emb:
            continue

        instance_id = rec.get("instance_id") or rec.get("repo") or "unknown"
        instance_type = rec.get(instance_type_key, "unknown") if instance_type_key else "unknown"

        divergences = {}
        for proc in structured_keys:
            proc_val = rec.get(proc)
            if proc_val is None:
                continue
            proc_emb = _embed_repr(proc_val, embed_model)
            if not proc_emb:
                continue
            dist = cosine_distance(baseline_emb, proc_emb)
            divergences[proc] = dist
            procedure_sums[proc].append(dist)

            if instance_type_key:
                if instance_type not in type_sums:
                    type_sums[instance_type] = {k: [] for k in structured_keys}
                type_sums[instance_type][proc].append(dist)

        if divergences:
            per_instance.append(
                {
                    "instance_id": instance_id,
                    "instance_type": instance_type if instance_type_key else None,
                    "baseline_key": baseline_key,
                    "divergences": divergences,
                }
            )

    per_procedure = {proc: sum(vals) / len(vals) if vals else 0.0 for proc, vals in procedure_sums.items()}

    per_instance_type = {}
    if instance_type_key and type_sums:
        for itype, proc_vals in type_sums.items():
            per_instance_type[itype] = {proc: sum(v) / len(v) if v else 0.0 for proc, v in proc_vals.items()}

    return {
        "per_instance": per_instance,
        "per_procedure": per_procedure,
        "per_instance_type": per_instance_type,
    }
