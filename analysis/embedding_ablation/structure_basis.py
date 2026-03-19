"""
Embedding ablation: structure as basis of representation.

Runs behavioral with (a) edits only, (b) code only, (c) both.
Compares claim embeddings: sim(emb_a, emb_c), sim(emb_b, emb_c), sim(emb_a, emb_b).

If sim(emb_a, emb_c) high -> structure is the basis.
If sim(emb_b, emb_c) high -> code semantics dominate.
If sim(emb_a, emb_b) low -> edits and code produce different bases.
"""

from typing import Any

import numpy as np


def _cosine_sim(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity. Returns 0 if either empty."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or len(a) != len(b):
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_embedding_ablation(
    records: list[dict[str, Any]],
    behavioral_fn: Any,
    embed_model: str | None = None,
) -> dict[str, Any]:
    """
    Run ablation: (a) edits only, (b) code only, (c) both. Compare embeddings.

    Each record must have:
      - before_fn, after_fn: str (from first code_change)
      - structural_certificate: dict or list (edits cert)

    behavioral_fn(before_fn, after_fn, structural_certificate, embed_model=...) -> dict
    with "claim" and "embedding" keys.

    Returns:
      - per_instance: list of {instance_id, sim_ac, sim_bc, sim_ab, claim_a, claim_b, claim_c}
      - aggregate: mean sim_ac, mean sim_bc, mean sim_ab, n_valid
    """
    per_instance = []
    sims_ac, sims_bc, sims_ab = [], [], []

    for rec in records:
        before = rec.get("before_fn") or ""
        after = rec.get("after_fn") or ""
        cert = rec.get("structural_certificate") or {}
        instance_id = rec.get("instance_id") or rec.get("repo") or "unknown"

        # (a) Edits only: empty code
        out_a = behavioral_fn(
            before_fn="(empty)",
            after_fn="(empty)",
            structural_certificate=cert,
            embed_model=embed_model,
        )
        # (b) Code only: empty cert
        out_b = behavioral_fn(
            before_fn=before or "(empty)",
            after_fn=after or "(empty)",
            structural_certificate={},
            embed_model=embed_model,
        )
        # (c) Both
        out_c = behavioral_fn(
            before_fn=before or "(empty)",
            after_fn=after or "(empty)",
            structural_certificate=cert,
            embed_model=embed_model,
        )

        emb_a = out_a.get("embedding") or []
        emb_b = out_b.get("embedding") or []
        emb_c = out_c.get("embedding") or []

        if not emb_c:
            continue

        sim_ac = _cosine_sim(emb_a, emb_c) if emb_a else 0.0
        sim_bc = _cosine_sim(emb_b, emb_c) if emb_b else 0.0
        sim_ab = _cosine_sim(emb_a, emb_b) if emb_a and emb_b else 0.0

        sims_ac.append(sim_ac)
        sims_bc.append(sim_bc)
        sims_ab.append(sim_ab)

        per_instance.append({
            "instance_id": instance_id,
            "sim_ac": sim_ac,
            "sim_bc": sim_bc,
            "sim_ab": sim_ab,
            "claim_a": out_a.get("claim") or "",
            "claim_b": out_b.get("claim") or "",
            "claim_c": out_c.get("claim") or "",
        })

    n = len(sims_ac)
    return {
        "per_instance": per_instance,
        "aggregate": {
            "n_valid": n,
            "mean_sim_ac": float(np.mean(sims_ac)) if n else 0.0,
            "mean_sim_bc": float(np.mean(sims_bc)) if n else 0.0,
            "mean_sim_ab": float(np.mean(sims_ab)) if n else 0.0,
            "structure_dominates": float(np.mean(sims_ac)) > float(np.mean(sims_bc)) if n else False,
        },
    }
