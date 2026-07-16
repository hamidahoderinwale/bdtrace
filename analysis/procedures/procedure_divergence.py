"""
Procedural divergence: compare procedure outputs by stage.

Procedure set: P_behavioral, P_mechanistic, P_functional — all share [localization, edits];
only annotation varies. Structural agreement is high by construction; semantic agreement
(embedding distance between output fields) is where divergence occurs. The gap between
structural and semantic agreement is the key metric.
"""

from typing import Any

from representations.inferred.utils import cosine_distance


def _jaccard_distance(a: set, b: set) -> float:
    """Jaccard distance 1 - |intersection|/|union|. 0 = identical."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else 1.0 - inter / union


def certificate_divergence(
    cert_a: dict | list | None,
    cert_b: dict | list | None,
    threshold: float = 0.0,
) -> tuple[bool, float]:
    """
    Length-2 divergence: do two structural outputs (edits certificates) differ?

    Extracts op types from certificates; Jaccard distance > threshold means diverge.
    Returns (diverged, distance).
    """
    def op_types_from(cert: dict | list | None) -> set[str]:
        if cert is None:
            return set()
        if isinstance(cert, list):
            types = set()
            for c in cert:
                if isinstance(c, dict):
                    for op in c.get("operations", []):
                        if isinstance(op, dict) and op.get("type"):
                            types.add(str(op["type"]))
            return types
        if isinstance(cert, dict):
            types = set()
            for op in cert.get("operations", []):
                if isinstance(op, dict) and op.get("type"):
                    types.add(str(op["type"]))
            return types
        return set()

    types_a = op_types_from(cert_a)
    types_b = op_types_from(cert_b)
    d = _jaccard_distance(types_a, types_b)
    return (d > threshold, d)


def _primary_embedding(ann: dict, proc_name: str) -> list[float]:
    """Primary embedding for procedure: claim, pattern, or role."""
    emb = ann.get("embedding")
    if emb is not None:
        return emb if isinstance(emb, (list, tuple)) else list(emb)
    field = {"behavioral": "claim", "mechanistic": "pattern", "functional": "role"}.get(
        proc_name, "claim"
    )
    return ann.get(f"{field}_embedding") or []


def annotation_divergence(
    ann_a: dict | None,
    ann_b: dict | None,
    proc_a: str = "behavioral",
    proc_b: str = "behavioral",
    threshold: float = 0.3,
) -> tuple[bool, float]:
    """
    Length-3 divergence: do two annotation outputs differ?

    Uses embedding distance on primary field (claim, pattern, role) per procedure.
    Cross-type ok: behavioral claim vs mechanistic pattern compared via embeddings.
    Returns (diverged, distance). Diverged when d > threshold.
    """
    if ann_a is None or ann_b is None:
        return (True, 1.0)

    emb_a = _primary_embedding(ann_a, proc_a)
    emb_b = _primary_embedding(ann_b, proc_b)
    if not emb_a or not emb_b:
        return (True, 1.0)
    d = cosine_distance(emb_a, emb_b)
    return (d > threshold, d)


def reconstruction_divergence(
    rec_a: dict | None,
    rec_b: dict | None,
    threshold: float = 0.0,
) -> tuple[bool, float]:
    """
    Length-4 divergence: fidelity difference. Stub — not implemented.
    """
    return (False, 0.0)


def procedural_summary(
    record: dict,
    proc_a: str,
    proc_b: str,
    stage_outputs_a: dict[str, Any],
    stage_outputs_b: dict[str, Any],
    annotation_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Compute S(P_a, P_b, stage): where did divergence originate?

    stage_outputs_a/b: {"structural": cert, "annotation": ann} or {"structural": cert} for length 2.
    Returns dict with per-stage divergence flags and provenance (inherited vs introduced).
    """
    result = {
        "structural_diverged": False,
        "structural_distance": 0.0,
        "annotation_diverged": False,
        "annotation_distance": 0.0,
        "structural_introduced": False,
        "annotation_introduced": False,
    }

    cert_a = stage_outputs_a.get("structural")
    cert_b = stage_outputs_b.get("structural")
    div_struct, d_struct = certificate_divergence(cert_a, cert_b, threshold=0.0)
    result["structural_diverged"] = div_struct
    result["structural_distance"] = d_struct

    if div_struct:
        result["structural_introduced"] = True

    ann_a = stage_outputs_a.get("annotation")
    ann_b = stage_outputs_b.get("annotation")
    if ann_a is not None or ann_b is not None:
        div_ann, d_ann = annotation_divergence(
            ann_a, ann_b, proc_a=proc_a, proc_b=proc_b, threshold=annotation_threshold
        )
        result["annotation_diverged"] = div_ann
        result["annotation_distance"] = d_ann
        if div_ann and not div_struct:
            result["annotation_introduced"] = True
        elif div_ann and div_struct:
            result["annotation_introduced"] = False

    return result


def procedure_pair_divergence(
    record: dict,
    proc_a: str,
    proc_b: str,
    length: int = 3,
    annotation_threshold: float = 0.3,
    cert_a: dict | list | None = None,
    cert_b: dict | list | None = None,
) -> dict[str, Any]:
    """
    Compute divergence between two procedures on one instance.

    Length 3 (default): procedures are annotation types (behavioral, mechanistic, functional).
    Structural stage shared (edits). Annotation stage differs by procedure.

    Length 2: pass cert_a, cert_b for procedures that produce different structural outputs.
    When cert_a/cert_b omitted, uses record["edits"] for both (same structural path).

    Returns dict with terminal_diverged, distances, and S (procedural_summary).
    """
    edits = record.get("edits")
    if isinstance(edits, str):
        import json
        try:
            edits = json.loads(edits)
        except json.JSONDecodeError:
            edits = []
    edits = edits or []

    struct_a = cert_a if cert_a is not None else edits
    struct_b = cert_b if cert_b is not None else edits

    ann_a = record.get(proc_a) if proc_a in record else None
    ann_b = record.get(proc_b) if proc_b in record else None

    stage_a = {"structural": struct_a, "annotation": ann_a}
    stage_b = {"structural": struct_b, "annotation": ann_b}

    S = procedural_summary(
        record, proc_a, proc_b, stage_a, stage_b, annotation_threshold
    )

    if length == 2:
        terminal_diverged = S["structural_diverged"]
        terminal_distance = S["structural_distance"]
    elif length == 3:
        terminal_diverged = S["annotation_diverged"]
        terminal_distance = S["annotation_distance"]
    else:
        terminal_diverged = False
        terminal_distance = 0.0

    structural_agreement = 1.0 - S["structural_distance"]
    sem_dist = S.get("annotation_distance", 1.0) or 1.0
    semantic_agreement = 1.0 - sem_dist
    gap = structural_agreement - semantic_agreement

    return {
        "instance_id": record.get("instance_id"),
        "proc_a": proc_a,
        "proc_b": proc_b,
        "length": length,
        "terminal_diverged": terminal_diverged,
        "terminal_distance": terminal_distance,
        "structural_agreement": structural_agreement,
        "semantic_agreement": semantic_agreement,
        "gap": gap,
        "S": S,
    }


def build_procedure_divergence_matrix(
    records: list[dict],
    procedures: list[str] | None = None,
    length: int = 3,
    annotation_threshold: float = 0.3,
) -> list[dict]:
    """
    For each instance and each procedure pair, compute divergence.

    procedures: e.g. ["behavioral", "mechanistic", "functional"].
    Returns list of procedure_pair_divergence outputs.
    """
    procedures = procedures or ["behavioral", "mechanistic", "functional"]
    results = []
    for rec in records:
        for i, pa in enumerate(procedures):
            for pb in procedures[i + 1:]:
                if pa != pb:
                    out = procedure_pair_divergence(
                        rec, pa, pb, length=length, annotation_threshold=annotation_threshold
                    )
                    results.append(out)
    return results
