"""
Build pairwise distance matrices from certificate records.

Input: list of records with edits, modules, motifs (and optionally behavioral, mechanistic).
Output: dict of (n, n) symmetric distance matrices + stratum labels.

Approaches:
- jaccard: Jaccard on sets (edits op types, module tokens), cosine on motifs
- structural: Levenshtein on tokens, tree edit on edits (when ast available), graph distance on modules
"""

from typing import Any

import numpy as np


def _jaccard_set(a: set, b: set) -> float:
    """Jaccard distance 1 - |intersection|/|union|."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else 1.0 - inter / union


def _set_diff_distance(a: set, b: set) -> float:
    """
    Symmetric set difference: |A Δ B| = number of add/remove ops to transform A into B.
    Normalized by |A| + |B| so output is in [0, 1].
    """
    if not a and not b:
        return 0.0
    sym_diff = len(a ^ b)
    total = len(a) + len(b)
    return sym_diff / total if total > 0 else 0.0


def _op_types(rec: dict) -> set[str]:
    """Extract operation type set from edits certificates."""
    edits = rec.get("edits") or []
    if isinstance(edits, str):
        import json

        try:
            edits = json.loads(edits)
        except json.JSONDecodeError:
            return set()
    if not isinstance(edits, list):
        return set()
    types = set()
    for cert in edits:
        if not isinstance(cert, dict):
            continue
        for op in cert.get("operations", []):
            if isinstance(op, dict) and op.get("type"):
                types.add(str(op["type"]))
    return types


def _edits_distance(rec_a: dict, rec_b: dict) -> float:
    """Jaccard distance on aggregated operation type sets."""
    return _jaccard_set(_op_types(rec_a), _op_types(rec_b))


def _edits_distance_set_diff(rec_a: dict, rec_b: dict) -> float:
    """Symmetric set difference on op types: |A Δ B| / (|A| + |B|). Number of ops to transform."""
    return _set_diff_distance(_op_types(rec_a), _op_types(rec_b))


def _edits_distance_tree(rec_a: dict, rec_b: dict) -> float:
    """Tree edit distance when ast_after_dump available; else Jaccard fallback."""
    def certs(rec: dict) -> list:
        edits = rec.get("edits") or []
        if isinstance(edits, str):
            import json

            try:
                edits = json.loads(edits)
            except json.JSONDecodeError:
                return []
        return [c for c in edits if isinstance(c, dict)] if isinstance(edits, list) else []

    ca, cb = certs(rec_a), certs(rec_b)
    if not ca or not cb:
        return _jaccard_set(_op_types(rec_a), _op_types(rec_b))

    try:
        from representations.computed.edits.distance import certificate_distance
    except ImportError:
        return _jaccard_set(_op_types(rec_a), _op_types(rec_b))

    best = float("inf")
    for c1 in ca:
        for c2 in cb:
            if c1.get("ast_after_dump") or c2.get("ast_after_dump"):
                d, _ = certificate_distance(c1, c2)
                best = min(best, d)
    if best != float("inf"):
        max_d = 100.0
        return min(best / max_d, 1.0)
    return _jaccard_set(_op_types(rec_a), _op_types(rec_b))


def _module_tokens(rec: dict) -> set[str]:
    """Extract module token set."""
    mods = rec.get("modules") or []
    if isinstance(mods, str):
        import json

        try:
            mods = json.loads(mods)
        except json.JSONDecodeError:
            return set()
    return set(mods) if isinstance(mods, list) else set()


def _modules_distance(rec_a: dict, rec_b: dict) -> float:
    """Jaccard distance on module token sets."""
    return _jaccard_set(_module_tokens(rec_a), _module_tokens(rec_b))


def _modules_edges(rec: dict) -> list[tuple[str, str]]:
    """Extract co-edit edges for graph distance."""
    edges = rec.get("modules_edges") or []
    if isinstance(edges, str):
        import json

        try:
            edges = json.loads(edges)
        except json.JSONDecodeError:
            return []
    if not isinstance(edges, list):
        return []
    out = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            out.append((str(e[0]), str(e[1])))
    return out


def _modules_distance_graph(rec_a: dict, rec_b: dict) -> float:
    """Graph distance (symmetric edge diff) when modules_edges available; else Jaccard fallback."""
    import networkx as nx

    edges_a = set(_modules_edges(rec_a))
    edges_b = set(_modules_edges(rec_b))
    if not edges_a and not edges_b:
        return _jaccard_set(_module_tokens(rec_a), _module_tokens(rec_b))

    try:
        from representations.computed.modules.distance import graph_distance
    except ImportError:
        return _jaccard_set(_module_tokens(rec_a), _module_tokens(rec_b))

    Ga = nx.Graph()
    Ga.add_edges_from(edges_a)
    Gb = nx.Graph()
    Gb.add_edges_from(edges_b)
    d, _ = graph_distance(Ga, Gb)
    return float(d)


def _tokens_distance_levenshtein(rec_a: dict, rec_b: dict) -> float:
    """Levenshtein distance on token sequences, normalized by max length."""
    def tokens(rec: dict) -> list[str]:
        t = rec.get("tokens") or []
        if isinstance(t, str):
            import json

            try:
                t = json.loads(t)
            except json.JSONDecodeError:
                return []
        return list(t) if isinstance(t, list) else []

    a, b = tokens(rec_a), tokens(rec_b)
    if not a and not b:
        return 0.0
    seq_a = " ".join(str(x) for x in a)
    seq_b = " ".join(str(x) for x in b)
    try:
        import Levenshtein

        dist = Levenshtein.distance(seq_a, seq_b)
        max_len = max(len(seq_a), len(seq_b), 1)
        return min(dist / max_len, 1.0)
    except ImportError:
        return _jaccard_set(set(a), set(b))


def _motifs_distance(rec_a: dict, rec_b: dict) -> float:
    """Cosine distance between soft_membership vectors."""
    def soft_membership(rec: dict) -> dict[str, float]:
        motifs = rec.get("motifs") or {}
        if isinstance(motifs, str):
            import json
            try:
                motifs = json.loads(motifs)
            except json.JSONDecodeError:
                return {}
        if not isinstance(motifs, dict):
            return {}
        sm = motifs.get("soft_membership", {})
        return {k: float(v) for k, v in (sm or {}).items()} if isinstance(sm, dict) else {}

    try:
        from representations.computed.motifs.distance import motif_distance
    except ImportError:
        return 0.0

    va, vb = soft_membership(rec_a), soft_membership(rec_b)
    if not va and not vb:
        return 0.0
    return motif_distance(va, vb)


def _stratum_label(rec: dict) -> str:
    """Derive stratum from repo or instance_id."""
    repo = rec.get("repo")
    if repo:
        return str(repo)
    instance_id = rec.get("instance_id", "")
    if "__" in instance_id:
        return instance_id.split("-")[0] if "-" in instance_id else instance_id.split("__")[0]
    return instance_id or "unknown"


APPROACH_JACCARD = "jaccard"
APPROACH_STRUCTURAL = "structural"
APPROACH_BOTH = "both"


def build_distance_matrices(
    records: list[dict[str, Any]],
    repr_keys: list[str] | None = None,
    approach: str = APPROACH_JACCARD,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Build pairwise distance matrices for each representation.

    approach: "jaccard" (default), "structural", or "both"
      - jaccard: edits (Jaccard), modules (Jaccard), motifs (cosine)
      - structural: tokens (Levenshtein), edits_tree, modules_graph, motifs (cosine)
      - both: all of the above

    Returns (matrices, labels) where matrices is {repr_name: D} and labels is (n,) stratum.
    """
    n = len(records)
    labels = np.array([_stratum_label(r) for r in records])

    distance_fns = {
        "edits": _edits_distance,
        "edits_set_diff": _edits_distance_set_diff,
        "modules": _modules_distance,
        "motifs": _motifs_distance,
    }
    if approach in (APPROACH_STRUCTURAL, APPROACH_BOTH):
        distance_fns.update({
            "tokens": _tokens_distance_levenshtein,
            "edits_tree": _edits_distance_tree,
            "modules_graph": _modules_distance_graph,
        })

    repr_keys = repr_keys or list(distance_fns)
    matrices = {}
    for key in repr_keys:
        fn = distance_fns.get(key)
        if not fn:
            continue
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = fn(records[i], records[j])
                D[i, j] = d
                D[j, i] = d
        matrices[key] = D

    return matrices, labels
