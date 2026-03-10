"""Base utilities for inferred representations: grounding serialization."""

from typing import Any


def format_structural_certificate(certificate: dict[str, Any] | list[dict]) -> str:
    """Serialize edits certificate for LLM input."""
    if isinstance(certificate, list):
        ops = certificate
    else:
        ops = certificate.get("operations", []) if certificate else []
    if not ops:
        return "No structural changes recorded."
    lines = []
    for i, op in enumerate(ops[:20], 1):
        if isinstance(op, dict):
            t, loc, nt = op.get("type", "unknown"), op.get("location", ""), op.get("node_type", "")
            parts = [f"{i}. {t}"]
            if loc:
                parts.append(f"at {loc}")
            if nt:
                parts.append(f"({nt})")
            lines.append(" ".join(parts))
        else:
            lines.append(str(op))
    return "\n".join(lines)


def format_module_context(module_graph: dict[str, Any] | list[str]) -> str:
    """Serialize module subgraph for LLM input."""
    if isinstance(module_graph, list):
        return "\n".join(module_graph) if module_graph else "No module context."
    if not module_graph:
        return "No module context."
    lines = [f"IMPORT: {a} -> {b}" for a, b in module_graph.get("import_edges", [])[:30]]
    for a, b, d in module_graph.get("coedit_edges", [])[:20]:
        lines.append(f"COEDIT: {a} <-> {b} (weight={d.get('weight', 1)})")
    return "\n".join(lines) if lines else "No import or co-edit edges."
