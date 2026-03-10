"""
Pipeline utilities: extraction, serialization, HF token.

Extraction: apply computed representations (edits, modules, motifs) to traces.
"""

import json
import os
from collections.abc import Iterator
from typing import Any

from representations import file_edit_graph_repr, semantic_edits_repr_trace
from representations.computed.motifs.motifs import motifs_repr_structural


def get_hf_token() -> str | None:
    """HF token from env or huggingface_hub default locations."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        return None


def serialize_for_storage(rec: dict[str, Any]) -> dict[str, Any]:
    """Convert complex values to JSON strings for parquet compatibility."""
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v, default=str)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _edits_certificate(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract structural edit certificates per code_change event."""
    structs = semantic_edits_repr_trace(
        trace,
        include_prompts=False,
        include_intent=False,
        return_structural=True,
    )
    if not isinstance(structs, list):
        return []
    certs = []
    for s in structs:
        if isinstance(s, dict):
            certs.append({
                "operations": s.get("operations", []),
                "delta": s.get("delta", 0),
            })
    return certs


def _modules_subgraph(trace: dict[str, Any]) -> list[str]:
    """Trace-based co-edit subgraph."""
    return file_edit_graph_repr(trace)


def _motifs_certificate(trace: dict[str, Any]) -> dict[str, Any]:
    """Event subsequences + soft membership over motif vocabulary."""
    return motifs_repr_structural(trace)


def apply_computed_representations(trace: dict[str, Any]) -> dict[str, Any]:
    """Run edits, modules, motifs. Return well-formed certificates."""
    rec: dict[str, Any] = {
        "instance_id": trace.get("instance_id"),
        "repo": trace.get("repo"),
        "base_commit": trace.get("base_commit"),
    }
    try:
        rec["edits"] = _edits_certificate(trace)
    except (TypeError, KeyError, ValueError) as e:
        rec["edits"] = []
        rec["edits_error"] = str(e)
    try:
        rec["modules"] = _modules_subgraph(trace)
    except (TypeError, KeyError, ValueError) as e:
        rec["modules"] = []
        rec["modules_error"] = str(e)
    try:
        rec["motifs"] = _motifs_certificate(trace)
    except (TypeError, KeyError, ValueError) as e:
        rec["motifs"] = {}
        rec["motifs_error"] = str(e)
    return rec


def extract_dataset(
    loader: Iterator[dict[str, Any]],
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield certificate records from a trace loader."""
    for i, trace in enumerate(loader):
        if limit is not None and i >= limit:
            break
        yield apply_computed_representations(trace)
