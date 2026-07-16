"""
Pipeline utilities: extraction, serialization, HF token.

Extraction: apply computed representations (edits, modules, motifs, tokens) to traces.
"""

import ast
import json
import os
from collections.abc import Iterator
from typing import Any

from representations import file_edit_graph_repr, semantic_edits_repr_trace, tokens_repr
from representations.computed.motifs.motifs import motifs_repr_from_certificates


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


def _edits_certificate(trace: dict[str, Any], include_ast_dump: bool = True) -> list[dict[str, Any]]:
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
        if not isinstance(s, dict):
            continue
        cert = {
            "operations": s.get("operations", []),
            "delta": s.get("delta", 0),
        }
        if include_ast_dump:
            tree = s.get("ast_after") or s.get("ast_before")
            if tree is not None:
                try:
                    cert["ast_after_dump"] = ast.dump(tree)
                except (TypeError, ValueError):
                    pass
        certs.append(cert)
    return certs


def _modules_subgraph(trace: dict[str, Any]) -> list[str]:
    """Trace-based co-edit subgraph. Preserve modules_from_repo when already computed (e.g. diff_resolution)."""
    if trace.get("modules_from_repo") and trace.get("modules") is not None:
        return trace["modules"]
    return file_edit_graph_repr(trace)


def _modules_edges(trace: dict[str, Any]) -> list[tuple[str, str]]:
    """Trace-based co-edit edges for graph distance. Returns [(n1, n2), ...]."""
    import json
    from datetime import datetime
    from pathlib import Path

    edits = []
    for event in trace.get("events", []):
        details = event.get("details", {})
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except (json.JSONDecodeError, TypeError):
                details = {}
        file_path = details.get("file_path") or details.get("file")
        ts = event.get("timestamp", 0)
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                ts = 0
        if file_path:
            edits.append({"time": ts or 0, "path": file_path})

    if len(edits) < 2:
        return []
    all_files = list({e["path"] for e in edits})
    name_map = {f: Path(f).stem for f in all_files}
    time_window_sec = 300
    edges = set()
    for i, e1 in enumerate(edits):
        for e2 in edits[i + 1 :]:
            if e1["path"] == e2["path"]:
                continue
            if 0 <= e2["time"] - e1["time"] <= time_window_sec:
                edges.add((name_map[e1["path"]], name_map[e2["path"]]))
                break
    return sorted(edges)


def _motifs_certificate(edits_certs: list[dict[str, Any]]) -> dict[str, Any]:
    """Structural co-occurrence motifs from edit certificates (diff-based, SWE-bench)."""
    return motifs_repr_from_certificates(edits_certs)


_TASK_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("refactoring",    ["refactor", "cleanup", "clean up", "reorganize", "restructure", "rename"]),
    ("api_usage",      ["import", "library", "package", "module", "install", "dependency", "api"]),
    ("algorithmic",    ["implement", "algorithm", "sort", "search", "dynamic programming",
                        "recursion", "complexity", "leetcode", "competitive"]),
    ("code_generation",["write a function", "write a program", "implement a function",
                        "create a function", "define a function", "given the following"]),
    ("bug_fix",        ["fix", "bug", "error", "exception", "traceback", "fails", "broken",
                        "incorrect", "wrong", "issue", "problem", "crash"]),
]


def _infer_task_type(trace: dict[str, Any]) -> str:
    """Infer semantic task type from problem statement when not set by loader."""
    if trace.get("task_type"):
        return trace["task_type"]
    text = ""
    for event in trace.get("events", []):
        if event.get("type") == "prompt":
            text = (event.get("details", {}).get("text") or "").lower()
            break
    if not text:
        prompts = trace.get("prompts", [])
        if prompts:
            text = (prompts[0].get("text") or "").lower()
    for task_type, keywords in _TASK_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return task_type
    return "bug_fix"


def _prompt_length(trace: dict[str, Any]) -> int:
    """Character length of the problem statement / first prompt."""
    for event in trace.get("events", []):
        if event.get("type") == "prompt":
            text = event.get("details", {}).get("text", "") or ""
            return len(text)
    prompts = trace.get("prompts", [])
    if prompts:
        return len((prompts[0].get("text") or prompts[0].get("content") or ""))
    return 0


def apply_computed_representations(
    trace: dict[str, Any],
    include_structural: bool = True,
) -> dict[str, Any]:
    """Run edits, modules, motifs, tokens. Return well-formed certificates."""
    rec: dict[str, Any] = {
        "instance_id": trace.get("instance_id"),
        "repo": trace.get("repo"),
        "base_commit": trace.get("base_commit"),
        "modules_from_repo": trace.get("modules_from_repo", False),
        "prompt_length": _prompt_length(trace),
        "task_type": _infer_task_type(trace),
    }
    try:
        rec["edits"] = _edits_certificate(trace, include_ast_dump=include_structural)
    except (TypeError, KeyError, ValueError) as e:
        rec["edits"] = []
        rec["edits_error"] = str(e)
    try:
        rec["modules"] = _modules_subgraph(trace)
    except (TypeError, KeyError, ValueError) as e:
        rec["modules"] = []
        rec["modules_error"] = str(e)
    if include_structural:
        try:
            if trace.get("modules_from_repo") and trace.get("modules_edges") is not None:
                rec["modules_edges"] = trace["modules_edges"]
            else:
                rec["modules_edges"] = _modules_edges(trace)
        except (TypeError, KeyError, ValueError):
            rec["modules_edges"] = []
    try:
        rec["motifs"] = _motifs_certificate(rec.get("edits", []))
    except (TypeError, KeyError, ValueError) as e:
        rec["motifs"] = {}
        rec["motifs_error"] = str(e)
    try:
        rec["tokens"] = tokens_repr(trace, include_prompts=True)
    except (TypeError, KeyError, ValueError) as e:
        rec["tokens"] = []
        rec["tokens_error"] = str(e)
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
