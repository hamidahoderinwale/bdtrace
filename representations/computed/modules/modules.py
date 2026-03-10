"""
Modules: computed, inter-file, import graph + co-edit history.

Input: repo snapshot + git log.
Grounding: import graph + co-edit history. Unit: file-level dependency edge.
Distance: distance.graph_edit_distance (networkx).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from pydriller import Repository

from .import_extractor import extract_imports_from_file, neighbors_of


def module_graph_repr(
    repo_path: str | Path,
    commit: str | None = None,
    touched_files: list[str] | None = None,
) -> dict[str, Any]:
    """Build module graph from repository snapshot."""
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        return _empty_graph()

    _skip_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git"}
    all_py = [p for p in repo_path.rglob("*.py") if not any(s in p.parts for s in _skip_dirs)]
    if touched_files:
        files_to_analyze = list(neighbors_of(touched_files, repo_path, all_py))
        files_to_analyze = [f for f in files_to_analyze if f.suffix == ".py"]
    else:
        files_to_analyze = all_py

    import_edges = []
    repo_resolved = repo_path.resolve()
    for fp in files_to_analyze:
        try:
            for src, tgt in extract_imports_from_file(fp, repo_path):
                cand = Path(tgt) if Path(tgt).is_absolute() else repo_path / tgt
                try:
                    if cand.exists():
                        cand_res = cand.resolve()
                        cand_res.relative_to(repo_resolved)
                        import_edges.append((src, str(cand_res)))
                except (ValueError, OSError):
                    pass
        except (SyntaxError, OSError):
            continue

    coedit_edges = []
    if commit:
        try:
            for c in Repository(str(repo_path), to_commit=commit).traverse_commits():
                touched = []
                for mf in c.modified_files:
                    path = getattr(mf, "new_path", None) or getattr(mf, "old_path", None) or mf.filename
                    if path and path.endswith(".py"):
                        touched.append(path)
                for i, f1 in enumerate(touched):
                    for f2 in touched[i + 1 :]:
                        if f1 != f2:
                            coedit_edges.append((f1, f2))
        except (OSError, ValueError):
            pass

    G = nx.DiGraph()
    for src, tgt in import_edges:
        G.add_edge(src, tgt, edge_type="import")

    coedit_counts: dict[tuple[str, str], int] = {}
    for f1, f2 in coedit_edges:
        key = (f1, f2) if f1 < f2 else (f2, f1)
        coedit_counts[key] = coedit_counts.get(key, 0) + 1
    for (f1, f2), w in coedit_counts.items():
        if G.has_edge(f1, f2):
            G[f1][f2]["coedit_weight"] = G[f1][f2].get("coedit_weight", 0) + w
        elif G.has_edge(f2, f1):
            G[f2][f1]["coedit_weight"] = G[f2][f1].get("coedit_weight", 0) + w
        else:
            G.add_edge(f1, f2, coedit_weight=w)

    coedit_edges_with_weight = [(f1, f2, {"weight": w}) for (f1, f2), w in coedit_counts.items()]

    return {
        "nodes": list(G.nodes()),
        "import_edges": import_edges,
        "coedit_edges": coedit_edges_with_weight,
        "graph": G,
    }


def _empty_graph() -> dict[str, Any]:
    return {"nodes": [], "import_edges": [], "coedit_edges": [], "graph": nx.DiGraph()}


def module_graph_repr_list(
    repo_path: str | Path,
    commit: str | None = None,
    touched_files: list[str] | None = None,
) -> list[str]:
    """Serialize module graph to list of strings."""
    out = module_graph_repr(repo_path, commit, touched_files)
    tokens = [f"IMPORT_{Path(a).stem}_{Path(b).stem}" for a, b in out["import_edges"]]
    for a, b, d in out["coedit_edges"]:
        tokens.append(f"COEDIT_{Path(a).stem}_{Path(b).stem}_{d.get('weight', 1)}")
    tokens.append(f"NODES_{len(out['nodes'])}")
    tokens.append(f"EDGES_{len(out['import_edges']) + len(out['coedit_edges'])}")
    return tokens


def file_edit_graph_repr(trace: dict, time_window_sec: int = 300) -> list[str]:
    """Trace-based co-edit graph (fallback when repo not on disk)."""
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
    edges = set()
    for i, e1 in enumerate(edits):
        for e2 in edits[i + 1 :]:
            if e1["path"] == e2["path"]:
                continue
            if 0 <= e2["time"] - e1["time"] <= time_window_sec:
                edges.add((name_map[e1["path"]], name_map[e2["path"]]))
                break
    tokens = [f"E_{s}_{d}" for s, d in sorted(edges)]
    edit_counts = {}
    for e in edits:
        f = name_map[e["path"]]
        edit_counts[f] = edit_counts.get(f, 0) + 1
    for f, count in sorted(edit_counts.items()):
        tokens.append(f"EDITS_{f}_{count}")
    tokens.append(f"NODES_{len(all_files)}")
    tokens.append(f"EDGES_{len(edges)}")
    return tokens


def file_edit_graph_repr_str(trace: dict, time_window_sec: int = 300, limit: int = 100) -> str:
    tokens = file_edit_graph_repr(trace, time_window_sec)
    if not tokens:
        return "EMPTY_GRAPH"
    res = " ".join(tokens[:limit])
    if len(tokens) > limit:
        res += " ... [truncated]"
    return res
