"""
SWE-bench trajectory fetcher and feature extractor.

Fetches agent trajectories from S3 (SWE-bench/experiments submissions),
extracts procedural features per instance, and caches locally.

Reuses:
- data/agent_trajectories.py: trajectory → events conversion pattern
- representations/computed/modules/import_extractor.py: import graph for hop distance
- fetch_swebench_results.py: model/split conventions

Features extracted per instance:
- n_steps, n_edits, n_searches, n_opens, n_runs, n_nav
- edit_retries: consecutive EDIT pairs (stuck-in-loop signal)
- files_opened: ordered list of .py files the agent opened
- files_edited: list of .py files the agent actually changed
- action_sequence: compressed action type sequence (e.g. CREATE RUN SEARCH OPEN EDIT ...)
- submitted: whether agent reached submit
- exit_status
- hop_distance_min: min hop distance from any opened test file to any edited file (via import graph)
"""

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com"


## Action classification

def _classify_action(action: str) -> str:
    a = action.strip().lower()
    if a.startswith("edit"):
        return "EDIT"
    if a.startswith("open"):
        return "OPEN"
    if a.startswith("search"):
        return "SEARCH"
    if a.startswith("python") or a.startswith("bash") or a.startswith("./"):
        return "RUN"
    if a.startswith("create"):
        return "CREATE"
    if a.startswith("scroll") or a.startswith("goto"):
        return "NAV"
    if "submit" in a:
        return "SUBMIT"
    return "OTHER"


def _extract_py_file(action: str) -> str | None:
    m = re.search(r"([\w/._-]+\.py)", action)
    return m.group(1) if m else None


## Core extractor

def extract_trajectory_features(traj_data: dict) -> dict[str, Any]:
    """
    Extract procedural features from a raw .traj dict.

    Returns flat feature dict ready for a DataFrame row.
    """
    traj = traj_data.get("trajectory", [])
    info = traj_data.get("info", {})

    actions = [s.get("action", "") for s in traj]
    types = [_classify_action(a) for a in actions]

    files_opened: list[str] = []
    files_edited: list[str] = []
    seen_opened: set[str] = set()

    for action, atype in zip(actions, types):
        f = _extract_py_file(action)
        if not f:
            continue
        if atype == "OPEN" and f not in seen_opened:
            files_opened.append(f)
            seen_opened.add(f)
        elif atype == "EDIT":
            files_edited.append(f)

    # Edit retry: consecutive EDIT steps (agent stuck editing same area repeatedly)
    edit_retries = sum(
        1 for i in range(1, len(types)) if types[i] == "EDIT" and types[i - 1] == "EDIT"
    )

    # Compressed action sequence: run-length encode types
    compressed: list[str] = []
    for t in types:
        if not compressed or compressed[-1] != t:
            compressed.append(t)

    return {
        "n_steps": len(traj),
        "n_edits": types.count("EDIT"),
        "n_searches": types.count("SEARCH"),
        "n_opens": types.count("OPEN"),
        "n_runs": types.count("RUN"),
        "n_nav": types.count("NAV"),
        "edit_retries": edit_retries,
        "edit_retry_rate": edit_retries / max(types.count("EDIT"), 1),
        "files_opened": files_opened,
        "files_edited": list(dict.fromkeys(files_edited)),  # deduplicated, ordered
        "action_sequence": " ".join(compressed),
        "submitted": "SUBMIT" in types,
        "exit_status": info.get("exit_status", ""),
        "n_files_opened": len(files_opened),
        "n_files_edited": len(set(files_edited)),
    }


## Hop distance via import graph

def compute_hop_distance(
    files_opened: list[str],
    files_edited: list[str],
    repo_path: Path,
) -> int | None:
    """
    Approximate hop distance between test files opened and edited fix files.

    Uses the import graph (BFS) from import_extractor. Returns min hops
    from any test file to any edited file, or None if unreachable.
    """
    try:
        from representations.computed.modules.import_extractor import extract_imports_from_file

        test_files = [f for f in files_opened if "test" in f]
        fix_files = files_edited
        if not test_files or not fix_files:
            return None

        # Build adjacency from imports in all relevant files
        all_files = list(dict.fromkeys(files_opened + files_edited))
        graph: dict[str, set[str]] = {}
        for rel in all_files:
            fp = repo_path / rel
            if not fp.exists():
                continue
            try:
                edges = extract_imports_from_file(fp, repo_path)
                src_stem = str(rel)
                graph.setdefault(src_stem, set())
                for _, tgt in edges:
                    # Normalize target to relative path
                    tgt_path = Path(tgt)
                    try:
                        rel_tgt = str(tgt_path.relative_to(repo_path)) if tgt_path.is_absolute() else tgt
                        graph[src_stem].add(rel_tgt)
                    except ValueError:
                        graph[src_stem].add(tgt)
            except (SyntaxError, OSError):
                continue

        # BFS from each test file
        fix_set = set(fix_files)
        min_hops = None
        for start in test_files:
            visited = {start}
            queue = [(start, 0)]
            while queue:
                node, hops = queue.pop(0)
                if node in fix_set:
                    if min_hops is None or hops < min_hops:
                        min_hops = hops
                    break
                for neighbor in graph.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, hops + 1))
        return min_hops
    except (ImportError, OSError):
        return None


## Fetcher

def fetch_trajectory(
    instance_id: str,
    model_id: str,
    split: str = "lite",
    cache_dir: Path | None = None,
    timeout: int = 20,
) -> dict | None:
    """
    Fetch one trajectory from S3, with local cache.

    Returns raw .traj dict or None on failure.
    """
    if cache_dir:
        cache_path = cache_dir / model_id / f"{instance_id}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    url = f"{S3_BASE}/{split}/{model_id}/trajs/{instance_id}.traj"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
        if cache_dir:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
        return data
    except Exception:
        return None


def fetch_and_extract(
    instance_id: str,
    model_id: str,
    split: str = "lite",
    cache_dir: Path | None = None,
    repo_path: Path | None = None,
) -> dict[str, Any] | None:
    """Fetch trajectory and return feature dict, or None on failure."""
    raw = fetch_trajectory(instance_id, model_id, split, cache_dir)
    if raw is None:
        return None
    features = extract_trajectory_features(raw)
    features["instance_id"] = instance_id
    features["model_id"] = model_id

    if repo_path and repo_path.exists():
        features["hop_distance_min"] = compute_hop_distance(
            features["files_opened"], features["files_edited"], repo_path
        )
    else:
        features["hop_distance_min"] = None

    return features


def fetch_all(
    instance_ids: list[str],
    model_id: str,
    split: str = "lite",
    cache_dir: Path | None = None,
    repos_dir: Path | None = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """
    Fetch + extract features for all instances in parallel.

    Uses ThreadPoolExecutor; HTTP requests are I/O-bound so threads work well.
    """
    from tqdm import tqdm

    results = []

    def _work(iid: str) -> dict | None:
        repo_path = None
        if repos_dir:
            repo_name = iid.split("__")[0].split("/")[-1] if "__" in iid else None
            if repo_name:
                rp = repos_dir / repo_name
                if rp.exists():
                    repo_path = rp
        return fetch_and_extract(iid, model_id, split, cache_dir, repo_path)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, iid): iid for iid in instance_ids}
        with tqdm(total=len(futures), desc=f"Fetching {model_id}") as bar:
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    results.append(result)
                bar.update(1)

    return results
