"""
Full-file diff resolution for SWE-bench Lite.

Patch parsing yields hunk content only; AST-based edits need full file content.
This module clones repos, checkouts base_commit, applies patch, and reads full before/after.

When repo is available, also computes full module graph (import + co-edit) and attaches
to trace. See docs/MODULES_GAP.md.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .swe_bench import _parse_patch, swe_bench_instance_to_trace


def _clone_or_use_repo(repo: str, cache_dir: Path) -> Path:
    """Clone repo to cache or return existing path. Full clone for base_commit access."""
    repo_name = repo.split("/")[-1]
    repo_path = cache_dir / repo_name
    if repo_path.exists() and (repo_path / ".git").exists():
        return repo_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(repo_path)],
        check=True,
        capture_output=True,
    )
    return repo_path


def resolve_patch_to_full_files(
    instance: dict[str, Any],
    repos_cache: Path,
) -> dict[str, Any]:
    """
    Resolve patch to full before/after file content via clone + checkout + apply.

    Returns trace with full before_content, after_content per code_change event.
    Falls back to hunk content if resolution fails.
    """
    trace = swe_bench_instance_to_trace(instance)
    repo = instance.get("repo")
    base_commit = instance.get("base_commit")
    patch = instance.get("patch") or ""
    if not repo or not patch:
        return trace

    ## SWE-Smith repos live at swesmith/{repo_name_with_hash} on GitHub.
    ## The cloned HEAD is already the base state — no checkout needed.
    is_swesmith = repo.startswith("swesmith/")
    if not is_swesmith and not base_commit:
        return trace

    file_changes = _parse_patch(patch)
    if not file_changes:
        return trace

    try:
        repo_path = _clone_or_use_repo(repo, repos_cache)
    except subprocess.CalledProcessError:
        return trace

    if not is_swesmith:
        # Standard SWE-bench: checkout the base commit
        result = subprocess.run(
            ["git", "checkout", "--force", base_commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return trace

    # Before: read each file at base_commit
    for fc in file_changes:
        fp = fc["file_path"]
        full_path = repo_path / fp
        if full_path.exists():
            try:
                fc["before_content"] = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

    # After: apply main patch only (test_patch adds test files, not in file_changes)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        patch_tmp = Path(f.name)
    try:
        subprocess.run(
            ["git", "apply", "--index", str(patch_tmp)],
            cwd=repo_path,
            capture_output=True,
        )
    finally:
        patch_tmp.unlink(missing_ok=True)

    for fc in file_changes:
        fp = fc["file_path"]
        full_path = repo_path / fp
        if full_path.exists():
            try:
                fc["after_content"] = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

    # Rebuild trace with resolved content
    events = []
    for fc in file_changes:
        events.append(
            {
                "type": "code_change",
                "details": {
                    "file_path": fc["file_path"],
                    "before_content": fc.get("before_content", ""),
                    "after_content": fc.get("after_content", ""),
                    "lines_added": fc.get("lines_added", 0),
                    "lines_removed": fc.get("lines_removed", 0),
                    "diff_summary": f"+{fc.get('lines_added', 0)}, -{fc.get('lines_removed', 0)}",
                },
                "timestamp": instance.get("created_at"),
            }
        )

    problem = instance.get("problem_statement") or ""
    if problem:
        events.insert(
            0,
            {
                "type": "prompt",
                "details": {"text": problem, "content": problem},
                "timestamp": instance.get("created_at"),
            },
        )

    trace["events"] = events

    # Import-only module graph. Only parse touched files + their direct neighbors
    # (not the full repo) — avoids O(all_repo_files) AST parsing per instance.
    try:
        from representations.computed.modules.import_extractor import extract_imports_from_file

        _skip_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git"}
        touched = [repo_path / fc["file_path"] for fc in file_changes]
        touched = [p for p in touched if p.exists()]

        # Parse only touched files to find their direct import neighbors
        import_edges = []
        neighbor_paths: set[Path] = set()
        repo_resolved = repo_path.resolve()
        for fp in touched:
            try:
                for src, tgt in extract_imports_from_file(fp, repo_path):
                    cand = Path(tgt) if Path(tgt).is_absolute() else repo_path / tgt
                    try:
                        if cand.exists():
                            cand_res = cand.resolve()
                            cand_res.relative_to(repo_resolved)
                            import_edges.append((Path(src).stem, cand_res.stem))
                            neighbor_paths.add(cand_res)
                    except (ValueError, OSError):
                        pass
            except (SyntaxError, OSError):
                continue

        # Parse neighbors to find back-edges into touched files
        touched_resolved = {p.resolve() for p in touched}
        for fp in neighbor_paths - touched_resolved:
            try:
                for src, tgt in extract_imports_from_file(fp, repo_path):
                    cand = Path(tgt) if Path(tgt).is_absolute() else repo_path / tgt
                    try:
                        if cand.exists() and cand.resolve() in touched_resolved:
                            import_edges.append((Path(src).stem, cand.resolve().stem))
                    except (ValueError, OSError):
                        pass
            except (SyntaxError, OSError):
                continue

        tokens = [f"IMPORT_{a}_{b}" for a, b in import_edges]
        n_nodes = len({p.stem for p in touched} | {p.stem for p in neighbor_paths})
        tokens.append(f"NODES_{n_nodes}")
        tokens.append(f"EDGES_{len(import_edges)}")
        trace["modules"] = tokens
        trace["modules_edges"] = sorted(set(import_edges))
        trace["modules_from_repo"] = True
    except (ImportError, OSError, ValueError, TypeError):
        pass

    return trace
