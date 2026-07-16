"""
Scoped edit operations: enriched certificates with file-level and scope-level info.

Extends the existing edit certificate abstraction with:
  - File path and module (top-2 path components)
  - Scopes touched (function/class AST nodes overlapping the diff)
  - Line-level stats (lines added, removed, patch size, hunk count)
  - Layered similarity (file match, scope Jaccard, edit Jaccard)

This feeds the scoped certificate pipeline that decomposes structural
agreement into where-in-the-codebase vs what-was-changed.
"""

import ast
import difflib
import re
from pathlib import PurePosixPath

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence, patch_to_chunks
from analysis.procedures.contextual_edit_ops import _parse_diff_line_ranges

# Same normalization map used everywhere else
_NORMALIZE_OPS = {
    "ADD_if": "ADD_If", "DEL_if": "DEL_If",
    "ADD_for": "ADD_For", "DEL_for": "DEL_For",
    "ADD_return": "ADD_Return", "DEL_return": "DEL_Return",
    "ADD_raise": "ADD_Raise", "DEL_raise": "DEL_Raise",
    "ADD_try": "ADD_Try", "DEL_try": "DEL_Try",
    "ADD_while": "ADD_While", "DEL_while": "DEL_While",
    "ADD_with": "ADD_With", "DEL_with": "DEL_With",
    "ADD_def": "ADD_FunctionDef", "DEL_def": "DEL_FunctionDef",
    "ADD_class": "ADD_ClassDef", "DEL_class": "DEL_ClassDef",
    "ADD_elif": "ADD_If", "DEL_elif": "DEL_If",
    "ADD_else": "ADD_If", "DEL_else": "DEL_If",
    "ADD_except": "ADD_ExceptHandler", "DEL_except": "DEL_ExceptHandler",
    "ADD_assert": "ADD_Assert",
}

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")


def _file_module(file_path: str) -> str:
    """Top-2 path components, e.g. 'django/db' from 'django/db/models/sql/compiler.py'."""
    parts = PurePosixPath(file_path).parts
    if len(parts) >= 2:
        return str(PurePosixPath(parts[0]) / parts[1])
    return parts[0] if parts else ""


def _scopes_from_ast(source: str, changed_lines: set[int]) -> list[str]:
    """
    Find function/class scopes that overlap with changed_lines.

    Returns list like ["FunctionDef:get_columns", "ClassDef:SQLCompiler"].
    Only reports the innermost enclosing scope per changed line.
    """
    if not source.strip() or not changed_lines:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Collect all named scope nodes (functions and classes)
    scope_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                scope_nodes.append(node)

    # Sort by span size (smallest first) so innermost scopes win
    scope_nodes.sort(key=lambda n: n.end_lineno - n.lineno)

    scopes = set()
    for line in changed_lines:
        for node in scope_nodes:
            if node.lineno <= line <= node.end_lineno:
                kind = "FunctionDef" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "ClassDef"
                scopes.add(f"{kind}:{node.name}")
                break  # innermost scope for this line

    return sorted(scopes)


def _count_hunks(diff_text: str) -> int:
    """Count @@ hunk headers in a unified diff."""
    return sum(1 for line in diff_text.splitlines() if _HUNK_HEADER_RE.match(line))


def _line_stats(diff_text: str) -> tuple[int, int]:
    """Count lines added and removed from a unified diff."""
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _files_from_diff(diff_text: str) -> list[str]:
    """Extract file paths from diff --git headers."""
    files = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r"b/([\w/._ -]+\.py)$", line)
            if m:
                files.append(m.group(1))
    return files


def trace_to_scoped_cert(trace: dict) -> dict | None:
    """Extract enriched certificate from an oracle trace.

    Returns dict with:
      instance_id, edit_cert, file_path, file_module,
      scopes_touched, scope_types, lines_added, lines_removed,
      patch_size, hunk_count
    """
    instance_id = trace["instance_id"]

    # Collect code changes
    file_changes: dict[str, tuple[str, str]] = {}
    diff_parts: list[str] = []

    for ev in trace["events"]:
        if ev["type"] != "code_change":
            continue
        d = ev["details"]
        if not d["file_path"].endswith(".py"):
            continue

        fp = d["file_path"]
        before = d["before_content"] or ""
        after = d["after_content"] or ""
        if before == after:
            continue

        if fp not in file_changes:
            file_changes[fp] = (before, after)
        else:
            file_changes[fp] = (file_changes[fp][0], after)

        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        raw = "".join(difflib.unified_diff(
            before_lines, after_lines, fromfile=fp, tofile=fp,
        ))
        if raw:
            diff_parts.append(f"diff --git a/{fp} b/{fp}\n" + raw)

    if not diff_parts or not file_changes:
        return None

    combined_diff = "\n".join(diff_parts)

    # Edit certificate (Level 1)
    ops = patch_to_ast_sequence(combined_diff)
    edit_cert = sorted(set(_NORMALIZE_OPS.get(op, op) for op in ops))

    # File info (all traces are single-file, but handle multi-file gracefully)
    file_paths = list(file_changes.keys())
    primary_file = file_paths[0]

    # Scope extraction: use both before and after ASTs
    line_ranges = _parse_diff_line_ranges(combined_diff)
    all_scopes = []
    for fp, (deleted_lines, added_lines) in line_ranges.items():
        if fp not in file_changes:
            continue
        before_content, after_content = file_changes[fp]

        # Scopes from before AST (for deletions)
        if deleted_lines:
            scopes = _scopes_from_ast(before_content, deleted_lines)
            all_scopes.extend(scopes)

        # Scopes from after AST (for additions)
        if added_lines:
            scopes = _scopes_from_ast(after_content, added_lines)
            all_scopes.extend(scopes)

    scopes_touched = sorted(set(all_scopes))
    scope_types = sorted(set(s.split(":")[0] for s in scopes_touched))

    # Line stats
    lines_added, lines_removed = _line_stats(combined_diff)

    return {
        "instance_id": instance_id,
        "edit_cert": edit_cert,
        "file_path": primary_file,
        "file_paths": file_paths,
        "file_module": _file_module(primary_file),
        "scopes_touched": scopes_touched,
        "scope_types": scope_types,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "patch_size": lines_added + lines_removed,
        "hunk_count": _count_hunks(combined_diff),
    }


def patch_to_scoped_cert(
    patch_text: str,
    oracle_before_content: str,
    file_path: str,
) -> dict | None:
    """Extract scoped cert from an agent's raw patch.

    Uses oracle before_content for AST parsing (since we don't have
    the agent's intermediate file states).
    """
    if not patch_text or not patch_text.strip():
        return None

    # Edit certificate
    ops = patch_to_ast_sequence(patch_text)
    if not ops:
        return None
    edit_cert = sorted(set(_NORMALIZE_OPS.get(op, op) for op in ops))

    # Files touched by the agent
    agent_files = _files_from_diff(patch_text)
    if not agent_files:
        # Try to find any file reference
        agent_files = [file_path]

    primary_agent_file = agent_files[0] if agent_files else file_path

    # Scope extraction: use oracle before_content for the known file
    line_ranges = _parse_diff_line_ranges(patch_text)
    all_scopes = []

    for fp, (deleted_lines, added_lines) in line_ranges.items():
        # Only parse scopes for the oracle file (we have its content)
        if fp == file_path and oracle_before_content:
            # Scopes from deletions against before AST
            if deleted_lines:
                scopes = _scopes_from_ast(oracle_before_content, deleted_lines)
                all_scopes.extend(scopes)

            # For additions, we'd need after content. Approximate by
            # applying the patch mentally: if the added lines are near
            # deleted lines, they likely touch the same scope.
            # Use the before AST with a broadened line range.
            if added_lines:
                scopes = _scopes_from_ast(oracle_before_content, added_lines)
                all_scopes.extend(scopes)

    scopes_touched = sorted(set(all_scopes))
    scope_types = sorted(set(s.split(":")[0] for s in scopes_touched))

    # Line stats
    lines_added, lines_removed = _line_stats(patch_text)

    return {
        "edit_cert": edit_cert,
        "file_paths": agent_files,
        "file_module": _file_module(primary_agent_file),
        "scopes_touched": scopes_touched,
        "scope_types": scope_types,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "patch_size": lines_added + lines_removed,
        "hunk_count": _count_hunks(patch_text),
    }


def compute_layered_similarity(cert_a: dict, cert_b: dict) -> dict:
    """Compute similarity at file, scope, and edit-type levels.

    Returns:
      file_match: bool (any overlap in file_paths)
      scope_jaccard: float (Jaccard of scopes_touched)
      edit_jaccard: float (Jaccard of edit_cert)
      scope_overlap_count: int
    """
    # File-level
    files_a = set(cert_a.get("file_paths") or [cert_a.get("file_path", "")])
    files_b = set(cert_b.get("file_paths") or [cert_b.get("file_path", "")])
    file_match = bool(files_a & files_b)

    # Scope-level
    scopes_a = set(cert_a.get("scopes_touched", []))
    scopes_b = set(cert_b.get("scopes_touched", []))
    scope_overlap = scopes_a & scopes_b
    scope_union = scopes_a | scopes_b
    scope_jaccard = len(scope_overlap) / len(scope_union) if scope_union else 0.0

    # Edit-type level
    edits_a = set(cert_a.get("edit_cert", []))
    edits_b = set(cert_b.get("edit_cert", []))
    edit_union = edits_a | edits_b
    edit_jaccard = len(edits_a & edits_b) / len(edit_union) if edit_union else 0.0

    return {
        "file_match": file_match,
        "scope_jaccard": scope_jaccard,
        "edit_jaccard": edit_jaccard,
        "scope_overlap_count": len(scope_overlap),
    }
