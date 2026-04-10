"""
Contextual edit operations: (direction, node_type, parent_type) triples.

Unlike the bag-of-edit-types in ast_edit_sequences.py, which parses hunk
fragments in isolation, this module:

  1. Parses the complete before and after file ASTs
  2. Maps diff line numbers to the containing AST nodes
  3. Retains only root-level changed nodes (no ancestor also changed)
  4. Records parent context: ADD_For@FunctionDef vs ADD_For@comprehension

This distinguishes structurally identical surface ops that serve different
strategic roles — the key limitation of the bag-of-edit-types abstraction.

Reference: Falleri et al. (2014) GumTree — we add parent context without
the full move/update action set.
"""

import ast
import re
from collections import defaultdict
from dataclasses import dataclass


# Node types with no semantic content — skip entirely
_SKIP_NODES = frozenset({
    "Module", "Load", "Store", "Del",
    "And", "Or", "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "FloorDiv",
    "Invert", "Not", "UAdd", "USub",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
    "expr_context", "boolop", "operator", "unaryop", "cmpop",
    "alias", "arg", "arguments", "keyword",
})

# Statement-level nodes — preferred over expression leaves when reporting changes.
# When a block is entirely new/deleted, report the topmost statement node,
# not the leaf expressions within it.
_STATEMENT_NODES = frozenset({
    "For", "While", "If", "Try", "With", "AsyncFor", "AsyncWith",
    "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Assign", "AugAssign", "AnnAssign",
    "Return", "Raise", "Delete", "Assert", "Pass", "Break", "Continue",
    "Expr", "Import", "ImportFrom", "Global", "Nonlocal",
    "ExceptHandler", "comprehension",
})

# Compact parent labels for common containers
_PARENT_ALIAS = {
    "FunctionDef": "FunctionDef",
    "AsyncFunctionDef": "FunctionDef",
    "ClassDef": "ClassDef",
    "Module": "Module",
    "If": "If",
    "For": "For",
    "While": "While",
    "Try": "Try",
    "With": "With",
    "comprehension": "comprehension",
    "ListComp": "comprehension",
    "SetComp": "comprehension",
    "DictComp": "comprehension",
    "GeneratorExp": "comprehension",
    "Lambda": "Lambda",
    "Assign": "Assign",
    "AnnAssign": "Assign",
    "AugAssign": "Assign",
    "Return": "Return",
    "Expr": "Expr",
}

_HUNK_HEADER_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


@dataclass(frozen=True)
class ContextualOp:
    direction: str   # "ADD" or "DEL"
    node_type: str   # AST node class name
    parent_type: str # parent AST node class name

    def __str__(self) -> str:
        return f"{self.direction}_{self.node_type}@{self.parent_type}"


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Map node id → parent node for the entire AST."""
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    return parent


def _nodes_in_lines(
    tree: ast.AST,
    changed_lines: set[int],
    parent_map: dict[int, ast.AST],
) -> list[tuple[ast.AST, ast.AST | None]]:
    """
    Return (node, parent) pairs representing the meaningful structural changes.

    A node is "the change" when the majority of its lines are in the changed
    set (coverage >= 0.5). A node is a "container" when only a small fraction
    of its lines changed. We report the deepest nodes that meet the coverage
    threshold — this finds the actual changed statements/expressions, not their
    enclosing function or class definitions.
    """
    if not changed_lines:
        return []

    # Compute coverage ratio for every node with line info
    coverage: dict[int, float] = {}
    node_by_id: dict[int, ast.AST] = {}

    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        if type(node).__name__ in _SKIP_NODES:
            continue
        node_lines = set(range(node.lineno, node.end_lineno + 1))
        if not node_lines:
            continue
        cov = len(node_lines & changed_lines) / len(node_lines)
        if cov > 0:
            coverage[id(node)] = cov
            node_by_id[id(node)] = node

    # Keep nodes with coverage >= 0.5 (majority of lines are changed)
    primary: set[int] = {nid for nid, cov in coverage.items() if cov >= 0.5}

    # Strategy: prefer statement-level nodes over expression leaves.
    #
    # For entirely new blocks (coverage=1.0 throughout), report the topmost
    # statement node — e.g. ADD_For@FunctionDef, not ADD_Name@For.
    # For partial changes (0.5 <= cov < 1.0), report the deepest primary node
    # since that's where the actual change occurred within existing code.

    # Separate fully-changed (cov=1.0) from partially-changed (0.5<=cov<1.0)
    full_ids = {nid for nid, cov in coverage.items() if cov >= 0.99}
    partial_ids = primary - full_ids

    result = []

    # From fully-changed nodes: report topmost statement nodes
    # (a statement node is "topmost" if its parent is not also full+statement)
    reported_full_ids: set[int] = set()
    for nid in full_ids:
        node = node_by_id[nid]
        if type(node).__name__ not in _STATEMENT_NODES:
            continue
        parent = parent_map.get(nid)
        parent_nid = id(parent) if parent is not None else None
        parent_is_full_stmt = (
            parent_nid in full_ids
            and type(parent).__name__ in _STATEMENT_NODES
        )
        if not parent_is_full_stmt:
            result.append((node, parent))
            reported_full_ids.add(nid)

    # From fully-changed non-statement nodes: report deepest ones that have
    # no fully-changed statement ancestor already reported.
    # This catches expression-level changes (e.g. a modified Call inside a comprehension).
    for nid in full_ids:
        node = node_by_id[nid]
        if type(node).__name__ in _STATEMENT_NODES or type(node).__name__ in _SKIP_NODES:
            continue
        # Check no ancestor is already in reported_full_ids
        has_stmt_ancestor = False
        cur = parent_map.get(nid)
        while cur is not None:
            if id(cur) in reported_full_ids:
                has_stmt_ancestor = True
                break
            cur = parent_map.get(id(cur))
        if has_stmt_ancestor:
            continue
        # Check this is deepest (no fully-changed non-skip child)
        has_full_child = any(
            id(child) in full_ids and type(child).__name__ not in _SKIP_NODES
            for child in ast.walk(node)
            if child is not node
        )
        if not has_full_child:
            result.append((node, parent_map.get(nid)))

    # From partially-changed nodes: report deepest primary nodes
    # (nodes whose children are not also primary)
    for nid in partial_ids:
        node = node_by_id[nid]
        if type(node).__name__ in _SKIP_NODES:
            continue
        has_primary_child = any(
            id(child) in primary
            for child in ast.walk(node)
            if child is not node
        )
        if not has_primary_child:
            parent = parent_map.get(nid)
            result.append((node, parent))

    return result


def _parse_diff_line_ranges(diff: str) -> dict[str, tuple[set[int], set[int]]]:
    """
    Parse a unified diff and return per-file (deleted_lines, added_lines)
    as sets of 1-based line numbers in the respective before/after files.
    """
    result: dict[str, tuple[set[int], set[int]]] = {}
    current_file = ""
    old_line = 0
    new_line = 0

    for line in diff.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r'b/(.+\.py)$', line)
            current_file = m.group(1) if m else ""
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            m = _HUNK_HEADER_RE.match(line)
            if m and current_file:
                old_line = int(m.group(1))
                new_line = int(m.group(3))
            continue
        if not current_file:
            continue

        if line.startswith("-"):
            deleted, added = result.setdefault(current_file, (set(), set()))
            deleted.add(old_line)
            old_line += 1
        elif line.startswith("+"):
            deleted, added = result.setdefault(current_file, (set(), set()))
            added.add(new_line)
            new_line += 1
        else:
            old_line += 1
            new_line += 1

    return result


def _parent_label(parent: ast.AST | None) -> str:
    if parent is None:
        return "Module"
    name = type(parent).__name__
    return _PARENT_ALIAS.get(name, name)


def contextual_ops_from_change(
    before_content: str,
    after_content: str,
    file_path: str,
    deleted_lines: set[int],
    added_lines: set[int],
) -> frozenset[str]:
    """
    Compute contextual edit ops for a single file change.
    Returns frozenset of strings like 'ADD_For@FunctionDef'.
    """
    ops: set[str] = set()

    # Deleted ops: from before AST
    if deleted_lines and before_content.strip():
        try:
            before_tree = ast.parse(before_content)
            parent_map = _build_parent_map(before_tree)
            for node, parent in _nodes_in_lines(before_tree, deleted_lines, parent_map):
                node_type = type(node).__name__
                if node_type in _SKIP_NODES:
                    continue
                ops.add(f"DEL_{node_type}@{_parent_label(parent)}")
        except SyntaxError:
            pass

    # Added ops: from after AST
    if added_lines and after_content.strip():
        try:
            after_tree = ast.parse(after_content)
            parent_map = _build_parent_map(after_tree)
            for node, parent in _nodes_in_lines(after_tree, added_lines, parent_map):
                node_type = type(node).__name__
                if node_type in _SKIP_NODES:
                    continue
                ops.add(f"ADD_{node_type}@{_parent_label(parent)}")
        except SyntaxError:
            pass

    return frozenset(ops)


def patch_to_contextual_ops(
    diff: str,
    file_contents: dict[str, tuple[str, str]],
) -> frozenset[str]:
    """
    Compute contextual edit ops for an entire patch.

    Args:
        diff: unified diff string (with diff --git headers)
        file_contents: {file_path: (before_content, after_content)}

    Returns:
        frozenset of strings like 'ADD_For@FunctionDef'
    """
    line_ranges = _parse_diff_line_ranges(diff)
    all_ops: set[str] = set()

    for file_path, (deleted_lines, added_lines) in line_ranges.items():
        if file_path not in file_contents:
            continue
        before, after = file_contents[file_path]
        ops = contextual_ops_from_change(
            before, after, file_path, deleted_lines, added_lines
        )
        all_ops.update(ops)

    return frozenset(all_ops)


def trace_to_contextual_ops(trace: dict) -> frozenset[str]:
    """
    Extract contextual edit ops from a resolved trace record.

    Collects all Python file changes across all code_change events,
    then computes contextual ops using the full before/after content.
    """
    import difflib

    # Collect per-file latest before/after content
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

        # Track cumulative before (first seen) and after (last seen)
        if fp not in file_changes:
            file_changes[fp] = (before, after)
        else:
            file_changes[fp] = (file_changes[fp][0], after)

        # Build diff for line range extraction
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        raw = "".join(difflib.unified_diff(
            before_lines, after_lines,
            fromfile=fp, tofile=fp,
        ))
        if raw:
            diff_parts.append(f"diff --git a/{fp} b/{fp}\n" + raw)

    if not diff_parts or not file_changes:
        return frozenset()

    combined_diff = "\n".join(diff_parts)
    return patch_to_contextual_ops(combined_diff, file_changes)
