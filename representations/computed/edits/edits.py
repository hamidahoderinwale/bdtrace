"""
Edits: computed, intra-function, AST delta.

Input: function source (before/after).
Grounding: AST delta. Unit: AST node operation.
Distance: distance.tree_edit_distance (zss).
"""

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ...core.intent import extract_emergent_intent, extract_event_intent, intent_tokens_for_event
from ...core.utils import extract_function_names_from_code


def _nodes_in(tree: ast.AST, source: str = "") -> set[tuple[str, int, str]]:
    """Extract (node_type, lineno, snippet) for each node."""
    result = set()
    for node in ast.walk(tree):
        node_type = type(node).__name__
        lineno = getattr(node, "lineno", 0) or 0
        snippet = ""
        if source:
            try:
                snippet = (ast.get_source_segment(source, node) or "")[:80].replace("\n", " ")
            except (ValueError, TypeError):
                pass
        result.add((node_type, lineno, snippet))
    return result


def _classify_operation(
    added: set[tuple[str, int, str]],
    removed: set[tuple[str, int, str]],
) -> list[dict[str, Any]]:
    """Map AST node changes to semantic operation types."""
    operations = []
    added_by_type = {}
    for nt, ln, _ in added:
        added_by_type.setdefault(nt, []).append({"location": f"line {ln}", "node_type": nt})
    removed_by_type = {}
    for nt, ln, _ in removed:
        removed_by_type.setdefault(nt, []).append({"location": f"line {ln}", "node_type": nt})

    for node_type, nodes in added_by_type.items():
        for n in nodes:
            if node_type == "If":
                operations.append({"type": "guard_clause_added", "location": n["location"], "node_type": node_type})
            elif node_type == "Call":
                operations.append({"type": "call_added", "location": n["location"], "node_type": node_type})
            elif node_type in ("FunctionDef", "AsyncFunctionDef"):
                operations.append({"type": "function_added", "location": n["location"], "node_type": node_type})
            elif node_type == "Return":
                operations.append({"type": "return_added", "location": n["location"], "node_type": node_type})
            else:
                operations.append(
                    {"type": f"{node_type.lower()}_added", "location": n["location"], "node_type": node_type}
                )

    for node_type, nodes in removed_by_type.items():
        for n in nodes:
            operations.append(
                {"type": f"{node_type.lower()}_removed", "location": n["location"], "node_type": node_type}
            )

    added_types = {nt for nt, _, _ in added}
    removed_types = {nt for nt, _, _ in removed}
    for nt in added_types & removed_types:
        operations.append(
            {"type": f"{nt.lower()}_modified", "location": "multiple", "node_type": nt, "before": nt, "after": nt}
        )
    return operations


def semantic_edits_repr(
    before_source: str,
    after_source: str,
    file_path: str | None = None,
) -> dict[str, Any]:
    """
    Structural edit representation from before/after source.
    Returns {operations, ast_before, ast_after, delta}.
    """
    if not before_source and not after_source:
        return {"operations": [], "ast_before": None, "ast_after": None, "delta": 0}

    tree_before = None
    tree_after = None
    if before_source:
        try:
            tree_before = ast.parse(before_source)
        except SyntaxError:
            pass
    if after_source:
        try:
            tree_after = ast.parse(after_source)
        except SyntaxError:
            pass

    if tree_before is None and tree_after is None:
        return {"operations": [], "ast_before": None, "ast_after": None, "delta": 0}

    nodes_before = _nodes_in(tree_before, before_source) if tree_before else set()
    nodes_after = _nodes_in(tree_after, after_source) if tree_after else set()
    added = nodes_after - nodes_before
    removed = nodes_before - nodes_after
    operations = _classify_operation(added, removed)
    delta = len(added) + len(removed)

    return {
        "operations": operations,
        "ast_before": tree_before,
        "ast_after": tree_after,
        "delta": delta,
    }


def semantic_edits_repr_trace(
    trace: dict,
    include_prompts: bool = True,
    include_intent: bool = True,
    use_emergent: bool = True,
    return_structural: bool = False,
) -> list[str] | list[dict[str, Any]]:
    """Extract edits from trace. Returns list[str] or list of structural dicts."""
    if not trace or not isinstance(trace, dict):
        return []

    events = trace.get("events", [])
    if not events:
        return []

    edits = []
    structural_results = []
    max_segment_size = 10

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        try:
            event_type = (event.get("type") or "").lower()
            details = event.get("details", {})
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}
            if not isinstance(details, dict):
                details = {}

            file_path = details.get("file_path") or details.get("file")
            before_content = details.get("before_content", "")
            after_content = details.get("after_content", "")
            lines_added = details.get("lines_added", 0) or 0
            lines_removed = details.get("lines_removed", 0) or 0
            diff_summary = details.get("diff_summary", "")

            if event_type in ("code_change", "file_change", "entry_created") and (before_content or after_content):
                struct = semantic_edits_repr(before_content, after_content, file_path)
                if return_structural:
                    structural_results.append(struct)

                op = "MODIFY"
                if lines_added > 0 and lines_removed == 0:
                    op = "ADD"
                elif lines_removed > 0 and lines_added == 0:
                    op = "REMOVE"

                target_base = Path(file_path).stem if file_path else "unknown"
                if before_content or after_content:
                    code = after_content or before_content
                    funcs = extract_function_names_from_code(code, file_path)
                    target = f"{target_base}::{funcs[0]}" if funcs else target_base
                else:
                    target = target_base

                size = "SMALL"
                if lines_added > 50 or lines_removed > 50:
                    size = "LARGE"
                elif lines_added > 10 or lines_removed > 10:
                    size = "MEDIUM"

                edit_str = f"{op}->{target}"
                if size != "SMALL":
                    edit_str += f"->{size}"
                if diff_summary:
                    words = diff_summary.split()[:2]
                    if words:
                        edit_str += f"->{'_'.join(words)}"

                if include_intent:
                    intents = intent_tokens_for_event(event, include_llm=False, use_emergent=use_emergent)
                    if intents:
                        edit_str += f"->{intents[0]}"
                        edits.append(edit_str)
                        for x in intents[1:]:
                            edits.append(f"INTENT->{x}")
                    else:
                        edits.append(edit_str)
                else:
                    edits.append(edit_str)

            elif event_type in ("entry_created", "file_created"):
                op = "CREATE"
                target = file_path or event.get("target", "unknown")
                if include_intent:
                    intents = intent_tokens_for_event(event, include_llm=False, use_emergent=use_emergent)
                    edits.append(f"{op}->{target}->{intents[0]}" if intents else f"{op}->{target}")
                else:
                    edits.append(f"{op}->{target}")

            elif event_type in ("entry_deleted", "file_deleted"):
                edits.append(f"DELETE->{file_path or event.get('target', 'unknown')}")

            elif (event.get("operation") or event.get("type")) and not file_path:
                op = event.get("operation") or event.get("type")
                if include_intent:
                    intents = intent_tokens_for_event(event, include_llm=False, use_emergent=use_emergent)
                    edits.append(f"{op}->{intents[0]}" if intents else str(op))
                else:
                    edits.append(str(op))

            if include_intent and (i + 1) % max_segment_size == 0:
                segment = events[max(0, i - max_segment_size + 1) : i + 1]
                counts = Counter()
                for seg in segment:
                    counts.update(
                        extract_emergent_intent(seg, use_llm=False) if use_emergent else extract_event_intent(seg)
                    )
                for intent, _ in counts.most_common(3):
                    edits.append(f"SEGMENT_INTENT->{intent}")

        except (KeyError, TypeError, ValueError, SyntaxError, ImportError):
            continue

    if return_structural:
        return structural_results
    return edits


def semantic_edits_repr_str(
    trace: dict,
    limit: int = 50,
    include_intent: bool = True,
    use_emergent: bool = True,
) -> str:
    """Extract edits as a string."""
    edits = semantic_edits_repr_trace(trace, include_intent=include_intent, use_emergent=use_emergent)
    if not edits:
        return "EMPTY_TRACE"
    s = " → ".join(edits[:limit])
    if len(edits) > limit:
        s += f" ... [truncated from {len(edits)} edits]"
    return s
