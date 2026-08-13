"""Per-scaffold canonicalizers (extended corpus).

Maps every scaffold's raw trajectory representation to the SAME 76-atom
canonical alphabet used by canonicalize.canonicalize_trajectory(), so all
agents share one BPE vocabulary downstream.

Format dispatch table — keyed off envelope["format"]:
    sweagent_traj_subdir       — same as legacy SWE-agent .traj content
    dars_traj_list             — list of {role, content, thought, action} dicts
    agentless_log_text         — markdown-structured log; ### headers are stages
    moatless_trajectory_json   — tree of nodes with action_steps; class names map to atoms

Scaffold-specific actions that don't fit the SWE-agent alphabet collapse
to OTHER (rather than UNKNOWN_*) — preserves the alphabet shape, accepts
small information loss in exchange for cross-scaffold comparability.

Usage:
    from analysis.preferences.canonicalize_extended import canonicalize_envelope
    envelope = json.load(open("output/trajectories/.cache/<sub>/<iid>.json"))
    atoms = canonicalize_envelope(envelope)
"""

from __future__ import annotations

import re
from typing import Any

from analysis.preferences.canonicalize import (
    canonicalize as canonicalize_swe_action,
    canonicalize_trajectory as canonicalize_swe_trajectory,
    file_type,
    _extract_path,
)


# ---------------------------------------------------------------------------
# 3a. SWE-agent (Claude 3.7 Thinking submission)
# ---------------------------------------------------------------------------

# Claude 3.7 uses Anthropic's text-editor tool (`str_replace_editor`) instead
# of the legacy SWE-agent open/edit verbs. Subcommands observed in the cache:
#     view          (n~249)  -> read-only file/dir inspection: NAV/OPEN
#     str_replace   (n~164)  -> in-place edit: EDIT
#     create        (n~126)  -> file creation: CREATE
#     insert        (rare)   -> in-place edit: EDIT
#     undo_edit     (rare)   -> revert: EDIT
_STR_REPLACE_EDITOR_RE = re.compile(
    r"^str_replace_editor\s+(\w+)(?:\s+(\S+))?", re.MULTILINE
)


def _rewrite_str_replace_editor(action: str) -> str | None:
    """Map a Claude-3.7 str_replace_editor action to the SWE-agent alphabet.

    Returns a canonical atom string, or None if not a str_replace_editor call.
    """
    m = _STR_REPLACE_EDITOR_RE.match(action.lstrip())
    if not m:
        return None
    sub = m.group(1).lower()
    arg = m.group(2) or ""
    path = _extract_path(arg) or arg
    ftype = file_type(path) if path else "UNKNOWN"
    if sub == "view":
        # Anthropic's view is a unified open+goto, treat as OPEN_<ftype>
        return f"OPEN_{ftype}"
    if sub == "create":
        return f"CREATE_{ftype}"
    if sub in ("str_replace", "insert", "undo_edit"):
        # Edit shape — file context here is the target path
        return f"EDIT_{ftype}" if ftype != "UNKNOWN" else "EDIT"
    # Unknown subcommand: collapse to OTHER rather than UNKNOWN_*
    return "OTHER"


def canonicalize_sweagent_traj_subdir(content: dict) -> list[str]:
    """Run the SWE-agent canonicalizer, then rewrite Claude-3.7 tool atoms."""
    trajectory = content.get("trajectory", []) if isinstance(content, dict) else []
    base = canonicalize_swe_trajectory(trajectory)
    out: list[str] = []
    for atom, step in zip(base, trajectory):
        if atom == "UNKNOWN_STR_REPLACE_EDITOR":
            rewritten = _rewrite_str_replace_editor(step.get("action") or "")
            out.append(rewritten or "OTHER")
        elif atom.startswith("UNKNOWN_"):
            # Non-tool unknowns (e.g. exit_cost variants) -> OTHER for shared alphabet
            out.append("OTHER")
        else:
            out.append(atom)
    return out


# ---------------------------------------------------------------------------
# 3b. DARS (action verbs are SWE-agent-shaped; reuse atom mapping)
# ---------------------------------------------------------------------------

def canonicalize_dars_traj_list(content: list) -> list[str]:
    """DARS message list — emit one atom per assistant action.

    DARS verbs largely overlap with SWE-agent (open / edit / search_file /
    search_dir / search_repo / find_file / goto / scroll_* / python /
    submit / create / append / rm / ls). Two non-SWE-agent verbs:

        search_repo  — semantically equivalent to search_dir, map to SEARCH
        append       — like edit (writes to current file), map to EDIT
    """
    if not isinstance(content, list):
        return []
    atoms: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "assistant":
            continue
        action = entry.get("action") or ""
        if not action or not action.strip():
            continue

        # search_repo / append are DARS-specific verbs; rewrite to SWE-agent
        # equivalents before canonicalizing
        first = action.strip().split(None, 1)[0].lower()
        if first == "search_repo":
            atoms.append("SEARCH")
            continue
        if first == "append":
            # append rewrites the open file; we don't have state here, so
            # default to EDIT atom (the SWE-agent canonicalizer collapses
            # bare edit to atom "EDIT", consistent with that behavior)
            atoms.append("EDIT")
            continue

        atom = canonicalize_swe_action(action)
        atoms.append(atom)
    return atoms


# ---------------------------------------------------------------------------
# 3c. Agentless (deterministic markdown pipeline)
# ---------------------------------------------------------------------------

# Top-level pipeline-stage headers. We recognize only headers that don't end
# with " ###" (those are nested prompt sub-sections, not pipeline stages)
# and aren't path-shaped (e.g. "astropy/modeling/separable.py" or
# "File: foo.py ###" — those are content artifacts inside a stage).
_AGENTLESS_HEADER_RE = re.compile(r"^### (.+)$", re.MULTILINE)

# Mapping from agentless stage label -> canonical atom. Lowercase compare.
_AGENTLESS_STAGE_MAP = {
    "localize to suspicious files":     "FIND_FILE",
    "model predicted suspicious files": "FIND_FILE",
    "model predicted irrelevant folders": "FIND_FILE",
    "embedding retrieval files":        "FIND_FILE",
    "localize to related elements":     "SEARCH",
    "localize to edit locations":       "NAV_SRC_PY",
    # Repair samples: Agentless emits N sampled patches per bug.
    # Each sample gets its own EDIT_SRC_PY atom — surfaces the parallel-sampling
    # signature in the canonical sequence.
    "repair sample 1":                  "EDIT_SRC_PY",
    "repair sample 2":                  "EDIT_SRC_PY",
    "repair sample 3":                  "EDIT_SRC_PY",
    "repair sample 4":                  "EDIT_SRC_PY",
    "regression test selection":        "RUN_PYTHON_TEST_PY",
    "reproduction test generation":     "RUN_PYTHON_REPRO_PY",
}


def canonicalize_agentless_log_text(content: str) -> list[str]:
    """Map Agentless ### pipeline-stage headers to canonical atoms.

    Allowlist-only: Agentless logs are full problem-statement plus pipeline
    headers. We only emit atoms for known pipeline stages (the keys of
    _AGENTLESS_STAGE_MAP); embedded GitHub bug descriptions, file paths,
    "Examples:", etc. are skipped.

    Each pipeline stage emits exactly one atom, in the order the stages
    appear in the log. Agentless is deterministic so all instances tend to
    have the same stages — that's the signature.
    """
    if not isinstance(content, str) or not content:
        return []
    atoms: list[str] = []
    for raw_header in _AGENTLESS_HEADER_RE.findall(content):
        header = raw_header.strip()
        # Skip nested prompt sub-headers like "GitHub Problem Description ###"
        if header.endswith("###"):
            continue
        h_lower = header.lower()
        atom = _AGENTLESS_STAGE_MAP.get(h_lower)
        if atom is not None:
            atoms.append(atom)
        # else: skip — it's a content artifact inside a stage (file path,
        # "Description", "Examples:", etc.), not a pipeline-stage header
    # Append a SUBMIT atom at the end — Agentless always submits a patch
    # (the patch is the artifact of the Repair stages); without an explicit
    # marker the trajectories would not show termination
    if atoms:
        atoms.append("SUBMIT")
    return atoms


# ---------------------------------------------------------------------------
# 3d. moatless (tree of action_steps; map action_args_class to atom)
# ---------------------------------------------------------------------------

# Class -> atom map, derived from inspection of 20 real moatless trajectories.
# Distribution observed:
#   StringReplaceArgs: 120  -> EDIT_SRC_PY (edits in code files)
#   SemanticSearchArgs: 36  -> SEARCH (semantic over the repo)
#   ViewCodeArgs: 22        -> OPEN_SRC_PY (read code into context)
#   FindClassArgs: 16       -> SEARCH (locate class)
#   FindFunctionArgs: 13    -> SEARCH
#   VerifiedFinishArgs: 13  -> SUBMIT
#   FindCodeSnippetArgs: 9  -> SEARCH
#   AppendStringArgs: 4     -> EDIT_SRC_PY (appends to code files)
_MOATLESS_CLASS_MAP = {
    "StringReplaceArgs":    "EDIT_SRC_PY",
    "AppendStringArgs":     "EDIT_SRC_PY",
    "CreateFileArgs":       "EDIT_SRC_PY",
    "ApplyChangeArgs":      "EDIT_SRC_PY",
    "SemanticSearchArgs":   "SEARCH",
    "FindClassArgs":        "SEARCH",
    "FindFunctionArgs":     "SEARCH",
    "FindCodeSnippetArgs":  "SEARCH",
    "ViewCodeArgs":         "OPEN_SRC_PY",
    "RequestMoreContextArgs": "OPEN_SRC_PY",
    "ListFilesArgs":        "FIND_FILE",
    "RunTestsArgs":         "RUN_PYTHON_TEST_PY",
    "VerifiedFinishArgs":   "SUBMIT",
    "FinishArgs":           "SUBMIT",
    "RejectArgs":           "SUBMIT",
}


def _walk_moatless_tree(node: Any, out: list[dict]) -> None:
    """Depth-first walk: collect all action_steps from a moatless tree."""
    if not isinstance(node, dict):
        return
    for step in node.get("action_steps") or []:
        out.append(step)
    for child in node.get("children") or []:
        _walk_moatless_tree(child, out)


def canonicalize_moatless_trajectory_json(content: dict) -> list[str]:
    """Walk the moatless tree, emit one atom per action_step."""
    if not isinstance(content, dict):
        return []
    # moatless stores actions in two places; some snapshots use the flat
    # "actions" list, most use the tree under "root". Try both.
    steps: list[dict] = []
    flat = content.get("actions") or []
    if flat:
        steps.extend(flat)
    root = content.get("root")
    if root:
        _walk_moatless_tree(root, steps)

    atoms: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action") or {}
        if not isinstance(action, dict):
            continue
        # Pull class name from action_args_class (full path) or fallback fields
        cls_full = action.get("action_args_class") or action.get("action_name") or action.get("name") or ""
        cls_short = cls_full.split(".")[-1] if cls_full else ""
        atom = _MOATLESS_CLASS_MAP.get(cls_short, "OTHER")
        atoms.append(atom)
    return atoms


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

_DISPATCH = {
    "sweagent_traj_subdir":     canonicalize_sweagent_traj_subdir,
    "dars_traj_list":           canonicalize_dars_traj_list,
    "agentless_log_text":       canonicalize_agentless_log_text,
    "moatless_trajectory_json": canonicalize_moatless_trajectory_json,
}


def canonicalize_envelope(envelope: dict) -> list[str]:
    """Dispatch on envelope["format"] and return canonical atom sequence.

    Legacy SWE-agent caches (the original 4 submissions) are NOT enveloped:
    they are raw {trajectory, history, info, environment} dicts. Detect that
    case and route to the SWE-agent handler.
    """
    if not isinstance(envelope, dict):
        return []
    # Un-enveloped legacy SWE-agent cache (raw .traj dict)
    if "format" not in envelope and "trajectory" in envelope:
        return canonicalize_sweagent_traj_subdir(envelope)
    fmt = envelope.get("format")
    content = envelope.get("content")
    handler = _DISPATCH.get(fmt)
    if handler is None:
        return []
    return handler(content)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke() -> None:
    import json
    from pathlib import Path

    cache = Path(__file__).resolve().parents[2] / "output" / "trajectories" / ".cache"

    samples = [
        ("20250226_sweagent_claude-3-7-sonnet-20250219", "astropy__astropy-12907"),
        ("20250205_dars_agent_claude_3.5_sonnet_deepseek_r1", "astropy__astropy-12907"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "astropy__astropy-12907"),
        ("20250111_moatless_deepseek_v3", "astropy__astropy-12907"),
    ]
    for submission, iid in samples:
        path = cache / submission / f"{iid}.json"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        env = json.loads(path.read_text())
        atoms = canonicalize_envelope(env)
        print(f"\n--- {submission} / {iid} ---")
        print(f"format={env['format']}  n_atoms={len(atoms)}")
        # show first 30 atoms
        snippet = atoms[:30]
        for a in snippet:
            print(f"  {a}")
        if len(atoms) > 30:
            print(f"  ... +{len(atoms)-30} more")


if __name__ == "__main__":
    _smoke()
