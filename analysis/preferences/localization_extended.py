"""Per-format gold-file localization for the extended corpus.

Determines, for each cached trajectory, the step at which the agent first
accessed the gold-patch file. Each scaffold's trajectory representation
needs its own file-extraction logic:

  sweagent_traj_subdir       — same as legacy SWE-agent: scan state.open_file
                                and action regex. Claude-3.7's str_replace_editor
                                also references file paths in its action arg.
  dars_traj_list             — assistant.action strings; regex over content
  agentless_log_text         — section-level only; localization not applicable
                                (returns "n/a")
  moatless_trajectory_json   — walk action_steps tree; action.path / files /
                                file_pattern carry the target file
  un-enveloped legacy SWE    — fallback: state.open_file + action regex

Returns:
    None if no localization observed for known formats, an int (0-indexed
    step) if localized, or "n/a" for Agentless (not applicable).

Per-format step counts intentionally use the same units as the canonical
sequence emitted by canonicalize_envelope, so steps_after = len(canonical) -
loc_step is comparable across scaffolds.

Usage:
    from analysis.preferences.localization_extended import (
        first_localization_step_extended,
    )
    step = first_localization_step_extended(envelope, gold_files)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional, Union

# A loose match for .py file paths anywhere in a free-text action.
_FILE_RE = re.compile(r"([a-zA-Z0-9_./\-]+\.py)")


def _matches_gold(file_str: str, gold: set[str]) -> bool:
    if not file_str:
        return False
    file_str = file_str.lstrip("/")
    if file_str in gold:
        return True
    # Match by suffix: gold paths are repo-relative; observed paths often
    # have a leading repo-name component or absolute path.
    for g in gold:
        if file_str.endswith("/" + g) or file_str == g:
            return True
        # Reverse: observed is shortened form
        if g.endswith("/" + file_str):
            return True
    return False


def _extract_files_from_text(text: str) -> set[str]:
    if not text:
        return set()
    return set(_FILE_RE.findall(text))


# ---------------------------------------------------------------------------
# Legacy + Claude-3.7 SWE-agent: scan state.open_file, then action regex
# ---------------------------------------------------------------------------

def _sweagent_files_at_step(step: dict) -> set[str]:
    files: set[str] = set()
    state = step.get("state")
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except Exception:
            state = {}
    if isinstance(state, dict):
        of = state.get("open_file") or ""
        if of and of != "n/a" and of.endswith(".py"):
            parts = Path(of).parts
            for i, p in enumerate(parts):
                if "__" in p:
                    rel = "/".join(parts[i + 1:])
                    if rel:
                        files.add(rel)
                    break
            else:
                files.add(of.lstrip("/"))
    files |= _extract_files_from_text(step.get("action", "") or "")
    return files


def _localize_sweagent(envelope: dict, gold: set[str]) -> Optional[int]:
    content = envelope.get("content")
    # Two cases: enveloped (content has trajectory) or raw legacy dict
    if isinstance(content, dict) and "trajectory" in content:
        traj = content.get("trajectory", [])
    elif isinstance(envelope, dict) and "trajectory" in envelope:
        traj = envelope.get("trajectory", [])
    else:
        return None
    for i, step in enumerate(traj):
        if any(_matches_gold(f, gold) for f in _sweagent_files_at_step(step)):
            return i
    return None


# ---------------------------------------------------------------------------
# DARS: regex over assistant.action content; step index = ith assistant atom
# (matches what canonicalize_dars_traj_list emits)
# ---------------------------------------------------------------------------

def _localize_dars(envelope: dict, gold: set[str]) -> Optional[int]:
    content = envelope.get("content") or []
    if not isinstance(content, list):
        return None
    atom_i = 0
    for entry in content:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "assistant":
            continue
        action = entry.get("action") or ""
        if not action.strip():
            continue
        files = _extract_files_from_text(action)
        if any(_matches_gold(f, gold) for f in files):
            return atom_i
        atom_i += 1
    return None


# ---------------------------------------------------------------------------
# Moatless: walk action_steps tree; per atom emitted, check args.path / files
# ---------------------------------------------------------------------------

def _moatless_step_files(action: dict) -> set[str]:
    files: set[str] = set()
    for k in ("path", "file_pattern"):
        v = action.get(k)
        if isinstance(v, str) and ".py" in v:
            files.add(v.lstrip("/"))
    f_list = action.get("files")
    if isinstance(f_list, list):
        for v in f_list:
            if isinstance(v, str):
                files.add(v.lstrip("/"))
            elif isinstance(v, dict):
                fp = v.get("file_path") or v.get("path")
                if isinstance(fp, str):
                    files.add(fp.lstrip("/"))
    # Also scan thoughts and snippets for .py mentions
    for k in ("thoughts", "code_snippet", "old_str", "new_str", "query"):
        v = action.get(k)
        if isinstance(v, str):
            files |= _extract_files_from_text(v)
    return files


def _localize_moatless(envelope: dict, gold: set[str]) -> Optional[int]:
    content = envelope.get("content") or {}
    if not isinstance(content, dict):
        return None
    steps: list[dict] = []
    flat = content.get("actions") or []
    if flat:
        steps.extend(flat)

    def _walk(node):
        if not isinstance(node, dict):
            return
        for s in node.get("action_steps") or []:
            steps.append(s)
        for ch in node.get("children") or []:
            _walk(ch)

    _walk(content.get("root") or {})

    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        action = s.get("action") or {}
        if not isinstance(action, dict):
            continue
        files = _moatless_step_files(action)
        if any(_matches_gold(f, gold) for f in files):
            return i
    return None


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def first_localization_step_extended(
    envelope: dict, gold: set[str]
) -> Union[int, str, None]:
    """Return the canonical-sequence step index at which gold was first hit,
    or None if never reached, or 'n/a' if localization is not applicable
    (Agentless: section-level only)."""
    if not isinstance(envelope, dict):
        return None

    # Un-enveloped legacy SWE-agent cache
    if "format" not in envelope and "trajectory" in envelope:
        return _localize_sweagent({"content": envelope}, gold)

    fmt = envelope.get("format")
    if fmt == "agentless_log_text":
        return "n/a"
    if fmt == "sweagent_traj_subdir":
        return _localize_sweagent(envelope, gold)
    if fmt == "dars_traj_list":
        return _localize_dars(envelope, gold)
    if fmt == "moatless_trajectory_json":
        return _localize_moatless(envelope, gold)
    return None


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def _smoke() -> None:
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from analysis.preferences.localization import load_gold_files

    gold_files = load_gold_files()
    cache = ROOT / "output" / "trajectories" / ".cache"
    samples = [
        ("20250226_sweagent_claude-3-7-sonnet-20250219", "astropy__astropy-12907"),
        ("20250205_dars_agent_claude_3.5_sonnet_deepseek_r1", "astropy__astropy-12907"),
        ("20241202_agentless-1.5_claude-3.5-sonnet-20241022", "astropy__astropy-12907"),
        ("20250111_moatless_deepseek_v3", "astropy__astropy-12907"),
        ("20240620_sweagent_claude3.5sonnet", "astropy__astropy-12907"),
    ]
    for sub, iid in samples:
        path = cache / sub / f"{iid}.json"
        if not path.exists():
            print(f"[skip] {path}")
            continue
        env = json.loads(path.read_text())
        gold = gold_files.get(iid, set())
        step = first_localization_step_extended(env, gold)
        print(f"  {sub:55s} {iid}  loc_step={step}  gold={list(gold)[:2]}")


if __name__ == "__main__":
    _smoke()
