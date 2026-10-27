"""Compile a trace into its typed entry and action skeleton.

The entry keeps only what a structured record proves: files from tool
inputs, verbs from tool names, commits and test runs from commands, span
from timestamps, asks from user text. Never from assistant prose. The
skeleton is the ordered sequence of canonical action labels, the part of a
session that carries behaviour (harness, pushback, next action) and the
least identity. Fields that measured as identity carriers (extensions,
error flags, full paths) are left out by construction; `--no-asks` drops
the task facet too, which is the procedure tier.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

VERSION = 1
MAX_ASKS = 5
ASK_CHARS = 120
MAX_FILES = 40

_INTERRUPT = re.compile(r"\[Request interrupted by user")
_COMMIT = re.compile(r"\bgit\s+commit\b")
_SUBAGENT_TOOLS = {"Task", "Agent", "TaskOutput"}
_WRITE_TOOLS = {"Write", "write_file", "NotebookEdit"}

LABELS = ("prompt", "interrupt", "read", "edit", "search", "run", "test", "subagent", "other")


def _label(event: dict) -> str:
    kind = event.get("type", "other")
    d = event.get("details") or {}
    if kind == "prompt":
        return "interrupt" if _INTERRUPT.search(str(d.get("text") or "")) else "prompt"
    if kind == "other" and d.get("tool") in _SUBAGENT_TOOLS:
        return "subagent"
    return kind if kind in LABELS else "other"


def _is_ask(text: str) -> bool:
    return bool(text) and not text.startswith("<") and not _INTERRUPT.search(text)


def compile_record(record: dict, asks: bool = True, files: bool = True) -> dict:
    """One typed entry per trace record. Pure function of structured fields."""
    events = record.get("events") or []
    stamps = [e["timestamp"] for e in events if e.get("timestamp")]
    ask_list: list[str] = []
    file_ops: dict[str, Counter] = {}
    tools: Counter = Counter()
    skeleton: list[str] = []
    commits = tests = interrupts = reads_before_edit = 0
    seen_edit = False
    for e in events:
        lab = _label(e)
        skeleton.append(lab)
        d = e.get("details") or {}
        if lab == "interrupt":
            interrupts += 1
            continue
        if lab == "prompt":
            text = " ".join(str(d.get("text") or "").split())
            if asks and len(ask_list) < MAX_ASKS and _is_ask(text):
                ask_list.append(text[:ASK_CHARS])
            continue
        tools[lab] += 1
        fp = d.get("file_path")
        if lab == "edit" and isinstance(fp, str) and fp:
            op = "WRITE" if d.get("tool") in _WRITE_TOOLS else "MODIFY"
            file_ops.setdefault(os.path.basename(fp), Counter())[op] += 1
            seen_edit = True
        elif lab == "read" and not seen_edit:
            reads_before_edit += 1
        if lab in ("run", "test"):
            cmd = str(d.get("command") or "")
            if _COMMIT.search(cmd):
                commits += 1
        if lab == "test":
            tests += 1
    cwd = record.get("cwd") or record.get("repo") or ""
    entry = {
        "version": VERSION,
        "instance_id": record.get("instance_id"),
        "agent": record.get("agent"),
        "project": Path(cwd).name if cwd else None,
        "started": min(stamps) if stamps else None,
        "ended": max(stamps) if stamps else None,
        "n_events": len(events),
        "asks": ask_list if asks else None,
        "files": ({b: dict(c) for b, c in sorted(file_ops.items())[:MAX_FILES]} if files else None),
        "files_total": len(file_ops),
        "tools": dict(tools),
        "commits": commits,
        "test_runs": tests,
        "interrupts": interrupts,
        "reads_before_first_edit": reads_before_edit,
        "distinct_verbs": len(tools),
        "skeleton": skeleton,
    }
    return entry


def render(entry: dict) -> str:
    """One-line map of a compiled entry, for eyes."""
    span = f"{(entry.get('started') or '?')[:16]} .. {(entry.get('ended') or '?')[:16]}"
    tools = " ".join(f"{k}×{v}" for k, v in sorted(entry["tools"].items(), key=lambda kv: -kv[1]))
    files = ", ".join(list(entry["files"] or {})[:6]) if entry.get("files") else "-"
    ask = (entry.get("asks") or ["-"])[0]
    return (f"{entry.get('instance_id', '?')} {entry.get('project') or ''}\n"
            f"  {span}  {entry['n_events']} events  {tools}\n"
            f"  commits {entry['commits']}  tests {entry['test_runs']}  interrupts {entry['interrupts']}\n"
            f"  files: {files}\n  ask: {ask}")


def compile_file(in_path: Path, out_path: Path | None, asks: bool, files: bool, show: int) -> dict:
    """Stream records in, entries out. Returns counts for the status line."""
    from bdtrace.spec import _iter_records
    n = 0
    skel_chars = 0
    out = open(out_path, "w") if out_path else None
    try:
        for rec in _iter_records(in_path):
            entry = compile_record(rec, asks=asks, files=files)
            n += 1
            skel_chars += sum(len(s) + 1 for s in entry["skeleton"])
            if out:
                out.write(json.dumps(entry) + "\n")
            if n <= show:
                print(render(entry))
    finally:
        if out:
            out.close()
    return {"records": n, "skeleton_chars": skel_chars, "out": str(out_path) if out_path else None}
