"""Prompt-anchored transcript view: what was asked, and what happened next.

A trace is a flat event list, which is the right storage shape and the wrong
reading shape. Sessions are actually a sequence of turns: one human prompt and
the run of agent actions it caused. This renders that, collapsing consecutive
same-type actions into one line, so a 400-event session reads as a page.

The same grouping is the analytic unit — a turn is what an instruction bought —
so `turns()` returns it as data and `render()` is only the presentation.
"""

import json
from pathlib import Path

_GLYPH = {"prompt": "▸", "edit": "✎", "read": "▪", "search": "⌕",
          "run": "$", "test": "✓", "other": "·", "code_change": "✎"}
_DETAIL_KEYS = ("command", "file_path", "text", "description", "tool", "content")


def _label(event: dict) -> str:
    d = event.get("details") or {}
    for key in _DETAIL_KEYS:
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())
    return ""


def turns(record: dict) -> list[dict]:
    """Split a trace into {prompt, events} turns. Events before the first prompt
    form a leading turn with prompt None, so nothing is dropped."""
    out: list[dict] = [{"prompt": None, "events": []}]
    for event in record.get("events", []):
        if event.get("type") == "prompt":
            out.append({"prompt": event, "events": []})
        else:
            out[-1]["events"].append(event)
    return [t for t in out if t["prompt"] or t["events"]]


def _runs(events: list[dict]) -> list[tuple[str, list[dict]]]:
    """Consecutive same-type events as one run: the abstraction that makes a
    400-event session legible without dropping what actually happened."""
    runs: list[tuple[str, list[dict]]] = []
    for event in events:
        kind = event.get("type", "other")
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(event)
        else:
            runs.append((kind, [event]))
    return runs


def render(record: dict, width: int = 88, max_examples: int = 3) -> str:
    header = " ".join(str(x) for x in (
        record.get("instance_id", "?"), record.get("agent") or "", record.get("cwd") or record.get("repo") or "",
    ) if x)
    stamps = [e["timestamp"] for e in record.get("events", []) if e.get("timestamp")]
    n_events = len(record.get("events", []))
    lines = [header, f"{n_events} events" + (f"  {stamps[0][:19]} .. {stamps[-1][:19]}" if stamps else "")]
    for turn in turns(record):
        if turn["prompt"]:
            text = _label(turn["prompt"]) or "(empty prompt)"
            lines.append("")
            lines.append(f"{_GLYPH['prompt']} {_clip(text, width - 2)}")
        for kind, events in _runs(turn["events"]):
            glyph = _GLYPH.get(kind, "·")
            count = f"×{len(events)}" if len(events) > 1 else "  "
            shown = [lbl for lbl in (_label(e) for e in events[:max_examples]) if lbl]
            detail = ", ".join(_clip(s, 40) for s in shown)
            if len(events) > max_examples:
                detail += f", +{len(events) - max_examples} more"
            lines.append(f"    {glyph} {kind:<6}{count}  {_clip(detail, width - 20)}")
    return "\n".join(lines)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def show(in_path: Path, instance_id: str | None, limit: int, width: int) -> None:
    from bdtrace.spec import _iter_records

    shown = 0
    for record in _iter_records(in_path):
        if instance_id and record.get("instance_id") != instance_id:
            continue
        if shown:
            print("\n" + "─" * width)
        print(render(record, width=width))
        shown += 1
        if shown >= limit:
            break
    if not shown:
        raise SystemExit(f"bdtrace: no trace matched {instance_id!r}" if instance_id else "bdtrace: no records")


def turns_summary(in_path: Path) -> dict:
    """Turn-level shape of a corpus: how much work an instruction actually buys."""
    from collections import Counter

    from bdtrace.spec import _iter_records

    per_turn: list[int] = []
    kinds: Counter = Counter()
    n_traces = 0
    for record in _iter_records(in_path):
        n_traces += 1
        for turn in turns(record):
            if turn["prompt"]:
                per_turn.append(len(turn["events"]))
                kinds.update(e.get("type", "other") for e in turn["events"])
    per_turn.sort()
    n = len(per_turn)
    return {
        "traces": n_traces,
        "turns": n,
        "actions_per_turn": {"min": per_turn[0], "median": per_turn[n // 2], "max": per_turn[-1]} if n else {},
        "action_mix": dict(kinds.most_common()),
    }


def _cli_json(obj) -> str:
    return json.dumps(obj, indent=2)
