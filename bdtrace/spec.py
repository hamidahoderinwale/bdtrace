"""Trace-data readouts: the canonical record spec, a file audit, and head/slice views.

`spec()` states what a trace record is; `describe()` audits an actual JSONL
against it (counts, field coverage, event-type distribution) so a file's
shape and fullness are readable before anything consumes it; `head()` shows
the first records legibly and can write a slice out as a segmented export.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# canonical trace record (the resolved_traces_lite_full.jsonl shape)
SPEC = {
    "record": "one trace: an agent's work on one instance/session",
    "fields": {
        "instance_id": "str — unique id (benchmark instance, or '<harness>-<session id>' for imports)",
        "repo": "str|null — repository worked on",
        "base_commit": "str|null — commit the work started from",
        "events": "list — ordered actions; each {type: prompt|edit|read|search|run|test|other, details: dict}",
        "prompts": "list — prompt events, extracted for convenience",
        "reprs": "dict — added by `bdtrace transform`: {transform name: representation}",
    },
    "interchange": "JSONL, one record per line; compressed serializations via `bdtrace trace export`",
}


def _iter_records(in_path: Path):
    if str(in_path) == "-":
        yield from (json.loads(line) for line in sys.stdin if line.strip())
        return
    with open(in_path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def spec_text() -> str:
    return json.dumps(SPEC, indent=2)


def _dist(values: list[int]) -> dict:
    values = sorted(values)
    n = len(values)
    return {"min": values[0], "median": values[n // 2], "max": values[-1], "total": sum(values)} if n else {}


def summarize(in_path: Path) -> dict:
    """One structured summary, shared verbatim by `trace spec --in` and export
    sidecars. Each block answers a decision: size (is it enough), events.by_type
    (what kind of work), prompts/text_chars (embedding and LLM-pass cost),
    time (longitudinal coverage), sources and workspaces (what corpus this is)."""
    n = 0
    field_counts: Counter = Counter()
    event_types: Counter = Counter()
    sources: Counter = Counter()
    workspaces: Counter = Counter()
    events_per: list[int] = []
    prompts_per: list[int] = []
    text_chars: list[int] = []
    earliest, latest = None, None
    for r in _iter_records(in_path):
        n += 1
        field_counts.update(k for k, v in r.items() if v not in (None, [], {}, ""))
        evs = r.get("events", [])
        events_per.append(len(evs))
        event_types.update(e.get("type", "?") for e in evs)
        prompts_per.append(len(r.get("prompts", [])))
        text_chars.append(sum(len(str(v)) for e in evs for v in e.get("details", {}).values()))
        sources[str(r.get("instance_id", "")).split("-")[0] or "?"] += 1
        workspaces[r.get("cwd") or r.get("repo") or "?"] += 1
        stamps = [e["timestamp"] for e in evs if e.get("timestamp")]
        if stamps:
            earliest = min(earliest or min(stamps), min(stamps))
            latest = max(latest or max(stamps), max(stamps))
    if n == 0:
        return {"n_records": 0}
    return {
        "n_records": n,
        "bytes": in_path.stat().st_size if str(in_path) != "-" else None,
        "field_coverage": dict(field_counts.most_common()),
        "events": {"by_type": dict(event_types.most_common()), "per_record": _dist(events_per)},
        "prompts": {"per_record": _dist(prompts_per)},
        "text_chars_per_record": _dist(text_chars),
        "time": {"earliest": earliest, "latest": latest},
        "sources": dict(sources.most_common()),
        "top_workspaces": dict(workspaces.most_common(8)),
    }


def describe(in_path: Path) -> str:
    s = summarize(in_path)
    n = s["n_records"]
    if n == 0:
        return f"{in_path}: empty"
    lines = [f"{in_path}: {n} records" + (f", {s['bytes']:,} bytes" if s.get("bytes") else ""),
             "field coverage (records with a non-empty value):"]
    lines += [f"  {k:<14} {c}/{n}" for k, c in s["field_coverage"].items()]
    ep = s["events"]["per_record"]
    lines.append(f"events per record: min {ep['min']}, median {ep['median']}, max {ep['max']} (total {ep['total']})")
    lines.append("event types: " + ", ".join(f"{t} {c}" for t, c in s["events"]["by_type"].items()))
    lines.append(f"prompts: {s['prompts']['per_record'].get('total', 0)} total, "
                 f"median {s['prompts']['per_record'].get('median', 0)}/record")
    tc = s["text_chars_per_record"]
    lines.append(f"detail text per record: median {tc['median']:,} chars (embedding/LLM cost proxy)")
    if s["time"]["earliest"]:
        lines.append(f"time span: {s['time']['earliest'][:19]} .. {s['time']['latest'][:19]}")
    lines.append("sources: " + ", ".join(f"{k} {v}" for k, v in s["sources"].items()))
    return "\n".join(lines)


# the event-type taxonomy (what tokens_repr sequences); "tools" = everything but prompt
EVENT_TYPES = ("prompt", "edit", "read", "search", "run", "test", "other")
_COMPACT_DETAIL_KEYS = ("tool", "command", "file_path", "description")


def parse_types(spec_str: str) -> set[str]:
    names = set(EVENT_TYPES) - {"prompt"} if spec_str == "tools" else set(spec_str.split(","))
    bad = names - set(EVENT_TYPES) - {"code_change"}  # legacy resolved-trace type allowed
    if bad:
        sys.exit(f"bdtrace: unknown event types {sorted(bad)}; taxonomy: {', '.join(EVENT_TYPES)} (or 'tools')")
    return names


def project(record: dict, types: set[str] | None = None, compact: bool = False) -> dict:
    """Project along the two orthogonal axes: WHICH event types survive, and how
    much detail each keeps (compact = action surface only, text bodies dropped)."""
    events = record.get("events", [])
    if types is not None:
        events = [e for e in events if e.get("type") in types]
    if compact:
        events = [{"type": e.get("type"), "timestamp": e.get("timestamp"),
                   "details": {k: e.get("details", {}).get(k) for k in _COMPACT_DETAIL_KEYS
                               if e.get("details", {}).get(k) is not None}}
                  for e in events]
    out = {**record, "events": events}
    if types is not None and "prompt" not in types:
        out["prompts"] = []
    return out


def _anon_str(s: str) -> str:
    import re
    s = re.sub(r"/Users/[^/\s'\"]+", "~", s)
    s = re.sub(r"/home/[^/\s'\"]+", "~", s)
    s = re.sub(r"-Users-[A-Za-z0-9_.]+", "-Users-anon", s)  # Claude Code workspace slugs
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<email>", s)


def anonymize(record: dict):
    """Strip identifying strings (home directories, usernames in paths, emails)
    from every string field, prompts included: the text survives, the identity doesn't."""
    if isinstance(record, str):
        return _anon_str(record)
    if isinstance(record, dict):
        return {k: anonymize(v) for k, v in record.items()}
    if isinstance(record, list):
        return [anonymize(v) for v in record]
    return record


def _truncate(v, cap: int = 240):
    if isinstance(v, str) and len(v) > cap:
        return v[:cap] + f"... [{len(v)} chars]"
    if isinstance(v, dict):
        return {k: _truncate(x, cap) for k, x in v.items()}
    if isinstance(v, list):
        return [_truncate(x, cap) for x in v[:8]] + ([f"... [{len(v)} items]"] if len(v) > 8 else [])
    return v


def interval_bounds(interval: str | None) -> tuple[str | None, str | None]:
    """'A..B' (either side may be empty) -> ISO bounds; '7d'/'24h'/'2w' -> the last N units."""
    if not interval:
        return None, None
    if ".." in interval:
        since, _, until = interval.partition("..")
        return since or None, until or None
    from datetime import datetime, timedelta, timezone
    unit = {"h": "hours", "d": "days", "w": "weeks"}.get(interval[-1])
    if unit is None or not interval[:-1].isdigit():
        sys.exit(f"bdtrace: bad interval `{interval}` (use A..B ISO bounds, or 7d / 24h / 2w)")
    since_dt = datetime.now(timezone.utc) - timedelta(**{unit: int(interval[:-1])})
    return since_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), None


def _in_window(r: dict, since: str | None, until: str | None) -> bool:
    """True when any event timestamp falls in [since, until). ISO strings compare
    lexicographically, so date-only bounds work; records without timestamps are
    excluded whenever a bound is set."""
    if not since and not until:
        return True
    stamps = [e.get("timestamp") for e in r.get("events", []) if e.get("timestamp")]
    return any((not since or s >= since) and (not until or s < until) for s in stamps)


def head(in_path: Path, n: int, skip: int, max_events: int, out: Path | None,
         since: str | None = None, until: str | None = None) -> None:
    shown = 0
    matched = 0
    out_f = open(out, "w") if out else None
    for r in _iter_records(in_path):
        if not _in_window(r, since, until):
            continue
        i, matched = matched, matched + 1
        if i < skip:
            continue
        if shown >= n:
            break
        shown += 1
        if out_f:
            out_f.write(json.dumps(r, default=str) + "\n")
        else:
            view = dict(r)
            evs = view.get("events", [])
            view["events"] = evs[:max_events] + ([f"... [{len(evs)} events]"] if len(evs) > max_events else [])
            print(json.dumps(_truncate(view), indent=2, default=str))
    if out_f:
        out_f.close()
        print(f"{shown} records (skip {skip}) -> {out}", file=sys.stderr)
    elif shown == 0:
        sys.exit(f"bdtrace: no records at offset {skip} in {in_path}")
