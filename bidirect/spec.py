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


def describe(in_path: Path) -> str:
    n = 0
    field_counts: Counter = Counter()
    event_types: Counter = Counter()
    events_per: list[int] = []
    for r in _iter_records(in_path):
        n += 1
        field_counts.update(k for k, v in r.items() if v not in (None, [], {}, ""))
        evs = r.get("events", [])
        events_per.append(len(evs))
        event_types.update(e.get("type", "?") for e in evs)
    if n == 0:
        return f"{in_path}: empty"
    events_per.sort()
    lines = [f"{in_path}: {n} records, {in_path.stat().st_size:,} bytes",
             "field coverage (records with a non-empty value):"]
    lines += [f"  {k:<14} {c}/{n}" for k, c in field_counts.most_common()]
    lines.append(f"events per record: min {events_per[0]}, median {events_per[n // 2]}, max {events_per[-1]}")
    lines.append("event types: " + ", ".join(f"{t} {c}" for t, c in event_types.most_common()))
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
