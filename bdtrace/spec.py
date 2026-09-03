"""Trace-data readouts: the canonical record spec, a file audit, and head/slice views.

`spec()` states what a trace record is; `describe()` audits an actual JSONL
against it (counts, field coverage, event-type distribution) so a file's
shape and fullness are readable before anything consumes it; `head()` shows
the first records legibly and can write a slice out as a segmented export.
"""

import getpass
import json
import re
import socket
import subprocess
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


# One rule per identity class, applied in this order to every string. Order is
# load-bearing twice over: a credential is masked before any path, URL or email rule
# can chew it into a shape the credential patterns no longer recognise, and the email
# rule runs before the IP and hostname rules so `jdoe@10.0.0.5` and
# `jdoe@some-mac.local` are removed whole rather than leaving the user half behind.
# Every replacement is a fixed point (its output cannot re-match), so anonymizing
# twice equals anonymizing once — test_anonymize.py asserts that.
_ANON_RULES = [(re.compile(pattern), repl) for pattern, repl in (
    # credentials, by issuer prefix
    (r"\b(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"
     r"|github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}"
     r"|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{16,}"
     r"|npm_[A-Za-z0-9]{20,}|ya29\.[A-Za-z0-9_-]{20,})", "<token>"),
    (r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}", "<token>"),  # JWT
    (r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9_\-.=+/]{12,}", r"\1 <token>"),
    # credentials named by their variable/field, whatever their shape; the key, the
    # separator and any quoting are kept so a JSON or shell fragment stays parseable
    (r"(?i)\b([A-Za-z0-9_-]*(?:api[_-]?key|token|secret|password|passwd|access[_-]?key)"
     r"[\"']?\s*[=:]\s*[\"']?)[A-Za-z0-9_\-./+]{8,}", r"\1<redacted>"),
    # git remotes and URLs: the account is the identity, and userinfo may carry a token
    (r"\bgit@[\w.-]+:[A-Za-z0-9._-]+/", "git@<host>:<user>/"),
    (r"\b([a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1"),
    (r"\b((?:https?|ssh|git)://(?:github\.com|gitlab\.com|bitbucket\.org)/)[A-Za-z0-9._-]+",
     r"\1<user>"),
    (r"\b(https?://huggingface\.co/(?:datasets|spaces)/)[A-Za-z0-9._-]+", r"\1<user>"),
    (r"\b(https?://huggingface\.co/)(?!datasets/|spaces/)[A-Za-z0-9._-]+", r"\1<user>"),
    # forge CLIs name the owner with no URL around it: `gh api repos/<owner>/<repo>`,
    # `gh repo view <owner>/<repo>`, `gh pr list --repo <owner>/<repo>`. Measured on a
    # 138-trace corpus these were the only surviving occurrences of the real username.
    # `repos/<owner>` with or without a repo after it, and forge hosts written without
    # a scheme (`github.com/acme`), which the URL rules above never see
    (r"\b(repos/)(?!<)[A-Za-z0-9._-]+", r"\1<user>"),
    (r"\b((?:github|gitlab)\.com/)(?!<)[A-Za-z0-9._-]+", r"\1<user>"),
    (r"\b(gh (?:repo|pr|issue|release|run|workflow|api)\s+(?:\w+\s+)*?(?:--repo[= ])?)"
     r"[A-Za-z0-9._-]+/(?=[A-Za-z0-9._-]+)", r"\1<user>/"),
    (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "<email>"),
    # network and hardware addresses. A four-part version string ("9.10.2.21" in a
    # `uv add` line) is indistinguishable from a dotted quad and is masked too — the
    # cheaper error. The hex rule wants >=4 groups, so an ISO timestamp is not one.
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>"),
    (r"\b(?:[0-9a-fA-F]{1,4}:){3,}[0-9a-fA-F]{1,4}\b", "<ip>"),
    # home directories, in every spelling that reaches a trace
    (r"/Users/[A-Za-z0-9._-]+", "~"),
    (r"/home/[A-Za-z0-9._-]+", "~"),
    (r"(?<![\w/])/root\b", "~"),
    (r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+", "~"),
    (r"(?i)%2fusers%2f[A-Za-z0-9._-]+", "~"),
    # Claude Code workspace slugs: the cwd with its separators flattened to hyphens
    (r"-Users-[A-Za-z0-9_.]+", "-Users-anon"),
    (r"-home-[A-Za-z0-9_.]+", "-home-anon"),
    # per-user scratch space: the macOS temp token and the uid in a session temp dir
    (r"/var/folders/[A-Za-z0-9_+-]{2}/[A-Za-z0-9_+-]+", "/var/folders/<tmp>"),
    (r"/claude-\d+/", "/claude-<uid>/"),
    (r"/Volumes/[^/\s'\"]+", "/Volumes/<volume>"),
    # machine names: a hyphenated mDNS name, which `settings.local.json` is not
    (r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\.local\b(?![\w.])", "<host>"),
)]

# names too common to blank out safely if this machine happens to use one
_GENERIC_NAMES = {"localhost", "root", "user", "admin", "runner", "ubuntu"}


def _local_identifiers() -> tuple[str, ...]:
    """This machine's own names — login name, home-directory name, hostname. No
    pattern recognises a bare `jdoe` in `su jdoe` or a shell prompt, so they are
    matched literally. Longest first, so `jdoe-mbp` goes before the `jdoe` in it."""
    names = {Path.home().name, socket.gethostname()}
    names.add(socket.gethostname().partition(".")[0])
    try:
        names.add(getpass.getuser())
    except Exception:  # no passwd entry and no LOGNAME/USER/USERNAME set
        pass
    names |= _forge_handles()
    keep = (n for n in names if len(n) > 2 and n.lower() not in _GENERIC_NAMES)
    return tuple(sorted(keep, key=len, reverse=True))


def _forge_handles() -> set[str]:
    """Git and GitHub handles, which appear as bare words no pattern can see:
    `--author <handle>`, `R=<handle>`, prose. Read from local config only, so on
    someone else's machine it learns their handle, not the author's."""
    handles: set[str] = set()
    try:
        name = subprocess.run(["git", "config", "--get", "user.name"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        email = subprocess.run(["git", "config", "--get", "user.email"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return handles
    if name:
        handles.add(name)
        handles.add(name.replace(" ", ""))
        # a full name is also written in parts ("Hamidah, 2026-08-29"); over-scrubbing
        # a common word is the cheaper error here, and _GENERIC_NAMES filters those
        handles.update(name.split())
    if "@" in email:
        handles.add(email.split("@")[0])
    for remote in _git_remote_owners():
        handles.add(remote)
    return handles


def _git_remote_owners() -> set[str]:
    """Owner segment of every configured git remote: the account that owns the work."""
    try:
        out = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()
    return set(re.findall(r"(?:[:/])([A-Za-z0-9._-]+)/[A-Za-z0-9._-]+(?:\.git)?\s", out))


_LOCAL_NAMES = _local_identifiers()


def _anon_str(s: str, extra: tuple[str, ...] = ()) -> str:
    for pattern, repl in _ANON_RULES:
        s = pattern.sub(repl, s)
    # longest first, so a term containing another is consumed before its substring
    for name in sorted({*_LOCAL_NAMES, *extra}, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(name)}\b", "anon", s)
    return s


def anonymize(record: dict, extra: tuple[str, ...] = ()):
    """Strip identity from every string field, prompts and command lines included:
    the text survives, the identity doesn't.

    Closed classes (`_ANON_RULES` plus this machine's own names): home directories in
    POSIX, Windows, URL-encoded and workspace-slug spellings; per-user temp and volume
    paths; the local login name, home-directory name and hostname; mDNS machine names;
    email addresses; IPv4, IPv6 and MAC addresses; git remotes and forge URLs, whose
    account segment names a person; credentials, both by issuer prefix (Anthropic,
    OpenAI, GitHub, Slack, AWS, Google, Hugging Face, GitLab, npm) and by field name
    (`*token`, `*secret`, `password`, `*api_key`, `*access_key`), plus JWTs and
    Authorization headers.

    Not closed, and not claimed to be: personal names written in prose, other people's
    usernames where they appear bare rather than in a path or URL, and any secret that
    is neither issuer-prefixed nor next to a field name that says what it is. Pass those
    as `extra` (the CLI's `--redact`), and run `bdtrace trace audit` on the output to see
    what is left: a denylist only removes what it was told about, so the audit, not this
    function, is what makes sharing a decision on evidence. Dict keys are left alone —
    they are schema names, and rewriting them would break consumers.
    """
    if isinstance(record, str):
        return _anon_str(record, extra)
    if isinstance(record, dict):
        return {k: anonymize(v, extra) for k, v in record.items()}
    if isinstance(record, list):
        return [anonymize(v, extra) for v in record]
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
