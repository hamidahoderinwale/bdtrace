"""Privacy-preserving rewrites of trace records: one operator per field type.

Each operator takes one record and returns a new record with a single field type
changed — `details.file_path`, `details.command`, prompt text, timestamps, event
order, or the details block as a whole — and everything else as it was: a record
without that field comes back equal to what went in, a missing or oddly typed field
is skipped rather than raised on, and the input record is never mutated. Operators
are grouped by the standard privacy families (pseudonymization, generalization,
suppression, perturbation, order) and compose by chaining `bdtrace transform` calls;
there is no built-in ladder, because the right order depends on what the release is
for. `min_users` is the one corpus-level operator: it is built from a pass over every
record and then applied record by record like the rest.

Prompt text lives in three places on a record — `details.text` of a prompt event,
its Claude Code duplicate `details.content`, and the record's `prompts` list — so the
prompt-text operator rewrites all three; leaving one would carry the text out under
a help string that says it left.

Measured on SWE-chat (rev f66cca95; 3,712 sessions of the 105 developers with >= 5
sessions; developer top-1 of a tf-idf + multinomial logistic regression over path
tokens only; majority-class floor 0.121; the trace-funnel paper's
results/hashing_paths.json):

- raw paths 0.630; `hash_paths` with one release salt 0.630 — identical by
  construction, since a keyed hash is a bijection on tokens and a classifier sees
  tokens, not spellings. Pseudonymization removes readability, not linkability.
- `hash_paths` with a salt per session 0.121, the floor: every token becomes unique
  to its session, so the field carries nothing for re-identification and nothing
  for any other consumer either.
- `paths_to_dir` then `hash_paths` with one salt 0.764, higher than raw: directory
  tokens recur across one developer's sessions more than file tokens do, so
  coarsening to the directory concentrates the fingerprint.
- `min_users` with k=2 on raw path tokens keeps 102 of 18,427 distinct tokens (99.4%
  of them occur under one developer only) and drops developer top-1 to 0.168; k=5
  keeps 4 tokens. On that field the threshold and "drop the field" are the same
  operator (results/invariance_and_kthreshold.json).

Those are that corpus's numbers, not properties of the operators in general.
"""

# ntpath splits on both separators, so the `C:\Users\<name>\...` paths Cursor traces
# carry generalize like POSIX ones; posixpath would return the whole path, name and
# all, as the basename.
import hmac
import ntpath
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from itertools import takewhile

_PROMPT_TEXT_KEYS = ("text", "content")
_SCOPE_LABEL = {"release": None, "user": "user_id", "session": "session_id"}
_MIN_USERS_KEYS = ("file_path", "command")
_TIMESPEC = {0: "seconds", 3: "milliseconds", 6: "microseconds"}


def _events(record: dict) -> list | None:
    events = record.get("events") if isinstance(record, dict) else None
    return events if isinstance(events, list) else None


def _map_details(record: dict, keys: tuple[str, ...], fn: Callable[[str], str],
                 types: tuple[str, ...] | None = None) -> dict:
    """Rewrite `details[key]` for each key, on every event (of the given types) where
    it is a string. A record with nothing to rewrite is returned as the same object."""
    events = _events(record)
    if events is None:
        return record
    changed = False
    out = []
    for e in events:
        d = e.get("details") if isinstance(e, dict) else None
        if isinstance(d, dict) and (types is None or e.get("type") in types):
            hit = {k: fn(d[k]) for k in keys if isinstance(d.get(k), str)}
            if hit:
                e = {**e, "details": {**d, **hit}}
                changed = True
        out.append(e)
    return {**record, "events": out} if changed else record


def _map_prompts(record: dict, keys: tuple[str, ...], fn: Callable[[str], str]) -> dict:
    """The same rewrite over the record's `prompts` list, the copy of its prompt events."""
    prompts = record.get("prompts") if isinstance(record, dict) else None
    if not isinstance(prompts, list):
        return record
    changed = False
    out = []
    for p in prompts:
        hit = {k: fn(p[k]) for k in keys if isinstance(p.get(k), str)} if isinstance(p, dict) else {}
        if hit:
            p = {**p, **hit}
            changed = True
        out.append(p)
    return {**record, "prompts": out} if changed else record


# ---- pseudonymization -----------------------------------------------------------

def _keyed(record: dict, salt: str, scope: str) -> Callable[[str], str]:
    """HMAC-SHA256 under a key that is the salt alone (release scope) or the salt joined
    with the record's `labels.user_id` / `labels.session_id`. The key is derived here and
    used here; nothing in the output names it. Records lacking the label share one key."""
    if not salt:
        raise ValueError("a non-empty salt is required")
    if scope not in _SCOPE_LABEL:
        raise ValueError(f"scope must be one of {', '.join(_SCOPE_LABEL)}, not {scope!r}")
    label = _SCOPE_LABEL[scope]
    if label is None:
        key = salt
    else:
        labels = record.get("labels")
        ident = labels.get(label) if isinstance(labels, dict) else None
        key = f"{salt}\x1f{ident or ''}"
    return lambda value: hmac.new(key.encode(), value.encode(), "sha256").hexdigest()


def hash_paths(record: dict, salt: str, scope: str = "release") -> dict:
    """`details.file_path` -> its HMAC-SHA256 hex digest under the salt. The path text
    leaves; the token keeps its identity across the release (scope=release), within one
    developer (scope=user), or within one session (scope=session). The salt is never
    written into the record."""
    return _map_details(record, ("file_path",), _keyed(record, salt, scope))


def hash_commands(record: dict, salt: str, scope: str = "release") -> dict:
    """`details.command` -> its HMAC-SHA256 hex digest under the salt; same scopes as
    `hash_paths`. The command text leaves; the token keeps its identity within the scope."""
    return _map_details(record, ("command",), _keyed(record, salt, scope))


# ---- generalization -------------------------------------------------------------

def paths_to_dir(record: dict) -> dict:
    """`details.file_path` -> its directory. The basename leaves; a bare filename becomes ''."""
    return _map_details(record, ("file_path",), ntpath.dirname)


def _ext(path: str) -> str:
    return ntpath.splitext(path)[1].lower().lstrip(".") or "<noext>"


def paths_to_ext(record: dict) -> dict:
    """`details.file_path` -> its lowercase extension without the dot, or `<noext>`.
    Directory and name leave."""
    return _map_details(record, ("file_path",), _ext)


def paths_to_basename(record: dict) -> dict:
    """`details.file_path` -> its basename. The directory leaves."""
    return _map_details(record, ("file_path",), ntpath.basename)


def _head(command: str) -> str:
    return (command.split() or [""])[0].lower()


def commands_to_head(record: dict) -> dict:
    """`details.command` -> its first whitespace token, lowercased. Every argument leaves."""
    return _map_details(record, ("command",), _head)


def commands_to_class(record: dict) -> dict:
    """`details.command` -> test | search | read | run | other, by the repo's own
    `analysis.ingest.harnesses.classify_command` (the ingest classifier, so a released
    class matches what `trace import` would have assigned). The command text leaves."""
    from analysis.ingest.harnesses import classify_command

    return _map_details(record, ("command",), classify_command)


def text_to_length_bucket(record: dict, short: int = 80, long: int = 400) -> dict:
    """Prompt text -> short (< `short` chars) | medium (< `long`) | long, on prompt
    events' `details.text` and `details.content` and on the `prompts` list. The prompt
    text leaves; only its length class stays."""
    def bucket(text: str) -> str:
        return "short" if len(text) < short else "medium" if len(text) < long else "long"

    return _map_prompts(_map_details(record, _PROMPT_TEXT_KEYS, bucket, types=("prompt",)),
                        ("text",), bucket)


# ---- suppression ----------------------------------------------------------------

def _without(d: dict, keys) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


def drop_field(record: dict, field: str) -> dict:
    """Remove one named `details` key from every event, and from the `prompts` copies.
    The key and its value leave; the events themselves stay, so count and order do not
    change."""
    out = record
    events = _events(record)
    if events is not None:
        new = []
        changed = False
        for e in events:
            d = e.get("details") if isinstance(e, dict) else None
            if isinstance(d, dict) and field in d:
                e = {**e, "details": _without(d, (field,))}
                changed = True
            new.append(e)
        if changed:
            out = {**out, "events": new}
    prompts = record.get("prompts") if isinstance(record, dict) else None
    if isinstance(prompts, list) and any(isinstance(p, dict) and field in p for p in prompts):
        out = {**out, "prompts": [_without(p, (field,)) if isinstance(p, dict) else p for p in prompts]}
    return out


def min_users(records: Iterable[dict], k: int = 2) -> Callable[[dict], dict]:
    """Corpus-level: one pass over `records` counts, for every `details.file_path` and
    `details.command` value, the distinct `labels.user_id` it occurs under; the returned
    per-record rewrite removes the key wherever its value was seen under fewer than k
    developers. A value that recurs across one developer's sessions still counts as one.
    Records without a user_id all count as a single anonymous developer, which
    under-counts distinctness and so suppresses more, never less.

    Two passes over the file: `transforms.apply` builds the operator from a scan of the
    same records (the same `--limit`) it then rewrites one by one."""
    seen: dict[tuple[str, str], set] = defaultdict(set)
    for r in records:
        labels = r.get("labels") if isinstance(r, dict) else None
        uid = labels.get("user_id") if isinstance(labels, dict) else None
        for e in _events(r) or []:
            d = e.get("details") if isinstance(e, dict) else None
            if isinstance(d, dict):
                for key in _MIN_USERS_KEYS:
                    if isinstance(d.get(key), str):
                        seen[(key, d[key])].add(uid)
    rare = {kv for kv, users in seen.items() if len(users) < k}

    def rewrite(record: dict) -> dict:
        events = _events(record)
        if events is None:
            return record
        new = []
        changed = False
        for e in events:
            d = e.get("details") if isinstance(e, dict) else None
            if isinstance(d, dict):
                drop = [key for key in _MIN_USERS_KEYS if isinstance(d.get(key), str) and (key, d[key]) in rare]
                if drop:
                    e = {**e, "details": _without(d, drop)}
                    changed = True
            new.append(e)
        return {**record, "events": new} if changed else record

    return rewrite


def truncate_events(record: dict, n: int) -> dict:
    """Keep the first n events; the rest leave, and the `prompts` list is cut to the
    prompt events that remain (it is extracted from them in order)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    events = _events(record)
    if events is None or len(events) <= n:
        return record
    kept = events[:n]
    out = {**record, "events": kept}
    prompts = record.get("prompts")
    if isinstance(prompts, list):
        out["prompts"] = prompts[:sum(1 for e in kept if isinstance(e, dict) and e.get("type") == "prompt")]
    return out


# ---- perturbation ---------------------------------------------------------------

def _rng(record: dict, seed: int) -> random.Random:
    """One generator per (seed, record), so a record's draw does not depend on where in
    the file it sits or on how many records came before it."""
    return random.Random(f"{seed}\x1f{record.get('instance_id', '')}")


def _shift_iso(ts, seconds: int):
    """The timestamp moved by `seconds`, in the input's own shape: the same fractional
    precision, a trailing `Z` kept. Anything that does not parse is returned as it was."""
    if not isinstance(ts, str) or "T" not in ts:
        return ts
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    clock = ts.split("T", 1)[1]
    frac = clock.partition(".")[2] if "." in clock else ""
    digits = sum(1 for _ in takewhile(str.isdigit, frac))
    out = (dt + timedelta(seconds=seconds)).isoformat(timespec=_TIMESPEC.get(digits, "auto"))
    if ts.endswith("Z") and out.endswith("+00:00"):
        out = out[:-6] + "Z"
    return out


def jitter_timestamps(record: dict, hours: float, seed: int = 0) -> dict:
    """Shift every event and prompt timestamp in the record by one offset: a whole number
    of seconds drawn uniformly from +-hours by a generator seeded with (seed, instance_id).
    Absolute time leaves; the intervals between a record's events stay exactly."""
    if hours < 0:
        raise ValueError("hours must be >= 0")
    span = int(hours * 3600)
    offset = _rng(record, seed).randint(-span, span)

    def shifted(items: list) -> list | None:
        new = [{**x, "timestamp": _shift_iso(x["timestamp"], offset)}
               if isinstance(x, dict) and isinstance(x.get("timestamp"), str) else x for x in items]
        return new if new != items else None

    out = record
    events = _events(record)
    if events is not None and (new := shifted(events)) is not None:
        out = {**out, "events": new}
    prompts = record.get("prompts") if isinstance(record, dict) else None
    if isinstance(prompts, list) and (new := shifted(prompts)) is not None:
        out = {**out, "prompts": new}
    return out


# ---- order ----------------------------------------------------------------------

def shuffle_events(record: dict, seed: int = 0) -> dict:
    """Permute the order of the record's events, and of its `prompts` copies, by a
    generator seeded with (seed, instance_id). The order leaves; the multiset of events
    stays, timestamps included, so a consumer can still sort them back if it must."""
    events = _events(record)
    if events is None or len(events) < 2:
        return record
    rng = _rng(record, seed)
    new = list(events)
    rng.shuffle(new)
    out = {**record, "events": new}
    prompts = record.get("prompts")
    if isinstance(prompts, list) and len(prompts) > 1:
        ps = list(prompts)
        rng.shuffle(ps)
        out["prompts"] = ps
    return out


def verbs_only(record: dict) -> dict:
    """Keep only each event's type and timestamp; every details block leaves, and the
    `prompts` copies with it (they are prompt details). The event sequence stays."""
    out = record
    events = _events(record)
    if events is not None:
        stripped = [{k: e[k] for k in ("type", "timestamp") if k in e} if isinstance(e, dict) else e
                    for e in events]
        if stripped != events:
            out = {**out, "events": stripped}
    if isinstance(record.get("prompts"), list) and record["prompts"]:
        out = {**out, "prompts": []}
    return out
