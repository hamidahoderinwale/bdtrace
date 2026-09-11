"""Privacy-preserving rewrites of trace records: one operator per field type.

Each operator takes one record and returns a new record with a single field type
changed — `details.file_path`, `details.command`, or prompt text — and everything
else as it was: a record without that field comes back equal to what went in, a
missing or oddly typed field is skipped rather than raised on, and the input record
is never mutated. Operators are grouped by the standard privacy families
(pseudonymization, generalization, suppression, perturbation, order) and compose by
chaining `bdtrace transform` calls; there is no built-in ladder, because the right
order depends on what the release is for.

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

Those are that corpus's numbers, not properties of the operators in general.
"""

# ntpath splits on both separators, so the `C:\Users\<name>\...` paths Cursor traces
# carry generalize like POSIX ones; posixpath would return the whole path, name and
# all, as the basename.
import hmac
import ntpath
from collections.abc import Callable

_PROMPT_TEXT_KEYS = ("text", "content")
_SCOPE_LABEL = {"release": None, "user": "user_id", "session": "session_id"}


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
