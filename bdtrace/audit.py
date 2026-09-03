"""Residual-identity audit: what identity is still in a file you are about to share.

`anonymize` is a denylist, so it removes what its rules name and nothing else.
Three leaks in one session (a workspace slug, a forge owner in `gh` commands, a
second machine's login) were each found by reading output, not by the rules. This
module is the other half: it reads a finished artifact and reports what still
looks like identity, so sharing is a decision made on evidence.

Two kinds of finding, deliberately separate:

- **residual**: matches a class the anonymizer claims to remove. A non-zero count
  is a defect in the rules, and `--strict` refuses to write on one.
- **candidate**: a token that no rule can recognise as identity but that sits in a
  position identity occupies (a handle after `--author`, a user-like path segment,
  a repeated capitalized word). These need a human. `--redact TERM` closes them.
"""

import json
import re
from collections import Counter
from pathlib import Path

# Classes the anonymizer claims to remove. A hit here means a rule missed.
RESIDUAL_PATTERNS = {
    "home path": re.compile(r"/(?:Users|home)/(?!<)[A-Za-z0-9_.-]+"),
    "windows home": re.compile(r"[A-Za-z]:\\\\?Users\\\\?(?!<)[A-Za-z0-9_.-]+"),
    "workspace slug": re.compile(r"-Users-(?!anon\b)[A-Za-z0-9_.]+"),
    "url-encoded home": re.compile(r"%2FUsers%2F(?!%3C)[A-Za-z0-9_.-]+", re.I),
    # a real address, not a font weight (wght@300..700) or a decorator (@app.tool)
    "email": re.compile(r"\b[A-Za-z][\w.+-]*@[A-Za-z][\w-]*\.[A-Za-z]{2,24}\b"),
    "credential": re.compile(r"\b(?:sk-[A-Za-z0-9-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}"
                             r"|hf_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})"),
    "forge owner": re.compile(r"(?:github\.com|gitlab\.com)/(?!<)[A-Za-z0-9._-]+"),
    "repos owner": re.compile(r"\brepos/(?!<)[A-Za-z0-9._-]+"),
}

# Positions identity occupies, whatever the value is. The captured group is the candidate.
CANDIDATE_PATTERNS = (
    re.compile(r"--author[= ]+([A-Za-z0-9._-]{3,})"),
    re.compile(r"\b(?:user|owner|login|account|username)\s*[=:]\s*[\"']?([A-Za-z0-9._-]{3,})"),
    re.compile(r"/(?:Users|home)/([A-Za-z0-9_.-]{3,})"),
    re.compile(r"-Users-([A-Za-z0-9_.]{3,})"),
    re.compile(r"\b([A-Za-z0-9._-]{3,})@[A-Za-z][\w-]*\.[A-Za-z]{2,24}\b"),
    re.compile(r"\b(?:repos|github\.com|gitlab\.com)/([A-Za-z0-9._-]{3,})/"),
)

# Placeholders the anonymizer writes, plus words that occupy identity positions
# without being identity. A candidate matching these is noise.
_NOT_IDENTITY = {
    "anon", "user", "users", "home", "root", "admin", "runner", "none", "null",
    "true", "false", "local", "shared", "public", "main", "master", "origin",
    "redacted", "token", "email", "host", "email>", "github", "gitlab",
    # ordinary words the identity-position patterns pick up out of prose
    "the", "and", "for", "you", "yes", "not", "add", "pass", "with", "this",
    "that", "from", "into", "name", "base", "mon", "tue", "wed", "thu", "fri",
    "repos", "repo", "self", "test", "data", "file", "path", "list", "all",
}


def _strings(value) -> list[str]:
    """Every string anywhere in a record: identity hides in nested detail blobs."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def audit(in_path: Path, extra_terms: tuple[str, ...] = ()) -> dict:
    """Scan a trace JSONL for identity that survived. Returns counts and masked samples."""
    residual: Counter = Counter()
    samples: dict[str, set] = {}
    candidates: Counter = Counter()
    n = 0
    with open(in_path) as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            for s in _strings(json.loads(line)):
                for name, pattern in RESIDUAL_PATTERNS.items():
                    for hit in pattern.findall(s):
                        residual[name] += 1
                        samples.setdefault(name, set()).add(_mask(hit))
                for pattern in CANDIDATE_PATTERNS:
                    for hit in pattern.findall(s):
                        if hit.lower() not in _NOT_IDENTITY and not hit.startswith("<"):
                            candidates[hit] += 1
                for term in extra_terms:
                    if term in s:
                        residual[f"term {term!r}"] += 1
    return {
        "records": n,
        "residual": dict(residual.most_common()),
        "residual_samples": {k: sorted(v)[:3] for k, v in samples.items()},
        "candidates": dict(candidates.most_common(12)),
    }


def _mask(hit: str) -> str:
    """Show the shape of a finding without reprinting the identity it contains."""
    tail = hit.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    keep = tail[:2] if len(tail) > 4 else tail[:1]
    return hit[: len(hit) - len(tail)] + keep + "*" * max(len(tail) - len(keep), 1)


def report(result: dict) -> str:
    lines = [f"{result['records']} records scanned"]
    if result["residual"]:
        lines.append("RESIDUAL — classes the anonymizer should have removed:")
        for name, count in result["residual"].items():
            eg = ", ".join(result["residual_samples"].get(name, [])[:3])
            lines.append(f"  {name:<18} {count:>6}" + (f"   e.g. {eg}" if eg else ""))
    else:
        lines.append("RESIDUAL — none: every class the anonymizer covers is clean")
    if result["candidates"]:
        lines.append("CANDIDATES — identity-shaped tokens no rule can name; review these:")
        for token, count in result["candidates"].items():
            lines.append(f"  {token:<28} {count:>6}   close with --redact {token}")
    return "\n".join(lines)
