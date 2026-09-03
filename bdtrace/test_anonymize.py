"""What `bdtrace trace export --anonymize` must strip before a trace leaves the machine.

One case per identity class, each stating the exact anonymized form, plus the two
properties the whole scrubber has to hold: anonymizing twice equals anonymizing once,
and the record's structure — event count, event types, field names, timestamps — comes
through untouched. The no-op cases are the other half of the bar: strings that look
like an identity class but are not (`settings.local.json`, an ISO timestamp, a commit
sha) must survive verbatim, or the anonymizer buys privacy with the trace's usefulness.

No value here is a real credential; the key-shaped strings are synthetic padding.
"""

from __future__ import annotations

import json

import pytest

from bdtrace import spec

# machine-derived names are pinned so the table reads the same on every machine
LOCAL_NAMES = ("jdoe-mbp", "jdoe")

BEARER = "x" * 40
GH_PAT = "ghp_" + "b" * 36
ANTHROPIC_KEY = "sk-ant-api03-" + "A" * 40
HF_TOKEN = "hf_" + "A" * 34
JWT = "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY.SflKxwRJSMeKKF2QT4"
# assembled, never written whole: a literal here trips GitHub push protection
SLACK_TOKEN = "xox" + "b-123456789012-abcdefghijklmnop"
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"


@pytest.fixture(autouse=True)
def _pinned_local_names(monkeypatch):
    monkeypatch.setattr(spec, "_LOCAL_NAMES", LOCAL_NAMES)


# (identity class, string as it occurs in a trace, string after anonymizing)
CASES = [
    ("home-path-macos", "cd /Users/jdoe/src/app && pytest -q", "cd ~/src/app && pytest -q"),
    ("home-path-linux", "/home/jdoe/.cache/bdtrace", "~/.cache/bdtrace"),
    ("home-path-root", "/root/.ssh/known_hosts", "~/.ssh/known_hosts"),
    ("home-path-windows", r"C:\Users\jdoe\Documents\notes.txt", r"~\Documents\notes.txt"),
    ("home-path-url-encoded", "%2FUsers%2Fjdoe%2Fsrc", "~%2Fsrc"),
    ("workspace-slug-macos",
     "~/.claude/projects/-Users-jdoe-learning-from-dev/a.jsonl",
     "~/.claude/projects/-Users-anon-learning-from-dev/a.jsonl"),
    ("workspace-slug-linux", "-home-jdoe-work/x", "-home-anon-work/x"),
    ("uid-temp-dir",
     "/private/tmp/claude-501/-Users-jdoe/7f09/scratchpad",
     "/private/tmp/claude-<uid>/-Users-anon/7f09/scratchpad"),
    ("per-user-temp-dir",
     "/var/folders/zz/8kq1n0s7xz/T/tmpabc.json",
     "/var/folders/<tmp>/T/tmpabc.json"),
    ("volume-name", "/Volumes/BackupSSD/data.jsonl", "/Volumes/<volume>/data.jsonl"),
    ("bare-username", "su jdoe && whoami", "su anon && whoami"),
    ("machine-name", "host jdoe-mbp is up", "host anon is up"),
    ("mdns-hostname", "resolved Janes-MacBook-Pro.local ok", "resolved <host> ok"),
    ("user-at-host", "ssh jdoe@Janes-MacBook-Pro.local", "ssh <email>"),
    ("email", "contact jane.doe+ci@example.co.uk", "contact <email>"),
    ("ipv4", "curl http://192.168.1.42:8080/health", "curl http://<ip>:8080/health"),
    ("mac-address", "arp -a 3c:22:fb:1a:2b:3c", "arp -a <ip>"),
    ("git-remote-ssh",
     "git remote add origin git@github.com:janedoe/proj.git",
     "git remote add origin git@<host>:<user>/proj.git"),
    ("git-remote-https",
     "git clone https://github.com/janedoe/proj.git",
     "git clone https://github.com/<user>/proj.git"),
    ("url-credentials",
     f"https://janedoe:{GH_PAT}@github.com/janedoe/proj",
     "https://github.com/<user>/proj"),
    ("hub-account", "https://huggingface.co/datasets/janedoe/traces",
     "https://huggingface.co/datasets/<user>/traces"),
    ("key-anthropic", f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}", "ANTHROPIC_API_KEY=<token>"),
    ("key-github-pat", f"gh auth login --with-token {GH_PAT}", "gh auth login --with-token <token>"),
    ("key-huggingface", f"login {HF_TOKEN}", "login <token>"),
    ("key-aws", AWS_KEY, "<token>"),
    ("key-slack", SLACK_TOKEN, "<token>"),
    ("jwt", JWT, "<token>"),
    ("bearer-header", f"-H 'Authorization: Bearer {BEARER}'", "-H 'Authorization: Bearer <token>'"),
    ("secret-by-field-name", '{"api_key": "abcdefgh12345678"}', '{"api_key": "<redacted>"}'),
]

# strings shaped like an identity class that carry no identity: they must survive intact
NO_OP_CASES = [
    ("dotted-local-filename", "settings.local.json"),
    ("iso-timestamp", "2026-09-02T20:41:33.123Z"),
    ("commit-sha", "base_commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"),
    ("session-id", "claude-7f091101-3bde-420e-9c86-5b20f6a930b1"),
    ("semver", "version 1.2.3 build 4"),
    ("root-mid-path", "/some/root/dir"),
    ("ordinary-command", "npm run build && pytest -q"),
]


@pytest.mark.parametrize("raw, expected", [pytest.param(r, e, id=name) for name, r, e in CASES])
def test_class_is_stripped(raw: str, expected: str):
    assert spec._anon_str(raw) == expected


@pytest.mark.parametrize("raw", [pytest.param(r, id=name) for name, r in NO_OP_CASES])
def test_lookalike_survives(raw: str):
    assert spec._anon_str(raw) == raw


@pytest.mark.parametrize("raw", [pytest.param(r, id=name) for name, r, _ in CASES]
                         + [pytest.param(r, id=name) for name, r in NO_OP_CASES])
def test_idempotent(raw: str):
    once = spec._anon_str(raw)
    assert spec._anon_str(once) == once


def test_local_identifiers_are_usable():
    """The machine-derived half of the scrubber: no generic name that would blank out
    ordinary prose, and longest first so `jdoe-mbp` is consumed before the `jdoe` in it."""
    names = spec._local_identifiers()
    assert all(len(n) > 2 for n in names)
    assert not (set(n.lower() for n in names) & spec._GENERIC_NAMES)
    assert list(names) == sorted(names, key=len, reverse=True)


# one trace record in the shape analysis/ingest/claude_code.py produces, carrying one
# instance of every class the scrubber claims to close
RECORD = {
    "instance_id": "claude-7f091101-3bde-420e-9c86-5b20f6a930b1",
    "repo": None,
    "base_commit": None,
    "cwd": "/Users/jdoe/learning-from-dev",
    "events": [
        {"type": "prompt",
         "details": {"text": "mail jane.doe@example.com the file at /Users/jdoe/src/x.py",
                     "content": "mail jane.doe@example.com the file at /Users/jdoe/src/x.py",
                     "cwd": "/Users/jdoe/learning-from-dev"},
         "timestamp": "2026-09-02T20:41:33.123Z"},
        {"type": "read",
         "details": {"tool": "Read", "file_path": "/Users/jdoe/src/spec.py",
                     "cwd": "/Users/jdoe/learning-from-dev"},
         "timestamp": "2026-09-02T20:42:00.000Z"},
        {"type": "run",
         "details": {"tool": "Bash",
                     "command": ("git push git@github.com:janedoe/proj.git && "
                                 f"curl -H 'Authorization: Bearer {BEARER}' https://10.0.0.5/api"),
                     "description": "push from jdoe-mbp",
                     "cwd": "/Users/jdoe/learning-from-dev"},
         "timestamp": "2026-09-02T20:43:00.000Z"},
        {"type": "other",
         "details": {"tool": "WebFetch",
                     "input": '{"url": "https://github.com/janedoe/proj", "token": "abcdefgh12345678"}'},
         "timestamp": "2026-09-02T20:44:00.000Z"},
    ],
    "prompts": [{"text": "mail jane.doe@example.com", "timestamp": "2026-09-02T20:41:33.123Z"}],
}

LEAKS = ["jdoe", "jdoe-mbp", "jane.doe@example.com", "janedoe", "10.0.0.5", BEARER,
         "abcdefgh12345678", "/Users/"]


def test_structure_is_preserved():
    out = spec.anonymize(RECORD)
    assert list(out) == list(RECORD)
    assert out["instance_id"] == RECORD["instance_id"]
    assert out["repo"] is None and out["base_commit"] is None
    assert len(out["events"]) == len(RECORD["events"])
    assert [e["type"] for e in out["events"]] == [e["type"] for e in RECORD["events"]]
    assert [e["timestamp"] for e in out["events"]] == [e["timestamp"] for e in RECORD["events"]]
    assert [list(e["details"]) for e in out["events"]] == [list(e["details"]) for e in RECORD["events"]]
    assert len(out["prompts"]) == len(RECORD["prompts"])


def test_no_identity_survives_the_record():
    text = json.dumps(spec.anonymize(RECORD))
    assert [leak for leak in LEAKS if leak in text] == []


def test_embedded_json_stays_parseable():
    """`details.input` is a JSON dump of a tool's arguments; redaction replaces the
    value inside the quotes, so a consumer can still parse it."""
    out = spec.anonymize(RECORD)
    payload = json.loads(out["events"][3]["details"]["input"])
    assert payload == {"url": "https://github.com/<user>/proj", "token": "<redacted>"}


def test_record_anonymize_is_idempotent():
    once = spec.anonymize(RECORD)
    assert spec.anonymize(once) == once


def test_containers_and_scalars():
    assert spec.anonymize("/Users/jdoe/x") == "~/x"
    assert spec.anonymize(["/Users/jdoe", 1, None, True]) == ["~", 1, None, True]
    assert spec.anonymize({"cwd": "/home/jdoe"}) == {"cwd": "~"}
