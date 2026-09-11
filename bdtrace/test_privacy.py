"""What each privacy operator takes out of a record, and what it must leave alone.

Three synthetic records: two from one developer (u1, sessions s1 and s2) and one from
another (u2, s3), sharing one path and one command across developers and holding one
path and one command that a single developer touches. Every operator is checked four
ways: it changes the field it names (applied), a record without that field comes back
equal (not applicable), two runs agree (determinism), and the input is never mutated.
The registry checks are the other half: each operator is listed under its family with
its parameters, runs through `bdtrace transform`, and stays out of `all`.

No path or name here is real; the salts are test strings.
"""

import copy
import json

import pytest

from bdtrace import privacy, transforms
from bdtrace.test_cli import run_cli

SHARED = "/Users/jdoe/src/app/main.py"        # read by both developers
PRIVATE = "/Users/jdoe/src/app/notes_u1.md"   # edited by u1 only, in both of u1's sessions
WINDOWS = r"C:\Users\jdoe\proj\Main.PY"
SHARED_CMD = "pytest -q tests/"
PRIVATE_CMD = "git push origin main"
SHORT_TEXT = "fix the failing test"
MEDIUM_TEXT = "y" * 200
LONG_TEXT = "z" * 500


def _ev(etype: str, details: dict, ts: str) -> dict:
    return {"type": etype, "details": details, "timestamp": ts}


def _rec(iid: str, user: str, session: str, events: list[dict]) -> dict:
    return {"instance_id": iid, "repo": "o/r", "base_commit": None, "events": events,
            "prompts": [{"text": e["details"]["text"], "timestamp": e["timestamp"]}
                        for e in events if e["type"] == "prompt"],
            "labels": {"user_id": user, "session_id": session, "model": "m"}}


RECORDS = [
    _rec("swechat-s1", "u1", "s1", [
        _ev("prompt", {"text": SHORT_TEXT, "content": SHORT_TEXT}, "2026-03-01T10:00:00Z"),
        _ev("read", {"tool": "Read", "file_path": SHARED}, "2026-03-01T10:00:05Z"),
        _ev("run", {"tool": "Bash", "command": SHARED_CMD}, "2026-03-01T10:00:09Z"),
        _ev("edit", {"tool": "Edit", "file_path": PRIVATE}, "2026-03-01T10:01:00.250Z"),
    ]),
    _rec("swechat-s2", "u1", "s2", [
        _ev("prompt", {"text": LONG_TEXT}, "2026-03-02T09:00:00Z"),
        _ev("edit", {"tool": "Edit", "file_path": PRIVATE}, "2026-03-02T09:00:30Z"),
        _ev("run", {"tool": "Bash", "command": "git status"}, "2026-03-02T09:01:00Z"),
    ]),
    _rec("swechat-s3", "u2", "s3", [
        _ev("prompt", {"text": MEDIUM_TEXT}, "2026-03-03T08:00:00Z"),
        _ev("read", {"tool": "Read", "file_path": SHARED}, "2026-03-03T08:00:02Z"),
        _ev("run", {"tool": "Bash", "command": SHARED_CMD}, "2026-03-03T08:00:04Z"),
        _ev("run", {"tool": "Bash", "command": PRIVATE_CMD}, "2026-03-03T08:00:20Z"),
        _ev("read", {"tool": "Read", "file_path": WINDOWS}, "2026-03-03T08:00:22Z"),
    ]),
]
R1, R2, R3 = RECORDS

# nothing any operator acts on: one event with no details and no timestamp, no labels
BARE = {"instance_id": "bare", "events": [{"type": "other"}]}
# no events key at all
EMPTY = {"instance_id": "empty", "repo": None}

# operator -> the parameters that make it runnable on the fixture
OPERATORS = {
    "hash_paths": {"salt": "release-salt"},
    "hash_commands": {"salt": "release-salt"},
    "paths_to_dir": {},
    "paths_to_ext": {},
    "paths_to_basename": {},
    "commands_to_head": {},
    "commands_to_class": {},
    "text_to_length_bucket": {},
}


def run(name: str, record: dict, **overrides):
    return getattr(privacy, name)(record, **{**OPERATORS[name], **overrides})


def paths(record: dict) -> list:
    return [e["details"]["file_path"] for e in record["events"] if "file_path" in e.get("details", {})]


def commands(record: dict) -> list:
    return [e["details"]["command"] for e in record["events"] if "command" in e.get("details", {})]


def without(record: dict, key: str) -> dict:
    """The record with one details key removed everywhere: what an operator on that key must not touch."""
    return {**record, "events": [{**e, "details": {k: v for k, v in e["details"].items() if k != key}}
                                 for e in record["events"]]}


# ---- properties every operator holds ----------------------------------------------

@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("record", [BARE, EMPTY], ids=["bare", "empty"])
def test_not_applicable_is_untouched(name: str, record: dict):
    assert run(name, record) == record


@pytest.mark.parametrize("name", OPERATORS)
def test_deterministic_and_non_mutating(name: str):
    before = copy.deepcopy(RECORDS)
    first = [run(name, r) for r in RECORDS]
    second = [run(name, r) for r in RECORDS]
    assert first == second
    assert RECORDS == before
    assert any(a != b for a, b in zip(first, RECORDS, strict=True)), "the fixture exercises every operator"


@pytest.mark.parametrize("name", OPERATORS)
def test_listed_under_its_family_with_parameters(name: str):
    t = transforms.TRANSFORMS[name]
    assert t.kind == "rewrite" and t.module == "bdtrace.privacy" and t.family in transforms.PRIVACY_FAMILIES
    table = transforms.list_table()
    assert f"{name}({transforms.signature_text(t)})" in table and f"  {t.family}:" in table
    assert "leave" in t.desc  # the help says what leaves the record


# ---- pseudonymization ---------------------------------------------------------------

def test_hash_paths_release_scope_is_a_bijection_across_records():
    h1, h3 = run("hash_paths", R1), run("hash_paths", R3)
    digests = paths(h1) + paths(h3)
    assert all(len(d) == 64 and set(d) <= set("0123456789abcdef") for d in digests)
    assert paths(h1)[0] == paths(h3)[0]                      # same path, same token
    assert paths(h1)[1] != paths(h1)[0]                      # different paths, different tokens
    assert paths(run("hash_paths", R1, salt="other")) != paths(h1)
    assert commands(h1) == commands(R1) and h1["labels"] == R1["labels"]
    assert without(h1, "file_path") == without(R1, "file_path")


def test_hash_paths_user_and_session_scopes():
    by_user = [paths(run("hash_paths", r, scope="user")) for r in RECORDS]
    assert by_user[0][1] == by_user[1][0]      # PRIVATE in s1 and s2: same developer, same token
    assert by_user[0][0] != by_user[2][0]      # SHARED across developers: different tokens
    by_session = [paths(run("hash_paths", r, scope="session")) for r in RECORDS]
    assert by_session[0][1] != by_session[1][0]  # PRIVATE in s1 vs s2: unique per session
    assert by_user[0] != by_session[0] != paths(run("hash_paths", R1))


def test_hash_never_writes_the_salt_and_refuses_a_bad_key():
    salt = "the-release-salt-2026"
    text = json.dumps([run("hash_paths", r, salt=salt) for r in RECORDS])
    assert salt not in text and SHARED not in text and "jdoe" not in text.replace("notes_u1", "")
    with pytest.raises(ValueError):
        run("hash_paths", R1, salt="")
    with pytest.raises(ValueError):
        run("hash_paths", R1, scope="team")


def test_hash_commands():
    h1, h3 = run("hash_commands", R1), run("hash_commands", R3)
    assert commands(h1)[0] == commands(h3)[0] and len(commands(h3)[0]) == 64
    assert commands(h3)[1] != commands(h3)[0]
    assert paths(h3) == paths(R3)
    assert commands(run("hash_commands", R3, scope="session")) != commands(h3)
    assert without(h3, "command") == without(R3, "command")


# ---- generalization -----------------------------------------------------------------

def test_paths_to_dir():
    assert paths(run("paths_to_dir", R1)) == ["/Users/jdoe/src/app", "/Users/jdoe/src/app"]
    assert paths(run("paths_to_dir", R3)) == ["/Users/jdoe/src/app", r"C:\Users\jdoe\proj"]
    assert without(run("paths_to_dir", R3), "file_path") == without(R3, "file_path")


def test_paths_to_ext():
    assert paths(run("paths_to_ext", R1)) == ["py", "md"]
    assert paths(run("paths_to_ext", R3)) == ["py", "py"]
    dotfile = {"events": [_ev("read", {"file_path": "/Users/jdoe/.env"}, "t"),
                          _ev("read", {"file_path": "Makefile"}, "t")]}
    assert paths(run("paths_to_ext", dotfile)) == ["<noext>", "<noext>"]


def test_paths_to_basename():
    assert paths(run("paths_to_basename", R1)) == ["main.py", "notes_u1.md"]
    assert paths(run("paths_to_basename", R3)) == ["main.py", "Main.PY"]  # both separators split


def test_commands_to_head():
    assert commands(run("commands_to_head", R3)) == ["pytest", "git"]
    assert commands(run("commands_to_head", {"events": [_ev("run", {"command": "  Make  build"}, "t")]})) == ["make"]
    assert commands(run("commands_to_head", {"events": [_ev("run", {"command": "   "}, "t")]})) == [""]


def test_commands_to_class_uses_the_ingest_classifier():
    from analysis.ingest.harnesses import classify_command

    out = run("commands_to_class", R3)
    assert commands(out) == ["test", "run"] == [classify_command(c) for c in commands(R3)]
    assert commands(run("commands_to_class", R2)) == ["run"]
    assert set(commands(out)) <= {"test", "search", "read", "run", "other"}
    assert "pytest" not in json.dumps(out["events"])


def test_text_to_length_bucket_covers_every_copy_of_the_prompt():
    outs = [run("text_to_length_bucket", r) for r in RECORDS]
    assert [o["events"][0]["details"]["text"] for o in outs] == ["short", "long", "medium"]
    assert outs[0]["events"][0]["details"]["content"] == "short"
    assert [o["prompts"][0]["text"] for o in outs] == ["short", "long", "medium"]
    assert [o["prompts"][0]["timestamp"] for o in outs] == [r["prompts"][0]["timestamp"] for r in RECORDS]
    assert outs[0]["events"][1:] == R1["events"][1:]  # non-prompt events untouched, text on them included
    assert SHORT_TEXT not in json.dumps(outs[0]) and LONG_TEXT not in json.dumps(outs[1])


def test_text_to_length_bucket_boundaries_are_explicit():
    def at(n: int, **kw) -> str:
        return run("text_to_length_bucket", {"events": [_ev("prompt", {"text": "a" * n}, "t")]}, **kw)["events"][0]["details"]["text"]

    assert [at(79), at(80), at(399), at(400)] == ["short", "medium", "medium", "long"]
    assert [at(5, short=3, long=6), at(6, short=3, long=6)] == ["medium", "long"]


# ---- through the registry and the CLI ------------------------------------------------

@pytest.fixture
def records_file(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    return p


def test_cli_applies_an_operator_with_params(records_file, monkeypatch, capsys):
    code, _, err = run_cli(["tf", "hash_paths", "--in", str(records_file), "-p", "salt=abc", "-p", "scope=user"],
                           monkeypatch, capsys)
    out_path = records_file.with_suffix(".hash_paths.jsonl")
    assert code == 0 and "3 records" in err and out_path.exists()
    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert rows == [privacy.hash_paths(r, salt="abc", scope="user") for r in RECORDS]
    assert "abc" not in out_path.read_text() and all("reprs" not in r for r in rows)


def test_cli_salt_comes_from_the_environment_not_the_record(records_file, monkeypatch, capsys):
    monkeypatch.setenv("BDTRACE_SALT", "env-salt-value")
    code, _, err = run_cli(["tf", "hash_commands", "--in", str(records_file)], monkeypatch, capsys)
    assert code == 0 and "salt: BDTRACE_SALT" in err and "env-salt-value" not in err
    text = records_file.with_suffix(".hash_commands.jsonl").read_text()
    assert "env-salt-value" not in text and SHARED_CMD not in text
    monkeypatch.delenv("BDTRACE_SALT")
    code, _, err = run_cli(["tf", "hash_commands", "--in", str(records_file)], monkeypatch, capsys)
    assert code != 0 and "BDTRACE_SALT" in err


def test_cli_refuses_unknown_or_mistyped_params(records_file, monkeypatch, capsys):
    code, _, err = run_cli(["tf", "paths_to_dir", "--in", str(records_file), "-p", "depth=2"], monkeypatch, capsys)
    assert code != 0 and "no parameter depth" in err
    code, _, err = run_cli(["tf", "text_to_length_bucket", "--in", str(records_file), "-p", "short=many"],
                           monkeypatch, capsys)
    assert code != 0 and "must be int" in err
    code, _, err = run_cli(["tf", "tokens", "--in", str(records_file), "-p", "x=1"], monkeypatch, capsys)
    assert code != 0 and "representation transforms take none" in err


def test_cli_all_excludes_privacy_operators(records_file, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(transforms, "apply", lambda names, *a, **k: seen.setdefault("names", names))
    code, _, _ = run_cli(["tf", "all", "--in", str(records_file)], monkeypatch, capsys)
    assert code == 0 and seen["names"] and not any(transforms.TRANSFORMS[n].family for n in seen["names"])


def test_cli_list_shows_families(monkeypatch, capsys):
    code, out, _ = run_cli(["tf", "list"], monkeypatch, capsys)
    assert code == 0 and "privacy operators" in out
    assert "hash_paths(salt, scope=release)" in out and "text_to_length_bucket(short=80, long=400)" in out
    assert out.index("pseudonymization:") < out.index("generalization:")
