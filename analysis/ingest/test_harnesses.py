"""Tests for analysis/ingest/harnesses.py against synthetic fixtures (never live stores)."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harnesses  # noqa: E402
from harnesses import (  # noqa: E402
    EVENT_TYPES,
    classify_command,
    iter_traces_cursor,
    iter_traces_openhands,
    iter_traces_swe_agent,
    parse,
)


def assert_trace_schema(trace: dict) -> None:
    assert set(trace) == {"instance_id", "repo", "base_commit", "events", "prompts"}
    assert isinstance(trace["instance_id"], str) and trace["instance_id"]
    assert trace["repo"] is None or isinstance(trace["repo"], str)
    assert trace["base_commit"] is None or isinstance(trace["base_commit"], str)
    assert isinstance(trace["prompts"], list)
    assert isinstance(trace["events"], list) and trace["events"]
    for event in trace["events"]:
        assert set(event) == {"type", "details"}
        assert event["type"] in EVENT_TYPES
        assert isinstance(event["details"], dict)
    json.dumps(trace)


@pytest.fixture
def cursor_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "state.vscdb"
    con = sqlite3.connect(db_path)
    con.execute("create table cursorDiskKV (key text primary key, value blob)")
    con.execute("create table ItemTable (key text primary key, value blob)")
    composer_id = "comp-1"
    bubbles = [
        ("bub-1", {"type": 1, "text": "fix the failing separability test", "createdAt": "2026-01-01T00:00:01Z"}),
        (
            "bub-2",
            {
                "type": 2,
                "text": "",
                "createdAt": "2026-01-01T00:00:02Z",
                "toolFormerData": {
                    "name": "read_file_v2",
                    "rawArgs": json.dumps({"path": "/repo/separable.py"}),
                },
            },
        ),
        (
            "bub-3",
            {
                "type": 2,
                "text": "",
                "createdAt": "2026-01-01T00:00:03Z",
                "toolFormerData": {
                    "name": "ripgrep_raw_search",
                    "rawArgs": json.dumps({"pattern": "separability_matrix"}),
                },
            },
        ),
        (
            "bub-4",
            {
                "type": 2,
                "text": "",
                "createdAt": "2026-01-01T00:00:04Z",
                "toolFormerData": {
                    "name": "edit_file_v2",
                    "rawArgs": json.dumps(
                        {"target_file": "/repo/separable.py", "code_edit": "def fixed(): pass"}
                    ),
                },
            },
        ),
        (
            "bub-5",
            {
                "type": 2,
                "text": "",
                "createdAt": "2026-01-01T00:00:05Z",
                "toolFormerData": {
                    "name": "run_terminal_command_v2",
                    "rawArgs": json.dumps({"command": "python -m pytest tests/"}),
                },
            },
        ),
        ("bub-6", {"type": 2, "text": "done, the fix is in.", "createdAt": "2026-01-01T00:00:06Z"}),
    ]
    composer = {
        "composerId": composer_id,
        "text": "",
        "createdAt": 1780000000000,
        "fullConversationHeadersOnly": [{"bubbleId": bid, "type": b["type"]} for bid, b in bubbles],
    }
    con.execute(
        "insert into cursorDiskKV values (?, ?)",
        (f"composerData:{composer_id}", json.dumps(composer)),
    )
    for bid, bubble in bubbles:
        con.execute(
            "insert into cursorDiskKV values (?, ?)",
            (f"bubbleId:{composer_id}:{bid}", json.dumps(bubble)),
        )
    # a composer with no bubbles and no text must be skipped
    con.execute(
        "insert into cursorDiskKV values (?, ?)",
        ("composerData:comp-empty", json.dumps({"composerId": "comp-empty", "text": ""})),
    )
    con.commit()
    con.close()
    return db_path


def test_cursor_sqlite(cursor_db: Path):
    traces = list(iter_traces_cursor(cursor_db))
    assert len(traces) == 1
    trace = traces[0]
    assert_trace_schema(trace)
    assert trace["instance_id"] == "cursor-comp-1"
    types = [e["type"] for e in trace["events"]]
    assert types == ["prompt", "read", "search", "edit", "test", "other"]
    assert trace["prompts"] == [{"text": "fix the failing separability test"}]
    edit = trace["events"][3]
    assert edit["details"]["file_path"] == "/repo/separable.py"
    assert edit["details"]["after_content"] == "def fixed(): pass"


def test_cursor_user_dir_layout(cursor_db: Path, tmp_path: Path):
    user_dir = tmp_path / "User"
    (user_dir / "globalStorage").mkdir(parents=True)
    (user_dir / "workspaceStorage" / "ws1").mkdir(parents=True)
    (user_dir / "globalStorage" / "state.vscdb").write_bytes(cursor_db.read_bytes())
    (user_dir / "workspaceStorage" / "ws1" / "state.vscdb").write_bytes(cursor_db.read_bytes())
    traces = list(iter_traces_cursor(user_dir))
    assert len(traces) == 2
    for trace in traces:
        assert_trace_schema(trace)
    assert len(list(iter_traces_cursor(user_dir, limit=1))) == 1


def test_cursor_raw_export(tmp_path: Path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "prompts_raw.txt").write_text(json.dumps([{"text": "add a cli flag"}]))
    (export / "conversations_raw.txt").write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "messages": [
                        {"role": "user", "content": "please add --limit"},
                        {"role": "assistant", "content": "added it"},
                    ],
                }
            ]
        )
    )
    traces = list(iter_traces_cursor(export))
    assert len(traces) == 1
    trace = traces[0]
    assert_trace_schema(trace)
    types = [e["type"] for e in trace["events"]]
    assert types == ["prompt", "prompt", "other"]
    assert len(trace["prompts"]) == 2


@pytest.fixture
def swe_agent_dir(tmp_path: Path) -> Path:
    traj_dir = tmp_path / "child_traj"
    traj_dir.mkdir()
    traj = {
        "history": [
            {"role": "system", "content": "boilerplate", "agent": "main"},
            {"role": "user", "content": "fix separability_matrix for nested models", "agent": "main"},
            {"role": "assistant", "content": "ok", "action": "find /testbed -name '*.py'", "agent": "main"},
        ],
        "trajectory": [
            {"action": "find /testbed -type f -name '*.py' | grep separable"},
            {"action": "str_replace_editor view /testbed/astropy/modeling/separable.py"},
            {"action": "str_replace_editor str_replace /testbed/astropy/modeling/separable.py"},
            {"action": "cd /testbed && python -m pytest astropy/modeling/tests/"},
            {"action": "grep -n 'class CompoundModel' /testbed/astropy/modeling/core.py"},
            {"action": "submit"},
        ],
        "replay_config": json.dumps(
            {"env": {"repo": {"repo_name": "testbed", "base_commit": "d16bfe05a744"}}}
        ),
        "info": {"exit_status": "submitted"},
    }
    (traj_dir / "astropy__astropy-12907.traj").write_text(json.dumps(traj))
    traj2 = dict(traj)
    traj2["replay_config"] = "not json"
    (traj_dir / "django__django-11099.traj").write_text(json.dumps(traj2))
    return traj_dir


def test_swe_agent(swe_agent_dir: Path):
    traces = list(iter_traces_swe_agent(swe_agent_dir))
    assert len(traces) == 2
    by_id = {t["instance_id"]: t for t in traces}
    trace = by_id["astropy__astropy-12907"]
    assert_trace_schema(trace)
    assert trace["repo"] == "astropy/astropy"
    assert trace["base_commit"] == "d16bfe05a744"
    types = [e["type"] for e in trace["events"]]
    assert types == ["prompt", "search", "read", "edit", "test", "search", "other"]
    assert trace["events"][3]["details"]["file_path"] == "/testbed/astropy/modeling/separable.py"
    assert trace["prompts"][0]["text"].startswith("fix separability_matrix")
    # unparseable replay_config degrades to None, never raises
    assert by_id["django__django-11099"]["base_commit"] is None
    assert by_id["django__django-11099"]["repo"] == "django/django"
    assert len(list(iter_traces_swe_agent(swe_agent_dir, limit=1))) == 1


def test_swe_agent_history_only(tmp_path: Path):
    traj = {
        "history": [
            {"role": "user", "content": "task statement"},
            {"role": "assistant", "content": "x", "action": "str_replace_editor view /f.py"},
            {"role": "user", "content": "OBSERVATION: ..."},
            {"role": "assistant", "content": "y", "action": "submit"},
        ]
    }
    p = tmp_path / "solo.traj"
    p.write_text(json.dumps(traj))
    traces = list(iter_traces_swe_agent(p))
    assert len(traces) == 1
    assert [e["type"] for e in traces[0]["events"]] == ["prompt", "read", "other"]


@pytest.fixture
def openhands_jsonl(tmp_path: Path) -> Path:
    records = [
        {
            "instance_id": "sympy__sympy-13437",
            "trajectory": [
                {"role": "user", "content": "bell numbers bug"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": json.dumps({"command": "view", "path": "/testbed/sympy/functions/combinatorial/numbers.py"}),
                            }
                        },
                        {
                            "function": {
                                "name": "execute_bash",
                                "arguments": json.dumps({"command": "grep -rn 'def bell' /testbed"}),
                            }
                        },
                    ],
                },
                {"role": "tool", "content": "observation"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "str_replace_editor",
                                "arguments": json.dumps(
                                    {
                                        "command": "str_replace",
                                        "path": "/testbed/sympy/functions/combinatorial/numbers.py",
                                        "old_str": "return oo",
                                        "new_str": "return S.Infinity",
                                    }
                                ),
                            }
                        },
                        {
                            "function": {
                                "name": "execute_bash",
                                "arguments": json.dumps({"command": "python -m pytest sympy/functions/combinatorial/tests -x"}),
                            }
                        },
                        {"function": {"name": "finish", "arguments": "{}"}},
                    ],
                },
            ],
        },
        {"instance_id": "empty-one", "trajectory": []},
    ]
    p = tmp_path / "openhands.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


def test_openhands(openhands_jsonl: Path):
    traces = list(iter_traces_openhands(openhands_jsonl))
    assert len(traces) == 1  # empty trajectory yields nothing
    trace = traces[0]
    assert_trace_schema(trace)
    assert trace["instance_id"] == "sympy__sympy-13437"
    assert trace["repo"] == "sympy/sympy"
    types = [e["type"] for e in trace["events"]]
    assert types == ["prompt", "read", "search", "edit", "test", "other"]
    edit = trace["events"][3]
    assert edit["details"]["before_content"] == "return oo"
    assert edit["details"]["after_content"] == "return S.Infinity"
    assert trace["prompts"] == [{"text": "bell numbers bug"}]


def test_classify_command():
    assert classify_command("python -m pytest tests/") == "test"
    assert classify_command("tox -e py311") == "test"
    assert classify_command("grep -rn foo .") == "search"
    assert classify_command("rg pattern src/") == "search"
    assert classify_command("cat /etc/hosts") == "read"
    assert classify_command("pip install -e .") == "run"
    assert classify_command("") == "other"


def test_parse_dispatch_and_main(openhands_jsonl: Path, tmp_path: Path):
    traces = list(parse("openhands", openhands_jsonl))
    assert len(traces) == 1
    with pytest.raises(ValueError):
        list(parse("aider", openhands_jsonl))
    out = tmp_path / "out.jsonl"
    rc = harnesses.main(
        ["--source", "openhands", "--input", str(openhands_jsonl), "--output", str(out)]
    )
    assert rc == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert_trace_schema(json.loads(lines[0]))


def test_missing_paths_raise(tmp_path: Path):
    for fn in (iter_traces_cursor, iter_traces_swe_agent, iter_traces_openhands):
        with pytest.raises(FileNotFoundError):
            list(fn(tmp_path / "nope"))
