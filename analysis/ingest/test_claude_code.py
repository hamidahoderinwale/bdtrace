import json

import pytest

from analysis.ingest.claude_code import (
    MAX_FIELD_CHARS,
    iter_traces,
    parse_session_file,
)

SESSION_ID = "abc12345-0000-0000-0000-000000000001"
CWD = "/Users/dev/myproject"


def _line(line_type, content=None, **extra):
    obj = {
        "type": line_type,
        "sessionId": SESSION_ID,
        "cwd": CWD,
        "timestamp": "2026-09-01T10:00:00.000Z",
    }
    if content is not None:
        role = "user" if line_type == "user" else "assistant"
        obj["message"] = {"role": role, "content": content}
    obj.update(extra)
    return obj


def _tool_use(name, tool_input):
    return {"type": "tool_use", "id": "toolu_x", "name": name, "input": tool_input}


@pytest.fixture
def session_file(tmp_path):
    lines = [
        _line("user", "Fix the failing parser test"),
        _line("user", [{"type": "text", "text": "second prompt as block list"}]),
        _line("user", "meta line to skip", isMeta=True),
        _line("user", [{"type": "tool_result", "tool_use_id": "toolu_x", "content": "grep output"}]),
        _line("user", "<command-name>/clear</command-name>"),
        _line("assistant", [
            {"type": "text", "text": "Looking at the file."},
            _tool_use("Read", {"file_path": "/Users/dev/myproject/parser.py"}),
        ]),
        _line("assistant", [_tool_use("Grep", {"pattern": "def parse", "path": "/Users/dev/myproject"})]),
        _line("assistant", [_tool_use("Edit", {
            "file_path": "/Users/dev/myproject/parser.py",
            "old_string": "return None",
            "new_string": "return result",
        })]),
        _line("assistant", [_tool_use("Write", {
            "file_path": "/Users/dev/myproject/new.py",
            "content": "x" * (MAX_FIELD_CHARS + 500),
        })]),
        _line("assistant", [_tool_use("Bash", {"command": "ls -la", "description": "List files"})]),
        _line("assistant", [_tool_use("Bash", {"command": "uv run pytest tests/ -x"})]),
        _line("assistant", [_tool_use("mcp__linear__save_issue", {"title": "t"})]),
        _line("file-history-snapshot"),
    ]
    slug_dir = tmp_path / "-Users-dev-myproject"
    slug_dir.mkdir()
    path = slug_dir / f"{SESSION_ID}.jsonl"
    with open(path, "w") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")
        f.write("not json at all\n")
    return path


def test_instance_id_and_top_level(session_file):
    trace = parse_session_file(session_file)
    assert trace["instance_id"] == f"claude-{SESSION_ID}"
    assert trace["repo"] is None
    assert trace["base_commit"] is None
    assert trace["cwd"] == CWD


def test_prompts(session_file):
    trace = parse_session_file(session_file)
    prompt_events = [e for e in trace["events"] if e["type"] == "prompt"]
    assert [e["details"]["text"] for e in prompt_events] == [
        "Fix the failing parser test",
        "second prompt as block list",
    ]
    assert [p["text"] for p in trace["prompts"]] == [
        "Fix the failing parser test",
        "second prompt as block list",
    ]
    assert prompt_events[0]["details"]["cwd"] == CWD
    assert prompt_events[0]["timestamp"] == "2026-09-01T10:00:00.000Z"


def test_event_taxonomy(session_file):
    trace = parse_session_file(session_file)
    assert [e["type"] for e in trace["events"]] == [
        "prompt", "prompt", "read", "search", "edit", "edit", "run", "test", "other",
    ]


def test_edit_details(session_file):
    trace = parse_session_file(session_file)
    edit = next(e for e in trace["events"] if e["details"].get("tool") == "Edit")
    assert edit["details"]["file_path"] == "/Users/dev/myproject/parser.py"
    assert edit["details"]["before_content"] == "return None"
    assert edit["details"]["after_content"] == "return result"


def test_truncation(session_file):
    trace = parse_session_file(session_file)
    write = next(e for e in trace["events"] if e["details"].get("tool") == "Write")
    assert len(write["details"]["after_content"]) == MAX_FIELD_CHARS


def test_run_vs_test(session_file):
    trace = parse_session_file(session_file)
    runs = [e for e in trace["events"] if e["type"] in ("run", "test")]
    assert runs[0]["type"] == "run"
    assert runs[0]["details"]["command"] == "ls -la"
    assert runs[1]["type"] == "test"
    assert "pytest" in runs[1]["details"]["command"]


def test_unknown_tool_is_other(session_file):
    trace = parse_session_file(session_file)
    other = next(e for e in trace["events"] if e["type"] == "other")
    assert other["details"]["tool"] == "mcp__linear__save_issue"


def test_iter_traces_and_limit(session_file, tmp_path):
    empty_dir = tmp_path / "-Users-dev-emptyproject"
    empty_dir.mkdir()
    empty_session = empty_dir / "def12345-0000-0000-0000-000000000002.jsonl"
    empty_session.write_text(json.dumps(_line("file-history-snapshot")) + "\n")

    traces = list(iter_traces(root=tmp_path))
    assert len(traces) == 1  # event-less session skipped
    assert traces[0]["instance_id"] == f"claude-{SESSION_ID}"
    assert list(iter_traces(root=tmp_path, limit=0)) == []


def test_records_are_json_serializable(session_file):
    trace = parse_session_file(session_file)
    round_tripped = json.loads(json.dumps(trace))
    assert round_tripped["events"] == trace["events"]
