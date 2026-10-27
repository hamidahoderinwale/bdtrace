from bdtrace.compile import compile_record, render, LABELS


def _ev(kind, **details):
    return {"type": kind, "timestamp": "2026-09-02T00:00:00Z", "details": details}


RECORD = {
    "instance_id": "claude-abc", "agent": "claude", "cwd": "/Users/x/w",
    "events": [
        _ev("run", command="setup.sh"),
        _ev("prompt", text="fix the parser"),
        _ev("read", file_path="/Users/x/w/a.py"), _ev("read", file_path="/Users/x/w/b.py"),
        _ev("edit", file_path="/Users/x/w/a.py", tool="Edit"),
        _ev("edit", file_path="/Users/x/w/NOTES.md", tool="Write"),
        _ev("prompt", text="[Request interrupted by user]"),
        _ev("other", tool="Task"),
        _ev("test", command="pytest -q"),
        _ev("run", command="git commit -m x"),
    ],
}


def test_entry_is_typed_and_identifier_free():
    e = compile_record(RECORD)
    assert e["project"] == "w" and e["asks"] == ["fix the parser"]
    assert e["files"] == {"NOTES.md": {"WRITE": 1}, "a.py": {"MODIFY": 1}}
    assert "/Users" not in str(e)
    assert e["commits"] == 1 and e["test_runs"] == 1 and e["interrupts"] == 1
    assert e["reads_before_first_edit"] == 2 and e["distinct_verbs"] == 5


def test_skeleton_uses_frozen_labels():
    e = compile_record(RECORD)
    assert e["skeleton"] == ["run", "prompt", "read", "read", "edit", "edit", "interrupt", "subagent", "test", "run"]
    assert set(e["skeleton"]) <= set(LABELS)


def test_procedure_tier_drops_task_and_files():
    e = compile_record(RECORD, asks=False, files=False)
    assert e["asks"] is None and e["files"] is None and e["skeleton"]


def test_render_is_one_screen():
    assert "claude-abc" in render(compile_record(RECORD))
