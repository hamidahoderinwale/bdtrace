"""Tests for the prompt-anchored transcript view."""

import json

from bdtrace.show import render, turns, turns_summary


def _ev(kind, **details):
    return {"type": kind, "timestamp": "2026-09-02T00:00:00Z", "details": details}


RECORD = {
    "instance_id": "claude-abc", "agent": "claude", "cwd": "/w",
    "events": [
        _ev("run", command="setup.sh"),
        _ev("prompt", text="fix the parser"),
        _ev("read", file_path="a.py"), _ev("read", file_path="b.py"),
        _ev("edit", file_path="a.py"),
        _ev("prompt", text="now add a test"),
        _ev("test", command="pytest -q"),
    ],
}


def test_turns_split_on_prompts_and_keep_leading_events():
    t = turns(RECORD)
    assert len(t) == 3
    assert t[0]["prompt"] is None and len(t[0]["events"]) == 1
    assert t[1]["prompt"]["details"]["text"] == "fix the parser"
    assert [e["type"] for e in t[1]["events"]] == ["read", "read", "edit"]
    assert [e["type"] for e in t[2]["events"]] == ["test"]


def test_no_event_is_dropped():
    assert sum(len(t["events"]) + bool(t["prompt"]) for t in turns(RECORD)) == len(RECORD["events"])


def test_render_collapses_a_run_and_shows_its_count():
    out = render(RECORD)
    assert "fix the parser" in out
    assert "read" in out and "×2" in out, out
    assert "claude-abc" in out and "7 events" in out


def test_render_survives_empty_and_detail_less_events():
    out = render({"instance_id": "x", "events": [{"type": "other"}, _ev("prompt")]})
    assert "(empty prompt)" in out
    assert render({"instance_id": "x", "events": []}).startswith("x")


def test_turns_summary_counts_only_prompted_turns(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(RECORD) + "\n")
    s = turns_summary(p)
    assert s == {"traces": 1, "turns": 2,
                 "actions_per_turn": {"min": 1, "median": 3, "max": 3},
                 "action_mix": {"read": 2, "edit": 1, "test": 1}}
