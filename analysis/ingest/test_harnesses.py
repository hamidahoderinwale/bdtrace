"""Tests for analysis/ingest/harnesses.py against synthetic fixtures (never live stores).

Provenance of the fixtures, per source (2026-09-02 audit):

- cursor: validated against a real Cursor SQLite store.
- swe_agent: DATA-DERIVED. docs/distillation_run/child_traj/ holds 499 real
  SWE-agent .traj files (SWE-agent-LM-32B on SWE-bench); the fixtures match their
  shape and test_swe_agent_real_corpus_* below parse the real files directly.
- openhands: SHAPE-DERIVED, NOT DATA-DERIVED. No OpenHands trajectory exists in
  this repo, anywhere in its git history, or in docs/distillation_run/ or output/.
  The record shape is reconstructed from what the two adapters that consume the
  real corpus prove they read (scripts/agent_trajectories_paper/openhands_adapter.py
  and fetch_openhands_rawtext.py against nvidia/SWE-Zero-openhands-trajectories):
  a record with instance_id / repo / trajectory, where trajectory is a list of
  OpenAI-chat messages and assistant turns carry
  tool_calls[].function.{name, arguments}, arguments being a JSON string.
  The tool CONTRACT (names, subcommands, parameter names) is corroborated by real
  data: the replay_config embedded in every child_traj/*.traj carries the verbatim
  str_replace_editor tool spec -- command in {view, create, str_replace, insert,
  undo_edit}, parameters path / file_text / old_str / new_str / insert_line /
  view_range. That is the same editor tool OpenHands exposes. What is NOT
  corroborated by any local data is the outer record envelope and the fact that
  OpenHands emits these as native tool_calls (the local .traj corpus uses
  xml_function_calling, so its tool_calls fields are all null).

Measured on the 499 real .traj files (grep-derived; see the module docstring notes
in the real-corpus tests for method): 20,930 trajectory steps + 499 task prompts.

The real-corpus tests skip when the .traj corpus is absent (it is gitignored, so they
run locally and skip in CI). No parser defect was found in this audit; the openhands
gap is coverage, not a known bug.
"""

import hashlib
import json
import sqlite3
import sys
import warnings
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harnesses  # noqa: E402
from harnesses import (  # noqa: E402
    EVENT_TYPES,
    classify_command,
    classify_openhands_tool_call,
    classify_str_replace_editor,
    iter_traces_cursor,
    iter_traces_openhands,
    iter_traces_specstory,
    iter_traces_swe_agent,
    parse,
)

REAL_TRAJ_DIR = Path(__file__).resolve().parents[2] / "docs" / "distillation_run" / "child_traj"


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


REAL_TRAJ_DIR = Path(__file__).resolve().parents[2] / "docs" / "distillation_run" / "child_traj"
real_corpus = pytest.mark.skipif(
    not REAL_TRAJ_DIR.is_dir() or not any(REAL_TRAJ_DIR.glob("*.traj")),
    reason=f"real SWE-agent corpus absent at {REAL_TRAJ_DIR} (see docs/distillation_run/MOVED.md)",
)


@real_corpus
def test_swe_agent_real_corpus_parses_every_file():
    """The 499 local SWE-agent-LM-32B rollouts, parsed as a whole."""
    traces = list(iter_traces_swe_agent(REAL_TRAJ_DIR))
    n_files = len(list(REAL_TRAJ_DIR.glob("*.traj")))
    assert len(traces) == n_files, f"{n_files} files parsed to {len(traces)} traces"
    for trace in traces:
        assert_trace_schema(trace)
    assert len({t["instance_id"] for t in traces}) == len(traces)


@real_corpus
def test_swe_agent_real_corpus_event_mix_is_plausible():
    """A parser that silently mismaps shows up as all-`other` or as empty traces."""
    traces = list(iter_traces_swe_agent(REAL_TRAJ_DIR))
    mix = Counter(e["type"] for t in traces for e in t["events"])
    assert not [t for t in traces if not t["events"]], "some real trajectories parsed to zero events"
    assert mix["other"] < sum(mix.values()) / 2, f"more than half the events are unclassified: {mix}"
    assert mix["prompt"] == len(traces), "every rollout should carry exactly one task prompt"
    assert {"edit", "run"} <= set(mix), f"no edit or run events in a repair corpus: {mix}"


# --- swechat ------------------------------------------------------------------------------
# DATA-DERIVED: column names and turn_type values verified 2026-09-10 against the SALT-NLP/SWE-chat
# dataset card and a live `sessions`/`conversations` preview; the tool-name vocabularies per
# harness (Claude Code `Bash`/`Read`/`Edit`, OpenCode `read`/`bash`/`apply_patch`) were counted
# from the real conversations table. The fixture is synthetic; no live rows are stored here.

def _swechat_fixture(tmp_path: Path) -> Path:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    rows = [
        # Claude Code session: prompt, read, edit, bash test, a tool_result (ignored), a response
        dict(session_id="s1", repo_id="o/r1", user_id="u1", agent="Claude Code", turn_number=0, turn_type="user_prompt",
             timestamp="2026-01-05T13:49:43Z", content="fix the flaky test", model=None, tool_name=None, file_path=None,
             command=None, pattern=None, tool_input_json=None, prompt_intent="debug", prompt_pushback="non_pushback"),
        dict(session_id="s1", repo_id="o/r1", user_id="u1", agent="Claude Code", turn_number=1, turn_type="tool_use",
             timestamp="2026-01-05T13:49:50Z", content=None, model=None, tool_name="Read", file_path="src/a.py",
             command=None, pattern=None, tool_input_json='{"file_path": "src/a.py"}', prompt_intent=None, prompt_pushback=None),
        dict(session_id="s1", repo_id="o/r1", user_id="u1", agent="Claude Code", turn_number=2, turn_type="tool_use",
             timestamp="2026-01-05T13:50:00Z", content=None, model=None, tool_name="Edit", file_path="src/a.py",
             command=None, pattern=None, tool_input_json='{"file_path": "src/a.py", "old_string": "x = 1", "new_string": "x = 2"}',
             prompt_intent=None, prompt_pushback=None),
        dict(session_id="s1", repo_id="o/r1", user_id="u1", agent="Claude Code", turn_number=3, turn_type="tool_result",
             timestamp="2026-01-05T13:50:01Z", content="ok", model=None, tool_name="Edit", file_path=None,
             command=None, pattern=None, tool_input_json=None, prompt_intent=None, prompt_pushback=None),
        dict(session_id="s1", repo_id="o/r1", user_id="u1", agent="Claude Code", turn_number=4, turn_type="tool_use",
             timestamp="2026-01-05T13:50:10Z", content=None, model=None, tool_name="Bash", file_path=None,
             command="pytest tests/test_a.py", pattern=None, tool_input_json='{"command": "pytest tests/test_a.py"}',
             prompt_intent=None, prompt_pushback=None),
        dict(session_id="s1", repo_id="o/r1", user_id="u1", agent="Claude Code", turn_number=5, turn_type="assistant_response",
             timestamp="2026-01-05T13:50:20Z", content="done", model="claude-opus-4-6", tool_name=None, file_path=None,
             command=None, pattern=None, tool_input_json=None, prompt_intent=None, prompt_pushback=None),
        # OpenCode session: lowercase tool names fold to the same verbs
        dict(session_id="s2", repo_id="o/r2", user_id=None, agent="OpenCode", turn_number=0, turn_type="user_prompt",
             timestamp="2026-02-01T09:00:00Z", content="add logging", model=None, tool_name=None, file_path=None,
             command=None, pattern=None, tool_input_json=None, prompt_intent="create new code", prompt_pushback=None),
        dict(session_id="s2", repo_id="o/r2", user_id=None, agent="OpenCode", turn_number=1, turn_type="tool_use",
             timestamp="2026-02-01T09:00:05Z", content=None, model=None, tool_name="grep", file_path=None,
             command=None, pattern="logger", tool_input_json='{"pattern": "logger"}', prompt_intent=None, prompt_pushback=None),
        dict(session_id="s2", repo_id="o/r2", user_id=None, agent="OpenCode", turn_number=2, turn_type="tool_use",
             timestamp="2026-02-01T09:00:09Z", content=None, model=None, tool_name="apply_patch", file_path="lib/log.py",
             command=None, pattern=None, tool_input_json='{"patch": "*** Update File: lib/log.py"}', prompt_intent=None, prompt_pushback=None),
        dict(session_id="s2", repo_id="o/r2", user_id=None, agent="OpenCode", turn_number=3, turn_type="tool_use",
             timestamp="2026-02-01T09:00:15Z", content=None, model=None, tool_name="bash", file_path=None,
             command="git status", pattern=None, tool_input_json='{"command": "git status"}', prompt_intent=None, prompt_pushback=None),
    ]
    table = pa.Table.from_pylist(rows)
    d = tmp_path / "swechat"; d.mkdir()
    pq.write_table(table, d / "conversations.parquet")
    return d


def test_swechat_folds_tool_names_to_verbs_and_keeps_identity_in_labels(tmp_path):
    from analysis.ingest.harnesses import parse, swechat_event_type
    traces = list(parse("swechat", _swechat_fixture(tmp_path)))
    assert [t["instance_id"] for t in traces] == ["swechat-s1", "swechat-s2"]
    s1, s2 = traces
    assert s1["agent"] == "Claude Code" and s2["agent"] == "OpenCode"
    assert s1["repo"] == "o/r1" and s1["labels"] == {"user_id": "u1", "session_id": "s1", "model": "claude-opus-4-6"}
    assert [e["type"] for e in s1["events"]] == ["prompt", "read", "edit", "test"], "tool_result and assistant rows are not events"
    assert s1["events"][2]["details"]["before_content"] == "x = 1" and s1["events"][2]["details"]["after_content"] == "x = 2"
    assert s1["prompts"][0]["details"]["prompt_intent"] == "debug"
    assert [e["type"] for e in s2["events"]] == ["prompt", "search", "edit", "run"]
    assert s2["labels"]["user_id"] is None and s2["labels"]["model"] is None
    assert swechat_event_type("Read", None) == swechat_event_type("read", None) == swechat_event_type("read_file", None) == "read"
    assert swechat_event_type("mcp__anything__tool", None) == "other"


def test_swechat_limit_stops_at_whole_sessions(tmp_path):
    from analysis.ingest.harnesses import parse
    traces = list(parse("swechat", _swechat_fixture(tmp_path), limit=1))
    assert len(traces) == 1 and traces[0]["instance_id"] == "swechat-s1"


# --- specstory ----------------------------------------------------------------------------
# DATA-DERIVED: the three fixture layouts reproduce six real .specstory/history exports read
# on 2026-09-11 (two Claude Code and two Cursor Markdown v2.1.0 files, one early Cursor file
# with bare `Tool use:` lines, one sidechain-only file with no user turn and no tool call);
# test_specstory_real_samples parses those files directly when the local cache holds them.
# Fixture paths live under a fake home; no live export is stored here.

SPECSTORY_SAMPLES = Path.home() / ".cache" / "trace-funnel-paper" / "specstory" / "samples"

SPECSTORY_CLAUDE_CODE = """<!-- Generated by SpecStory, Markdown v2.1.0 -->

# 2026-07-13 13:40:49Z

<!-- Claude Code Session 5c7d5c23-0691-4467-b98e-0fb865b4a639 (2026-07-13 13:40:49Z) -->

_**User (2026-07-13 13:40:49Z)**_

fix the flaky test in app.py

---

_**Agent (claude-opus-4-8 2026-07-13 13:40:56Z)**_

I'll look at the test first.

_**Agent (claude-opus-4-8 2026-07-13 13:41:01Z)**_

<tool-use data-tool-type="shell" data-tool-name="Bash"><details>
<summary>Tool use: **Bash**</summary>
Run the suite

`pytest tests/ -q`

```text
1 failed, 3 passed
```
</details></tool-use>

_**Agent (claude-opus-4-8 2026-07-13 13:41:05Z)**_

<tool-use data-tool-type="read" data-tool-name="Read"><details>
<summary>Tool use: **Read**</summary>
 `./src/app.py`

```text
1	x = 1
```
</details></tool-use>

_**Agent (claude-opus-4-8 2026-07-13 13:41:20Z)**_

<tool-use data-tool-type="shell" data-tool-name="Bash"><details>
<summary>Tool use: **Bash**</summary>
Find callers

```bash
cd /home/dev/widgets
grep -rn "flaky" src/
```

```text
src/app.py:3:# flaky
```
</details></tool-use>

_**Agent (claude-opus-4-8 2026-07-13 13:41:40Z)**_

<tool-use data-tool-type="write" data-tool-name="Edit"><details>
<summary>Tool use: **Edit**</summary>

```text
The file /home/dev/widgets/src/app.py has been updated successfully. (file state is current in your context — no need to Read it back)
```
</details></tool-use>

_**Agent (claude-opus-4-8 2026-07-13 13:42:00Z)**_

<tool-use data-tool-type="write" data-tool-name="Write"><details>
<summary>Tool use: **Write**</summary>

```md
# Plan

- step one
```

```text
File created successfully at: /home/dev/.claude/plans/plan.md
```
</details></tool-use>

_**Agent (claude-opus-4-8 2026-07-13 13:42:10Z)**_

<tool-use data-tool-type="unknown" data-tool-name="ToolSearch"><details>
<summary>Tool use: **ToolSearch**</summary>

**Input:**

- query: `select:ExitPlanMode`

**Result:**

```

```
</details></tool-use>

_**Agent (claude-opus-4-8 2026-07-13 13:42:30Z)**_

Done. The test passes now; to repeat it:

```bash
_**User**_
pytest tests/ -q
```

---

_**User (2026-07-13 13:46:28Z)**_

thanks, also run it twice

---
"""

SPECSTORY_CURSOR_NEW = """<!-- Generated by SpecStory, Markdown v2.1.0 -->

<!-- cursor Session 227b9a24-168c-4d87-810b-9ed83e62101e (2025-12-02 12:49Z) -->

# Bunch of updates (2025-12-02 12:49Z)

_**User (2025-12-02 12:49Z)**_

remove smithery entirely

---

_**Agent (model claude-4.5-opus-high-thinking, mode Agent)**_

<think><details><summary>Thought Process</summary>
The user wants smithery gone. Let me grep for it.
Tool use: **read_file**
</details></think>

---

<tool-use data-tool-type="read" data-tool-name="read_file">
<details><summary>Tool use: **read_file** • Read file: /home/dev/widgets/README.md</summary>

</details>
</tool-use>

---

<tool-use data-tool-type="search" data-tool-name="grep">
<details>
<summary>Tool use: **grep** • Grep for "smithery" • 2 matches</summary>

Output mode: files_with_matches

| File |
|------|
| `./src/types.ts` |

</details>
</tool-use>

---

Let me remove it now.

---

<tool-use data-tool-type="task" data-tool-name="todo_write">
<details><summary>Tool use: **todo_write** • Todo List</summary>

- [-] Remove smitheryApiKey from types.ts

</details>
</tool-use>

---

<tool-use data-tool-type="write" data-tool-name="search_replace">

<details><summary>Tool use: **code_edit** • Edit file: /home/dev/widgets/src/types.ts</summary>

**Chunk 1**
Lines added: undefined, lines removed: 1

```diff
@@ -12,5 +12,4 @@
    langwatchApiKey: string;
-   smitheryApiKey?: string;
    projectGoal: string;
```

</details>

</tool-use>

---

<tool-use data-tool-type="shell" data-tool-name="run_terminal_cmd">
<details><summary>Tool use: **command** • Run command: cd /home/dev/widgets &amp;&amp; pytest -q</summary>

```bash
cd /home/dev/widgets && pytest -q
```

```
4 passed
```

</details>
</tool-use>

---

_**User (2025-12-03 05:21Z)**_

dont change changelog by hand

---

_**Agent (model claude-4.5-opus-high-thinking, mode Agent)**_

Got it.

---
"""

SPECSTORY_CURSOR_OLD = """<!-- Generated by SpecStory -->

<!-- cursor Session 198f391d-8ae2-4788-975c-8c8e23345506 (2025-10-01 23:30Z) -->

# Avatare fliegen nicht mehr (2025-10-01 23:30Z)

_**User (2025-10-01 23:30Z)**_

die Avatare fliegen nicht mehr

---

_**Assistant (claude-4.5-sonnet-thinking)**_

<think><details><summary>Thought Process</summary>Ich schaue mir script.js an.</details></think>

---

Ich schaue mir den Code an.

---

Tool use: **read_file**

Read file: /home/dev/mastowall/script.js

---

Tool use: **grep**

<details>
<summary>Grep for "representativeNode" in "/home/dev/mastowall/script.js" • 5 matches</summary>

Output mode: content

| Content | Line |
|------|------|
| `representativeNode: null` | L109 |

</details>

---

Tool use: **run_terminal_cmd**

```bash
cd /home/dev/mastowall && grep -n "avatar-img" script.js
```

```
145: <img class="avatar-img">
```

---

Tool use: **search_replace**


<details><summary>Edit file: /home/dev/mastowall/script.js</summary>

**Chunk 1**
Lines added: 1, lines removed: 1

```diff
@@ -159,7 +159,7 @@
-     if (authorData.has(authorId)) {
+     if (avatarNode && authorData.has(authorId)) {
```

</details>

---

Tool use: **search_replace**


<details><summary>Edit file: /home/dev/mastowall/styles.css</summary>
Tool use: **read_lints**

<details>
          <summary>Read lints for 1 file</summary>

Lint paths:

- `/home/dev/mastowall/script.js`

Lint results

**No lint errors found**
</details>

---

Ich habe Debug-Logs hinzugefügt.

---

_**User (2025-10-01 23:30Z)**_



---

_**Assistant (claude-4.5-sonnet-thinking)**_

Bitte lade die Seite neu.

---
"""

SPECSTORY_EARLIEST = """<!-- Generated by SpecStory -->

# Untitled (2025-04-24 13:10:01)

_**User**_

Can you push this to my acme/widgets repository?

---

_**Assistant**_



---



---
"""


def _specstory_fixture(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    d = raw / "acme__widgets"
    d.mkdir(parents=True)
    (d / "2026-07-13_13-40-49Z-fix-the-flaky.md").write_text(SPECSTORY_CLAUDE_CODE)
    (d / "2025-12-02_12-49Z-bunch-of-updates.md").write_text(SPECSTORY_CURSOR_NEW)
    (d / "2025-10-01_23-30Z-avatare.md").write_text(SPECSTORY_CURSOR_OLD)
    (d / "2025-04-24_13-10-01-untitled.md").write_text(SPECSTORY_EARLIEST)
    return raw


def _by_agent_model(traces: list[dict]) -> dict:
    return {(t["agent"], t["labels"]["model"]): t for t in traces}


def test_specstory_three_layouts(tmp_path):
    traces = list(iter_traces_specstory(_specstory_fixture(tmp_path)))
    assert len(traces) == 4
    for t in traces:
        assert t["repo"] == "acme/widgets" and t["base_commit"] is None
        assert t["labels"]["repo"] == "acme/widgets"
    by = _by_agent_model(traces)

    cc = by[("Claude Code", "claude-opus-4-8")]
    assert cc["instance_id"] == "specstory-5c7d5c23-0691-4467-b98e-0fb865b4a639"
    assert cc["labels"] == {"session_id": "5c7d5c23-0691-4467-b98e-0fb865b4a639", "model": "claude-opus-4-8",
                            "specstory_version": "2.1.0", "repo": "acme/widgets", "format": "v2",
                            "tool_markers": ["Bash", "Edit", "Read", "ToolSearch", "Write"]}
    assert [e["type"] for e in cc["events"]] == ["prompt", "test", "read", "search", "edit", "edit", "other", "prompt"]
    ev = cc["events"]
    assert ev[0]["details"]["text"] == "fix the flaky test in app.py" and ev[0]["timestamp"] == "2026-07-13 13:40:49Z"
    assert ev[1]["details"] == {"tool": "Bash", "command": "pytest tests/ -q"}
    assert ev[2]["details"] == {"tool": "Read", "file_path": "./src/app.py"}
    assert ev[3]["details"]["command"] == 'cd /home/dev/widgets\ngrep -rn "flaky" src/'
    assert ev[4]["details"] == {"tool": "Edit", "file_path": "/home/dev/widgets/src/app.py"}
    assert ev[5]["details"]["file_path"] == "/home/dev/.claude/plans/plan.md"
    assert ev[5]["details"]["after_content"].startswith("# Plan")
    assert ev[6]["details"] == {"tool": "ToolSearch"}
    assert ev[7]["details"]["text"] == "thanks, also run it twice", "a turn header inside a fenced block is not a turn"
    assert cc["prompts"] == [ev[0], ev[7]]

    cur = by[("Cursor", "claude-4.5-opus-high-thinking")]
    assert cur["instance_id"] == "specstory-227b9a24-168c-4d87-810b-9ed83e62101e"
    assert cur["labels"]["specstory_version"] == "2.1.0" and cur["labels"]["format"] == "v2"
    assert cur["labels"]["tool_markers"] == ["grep", "read_file", "run_terminal_cmd", "search_replace", "todo_write"]
    assert [e["type"] for e in cur["events"]] == ["prompt", "read", "search", "other", "edit", "test", "prompt"], \
        "think blocks (even ones quoting a `Tool use:` line) and agent prose are not events"
    ev = cur["events"]
    assert ev[1]["details"] == {"tool": "read_file", "file_path": "/home/dev/widgets/README.md"}
    assert ev[2]["details"] == {"tool": "grep", "query": "smithery"}
    assert ev[3]["details"] == {"tool": "todo_write"}
    assert ev[4]["details"]["tool"] == "search_replace" and ev[4]["details"]["file_path"] == "/home/dev/widgets/src/types.ts"
    assert "-   smitheryApiKey?: string;" in ev[4]["details"]["diff"]
    assert ev[5]["details"]["command"] == "cd /home/dev/widgets && pytest -q", "summary text is HTML-unescaped"
    assert ev[6]["timestamp"] == "2025-12-03 05:21Z"

    old = by[("Cursor", "claude-4.5-sonnet-thinking")]
    assert old["labels"]["specstory_version"] is None and old["labels"]["session_id"] == "198f391d-8ae2-4788-975c-8c8e23345506"
    assert old["labels"]["format"] == "v2", "a header comment marks the file v2 even without <tool-use> tags"
    assert [e["type"] for e in old["events"]] == ["prompt", "read", "search", "search", "edit", "edit", "read"], \
        "bare `Tool use:` lines map by tool name; an empty user turn is not a prompt"
    ev = old["events"]
    assert ev[1]["details"] == {"tool": "read_file", "file_path": "/home/dev/mastowall/script.js"}
    assert ev[2]["details"] == {"tool": "grep", "query": "representativeNode", "file_path": "/home/dev/mastowall/script.js"}
    assert ev[3]["details"]["command"] == 'cd /home/dev/mastowall && grep -n "avatar-img" script.js'
    assert ev[4]["details"]["file_path"] == "/home/dev/mastowall/script.js" and "avatarNode &&" in ev[4]["details"]["diff"]
    assert ev[5]["details"] == {"tool": "search_replace", "file_path": "/home/dev/mastowall/styles.css"}, "a truncated block still yields its event"
    assert ev[6]["details"]["tool"] == "read_lints"

    earliest = by[("unknown", None)]
    assert earliest["labels"] == {"session_id": None, "model": None, "specstory_version": None, "repo": "acme/widgets",
                                  "format": "v2", "tool_markers": []}
    expected = hashlib.sha1(b"acme/widgets/2025-04-24_13-10-01-untitled.md").hexdigest()
    assert earliest["instance_id"] == f"specstory-{expected}"
    assert [e["type"] for e in earliest["events"]] == ["prompt"] and "timestamp" not in earliest["events"][0]


def test_specstory_malformed_files_are_skipped_with_a_warning(tmp_path):
    raw = _specstory_fixture(tmp_path)
    (raw / "acme__widgets" / "broken.md").write_bytes(b"\xff\xfe\x00 not utf-8")
    (raw / "acme__widgets" / "notes.md").write_text("# just a markdown file\n\nno header comment, no `## SpecStory` heading, no turn\n")
    with pytest.warns(UserWarning, match="specstory: skipped malformed file") as record:
        traces = list(iter_traces_specstory(raw))
    assert len(traces) == 4
    assert {str(w.message).split()[4] for w in record} == {"broken.md", "notes.md"}
    assert any("2 skipped so far" in str(w.message) for w in record)


def test_specstory_single_file_limit_and_parse_dispatch(tmp_path):
    raw = _specstory_fixture(tmp_path)
    one = list(iter_traces_specstory(raw / "acme__widgets" / "2025-10-01_23-30Z-avatare.md"))
    assert len(one) == 1 and one[0]["agent"] == "Cursor" and one[0]["repo"] == "acme/widgets"
    assert len(list(iter_traces_specstory(raw, limit=2))) == 2
    with pytest.raises(FileNotFoundError):
        list(iter_traces_specstory(tmp_path / "nope"))
    traces = list(parse("specstory", raw))
    assert len(traces) == 4
    for t in traces:
        # base shape, minus the swechat-precedent extras: top-level agent/labels and per-event timestamp
        base = {k: v for k, v in t.items() if k not in ("agent", "labels")}
        base["events"] = [{"type": e["type"], "details": e["details"]} for e in t["events"]]
        assert_trace_schema(base)
        assert all(set(e) <= {"type", "details", "timestamp"} for e in t["events"])
        assert set(t) == {"instance_id", "repo", "base_commit", "events", "prompts", "agent", "labels"}
        assert set(t["labels"]) == {"session_id", "model", "specstory_version", "repo", "format", "tool_markers"}
        assert all(p["type"] == "prompt" for p in t["prompts"])
    # a lone file outside the <owner__repo> layout has no repo and a path-derived id
    lone = tmp_path / "2026-07-13_13-40-49Z-fix-the-flaky.md"
    lone.write_text(SPECSTORY_EARLIEST)
    (t,) = iter_traces_specstory(lone)
    assert t["repo"] is None and t["labels"]["repo"] is None
    assert t["instance_id"] == "specstory-" + hashlib.sha1(f"None/{lone.name}".encode()).hexdigest()


@pytest.mark.skipif(not SPECSTORY_SAMPLES.is_dir() or not any(SPECSTORY_SAMPLES.glob("*.md")),
                    reason=f"SpecStory samples absent at {SPECSTORY_SAMPLES}")
def test_specstory_real_samples():
    """The six real exports: every record carries events, the two big Cursor sessions carry edits,
    the harness comment is read for all four Markdown v2.1.0 files, and nothing warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        per_file = {p.name: list(iter_traces_specstory(p)) for p in sorted(SPECSTORY_SAMPLES.glob("*.md"))}
    assert not [str(w.message) for w in caught]
    records = [t for ts in per_file.values() for t in ts]
    assert records and all(t["events"] for t in records)
    assert len({t["instance_id"] for t in records}) == len(records)
    for t in records:
        assert set(t) == {"instance_id", "repo", "base_commit", "events", "prompts", "agent", "labels"}
        assert all(e["type"] in EVENT_TYPES for e in t["events"])
    for name, harness in (("big_1.md", "Claude Code"), ("big_3.md", "Claude Code"), ("big_2.md", "Cursor"), ("big_4.md", "Cursor")):
        if name in per_file:
            (t,) = per_file[name]
            assert t["agent"] == harness and t["labels"]["session_id"] and t["labels"]["model"]
    for name in ("big_2.md", "big_4.md"):
        if name in per_file:
            (t,) = per_file[name]
            edits = [e for e in t["events"] if e["type"] == "edit"]
            assert edits and all(e["details"].get("file_path") for e in edits)
            assert any(e["details"].get("diff") for e in edits)


# The early 2025 layout: no <tool-use> tags; each tool action is its own section between `---`
# (and, in the oldest files, `_****_`) separators. Shapes and their corpus counts were read from
# 1,644 harvested files on 2026-09-11; paths here are fake.

SPECSTORY_V1_OLDEST = """## SpecStory

## Setting Up The Parking Page (3/11/2025, 8:11:06 PM)

_**User**_

make the svg full screen

---

_**Assistant**_

I'll read the CSS file first.

---

_****_

Read file: src/App.css

---

_****_

Read file: undefined

---

_****_

<details>
            <summary>Listed current directory • **2** results</summary>

| Name |
|-------|
| 📁 `src` |
| 📄 `package.json` |

</details>

---

_****_

<details>
            <summary>Searched codebase "How is the svg sized?" • **3** results</summary>

| File | Lines |
|------|-------|
| `src/App.tsx` | L1-20 |

</details>

---

_****_

<details>
<summary>Grep search for "parking-container" • **2** files</summary>

| File | Line | Match |
|------|------|-------|
| `src/App.css` | L12 | `.parking-container {` |

</details>

---

_****_

<details><summary>Tool use: **edit_file** • Edit file: src/App.css</summary>

**Chunk 1**
Lines added: 2, lines removed: 1

```diff
@@ -1,3 +1,4 @@
  :root {
-   --primary-color: #2dd4bf;
+   --text-color: #2055a4;
+   --bg: white;
  }
```

</details>

---

_****_

```diff
  body {
-   margin: 1px;
+   margin: 0;
  }
```

Lines added: 1, lines removed: 1

---

_****_

Now install the dependency:

---

_****_

Cancelled

---

_****_

```bash
npm install express dotenv
```

```
added 2 packages
```

---

_****_

Tool call timed out after 5000ms

---

_****_

To run it yourself:

```bash
npm run dev
```

---

_**User**_



---

_**User**_

now the layout

---

_**Assistant**_

```diff
  a
- b
+ c
```

---
"""

SPECSTORY_V1_WITH_HEADER = """<!-- Generated by SpecStory -->

# 查找插件系统日历功能 (2025-04-30 13:02:42)

_**User**_

查找下我的插件系统里面的日历功能

---

_**Assistant**_



---

我明白您的意思了。让我来实现这个解决方案。

---

<details>
            <summary>Listed directory src/renderer/src/plugins • **4** results</summary>

| Name |
|-------|
| 📄 `index.ts` |

</details>

---

Read file: src/renderer/src/plugins/index.ts

---

```diff
  import SimpleTextTools from './SimpleTextTools'
+ import Calendar from './Calendar'
```

---

Error calling tool.

---

<details><summary>Tool use: **read_file** • Read file: src/renderer/src/plugins/Calendar.ts</summary>

</details>

---

_**Agent (mode Agent)**_

完成了。

---
"""


def test_specstory_early_layout_sections(tmp_path):
    raw = tmp_path / "raw"
    d = raw / "acme__parking"
    d.mkdir(parents=True)
    (d / "2025-03-11_20-11-setting-up.md").write_text(SPECSTORY_V1_OLDEST)
    (d / "2025-04-30_05-02-plugins.md").write_text(SPECSTORY_V1_WITH_HEADER)
    oldest, with_header = list(iter_traces_specstory(raw))

    assert oldest["agent"] == "unknown" and oldest["repo"] == "acme/parking"
    assert oldest["labels"] == {"session_id": None, "model": None, "specstory_version": None, "repo": "acme/parking",
                                "format": "v1", "tool_markers": ["codebase_search", "edit_file", "grep_search",
                                                                 "list_dir", "read_file", "run_terminal_cmd"]}
    assert [e["type"] for e in oldest["events"]] == \
        ["prompt", "read", "read", "search", "search", "search", "edit", "edit", "run", "prompt", "edit"], \
        "prose sections, fences inside prose sections and an empty user turn are not events"
    ev = oldest["events"]
    assert len(oldest["prompts"]) == 2, "one prompt per user turn with text"
    assert ev[1]["details"] == {"tool": "read_file", "file_path": "src/App.css"}
    assert ev[2]["details"] == {"tool": "read_file"}, "`Read file: undefined` is a read with no path"
    assert ev[3]["details"] == {"tool": "list_dir"}
    assert ev[4]["details"] == {"tool": "codebase_search", "query": "How is the svg sized?"}
    assert ev[5]["details"] == {"tool": "grep_search", "query": "parking-container"}
    assert ev[6]["details"]["tool"] == "edit_file" and ev[6]["details"]["file_path"] == "src/App.css"
    assert "+   --text-color: #2055a4;" in ev[6]["details"]["diff"]
    assert ev[7]["details"]["file_path"] == "src/App.css", "a bare diff takes the path of the turn's last Edit file"
    assert "+   margin: 0;" in ev[7]["details"]["diff"] and ev[7]["details"]["outcome"] == "Cancelled"
    assert ev[8]["details"] == {"tool": "run_terminal_cmd", "command": "npm install express dotenv",
                                "outcome": "Tool call timed out after 5000ms"}
    assert ev[10]["details"] == {"tool": "edit_file", "diff": "  a\n- b\n+ c\n"}, "no Edit file in this turn: no path"
    assert all("timestamp" not in e for e in ev)

    assert with_header["labels"]["format"] == "v2" and with_header["labels"]["specstory_version"] is None
    assert with_header["labels"]["model"] is None, "`(mode Agent)` is not a model"
    assert with_header["labels"]["tool_markers"] == ["edit_file", "list_dir", "read_file"]
    assert [e["type"] for e in with_header["events"]] == ["prompt", "search", "read", "edit", "read"]
    ev = with_header["events"]
    assert ev[1]["details"] == {"tool": "list_dir", "file_path": "src/renderer/src/plugins"}
    assert ev[2]["details"] == {"tool": "read_file", "file_path": "src/renderer/src/plugins/index.ts"}
    assert ev[3]["details"]["outcome"] == "Error calling tool." and "file_path" not in ev[3]["details"]
    assert ev[4]["details"] == {"tool": "read_file", "file_path": "src/renderer/src/plugins/Calendar.ts"}


SPECSTORY_RAW = Path.home() / ".cache" / "trace-funnel-paper" / "specstory" / "raw"


@pytest.mark.skipif(not SPECSTORY_RAW.is_dir() or not any(SPECSTORY_RAW.rglob("*.md")),
                    reason=f"SpecStory harvest absent at {SPECSTORY_RAW}")
def test_specstory_real_harvest_first_300_files():
    """The harvested corpus, first 300 files by path: under 2% skipped, and more than half of the
    records carry at least one tool event (a section walk that misses a layout shows up here)."""
    files = sorted(SPECSTORY_RAW.rglob("*.md"))[:300]
    records, skipped = [], 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for f in files:
            records.extend(iter_traces_specstory(f))
    skipped = sum(1 for w in caught if "skipped malformed file" in str(w.message))
    assert skipped / len(files) < 0.02, f"{skipped} of {len(files)} files skipped"
    assert records
    with_tools = sum(1 for t in records if any(e["type"] != "prompt" for e in t["events"]))
    assert with_tools / len(records) > 0.5, f"only {with_tools} of {len(records)} records have a tool event"
    for t in records:
        assert t["repo"] and "/" in t["repo"], "repo is read back from the <owner__repo> directory"
        assert set(t["labels"]) == {"session_id", "model", "specstory_version", "repo", "format", "tool_markers"}
