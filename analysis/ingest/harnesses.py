"""Ingest other agent harnesses' local trace stores into standardized trace records.

Sources:
- cursor: Cursor's local SQLite stores (state.vscdb, tables cursorDiskKV/ItemTable)
  or raw text dumps of them (prompts_raw.txt / conversations_raw.txt, the shape
  scripts/parse_to_traces.py reads; its helpers are ported here).
- swe_agent: SWE-agent .traj JSON files (the shape distillation_run/child_traj/*.traj
  uses and scripts/export_traces_for_sessiongrep.py reads).
- openhands: OpenHands trajectory records (the nvidia/SWE-Zero-openhands-trajectories
  shape scripts/agent_trajectories_paper/openhands_adapter.py maps; its classify
  heuristic is ported here), as local .json/.jsonl files.
- specstory: SpecStory session exports (.specstory/history/*.md written inside a repo
  by the Cursor / Claude Code extension), one record per file.

Output records match output/resolved_traces_lite_full.jsonl rows:
{"instance_id", "repo", "base_commit", "events": [{"type", "details"}], "prompts"}
with event types from the taxonomy: prompt/edit/read/search/run/test/other
(representations/encoders/tokens.py consumes event "type" and, for edit events,
details.after_content/before_content).

Usage:
    python -m analysis.ingest.harnesses --source cursor \
        --input ~/Library/Application\\ Support/Cursor/User --output traces.jsonl
"""

import argparse
import hashlib
import html
import json
import re
import sqlite3
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

EVENT_TYPES = {"prompt", "edit", "read", "search", "run", "test", "other"}


def make_trace(
    instance_id: str,
    repo: str | None,
    base_commit: str | None,
    events: list[dict],
    prompts: list[dict],
) -> dict:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "events": events,
        "prompts": prompts,
    }


def _event(etype: str, details: dict) -> dict:
    assert etype in EVENT_TYPES, etype
    return {"type": etype, "details": details}


# Command classification, ported from scripts/agent_trajectories_paper/openhands_adapter.py
# classify() and extended to str_replace_editor CLI-style actions in SWE-agent .traj files.

TEST_MARKERS = ("pytest", "unittest", "tox", "nosetests")
SEARCH_MARKERS = ("grep", "find ", "rg ", "locate", "ls ", "glob")
READ_MARKERS = ("cat ", "head ", "tail ", "less ")


def classify_command(command: str) -> str:
    c = command.strip().lower()
    if not c:
        return "other"
    if any(k in c for k in TEST_MARKERS) or ("python" in c and "test" in c):
        return "test"
    first = c.split()[0]
    if first in ("grep", "rg", "find", "locate", "ls", "ack", "ag") or any(
        k in c for k in SEARCH_MARKERS
    ):
        return "search"
    if first in ("cat", "head", "tail", "less", "more") or any(
        k in c for k in READ_MARKERS
    ):
        return "read"
    return "run"


def classify_str_replace_editor(subcommand: str) -> str:
    sub = subcommand.strip().lower()
    if sub == "view":
        return "read"
    if sub in ("create", "str_replace", "insert", "edit", "undo_edit"):
        return "edit"
    return "other"


def classify_openhands_tool_call(name: str, raw_args: Any) -> str:
    try:
        a = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        a = {}
    if not isinstance(a, dict):
        a = {}
    if name == "str_replace_editor":
        return classify_str_replace_editor(str(a.get("command", "")))
    if name == "execute_bash":
        return classify_command(str(a.get("command", "")))
    if name in ("finish", "submit"):
        return "other"
    return "other"


# Cursor source.

# Ported from scripts/parse_to_traces.py parse_raw_json_value().
def parse_raw_json_value(raw_text: str) -> Any:
    raw_text = raw_text.strip()
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            results = []
            for line in raw_text.split("\n"):
                line = line.strip()
                if line:
                    results.append(json.loads(line))
            return results if len(results) > 1 else results[0] if results else None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


CURSOR_TOOL_TYPES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(edit_file|search_replace|write|apply_patch|multiedit)"), "edit"),
    (re.compile(r"^(read_file|open_file|read_lints)"), "read"),
    (re.compile(r"^(ripgrep|grep|glob_file_search|codebase_search|file_search|list_dir|search)"), "search"),
    (re.compile(r"^(run_terminal_cmd|run_terminal_command|terminal|bash)"), "run"),
]


def classify_cursor_tool(name: str, raw_args: Any) -> str:
    n = (name or "").lower()
    for pattern, etype in CURSOR_TOOL_TYPES:
        if pattern.search(n):
            if etype == "run":
                try:
                    a = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (json.JSONDecodeError, TypeError, ValueError):
                    a = {}
                cmd = str(a.get("command", "")) if isinstance(a, dict) else ""
                return classify_command(cmd) if cmd else "run"
            return etype
    return "other"


def _cursor_tool_details(tool: dict) -> dict:
    details: dict = {"tool_name": tool.get("name")}
    raw_args = tool.get("rawArgs")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        args = {}
    if isinstance(args, dict):
        path = args.get("path") or args.get("target_file") or args.get("file_path")
        if path:
            details["file_path"] = path
        if args.get("command"):
            details["command"] = args["command"]
        query = args.get("query") or args.get("pattern") or args.get("globPattern")
        if query:
            details["query"] = query
        if args.get("code_edit"):
            details["after_content"] = args["code_edit"]
        if args.get("new_string"):
            details["after_content"] = args["new_string"]
        if args.get("old_string"):
            details["before_content"] = args["old_string"]
    return details


def _cursor_bubble_events(bubble: dict) -> list[dict]:
    events = []
    text = (bubble.get("text") or "").strip()
    btype = bubble.get("type")
    if btype == 1 and text:
        events.append(_event("prompt", {"text": text}))
    tool = bubble.get("toolFormerData")
    if isinstance(tool, dict) and tool.get("name"):
        etype = classify_cursor_tool(tool.get("name", ""), tool.get("rawArgs"))
        events.append(_event(etype, _cursor_tool_details(tool)))
    if btype == 2 and text and not events:
        events.append(_event("other", {"role": "assistant", "text": text[:2000]}))
    return events


def _iter_cursor_sqlite(db_path: Path, limit: int | None) -> Iterator[dict]:
    """Read composers + bubbles from one state.vscdb, strictly read-only."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
        if "cursorDiskKV" not in tables:
            return
        composers = {}
        for key, value in cur.execute(
            "select key, value from cursorDiskKV where key like 'composerData:%'"
        ):
            try:
                composers[key.split(":", 1)[1]] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
        bubbles: dict[str, dict[str, dict]] = {}
        for key, value in cur.execute(
            "select key, value from cursorDiskKV where key like 'bubbleId:%'"
        ):
            parts = key.split(":")
            if len(parts) != 3:
                continue
            _, composer_id, bubble_id = parts
            try:
                bubbles.setdefault(composer_id, {})[bubble_id] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
        n = 0
        for composer_id in sorted(set(composers) | set(bubbles)):
            data = composers.get(composer_id, {})
            cbubbles = bubbles.get(composer_id, {})
            headers = data.get("fullConversationHeadersOnly") or []
            ordered_ids = [h.get("bubbleId") for h in headers if h.get("bubbleId") in cbubbles]
            remaining = [b for b in cbubbles if b not in set(ordered_ids)]
            remaining.sort(key=lambda b: cbubbles[b].get("createdAt") or "")
            events: list[dict] = []
            prompts: list[dict] = []
            for bubble_id in ordered_ids + remaining:
                bubble = cbubbles[bubble_id]
                for event in _cursor_bubble_events(bubble):
                    events.append(event)
                    if event["type"] == "prompt":
                        prompts.append({"text": event["details"]["text"]})
            first_text = (data.get("text") or "").strip()
            if first_text and not prompts:
                prompts.append({"text": first_text})
                events.insert(0, _event("prompt", {"text": first_text}))
            if not events:
                continue
            yield make_trace(f"cursor-{composer_id}", None, None, events, prompts)
            n += 1
            if limit is not None and n >= limit:
                return
    finally:
        con.close()


def _iter_cursor_raw_export(export_dir: Path, limit: int | None) -> Iterator[dict]:
    """Raw text dumps, the scripts/parse_to_traces.py input shape."""

    def load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        data = parse_raw_json_value(path.read_text())
        if isinstance(data, dict):
            data = data.get("conversations", [data])
        return [d for d in (data or []) if isinstance(d, dict)]

    def build(workspace_id: str, prompts_raw: list[dict], conversations: list[dict]) -> dict | None:
        events: list[dict] = []
        prompts: list[dict] = []
        for prompt in prompts_raw:
            text = prompt.get("text") or prompt.get("prompt", "")
            details = {"text": text}
            if "context" in prompt:
                details["context"] = prompt["context"]
            events.append(_event("prompt", details))
            prompts.append({"text": text})
        for conv in conversations:
            for msg in conv.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown")
                details = {
                    "role": role,
                    "text": str(msg.get("content", ""))[:4000],
                    "conversation_id": conv.get("id", ""),
                }
                if role == "user":
                    events.append(_event("prompt", {"text": details["text"]}))
                    prompts.append({"text": details["text"]})
                else:
                    events.append(_event("other", details))
        if not events:
            return None
        return make_trace(f"cursor-{workspace_id}", None, None, events, prompts)

    n = 0
    trace = build(
        "global",
        load(export_dir / "prompts_raw.txt"),
        load(export_dir / "conversations_raw.txt"),
    )
    if trace:
        yield trace
        n += 1
    for workspace_file in sorted(export_dir.glob("workspace_*.txt")):
        if limit is not None and n >= limit:
            return
        data = parse_raw_json_value(workspace_file.read_text())
        wprompts, wconvs = [], []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "prompt" in item or "text" in item:
                        wprompts.append(item)
                    elif "messages" in item:
                        wconvs.append(item)
        trace = build(workspace_file.stem.replace("workspace_", ""), wprompts, wconvs)
        if trace:
            yield trace
            n += 1


def iter_traces_cursor(path: Path, limit: int | None = None) -> Iterator[dict]:
    """path: a state.vscdb file, a Cursor User dir (globalStorage +
    workspaceStorage/*/state.vscdb), or a raw-export directory."""
    path = Path(path).expanduser()
    n = 0

    def bounded(it: Iterator[dict]) -> Iterator[dict]:
        nonlocal n
        for trace in it:
            yield trace
            n += 1
            if limit is not None and n >= limit:
                return

    if path.is_file() and path.suffix == ".vscdb":
        yield from bounded(_iter_cursor_sqlite(path, limit))
        return
    if path.is_dir():
        dbs = []
        global_db = path / "globalStorage" / "state.vscdb"
        if global_db.exists():
            dbs.append(global_db)
        workspace_storage = path / "workspaceStorage"
        if workspace_storage.is_dir():
            dbs.extend(sorted(workspace_storage.glob("*/state.vscdb")))
        if not dbs:
            dbs = sorted(path.glob("*.vscdb"))
        if dbs:
            for db in dbs:
                if limit is not None and n >= limit:
                    return
                yield from bounded(_iter_cursor_sqlite(db, None if limit is None else limit - n))
            return
        yield from bounded(_iter_cursor_raw_export(path, limit))
        return
    raise FileNotFoundError(f"no Cursor store at {path}")


# SWE-agent source.

def _swe_agent_action_event(action: str) -> dict:
    action = (action or "").strip()
    if not action:
        return _event("other", {"action": action})
    head = action.split()[0]
    if head == "str_replace_editor":
        parts = action.split()
        sub = parts[1] if len(parts) > 1 else ""
        etype = classify_str_replace_editor(sub)
        details: dict = {"command": action}
        for token in parts[2:]:
            if token.startswith("/") or token.startswith("~"):
                details["file_path"] = token
                break
        return _event(etype, details)
    if head in ("submit", "exit", "skip"):
        return _event("other", {"command": action})
    if head in ("edit", "insert", "create"):
        return _event("edit", {"command": action})
    if head in ("open", "goto", "scroll_up", "scroll_down"):
        return _event("read", {"command": action})
    if head in ("search_dir", "search_file", "find_file"):
        return _event("search", {"command": action})
    return _event(classify_command(action), {"command": action})


def _swe_agent_trace_from_obj(traj: dict, instance_id: str) -> dict:
    repo = None
    base_commit = None
    if "__" in instance_id:
        org_repo = instance_id.rsplit("-", 1)[0]
        if "__" in org_repo:
            repo = org_repo.replace("__", "/", 1)
    replay_config = traj.get("replay_config")
    if isinstance(replay_config, str):
        try:
            replay_config = json.loads(replay_config)
        except (json.JSONDecodeError, TypeError):
            replay_config = None
    if isinstance(replay_config, dict):
        base_commit = (
            replay_config.get("env", {}).get("repo", {}) or {}
        ).get("base_commit") or None

    events: list[dict] = []
    prompts: list[dict] = []
    history = traj.get("history") or []
    for item in history:
        if item.get("role") == "user":
            text = str(item.get("content", ""))
            if text.strip():
                events.append(_event("prompt", {"text": text}))
                prompts.append({"text": text})
            break  # first user message is the task statement; later ones are observations
    steps = traj.get("trajectory") or []
    if steps:
        for step in steps:
            events.append(_swe_agent_action_event(str(step.get("action", ""))))
    else:
        for item in history:
            if item.get("role") == "assistant" and item.get("action"):
                events.append(_swe_agent_action_event(str(item["action"])))
    return make_trace(instance_id, repo, base_commit, events, prompts)


def iter_traces_swe_agent(path: Path, limit: int | None = None) -> Iterator[dict]:
    """path: a .traj file or a directory of *.traj files."""
    path = Path(path).expanduser()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.traj"))
    else:
        raise FileNotFoundError(f"no SWE-agent trajectories at {path}")
    n = 0
    for traj_path in files:
        try:
            traj = json.loads(traj_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        yield _swe_agent_trace_from_obj(traj, traj_path.stem)
        n += 1
        if limit is not None and n >= limit:
            return


# OpenHands source.

def _openhands_tool_details(fn: dict) -> dict:
    details: dict = {"tool_name": fn.get("name")}
    raw = fn.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        args = {}
    if isinstance(args, dict):
        if args.get("path"):
            details["file_path"] = args["path"]
        if args.get("command") and fn.get("name") == "execute_bash":
            details["command"] = args["command"]
        if args.get("file_text"):
            details["after_content"] = args["file_text"]
        if args.get("new_str"):
            details["after_content"] = args["new_str"]
        if args.get("old_str"):
            details["before_content"] = args["old_str"]
    return details


def _openhands_trace_from_record(record: dict, fallback_id: str) -> dict | None:
    trajectory = record.get("trajectory") or record.get("history") or []
    instance_id = str(record.get("instance_id") or record.get("id") or fallback_id)
    repo = record.get("repo")
    if repo is None and "__" in instance_id:
        org_repo = instance_id.rsplit("-", 1)[0]
        if "__" in org_repo:
            repo = org_repo.replace("__", "/", 1)
    events: list[dict] = []
    prompts: list[dict] = []
    for msg in trajectory:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user" and not prompts:
            text = str(msg.get("content", ""))
            if text.strip():
                events.append(_event("prompt", {"text": text}))
                prompts.append({"text": text})
            continue
        if role != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            etype = classify_openhands_tool_call(fn.get("name", ""), fn.get("arguments"))
            events.append(_event(etype, _openhands_tool_details(fn)))
    if not events:
        return None
    return make_trace(instance_id, repo, record.get("base_commit"), events, prompts)


def iter_traces_openhands(path: Path, limit: int | None = None) -> Iterator[dict]:
    """path: a .jsonl file of trajectory records, a single .json record,
    or a directory of such files."""
    path = Path(path).expanduser()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(list(path.glob("*.jsonl")) + list(path.glob("*.json")))
    else:
        raise FileNotFoundError(f"no OpenHands trajectories at {path}")
    n = 0
    for fpath in files:
        if fpath.suffix == ".jsonl":
            with fpath.open() as f:
                records = (json.loads(line) for line in f if line.strip())
                for i, record in enumerate(records):
                    trace = _openhands_trace_from_record(record, f"{fpath.stem}-{i}")
                    if trace is None:
                        continue
                    yield trace
                    n += 1
                    if limit is not None and n >= limit:
                        return
        else:
            try:
                record = json.loads(fpath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            trace = _openhands_trace_from_record(record, fpath.stem)
            if trace is None:
                continue
            yield trace
            n += 1
            if limit is not None and n >= limit:
                return



# --- swechat: SALT-NLP/SWE-chat (Hugging Face) ---------------------------------------------
# Schema verified 2026-09-10 against the dataset card and a live preview: conversations.parquet
# has one row per transcript entry with turn_type in {user_prompt, tool_use, tool_result,
# assistant_response, assistant_thinking, ...}, the harness in `agent`, and tool inputs already
# projected to file_path / command / pattern / tool_input_json. Tool names differ by harness
# by construction (Claude Code `Read`, OpenCode `read`, Gemini CLI `read_file`), so they are
# folded to one verb table before anything downstream compares harnesses.

SWECHAT_TOOL_VERB = {
    # edit
    "edit": "edit", "multiedit": "edit", "write": "edit", "notebookedit": "edit",
    "apply_patch": "edit", "replace": "edit", "write_file": "edit",
    # read
    "read": "read", "read_file": "read",
    # search
    "grep": "search", "glob": "search", "toolsearch": "search", "websearch": "search",
    "webfetch": "search", "codesearch": "search", "grep_search": "search",
    "google_web_search": "search", "list_directory": "search", "github_search_code": "search",
    # run (refined to test/search/read by classify_command)
    "bash": "run", "run_shell_command": "run",
}
SWECHAT_MAX_CHARS = 2000


def swechat_event_type(tool_name: str | None, command: str | None) -> str:
    verb = SWECHAT_TOOL_VERB.get((tool_name or "").strip().lower(), "other")
    if verb == "run":
        return classify_command(command or "") if command else "run"
    return verb


def _swechat_tool_event(row: dict) -> dict:
    etype = swechat_event_type(row.get("tool_name"), row.get("command"))
    details: dict[str, Any] = {"tool": row.get("tool_name")}
    if row.get("file_path"):
        details["file_path"] = row["file_path"]
    if row.get("command"):
        details["command"] = str(row["command"])[:SWECHAT_MAX_CHARS]
    if row.get("pattern"):
        details["query"] = str(row["pattern"])[:SWECHAT_MAX_CHARS]
    if etype == "edit" and row.get("tool_input_json"):
        try:
            inp = json.loads(row["tool_input_json"])
        except (TypeError, ValueError):
            inp = {}
        before = inp.get("old_string") or ""
        after = inp.get("new_string") or inp.get("content") or inp.get("patch") or inp.get("new_source") or ""
        if isinstance(before, str) and before:
            details["before_content"] = before[:SWECHAT_MAX_CHARS]
        if isinstance(after, str) and after:
            details["after_content"] = after[:SWECHAT_MAX_CHARS]
    ev = _event(etype, details)
    if row.get("timestamp") is not None:
        ev["timestamp"] = str(row["timestamp"])
    return ev


def _swechat_prompt_event(row: dict) -> dict:
    ev = _event("prompt", {"text": str(row.get("content") or "")[:SWECHAT_MAX_CHARS]})
    if row.get("timestamp") is not None:
        ev["timestamp"] = str(row["timestamp"])
    for k in ("prompt_intent", "prompt_pushback"):
        if row.get(k):
            ev["details"][k] = row[k]
    return ev


def _swechat_trace(session_id: str, rows: list[dict]) -> dict:
    events, prompts = [], []
    models: dict[str, int] = {}
    for r in rows:
        if r["turn_type"] == "user_prompt":
            ev = _swechat_prompt_event(r)
            events.append(ev)
            prompts.append(ev)
        elif r["turn_type"] == "tool_use":
            events.append(_swechat_tool_event(r))
        if r.get("model"):
            models[r["model"]] = models.get(r["model"], 0) + 1
    first = rows[0]
    trace = make_trace(f"swechat-{session_id}", first.get("repo_id"), None, events, prompts)
    trace["agent"] = first.get("agent") or "unknown"
    # identity travels in `labels` only, so a release step can drop one key
    trace["labels"] = {"user_id": first.get("user_id"), "session_id": session_id,
                       "model": max(models, key=models.get) if models else None}
    return trace


def iter_traces_swechat(path: Path, limit: int | None = None) -> Iterator[dict]:
    """`path` is a directory holding conversations.parquet (a SALT-NLP/SWE-chat snapshot).

    Reads only the structural columns and only prompt and tool-call rows, sorted by session
    then turn, so the 2.7M-row table costs a few hundred MB rather than its 1.3 GB.
    """
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as e:  # same extra the exporter uses
        raise ImportError('swechat import needs the "parquet" extra; install it with `uv sync --extra parquet`') from e
    src = path / "conversations.parquet" if path.is_dir() else path
    cols = ["session_id", "repo_id", "user_id", "agent", "turn_number", "turn_type", "timestamp",
            "content", "model", "tool_name", "file_path", "command", "pattern", "tool_input_json",
            "prompt_intent", "prompt_pushback"]
    table = pq.read_table(src, columns=cols, filters=[("turn_type", "in", ["user_prompt", "tool_use", "assistant_response"])])
    table = table.sort_by([("session_id", "ascending"), ("turn_number", "ascending")])
    n = 0
    current: str | None = None
    rows: list[dict] = []
    for batch in table.to_batches(max_chunksize=65536):
        for r in batch.to_pylist():
            if r["session_id"] != current:
                if rows:
                    yield _swechat_trace(current, rows)
                    n += 1
                    if limit is not None and n >= limit:
                        return
                current, rows = r["session_id"], []
            rows.append(r)
    if rows and (limit is None or n < limit):
        yield _swechat_trace(current, rows)



# --- specstory: .specstory/history/<timestamp>-<slug>.md session exports ---------------------
# Shape read from six real exports (2026-09-11; two Claude Code, two Cursor, one early Cursor,
# one sidechain-only) rather than from the SpecStory docs. Three layouts coexist:
#   Markdown v2.1.0 (both harnesses): every tool call is a <tool-use data-tool-type=".."
#     data-tool-name=".."> block wrapping <details><summary>Tool use: **name** [• Read file: P |
#     • Edit file: P | • Run command: C | • Grep for "q" ...]</summary>. Claude Code blocks put a
#     description line, then the command in backticks or a ```bash fence, then a ```text result
#     (Write/Edit carry the path only in that result text); Cursor blocks carry a ```diff.
#   Early Cursor (no version tag): a bare `Tool use: **name**` line followed by `Read file: P`,
#     a ```bash fence, or <details><summary>Edit file: P</summary> with a ```diff.
#   Turn headers: _**User**_, _**User (ts)**_, _**Agent (model ts)**_, _**Agent (model m, mode
#     Agent)**_, _**Assistant (model)**_, _**Agent (model) (sidechain)**_; a session comment
#     `<!-- <harness> Session <uuid> (<ts>) -->` names the harness in newer files only.
# Agent prose and <think> blocks are not events. data-tool-type is the harness-independent
# key (Claude Code's Edit and Write are both type "write"), so the verb comes from it.

SPECSTORY_MAX_CHARS = 2000
SPECSTORY_TYPE_VERB = {"read": "read", "write": "edit", "edit": "edit", "search": "search",
                       "shell": "run", "command": "run"}
SPECSTORY_HARNESS = {"claude code": "Claude Code", "cursor": "Cursor", "copilot": "Copilot",
                     "github copilot": "Copilot"}

_SS_HEADER = re.compile(r"<!--\s*Generated by SpecStory(?:,\s*([^>]*?))?\s*-->")
_SS_SESSION = re.compile(r"<!--\s*(.+?)\s+Session\s+([0-9a-fA-F-]{36})\s*(?:\(([^)]*)\))?\s*-->")
_SS_TURN = re.compile(r"^_\*\*(User|Agent|Assistant)((?:\s*\([^)]*\))*)\*\*_\s*$")
_SS_PAREN = re.compile(r"\(([^)]*)\)")
_SS_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T_]\d{2}[:-]\d{2}(?:[:-]\d{2})?Z?")
_SS_TOOL_OPEN = re.compile(r"<tool-use\s+([^>]*)>")
_SS_ATTR = re.compile(r'data-(tool-type|tool-name)="([^"]*)"')
_SS_OLD_TOOL = re.compile(r"^Tool use:\s*\*\*([^*]+)\*\*\s*$")
_SS_SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S)
_SS_FENCE = re.compile(r"^[ \t]*```([\w+-]*)[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.S | re.M)
_SS_INLINE_CODE = re.compile(r"^\s*`([^`\n]+)`\s*$", re.M)
_SS_QUOTED = re.compile(r'"([^"]*)"|`([^`]*)`')
_SS_RESULT_PATH = re.compile(
    r"File created successfully at:\s*(\S+)|The file\s+(\S+)\s+has been updated successfully")


def _ss_clip(s: str) -> str:
    return html.unescape(s)[:SPECSTORY_MAX_CHARS]


def specstory_event_type(tool_type: str | None, command: str | None) -> str:
    verb = SPECSTORY_TYPE_VERB.get((tool_type or "").strip().lower(), "other")
    if verb == "run":
        return classify_command(command) if command else "run"
    return verb


def _ss_summary_fields(summary: str, details: dict) -> None:
    """`Tool use: **x** • Read file: P • 9 matches` -> file_path / command / query."""
    for piece in html.unescape(summary).split(" • "):
        piece = piece.strip()
        for prefix, key in (("Read file:", "file_path"), ("Edit file:", "file_path"),
                            ("Run command:", "command")):
            if piece.startswith(prefix):
                details.setdefault(key, piece[len(prefix):].strip())
        if piece.startswith(("Grep for", "Searched codebase", "Searched web", "Listed directory")):
            quoted = [a or b for a, b in _SS_QUOTED.findall(piece)]
            if quoted:
                details.setdefault("query", quoted[0][:SPECSTORY_MAX_CHARS])
            m = re.search(r'\bin "([^"]+)"', piece)
            if m:
                details.setdefault("file_path", m.group(1))


def _ss_body_fields(body: str, tool_type: str, details: dict) -> None:
    """Fenced blocks and result lines: the command, an edit's diff or content, a path."""
    fences = _SS_FENCE.findall(body)
    head = body.split("```", 1)[0]
    if "command" not in details:
        for lang, content in fences:
            if lang.lower() in ("bash", "sh", "shell", "zsh"):
                details["command"] = _ss_clip(content.strip())
                break
        else:
            m = _SS_INLINE_CODE.search(head)
            if m and tool_type in ("shell", "command"):
                details["command"] = _ss_clip(m.group(1))
    if "file_path" not in details:
        if tool_type == "read":
            m = _SS_INLINE_CODE.search(head)
            if m:
                details["file_path"] = m.group(1).strip()
        m = _SS_RESULT_PATH.search(body)
        if m and "file_path" not in details:
            details["file_path"] = (m.group(1) or m.group(2)).rstrip(".")
    for lang, content in fences:
        if lang.lower() == "diff":
            details["diff"] = _ss_clip(content)
            break
    else:
        if tool_type in ("write", "edit"):
            for lang, content in fences:
                if lang.lower() not in ("text", "bash", "sh", "shell", ""):
                    details["after_content"] = _ss_clip(content)
                    break


def _specstory_tool_event(block: str, timestamp: str | None) -> dict:
    """One <tool-use ...>...</tool-use> block from a Markdown v2.1.0 export."""
    m = _SS_TOOL_OPEN.search(block)
    attrs = dict(_SS_ATTR.findall(m.group(1))) if m else {}
    tool_type = attrs.get("tool-type", "").lower()
    body = block[m.end():] if m else block
    details: dict[str, Any] = {"tool": attrs.get("tool-name")}
    for summary in _SS_SUMMARY.findall(body):
        _ss_summary_fields(" ".join(summary.split()), details)
    _ss_body_fields(_SS_SUMMARY.sub("", body), tool_type, details)
    ev = _event(specstory_event_type(tool_type, details.get("command")), details)
    if timestamp:
        ev["timestamp"] = timestamp
    return ev


def _specstory_old_tool_event(name: str, body: str, timestamp: str | None) -> dict:
    """A bare `Tool use: **name**` line plus what follows it, in the early Cursor layout."""
    details: dict[str, Any] = {"tool": name}
    for summary in _SS_SUMMARY.findall(body):
        _ss_summary_fields(" ".join(summary.split()), details)
    for line in _SS_SUMMARY.sub("", body).split("```", 1)[0].splitlines():
        _ss_summary_fields(line, details)
    _ss_body_fields(body, "", details)
    etype = classify_cursor_tool(name, {"command": details["command"]} if details.get("command") else None)
    ev = _event(etype, details)
    if timestamp:
        ev["timestamp"] = timestamp
    return ev


def _specstory_turn(header: str) -> tuple[str, str | None, str | None]:
    """_**Agent (claude-opus-5 2026-08-09 13:10:30Z)**_ -> ('agent', model, timestamp)."""
    m = _SS_TURN.match(header)
    role = "user" if m.group(1) == "User" else "agent"
    model = timestamp = None
    for group in _SS_PAREN.findall(m.group(2)):
        ts = _SS_TIMESTAMP.search(group)
        if ts:
            timestamp = ts.group(0)
            group = group[:ts.start()] + group[ts.end():]
        if role == "agent" and model is None:
            first = group.split(",")[0].strip()
            if first.startswith("model "):
                first = first[len("model "):].strip()
            if first and first != "sidechain":
                model = first
    return role, model, timestamp


def _specstory_events(text: str) -> tuple[list[dict], dict[str, int]]:
    """Walk the export line by line: prompt events from user turns, tool events from tool
    blocks, in document order; agent prose, <think> blocks and fenced text are skipped."""
    lines = text.split("\n")
    events: list[dict] = []
    models: dict[str, int] = {}
    role: str | None = None
    turn_ts: str | None = None
    user_buf: list[str] = []
    pending: tuple[str, list[str]] | None = None
    in_fence = False

    def flush_user() -> None:
        prompt = "\n".join(user_buf).strip()
        user_buf.clear()
        if prompt:
            ev = _event("prompt", {"text": prompt[:SPECSTORY_MAX_CHARS]})
            if turn_ts:
                ev["timestamp"] = turn_ts
            events.append(ev)

    def flush_tool() -> None:
        nonlocal pending
        if pending is not None:
            events.append(_specstory_old_tool_event(pending[0], "\n".join(pending[1]), turn_ts))
            pending = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not in_fence:
            if _SS_TURN.match(stripped):
                flush_tool()
                flush_user()
                role, model, turn_ts = _specstory_turn(stripped)
                if model:
                    models[model] = models.get(model, 0) + 1
                i += 1
                continue
            if _SS_TOOL_OPEN.match(stripped):
                flush_tool()
                j = i
                while j < len(lines) and "</tool-use>" not in lines[j]:
                    j += 1
                events.append(_specstory_tool_event("\n".join(lines[i:j + 1]), turn_ts))
                i = j + 1
                continue
            if stripped.startswith("<think>"):
                flush_tool()
                while i < len(lines) and "</think>" not in lines[i]:
                    i += 1
                i += 1
                continue
            m = _SS_OLD_TOOL.match(stripped)
            if m:
                flush_tool()
                pending = (m.group(1).strip(), [])
                i += 1
                continue
            if stripped == "---":
                flush_tool()
                i += 1
                continue
        if stripped.startswith("```"):
            in_fence = not in_fence
        if pending is not None:
            pending[1].append(line)
        elif role == "user":
            user_buf.append(line)
        i += 1
    flush_tool()
    flush_user()
    return events, models


def _specstory_trace(path: Path, repo: str | None) -> dict | None:
    text = path.read_text(encoding="utf-8")
    header = _SS_HEADER.search(text)
    if header is None:
        raise ValueError("no `Generated by SpecStory` header")
    version = None
    if header.group(1):
        version = header.group(1).split()[-1].lstrip("v")
    # the session comment sits above the first turn; a quoted export inside a tool result must not win
    first_turn = re.search(r"^_\*\*(?:User|Agent|Assistant)", text, re.M)
    session = _SS_SESSION.search(text, 0, first_turn.start() if first_turn else len(text))
    session_id = session.group(2) if session else None
    harness = session.group(1).strip().lower() if session else ""
    events, models = _specstory_events(text)
    if not events:
        return None
    key = session_id or hashlib.sha1(f"{repo}/{path.name}".encode()).hexdigest()
    prompts = [e for e in events if e["type"] == "prompt"]
    trace = make_trace(f"specstory-{key}", repo, None, events, prompts)
    trace["agent"] = SPECSTORY_HARNESS.get(harness, "unknown")
    trace["labels"] = {"session_id": session_id,
                       "model": max(models, key=models.get) if models else None,
                       "specstory_version": version, "repo": repo}
    return trace


def iter_traces_specstory(path: Path, limit: int | None = None) -> Iterator[dict]:
    """`path` is one exported session (.md) or a directory laid out as
    <cache>/raw/<owner__repo>/<file>.md; "owner/repo" is read back from the directory name.

    A file that is not a SpecStory export, or cannot be parsed, is skipped with a counted
    warning; a file with no user turn and no tool call (a sidechain-only export) yields nothing.
    """
    path = Path(path).expanduser()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.rglob("*.md") if p.is_file())
    else:
        raise FileNotFoundError(f"no SpecStory history at {path}")
    n = skipped = empty = 0
    for p in files:
        repo = p.parent.name.replace("__", "/", 1) if "__" in p.parent.name else None
        try:
            trace = _specstory_trace(p, repo)
        except Exception as e:  # noqa: BLE001 - one bad export must not stop the corpus
            skipped += 1
            warnings.warn(f"specstory: skipped malformed file {p.name} ({type(e).__name__}: {e}); "
                          f"{skipped} skipped so far", stacklevel=2)
            continue
        if trace is None:
            empty += 1
            continue
        yield trace
        n += 1
        if limit is not None and n >= limit:
            return
    if skipped or empty:
        print(f"specstory: {n} traces; {skipped} malformed files skipped, "
              f"{empty} with no prompt or tool call", file=sys.stderr)


SOURCES = {
    "cursor": iter_traces_cursor,
    "swe_agent": iter_traces_swe_agent,
    "openhands": iter_traces_openhands,
    "swechat": iter_traces_swechat,
    "specstory": iter_traces_specstory,
}


def parse(source: str, path: Path, limit: int | None = None) -> Iterator[dict]:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(SOURCES)}")
    return SOURCES[source](Path(path), limit=limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest agent-harness trace stores into standardized trace JSONL"
    )
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--input", required=True, help="store path (file or directory)")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for trace in parse(args.source, Path(args.input), limit=args.limit):
            f.write(json.dumps(trace) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  {n} traces written", file=sys.stderr)
    print(f"wrote {n} {args.source} traces to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
