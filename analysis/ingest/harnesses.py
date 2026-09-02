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
import json
import re
import sqlite3
import sys
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


SOURCES = {
    "cursor": iter_traces_cursor,
    "swe_agent": iter_traces_swe_agent,
    "openhands": iter_traces_openhands,
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
