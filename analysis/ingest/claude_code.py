"""
Parse a local Claude Code session store into standardized trace records.

Sessions live at ~/.claude/projects/<workspace-slug>/<session-uuid>.jsonl, one JSON
object per line. User turns become "prompt" events; assistant tool_use blocks map onto
the event taxonomy (prompt, edit, read, search, run, test, other). Output records match
the shape of output/resolved_traces_lite_full.jsonl rows:

    {"instance_id": "claude-<sessionId>", "repo": None, "base_commit": None,
     "events": [{"type": ..., "details": {...}, "timestamp": ...}], "prompts": [...]}
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

MAX_FIELD_CHARS = 2000  # per content field; head of the text is the searchable part

TOOL_TO_EVENT_TYPE = {
    "Edit": "edit",
    "MultiEdit": "edit",
    "Write": "edit",
    "NotebookEdit": "edit",
    "Read": "read",
    "Grep": "search",
    "Glob": "search",
    "ToolSearch": "search",
    "WebSearch": "search",
    "Bash": "run",
}

_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|py\.test|unittest|npm +test|yarn +test|pnpm +test|go +test|cargo +test|tox)\b"
)


def _truncate(value: str) -> str:
    return value[:MAX_FIELD_CHARS]


def _text_from_content(content) -> str:
    """Extract user-visible text from a message content field (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _prompt_event(text: str, timestamp, cwd) -> dict:
    text = _truncate(text)
    return {
        "type": "prompt",
        "details": {"text": text, "content": text, "cwd": cwd},
        "timestamp": timestamp,
    }


def _tool_event(name: str, tool_input: dict, timestamp, cwd) -> dict:
    event_type = TOOL_TO_EVENT_TYPE.get(name, "other")
    details: dict = {"tool": name, "cwd": cwd}

    if event_type == "edit":
        details["file_path"] = tool_input.get("file_path") or tool_input.get("notebook_path")
        before = tool_input.get("old_string", "")
        after = tool_input.get("new_string") or tool_input.get("content") or tool_input.get("new_source") or ""
        if isinstance(before, str) and before:
            details["before_content"] = _truncate(before)
        if isinstance(after, str) and after:
            details["after_content"] = _truncate(after)
    elif event_type == "read":
        details["file_path"] = tool_input.get("file_path")
    elif event_type == "search":
        details["query"] = _truncate(
            str(tool_input.get("pattern") or tool_input.get("query") or "")
        )
        if tool_input.get("path"):
            details["path"] = tool_input["path"]
    elif event_type == "run":
        command = str(tool_input.get("command") or "")
        details["command"] = _truncate(command)
        if tool_input.get("description"):
            details["description"] = _truncate(str(tool_input["description"]))
        if _TEST_COMMAND_RE.search(command):
            event_type = "test"
    else:
        details["input"] = _truncate(json.dumps(tool_input, default=str))

    return {"type": event_type, "details": details, "timestamp": timestamp}


def parse_session_file(path: Path) -> dict | None:
    """Parse one session .jsonl file into a trace record; None if it yields no events."""
    path = Path(path)
    session_id = path.stem
    cwd = None
    events = []
    prompts = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            if obj.get("sessionId"):
                session_id = obj["sessionId"]
            if cwd is None and obj.get("cwd"):
                cwd = obj["cwd"]

            line_type = obj.get("type")
            message = obj.get("message") or {}
            timestamp = obj.get("timestamp")

            if line_type == "user":
                if obj.get("isMeta"):
                    continue
                text = _text_from_content(message.get("content")).strip()
                if not text or text.startswith("<command-name>") or text.startswith("<local-command"):
                    continue
                events.append(_prompt_event(text, timestamp, cwd))
                prompts.append({"text": _truncate(text), "timestamp": timestamp})

            elif line_type == "assistant":
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name") or ""
                    tool_input = block.get("input")
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    events.append(_tool_event(name, tool_input, timestamp, cwd))

    if not events:
        return None

    return {
        "instance_id": f"claude-{session_id}",
        "repo": None,
        "base_commit": None,
        "cwd": cwd,
        "events": events,
        "prompts": prompts,
    }


def iter_traces(root: Path | None = None, limit: int | None = None) -> Iterator[dict]:
    """Yield trace records for every parseable session under root (~/.claude/projects)."""
    root = Path(root) if root is not None else Path.home() / ".claude" / "projects"
    count = 0
    for path in sorted(root.glob("*/*.jsonl")):
        if limit is not None and count >= limit:
            return
        trace = parse_session_file(path)
        if trace is None:
            continue
        count += 1
        yield trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="session store root (default ~/.claude/projects)")
    parser.add_argument("--output", type=Path, default=None, help="output JSONL path (default stdout)")
    parser.add_argument("--limit", type=int, default=None, help="max sessions to parse")
    args = parser.parse_args()

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    n = 0
    try:
        for trace in iter_traces(root=args.root, limit=args.limit):
            out.write(json.dumps(trace, default=str) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"parsed {n} sessions", file=sys.stderr)
    finally:
        if args.output:
            out.close()
    print(f"done: {n} sessions", file=sys.stderr)


if __name__ == "__main__":
    main()
