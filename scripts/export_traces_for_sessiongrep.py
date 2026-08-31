#!/usr/bin/env python3
"""
Export resolved dev traces into sessiongrep-indexable session files.

sessiongrep (github.com/braincompany/sessiongrep) indexes CLI-agent session
history into SQLite+FTS5 and serves it to humans (CLI/TUI) and agents (MCP).
Its Claude adapter parses any directory of JSONL files where each line is
{"message": {"role", "content"}, "timestamp"?, "sessionId"?, "cwd"?}.

This script renders each trace in output/resolved_traces_lite_full.jsonl
(one SWE-bench instance: a prompt event + a code_change event) into that
shape, one file per instance, under output/sessiongrep_export/. Adding that
directory to `providers.claude.paths` in ~/.config/sessiongrep/config.toml
makes every dev trace searchable from the same surface as session history
("which trace touched separability_matrix?"). The export is a derived,
regenerable artifact: delete the directory and re-run to rebuild.

Bounded fields: code_change content is truncated per event so the FTS index
carries the searchable head of a patch, not megabytes of file bodies.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "output" / "resolved_traces_lite_full.jsonl"
DEST = REPO / "output" / "sessiongrep_export"
MAX_CONTENT_CHARS = 4000  # per code_change event; head of patch is the searchable part


def render_code_change(details: dict) -> str:
    parts = [f"[code_change] {details.get('file_path', '(unknown file)')}"]
    if details.get("diff_summary"):
        parts.append(str(details["diff_summary"]))
    added, removed = details.get("lines_added"), details.get("lines_removed")
    if added is not None or removed is not None:
        parts.append(f"+{added or 0}/-{removed or 0} lines")
    content = str(details.get("content", ""))[:MAX_CONTENT_CHARS]
    if content:
        parts.append(content)
    return "\n".join(parts)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    written = 0
    for line in SOURCE.open():
        trace = json.loads(line)
        instance_id = trace["instance_id"]
        session_id = f"bidirect-{instance_id}"
        title = f"bidirect trace {instance_id} ({trace['repo']})"
        lines = [
            {
                "type": "last-prompt",
                "lastPrompt": title,
                "sessionId": session_id,
                "cwd": str(REPO),
            }
        ]
        for event in trace["events"]:
            details = event.get("details", {})
            if event["type"] == "prompt":
                role, text = "user", str(details.get("text", ""))
            else:
                role, text = "assistant", render_code_change(details)
            if not text.strip():
                continue
            lines.append({"message": {"role": role, "content": text}, "cwd": str(REPO)})
        out = DEST / f"{session_id}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(entry) for entry in lines) + "\n")
        tmp.replace(out)
        written += 1
    print(f"wrote {written} trace sessions to {DEST}")


if __name__ == "__main__":
    main()
