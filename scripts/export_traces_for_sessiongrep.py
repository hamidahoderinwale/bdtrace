#!/usr/bin/env python3
"""
Export resolved dev traces into sessiongrep-indexable session files.

sessiongrep (github.com/braincompany/sessiongrep) indexes CLI-agent session
history into SQLite+FTS5 and serves it to humans (CLI/TUI) and agents (MCP).
Its Claude adapter parses any directory of JSONL files where each line is
{"message": {"role", "content"}, "timestamp"?, "sessionId"?, "cwd"?}.

This script renders two trace sources into that shape, one file per
instance, under output/sessiongrep_export/:

1. output/resolved_traces_lite_full.jsonl — 300 resolved SWE-bench traces
   (a prompt event + a code_change event each) -> bidirect-<instance_id>
2. distillation_run/child_traj/*.traj — 499 SWE-agent-LM-32B rollout
   trajectories (HF midah/bidirect-distillation-traces; fetch per
   distillation_run/MOVED.md) -> bidirect-rollout-<instance_id>

Adding the export directory to `providers.claude.paths` in
~/.config/sessiongrep/config.toml makes every dev trace searchable from the
same surface as session history ("which trace touched separability_matrix?").
The export is a derived, regenerable artifact: delete and re-run to rebuild.

Bounded fields: per-message content is truncated so the FTS index carries
the searchable head of each step, not megabytes of tool payloads; identical
system prompts are skipped as boilerplate.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "output" / "resolved_traces_lite_full.jsonl"
ROLLOUTS = REPO / "distillation_run" / "child_traj"
DEST = REPO / "output" / "sessiongrep_export"
MAX_CONTENT_CHARS = 4000  # per code_change event; head of patch is the searchable part
MAX_ROLLOUT_MSG_CHARS = 1500  # per rollout history message; bounds index size


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


def write_session(session_id: str, title: str, messages: list[tuple[str, str]]) -> None:
    lines = [
        {"type": "last-prompt", "lastPrompt": title, "sessionId": session_id, "cwd": str(REPO)}
    ]
    for role, text in messages:
        if text.strip():
            lines.append({"message": {"role": role, "content": text}, "cwd": str(REPO)})
    out = DEST / f"{session_id}.jsonl"
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(entry) for entry in lines) + "\n")
    tmp.replace(out)


def export_rollouts() -> int:
    if not ROLLOUTS.is_dir():
        print(f"rollouts not present at {ROLLOUTS} — see distillation_run/MOVED.md; skipping")
        return 0
    written = 0
    for traj_path in sorted(ROLLOUTS.glob("*.traj")):
        instance_id = traj_path.stem
        traj = json.loads(traj_path.read_text())
        messages = []
        for item in traj.get("history", []):
            role = item.get("role")
            if role not in ("user", "assistant"):
                continue  # system prompts are identical boilerplate across rollouts
            messages.append((role, str(item.get("content", ""))[:MAX_ROLLOUT_MSG_CHARS]))
        write_session(
            f"bidirect-rollout-{instance_id}",
            f"bidirect rollout {instance_id} (SWE-agent-LM-32B child)",
            messages,
        )
        written += 1
    return written


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    written = 0
    for line in SOURCE.open():
        trace = json.loads(line)
        instance_id = trace["instance_id"]
        messages = []
        for event in trace["events"]:
            details = event.get("details", {})
            if event["type"] == "prompt":
                messages.append(("user", str(details.get("text", ""))))
            else:
                messages.append(("assistant", render_code_change(details)))
        write_session(
            f"bidirect-{instance_id}",
            f"bidirect trace {instance_id} ({trace['repo']})",
            messages,
        )
        written += 1
    rollouts = export_rollouts()
    print(f"wrote {written} resolved traces + {rollouts} rollouts to {DEST}")


if __name__ == "__main__":
    main()
