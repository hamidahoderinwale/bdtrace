"""
Agent trajectory loader for SWE-agent-style datasets.

Loads trajectories from Hugging Face (e.g. nebius/SWE-agent-trajectories)
and converts them to trace format with intermediate states.
"""

import json
import re
from collections.abc import Iterator

from datasets import load_dataset


def _parse_ai_content(text: str) -> tuple[str, list[str]]:
    """
    Extract reasoning and command blocks from AI message content.

    Commands appear in fenced code blocks (```...```).
    Returns (reasoning_text, [command1, command2, ...]).
    """
    if not text or not text.strip():
        return "", []

    commands: list[str] = []
    # Match ```...``` blocks; capture content
    block_pattern = re.compile(r"```(?:[\w-]*)\n?(.*?)```", re.DOTALL)
    blocks = block_pattern.findall(text)
    for block in blocks:
        cmd = block.strip()
        if cmd:
            commands.append(cmd)

    # Reasoning is text outside blocks
    reasoning = block_pattern.sub("", text).strip()
    return reasoning, commands


def _trajectory_to_events(trajectory: list[dict], step_index_offset: int = 0) -> list[dict]:
    """
    Convert trajectory messages to trace events.

    Message format: role (system|user|ai), text.
    - system: skip (or could capture as system_prompt)
    - user: issue text → prompt; observations → observation
    - ai: reasoning → model_reasoning; command blocks → terminal_command
    """
    events: list[dict] = []
    is_first_user = True

    for i, msg in enumerate(trajectory):
        role = (msg.get("role") or "").lower()
        text = msg.get("text") or msg.get("content") or ""
        step_idx = step_index_offset + i

        if role == "system":
            continue

        if role == "user":
            if is_first_user and text.strip():
                events.append(
                    {
                        "type": "prompt",
                        "details": {"text": text, "content": text},
                        "step_index": step_idx,
                        "source": "agent_trajectory",
                    }
                )
                is_first_user = False
            elif text.strip():
                events.append(
                    {
                        "type": "observation",
                        "details": {"text": text, "content": text},
                        "step_index": step_idx,
                        "source": "agent_trajectory",
                    }
                )

        elif role == "ai":
            reasoning, commands = _parse_ai_content(text)
            if reasoning:
                events.append(
                    {
                        "type": "model_reasoning",
                        "details": {"text": reasoning, "content": reasoning},
                        "step_index": step_idx,
                        "source": "agent_trajectory",
                    }
                )
            for cmd in commands:
                events.append(
                    {
                        "type": "terminal_command",
                        "details": {"command": cmd, "text": cmd},
                        "step_index": step_idx,
                        "source": "agent_trajectory",
                    }
                )

    return events


def agent_trajectory_to_trace(row: dict) -> dict:
    """
    Convert a single agent trajectory row to trace format.

    Trace format: {events: [...], prompts: [...], instance_id, model_name, ...}
    Events include prompt, model_reasoning, terminal_command, observation.
    """
    trajectory_raw = row.get("trajectory") or "[]"
    trajectory = (
        json.loads(trajectory_raw) if isinstance(trajectory_raw, str) else trajectory_raw
    )
    if not isinstance(trajectory, list):
        trajectory = []

    events = _trajectory_to_events(trajectory)

    # Prompts: first user message (issue)
    prompts: list[dict] = []
    for ev in events:
        if ev.get("type") == "prompt":
            details = ev.get("details") or {}
            prompts.append(
                {
                    "text": details.get("text", ""),
                    "content": details.get("content", ""),
                }
            )
            break

    return {
        "instance_id": row.get("instance_id"),
        "model_name": row.get("model_name"),
        "target": row.get("target"),
        "exit_status": row.get("exit_status"),
        "generated_patch": row.get("generated_patch"),
        "eval_logs": row.get("eval_logs"),
        "events": events,
        "prompts": prompts,
    }


def load_agent_trajectories(
    dataset_id: str = "nebius/SWE-agent-trajectories",
    split: str = "train",
    limit: int | None = None,
) -> Iterator[dict]:
    """Load agent trajectories from Hugging Face and yield traces."""
    ds = load_dataset(dataset_id, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield agent_trajectory_to_trace(dict(row))
