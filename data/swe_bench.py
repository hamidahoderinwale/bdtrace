"""
SWE-bench Lite loader.

Fetches from Hugging Face and converts to trace format for representation encoders.
"""

import re
from collections.abc import Iterator

from datasets import load_dataset


def _parse_patch(patch: str) -> list[dict]:
    """
    Parse unified diff: build before/after content per file.
    - lines go to before only, + to after only, context to both.
    """
    if not patch or not patch.strip():
        return []

    result = []
    current_file = None
    before_lines = []
    after_lines = []
    added_count = 0
    removed_count = 0
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("diff --git "):
            if current_file and (before_lines or after_lines):
                result.append(
                    {
                        "file_path": current_file,
                        "before_content": "\n".join(before_lines),
                        "after_content": "\n".join(after_lines),
                        "lines_added": added_count,
                        "lines_removed": removed_count,
                    }
                )
            m = re.match(r"diff --git a/(.+?) b/.+", line)
            current_file = m.group(1) if m else None
            before_lines = []
            after_lines = []
            added_count = 0
            removed_count = 0
            in_hunk = False
        elif line.startswith("--- ") or line.startswith("+++ "):
            continue
        elif line.startswith("@@ "):
            in_hunk = True
            continue
        elif in_hunk and current_file:
            if line.startswith("-") and not line.startswith("---"):
                before_lines.append(line[1:])
                removed_count += 1
            elif line.startswith("+") and not line.startswith("+++"):
                after_lines.append(line[1:])
                added_count += 1
            elif line.startswith(" "):
                ctx = line[1:]
                before_lines.append(ctx)
                after_lines.append(ctx)

    if current_file and (before_lines or after_lines):
        result.append(
            {
                "file_path": current_file,
                "before_content": "\n".join(before_lines),
                "after_content": "\n".join(after_lines),
                "lines_added": added_count,
                "lines_removed": removed_count,
            }
        )

    return result


def swe_bench_instance_to_trace(instance: dict) -> dict:
    """
    Convert a single SWE-bench instance to trace format.

    Trace format: {events: [...], prompts: [...]}
    Events have type, details (file_path, before_content, after_content, etc.)
    """
    events = []

    # Parse main patch
    patch = instance.get("patch") or ""
    for file_change in _parse_patch(patch):
        events.append(
            {
                "type": "code_change",
                "details": {
                    "file_path": file_change["file_path"],
                    "before_content": file_change["before_content"],
                    "after_content": file_change["after_content"],
                    "lines_added": file_change["lines_added"],
                    "lines_removed": file_change["lines_removed"],
                    "diff_summary": f"+{file_change['lines_added']}, -{file_change['lines_removed']}",
                },
                "timestamp": instance.get("created_at"),
            }
        )

    # Parse test patch if present
    test_patch = instance.get("test_patch") or ""
    if test_patch:
        for file_change in _parse_patch(test_patch):
            events.append(
                {
                    "type": "code_change",
                    "details": {
                        "file_path": file_change["file_path"],
                        "before_content": file_change["before_content"],
                        "after_content": file_change["after_content"],
                        "lines_added": file_change["lines_added"],
                        "lines_removed": file_change["lines_removed"],
                        "diff_summary": f"test +{file_change['lines_added']}, -{file_change['lines_removed']}",
                    },
                    "timestamp": instance.get("created_at"),
                }
            )

    # Problem statement as prompt
    problem = instance.get("problem_statement") or ""
    if problem:
        events.insert(
            0,
            {
                "type": "prompt",
                "details": {"text": problem, "content": problem},
                "timestamp": instance.get("created_at"),
            },
        )

    return {
        "instance_id": instance.get("instance_id"),
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "events": events,
        "prompts": [{"text": problem, "content": problem}] if problem else [],
    }


def load_swe_bench_lite(
    split: str = "dev",
    limit: int | None = None,
) -> Iterator[dict]:
    """Load SWE-bench Lite from Hugging Face and yield traces."""
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        yield swe_bench_instance_to_trace(dict(row))


def load_swe_smith(
    split: str = "train",
    limit: int | None = None,
    repo_filter: list[str] | None = None,
) -> Iterator[dict]:
    """
    Load SWE-smith from Hugging Face and yield traces.

    SWE-smith shares the SWE-bench schema (instance_id, patch, repo, base_commit,
    problem_statement, created_at) so we can reuse swe_bench_instance_to_trace directly.
    Optionally filter to a subset of repos.
    """
    ds = load_dataset("SWE-bench/SWE-smith", split=split)
    count = 0
    for row in ds:
        if limit and count >= limit:
            break
        instance = dict(row)
        if repo_filter and instance.get("repo") not in repo_filter:
            continue
        yield swe_bench_instance_to_trace(instance)
        count += 1


