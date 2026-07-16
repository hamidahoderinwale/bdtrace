"""
Loaders for non-SWE-bench evaluation datasets.

Each loader yields traces in the standard format:
  {instance_id, events: [{type, details}], prompts: [...]}

Supported:
  - HumanEval  (openai_humaneval, 164 function synthesis tasks)
  - MBPP       (google-research-datasets/mbpp, 500 docstring-to-code tasks)
  - LiveCodeBench (livecodebench/code_generation_lite, competitive programming)
  - BigCodeBench  (bigcode/bigcodebench, library API tasks)
"""

from collections.abc import Iterator


def _code_change_event(
    file_path: str,
    before_content: str,
    after_content: str,
) -> dict:
    before_lines = before_content.splitlines()
    after_lines = after_content.splitlines()
    added = sum(1 for l in after_lines if l not in before_lines)
    removed = sum(1 for l in before_lines if l not in after_lines)
    return {
        "type": "code_change",
        "details": {
            "file_path": file_path,
            "before_content": before_content,
            "after_content": after_content,
            "lines_added": added,
            "lines_removed": removed,
            "diff_summary": f"+{added}, -{removed}",
        },
    }


def _prompt_event(text: str) -> dict:
    return {
        "type": "prompt",
        "details": {"text": text, "content": text},
    }


def load_humaneval(
    split: str = "test",
    limit: int | None = None,
    **kwargs,
) -> Iterator[dict]:
    """
    Load HumanEval (openai_humaneval).
    before = function signature + docstring (prompt), after = prompt + canonical_solution.
    """
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split=split, trust_remote_code=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        task_id = str(row["task_id"])
        prompt = row["prompt"] or ""
        solution = row["canonical_solution"] or ""
        after = prompt + solution
        yield {
            "instance_id": task_id.replace("/", "__"),
            "repo": "humaneval",
            "events": [
                _prompt_event(prompt),
                _code_change_event(
                    f"{row['entry_point']}.py",
                    before_content=prompt,
                    after_content=after,
                ),
            ],
            "prompts": [{"text": prompt, "content": prompt}],
            "task_type": "code_generation",
        }


def load_mbpp(
    split: str = "test",
    limit: int | None = None,
    **kwargs,
) -> Iterator[dict]:
    """
    Load MBPP (google-research-datasets/mbpp).
    before = empty, after = canonical solution code.
    """
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split=split, trust_remote_code=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        task_id = str(row["task_id"])
        text = row["text"] or ""
        code = row["code"] or ""
        yield {
            "instance_id": f"mbpp__{task_id}",
            "repo": "mbpp",
            "events": [
                _prompt_event(text),
                _code_change_event("solution.py", before_content="", after_content=code),
            ],
            "prompts": [{"text": text, "content": text}],
            "task_type": "code_generation",
        }


def load_livecodebench(
    split: str = "test",
    limit: int | None = None,
    **kwargs,
) -> Iterator[dict]:
    """
    Load LiveCodeBench (livecodebench/code_generation_lite).
    Loads directly from JSONL via HF hub (dataset script no longer supported).
    before = empty, after = starter_code (canonical solutions not available).
    """
    import json
    from huggingface_hub import hf_hub_download
    # test6.jsonl = release_v6 (most recent); fall back to test.jsonl
    for filename in ("test6.jsonl", "test.jsonl"):
        try:
            path = hf_hub_download(
                "livecodebench/code_generation_lite", filename, repo_type="dataset"
            )
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not download any LiveCodeBench JSONL from HF hub")

    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            row = json.loads(line)
            qid = str(row.get("question_id", i))
            prompt = row.get("question_content") or ""
            starter = row.get("starter_code") or ""
            yield {
                "instance_id": f"lcb__{qid}",
                "repo": "livecodebench",
                "events": [
                    _prompt_event(prompt),
                    _code_change_event("solution.py", before_content="", after_content=starter),
                ],
                "prompts": [{"text": prompt, "content": prompt}],
                "task_type": "algorithmic",
            }


def load_bigcodebench(
    split: str = "v0.1.2",
    limit: int | None = None,
    **kwargs,
) -> Iterator[dict]:
    """
    Load BigCodeBench (bigcode/bigcodebench).
    before = code_prompt (function stub), after = canonical_solution.
    """
    from datasets import load_dataset
    ds = load_dataset("bigcode/bigcodebench", split=split, trust_remote_code=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        task_id = str(row.get("task_id", i))
        prompt = row.get("instruct_prompt") or row.get("complete_prompt") or ""
        stub = row.get("code_prompt") or ""
        solution = row.get("canonical_solution") or ""
        yield {
            "instance_id": f"bcb__{task_id.replace('/', '__')}",
            "repo": "bigcodebench",
            "events": [
                _prompt_event(prompt),
                _code_change_event("solution.py", before_content=stub, after_content=stub + solution),
            ],
            "prompts": [{"text": prompt, "content": prompt}],
            "task_type": "api_usage",
        }
