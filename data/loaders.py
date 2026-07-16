"""Generic loaders for extraction pipeline."""

import json
from collections.abc import Iterator
from pathlib import Path


def load_from_jsonl(
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
    **kwargs: object,
) -> Iterator[dict]:
    """Yield records from JSONL file. Ignores split; path may include {split} placeholder."""
    path = Path(path)
    if split and "{split}" in str(path):
        path = Path(str(path).format(split=split))
    if not path.exists():
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec
            except json.JSONDecodeError:
                continue
