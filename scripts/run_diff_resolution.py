#!/usr/bin/env python3
"""
Resolve SWE-bench Lite patches to full before/after file content.

Clones repos, checkouts base_commit, applies patch, reads full files.
Output: JSONL of traces with full before_content, after_content.

Usage:
  python scripts/run_diff_resolution.py --output output/resolved_traces.jsonl --limit 3
  python scripts/run_diff_resolution.py --output output/resolved_traces.jsonl --repos-cache output/repos
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from data.diff_resolution import resolve_patch_to_full_files


def _n_files_in_patch(patch: str) -> int:
    """Count distinct files touched by patch."""
    from data.swe_bench import _parse_patch

    return len({fc["file_path"] for fc in _parse_patch(patch or "")})


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Resolve patches to full files")
    parser.add_argument("--output", "-o", type=Path, default=Path("output/resolved_traces.jsonl"))
    parser.add_argument("--repos-cache", type=Path, default=Path("output/repos"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--multi-file-only",
        action="store_true",
        help="Only include instances where patch touches 2+ files (for modules signal)",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset (e.g. SWE-bench/SWE-smith for SWE-smith)",
    )
    parser.add_argument("--hf-split", default=None, help="HF split override (default: test for SWE-bench, train for SWE-smith)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.repos_cache.mkdir(parents=True, exist_ok=True)

    # Default HF split: SWE-smith uses "train", SWE-bench uses "test"
    hf_split = args.hf_split or args.split
    if hf_split is None or hf_split == "test":
        hf_split = "train" if "SWE-smith" in args.dataset else "test"

    ds = load_dataset(args.dataset, split=hf_split)
    total = len(ds)
    count = 0
    skipped = 0
    with open(args.output, "w") as f:
        for i, row in enumerate(ds):
            if args.limit and count >= args.limit:
                break
            instance = dict(row)
            if args.multi_file_only and _n_files_in_patch(instance.get("patch", "")) < 2:
                skipped += 1
                continue
            trace = resolve_patch_to_full_files(instance, args.repos_cache)
            f.write(json.dumps(trace, default=str) + "\n")
            f.flush()
            count += 1
            print(f"  [{count}/{total - skipped}] {instance['instance_id']}", flush=True)

    print(f"Wrote {count} traces to {args.output}", flush=True)


if __name__ == "__main__":
    main()
