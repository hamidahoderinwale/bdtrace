#!/usr/bin/env python3
"""
Resolve a stratified sample from SWE-Smith across diverse repos.

Samples PER_REPO instances from each target repo, then resolves patches.
Repos are the swesmith forks on GitHub (require network access).

Usage:
  uv run python scripts/run_smith_stratified_resolution.py \
    --output output/resolved_traces_swe_smith_stratified.jsonl \
    --repos-cache output/repos
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from data.diff_resolution import resolve_patch_to_full_files

## 10 repos covering diverse domains: data, parsing, networking, testing, etc.
TARGET_REPOS = [
    "pandas-dev__pandas",
    "pylint-dev__astroid",
    "pygments__pygments",
    "pallets__jinja",
    "paramiko__paramiko",
    "tobymao__sqlglot",
    "python-trio__trio",
    "sunpy__sunpy",
    "pydata__patsy",
    "seperman__deepdiff",
]
PER_REPO = 30
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/resolved_traces_swe_smith_stratified.jsonl"))
    parser.add_argument("--repos-cache", type=Path, default=Path("output/repos"))
    parser.add_argument("--per-repo", type=int, default=PER_REPO)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.repos_cache.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    print("Loading SWE-Smith from HuggingFace...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    ## Group by base repo name (strip the commit hash suffix)
    by_repo: dict[str, list] = defaultdict(list)
    for row in ds:
        repo_key = row["repo"].split("/")[-1]   # e.g. pandas-dev__pandas.27649ebb
        base = repo_key.split(".")[0]            # pandas-dev__pandas
        if base in TARGET_REPOS:
            by_repo[base].append(dict(row))

    ## Stratified sample
    sampled: list[dict] = []
    for repo in TARGET_REPOS:
        pool = by_repo.get(repo, [])
        n = min(args.per_repo, len(pool))
        chosen = random.sample(pool, n)
        sampled.extend(chosen)
        print(f"  {repo}: {n} sampled (pool={len(pool)})")

    print(f"Total: {len(sampled)} tasks across {len(by_repo)} repos")

    count = 0
    errors = 0
    with open(args.output, "w") as f:
        for instance in sampled:
            try:
                trace = resolve_patch_to_full_files(instance, args.repos_cache)
                f.write(json.dumps(trace, default=str) + "\n")
                f.flush()
                count += 1
                print(f"  [{count}/{len(sampled)}] {instance['instance_id']}", flush=True)
            except Exception as e:
                errors += 1
                print(f"  [skip] {instance['instance_id']}: {e}", flush=True)

    print(f"Wrote {count} traces to {args.output} ({errors} errors)")


if __name__ == "__main__":
    main()
