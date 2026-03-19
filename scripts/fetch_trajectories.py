#!/usr/bin/env python3
"""
Fetch SWE-bench agent trajectories and extract procedural features.

Fetches from S3 (public SWE-bench submissions), caches locally, and saves
a parquet of per-instance features for downstream analysis.

Reuses:
- data/swebench_trajectories.py: fetcher + feature extractor
- output/swebench_results/: existing pass/fail files
- output/repos/: cached repo clones for hop distance

Usage:
  uv run python scripts/fetch_trajectories.py --models 20240402_sweagent_gpt4 20240620_sweagent_claude3.5sonnet
  uv run python scripts/fetch_trajectories.py --models 20240402_sweagent_gpt4 --limit 20
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.swebench_trajectories import fetch_all

ROOT = Path(__file__).resolve().parent.parent


def load_instance_ids(split: str = "lite") -> list[str]:
    from datasets import load_dataset
    hf_id = "princeton-nlp/SWE-bench_Verified" if split == "verified" else "princeton-nlp/SWE-bench_Lite"
    ds = load_dataset(hf_id, split="test")
    return [str(x) for x in ds["instance_id"]]


def main():
    parser = argparse.ArgumentParser(description="Fetch SWE-bench trajectories and extract features")
    parser.add_argument("--models", nargs="+", required=True, help="Model IDs (e.g. 20240402_sweagent_gpt4)")
    parser.add_argument("--split", default="lite", choices=["lite", "verified"])
    parser.add_argument("--limit", type=int, default=None, help="Limit instances per model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "trajectories")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "output" / "trajectories" / ".cache")
    parser.add_argument("--repos-dir", type=Path, default=ROOT / "output" / "repos")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading instance IDs...")
    instance_ids = load_instance_ids(args.split)
    if args.limit:
        instance_ids = instance_ids[:args.limit]
    print(f"  {len(instance_ids)} instances")

    # Load pass/fail for annotation
    pf_dir = ROOT / "output" / "swebench_results"
    pf_by_model: dict[str, dict[str, bool]] = {}
    for model_id in args.models:
        pf_path = pf_dir / f"{args.split}_{model_id}.json"
        if pf_path.exists():
            with open(pf_path) as f:
                pf_by_model[model_id] = {r["instance_id"]: r["resolved"] for r in json.load(f)}
            print(f"  Loaded pass/fail for {model_id}: {sum(pf_by_model[model_id].values())} passed")
        else:
            print(f"  No pass/fail found for {model_id} at {pf_path}")

    all_frames = []
    for model_id in args.models:
        print(f"\nFetching {model_id}...")
        features = fetch_all(
            instance_ids=instance_ids,
            model_id=model_id,
            split=args.split,
            cache_dir=args.cache_dir,
            repos_dir=args.repos_dir if args.repos_dir.exists() else None,
            max_workers=args.workers,
        )
        print(f"  Fetched {len(features)}/{len(instance_ids)} trajectories")

        # Annotate with pass/fail
        pf = pf_by_model.get(model_id, {})
        for row in features:
            row["passed"] = pf.get(row["instance_id"])

        df = pd.DataFrame(features)

        # Serialize list columns to JSON strings for parquet compatibility
        for col in ["files_opened", "files_edited"]:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

        out_path = args.output_dir / f"{args.split}_{model_id}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {out_path}")
        all_frames.append(df)

    # Combined across models
    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = args.output_dir / f"{args.split}_all_models.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\nCombined: {combined_path} ({len(combined)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
