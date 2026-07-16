#!/usr/bin/env python3
"""
Fetch SWE-bench model results (pass/fail per instance) from the experiments repo.

The SWE-bench/experiments repo has pre-populated results/results.json per model
with {"resolved": [instance_id, ...], ...}. This script converts to our format.

Usage:
  # Clone experiments repo first (one-time):
  git clone --depth 1 https://github.com/SWE-bench/experiments.git /path/to/experiments

  # Fetch one model, output JSON for --pass-fail:
  uv run python scripts/fetch_swebench_results.py \\
    --experiments-dir /path/to/experiments \\
    --split verified \\
    --model 20240402_sweagent_gpt4 \\
    --output output/swebench_results/verified_20240402_sweagent_gpt4.json

  # Fetch multiple models to a directory (for multi-model plots):
  uv run python scripts/fetch_swebench_results.py \\
    --experiments-dir /path/to/experiments \\
    --split verified \\
    --models 20240402_rag_gpt4 20240402_sweagent_gpt4 20240620_sweagent_claude3.5sonnet \\
    --output-dir output/swebench_results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_instance_ids(split: str) -> list[str]:
    """Load full instance_id list for the split from Hugging Face."""
    from datasets import load_dataset

    if split == "lite":
        ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    elif split in ("verified", "bash-only"):
        ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    elif split == "test":
        ds = load_dataset("princeton-nlp/SWE-bench", split="test")
    else:
        raise ValueError(f"Unknown split: {split}")
    return [str(x) for x in ds["instance_id"]]


def load_model_results(experiments_dir: Path, split: str, model_id: str) -> dict[str, bool]:
    """
    Load pass/fail map from experiments repo.
    Returns {instance_id: True if resolved, False otherwise}.
    """
    results_path = experiments_dir / "evaluation" / split / model_id / "results" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results at {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    resolved_set = set(data.get("resolved", []))
    instance_ids = load_instance_ids(split)
    return {iid: iid in resolved_set for iid in instance_ids}


def main():
    parser = argparse.ArgumentParser(description="Fetch SWE-bench model results from experiments repo")
    parser.add_argument("--experiments-dir", type=Path, required=True, help="Path to cloned SWE-bench/experiments repo")
    parser.add_argument("--split", type=str, default="verified", choices=["lite", "verified", "test", "bash-only"])
    parser.add_argument("--model", type=str, help="Single model id (e.g. 20240402_sweagent_gpt4)")
    parser.add_argument("--models", type=str, nargs="+", help="Multiple model ids")
    parser.add_argument("--output", type=Path, help="Output path for single model (JSON)")
    parser.add_argument("--output-dir", type=Path, help="Output dir for multiple models (one JSON per model)")
    args = parser.parse_args()

    models = []
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    else:
        # List available models
        split_dir = args.experiments_dir / "evaluation" / args.split
        if not split_dir.exists():
            print(f"Split dir not found: {split_dir}", file=sys.stderr)
            sys.exit(1)
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and (d / "results" / "results.json").exists():
                models.append(d.name)
        if not models:
            print("No models with results found.", file=sys.stderr)
            sys.exit(1)
        print(f"Available models ({len(models)}): {models[:15]}{'...' if len(models) > 15 else ''}")
        if not args.output and not args.output_dir:
            return

    if not models:
        print("Specify --model or --models", file=sys.stderr)
        sys.exit(1)

    for model_id in models:
        try:
            pass_fail = load_model_results(args.experiments_dir, args.split, model_id)
        except FileNotFoundError as e:
            print(f"Skip {model_id}: {e}", file=sys.stderr)
            continue

        # Output format: list of {instance_id, resolved} for compatibility
        records = [{"instance_id": iid, "resolved": passed} for iid, passed in pass_fail.items()]
        n_resolved = sum(1 for p in pass_fail.values() if p)
        print(f"{model_id}: {n_resolved}/{len(pass_fail)} resolved ({100*n_resolved/len(pass_fail):.1f}%)")

        if args.output and len(models) == 1:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(records, f, indent=2)
            print(f"Wrote {args.output}")
        elif args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / f"{args.split}_{model_id}.json"
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
