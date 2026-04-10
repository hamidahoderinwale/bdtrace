#!/usr/bin/env python3
"""
Label each SWE-bench Lite instance with a fix type.

Pipeline:
1. Load raw patches from HF (princeton-nlp/SWE-bench_Lite).
2. Load problem statements from the same dataset.
3. Extract AST stage features from each patch.
4. Call LLM (DSPy) for fix_type classification.
5. Write output/datasets/swe_bench_lite_resolved/fix_types.json.

Usage:
    uv run python scripts/label_fix_types.py
    uv run python scripts/label_fix_types.py --limit 20 --model openai/gpt-4o-mini
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_paths = [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]
for _p in _env_paths:
    if _p.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_p)
        except ImportError:
            pass
        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process first N instances")
    parser.add_argument("--model", type=str, default=None, help="LM model override (e.g. openai/gpt-4o-mini)")
    parser.add_argument("--output", type=str, default=None, help="Output path override")
    parser.add_argument("--workers", type=int, default=4, help="Parallel LLM workers")
    parser.add_argument("--resume", action="store_true", help="Skip already-labeled instances")
    parser.add_argument(
        "--dataset", type=str, default="swe_bench_lite",
        choices=["swe_bench_lite", "swe_smith", "swe_bench_verified"],
        help="Which dataset to label (default: swe_bench_lite)",
    )
    args = parser.parse_args()

    from configs.dspy_config import configure_dspy
    configure_dspy(model=args.model)

    from datasets import load_dataset
    from representations.inferred.fix_type import FixTypeModule, extract_ast_stage, FIX_TYPES

    HF_CONFIGS = {
        "swe_bench_lite": ("princeton-nlp/SWE-bench_Lite", "test",  "output/datasets/swe_bench_lite_resolved/fix_types.json"),
        "swe_smith":      ("SWE-bench/SWE-smith",           "train", "output/datasets/swe_smith_resolved/fix_types.json"),
        "swe_bench_verified": ("princeton-nlp/SWE-bench_Verified", "test", "output/datasets/swe_bench_verified_resolved_full/fix_types.json"),
    }
    hf_repo, hf_split, default_out = HF_CONFIGS[args.dataset]

    out_path = Path(args.output or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ## Load existing results if resuming
    existing: dict[str, dict] = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        existing = {r["instance_id"]: r for r in data.get("results", [])}
        print(f"Resuming: {len(existing)} already labeled")

    print(f"Loading {args.dataset} from HuggingFace ({hf_repo}, split={hf_split})...")
    ds = load_dataset(hf_repo, split=hf_split)
    rows = list(ds)
    if args.limit:
        rows = rows[: args.limit]

    labeler = FixTypeModule()
    results: list[dict] = []
    errors: list[dict] = []

    for row in tqdm.tqdm(rows, desc="Labeling fix types"):
        iid = row["instance_id"]
        if iid in existing:
            results.append(existing[iid])
            continue

        patch = row.get("patch", "") or ""
        problem = row.get("problem_statement", "") or ""
        repo = row.get("repo", "") or ""

        if not patch.strip():
            results.append({
                "instance_id": iid,
                "repo": repo,
                "fix_type": "other",
                "summary": "no patch",
                "library_pattern": "none",
                "confidence": "low",
                "error": "empty_patch",
            })
            continue

        try:
            stage = extract_ast_stage(patch)
            label = labeler(stage=stage, problem_statement=problem[:400])
            results.append({
                "instance_id": iid,
                "repo": repo,
                "fix_type": label["fix_type"],
                "summary": label["summary"],
                "library_pattern": label["library_pattern"],
                "confidence": label["confidence"],
                # Include key stage signals for downstream auditing
                "n_files_changed": stage["n_files_changed"],
                "is_test_file": stage["is_test_file"],
                "net_lines": stage["net_lines"],
                "added_has_if": stage["added"]["has_if"],
                "added_has_raise": stage["added"]["has_raise"],
                "added_has_try": stage["added"]["has_try"],
                "added_api_calls": stage["added"]["api_calls"],
            })
        except Exception as exc:
            errors.append({"instance_id": iid, "error": str(exc)})
            results.append({
                "instance_id": iid,
                "repo": repo,
                "fix_type": "other",
                "summary": "",
                "library_pattern": "none",
                "confidence": "low",
                "error": str(exc),
            })

    ## Count distribution
    from collections import Counter
    dist = Counter(r["fix_type"] for r in results)

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_instances": len(results),
        "n_errors": len(errors),
        "fix_type_distribution": dict(dist.most_common()),
        "fix_type_vocabulary": FIX_TYPES,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(results)} labels to {out_path}")
    print(f"Errors: {len(errors)}")
    print("\nFix type distribution:")
    for ft, count in dist.most_common():
        print(f"  {ft:20s}: {count:4d}  ({100*count/len(results):.1f}%)")


if __name__ == "__main__":
    main()
