#!/usr/bin/env python3
"""
Generate free-form structural descriptions for each diff hunk across all traces.

Uses ChunkDescriber (DSPy) to produce one grounded sentence per hunk, with no
pre-specified vocabulary. The resulting descriptions are then clustered in
cluster_hunk_descriptions.py to discover mechanism categories from data.

Outputs:
  output/hunk_descriptions/descriptions.json  — {instance_id: [sentence, ...]}

Usage:
  uv run python scripts/build_hunk_descriptions.py
  uv run python scripts/build_hunk_descriptions.py --limit 50  # quick test
  uv run python scripts/build_hunk_descriptions.py --resume
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _env in [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
        except ImportError:
            pass

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "hunk_descriptions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--traces", type=str,
        default="output/resolved_traces_lite_full.jsonl",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from configs.dspy_config import configure_dspy
    configure_dspy(model=args.model)

    from representations.inferred.fix_type.chunk_describer import ChunkDescriber

    descriptions_path = OUTPUT_DIR / "descriptions.json"
    existing: dict[str, list[str]] = {}
    if args.resume and descriptions_path.exists():
        with open(descriptions_path) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already described")

    describer = ChunkDescriber()

    traces_path = ROOT / args.traces
    print(f"Loading traces from {traces_path}...")

    results: dict[str, list[str]] = dict(existing)
    n_processed = 0

    with open(traces_path) as f:
        for line in f:
            if args.limit and n_processed >= args.limit:
                break
            trace = json.loads(line)
            iid = trace["instance_id"]
            if iid in existing:
                continue
            descs = describer.describe_trace(trace)
            results[iid] = descs
            n_processed += 1
            if n_processed % 10 == 0:
                print(f"  {n_processed} processed, last: {iid} → {len(descs)} hunks")
                with open(descriptions_path, "w") as f2:
                    json.dump(results, f2, indent=2)

    with open(descriptions_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved descriptions.json ({len(results)} instances)")

    all_descs = [d for descs in results.values() for d in descs]
    print(f"Total hunk descriptions: {len(all_descs)}")
    non_empty = [iid for iid, descs in results.items() if descs]
    print(f"Instances with >=1 description: {len(non_empty)}/{len(results)}")
    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
