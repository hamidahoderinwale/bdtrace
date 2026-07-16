#!/usr/bin/env python3
"""
Run procedural divergence analysis.

For each instance and each procedure pair (behavioral, mechanistic, functional),
computes divergence at terminal stage and procedural summary S(P_a, P_b, stage).

Requires records with edits + behavioral, mechanistic, functional (from eval/run_eval.py).

Usage:
  python scripts/run_procedure_divergence.py --input records.jsonl --output divergence.parquet
  python scripts/run_procedure_divergence.py --input output/eval_results.json --threshold 0.3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.procedures.procedure_divergence import build_procedure_divergence_matrix


def load_records(path: Path) -> list[dict]:
    """Load records from JSONL or JSON array."""
    with open(path) as f:
        if path.suffix == ".jsonl":
            records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return records
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return [data]


def main():
    parser = argparse.ArgumentParser(description="Procedural divergence analysis")
    parser.add_argument("--input", type=Path, required=True, help="JSONL or JSON of records")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.3, help="Annotation divergence threshold")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]

    if not records:
        print("No records loaded.")
        sys.exit(1)

    results = build_procedure_divergence_matrix(
        records,
        procedures=["behavioral", "mechanistic", "functional"],
        length=3,
        annotation_threshold=args.threshold,
    )

    flat = []
    for r in results:
        flat.append({
            "instance_id": r["instance_id"],
            "proc_a": r["proc_a"],
            "proc_b": r["proc_b"],
            "terminal_diverged": r["terminal_diverged"],
            "terminal_distance": r["terminal_distance"],
            "structural_agreement": r.get("structural_agreement"),
            "semantic_agreement": r.get("semantic_agreement"),
            "gap": r.get("gap"),
            "structural_diverged": r["S"]["structural_diverged"],
            "annotation_introduced": r["S"]["annotation_introduced"],
        })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".parquet":
            import pandas as pd
            df = pd.DataFrame(flat)
            df.to_parquet(args.output, index=False)
        else:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
        print(f"Wrote {len(flat)} procedure-pair results to {args.output}")
    else:
        n_diverged = sum(1 for r in flat if r["terminal_diverged"])
        print(f"Instances: {len(records)}, procedure pairs: {len(flat)}, diverged: {n_diverged}")


if __name__ == "__main__":
    main()
