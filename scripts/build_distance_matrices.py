#!/usr/bin/env python3
"""
Build distance matrices from certificate records.

Loads parquet or JSON, computes pairwise distances for edits, modules, motifs.
Output: Parquet (primary) + npz (for diversity script).

Parquet schema:
  distances.parquet: i, j, d_edits, d_modules, d_motifs [, d_tokens, d_edits_tree, d_modules_graph]
  labels.parquet: index, instance_id, stratum

Usage:
  python scripts/build_distance_matrices.py --input output/datasets/swe_bench_lite/test.parquet
  python scripts/build_distance_matrices.py --input output/datasets/swe_bench_lite/test.parquet --approach structural
  python scripts/build_distance_matrices.py --input output/datasets/swe_bench_lite/test.parquet --approach both
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.procedures.divergence_matrix import (
    APPROACH_BOTH,
    APPROACH_JACCARD,
    APPROACH_STRUCTURAL,
    build_distance_matrices,
)


def load_records(path: Path) -> list[dict]:
    """Load records from parquet or JSON."""
    json_cols = ["edits", "modules", "motifs", "tokens", "modules_edges"]
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        records = df.to_dict("records")
        for rec in records:
            for col in json_cols:
                if col in rec and isinstance(rec[col], str):
                    try:
                        rec[col] = json.loads(rec[col])
                    except json.JSONDecodeError:
                        pass
        return records

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return data.get("records", data) if isinstance(data, dict) else data

    raise ValueError(f"Unsupported format: {path.suffix}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build distance matrices from certificates")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Parquet or JSON with records")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output dir (default: same as input)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--approach",
        choices=[APPROACH_JACCARD, APPROACH_STRUCTURAL, APPROACH_BOTH],
        default=APPROACH_STRUCTURAL,
        help="Distance approach: structural (default), jaccard, or both",
    )
    parser.add_argument(
        "--reprs",
        nargs="*",
        default=None,
        help="Restrict to these representations only (e.g. tokens edits_tree modules_graph)",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]

    if len(records) < 2:
        print("Need at least 2 records")
        sys.exit(1)

    matrices, labels = build_distance_matrices(
        records, approach=args.approach, repr_keys=args.reprs
    )

    out_dir = args.output or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet (primary): queryable, portable, schema-preserving
    import pandas as pd

    n = len(records)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            row = {"i": i, "j": j}
            for k, D in matrices.items():
                row[f"d_{k}"] = float(D[i, j])
            rows.append(row)
    dist_df = pd.DataFrame(rows)
    dist_df.to_parquet(out_dir / "distances.parquet", index=False)

    labels_df = pd.DataFrame({
        "index": range(n),
        "instance_id": [r.get("instance_id", "") for r in records],
        "stratum": labels.tolist(),
    })
    labels_df.to_parquet(out_dir / "labels.parquet", index=False)

    # npz (for diversity script backward compat)
    np.savez(out_dir / "matrices.npz", **matrices)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels.tolist(), f)

    print(f"Saved to {out_dir}")
    print(f"  distances.parquet: {len(rows)} pairs, {list(matrices.keys())}")
    print(f"  labels.parquet: {n} rows, {len(set(labels))} strata")

if __name__ == "__main__":
    main()
