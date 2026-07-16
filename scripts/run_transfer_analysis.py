#!/usr/bin/env python3
"""
Run transfer / eval saturation analysis.

If an agent passes structurally similar tasks, new tasks in the same structural
region are likely to pass. Uses distance-to-passed-centroid and kNN transfer.

Usage:
  python scripts/run_transfer_analysis.py --dataset swe_bench_verified_resolved_multifile --synthetic-pass-rate 0.6
  python scripts/run_transfer_analysis.py \\
    --distances output/datasets/swe_bench_verified_resolved_multifile/distances.parquet \\
    --labels output/datasets/swe_bench_verified_resolved_multifile/labels.parquet \\
    --pass-fail output/model_results.json \\
    --output output/datasets/swe_bench_verified_resolved_multifile/transfer_metrics.json

Pass/fail format (one of):
  - JSON: {"instance_id": true, "instance_id2": false, ...}
  - JSON: [{"instance_id": "...", "resolved": true}, ...]
  - JSONL: {"instance_id": "...", "resolved": true} per line
  - Parquet/CSV: columns instance_id, resolved (or pass)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.io import load_matrices
from analysis.transfer.saturation import run_transfer_analysis


def load_pass_fail(path: Path) -> dict[str, bool]:
    """Load instance_id -> pass (True) / fail (False) mapping."""
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: bool(v) for k, v in data.items()}
        if isinstance(data, list):
            return {r["instance_id"]: bool(r.get("resolved", r.get("pass", False))) for r in data}
        raise ValueError("JSON must be dict or list of records")

    if path.suffix == ".jsonl":
        result = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                result[r["instance_id"]] = bool(r.get("resolved", r.get("pass", False)))
        return result

    if path.suffix in (".parquet", ".csv"):
        import pandas as pd

        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        col = "resolved" if "resolved" in df.columns else "pass"
        if col not in df.columns:
            raise ValueError(f"Need 'resolved' or 'pass' column. Got: {list(df.columns)}")
        return dict(zip(df["instance_id"].astype(str), df[col].astype(bool)))

    raise ValueError(f"Unsupported format: {path.suffix}")


def load_labels_with_ids(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load stratum labels and instance_ids from parquet."""
    import pandas as pd

    df = pd.read_parquet(path)
    stratum = np.array(df["stratum"].tolist()) if "stratum" in df.columns else np.array(df.iloc[:, -1].tolist())
    instance_ids = (
        df["instance_id"].astype(str).tolist()
        if "instance_id" in df.columns
        else [str(i) for i in range(len(df))]
    )
    return stratum, instance_ids


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Transfer / eval saturation analysis")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. swe_bench_verified_resolved_multifile). Infers paths when set.")
    parser.add_argument("--distances", "-d", type=Path, default=None, help="distances.parquet or npz (required if --dataset not set)")
    parser.add_argument("--labels", "-l", type=Path, default=None, help="labels.parquet (required if --dataset not set)")
    parser.add_argument("--pass-fail", "-p", type=Path, default=None, help="Pass/fail per instance (JSON/JSONL/parquet)")
    parser.add_argument("--synthetic-pass-rate", type=float, default=None, help="If set, use random pass/fail for testing (e.g. 0.7)")
    parser.add_argument("--repr", type=str, default="edits_set_diff", help="Representation to use (e.g. edits_set_diff, modules_graph)")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Base output dir for dataset-centric paths")
    parser.add_argument("--per-instance", action="store_true", help="Include per-instance output")
    args = parser.parse_args()

    if args.dataset:
        base = args.output_dir / "datasets" / args.dataset
        args.distances = args.distances or base / "distances.parquet"
        args.labels = args.labels or base / "labels.parquet"
        if args.output is None:
            args.output = base / "transfer_metrics.json"
    if not args.distances or not args.labels:
        parser.error("Provide --dataset or both --distances and --labels")

    matrices = load_matrices(args.distances)
    if args.repr not in matrices:
        avail = list(matrices.keys())
        print(f"Representation '{args.repr}' not found. Available: {avail}", file=sys.stderr)
        if avail:
            args.repr = avail[0]
            print(f"Using '{args.repr}'", file=sys.stderr)

    D = matrices[args.repr]
    region_labels, instance_ids = load_labels_with_ids(args.labels)
    n = D.shape[0]

    if args.synthetic_pass_rate is not None:
        rng = np.random.default_rng(42)
        passed_mask = rng.random(n) < args.synthetic_pass_rate
        print(f"Using synthetic pass/fail (rate={args.synthetic_pass_rate})", file=sys.stderr)
    elif args.pass_fail:
        pass_fail_map = load_pass_fail(args.pass_fail)
        passed_mask = np.zeros(n, dtype=bool)
        matched = 0
        for i, iid in enumerate(instance_ids):
            if iid in pass_fail_map:
                passed_mask[i] = pass_fail_map[iid]
                matched += 1
        if matched == 0:
            print("No instance_id matches between labels and pass-fail. Check instance_id format.", file=sys.stderr)
            sys.exit(1)
        print(f"Matched {matched}/{n} instances with pass/fail data", file=sys.stderr)
    else:
        print("Provide --pass-fail or --synthetic-pass-rate", file=sys.stderr)
        sys.exit(1)

    results = run_transfer_analysis(
        D,
        passed_mask,
        instance_ids=instance_ids if args.per_instance else None,
        region_labels=region_labels,
        repr_name=args.repr,
    )

    out = json.dumps(results, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out)
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
