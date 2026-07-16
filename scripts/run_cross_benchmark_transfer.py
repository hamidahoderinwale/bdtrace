#!/usr/bin/env python3
"""
Cross-benchmark transfer: does structural similarity in A predict pass in B?

For instances that appear in both benchmarks, we use distance-to-passed-centroid
from benchmark A to predict pass/fail in benchmark B. Requires overlapping instance_ids.

Usage:
  uv run python scripts/run_cross_benchmark_transfer.py \\
    --source swe_bench_verified_resolved_multifile \\
    --target swe_bench_lite_resolved \\
    --pass-fail-source output/swebench_results/verified_20240402_sweagent_gpt4.json \\
    --pass-fail-target output/swebench_results/lite_20240402_sweagent_gpt4.json \\
    --data-dir output
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.io import load_matrices
from analysis.transfer.saturation import distance_to_passed_centroid


def load_pass_fail(path: Path) -> dict[str, bool]:
    """Load instance_id -> pass (True) / fail (False) mapping."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {k: bool(v) for k, v in data.items()}
    if isinstance(data, list):
        return {r["instance_id"]: bool(r.get("resolved", r.get("pass", False))) for r in data}
    raise ValueError("JSON must be dict or list of records")


def load_benchmark_data(data_dir: Path, dataset: str, pass_fail_path: Path | None) -> dict | None:
    """Load distances, instance_ids, and pass/fail for a dataset."""
    base = data_dir / "datasets" / dataset
    dist_path = base / "distances.parquet"
    lbl_path = base / "labels.parquet"
    if not dist_path.exists() or not lbl_path.exists():
        return None

    import pandas as pd

    matrices = load_matrices(dist_path)
    D = matrices.get("edits_set_diff") if matrices.get("edits_set_diff") is not None else \
        matrices.get("edits") if matrices.get("edits") is not None else \
        next(iter(matrices.values()))
    df = pd.read_parquet(lbl_path)
    instance_ids = (
        df["instance_id"].astype(str).tolist()
        if "instance_id" in df.columns
        else [str(i) for i in range(len(df))]
    )

    if pass_fail_path and pass_fail_path.exists():
        pf = load_pass_fail(pass_fail_path)
        passed = np.array([pf.get(iid, False) for iid in instance_ids])
    else:
        return None

    return {
        "D": D,
        "instance_ids": instance_ids,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark transfer analysis")
    parser.add_argument("--source", "-s", required=True, help="Source dataset (train on this)")
    parser.add_argument("--target", "-t", required=True, help="Target dataset (predict on this)")
    parser.add_argument("--pass-fail-source", type=Path, required=True, help="Pass/fail for source")
    parser.add_argument("--pass-fail-target", type=Path, required=True, help="Pass/fail for target")
    parser.add_argument("--data-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    src = load_benchmark_data(args.data_dir, args.source, args.pass_fail_source)
    tgt = load_benchmark_data(args.data_dir, args.target, args.pass_fail_target)
    if src is None:
        print(f"Could not load source {args.source}", file=sys.stderr)
        sys.exit(1)
    if tgt is None:
        print(f"Could not load target {args.target}", file=sys.stderr)
        sys.exit(1)

    src_ids = {iid: i for i, iid in enumerate(src["instance_ids"])}
    tgt_ids = {iid: i for i, iid in enumerate(tgt["instance_ids"])}
    overlap = [iid for iid in src_ids if iid in tgt_ids]
    if not overlap:
        print("No overlapping instance_ids between source and target.", file=sys.stderr)
        sys.exit(1)
    print(f"Overlap: {len(overlap)} instances")

    dist_src = distance_to_passed_centroid(src["D"], src["passed"])
    passed_src = src["passed"]
    passed_tgt = tgt["passed"]

    # Calibrate threshold on source (using overlap instances that are in source)
    dist_overlap = []
    passed_src_overlap = []
    passed_tgt_overlap = []
    for iid in overlap:
        i_src = src_ids[iid]
        i_tgt = tgt_ids[iid]
        dist_overlap.append(dist_src[i_src])
        passed_src_overlap.append(passed_src[i_src])
        passed_tgt_overlap.append(passed_tgt[i_tgt])

    dist_overlap = np.array(dist_overlap)
    passed_src_overlap = np.array(passed_src_overlap)
    passed_tgt_overlap = np.array(passed_tgt_overlap)

    # Find best threshold (maximize accuracy on source labels within overlap)
    thresholds = np.percentile(dist_overlap, np.linspace(5, 95, 19))
    best_acc = 0
    best_thr = 0
    for thr in thresholds:
        pred = dist_overlap < thr
        acc = np.mean(pred == passed_src_overlap)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    # Transfer: predict target from source distance
    pred_tgt = dist_overlap < best_thr
    transfer_acc = np.mean(pred_tgt == passed_tgt_overlap)
    n_correct = int(np.sum(pred_tgt == passed_tgt_overlap))

    print(f"Threshold (calibrated on source): {best_thr:.4f}")
    print(f"Source accuracy (on overlap): {best_acc:.2%}")
    print(f"Transfer accuracy (A->B on overlap): {transfer_acc:.2%} ({n_correct}/{len(overlap)})")


if __name__ == "__main__":
    main()
