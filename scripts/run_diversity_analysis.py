#!/usr/bin/env python3
"""
Run diversity analysis on distance matrices.

Input: distances (parquet or npz) + labels (parquet, json, or npy).
Output: Parquet (primary) + JSON (full nested).

Parquet schema:
  diversity_metrics.parquet: repr, stratum_ratio, stratum_overlap, silhouette, unique_variance
  rank_correlation.parquet: repr_i, repr_j, rho
  per_instance_rep_correlation.parquet: index, instance_id, stratum, mean_rho, min_rho
  per_instance_pair_rho.parquet: index, instance_id, repr_i, repr_j, rho

Usage:
  python scripts/run_diversity_analysis.py --matrices output/distances.parquet --labels output/labels.parquet
  python scripts/run_diversity_analysis.py --matrices output/matrices.npz --labels output/labels.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.diversity import run_diversity_analysis
from analysis.io import load_labels, load_matrices


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diversity analysis on distance matrices")
    parser.add_argument(
        "--matrices",
        type=Path,
        required=True,
        help="distances.parquet, matrices.npz, or json",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="labels.parquet, labels.json, or npy",
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output dir (default: same as matrices)")
    args = parser.parse_args()

    matrices = load_matrices(args.matrices)
    if len(matrices) < 2:
        print("Need at least 2 distance matrices")
        sys.exit(1)

    labels = load_labels(args.labels)
    n = next(iter(matrices.values())).shape[0]
    if len(labels) != n:
        print("Labels length must match matrix dimension")
        sys.exit(1)

    results = run_diversity_analysis(matrices, labels)

    out_dir = args.output or args.matrices.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet (primary)
    import pandas as pd

    metrics_rows = []
    for repr_name in matrices:
        metrics_rows.append({
            "repr": repr_name,
            "stratum_ratio": results["stratum_ratios"].get(repr_name, float("nan")),
            "stratum_overlap": results.get("stratum_overlaps", {}).get(repr_name, float("nan")),
            "silhouette": results["silhouette_scores"].get(repr_name, float("nan")),
            "unique_variance": results["unique_variances"].get(repr_name, float("nan")),
        })
    pd.DataFrame(metrics_rows).to_parquet(out_dir / "diversity_metrics.parquet", index=False)

    names = list(matrices.keys())
    rho = np.array(results["rank_correlation"])
    rank_rows = []
    for i in range(len(names)):
        for j in range(len(names)):
            rank_rows.append({"repr_i": names[i], "repr_j": names[j], "rho": float(rho[i, j])})
    pd.DataFrame(rank_rows).to_parquet(out_dir / "rank_correlation.parquet", index=False)

    # Per-instance representation correlation
    pic = results.get("per_instance_rep_correlation", {})
    if pic:
        n = len(pic.get("mean_rho", []))
        instance_ids = [""] * n
        strata = list(labels) if hasattr(labels, "__len__") else [""] * n
        if args.labels.suffix == ".parquet":
            try:
                lbl_df = pd.read_parquet(args.labels)
                if "instance_id" in lbl_df.columns:
                    instance_ids = lbl_df["instance_id"].tolist()
                if "stratum" in lbl_df.columns:
                    strata = lbl_df["stratum"].tolist()
            except Exception:
                pass
        pi_rows = []
        for k in range(n):
            pi_rows.append({
                "index": k,
                "instance_id": instance_ids[k] if k < len(instance_ids) else "",
                "stratum": strata[k] if k < len(strata) else "",
                "mean_rho": float(pic["mean_rho"][k]) if not np.isnan(pic["mean_rho"][k]) else None,
                "min_rho": float(pic["min_rho"][k]) if not np.isnan(pic["min_rho"][k]) else None,
            })
        pd.DataFrame(pi_rows).to_parquet(out_dir / "per_instance_rep_correlation.parquet", index=False)

        pair_rows = []
        for (ri, rj), arr in pic.get("pair_rho", {}).items():
            for k in range(len(arr)):
                v = arr[k]
                if not np.isnan(v):
                    pair_rows.append({
                        "index": k,
                        "instance_id": instance_ids[k] if k < len(instance_ids) else "",
                        "repr_i": ri,
                        "repr_j": rj,
                        "rho": float(v),
                    })
        if pair_rows:
            pd.DataFrame(pair_rows).to_parquet(out_dir / "per_instance_pair_rho.parquet", index=False)

    # JSON (full nested, for inspection)
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                key = f"{k[0]}|{k[1]}" if isinstance(k, tuple) else str(k)
                out[key] = to_serializable(v)
            return out
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    with open(out_dir / "diversity_results.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    out_files = ["diversity_metrics.parquet", "rank_correlation.parquet", "diversity_results.json"]
    if pic:
        out_files.extend(["per_instance_rep_correlation.parquet", "per_instance_pair_rho.parquet"])
    print(f"Saved to {out_dir}")
    print(f"  {', '.join(out_files)}")


if __name__ == "__main__":
    main()
