#!/usr/bin/env python3
"""
Cross-dataset coverage and saturation analysis.

Measures how well a base dataset (e.g. SWE-bench Lite) covers the problem space
of a comparison dataset (e.g. SWE-bench Verified, SWE-Smith) in representation space.

A. Coverage curve: for each instance in --add, find its nearest neighbor in --base.
   Coverage(τ) = fraction of --add within τ of some --base instance.

B. Saturation curve: greedy farthest-first ordering of --add instances relative to
   --base. Shows how many instances from --add are needed before all problem types
   in --add are "covered."

Outputs (saved to --output-dir):
  coverage_nn.parquet          per-instance nearest-neighbor distances (B→A)
  coverage_curve.parquet       (tau, fraction) coverage curve
  saturation_curve.parquet     (step, n_added, coverage_fraction) saturation curve
  saturation_summary.json      scalar summary: tau, initial/final coverage, steps to 50/90%

Usage:
  uv run python scripts/run_coverage_analysis.py
  uv run python scripts/run_coverage_analysis.py \\
    --base swe_bench_lite_resolved \\
    --add swe_bench_verified_resolved_full \\
    --repr motifs
  uv run python scripts/run_coverage_analysis.py \\
    --base swe_bench_lite_resolved \\
    --add swe_smith_resolved \\
    --repr edits_set_diff \\
    --tau 0.4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.io import load_matrices
from analysis.transfer.coverage import (
    coverage_curve,
    cross_dataset_distances,
    nn_coverage,
    repr_sparsity,
    saturation_curve,
)

SPLIT_FOR = {
    "swe_bench_lite_resolved": "test",
    "swe_bench_lite": "test",
    "swe_bench_verified_resolved_multifile": "test",
    "swe_bench_verified_resolved_full": "test",
    "swe_smith_resolved": "train",
    "swe_smith_stratified": "train",
    "humaneval": "test",
    "mbpp": "test",
}

SPARSE_THRESHOLD = 0.2  # warn if fewer than 20% of records have non-empty repr


def load_records(data_dir: Path, dataset: str) -> list[dict]:
    split = SPLIT_FOR.get(dataset, "test")
    path = data_dir / "datasets" / dataset / f"{split}.parquet"
    if not path.exists():
        # try the other split
        alt = "train" if split == "test" else "test"
        path = data_dir / "datasets" / dataset / f"{alt}.parquet"
    if not path.exists():
        print(f"ERROR: no parquet found for {dataset} in {data_dir / 'datasets' / dataset}", file=sys.stderr)
        sys.exit(1)
    import pandas as pd
    import json as _json
    df = pd.read_parquet(path)
    records = df.to_dict("records")
    for col in ("edits", "modules", "motifs", "tokens", "modules_edges"):
        for r in records:
            if col in r and isinstance(r[col], str):
                try:
                    r[col] = _json.loads(r[col])
                except Exception:
                    pass
    return records


def load_intra_distances(data_dir: Path, dataset: str, repr_key: str) -> np.ndarray | None:
    """Load pre-computed intra-dataset distance matrix for repr_key.

    motifs_sequence has no pre-computed equivalent — returns None so the caller
    recomputes from records.
    """
    if repr_key == "motifs_sequence":
        return None
    dist_path = data_dir / "datasets" / dataset / "distances.parquet"
    if not dist_path.exists():
        return None
    matrices = load_matrices(dist_path)
    D = matrices.get(repr_key)
    if D is None and repr_key == "edits_set_diff":
        D = matrices.get("edits")
    if D is None:
        key = next(iter(matrices), None)
        if key:
            print(f"  repr '{repr_key}' not in {dataset} distances; using '{key}'")
            D = matrices[key]
    return D


def select_repr(records_a: list[dict], records_b: list[dict], requested: str) -> str:
    """Warn if requested repr is sparse; suggest fallback."""
    sp_a = repr_sparsity(records_a, requested)
    sp_b = repr_sparsity(records_b, requested)
    if sp_a < SPARSE_THRESHOLD or sp_b < SPARSE_THRESHOLD:
        print(
            f"WARNING: repr '{requested}' is sparse "
            f"(base={sp_a:.0%}, add={sp_b:.0%}). "
            "Consider --repr edits_set_diff instead."
        )
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset coverage and saturation")
    parser.add_argument("--base", default="swe_bench_lite_resolved",
                        help="Base dataset (A). Default: swe_bench_lite_resolved")
    parser.add_argument("--add", default="swe_bench_verified_resolved_full",
                        help="Comparison dataset (B). Default: swe_bench_verified_resolved_full")
    parser.add_argument("--repr", default="motifs_sequence",
                        choices=["motifs_sequence", "motifs", "edits_set_diff", "edits", "modules"],
                        help="Representation for distance. Default: motifs_sequence")
    parser.add_argument("--tau", type=float, default=None,
                        help="Coverage threshold τ. Default: median(nn_dists)")
    parser.add_argument("--data-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir. Default: output/cross_coverage/<base>_vs_<add>/")
    parser.add_argument("--limit-base", type=int, default=None)
    parser.add_argument("--limit-add", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or (
        args.data_dir / "cross_coverage" / f"{args.base}_vs_{args.add}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.base}...")
    records_a = load_records(args.data_dir, args.base)
    if args.limit_base:
        records_a = records_a[: args.limit_base]

    print(f"Loading {args.add}...")
    records_b = load_records(args.data_dir, args.add)
    if args.limit_add:
        records_b = records_b[: args.limit_add]

    repr_key = select_repr(records_a, records_b, args.repr)
    print(f"Computing cross-dataset distances ({repr_key}, {len(records_a)}×{len(records_b)})...")
    D_ab = cross_dataset_distances(records_a, records_b, repr_key)

    # ── A. Coverage ────────────────────────────────────────────────────────────
    print("Computing coverage...")
    nn_dists, nn_indices = nn_coverage(D_ab)
    thresholds, fractions = coverage_curve(nn_dists)

    import pandas as pd

    # Per-instance nearest-neighbor distances
    nn_df = pd.DataFrame({
        "instance_id_b": [r.get("instance_id", str(j)) for j, r in enumerate(records_b)],
        "nn_dist": nn_dists.tolist(),
        "nn_index_in_a": nn_indices.tolist(),
        "nn_instance_id_a": [
            records_a[int(i)].get("instance_id", str(i)) for i in nn_indices
        ],
    })
    nn_df.to_parquet(out_dir / "coverage_nn.parquet", index=False)

    curve_df = pd.DataFrame({"tau": thresholds.tolist(), "coverage_fraction": fractions.tolist()})
    curve_df.to_parquet(out_dir / "coverage_curve.parquet", index=False)

    tau_used = args.tau if args.tau is not None else float(np.median(nn_dists))
    cov_at_tau = float(np.mean(nn_dists <= tau_used))
    print(f"  Coverage at τ={tau_used:.3f}: {cov_at_tau:.1%} of {args.add} covered by {args.base}")

    # ── B. Saturation curve ────────────────────────────────────────────────────
    print("Loading intra-dataset distances for saturation curve...")
    D_bb = load_intra_distances(args.data_dir, args.add, repr_key)

    if D_bb is None:
        print(f"  No pre-computed distances for {args.add}; computing from records...")
        D_bb_raw = cross_dataset_distances(records_b, records_b, repr_key)
        D_bb = D_bb_raw
    else:
        n_b = len(records_b)
        if D_bb.shape[0] != n_b:
            print(
                f"  WARNING: stored D_bb shape {D_bb.shape} doesn't match n_b={n_b}; "
                "recomputing from records."
            )
            D_bb = cross_dataset_distances(records_b, records_b, repr_key)

    print("Computing saturation curve...")
    sat = saturation_curve(D_ab, D_bb, tau=args.tau)

    sat_df = pd.DataFrame({
        "step": list(range(len(sat["coverage"]))),
        "n_added": list(range(len(sat["coverage"]))),
        "coverage_fraction": sat["coverage"],
    })
    sat_df.to_parquet(out_dir / "saturation_curve.parquet", index=False)

    summary = {
        "base": args.base,
        "add": args.add,
        "repr": repr_key,
        "n_base": len(records_a),
        "n_add": len(records_b),
        "tau": sat["tau"],
        "initial_coverage": sat["initial_coverage"],
        "final_coverage": sat["final_coverage"],
        "n_steps_to_50pct": sat["n_steps_to_50pct"],
        "n_steps_to_90pct": sat["n_steps_to_90pct"],
    }
    with open(out_dir / "saturation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaturation summary (τ={sat['tau']:.3f}):")
    print(f"  Initial coverage (A alone):  {sat['initial_coverage']:.1%}")
    print(f"  Final coverage (all B added): {sat['final_coverage']:.1%}")
    if sat["n_steps_to_50pct"] is not None:
        print(f"  Steps to 50% coverage: {sat['n_steps_to_50pct']}")
    if sat["n_steps_to_90pct"] is not None:
        print(f"  Steps to 90% coverage: {sat['n_steps_to_90pct']}")
    print(f"\nSaved to {out_dir}/")
    print("  coverage_nn.parquet, coverage_curve.parquet, saturation_curve.parquet, saturation_summary.json")


if __name__ == "__main__":
    main()
