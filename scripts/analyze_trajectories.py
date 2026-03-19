#!/usr/bin/env python3
"""
Trajectory procedural analysis.

Given fetched trajectory features (from fetch_trajectories.py), produces:
1. Efficiency metrics: pass rate by edit-retry bucket
2. Hop distance: distribution of fix-site distance from test by outcome
3. Sequence alignment: edit distance from each trajectory to the modal
   "good procedure" template for its fix-type cluster
4. Cross-model comparison table

Reuses:
- output/trajectories/: parquet files from fetch_trajectories.py
- output/datasets/swe_bench_lite_resolved/: distances + labels (fix clustering)
- analysis/transfer/saturation.py: AUC + kNN utilities

Usage:
  uv run python scripts/analyze_trajectories.py
  uv run python scripts/analyze_trajectories.py --models 20240402_sweagent_gpt4 20240620_sweagent_claude3.5sonnet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = ROOT / "output" / "trajectories"
PLOTS_DIR = ROOT / "notebooks" / "plots" / "trajectories"


def _action_edit_distance(seq_a: str, seq_b: str) -> int:
    """Levenshtein distance on space-split action token sequences."""
    a, b = seq_a.split(), seq_b.split()
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def analysis_efficiency(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Pass rate by edit-retry bucket."""
    sub = df[df["model_id"] == model_id].copy()
    sub["retry_bucket"] = pd.cut(
        sub["edit_retries"],
        bins=[-1, 0, 2, 5, 10, 999],
        labels=["0", "1-2", "3-5", "6-10", "11+"],
    )
    return (
        sub.groupby("retry_bucket", observed=True)
        .agg(n=("passed", "count"), n_pass=("passed", "sum"), pass_rate=("passed", "mean"))
        .reset_index()
        .assign(model_id=model_id)
    )


def analysis_hop_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Mean hop distance by outcome and model."""
    sub = df[df["hop_distance_min"].notna()].copy()
    return (
        sub.groupby(["model_id", "passed"])["hop_distance_min"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )


def analysis_sequence_alignment(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    For passed instances, compute the modal action sequence (template).
    Then measure edit distance from each trajectory to that template.
    Check if closer-to-template → higher pass rate.
    """
    sub = df[df["model_id"] == model_id].copy()
    passed = sub[sub["passed"] == True]
    if len(passed) < 3:
        return pd.DataFrame()

    # Modal template: most common sequence among passed instances
    template = passed["action_sequence"].mode().iloc[0]

    sub["seq_dist_to_template"] = sub["action_sequence"].apply(
        lambda s: _action_edit_distance(s, template)
    )

    # Bucket by distance
    sub["seq_bucket"] = pd.cut(
        sub["seq_dist_to_template"],
        bins=5,
        labels=["Q1 (closest)", "Q2", "Q3", "Q4", "Q5 (farthest)"],
    )
    result = (
        sub.groupby("seq_bucket", observed=True)
        .agg(n=("passed", "count"), n_pass=("passed", "sum"), pass_rate=("passed", "mean"))
        .reset_index()
        .assign(model_id=model_id, template=template)
    )
    return result


def analysis_cross_model(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table comparing models across key procedural metrics."""
    rows = []
    for model_id, g in df.groupby("model_id"):
        passed = g[g["passed"] == True]
        failed = g[g["passed"] == False]

        def _m(col, subset):
            return subset[col].mean() if len(subset) > 0 else float("nan")

        rows.append({
            "model_id": model_id,
            "n_total": len(g),
            "n_passed": len(passed),
            "pass_rate": len(passed) / len(g) if len(g) else 0,
            "steps_pass": _m("n_steps", passed),
            "steps_fail": _m("n_steps", failed),
            "edits_pass": _m("n_edits", passed),
            "edits_fail": _m("n_edits", failed),
            "retries_pass": _m("edit_retries", passed),
            "retries_fail": _m("edit_retries", failed),
            "retry_rate_pass": _m("edit_retry_rate", passed),
            "retry_rate_fail": _m("edit_retry_rate", failed),
            "submitted_rate": g["submitted"].mean(),
        })
    return pd.DataFrame(rows)


def plot_efficiency(eff_frames: list[pd.DataFrame], output_dir: Path) -> None:
    import altair as alt

    df = pd.concat(eff_frames, ignore_index=True)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("retry_bucket:O", title="Edit retries"),
            y=alt.Y("pass_rate:Q", title="Pass rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_id:N"),
            column=alt.Column("model_id:N"),
            tooltip=["retry_bucket", "pass_rate", "n", "n_pass"],
        )
        .properties(title="Pass rate by edit-retry count", width=180)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_dir / "efficiency_by_retries.html"))
    print(f"  Saved efficiency_by_retries.html")


def plot_cross_model(cross: pd.DataFrame, output_dir: Path) -> None:
    import altair as alt

    # Melt for grouped bar chart
    metric_cols = ["steps_pass", "steps_fail", "retries_pass", "retries_fail", "retry_rate_pass", "retry_rate_fail"]
    melted = cross[["model_id"] + metric_cols].melt(id_vars="model_id", var_name="metric", value_name="value")
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("model_id:N", title=None),
            y=alt.Y("value:Q"),
            color="model_id:N",
            column=alt.Column("metric:N", title=None),
            tooltip=["model_id", "metric", "value"],
        )
        .properties(title="Procedural metrics: pass vs fail by model", width=100)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_dir / "cross_model_comparison.html"))
    print(f"  Saved cross_model_comparison.html")


def main():
    parser = argparse.ArgumentParser(description="Trajectory procedural analysis")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["20240402_sweagent_gpt4", "20240620_sweagent_claude3.5sonnet"],
    )
    parser.add_argument("--split", default="lite")
    parser.add_argument("--traj-dir", type=Path, default=TRAJ_DIR)
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()

    # Load available parquets
    frames = []
    for model_id in args.models:
        p = args.traj_dir / f"{args.split}_{model_id}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            frames.append(df)
            print(f"Loaded {model_id}: {len(df)} rows, {df['passed'].sum()} passed")
        else:
            print(f"Missing: {p}")

    if not frames:
        print("No trajectory data found. Run fetch_trajectories.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(df)} rows across {df['model_id'].nunique()} models\n")

    # 1. Efficiency
    print("=== Efficiency: pass rate by edit-retry bucket ===")
    eff_frames = []
    for model_id in df["model_id"].unique():
        eff = analysis_efficiency(df, model_id)
        eff_frames.append(eff)
        print(f"\n{model_id}")
        print(eff[["retry_bucket", "n", "n_pass", "pass_rate"]].to_string(index=False))

    # 2. Hop distance
    print("\n=== Hop distance from test to fix site ===")
    hop = analysis_hop_distance(df)
    if len(hop):
        print(hop.to_string(index=False))
    else:
        print("  No hop distance data (repos not available during fetch)")

    # 3. Sequence alignment
    print("\n=== Sequence alignment to passed template ===")
    seq_frames = []
    for model_id in df["model_id"].unique():
        seq = analysis_sequence_alignment(df, model_id)
        if len(seq):
            seq_frames.append(seq)
            print(f"\n{model_id} (template: {seq['template'].iloc[0]!r})")
            print(seq[["seq_bucket", "n", "n_pass", "pass_rate"]].to_string(index=False))

    # 4. Cross-model summary
    print("\n=== Cross-model comparison ===")
    cross = analysis_cross_model(df)
    print(cross.to_string(index=False))

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cross.to_csv(args.output_dir / "cross_model_summary.csv", index=False)
    if eff_frames:
        pd.concat(eff_frames).to_csv(args.output_dir / "efficiency_by_retries.csv", index=False)
    print(f"\nSaved CSVs to {args.output_dir}")

    # Plots
    try:
        plot_efficiency(eff_frames, args.output_dir)
        plot_cross_model(cross, args.output_dir)
    except ImportError:
        print("altair not available, skipping HTML plots")

    # Save combined parquet
    out = args.traj_dir / f"{args.split}_all_models.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved combined parquet: {out}")


if __name__ == "__main__":
    main()
