#!/usr/bin/env python3
"""
Charted vs uncharted instance analysis.

Asks: does LLM performance degrade on structurally novel (uncharted) instances,
and does procedural context help close that gap?

Three outputs:
  1. Score by condition x charted/uncharted (per model and pooled)
  2. Charted/uncharted gap by model
  3. Cross-model convergence by condition x charted/uncharted

Usage:
  uv run python scripts/analyze_charted_vs_uncharted.py
  uv run python scripts/analyze_charted_vs_uncharted.py --k 10 --output-dir output/charted_analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "charted_analysis"

SCORE_DIMS = ["plan_quality", "localization", "edit_type", "explanation"]
CONDITIONS = ["no_context", "procedural", "raw_logs"]
CONDITION_LABELS = {"no_context": "No context", "procedural": "Procedural", "raw_logs": "Raw logs"}

MODEL_DIRS = {
    "gpt_4o": "GPT-4o",
    "gpt_4o_mini": "GPT-4o mini",
    "qwen_2.5_72b_instruct": "Qwen 2.5 72B",
    "llama_3.3_70b_instruct": "Llama 3.3 70B",
}

# Project palette
CHARTED_COLOR = "#0C6583"
UNCHARTED_COLOR = "#EE7733"
GRAY = "#AAAAAA"
NAVY = "#2B2D42"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"


# --- Data loading ---

def load_records(model_dir: Path, model_label: str) -> pd.DataFrame:
    path = model_dir / "records.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        records = json.load(f)

    rows = []
    for r in records:
        instance_id = r["instance_id"]
        for cond in CONDITIONS:
            cond_data = r["conditions"].get(cond, {})
            scores = cond_data.get("scores") or {}
            if not scores:
                continue
            row = {
                "instance_id": instance_id,
                "model": model_label,
                "condition": cond,
            }
            for dim in SCORE_DIMS:
                val = scores.get(dim)
                row[dim] = float(val) if val is not None and val != -1 else np.nan
            row["divergence_level"] = scores.get("divergence_level", "")
            rows.append(row)

    return pd.DataFrame(rows)


def load_all_records(study_dir: Path) -> pd.DataFrame:
    frames = []
    for model_key, model_label in MODEL_DIRS.items():
        model_dir = study_dir / model_key
        if not model_dir.exists():
            continue
        df = load_records(model_dir, model_label)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_chartedness(distances_path: Path, labels_path: Path, k: int) -> pd.DataFrame:
    dist = pd.read_parquet(distances_path)
    labels = pd.read_parquet(labels_path)

    # For each instance index, find k nearest neighbors by d_edits
    idx_to_iid = labels.set_index("index")["instance_id"].to_dict()

    rows = []
    for idx in labels["index"]:
        # Rows where this instance appears as either i or j
        mask = (dist["i"] == idx) | (dist["j"] == idx)
        neighbors = dist.loc[mask, "d_edits"].sort_values()
        knn_mean = neighbors.iloc[:k].mean() if len(neighbors) >= k else neighbors.mean()
        rows.append({"instance_id": idx_to_iid[idx], "knn_edit_dist": knn_mean})

    return pd.DataFrame(rows)


def load_fix_types(merged_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(merged_path)
    return df[["instance_id", "fix_type", "confidence", "passed"]]


def load_epistemic(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "epistemic_confidence.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        ec = json.load(f)
    rows = [{"instance_id": iid, "epistemic_confidence": v.get("epistemic_confidence")}
            for iid, v in ec.items()]
    return pd.DataFrame(rows)


# --- Analysis ---

def add_charted_label(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["charted"] = df["knn_edit_dist"] <= threshold
    df["charted_label"] = df["charted"].map({True: "Charted", False: "Uncharted"})
    return df


def compute_gap(df: pd.DataFrame, score: str) -> pd.DataFrame:
    grouped = df.groupby(["model", "condition", "charted_label"])[score].mean().reset_index()
    pivoted = grouped.pivot_table(index=["model", "condition"], columns="charted_label", values=score).reset_index()
    if "Charted" in pivoted.columns and "Uncharted" in pivoted.columns:
        pivoted["gap"] = pivoted["Charted"] - pivoted["Uncharted"]
    return pivoted


def compute_convergence(df: pd.DataFrame, score: str) -> pd.DataFrame:
    models = df["model"].unique()
    rows = []
    for instance_id, grp in df.groupby("instance_id"):
        for cond, cgrp in grp.groupby("condition"):
            model_scores = cgrp.set_index("model")[score].dropna()
            if len(model_scores) < 2:
                continue
            pairs = [(a, b) for i, a in enumerate(model_scores) for b in list(model_scores)[i+1:]]
            vals = list(model_scores.values)
            disagreement = np.mean([abs(vals[i] - vals[j])
                                    for i in range(len(vals)) for j in range(i+1, len(vals))])
            charted_label = cgrp["charted_label"].iloc[0]
            rows.append({
                "instance_id": instance_id,
                "condition": cond,
                "charted_label": charted_label,
                "disagreement": disagreement,
                "n_models": len(model_scores),
            })
    return pd.DataFrame(rows)


# --- Plotting ---

def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig1_scores_by_condition(df: pd.DataFrame, output_dir: Path):
    cond_order = CONDITIONS
    cond_xs = np.arange(len(cond_order))
    width = 0.35

    fig, axes = plt.subplots(1, len(SCORE_DIMS), figsize=(14, 4), sharey=False)
    fig.subplots_adjust(wspace=0.35, bottom=0.22)

    for ax, score in zip(axes, SCORE_DIMS):
        style_panel(ax)
        for offset, label, color in [(-width/2, "Charted", CHARTED_COLOR),
                                     (width/2, "Uncharted", UNCHARTED_COLOR)]:
            means = []
            for cond in cond_order:
                subset = df[(df["condition"] == cond) & (df["charted_label"] == label)]
                means.append(subset[score].mean())
            ax.bar(cond_xs + offset, means, width, color=color, alpha=0.85)

        ax.set_xticks(cond_xs)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in cond_order], fontsize=9)
        ax.set_title(score.replace("_", " ").title(), fontsize=10, pad=6, fontweight="normal")
        ax.set_ylim(1, 3)
        ax.set_ylabel("Mean score (1-3)" if score == SCORE_DIMS[0] else "", fontsize=9)
        ax.yaxis.set_tick_params(labelsize=8)

    patches = [
        mpatches.Patch(color=CHARTED_COLOR, label="Charted"),
        mpatches.Patch(color=UNCHARTED_COLOR, label="Uncharted"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.02), frameon=False)

    fig.suptitle("Score by condition and structural novelty (all models pooled)", fontsize=11, y=1.01, fontweight="normal")
    fig.savefig(output_dir / "fig1_scores_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_scores_by_condition.png")


def fig2_gap_by_model(df: pd.DataFrame, output_dir: Path):
    score = "plan_quality"
    gap_df = compute_gap(df, score)
    if "gap" not in gap_df.columns:
        print("Skipping fig2: insufficient data for gap computation")
        return

    models = list(MODEL_DIRS.values())
    models_present = [m for m in models if m in gap_df["model"].unique()]
    cond_order = CONDITIONS

    fig, axes = plt.subplots(1, len(cond_order), figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.15, bottom=0.25)

    for ax, cond in zip(axes, cond_order):
        style_panel(ax)
        cond_data = gap_df[gap_df["condition"] == cond]
        xs = np.arange(len(models_present))
        gaps = [cond_data.loc[cond_data["model"] == m, "gap"].values[0]
                if m in cond_data["model"].values else np.nan
                for m in models_present]
        colors = [CHARTED_COLOR if g >= 0 else UNCHARTED_COLOR for g in gaps]
        ax.bar(xs, gaps, color=colors, alpha=0.85)
        ax.axhline(0, color=NAVY, linewidth=0.8, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(models_present, fontsize=8, rotation=30, ha="right")
        ax.set_title(CONDITION_LABELS[cond], fontsize=10, pad=6, fontweight="normal")
        if cond == cond_order[0]:
            ax.set_ylabel("Charted minus uncharted (plan quality)", fontsize=9)

    patches = [
        mpatches.Patch(color=CHARTED_COLOR, label="Charted outperforms"),
        mpatches.Patch(color=UNCHARTED_COLOR, label="Uncharted outperforms"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 0.0), frameon=False)

    fig.suptitle("Charted vs uncharted gap in plan quality by model and condition", fontsize=11, y=1.01, fontweight="normal")
    fig.savefig(output_dir / "fig2_gap_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_gap_by_model.png")


def fig3_convergence(df: pd.DataFrame, output_dir: Path):
    score = "plan_quality"
    conv_df = compute_convergence(df, score)
    if conv_df.empty:
        print("Skipping fig3: no instances with multiple model evaluations")
        return

    cond_order = CONDITIONS
    cond_xs = np.arange(len(cond_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.22)
    style_panel(ax)

    for offset, label, color in [(-width/2, "Charted", CHARTED_COLOR),
                                  (width/2, "Uncharted", UNCHARTED_COLOR)]:
        means = []
        for cond in cond_order:
            subset = conv_df[(conv_df["condition"] == cond) & (conv_df["charted_label"] == label)]
            means.append(subset["disagreement"].mean())
        ax.bar(cond_xs + offset, means, width, color=color, alpha=0.85)

    ax.set_xticks(cond_xs)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in cond_order], fontsize=9)
    ax.set_ylabel("Mean pairwise score disagreement", fontsize=9)
    ax.set_ylim(0, None)

    patches = [
        mpatches.Patch(color=CHARTED_COLOR, label="Charted"),
        mpatches.Patch(color=UNCHARTED_COLOR, label="Uncharted"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9, frameon=False)

    ax.set_title("Cross-model convergence on plan quality by condition and structural novelty", fontsize=10, pad=6, fontweight="normal")
    fig.savefig(output_dir / "fig3_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_convergence.png")


def fig4_fix_type_breakdown(df: pd.DataFrame, output_dir: Path):
    score = "plan_quality"
    cond = "no_context"
    subset = df[df["condition"] == cond].copy()

    type_order = (subset.groupby("fix_type")[score].mean()
                  .sort_values(ascending=False).index.tolist())
    type_order = [t for t in type_order if subset["fix_type"].value_counts().get(t, 0) >= 5]

    if not type_order:
        print("Skipping fig4: too few instances per fix type")
        return

    xs = np.arange(len(type_order))
    charted_means = []
    uncharted_means = []
    for ft in type_order:
        ft_df = subset[subset["fix_type"] == ft]
        charted_means.append(ft_df[ft_df["charted_label"] == "Charted"][score].mean())
        uncharted_means.append(ft_df[ft_df["charted_label"] == "Uncharted"][score].mean())

    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.3)
    style_panel(ax)

    ax.bar(xs - width/2, charted_means, width, color=CHARTED_COLOR, alpha=0.85, label="Charted")
    ax.bar(xs + width/2, uncharted_means, width, color=UNCHARTED_COLOR, alpha=0.85, label="Uncharted")
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace("_", " ") for t in type_order], fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("Mean plan quality (no context)", fontsize=9)
    ax.set_ylim(1, 3)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title("Plan quality by fix type and structural novelty (no context condition)", fontsize=10, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig4_fix_type_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_fix_type_breakdown.png")


# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="k for kNN chartedness (default 5)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records...")
    study_dir = ROOT / "output" / "prompting_study"
    records_df = load_all_records(study_dir)
    print(f"  {len(records_df)} scored condition rows across {records_df['model'].nunique()} models")

    print("Computing chartedness scores...")
    chart_df = load_chartedness(
        ROOT / "output" / "distances.parquet",
        ROOT / "output" / "labels.parquet",
        k=args.k,
    )
    threshold = chart_df["knn_edit_dist"].median()
    chart_df = add_charted_label(chart_df, threshold)
    n_charted = chart_df["charted"].sum()
    print(f"  Median kNN distance: {threshold:.3f}")
    print(f"  Charted: {n_charted}, Uncharted: {len(chart_df) - n_charted}")

    print("Loading fix types...")
    fix_df = load_fix_types(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )

    print("Joining...")
    df = records_df.merge(chart_df[["instance_id", "knn_edit_dist", "charted", "charted_label"]], on="instance_id", how="left")
    df = df.merge(fix_df, on="instance_id", how="left")

    # Save joined dataset
    df.to_parquet(args.output_dir / "charted_analysis.parquet", index=False)
    print(f"  Saved charted_analysis.parquet ({len(df)} rows)")

    # Print summary table
    print("\nMean scores by condition x charted/uncharted (all models, plan_quality):")
    summary = df.groupby(["condition", "charted_label"])["plan_quality"].agg(["mean", "count"]).round(3)
    print(summary.to_string())

    print("\nGenerating figures...")
    fig1_scores_by_condition(df, args.output_dir)
    fig2_gap_by_model(df, args.output_dir)
    fig3_convergence(df, args.output_dir)
    fig4_fix_type_breakdown(df, args.output_dir)

    # Print convergence summary
    conv = compute_convergence(df, "plan_quality")
    if not conv.empty:
        print("\nMean cross-model disagreement by condition x charted/uncharted (plan_quality):")
        print(conv.groupby(["condition", "charted_label"])["disagreement"].mean().round(3).to_string())

    print(f"\nDone. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
