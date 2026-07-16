#!/usr/bin/env python3
"""
Strategy coverage analysis across agents.

Asks: do agents differ in which fix strategies they cover, or do they all
rely on the same dominant strategies regardless of overall score?

Outputs:
  fig1_coverage_heatmap.png   -- pass rate per agent x fix type
  fig2_strategy_breadth.png   -- overall score vs strategy breadth (n types covered)
  fig3_dominance.png          -- what fraction of each agent's passes come from logic_fix
  strategy_coverage.parquet   -- underlying data

Usage:
  uv run python scripts/analyze_strategy_coverage.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "strategy_coverage"

# Project palette
COLORS = ["#0C6583", "#EE7733", "#2B2D42", "#AAAAAA", "#56B4E9"]
PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"

AGENT_LABELS = {
    "lite_20240402_sweagent_gpt4": "SWE-agent GPT-4",
    "lite_20240620_sweagent_claude3.5sonnet": "SWE-agent Claude 3.5",
    "lite_20240728_sweagent_gpt4o": "SWE-agent GPT-4o",
    "lite_20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": "SWE-Fixer Qwen2.5",
}

# Minimum instances in a fix type to include in analysis
MIN_INSTANCES = 5


def load_agent_results(agents_dir: Path) -> pd.DataFrame:
    rows = []
    for fpath in sorted(agents_dir.glob("*.json")):
        key = fpath.stem
        if key not in AGENT_LABELS:
            continue
        with open(fpath) as f:
            data = json.load(f)
        for entry in data:
            rows.append({
                "instance_id": entry["instance_id"],
                "agent": AGENT_LABELS[key],
                "resolved": bool(entry["resolved"]),
            })
    return pd.DataFrame(rows)


def load_fix_types(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[["instance_id", "fix_type", "confidence"]]


def build_coverage_table(agent_df: pd.DataFrame, fix_df: pd.DataFrame, min_instances: int) -> pd.DataFrame:
    merged = agent_df.merge(fix_df, on="instance_id", how="inner")
    counts = fix_df["fix_type"].value_counts()
    valid_types = counts[counts >= min_instances].index.tolist()
    merged = merged[merged["fix_type"].isin(valid_types)]

    coverage = (
        merged.groupby(["agent", "fix_type"])["resolved"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "n_resolved", "count": "n_total", "mean": "pass_rate"})
    )
    return coverage


def overall_stats(agent_df: pd.DataFrame) -> pd.DataFrame:
    return (
        agent_df.groupby("agent")["resolved"]
        .agg(n_resolved="sum", n_total="count", pass_rate="mean")
        .reset_index()
    )


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig1_heatmap(coverage: pd.DataFrame, output_dir: Path):
    agents = list(AGENT_LABELS.values())
    fix_types = (
        coverage.groupby("fix_type")["pass_rate"].mean()
        .sort_values(ascending=False).index.tolist()
    )

    matrix = np.full((len(agents), len(fix_types)), np.nan)
    for i, agent in enumerate(agents):
        for j, ft in enumerate(fix_types):
            row = coverage[(coverage["agent"] == agent) & (coverage["fix_type"] == ft)]
            if not row.empty:
                matrix[i, j] = row["pass_rate"].values[0]

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.subplots_adjust(bottom=0.3, left=0.22)

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(fix_types)))
    ax.set_xticklabels([ft.replace("_", " ") for ft in fix_types],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=9)

    for i in range(len(agents)):
        for j in range(len(fix_types)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "white" if val > 0.5 else "#2B2D42"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pass rate")
    ax.set_title("Pass rate by agent and fix strategy", fontsize=11, pad=8, fontweight="normal")

    fig.savefig(output_dir / "fig1_coverage_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_coverage_heatmap.png")


def fig2_breadth_vs_score(coverage: pd.DataFrame, overall: pd.DataFrame,
                           output_dir: Path, threshold: float = 0.2):
    breadth = (
        coverage[coverage["pass_rate"] >= threshold]
        .groupby("agent")["fix_type"]
        .nunique()
        .reset_index()
        .rename(columns={"fix_type": "n_strategies_covered"})
    )
    df = overall.merge(breadth, on="agent", how="left").fillna({"n_strategies_covered": 0})

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(bottom=0.15)
    style_panel(ax)

    for i, row in df.iterrows():
        ax.scatter(row["n_strategies_covered"], row["pass_rate"],
                   color=COLORS[i % len(COLORS)], s=80, zorder=3)
        ax.annotate(row["agent"], (row["n_strategies_covered"], row["pass_rate"]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8)

    ax.set_xlabel(f"Strategies covered (pass rate >= {threshold:.0%})", fontsize=9)
    ax.set_ylabel("Overall pass rate", fontsize=9)
    ax.set_xlim(-0.5, None)
    ax.set_ylim(0, 0.35)
    ax.set_title("Overall score vs strategy breadth", fontsize=11, pad=8, fontweight="normal")

    fig.savefig(output_dir / "fig2_strategy_breadth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_strategy_breadth.png")


def fig3_dominance(agent_df: pd.DataFrame, fix_df: pd.DataFrame, output_dir: Path):
    merged = agent_df[agent_df["resolved"]].merge(fix_df, on="instance_id", how="inner")
    agent_totals = merged.groupby("agent")["resolved"].sum()
    logic_counts = merged[merged["fix_type"] == "logic_fix"].groupby("agent")["resolved"].sum()

    dominance = (logic_counts / agent_totals).reset_index()
    dominance.columns = ["agent", "logic_fix_share"]
    dominance = dominance.sort_values("logic_fix_share", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.2)
    style_panel(ax)

    xs = np.arange(len(dominance))
    ax.bar(xs, dominance["logic_fix_share"], color="#0C6583", alpha=0.85)
    ax.axhline(0.49, color="#EE7733", linewidth=1.2, linestyle="--")
    ax.text(len(dominance) - 0.5, 0.50, "corpus share (49%)", fontsize=8,
            color="#EE7733", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(dominance["agent"], fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Share of passes from logic_fix", fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.set_title("Reliance on dominant strategy across agents", fontsize=11, pad=8, fontweight="normal")

    fig.savefig(output_dir / "fig3_dominance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_dominance.png")


def print_summary(coverage: pd.DataFrame, overall: pd.DataFrame, threshold: float = 0.2):
    print("\nOverall pass rates:")
    print(overall[["agent", "n_resolved", "n_total", "pass_rate"]]
          .sort_values("pass_rate", ascending=False)
          .to_string(index=False))

    print(f"\nStrategies covered per agent (pass rate >= {threshold:.0%}):")
    breadth = (
        coverage[coverage["pass_rate"] >= threshold]
        .groupby("agent")[["fix_type", "pass_rate"]]
        .apply(lambda g: g.set_index("fix_type")["pass_rate"].to_dict())
        .reset_index()
    )
    for _, row in breadth.iterrows():
        types = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(row[0].items(), key=lambda x: -x[1]))
        print(f"  {row['agent']}: {types}")

    print("\nFix types NO agent covers (pass rate < 20%):")
    worst = coverage.groupby("fix_type")["pass_rate"].max()
    uncovered = worst[worst < 0.2].index.tolist()
    print(f"  {uncovered}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading agent results...")
    agent_df = load_agent_results(ROOT / "output" / "swebench_results_lite_agents")
    print(f"  {agent_df['agent'].nunique()} agents, {agent_df['instance_id'].nunique()} instances")

    print("Loading fix types...")
    fix_df = load_fix_types(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )

    coverage = build_coverage_table(agent_df, fix_df, min_instances=MIN_INSTANCES)
    overall = overall_stats(agent_df)

    coverage.to_parquet(OUTPUT_DIR / "strategy_coverage.parquet", index=False)
    print("  Saved strategy_coverage.parquet")

    print_summary(coverage, overall)

    print("\nGenerating figures...")
    fig1_heatmap(coverage, OUTPUT_DIR)
    fig2_breadth_vs_score(coverage, overall, OUTPUT_DIR)
    fig3_dominance(agent_df, fix_df, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
