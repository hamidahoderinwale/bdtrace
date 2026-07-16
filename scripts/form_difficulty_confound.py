#!/usr/bin/env python3
"""
Confound analysis: is form pass rate about strategy or problem difficulty?

If form pass rates just reflect problem difficulty (hard problems happen to require
certain edit patterns), then forms are not measuring strategy — they're measuring
problem hardness. This script disentangles the two.

Method:
  1. For each instance, compute agent consensus difficulty:
       difficulty = fraction of agents that fail it (0 = all pass, 1 = none pass)
  2. Bin instances into difficulty quartiles
  3. Within each quartile, show form pass rates — if forms still vary within quartile,
     the structural signal is independent of difficulty
  4. Interaction plot: agent x form pass rates — if agents differ within a form,
     that's strategy signal (not difficulty)

Usage:
  uv run python scripts/form_difficulty_confound.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "form_difficulty_confound"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"

AGENT_COLORS = [TEAL, ORANGE, NAVY, GRAY]
AGENT_SHORT = {
    "lite_20240402_sweagent_gpt4": "SWE-agent GPT-4",
    "lite_20240620_sweagent_claude3.5sonnet": "SWE-agent Claude 3.5",
    "lite_20240728_sweagent_gpt4o": "SWE-agent GPT-4o",
    "lite_20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": "SWE-Fixer Qwen",
}


def load_agent_results(agent_dir: Path) -> dict[str, dict[str, bool]]:
    results = {}
    for p in sorted(agent_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            results[p.stem] = {r["instance_id"]: bool(r.get("resolved", False))
                               for r in data}
    return results


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig_within_quartile(form_df: pd.DataFrame, agent_matrix: pd.DataFrame,
                        output_dir: Path):
    # Compute consensus difficulty: fraction of agents that fail each instance
    form_df = form_df.copy()
    ids = form_df["instance_id"].tolist()
    agent_cols = agent_matrix.columns.tolist()
    sub = agent_matrix.loc[agent_matrix.index.isin(ids)]

    # Mean pass rate across agents per instance = "ease"
    ease = sub.mean(axis=1).rename("ease")
    form_df = form_df.set_index("instance_id").join(ease).reset_index()
    form_df["ease"] = form_df["ease"].fillna(form_df["passed"].astype(float))

    # Bin into quartiles
    # Bin by agent consensus: how many of the 4 agents pass each instance
    # Use agent_matrix directly to count passes per instance
    pass_counts = agent_matrix.reindex(form_df["instance_id"].tolist()).sum(axis=1)
    pass_counts.index = form_df.index
    bins = [-0.5, 0.5, 1.5, 2.5, 4.5]
    bin_labels = ["0 agents pass", "1 agent passes", "2 agents pass", "3-4 agents pass"]
    form_df["difficulty_q"] = pd.cut(pass_counts, bins=bins, labels=bin_labels)

    quartiles = ["0 agents pass", "1 agent passes", "2 agents pass", "3-4 agents pass"]
    form_order = (form_df.groupby("form_label")["passed"]
                  .mean().sort_values(ascending=False).index.tolist())

    n_forms = len(form_order)
    n_q = len(quartiles)
    fig, axes = plt.subplots(1, n_q, figsize=(4 * n_q, 5), sharey=True)
    fig.subplots_adjust(wspace=0.08, bottom=0.4)

    for ax, q in zip(axes, quartiles):
        style_panel(ax)
        sub_q = form_df[form_df["difficulty_q"] == q]
        if len(sub_q) == 0:
            ax.set_title(q, fontsize=9)
            continue
        rates = [sub_q[sub_q["form_label"] == f]["passed"].mean()
                 if (sub_q["form_label"] == f).any() else np.nan
                 for f in form_order]
        ns = [(sub_q["form_label"] == f).sum() for f in form_order]
        xs = np.arange(n_forms)
        colors = [TEAL if not np.isnan(r) and r >= 0.3 else
                  ORANGE if not np.isnan(r) and r >= 0.15 else GRAY
                  for r in rates]
        ax.bar(xs, [r if not np.isnan(r) else 0 for r in rates],
               color=colors, alpha=0.85)
        for xi, (r, n) in enumerate(zip(rates, ns)):
            if not np.isnan(r) and n > 0:
                ax.text(xi, r + 0.02, f"n={n}", ha="center", va="bottom",
                        fontsize=6, color=NAVY)
        ax.set_xticks(xs)
        ax.set_xticklabels(form_order, fontsize=6, rotation=45, ha="right")
        ax.set_title(f"{q}\n(n={len(sub_q)})", fontsize=8, pad=4)
        if ax == axes[0]:
            ax.set_ylabel("Pass rate within quartile", fontsize=9)

    fig.suptitle("Form pass rates within difficulty quartiles\n"
                 "(if forms vary within a quartile, signal is independent of difficulty)",
                 fontsize=10, y=1.01, fontweight="normal")

    fig.savefig(output_dir / "fig1_within_quartile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_within_quartile.png")


def fig_agent_x_form(form_df: pd.DataFrame, agent_results: dict[str, dict[str, bool]],
                     output_dir: Path):
    agents = sorted(agent_results.keys())
    form_order = (form_df.groupby("form_label")["passed"]
                  .mean().sort_values(ascending=False).index.tolist())

    n_forms = len(form_order)
    n_agents = len(agents)

    mat = np.full((n_agents, n_forms), np.nan)
    counts = np.zeros((n_agents, n_forms), dtype=int)

    for ai, agent in enumerate(agents):
        res = agent_results[agent]
        for fi, form in enumerate(form_order):
            members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
            vals = [res[iid] for iid in members if iid in res]
            if vals:
                mat[ai, fi] = np.mean(vals)
                counts[ai, fi] = len(vals)

    # Per-form agent range (max - min): high range = strategy signal, not difficulty
    form_range = np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_forms * 0.8), 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.35, bottom=0.3, left=0.15, right=0.97)

    # Top: heatmap
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_forms))
    ax.set_xticklabels(form_order, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels([AGENT_SHORT.get(a, a) for a in agents], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Pass rate")
    for ai in range(n_agents):
        for fi in range(n_forms):
            if not np.isnan(mat[ai, fi]):
                ax.text(fi, ai, f"{mat[ai, fi]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[ai, fi] > 0.5 else NAVY)
    ax.set_title("Agent x form pass rates", fontsize=11, pad=6, fontweight="normal")

    # Bottom: per-form agent range bar
    ax2 = axes[1]
    style_panel(ax2)
    xs = np.arange(n_forms)
    colors = [TEAL if r >= 0.3 else ORANGE if r >= 0.15 else GRAY
              for r in form_range]
    ax2.bar(xs, form_range, color=colors, alpha=0.85)
    ax2.axhline(0.15, color=NAVY, linewidth=0.8, linestyle=":",
                label="0.15 threshold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(form_order, fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("Agent pass rate range\n(max - min)", fontsize=8)
    ax2.set_title("Per-form agent divergence (high = strategy gap, not just difficulty)",
                  fontsize=9, pad=4, fontweight="normal")
    ax2.legend(fontsize=7, frameon=False)

    fig.savefig(output_dir / "fig2_agent_x_form.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_agent_x_form.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    form_df = pd.read_parquet(ROOT / "output" / "fix_forms" / "form_assignments.parquet")
    print(f"Loaded {len(form_df)} form assignments, {form_df['form_label'].nunique()} forms")

    agent_results = load_agent_results(ROOT / "output" / "swebench_results_lite_agents")
    print(f"Loaded {len(agent_results)} agents")

    # Build agent pass/fail matrix: instance_id x agent
    all_ids = form_df["instance_id"].tolist()
    agent_matrix = pd.DataFrame(
        {agent: {iid: float(res.get(iid, np.nan))
                 for iid in all_ids}
         for agent, res in agent_results.items()},
        index=all_ids
    )

    # Print summary stats
    print("\nAgents:")
    for agent, res in agent_results.items():
        n = sum(1 for iid in all_ids if iid in res)
        rate = np.mean([res[iid] for iid in all_ids if iid in res])
        print(f"  {AGENT_SHORT.get(agent, agent):35s}: "
              f"{rate:.1%} pass, {n} instances")

    # Confound analysis
    form_df_merged = form_df.copy()
    form_df_merged["ease"] = agent_matrix.mean(axis=1).reindex(all_ids).values

    print("\nForm pass rates vs mean agent ease:")
    for form in form_df["form_label"].unique():
        sub = form_df_merged[form_df_merged["form_label"] == form]
        pr = sub["passed"].mean()
        ease = sub["ease"].mean()
        n = len(sub)
        print(f"  {form:30s}: pass={pr:.2f}, mean_ease={ease:.2f}, n={n}")

    # Correlation: is form pass rate just difficulty?
    form_stats = form_df_merged.groupby("form_label").agg(
        pass_rate=("passed", "mean"),
        mean_ease=("ease", "mean"),
        n=("passed", "count")
    ).reset_index()
    corr = form_stats[["pass_rate", "mean_ease"]].corr().iloc[0, 1]
    print(f"\nCorrelation between form pass rate and mean agent ease: r={corr:.3f}")
    if corr > 0.8:
        print("  High correlation: forms largely track problem difficulty")
    elif corr > 0.5:
        print("  Moderate correlation: partial confound, forms have some independent signal")
    else:
        print("  Low correlation: form pass rates are NOT just difficulty — "
              "structural signal is independent")

    # Save stats
    form_stats.to_csv(OUTPUT_DIR / "form_difficulty_stats.csv", index=False)
    print("\nSaved form_difficulty_stats.csv")

    print("\nGenerating figures...")
    fig_within_quartile(form_df, agent_matrix, OUTPUT_DIR)
    fig_agent_x_form(form_df, agent_results, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
