#!/usr/bin/env python3
"""
Agent-level form saturation: does adding more agents expand form coverage?

The benchmark saturation claim — adding more instances doesn't add new strategy
coverage — has a sharper agent-level version: some fix forms are unsolved by ALL
current agents regardless of how many you add. These are capability frontiers,
not difficulty noise.

Analyses:
  1. How many agents solve at least one instance in each form?
     (form frontier = zero agents solve any instance)
  2. Agent accumulation curve: as you add agents, how many forms get at least one pass?
  3. Does the frontier persist on Verified? (forms unsolved on Lite also unsolved on Verified)
  4. Within-form pass rate spread: where do agents most disagree?

Usage:
  uv run python scripts/agent_form_saturation.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import permutations

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "agent_form_saturation"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"
GREEN = "#009E73"

AGENT_SHORT = {
    "lite_20240402_sweagent_gpt4": "SWE-agent\nGPT-4",
    "lite_20240620_sweagent_claude3.5sonnet": "SWE-agent\nClaude 3.5",
    "lite_20240728_sweagent_gpt4o": "SWE-agent\nGPT-4o",
    "lite_20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": "SWE-Fixer\nQwen",
    "verified_20240402_sweagent_gpt4": "SWE-agent\nGPT-4",
    "verified_20240620_sweagent_claude3.5sonnet": "SWE-agent\nClaude 3.5",
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


def agent_accumulation_curves(
    form_df: pd.DataFrame,
    agent_results: dict[str, dict[str, bool]],
    criterion: str = "any_pass",
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    For each ordering of agents, track how many forms get >=1 pass as agents added.
    criterion: 'any_pass' = at least one instance in form is solved
    """
    agents = list(agent_results.keys())
    n_agents = len(agents)
    form_order = form_df["form_label"].unique().tolist()
    n_forms = len(form_order)

    # Precompute: which forms does each agent solve (>=1 pass in form)?
    agent_solved = {}
    for agent, res in agent_results.items():
        solved = set()
        for form in form_order:
            members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
            if any(res.get(iid, False) for iid in members):
                solved.add(form)
        agent_solved[agent] = solved

    # Enumerate all orderings (4! = 24 for 4 agents)
    all_curves = []
    for perm in permutations(agents):
        curve = []
        seen = set()
        for agent in perm:
            seen |= agent_solved[agent]
            curve.append(len(seen))
        all_curves.append(curve)

    curves = np.array(all_curves)
    xs = list(range(1, n_agents + 1))
    return xs, curves.mean(axis=0), curves.std(axis=0)


def fig_solver_count(form_df: pd.DataFrame,
                     agent_results: dict[str, dict[str, bool]],
                     output_dir: Path):
    agents = list(agent_results.keys())
    n_agents = len(agents)
    form_order = (form_df.groupby("form_label")["passed"]
                  .mean().sort_values(ascending=False).index.tolist())

    # Count: how many agents solve >=1 instance per form
    solver_counts = {}
    for form in form_order:
        members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
        count = sum(
            1 for agent, res in agent_results.items()
            if any(res.get(iid, False) for iid in members)
        )
        solver_counts[form] = count

    fig, ax = plt.subplots(figsize=(max(10, len(form_order) * 0.9), 5))
    fig.subplots_adjust(bottom=0.35)
    style_panel(ax)

    xs = np.arange(len(form_order))
    colors = [
        TEAL if solver_counts[f] == n_agents else
        ORANGE if solver_counts[f] >= 2 else
        NAVY if solver_counts[f] == 1 else
        GRAY
        for f in form_order
    ]
    bars = ax.bar(xs, [solver_counts[f] for f in form_order], color=colors, alpha=0.85)

    ax.set_xticks(xs)
    ax.set_xticklabels(form_order, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Agents solving >=1 instance in form", fontsize=9)
    ax.set_yticks(range(n_agents + 1))
    ax.set_title("Form reachability: how many agents solve each strategy form?",
                 fontsize=11, pad=6, fontweight="normal")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TEAL, alpha=0.85, label=f"All {n_agents} agents"),
        Patch(facecolor=ORANGE, alpha=0.85, label="2+ agents"),
        Patch(facecolor=NAVY, alpha=0.85, label="1 agent only"),
        Patch(facecolor=GRAY, alpha=0.85, label="No agent (frontier)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False)

    # Annotate frontier forms
    for xi, form in enumerate(form_order):
        if solver_counts[form] == 0:
            ax.text(xi, 0.05, "frontier", ha="center", va="bottom",
                    fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=GRAY, ec="none"))

    fig.savefig(output_dir / "fig1_solver_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_solver_count.png")


def fig_accumulation(xs, means, stds, n_forms: int, output_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(bottom=0.15)
    style_panel(ax)

    ax.plot(xs, means, color=TEAL, linewidth=2, marker="o", markersize=6)
    ax.fill_between(xs, means - stds, means + stds, color=TEAL, alpha=0.15)
    ax.axhline(n_forms, color=GRAY, linewidth=0.8, linestyle=":",
               label=f"All {n_forms} forms")

    # Annotate final coverage
    final = means[-1]
    ax.annotate(
        f"{final:.0f}/{n_forms} forms\nreachable",
        xy=(xs[-1], final), xytext=(xs[-1] - 0.6, final - 1.5),
        fontsize=8, color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8),
    )

    ax.set_xlabel("Number of agents", fontsize=9)
    ax.set_ylabel("Forms with >=1 solved instance", fontsize=9)
    ax.set_xticks(xs)
    ax.set_title("Agent accumulation: form coverage vs number of agents",
                 fontsize=11, pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False)

    fig.savefig(output_dir / "fig2_agent_accumulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_agent_accumulation.png")


def fig_pass_rate_spread(form_df: pd.DataFrame,
                         agent_results: dict[str, dict[str, bool]],
                         output_dir: Path):
    agents = sorted(agent_results.keys())
    form_order = (form_df.groupby("form_label")["passed"]
                  .mean().sort_values(ascending=False).index.tolist())
    n_forms = len(form_order)
    n_agents = len(agents)

    mat = np.full((n_agents, n_forms), np.nan)
    for ai, agent in enumerate(agents):
        res = agent_results[agent]
        for fi, form in enumerate(form_order):
            members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
            vals = [res[iid] for iid in members if iid in res]
            if vals:
                mat[ai, fi] = np.mean(vals)

    spread = np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0)
    mean_rate = np.nanmean(mat, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_forms * 0.85), 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(hspace=0.5, bottom=0.3)

    # Top: grouped bars per form, one bar per agent
    ax = axes[0]
    style_panel(ax)
    xs = np.arange(n_forms)
    width = 0.8 / n_agents
    agent_colors = [TEAL, ORANGE, NAVY, GREEN]
    for ai, agent in enumerate(agents):
        offset = (ai - n_agents / 2 + 0.5) * width
        vals = mat[ai]
        ax.bar(xs + offset, np.nan_to_num(vals), width=width * 0.9,
               color=agent_colors[ai % len(agent_colors)], alpha=0.85,
               label=AGENT_SHORT.get(agent, agent).replace("\n", " "))
    ax.set_xticks(xs)
    ax.set_xticklabels(form_order, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_title("Per-form pass rates by agent", fontsize=10, pad=4, fontweight="normal")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # Bottom: spread bar
    ax2 = axes[1]
    style_panel(ax2)
    colors = [TEAL if s >= 0.25 else ORANGE if s >= 0.1 else GRAY for s in spread]
    ax2.bar(xs, spread, color=colors, alpha=0.85)
    ax2.axhline(0.2, color=NAVY, linewidth=0.8, linestyle=":")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(form_order, fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("Agent spread\n(max - min pass rate)", fontsize=8)
    ax2.set_title("Forms where agents most disagree = strategic differentiation",
                  fontsize=9, pad=4, fontweight="normal")

    fig.savefig(output_dir / "fig3_pass_rate_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_pass_rate_spread.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    form_df = pd.read_parquet(ROOT / "output" / "fix_forms" / "form_assignments.parquet")
    agent_results = load_agent_results(ROOT / "output" / "swebench_results_lite_agents")
    n_forms = form_df["form_label"].nunique()

    print(f"Loaded {len(form_df)} instances, {n_forms} forms, {len(agent_results)} agents\n")

    # Which forms does each agent solve?
    print("Forms solved (>=1 instance) per agent:")
    for agent, res in sorted(agent_results.items()):
        solved = []
        for form in form_df["form_label"].unique():
            members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
            if any(res.get(iid, False) for iid in members):
                solved.append(form)
        name = AGENT_SHORT.get(agent, agent).replace("\n", " ")
        print(f"  {name:30s}: {len(solved)}/{n_forms} forms  {solved}")

    # Frontier forms
    print("\nFrontier forms (no agent solves ANY instance):")
    for form in form_df["form_label"].unique():
        members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
        n_members = len(members)
        if not any(
            any(res.get(iid, False) for iid in members)
            for res in agent_results.values()
        ):
            print(f"  {form:30s} (n={n_members})")

    # Accumulation curves
    print("\nComputing agent accumulation curves (all orderings)...")
    xs, means, stds = agent_accumulation_curves(form_df, agent_results)
    print(f"  After {len(agent_results)} agents: {means[-1]:.1f}/{n_forms} forms reachable")
    print(f"  Saturation gap: {n_forms - means[-1]:.1f} forms unreachable by any agent combination")

    # Strategic differentiation: forms with high agent spread
    print("\nForms with highest agent disagreement (strategic differentiation):")
    agents = sorted(agent_results.keys())
    rows = []
    for form in form_df["form_label"].unique():
        members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
        rates = []
        for agent in agents:
            res = agent_results[agent]
            vals = [res[iid] for iid in members if iid in res]
            if vals:
                rates.append(np.mean(vals))
        if len(rates) >= 2:
            spread = max(rates) - min(rates)
            rows.append((form, spread, np.mean(rates), len(members)))
    rows.sort(key=lambda x: -x[1])
    for form, spread, mean_rate, n in rows:
        print(f"  {form:30s}: spread={spread:.2f}, mean={mean_rate:.2f}, n={n}")

    # Save summary
    summary = {
        "n_forms": n_forms,
        "n_agents": len(agent_results),
        "forms_reachable_all_agents": float(means[-1]),
        "accumulation": [{"n_agents": x, "mean_forms": float(m), "std": float(s)}
                         for x, m, s in zip(xs, means, stds)],
        "form_solver_counts": {},
    }
    for form in form_df["form_label"].unique():
        members = form_df[form_df["form_label"] == form]["instance_id"].tolist()
        count = sum(
            1 for res in agent_results.values()
            if any(res.get(iid, False) for iid in members)
        )
        summary["form_solver_counts"][form] = count
    with open(OUTPUT_DIR / "agent_saturation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved agent_saturation_results.json")

    print("\nGenerating figures...")
    fig_solver_count(form_df, agent_results, OUTPUT_DIR)
    fig_accumulation(xs, means, stds, n_forms, OUTPUT_DIR)
    fig_pass_rate_spread(form_df, agent_results, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
