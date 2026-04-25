"""
Five standalone figures for the five findings not in the paper.
Each uses a unique visual form matched to the finding, all from real data.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# Project palette
TEAL = "#0C6583"
AMBER = "#FFBA08"
GREEN = "#2CA02C"
GRAY = "#AAAAAA"
NAVY = "#2B2D42"
ORANGE = "#EE7733"

OUT = Path(__file__).resolve().parent.parent / "figures" / "gap"
OUT.mkdir(exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# A. FIM separates difficulty.
# Paired comparison: semantic ceiling vs structural floor, with multiplier.
# Form: two stacked groups with a ratio annotation between them.

def fig_a():
    fig, ax = plt.subplots(figsize=(8, 5))

    semantic = [
        ("Issue text (k-means, k=10)", 0.0073),
        ("Predicted fix (k-means, k=10)", 0.0087),
        ("Fix from traces (k-means, k=10)", 0.0083),
    ]
    structural = [
        ("AST cert decision tree (10 forms)", 0.0257),
        ("FIM closed itemsets (15 forms)", 0.0333),
    ]

    all_items = semantic + structural
    n = len(all_items)
    y = np.arange(n)

    for i, (label, val) in enumerate(all_items):
        color = GRAY if i < len(semantic) else (AMBER if i == len(semantic) else TEAL)
        ax.plot([0, val], [i, i], color=color, lw=2, alpha=0.5, zorder=2)
        ax.scatter(val, i, s=140, color=color, zorder=5,
                   edgecolors="white", linewidth=1.5)

    ax.set_yticks(y)
    ax.set_yticklabels([item[0] for item in all_items], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-0.001, 0.038)
    ax.set_xlabel("Variance of per-group mean agent ease", fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    sep_y = len(semantic) - 0.5
    ax.axhline(sep_y, color="#e0e0e0", lw=0.8, linestyle="-")

    fig.suptitle("Which grouping separates difficulty?", fontsize=14, color=NAVY, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "a_fim_separates_difficulty.png", dpi=200, facecolor="white")
    plt.close()
    print("  A: lollipop comparison")


# B. Composition failures.
# Form: cliff chart. One row per primitive in the hardest FIM pattern.
# Left bar = fraction of agents that have this primitive (near 100%).
# Right bar = fraction of agents that solve an instance requiring all six (2%).
# The cliff between them is the finding.

def fig_b():
    fig, ax = plt.subplots(figsize=(9, 6))

    # Real data: hardest FIM pattern (pass_rate=0.000, n=7)
    # pattern: ADD_Attribute, ADD_Call, ADD_Compare, ADD_Constant, ADD_If, ADD_Name
    # From findings.md: each primitive in 98-100% of agent libraries
    primitives = [
        ("ADD_If", 0.99),
        ("ADD_Compare", 0.98),
        ("ADD_Constant", 1.00),
        ("ADD_Attribute", 1.00),
        ("ADD_Call", 1.00),
        ("ADD_Name", 1.00),
    ]
    combo_ease = 0.02  # from findings.md

    n = len(primitives) + 1  # +1 for the combination row
    y = np.arange(n)

    # Draw primitive bars
    for i, (name, freq) in enumerate(primitives):
        ax.barh(i, freq, height=0.6, color=TEAL, edgecolor="white", linewidth=1)
        ax.text(freq - 0.03, i, f"{freq:.0%}", va="center", ha="right",
                fontsize=10, color="white")
        ax.text(-0.01, i, name, va="center", ha="right", fontsize=10,
                family="monospace", color=NAVY)

    # Gap row
    gap_y = len(primitives)
    ax.barh(gap_y, combo_ease, height=0.6, color=ORANGE, edgecolor="white", linewidth=1)
    ax.text(combo_ease + 0.015, gap_y, f"{combo_ease:.0%}", va="center",
            fontsize=12, color=ORANGE)
    ax.text(-0.01, gap_y, "Six combined", va="center", ha="right", fontsize=10,
            color=ORANGE)

    # Separator
    ax.axhline(len(primitives) - 0.5, color="#e0e0e0", lw=0.8)

    ax.set_xlim(0, 1.05)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of 84 agents", fontsize=10)

    # Right-side annotation: failure classification from real data
    # per_agent_mean_fractions: novel_primitive=0.213, novel_composition=0.458, familiar=0.329
    txt = (
        "Failure classification (84 agents)\n\n"
        "  Novel composition     45.8%\n"
        "  Familiar pattern        32.9%\n"
        "  Novel primitive          21.3%"
    )
    ax.text(0.62, 1.8, txt, fontsize=10, color=NAVY, va="top",
            family="monospace", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                      edgecolor="#e0e0e0", linewidth=0.5))

    fig.suptitle("The hard part is composition, not primitives",
                 fontsize=14, color=NAVY, y=0.97)
    ax.set_title("In the hardest FIM pattern every piece is universal, the combination is unsolved",
                 fontsize=10, color="#999999", pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / "b_composition_failures.png", dpi=200, facecolor="white")
    plt.close()
    print("  B: cliff chart")


# C. Grounding failure.
# Form: precision-recall scatter with F1 isolines.
# Real data from grounding_gpt_4o.parquet.

def fig_c():
    fig, ax = plt.subplots(figsize=(7, 7))

    # Real per-condition means from the parquet
    conditions = [
        ("No context", 0.457, 0.139, GRAY),
        ("Procedural scaffolds", 0.445, 0.139, AMBER),
        ("Raw behavioral logs", 0.527, 0.181, TEAL),
    ]

    # F1 isolines
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        p = np.linspace(f1_val / 2 + 0.001, 1.0, 300)
        r = (f1_val * p) / (2 * p - f1_val)
        valid = (r > 0) & (r <= 1)
        ax.plot(p[valid], r[valid], color="#e8e8e8", lw=1, zorder=1)
        # Label near the y-axis
        label_idx = min(5, valid.sum() - 1)
        if valid.sum() > label_idx:
            ax.text(p[valid][label_idx] + 0.01, r[valid][label_idx],
                    f"F1={f1_val}", fontsize=8, color="#d0d0d0")

    # "Useful" region
    ax.axvspan(0.5, 1.0, ymin=0.5, ymax=1.0, alpha=0.03, color=GREEN, zorder=0)
    ax.text(0.74, 0.74, "useful", fontsize=14, color=GREEN, alpha=0.3,
            ha="center", style="italic")

    # Plot conditions
    for label, prec, rec, color in conditions:
        ax.scatter(prec, rec, s=220, color=color, zorder=5,
                   edgecolors="white", linewidth=2)
        # Offset labels to avoid overlap
        dx, dy = 0.025, 0.012
        if "Raw" in label:
            dy = 0.018
        ax.text(prec + dx, rec + dy, label, fontsize=10, color=color)

    # Self-report zone box
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((0.38, 0.08), 0.22, 0.14,
                          boxstyle="round,pad=0.015",
                          facecolor=ORANGE, alpha=0.06,
                          edgecolor=ORANGE, linewidth=1, linestyle="--", zorder=0)
    ax.add_patch(rect)
    ax.text(0.49, 0.065, "self-report zone", fontsize=9, color=ORANGE,
            ha="center", style="italic")

    # Overall F1 callout
    ax.text(0.96, 0.96, "Overall F1 = 0.216", fontsize=11, color=NAVY,
            ha="right", va="top", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Precision (of claimed operations, how many were real)", fontsize=10)
    ax.set_ylabel("Recall (of real operations, how many were claimed)", fontsize=10)
    ax.set_aspect("equal")

    fig.suptitle("Agents cannot describe their own patches",
                 fontsize=14, color=NAVY, y=0.96)
    ax.set_title("GPT-4o self-reported edit operations vs actual patch structure",
                 fontsize=10, color="#999999", pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "c_grounding_failure.png", dpi=200, facecolor="white")
    plt.close()
    print("  C: precision-recall scatter")


# D. Localization bottleneck.
# Form: waterfall with per-pair dots showing spread.
# Real data from scoped certificates.

def fig_d():
    fig, ax = plt.subplots(figsize=(8, 6))

    layers = ["File\n(same file?)", "Edit type\n(same structural change?)",
              "Scope\n(same function?)"]
    # Real per-pair data from scope_decomposition figure
    pair_data = {
        "file":  [0.76, 0.76, 0.65, 0.61, 0.69, 0.68],
        "edit":  [0.45, 0.44, 0.46, 0.54, 0.51, 0.51],
        "scope": [0.26, 0.28, 0.25, 0.26, 0.28, 0.31],
    }
    means = [np.mean(v) for v in pair_data.values()]
    colors = [TEAL, AMBER, GREEN]
    x = np.arange(len(layers))

    # Bars
    bars = ax.bar(x, means, color=colors, width=0.5, edgecolor="white",
                  linewidth=1.5, zorder=3)

    # Connecting slope lines between bar tops
    for i in range(len(means) - 1):
        ax.plot([i + 0.25, i + 0.75], [means[i], means[i + 1]],
                color=ORANGE, lw=2.5, zorder=4)
        drop = means[i] - means[i + 1]
        mid_x = i + 0.5
        mid_y = (means[i] + means[i + 1]) / 2
        ax.text(mid_x + 0.08, mid_y, f"{drop:+.2f}",
                fontsize=10, color=ORANGE, va="center")

    # Per-pair dots
    rng = np.random.default_rng(42)
    for i, key in enumerate(["file", "edit", "scope"]):
        vals = pair_data[key]
        jitter = rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=35, color=NAVY, alpha=0.3, zorder=5, edgecolors="none")

    # Mean value labels above bars
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=12, color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=11)
    ax.set_ylabel("Mean pairwise agreement", fontsize=11)
    ax.set_ylim(0, 0.9)

    fig.suptitle("The bottleneck is localization, not edit strategy",
                 fontsize=14, color=NAVY, y=0.97)
    ax.set_title("Agreement across 6 agent pairs, decomposed by layer",
                 fontsize=10, color="#999999", pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "d_localization_bottleneck.png", dpi=200, facecolor="white")
    plt.close()
    print("  D: waterfall")


# E. Semantic independence.
# Form: ARI sweep across k values. Real data from alignment_results.json.
# Shows that no matter how many semantic clusters you try, ARI stays at zero.

def fig_e():
    fig, ax = plt.subplots(figsize=(9, 5))

    data = json.loads((ROOT / "output" / "form_alignment" / "alignment_results.json").read_text())
    sweep = data["sweep"]
    ks = [s["k"] for s in sweep]
    aris = [s["ari"] for s in sweep]

    ax.plot(ks, aris, color=TEAL, lw=1.5, zorder=3)
    ax.scatter(ks, aris, s=40, color=TEAL, zorder=4, edgecolors="white", linewidth=1)

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axhspan(-0.02, 0.02, color=ORANGE, alpha=0.04, zorder=0)

    ax.set_xlabel("Number of semantic clusters (k)", fontsize=10)
    ax.set_ylabel("Adjusted Rand Index vs 11 structural forms", fontsize=10)
    ax.set_ylim(-0.025, 0.035)
    ax.set_xlim(2, 26)
    ax.set_xticks(ks)

    fig.suptitle("Structure and semantics are independent",
                 fontsize=14, color=NAVY, y=0.97)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "e_semantic_independence.png", dpi=200, facecolor="white")
    plt.close()
    print("  E: ARI sweep")


if __name__ == "__main__":
    print("Generating gap figures from real data...")
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    fig_e()
    print(f"\nAll saved to {OUT}/")
