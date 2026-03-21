#!/usr/bin/env python3
"""
Build three missing procedural divergence plots.

1. Per-stage distance distributions (tokens → edits → modules)
2. Divergence gap distribution by procedure pair (behavioral, mechanistic, functional)
3. Cross-representation rank correlation heatmap

Usage:
    uv run python scripts/build_divergence_plots.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "notebooks" / "plots" / "swe_bench_lite_resolved"
OUT.mkdir(parents=True, exist_ok=True)

DS = ROOT / "output" / "datasets" / "swe_bench_lite_resolved"

# Wong colorblind-safe palette
BLUE    = "#0072B2"
ORANGE  = "#E69F00"
GREEN   = "#009E73"
VERMIL  = "#D55E00"
PINK    = "#CC79A7"
GRAY    = "#999999"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})


# ---------------------------------------------------------------------------
# 1. Per-stage distance distributions
# ---------------------------------------------------------------------------

def plot_distance_distributions():
    matrices_path = DS / "matrices.npz"
    if not matrices_path.exists():
        print(f"Skipping distance distributions: {matrices_path} not found")
        return

    data = np.load(matrices_path)
    # Three-stage structural hierarchy: tokens → edits (AST) → modules (graph)
    # edits_set_diff is the tractable proxy for tree edit distance (Jaccard fallback)
    stages = {
        "tokens":         ("Tokens\n(Levenshtein, normalised)",  BLUE),
        "edits_set_diff": ("Edits\n(sym diff / total — AST proxy)", ORANGE),
        "modules":        ("Modules\n(Jaccard on graph tokens)", GREEN),
    }

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), sharey=False)
    fig.suptitle("Per-stage distance distributions — structural hierarchy\n"
                 "(SWE-bench Lite resolved, 300 instances, 44 850 pairs)",
                 fontsize=10, y=1.04)

    for ax, (key, (label, color)) in zip(axes, stages.items()):
        if key not in data:
            ax.set_visible(False)
            continue
        D = data[key]
        upper = D[np.triu_indices_from(D, k=1)]
        mean, std = upper.mean(), upper.std()

        ax.hist(upper, bins=40, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(mean, color="black", linewidth=1.2, linestyle="--", label=f"mean={mean:.3f}")
        ax.set_xlabel(label, fontsize=8.5)
        ax.set_ylabel("pairs" if ax is axes[0] else "")
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_title(f"σ = {std:.3f}", fontsize=8, color=GRAY)

    fig.tight_layout()
    out_path = OUT / "distance_distributions_by_stage.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 2. Divergence gap by procedure pair
# ---------------------------------------------------------------------------

def plot_divergence_gap():
    div_path = DS / "eval" / "divergence.parquet"
    if not div_path.exists():
        print(f"Skipping divergence gap: {div_path} not found")
        return

    df = pd.read_parquet(div_path)
    df["pair"] = df["proc_a"].str.capitalize() + " / " + df["proc_b"].str.capitalize()

    pairs = df["pair"].unique()
    pair_colors = {p: c for p, c in zip(sorted(pairs), [BLUE, ORANGE, GREEN])}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.suptitle(
        "Procedural divergence: annotation gap by procedure pair\n"
        "(gap = structural agreement − semantic agreement; structural agreement = 1.0 for all pairs)",
        fontsize=10, y=1.03,
    )

    for ax, pair in zip(axes, sorted(pairs)):
        sub = df[df["pair"] == pair]["gap"].dropna()
        color = pair_colors[pair]

        # KDE-like histogram
        ax.hist(sub, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        mean = sub.mean()
        pct_diverged = (df[df["pair"] == pair]["terminal_diverged"]).mean() * 100
        ax.axvline(mean, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(pair, fontsize=9, fontweight="bold")
        ax.set_xlabel("gap", fontsize=8.5)
        ax.set_ylabel("instances" if ax is axes[0] else "")
        ax.text(
            0.97, 0.95,
            f"mean = {mean:.3f}\ndivergence = {pct_diverged:.0f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY, linewidth=0.6),
        )

    fig.tight_layout()
    out_path = OUT / "procedure_divergence_gap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 3. Cross-representation rank correlation heatmap
# ---------------------------------------------------------------------------

def plot_rank_correlation_heatmap():
    rc_path = DS / "rank_correlation.parquet"
    if not rc_path.exists():
        print(f"Skipping rank correlation heatmap: {rc_path} not found")
        return

    rc = pd.read_parquet(rc_path)

    LABELS = {
        "tokens":         "Tokens\n(Levenshtein)",
        "edits_set_diff": "Edits\n(AST proxy)",
        "modules":        "Modules\n(graph)",
    }
    reprs = [r for r in ["tokens", "edits_set_diff", "modules"]
             if r in rc["repr_i"].values]
    n = len(reprs)

    mat = np.zeros((n, n))
    for _, row in rc.iterrows():
        if row["repr_i"] in reprs and row["repr_j"] in reprs:
            i = reprs.index(row["repr_i"])
            j = reprs.index(row["repr_j"])
            mat[i, j] = row["rho"]

    tick_labels = [LABELS.get(r, r) for r in reprs]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=8.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

    ax.set_title(
        "Cross-representation rank correlation (Spearman ρ)\n"
        "SWE-bench Lite resolved — 300 instances, pairwise distances",
        fontsize=9,
    )
    ax.spines[:].set_visible(False)

    fig.tight_layout()
    out_path = OUT / "cross_repr_rank_correlation.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    plot_distance_distributions()
    plot_divergence_gap()
    plot_rank_correlation_heatmap()
    print("Done.")
