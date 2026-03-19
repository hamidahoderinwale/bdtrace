#!/usr/bin/env python3
"""
Build load-bearing paper figures.

Figure 1: Reframing — current harness vs structural harness (conceptual)
Figure 2: Three-level representation hierarchy (token / syntactic / graph)
Figure 3: Procedure-space UMAP — all datasets, colored by task type

Usage:
    uv run python scripts/build_figures.py
    uv run python scripts/build_figures.py --figures 1 2   # specific figures only
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Wong colorblind-safe palette
GRAY    = "#999999"
ORANGE  = "#E69F00"
BLUE    = "#0072B2"
GREEN   = "#009E73"
VERMIL  = "#D55E00"
PINK    = "#CC79A7"
SKY     = "#56B4E9"
YELLOW  = "#F0E442"

DATASET_COLORS = {
    "SWE-bench":     BLUE,
    "HumanEval":     ORANGE,
    "MBPP":          GREEN,
    "BigCodeBench":  PINK,
    "LiveCodeBench": VERMIL,
}
DATASET_ORDER = ["SWE-bench", "HumanEval", "MBPP", "BigCodeBench", "LiveCodeBench"]

TASK_TYPE_COLORS = {
    "bug_fix":         BLUE,
    "code_generation": ORANGE,
    "algorithmic":     GREEN,
    "api_usage":       PINK,
    "refactoring":     VERMIL,
}

TASK_TYPE_LABELS = {
    "bug_fix":         "Bug fix",
    "code_generation": "Code generation",
    "algorithmic":     "Algorithmic",
    "api_usage":       "API usage",
    "refactoring":     "Refactoring",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Reframing
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, text, facecolor, textcolor="white", fontsize=8.5,
         radius=0.03, bold=False, subtext=None):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor="none", zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x, y + 0.015, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)
        ax.text(x, y - 0.025, subtext, ha="center", va="center",
                fontsize=6.5, color=textcolor, alpha=0.82, zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10))


def build_figure1():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── divider ──────────────────────────────────────────────────────────────
    ax.axvline(0.5, color="#dddddd", lw=1.5, zorder=0)

    # ── column headers ────────────────────────────────────────────────────────
    ax.text(0.25, 0.95, "Current harnesses", ha="center", va="top",
            fontsize=10, fontweight="bold", color="#444444")
    ax.text(0.75, 0.95, "This work", ha="center", va="top",
            fontsize=10, fontweight="bold", color=BLUE)

    # ══ LEFT SIDE ════════════════════════════════════════════════════════════
    lx = 0.25
    bw, bh = 0.17, 0.09

    _box(ax, lx, 0.80, bw, bh, "Agent",  "#555555", bold=True)
    _arrow(ax, lx, 0.755, lx, 0.685)
    _box(ax, lx, 0.64, bw, bh, "Task",   "#777777")
    _arrow(ax, lx, 0.595, lx, 0.525)

    # binary outcome
    _box(ax, lx, 0.48, bw * 1.1, bh, "Pass / Fail", GRAY, bold=True)

    # annotation
    ax.text(lx, 0.36, "One bit of signal.\nNo structure visible.",
            ha="center", va="top", fontsize=7.5, color="#888888",
            linespacing=1.5, style="italic")

    # ══ RIGHT SIDE ═══════════════════════════════════════════════════════════
    rx = 0.75
    bw2 = 0.18

    _box(ax, rx, 0.80, bw2, bh, "Agent",  "#555555", bold=True)
    _arrow(ax, rx, 0.755, rx, 0.685)
    _box(ax, rx, 0.64, bw2, bh, "Task",   "#777777")
    _arrow(ax, rx, 0.595, rx, 0.525)

    # structural trace box
    _box(ax, rx, 0.47, bw2 * 1.3, bh * 1.05,
         "Structural trace", "#2a5ca8", subtext="edits · modules · tokens",
         fontsize=8, bold=True)

    # three level boxes
    levels = [
        (rx - 0.13, 0.33, "Token",     GRAY,   "sequence"),
        (rx,        0.33, "Syntactic", ORANGE, "edits / AST"),
        (rx + 0.13, 0.33, "Graph",     BLUE,   "modules"),
    ]
    for lx2, ly, label, col, sub in levels:
        _arrow(ax, rx, 0.445, lx2, 0.375)
        _box(ax, lx2, ly, 0.11, 0.08, label, col, fontsize=7.5,
             subtext=sub, radius=0.025)

    # funnel convergence → rubric
    # short stubs from each level box bottom to a shared horizontal rail
    rail_y   = 0.258
    stub_top = 0.290
    for lx2 in [rx - 0.13, rx, rx + 0.13]:
        ax.plot([lx2, lx2], [stub_top, rail_y], color="#888888", lw=1.0, zorder=2)
    ax.plot([rx - 0.13, rx + 0.13], [rail_y, rail_y],
            color="#888888", lw=1.0, zorder=2)
    # single arrow from rail centre down to rubric box
    _arrow(ax, rx, rail_y, rx, 0.220, color="#888888")
    _box(ax, rx, 0.185, bw2 * 1.35, bh * 1.0,
         "Rubric match + partial score", GREEN,
         subtext="verify before tests run", fontsize=7.5, bold=True)

    # saturation inset ────────────────────────────────────────────────────────
    inset = fig.add_axes([0.055, 0.06, 0.16, 0.14])
    xs = np.linspace(0, 100, 300)
    ys = 100 * (1 - np.exp(-xs / 8))
    inset.plot(xs, ys, color=BLUE, lw=1.5)
    inset.axvline(5.5, color=VERMIL, lw=1, linestyle="--", alpha=0.8)
    inset.fill_betweenx([0, 100], 0, 5.5, alpha=0.08, color=VERMIL)
    inset.set_xlim(0, 100)
    inset.set_ylim(0, 105)
    inset.set_xlabel("Tasks (%)", fontsize=6, labelpad=2)
    inset.set_ylabel("Variety\ncovered (%)", fontsize=6, labelpad=2)
    inset.tick_params(labelsize=5.5)
    inset.text(5.5, 55, "~5%", fontsize=6, color=VERMIL, ha="left")
    inset.set_title("Saturation", fontsize=6.5, pad=2)
    for sp in ["top", "right"]:
        inset.spines[sp].set_visible(False)

    fig.savefig(OUT / "figure1_reframing.pdf")
    fig.savefig(OUT / "figure1_reframing.png")
    print("Saved figure1_reframing.pdf/.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Three-level hierarchy
# ─────────────────────────────────────────────────────────────────────────────

def build_figure2():
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    ax.text(0.5, 1.01, "Three levels of structural abstraction",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="#333333")

    levels = [
        {
            "name":    "Token",
            "sublabel":"sequence",
            "color":   GRAY,
            "bar_w":   0.92,
            "ratio":   "10×",
            "example": "ADD  MODIFY  CALL  RETURN  ASSIGN  IMPORT  ...",
            "use":     "distance baseline · task sequencing",
            "method":  "Levenshtein distance",
            "y":       0.74,
        },
        {
            "name":    "Syntactic",
            "sublabel":"edits / AST",
            "color":   ORANGE,
            "bar_w":   0.70,
            "ratio":   "11×",
            "example": "{op: MODIFY,  target: parse_input,  delta: 3 lines}",
            "use":     "rubric constraints · partial credit · tree edit distance",
            "method":  "AST diff (GumTree-style)",
            "y":       0.44,
        },
        {
            "name":    "Graph",
            "sublabel":"modules",
            "color":   BLUE,
            "bar_w":   0.44,
            "ratio":   "100×",
            "example": "utils  ↔  parser  ↔  validator",
            "use":     "dependency footprint · workflow pattern · graph edit distance",
            "method":  "co-edit subgraph",
            "y":       0.14,
        },
    ]

    bar_x0   = 0.12
    bar_h    = 0.18
    label_x  = 0.01
    ratio_x  = 0.96
    ex_y_off = -0.045
    use_y_off = -0.085

    for lvl in levels:
        y = lvl["y"]
        # bar
        bar = FancyBboxPatch(
            (bar_x0, y - bar_h / 2), lvl["bar_w"], bar_h,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=lvl["color"], alpha=0.18, edgecolor=lvl["color"],
            linewidth=1.2, zorder=2,
        )
        ax.add_patch(bar)

        # level name
        ax.text(label_x, y + 0.01, lvl["name"], ha="left", va="center",
                fontsize=10, fontweight="bold", color=lvl["color"])
        ax.text(label_x, y - 0.028, lvl["sublabel"], ha="left", va="center",
                fontsize=7, color=lvl["color"], alpha=0.75)

        # example (top of bar interior)
        ax.text(bar_x0 + 0.22, y + 0.030,
                lvl["example"], ha="left", va="center",
                fontsize=7.5, color="#333333",
                fontfamily="monospace")

        # method tag (second line, italic)
        ax.text(bar_x0 + 0.22, y - 0.015,
                lvl["method"], ha="left", va="center",
                fontsize=6.5, color=lvl["color"], alpha=0.85, style="italic")

        # use case (third line, below method)
        ax.text(bar_x0 + 0.22, y - 0.055,
                lvl["use"], ha="left", va="center",
                fontsize=6.5, color="#666666")

        # compression ratio
        ax.text(ratio_x, y, lvl["ratio"], ha="right", va="center",
                fontsize=9, color=lvl["color"], fontweight="bold")

    # compression arrow on right
    ax.annotate("", xy=(ratio_x + 0.025, levels[2]["y"]),
                xytext=(ratio_x + 0.025, levels[0]["y"]),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa",
                                lw=1.0, mutation_scale=8))
    ax.text(ratio_x + 0.035, 0.44, "compression", ha="left", va="center",
            fontsize=6, color="#aaaaaa", rotation=270)

    # column headers
    ax.text(bar_x0 + 0.22, 0.975, "Example",
            ha="left", va="center", fontsize=7, color="#999999", fontweight="bold")
    ax.text(ratio_x, 0.975, "Compression",
            ha="right", va="center", fontsize=7, color="#999999", fontweight="bold")

    fig.savefig(OUT / "figure2_hierarchy.pdf")
    fig.savefig(OUT / "figure2_hierarchy.png")
    print("Saved figure2_hierarchy.pdf/.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Procedure-space UMAP
# ─────────────────────────────────────────────────────────────────────────────

def _featurize(df, ds_name: str) -> np.ndarray:
    """Extract fixed-dim feature vector from token / edits / modules columns."""
    import json

    TOKEN_TYPES = ["ADD", "MODIFY", "REMOVE", "CALL", "RETURN", "ASSIGN",
                   "IMPORT", "DEFINE", "CONDITIONAL", "LOOP"]
    OP_TYPES    = ["add", "remove", "modify", "move", "update",
                   "insert", "delete", "replace"]

    rows = []
    for _, row in df.iterrows():
        feat = []

        # ── token level ──────────────────────────────────────────────────────
        toks = row.get("tokens", [])
        if isinstance(toks, str):
            try: toks = json.loads(toks)
            except Exception: toks = []
        toks = [t for t in (toks or []) if isinstance(t, str)]
        total = max(len(toks), 1)
        feat.append(np.log1p(total))
        for tt in TOKEN_TYPES:
            feat.append(sum(1 for t in toks if tt.lower() in t.lower()) / total)

        # ── syntactic (edits) level ───────────────────────────────────────────
        edits = row.get("edits", [])
        if isinstance(edits, str):
            try: edits = json.loads(edits)
            except Exception: edits = []
        edits = edits or []
        n_sites = len(edits)
        n_ops = sum(len(e.get("operations", [])) if isinstance(e, dict) else 0
                    for e in edits)
        total_delta = sum(abs(e.get("delta", 0)) if isinstance(e, dict) else 0
                          for e in edits)
        feat.append(np.log1p(n_sites))
        feat.append(np.log1p(n_ops))
        feat.append(np.log1p(total_delta))
        all_ops = []
        for e in edits:
            if isinstance(e, dict):
                for op in e.get("operations", []):
                    if isinstance(op, dict):
                        all_ops.append(str(op.get("type", "")).lower())
        op_total = max(len(all_ops), 1)
        for ot in OP_TYPES:
            feat.append(sum(1 for o in all_ops if ot in o) / op_total)

        # ── graph (modules) level ─────────────────────────────────────────────
        mods = row.get("modules", [])
        if isinstance(mods, str):
            try: mods = json.loads(mods)
            except Exception: mods = []
        mods = mods or []
        edges = row.get("modules_edges", [])
        if isinstance(edges, str):
            try: edges = json.loads(edges)
            except Exception: edges = []
        edges = edges or []
        n_nodes = len(set(mods)) if isinstance(mods, list) else 0
        n_edges = len(edges)
        feat.append(np.log1p(n_nodes))
        feat.append(np.log1p(n_edges))
        feat.append(float(n_nodes > 1))  # multi-file flag

        rows.append(feat)

    return np.array(rows, dtype=np.float32)


def _style_scatter(ax, title):
    ax.set_title(title, fontsize=9, fontweight="bold", color="#333333", pad=5)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure3():
    """
    Three-panel figure:
      A  UMAP scatter, colored by benchmark — shows structural territory per dataset
      B  UMAP scatter, colored by task type — shows task-type alignment is cross-benchmark
      C  Ridgeline (joy-plot) along UMAP-1 — shows structural spread / overlap per benchmark
         Inspired by density-track visualisations in genomics (Krzywinski et al., Circos 2009)
         and Wilke (2019) 'Fundamentals of Data Visualization', ridgeline chapter.
    """
    import pandas as pd
    from scipy.stats import gaussian_kde

    try:
        import umap as umap_lib
    except ImportError:
        print("umap-learn not installed — skipping figure 3")
        return

    ds_paths = {
        "SWE-bench":     ROOT / "output/datasets/swe_bench_lite_resolved/test.parquet",
        "HumanEval":     ROOT / "output/datasets/humaneval/test.parquet",
        "MBPP":          ROOT / "output/datasets/mbpp/test.parquet",
        "BigCodeBench":  ROOT / "output/datasets/bigcodebench/v0.1.2.parquet",
        "LiveCodeBench": ROOT / "output/datasets/livecodebench/test.parquet",
    }

    frames, labels_ds, labels_type = [], [], []
    for ds_name in DATASET_ORDER:
        path = ds_paths.get(ds_name)
        if path is None or not path.exists():
            print(f"  Skipping {ds_name} — not found")
            continue
        df = pd.read_parquet(path)
        needed = {"tokens", "edits", "modules"}
        if not needed.issubset(df.columns):
            print(f"  Skipping {ds_name} — missing columns: {needed - set(df.columns)}")
            continue
        feats = _featurize(df, ds_name)
        frames.append(feats)
        labels_ds.extend([ds_name] * len(df))
        if "task_type" in df.columns:
            labels_type.extend(df["task_type"].fillna("bug_fix").tolist())
        else:
            default = "bug_fix" if "swe" in ds_name.lower() else "code_generation"
            labels_type.extend([default] * len(df))

    if not frames:
        print("No datasets available for UMAP — skipping figure 3")
        return

    X = np.vstack(frames)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # z-score normalise before UMAP so all 25 features contribute equally
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_scaled = (X - mu) / sigma

    labels_ds_arr  = np.array(labels_ds)
    labels_type_arr = np.array(labels_type)
    ds_present = [d for d in DATASET_ORDER if d in set(labels_ds)]

    print(f"  Running UMAP on {X.shape[0]} instances × {X.shape[1]} features...")
    reducer = umap_lib.UMAP(
        n_neighbors=15, min_dist=0.10,
        metric="euclidean", random_state=42, verbose=False,
    )
    embedding = reducer.fit_transform(X_scaled)

    # ── layout: 2×2 grid, bottom row spans both columns ───────────────────────
    fig = plt.figure(figsize=(8.5, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.8], hspace=0.52, wspace=0.38)
    ax_ds   = fig.add_subplot(gs[0, 0])
    ax_type = fig.add_subplot(gs[0, 1])
    ax_rdg  = fig.add_subplot(gs[1, :])

    # ── panel A: by benchmark ─────────────────────────────────────────────────
    e0_min, e0_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
    e1_min, e1_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
    xx, yy = np.mgrid[e0_min:e0_max:70j, e1_min:e1_max:70j]

    for ds_name in ds_present:
        col  = DATASET_COLORS[ds_name]
        mask = labels_ds_arr == ds_name
        xy   = embedding[mask]
        ax_ds.scatter(xy[:, 0], xy[:, 1], c=col, s=3, alpha=0.38,
                      linewidths=0, label=ds_name, rasterized=True, zorder=2)
        if mask.sum() > 25:
            try:
                kde = gaussian_kde(xy.T, bw_method=0.35)
                z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax_ds.contour(xx, yy, z, levels=2, colors=[col],
                              alpha=0.55, linewidths=0.9, zorder=3)
            except Exception:
                pass

    _style_scatter(ax_ds, "Structural space — by benchmark")
    ax_ds.legend(fontsize=6.2, markerscale=3.0, frameon=True, framealpha=0.92,
                 edgecolor="#dddddd", loc="best", handletextpad=0.3, borderpad=0.5)

    # ── panel B: by task type ─────────────────────────────────────────────────
    for tt, col in TASK_TYPE_COLORS.items():
        mask = labels_type_arr == tt
        if not mask.any():
            continue
        xy = embedding[mask]
        ax_type.scatter(xy[:, 0], xy[:, 1], c=col, s=3, alpha=0.38,
                        linewidths=0, label=TASK_TYPE_LABELS[tt], rasterized=True, zorder=2)

    _style_scatter(ax_type, "Structural space — by task type")
    ax_type.legend(fontsize=6.2, markerscale=3.0, frameon=True, framealpha=0.92,
                   edgecolor="#dddddd", loc="best", handletextpad=0.3, borderpad=0.5)

    # ── panel C: ridgeline density along UMAP-1 ───────────────────────────────
    # Each benchmark gets one filled KDE ridge stacked vertically.
    # The structural spread of a benchmark is readable as ridge width;
    # overlap between ridges indicates shared structural patterns.
    umap1  = embedding[:, 0]
    x_pad  = (umap1.max() - umap1.min()) * 0.04
    x_min  = umap1.min() - x_pad
    x_max  = umap1.max() + x_pad
    xs     = np.linspace(x_min, x_max, 400)

    n_ds     = len(ds_present)
    y_step   = 1.0 / (n_ds + 0.5)
    y_scale  = y_step * 2.2   # ridge height relative to spacing

    ax_rdg.set_xlim(x_min, x_max)
    ax_rdg.set_ylim(-y_step * 0.4, 1.0 + y_step * 0.8)
    ax_rdg.set_xlabel("UMAP dimension 1  (structural variety)", fontsize=8)
    ax_rdg.spines["left"].set_visible(False)
    ax_rdg.spines["top"].set_visible(False)
    ax_rdg.spines["right"].set_visible(False)
    ax_rdg.yaxis.set_visible(False)
    ax_rdg.tick_params(axis="x", labelsize=7)
    ax_rdg.set_title(
        "Structural spread per benchmark  (density along UMAP 1)",  # noqa: RUF001
        fontsize=9, fontweight="bold", color="#333333", pad=5, loc="left",
    )

    for i, ds_name in enumerate(reversed(ds_present)):
        col   = DATASET_COLORS[ds_name]
        mask  = labels_ds_arr == ds_name
        vals  = umap1[mask]
        y_base = (i + 1) * y_step

        try:
            kde     = gaussian_kde(vals, bw_method=0.28)
            density = kde(xs)
            density = density / density.max() * y_scale
        except Exception:
            continue

        # filled ridge with subtle white base line
        ax_rdg.fill_between(xs, y_base, y_base + density,
                             color=col, alpha=0.22, zorder=i + 1)
        ax_rdg.plot(xs, y_base + density, color=col, lw=1.4, zorder=i + 2)
        ax_rdg.axhline(y_base, color="#e8e8e8", lw=0.5, zorder=0)

        # IQR tick marks on baseline
        q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
        ax_rdg.plot([q25, q75], [y_base, y_base], color=col, lw=2.0,
                    solid_capstyle="round", alpha=0.6, zorder=i + 3)

        # label (left) and count (right)
        ax_rdg.text(x_min - (x_max - x_min) * 0.01, y_base + y_scale * 0.42,
                    ds_name, ha="right", va="center",
                    fontsize=7.5, color=col, fontweight="bold")
        ax_rdg.text(x_max + (x_max - x_min) * 0.01, y_base + y_scale * 0.42,
                    f"n={mask.sum()}", ha="left", va="center",
                    fontsize=6.5, color="#999999")

    fig.suptitle(
        "Procedure-space: structural representations across evaluation benchmarks",
        fontsize=10, fontweight="bold", color="#333333", y=0.995,
    )

    fig.savefig(OUT / "figure3_umap.pdf")
    fig.savefig(OUT / "figure3_umap.png")
    print("Saved figure3_umap.pdf/.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", nargs="*", type=int, default=[1, 2, 3],
                        help="Which figures to build (default: all)")
    args = parser.parse_args()

    figs = set(args.figures)
    if 1 in figs:
        print("Building figure 1 (reframing)...")
        build_figure1()
    if 2 in figs:
        print("Building figure 2 (hierarchy)...")
        build_figure2()
    if 3 in figs:
        print("Building figure 3 (UMAP)...")
        build_figure3()

    print(f"\nFigures written to {OUT}/")


if __name__ == "__main__":
    main()
