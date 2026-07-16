#!/usr/bin/env python3
"""
Strategy form discovery via frequent itemset mining on edit certificates.

Asks: what is the right level of abstraction for fix strategies?
  - 13 hand-labeled fix types = too coarse?
  - 119 raw AST op types = too fine
  - FIM closed frequent itemsets = principled middle ground?

Sweeps support thresholds and finds the level where itemsets are most
predictive of pass/fail. Compares to existing 13-type taxonomy.

Outputs:
  fig1_support_sweep.png         -- n_patterns and pass_rate_variance vs support
  fig2_top_patterns.png          -- top closed itemsets by lift on pass rate
  fig3_vs_fix_types.png          -- FIM patterns vs hand-labeled fix types
  strategy_forms.parquet         -- per-instance FIM pattern assignments
  frequent_itemsets.json         -- all closed itemsets at chosen support

Usage:
  uv run python scripts/analyze_strategy_forms.py
  uv run python scripts/analyze_strategy_forms.py --support 0.08
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "strategy_forms"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"

# Normalize fallback keyword tokens to their proper AST equivalents
# (patch_to_ast_sequence emits lowercase fallbacks when AST parse fails)
_NORMALIZE = {
    "ADD_if": "ADD_If", "DEL_if": "DEL_If",
    "ADD_for": "ADD_For", "DEL_for": "DEL_For",
    "ADD_return": "ADD_Return", "DEL_return": "DEL_Return",
    "ADD_raise": "ADD_Raise", "DEL_raise": "DEL_Raise",
    "ADD_try": "ADD_Try", "DEL_try": "DEL_Try",
    "ADD_while": "ADD_While", "DEL_while": "DEL_While",
    "ADD_with": "ADD_With", "DEL_with": "DEL_With",
    "ADD_def": "ADD_FunctionDef", "DEL_def": "DEL_FunctionDef",
    "ADD_class": "ADD_ClassDef", "DEL_class": "DEL_ClassDef",
    "ADD_elif": "ADD_If", "DEL_elif": "DEL_If",
    "ADD_else": "ADD_If", "DEL_else": "DEL_If",
    "ADD_except": "ADD_ExceptHandler", "DEL_except": "DEL_ExceptHandler",
    "ADD_assert": "ADD_Assert",
}


def normalize_cert(ops: list[str]) -> frozenset[str]:
    return frozenset(_NORMALIZE.get(op, op) for op in ops)


def load_certs(traces_path: Path) -> dict[str, frozenset[str]]:
    certs = {}
    with open(traces_path) as f:
        for line in f:
            trace = json.loads(line)
            ops = []
            for ev in trace["events"]:
                if ev["type"] != "code_change":
                    continue
                d = ev["details"]
                if not d["file_path"].endswith(".py"):
                    continue
                before = d["before_content"].splitlines(keepends=True)
                after = d["after_content"].splitlines(keepends=True)
                raw = "".join(difflib.unified_diff(
                    before, after, fromfile=d["file_path"], tofile=d["file_path"]
                ))
                if not raw:
                    continue
                diff = f"diff --git a/{d['file_path']} b/{d['file_path']}\n" + raw
                ops.extend(patch_to_ast_sequence(diff))
            if ops:
                certs[trace["instance_id"]] = normalize_cert(ops)
    return certs


def certs_to_df(certs: dict[str, frozenset[str]]) -> pd.DataFrame:
    transactions = [list(cert) for cert in certs.values()]
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_,
                      index=list(certs.keys()))
    return df


def run_fim(binary_df: pd.DataFrame, min_support: float) -> pd.DataFrame:
    return fpgrowth(binary_df, min_support=min_support, use_colnames=True)


def assign_patterns(certs: dict, itemsets: pd.DataFrame,
                    min_size: int = 2) -> dict[str, list[frozenset]]:
    large = itemsets[itemsets["itemsets"].apply(len) >= min_size].copy()
    large["size"] = large["itemsets"].apply(len)
    large = large.sort_values(["size", "support"], ascending=[False, False])

    assignments: dict[str, list[frozenset]] = {}
    for iid, cert in certs.items():
        matching = [fs for fs in large["itemsets"] if fs.issubset(cert)]
        assignments[iid] = matching
    return assignments


def pass_rate_variance(itemsets: pd.DataFrame, certs: dict,
                       fix_df: pd.DataFrame) -> pd.Series:
    pass_map = fix_df.set_index("instance_id")["passed"].to_dict()
    variances = []
    for _, row in itemsets.iterrows():
        pattern = row["itemsets"]
        if len(pattern) < 2:
            variances.append(np.nan)
            continue
        has_pattern = [iid for iid, cert in certs.items()
                       if pattern.issubset(cert) and iid in pass_map]
        not_pattern = [iid for iid, cert in certs.items()
                       if not pattern.issubset(cert) and iid in pass_map]
        r1 = np.mean([pass_map[i] for i in has_pattern]) if has_pattern else np.nan
        r2 = np.mean([pass_map[i] for i in not_pattern]) if not_pattern else np.nan
        if not np.isnan(r1) and not np.isnan(r2):
            variances.append(abs(r1 - r2))
        else:
            variances.append(np.nan)
    return pd.Series(variances, index=itemsets.index)


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig1_support_sweep(sweep_results: list[dict], output_dir: Path):
    supports = [r["support"] for r in sweep_results]
    n_patterns = [r["n_patterns"] for r in sweep_results]
    mean_variance = [r["mean_variance"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.15)
    style_panel(ax1)

    ax1.plot(supports, n_patterns, color=TEAL, marker="o", markersize=4, linewidth=1.5)
    ax1.set_xlabel("Min support threshold", fontsize=9)
    ax1.set_ylabel("Number of frequent itemsets (size >= 2)", fontsize=9, color=TEAL)
    ax1.tick_params(axis="y", labelcolor=TEAL)

    ax2 = ax1.twinx()
    ax2.plot(supports, mean_variance, color=ORANGE, marker="s", markersize=4,
             linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Mean pass-rate lift (has pattern vs not)", fontsize=9, color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)

    patches = [
        mpatches.Patch(color=TEAL, label="N patterns"),
        mpatches.Patch(color=ORANGE, label="Pass-rate lift"),
    ]
    ax1.legend(handles=patches, fontsize=9, frameon=False, loc="upper right")
    ax1.set_title("FIM granularity sweep", fontsize=11, pad=6, fontweight="normal")
    ax1.invert_xaxis()

    fig.savefig(output_dir / "fig1_support_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_support_sweep.png")


def fig2_top_patterns(itemsets: pd.DataFrame, variances: pd.Series,
                      certs: dict, fix_df: pd.DataFrame, output_dir: Path,
                      top_n: int = 15):
    pass_map = fix_df.set_index("instance_id")["passed"].to_dict()
    large = itemsets[itemsets["itemsets"].apply(len) >= 2].copy()
    large["lift"] = variances[large.index]
    large = large.dropna(subset=["lift"]).sort_values("lift", ascending=False).head(top_n)

    labels = []
    pass_rates_with = []
    pass_rates_without = []

    for _, row in large.iterrows():
        pattern = row["itemsets"]
        label = " + ".join(sorted(op.replace("ADD_", "+").replace("DEL_", "-")
                                  for op in pattern))
        labels.append(label)
        with_p = [iid for iid, cert in certs.items()
                  if pattern.issubset(cert) and iid in pass_map]
        without_p = [iid for iid, cert in certs.items()
                     if not pattern.issubset(cert) and iid in pass_map]
        pass_rates_with.append(np.mean([pass_map[i] for i in with_p]) if with_p else 0)
        pass_rates_without.append(np.mean([pass_map[i] for i in without_p]) if without_p else 0)

    xs = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(bottom=0.4)
    style_panel(ax)

    ax.bar(xs - width/2, pass_rates_with, width, color=TEAL, alpha=0.85, label="Has pattern")
    ax.bar(xs + width/2, pass_rates_without, width, color=GRAY, alpha=0.85, label="No pattern")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title(f"Top {top_n} patterns by pass-rate lift", fontsize=11, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig2_top_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_top_patterns.png")


def fig3_vs_fix_types(itemsets: pd.DataFrame, variances: pd.Series,
                      certs: dict, fix_df: pd.DataFrame, output_dir: Path):
    pass_map = fix_df.set_index("instance_id")["passed"].to_dict()
    type_map = fix_df.set_index("instance_id")["fix_type"].to_dict()

    # For each fix type, what fraction of instances are "explained" by a pattern?
    types = fix_df["fix_type"].value_counts()
    types = types[types >= 5].index.tolist()

    large = itemsets[itemsets["itemsets"].apply(len) >= 2].copy()

    coverage_by_type = {}
    for ft in types:
        ft_instances = [iid for iid in certs if type_map.get(iid) == ft]
        if not ft_instances:
            continue
        covered = sum(
            1 for iid in ft_instances
            if any(pat.issubset(certs[iid]) for pat in large["itemsets"])
        )
        coverage_by_type[ft] = covered / len(ft_instances)

    pass_rate_by_type = fix_df[fix_df["fix_type"].isin(types)].groupby("fix_type")["passed"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.35, bottom=0.25)

    # Left: pass rate by fix type
    ax = axes[0]
    style_panel(ax)
    sorted_types = pass_rate_by_type.sort_values(ascending=False).index
    xs = np.arange(len(sorted_types))
    ax.bar(xs, pass_rate_by_type[sorted_types], color=TEAL, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace("_", " ") for t in sorted_types],
                       fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_title("Pass rate by hand-labeled fix type", fontsize=10, pad=6, fontweight="normal")

    # Right: FIM coverage by fix type
    ax = axes[1]
    style_panel(ax)
    cov_sorted = sorted(coverage_by_type.items(), key=lambda x: -x[1])
    ft_labels = [t.replace("_", " ") for t, _ in cov_sorted]
    cov_vals = [v for _, v in cov_sorted]
    xs2 = np.arange(len(ft_labels))
    ax.bar(xs2, cov_vals, color=ORANGE, alpha=0.85)
    ax.set_xticks(xs2)
    ax.set_xticklabels(ft_labels, fontsize=8, rotation=40, ha="right")
    ax.set_ylabel("Fraction covered by a frequent pattern", fontsize=9)
    ax.set_title("FIM coverage by fix type", fontsize=10, pad=6, fontweight="normal")
    ax.set_ylim(0, 1)

    fig.suptitle("Hand-labeled fix types vs data-driven FIM patterns", fontsize=11,
                 y=1.01, fontweight="normal")
    fig.savefig(output_dir / "fig3_vs_fix_types.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_vs_fix_types.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=float, default=None,
                        help="Fixed support threshold (default: sweep and pick best)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading edit certificates...")
    certs = load_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")
    print(f"  {len(certs)} instances, vocab size: "
          f"{len(set(op for ops in certs.values() for op in ops))}")

    print("Loading fix type labels...")
    fix_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "fix_type", "passed"]]

    binary_df = certs_to_df(certs)
    print(f"  Binary matrix: {binary_df.shape}")

    # Support sweep
    supports = [0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03]
    sweep_results = []

    print("Sweeping support thresholds...")
    for sup in supports:
        items = run_fim(binary_df, min_support=sup)
        large = items[items["itemsets"].apply(len) >= 2]
        variances = pass_rate_variance(large, certs, fix_df)
        mean_var = variances.dropna().mean()
        sweep_results.append({
            "support": sup,
            "n_patterns": len(large),
            "mean_variance": mean_var,
        })
        print(f"  support={sup:.2f}: {len(large)} patterns (size>=2), "
              f"mean lift={mean_var:.3f}")

    fig1_support_sweep(sweep_results, args.output_dir)

    # Pick best support: highest mean variance (most discriminative)
    if args.support:
        chosen_support = args.support
    else:
        best = max(sweep_results, key=lambda r: r["mean_variance"] or 0)
        chosen_support = best["support"]
        print(f"\nBest support by pass-rate lift: {chosen_support:.2f} "
              f"({best['n_patterns']} patterns)")

    print(f"\nRunning FIM at support={chosen_support:.2f}...")
    itemsets = run_fim(binary_df, min_support=chosen_support)
    large = itemsets[itemsets["itemsets"].apply(len) >= 2]
    variances = pass_rate_variance(large, certs, fix_df)

    print(f"  {len(large)} frequent itemsets (size >= 2)")
    top = large.assign(lift=variances).sort_values("lift", ascending=False).head(20)
    print("\nTop 20 patterns by pass-rate lift:")
    for _, row in top.iterrows():
        ops = " + ".join(sorted(row["itemsets"]))
        print(f"  sup={row['support']:.2f} lift={row['lift']:.3f}  {ops}")

    fig2_top_patterns(large, variances, certs, fix_df, args.output_dir)
    fig3_vs_fix_types(large, variances, certs, fix_df, args.output_dir)

    # Save itemsets
    out = [
        {
            "itemset": sorted(row["itemsets"]),
            "support": float(row["support"]),
            "lift": float(variances[i]) if not np.isnan(variances[i]) else None,
        }
        for i, row in large.iterrows()
    ]
    with open(args.output_dir / "frequent_itemsets.json", "w") as f:
        json.dump({"support": chosen_support, "n_patterns": len(out), "patterns": out}, f, indent=2)
    print(f"\nSaved frequent_itemsets.json ({len(out)} patterns)")
    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
