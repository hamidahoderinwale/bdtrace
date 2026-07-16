#!/usr/bin/env python3
"""
Strategy saturation: does adding more benchmark instances add new strategy coverage?

Two analyses:
  1. Within-benchmark accumulation: how quickly do all 10 forms appear as instances
     are added in random order? (saturates early = redundant volume)
  2. Cross-benchmark form distribution: do Lite, Verified, SWE-smith sample the
     same strategy space or reveal coverage gaps?

The decision tree fitted on Lite is applied to Verified and SWE-smith edit certs.
Form assignments from discover_fix_forms.py are used as the reference taxonomy.

Usage:
  uv run python scripts/strategy_saturation.py
"""

import difflib
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "strategy_saturation"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"
GREEN = "#009E73"

BENCH_COLORS = {
    "SWE-bench Lite": TEAL,
    "SWE-bench Verified": ORANGE,
    "SWE-smith": GREEN,
}

_NORMALIZE_OPS = {
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


def load_certs(traces_path: Path, limit: int = None) -> dict[str, frozenset[str]]:
    certs = {}
    with open(traces_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
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
                norm = frozenset(_NORMALIZE_OPS.get(op, op) for op in ops)
                certs[trace["instance_id"]] = norm
    return certs


def build_feature_vector(cert: frozenset[str], feature_names: list[str]) -> np.ndarray:
    op_idx = {op: i for i, op in enumerate(feature_names)}
    x = np.zeros(len(feature_names), dtype=np.float32)
    for op in cert:
        if op in op_idx:
            x[op_idx[op]] = 1.0
    return x


def fit_tree_on_lite(
    certs: dict[str, frozenset[str]],
    labels_df: pd.DataFrame,
) -> tuple[DecisionTreeClassifier, list[str]]:
    common = sorted(set(certs) & set(labels_df["instance_id"]))
    all_ops = sorted({op for iid in common for op in certs[iid]})

    X = np.array([build_feature_vector(certs[iid], all_ops) for iid in common])
    y = np.array([int(labels_df.set_index("instance_id").loc[iid, "passed"])
                  for iid in common])

    tree = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=42, min_samples_leaf=5
    )
    tree.fit(X, y)
    return tree, all_ops


def assign_forms(
    certs: dict[str, frozenset[str]],
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    form_labels: dict[int, str],
) -> dict[str, str]:
    assignments = {}
    for iid, cert in certs.items():
        x = build_feature_vector(cert, feature_names).reshape(1, -1)
        leaf_id = int(tree.apply(x)[0])
        assignments[iid] = form_labels.get(leaf_id, f"leaf_{leaf_id}")
    return assignments


def accumulation_curve(
    assignments: dict[str, str],
    n_shuffles: int = 30,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)
    ids = list(assignments.keys())
    n = len(ids)
    curves = []
    for _ in range(n_shuffles):
        order = rng.permutation(ids)
        seen = set()
        curve = []
        for iid in order:
            seen.add(assignments[iid])
            curve.append(len(seen))
        curves.append(curve)
    curves = np.array(curves)
    return np.arange(1, n + 1), curves.mean(axis=0), curves.std(axis=0)


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig_accumulation(bench_curves: dict[str, tuple], n_forms: int, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(bottom=0.15, right=0.75)
    style_panel(ax)

    for bench, (xs, means, stds) in bench_curves.items():
        color = BENCH_COLORS.get(bench, GRAY)
        ax.plot(xs, means, color=color, linewidth=1.8, label=bench)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)

    ax.axhline(n_forms, color=GRAY, linewidth=0.8, linestyle=":",
               label=f"All {n_forms} forms")
    ax.set_xlabel("Instances added", fontsize=9)
    ax.set_ylabel("Distinct strategy forms seen", fontsize=9)
    ax.set_title("Strategy form accumulation across benchmarks",
                 fontsize=11, pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.savefig(output_dir / "fig1_accumulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_accumulation.png")


def fig_form_distribution(bench_distributions: dict[str, dict[str, float]],
                          form_order: list[str], output_dir: Path):
    benches = list(bench_distributions.keys())
    n_forms = len(form_order)
    n_benches = len(benches)

    x = np.arange(n_forms)
    width = 0.8 / n_benches

    fig, ax = plt.subplots(figsize=(max(12, n_forms * 0.8), 5))
    fig.subplots_adjust(bottom=0.35, left=0.08, right=0.97)
    style_panel(ax)

    for bi, bench in enumerate(benches):
        dist = bench_distributions[bench]
        vals = [dist.get(f, 0.0) for f in form_order]
        offset = (bi - n_benches / 2 + 0.5) * width
        color = BENCH_COLORS.get(bench, GRAY)
        ax.bar(x + offset, vals, width=width * 0.9, color=color, alpha=0.85,
               label=bench)

    ax.set_xticks(x)
    ax.set_xticklabels(form_order, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Fraction of benchmark instances", fontsize=9)
    ax.set_title("Strategy form distribution across benchmarks",
                 fontsize=11, pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False)

    fig.savefig(output_dir / "fig2_form_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_form_distribution.png")


def fig_coverage_gap(bench_distributions: dict[str, dict[str, float]],
                     form_order: list[str], output_dir: Path):
    benches = list(bench_distributions.keys())
    n_forms = len(form_order)
    n_benches = len(benches)

    mat = np.zeros((n_benches, n_forms))
    for bi, bench in enumerate(benches):
        dist = bench_distributions[bench]
        for fi, f in enumerate(form_order):
            mat[bi, fi] = dist.get(f, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, n_forms * 0.7), max(3, n_benches * 0.8)))
    fig.subplots_adjust(bottom=0.35, left=0.2)

    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(n_forms))
    ax.set_xticklabels(form_order, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_benches))
    ax.set_yticklabels(benches, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Fraction of benchmark")

    for bi in range(n_benches):
        for fi in range(n_forms):
            v = mat[bi, fi]
            ax.text(fi, bi, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if v > 0.3 else NAVY)

    ax.set_title("Strategy coverage by benchmark (fraction of instances per form)",
                 fontsize=10, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig3_coverage_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_coverage_gap.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Lite form assignments as reference taxonomy
    form_df = pd.read_parquet(ROOT / "output" / "fix_forms" / "form_assignments.parquet")
    lite_labels_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "passed"]]

    # Build form_labels map: leaf_id -> label (from Lite assignments)
    leaf_to_label = dict(zip(form_df["form_leaf_id"], form_df["form_label"]))
    n_forms = len(leaf_to_label)
    print(f"Reference taxonomy: {n_forms} forms from SWE-bench Lite decision tree")

    # Load Lite edit certs and refit tree
    print("\nLoading Lite edit certs...")
    lite_certs = load_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")
    print(f"  {len(lite_certs)} instances")

    print("Fitting decision tree on Lite...")
    tree, feature_names = fit_tree_on_lite(lite_certs, lite_labels_df)

    # Assign forms to Lite instances
    lite_assignments = assign_forms(lite_certs, tree, feature_names, leaf_to_label)

    # Load Verified edit certs
    print("\nLoading Verified edit certs...")
    verified_certs = load_certs(ROOT / "output" / "resolved_traces_verified_full.jsonl")
    print(f"  {len(verified_certs)} instances")
    verified_assignments = assign_forms(verified_certs, tree, feature_names, leaf_to_label)

    # Load SWE-smith (stratified sample)
    print("\nLoading SWE-smith edit certs (stratified sample)...")
    smith_certs = load_certs(ROOT / "output" / "resolved_traces_swe_smith_stratified.jsonl")
    print(f"  {len(smith_certs)} instances")
    smith_assignments = assign_forms(smith_certs, tree, feature_names, leaf_to_label)

    # Accumulation curves
    print("\nComputing accumulation curves...")
    rng = np.random.default_rng(42)
    bench_curves = {}
    for bench, assignments in [
        ("SWE-bench Lite", lite_assignments),
        ("SWE-bench Verified", verified_assignments),
        ("SWE-smith", smith_assignments),
    ]:
        xs, means, stds = accumulation_curve(assignments, n_shuffles=50, rng=rng)
        bench_curves[bench] = (xs, means, stds)
        saturation_point = next(
            (i + 1 for i, m in enumerate(means) if m >= n_forms), len(xs)
        )
        print(f"  {bench}: saturates at ~{saturation_point} instances "
              f"(of {len(xs)} total), "
              f"final coverage={means[-1]:.1f}/{n_forms} forms")

    # Form distributions
    print("\nComputing form distributions...")
    bench_distributions = {}
    for bench, assignments in [
        ("SWE-bench Lite", lite_assignments),
        ("SWE-bench Verified", verified_assignments),
        ("SWE-smith", smith_assignments),
    ]:
        total = len(assignments)
        from collections import Counter
        counts = Counter(assignments.values())
        bench_distributions[bench] = {f: counts.get(f, 0) / total for f in leaf_to_label.values()}
        print(f"\n  {bench} form distribution:")
        for form, frac in sorted(bench_distributions[bench].items(), key=lambda x: -x[1]):
            n = counts.get(form, 0)
            print(f"    {form:30s}: {frac:.2f} (n={n})")

    # Form order: sorted by Lite fraction descending
    form_order = sorted(
        leaf_to_label.values(),
        key=lambda f: -bench_distributions["SWE-bench Lite"].get(f, 0)
    )

    # Save results
    results = {
        "n_forms": n_forms,
        "benchmarks": {}
    }
    for bench, assignments in [
        ("SWE-bench Lite", lite_assignments),
        ("SWE-bench Verified", verified_assignments),
        ("SWE-smith", smith_assignments),
    ]:
        xs, means, _ = bench_curves[bench]
        saturation_point = next(
            (i + 1 for i, m in enumerate(means) if m >= n_forms), len(xs)
        )
        results["benchmarks"][bench] = {
            "n_instances": len(assignments),
            "n_forms_covered": int(means[-1].round()),
            "saturation_at": saturation_point,
            "form_distribution": bench_distributions[bench],
        }
    with open(OUTPUT_DIR / "saturation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved saturation_results.json")

    print("\nGenerating figures...")
    fig_accumulation(bench_curves, n_forms, OUTPUT_DIR)
    fig_form_distribution(bench_distributions, form_order, OUTPUT_DIR)
    fig_coverage_gap(bench_distributions, form_order, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
