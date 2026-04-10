#!/usr/bin/env python3
"""
Discover emergent fix forms via a decision tree on edit cert features.

A decision tree trained on the binary edit-op presence matrix to predict
pass/fail partitions instances into leaf nodes. Each leaf is an emergent
fix form: structurally coherent, discriminatively defined, no hand labeling.

Method:
  1. Build binary feature matrix: instance × op_type (1 if op present)
  2. Sweep tree depth 2-6, record leaf-level pass rate variance
  3. Pick elbow depth; label each leaf node
  4. Output: form assignments, pass rates, depth sweep figure, agent heatmap

Usage:
  uv run python scripts/discover_fix_forms.py
"""

import difflib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

for _env in [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
        except ImportError:
            pass

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "fix_forms"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"

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
                norm = frozenset(_NORMALIZE_OPS.get(op, op) for op in ops)
                certs[trace["instance_id"]] = norm
    return certs


def build_feature_matrix(
    instances: list[str],
    certs: dict[str, frozenset[str]],
) -> tuple[np.ndarray, list[str]]:
    all_ops = sorted({op for iid in instances for op in certs.get(iid, set())})
    op_idx = {op: i for i, op in enumerate(all_ops)}
    X = np.zeros((len(instances), len(all_ops)), dtype=np.float32)
    for row, iid in enumerate(instances):
        for op in certs.get(iid, set()):
            if op in op_idx:
                X[row, op_idx[op]] = 1.0
    return X, all_ops


def leaf_pass_rate_variance(tree: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray) -> float:
    leaf_ids = tree.apply(X)
    rates = []
    for lid in np.unique(leaf_ids):
        mask = leaf_ids == lid
        if mask.sum() >= 3:
            rates.append(y[mask].mean())
    return float(np.var(rates)) if len(rates) > 1 else 0.0


def leaf_summary(tree: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray,
                 feature_names: list[str], instances: list[str]) -> list[dict]:
    leaf_ids = tree.apply(X)
    forms = []
    for lid in sorted(np.unique(leaf_ids)):
        mask = leaf_ids == lid
        members = [instances[i] for i in np.where(mask)[0]]
        y_sub = y[mask]
        pass_rate = float(y_sub.mean())
        n = int(mask.sum())

        # Identify defining ops: ops present in majority of leaf members
        X_sub = X[mask]
        op_freq = X_sub.mean(axis=0)
        dominant = [feature_names[i] for i in np.argsort(-op_freq) if op_freq[i] >= 0.5]
        rare = [feature_names[i] for i in np.argsort(-op_freq) if op_freq[i] < 0.1]

        # Short label: top 2 ops or "simple" if none dominant
        if dominant:
            label = "+".join(op.replace("ADD_", "+").replace("DEL_", "-") for op in dominant[:2])
        else:
            label = "minimal_edit"

        forms.append({
            "leaf_id": int(lid),
            "label": label,
            "n": n,
            "pass_rate": pass_rate,
            "dominant_ops": dominant[:6],
            "rare_ops": rare[:4],
            "instance_ids": members,
        })
    return forms


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig_depth_sweep(depths, variances, n_leaves, output_dir: Path):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.15)
    style_panel(ax1)

    ax1.plot(depths, variances, color=TEAL, marker="o", markersize=5,
             linewidth=1.8, label="Leaf pass rate variance")
    ax1.set_xlabel("Tree depth", fontsize=9)
    ax1.set_ylabel("Leaf pass rate variance", fontsize=9, color=TEAL)
    ax1.tick_params(axis="y", labelcolor=TEAL)

    ax2 = ax1.twinx()
    ax2.plot(depths, n_leaves, color=ORANGE, marker="s", markersize=5,
             linewidth=1.8, linestyle="--", label="N leaf forms")
    ax2.set_ylabel("Number of leaf forms", fontsize=9, color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False,
               loc="upper left")
    ax1.set_title("Decision tree depth sweep: form granularity vs discrimination",
                  fontsize=11, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig1_depth_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_depth_sweep.png")


def fig_form_pass_rates(forms: list[dict], depth: int, output_dir: Path):
    forms_sorted = sorted(forms, key=lambda f: -f["pass_rate"])
    labels = [f["label"] for f in forms_sorted]
    pass_rates = [f["pass_rate"] for f in forms_sorted]
    sizes = [f["n"] for f in forms_sorted]

    fig, ax = plt.subplots(figsize=(max(10, len(forms_sorted) * 0.7), 5))
    fig.subplots_adjust(bottom=0.4, left=0.08, right=0.97)
    style_panel(ax)

    xs = np.arange(len(forms_sorted))
    colors = [TEAL if p >= 0.3 else ORANGE if p >= 0.15 else GRAY for p in pass_rates]
    bars = ax.bar(xs, pass_rates, color=colors, alpha=0.85)

    for bar, n in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=7, color=NAVY)

    ax.axhline(0.23, color=NAVY, linewidth=0.8, linestyle=":", label="Baseline (23%)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_title(f"Emergent fix forms at depth={depth}: pass rate per form",
                 fontsize=11, pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False)

    fig.savefig(output_dir / "fig2_form_pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_form_pass_rates.png")


def fig_agent_heatmap(forms: list[dict], agent_results: dict[str, dict[str, bool]],
                      output_dir: Path):
    form_labels = [f["label"] for f in forms]
    agent_names = sorted(agent_results)
    n_forms = len(forms)
    n_agents = len(agent_names)

    # Build per-agent per-form pass rate matrix
    mat = np.full((n_agents, n_forms), np.nan)
    for ai, agent in enumerate(agent_names):
        results = agent_results[agent]
        for fi, form in enumerate(forms):
            members = form["instance_ids"]
            results_in_form = [results[iid] for iid in members if iid in results]
            if results_in_form:
                mat[ai, fi] = np.mean(results_in_form)

    fig, ax = plt.subplots(figsize=(max(12, n_forms * 0.6), max(4, n_agents * 0.7)))
    fig.subplots_adjust(bottom=0.35, left=0.2)

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n_forms))
    ax.set_xticklabels(form_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(agent_names, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Pass rate")
    ax.set_title("Agent coverage by emergent fix form", fontsize=11, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig3_agent_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_agent_heatmap.png")


def load_agent_results() -> dict[str, dict[str, bool]]:
    agent_dir = ROOT / "output" / "swebench_results_lite_agents"
    if not agent_dir.exists():
        return {}
    results = {}
    for p in sorted(agent_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        agent_name = p.stem
        # Support both flat list and nested formats
        if isinstance(data, list):
            results[agent_name] = {r["instance_id"]: bool(r.get("resolved", r.get("passed", False)))
                                   for r in data}
        elif isinstance(data, dict):
            if "results" in data:
                results[agent_name] = {r["instance_id"]: bool(r.get("resolved", r.get("passed", False)))
                                       for r in data["results"]}
            else:
                results[agent_name] = {k: bool(v) for k, v in data.items()}
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pass/fail labels
    fix_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "fix_type", "passed"]]
    print(f"Loaded {len(fix_df)} instances  "
          f"(pass={fix_df['passed'].sum()}, fail={(~fix_df['passed']).sum()})")

    # Load edit certificates
    print("Loading edit certificates...")
    certs = load_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")
    print(f"  {len(certs)} instances have edit certs")

    # Align
    common = sorted(set(certs) & set(fix_df["instance_id"]))
    print(f"  {len(common)} instances in intersection")

    fix_sub = fix_df[fix_df["instance_id"].isin(common)].set_index("instance_id")
    y = np.array([int(fix_sub.loc[iid, "passed"]) for iid in common])

    # Build feature matrix
    print("Building feature matrix...")
    X, feature_names = build_feature_matrix(common, certs)
    print(f"  Shape: {X.shape}")

    # Depth sweep
    depths = list(range(2, 8))
    variances = []
    n_leaves_list = []

    print("\nSweeping tree depth...")
    for d in depths:
        tree = DecisionTreeClassifier(
            max_depth=d, class_weight="balanced", random_state=42, min_samples_leaf=5
        )
        tree.fit(X, y)
        v = leaf_pass_rate_variance(tree, X, y)
        n_leaves = tree.get_n_leaves()
        variances.append(v)
        n_leaves_list.append(n_leaves)
        print(f"  depth={d}: {n_leaves:2d} leaves, pass rate variance={v:.4f}")

    fig_depth_sweep(depths, variances, n_leaves_list, OUTPUT_DIR)

    # Pick elbow: max variance per leaf (variance / n_leaves)
    scores = [v / max(n, 1) for v, n in zip(variances, n_leaves_list)]
    best_depth = depths[np.argmax(scores)]
    print(f"\nSelected depth={best_depth} (best variance/leaf ratio)")

    # Fit final tree at selected depth
    tree_final = DecisionTreeClassifier(
        max_depth=best_depth, class_weight="balanced", random_state=42, min_samples_leaf=5
    )
    tree_final.fit(X, y)
    forms = leaf_summary(tree_final, X, y, feature_names, common)

    print(f"\n{len(forms)} emergent fix forms at depth={best_depth}:")
    for f in sorted(forms, key=lambda x: -x["pass_rate"]):
        print(f"  [{f['label']:30s}] n={f['n']:3d}  pass={f['pass_rate']:.2f}  "
              f"ops={f['dominant_ops'][:3]}")

    # Figures
    print("\nGenerating figures...")
    fig_form_pass_rates(forms, best_depth, OUTPUT_DIR)

    # Agent heatmap
    agent_results = load_agent_results()
    if agent_results:
        print(f"Loaded {len(agent_results)} agents for heatmap")
        fig_agent_heatmap(forms, agent_results, OUTPUT_DIR)
    else:
        print("No agent results found, skipping heatmap")

    # Save assignments
    rows = []
    for form in forms:
        for iid in form["instance_ids"]:
            rows.append({
                "instance_id": iid,
                "form_label": form["label"],
                "form_leaf_id": form["leaf_id"],
                "form_pass_rate": form["pass_rate"],
                "form_n": form["n"],
                "passed": bool(fix_sub.loc[iid, "passed"]),
                "fix_type_hand": fix_sub.loc[iid, "fix_type"],
            })
    assignments_df = pd.DataFrame(rows)
    assignments_df.to_parquet(OUTPUT_DIR / "form_assignments.parquet", index=False)
    print(f"\nSaved form_assignments.parquet ({len(assignments_df)} rows)")

    # Save form summaries
    summaries = [
        {k: v for k, v in f.items() if k != "instance_ids"}
        for f in forms
    ]
    with open(OUTPUT_DIR / "form_summaries.json", "w") as fh:
        json.dump({"depth": best_depth, "n_forms": len(forms), "forms": summaries}, fh, indent=2)
    print("Saved form_summaries.json")

    # Print tree structure for inspection
    print("\nDecision tree structure:")
    print(export_text(tree_final, feature_names=feature_names, max_depth=best_depth))

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
