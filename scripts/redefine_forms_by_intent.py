#!/usr/bin/env python3
"""
Redefine fix forms using intent sequences instead of raw AST op types.

Replaces the bag-of-edit-types decision tree (discover_fix_forms.py) with
one trained on binary intent-mechanism features. Forms are now labeled by
mechanism names (algorithm_replace, refactor_iteration) rather than AST ops
(+For++If), making the capability frontier claim coherent and interpretable.

Method:
  1. Load intent sequences from build_intent_sequences.py output
  2. Build binary feature matrix: instance × mechanism_label
  3. Sweep decision tree depth 2-6
  4. Pick elbow; label leaf nodes by dominant mechanisms
  5. Re-run frontier analysis: which forms are still unreachable?

Usage:
  uv run python scripts/redefine_forms_by_intent.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "intent_forms"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"
GREEN = "#009E73"

AGENT_SHORT = {
    "lite_20240402_rag_gpt4": "RAG GPT-4",
    "lite_20240402_sweagent_gpt4": "SWE-agent GPT-4",
    "lite_20240620_sweagent_claude3.5sonnet": "SWE-agent Claude 3.5",
    "lite_20240728_sweagent_gpt4o": "SWE-agent GPT-4o",
    "lite_20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": "SWE-Fixer Qwen",
}


def build_feature_matrix(
    instance_ids: list[str],
    sequences: dict[str, list[str]],
    mechanism_labels: list[str],
) -> np.ndarray:
    label_idx = {m: i for i, m in enumerate(mechanism_labels)}
    X = np.zeros((len(instance_ids), len(mechanism_labels)), dtype=np.float32)
    for row, iid in enumerate(instance_ids):
        for mech in sequences.get(iid, []):
            if mech in label_idx:
                X[row, label_idx[mech]] = 1.0
    return X


def leaf_pass_rate_variance(tree, X, y):
    leaf_ids = tree.apply(X)
    rates = [y[leaf_ids == lid].mean()
             for lid in np.unique(leaf_ids)
             if (leaf_ids == lid).sum() >= 3]
    return float(np.var(rates)) if len(rates) > 1 else 0.0


def leaf_summary(tree, X, y, feature_names, instances):
    leaf_ids = tree.apply(X)
    forms = []
    for lid in sorted(np.unique(leaf_ids)):
        mask = leaf_ids == lid
        members = [instances[i] for i in np.where(mask)[0]]
        y_sub = y[mask]
        X_sub = X[mask]
        pass_rate = float(y_sub.mean())
        n = int(mask.sum())
        op_freq = X_sub.mean(axis=0)
        dominant = [feature_names[i] for i in np.argsort(-op_freq) if op_freq[i] >= 0.5]
        label = " + ".join(dominant[:3]) if dominant else "minimal"
        forms.append({
            "leaf_id": int(lid),
            "label": label,
            "n": n,
            "pass_rate": pass_rate,
            "dominant_mechanisms": dominant[:4],
            "instance_ids": members,
        })
    return forms


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig_depth_sweep(depths, variances, n_leaves, best_depth, output_dir):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.15)
    style_panel(ax1)
    ax1.plot(depths, variances, color=TEAL, marker="o", markersize=5, linewidth=1.8)
    ax1.axvline(best_depth, color=GRAY, linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Tree depth", fontsize=9)
    ax1.set_ylabel("Leaf pass rate variance", fontsize=9, color=TEAL)
    ax1.tick_params(axis="y", labelcolor=TEAL)
    ax2 = ax1.twinx()
    ax2.plot(depths, n_leaves, color=ORANGE, marker="s", markersize=5,
             linewidth=1.8, linestyle="--")
    ax2.set_ylabel("Number of forms", fontsize=9, color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=9)
    ax1.set_title("Intent-based form depth sweep", fontsize=11, pad=6, fontweight="normal")
    fig.savefig(output_dir / "fig1_depth_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_depth_sweep.png")


def fig_form_pass_rates(forms, depth, output_dir):
    forms_sorted = sorted(forms, key=lambda f: -f["pass_rate"])
    fig, ax = plt.subplots(figsize=(max(10, len(forms_sorted) * 0.8), 5))
    fig.subplots_adjust(bottom=0.45, left=0.07, right=0.97)
    style_panel(ax)
    xs = np.arange(len(forms_sorted))
    colors = [TEAL if f["pass_rate"] >= 0.3 else ORANGE if f["pass_rate"] >= 0.15
              else GRAY for f in forms_sorted]
    bars = ax.bar(xs, [f["pass_rate"] for f in forms_sorted], color=colors, alpha=0.85)
    for bar, f in zip(bars, forms_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={f['n']}", ha="center", va="bottom", fontsize=7, color=NAVY)
    ax.axhline(0.23, color=NAVY, linewidth=0.8, linestyle=":", label="Baseline (23%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f["label"] for f in forms_sorted], fontsize=7,
                       rotation=45, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_title(f"Intent-defined fix forms at depth={depth}: pass rate per form",
                 fontsize=11, pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(output_dir / "fig2_form_pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_form_pass_rates.png")


def fig_frontier(forms, agent_results, output_dir):
    n_agents = len(agent_results)
    form_order = sorted(forms, key=lambda f: -f["pass_rate"])
    n_forms = len(form_order)

    # Per form: fraction of instances unsolved by all agents
    frac_unsolved = []
    for form in form_order:
        members = set(form["instance_ids"])
        n_unsolved = sum(
            1 for iid in members
            if not any(res.get(iid, False) for res in agent_results.values())
        )
        frac_unsolved.append(n_unsolved / max(len(members), 1))

    fig, ax = plt.subplots(figsize=(max(10, n_forms * 0.8), 5))
    fig.subplots_adjust(bottom=0.45)
    style_panel(ax)
    xs = np.arange(n_forms)
    colors = [GRAY if f >= 0.5 else ORANGE if f >= 0.2 else TEAL for f in frac_unsolved]
    ax.bar(xs, frac_unsolved, color=colors, alpha=0.85)
    for xi, (f, form) in enumerate(zip(frac_unsolved, form_order)):
        ax.text(xi, f + 0.01, f"n={form['n']}", ha="center", va="bottom",
                fontsize=7, color=NAVY)
    ax.set_xticks(xs)
    ax.set_xticklabels([f["label"] for f in form_order], fontsize=7,
                       rotation=45, ha="right")
    ax.set_ylabel("Fraction of instances unsolved by all agents", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Instance-level coverage gaps across {n_agents} leaderboard agents",
                 fontsize=11, pad=6, fontweight="normal")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GRAY, alpha=0.85, label=">=50% unsolved (structural frontier)"),
        Patch(facecolor=ORANGE, alpha=0.85, label="20-50% unsolved"),
        Patch(facecolor=TEAL, alpha=0.85, label="<20% unsolved"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False)
    fig.savefig(output_dir / "fig3_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_frontier.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load intent sequences
    seq_path = ROOT / "output" / "intent_sequences" / "sequences.json"
    if not seq_path.exists():
        print("ERROR: Run build_intent_sequences.py first.")
        sys.exit(1)
    with open(seq_path) as f:
        sequences = json.load(f)
    print(f"Loaded sequences for {len(sequences)} instances")

    # Load pass/fail labels
    fix_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "passed"]]

    # Align
    common = sorted(set(sequences) & set(fix_df["instance_id"]))
    # Exclude empty sequences
    common = [iid for iid in common if sequences[iid]]
    print(f"Instances with non-empty sequences: {len(common)}")

    fix_sub = fix_df[fix_df["instance_id"].isin(common)].set_index("instance_id")
    y = np.array([int(fix_sub.loc[iid, "passed"]) for iid in common])
    print(f"Pass: {y.sum()}, Fail: {(1-y).sum()}")

    from representations.inferred.fix_type.chunk_intent import MECHANISM_LABELS
    X = build_feature_matrix(common, sequences, MECHANISM_LABELS)
    print(f"Feature matrix: {X.shape}  ({len(MECHANISM_LABELS)} mechanisms)")

    # Depth sweep
    depths = list(range(2, 7))
    variances, n_leaves_list = [], []
    print("\nSweeping depth...")
    for d in depths:
        tree = DecisionTreeClassifier(
            max_depth=d, class_weight="balanced", random_state=42, min_samples_leaf=3
        )
        tree.fit(X, y)
        v = leaf_pass_rate_variance(tree, X, y)
        n = tree.get_n_leaves()
        variances.append(v)
        n_leaves_list.append(n)
        print(f"  depth={d}: {n} leaves, variance={v:.4f}")

    # Pick depth at the largest marginal gain in variance (elbow by delta)
    deltas = [variances[i] - variances[i - 1] for i in range(1, len(variances))]
    best_depth = depths[np.argmax(deltas) + 1]
    print(f"\nSelected depth={best_depth} (largest variance gain: +{max(deltas):.4f})")

    tree_final = DecisionTreeClassifier(
        max_depth=best_depth, class_weight="balanced", random_state=42, min_samples_leaf=3
    )
    tree_final.fit(X, y)
    forms = leaf_summary(tree_final, X, y, MECHANISM_LABELS, common)

    print(f"\n{len(forms)} intent-defined forms:")
    for f in sorted(forms, key=lambda x: -x["pass_rate"]):
        print(f"  [{f['label']:45s}] n={f['n']:3d}  pass={f['pass_rate']:.2f}")

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
                "mechanisms": sequences.get(iid, []),
            })
    pd.DataFrame(rows).to_parquet(OUTPUT_DIR / "form_assignments.parquet", index=False)
    print(f"\nSaved form_assignments.parquet ({len(rows)} rows)")

    with open(OUTPUT_DIR / "form_summaries.json", "w") as f:
        json.dump({
            "depth": best_depth,
            "n_forms": len(forms),
            "forms": [{k: v for k, v in f.items() if k != "instance_ids"}
                      for f in forms]
        }, f, indent=2)
    print("Saved form_summaries.json")

    # Frontier analysis — prefer 84-agent leaderboard msgpack
    agent_results = {}
    lb_path = ROOT / "output" / "leaderboard" / "lite_results.msgpack"
    if lb_path.exists():
        import msgpack
        with open(lb_path, "rb") as f:
            lb_data = msgpack.unpack(f, raw=False)
        for agent_id, pf in lb_data.items():
            agent_results[agent_id] = {iid: bool(v) for iid, v in pf.items()}
        print(f"Loaded {len(agent_results)} agents from leaderboard msgpack")
    else:
        for agent_dir in [
            ROOT / "output" / "swebench_results_lite_agents",
            ROOT / "output" / "swebench_results",
        ]:
            if not agent_dir.exists():
                continue
            for p in sorted(agent_dir.glob("lite_*.json")):
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    agent_results[p.stem] = {
                        r["instance_id"]: bool(r.get("resolved", False)) for r in data
                    }

    n_agents = len(agent_results)
    print(f"\nFrontier analysis with {n_agents} agents:")
    frontier_instances_total = 0
    all_forms_stats = []
    for form in sorted(forms, key=lambda f: -f["pass_rate"]):
        members = set(form["instance_ids"])
        agent_solver_count = sum(
            1 for res in agent_results.values()
            if any(res.get(iid, False) for iid in members)
        )
        # Instance-level: how many instances are unsolved by ALL agents?
        unsolved_instances = [
            iid for iid in members
            if not any(res.get(iid, False) for res in agent_results.values())
        ]
        pct_unsolved = 100 * len(unsolved_instances) / max(len(members), 1)
        form["unsolved_instances"] = unsolved_instances
        form["pct_unsolved"] = pct_unsolved
        frontier_instances_total += len(unsolved_instances)
        status = f"{agent_solver_count}/{n_agents} agents  |  {len(unsolved_instances)}/{len(members)} instances unsolved ({pct_unsolved:.0f}%)"
        print(f"  [{form['label']:45s}] n={form['n']:3d}  pass={form['pass_rate']:.2f}  {status}")
        all_forms_stats.append(form)

    frontier_forms = [f for f in all_forms_stats if f["pct_unsolved"] >= 50]
    if frontier_forms:
        print(f"\nForms with >=50% instances unsolved by all {n_agents} agents:")
        for f in frontier_forms:
            print(f"  {f['label']}  (n={f['n']}, unsolved={len(f['unsolved_instances'])}/{f['n']}, mechanisms={f['dominant_mechanisms']})")

    # Global: instances unsolved by any leaderboard agent
    all_members = set(iid for form in forms for iid in form["instance_ids"])
    global_unsolved = [
        iid for iid in all_members
        if not any(res.get(iid, False) for res in agent_results.values())
    ]
    print(f"\nGlobal: {len(global_unsolved)}/{len(all_members)} instances unsolved by all {n_agents} agents")

    print("\nTree structure:")
    print(export_text(tree_final, feature_names=MECHANISM_LABELS, max_depth=best_depth))

    print("\nGenerating figures...")
    fig_depth_sweep(depths, variances, n_leaves_list, best_depth, OUTPUT_DIR)
    fig_form_pass_rates(forms, best_depth, OUTPUT_DIR)
    if agent_results:
        fig_frontier(forms, agent_results, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
