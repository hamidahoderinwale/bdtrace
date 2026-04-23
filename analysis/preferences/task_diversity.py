"""Task-level procedural diversity: for each task, how much do agents diverge?

For each task (instance) where >=2 agents have trajectories, measure:
  - range of n_steps across agents (permissive = high range)
  - coefficient of variation on n_steps (normalized diversity)
  - mean pairwise action-sequence Levenshtein across agent-pairs on that task

Output:
  - output/paper2_pilot/task_diversity.json
  - output/paper2_pilot/task_diversity.csv
  - output/paper2_pilot/task_diversity_distribution.png
  - output/paper2_pilot/task_diversity_by_resolved.png

Addresses pluralism-vs-convergence (Tier 1 phenomenon 4): tasks vary in how
much procedural diversity they permit across agents.

Usage:
    python -m analysis.preferences.task_diversity
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "paper2_pilot"

TRAJECTORY_PATHS = {
    "GPT-4": "output/trajectories/lite_20240402_sweagent_gpt4.parquet",
    "Claude-3.5": "output/trajectories/lite_20240620_sweagent_claude3.5sonnet.parquet",
    "GPT-4o": "output/trajectories/lite_20240728_sweagent_gpt4o.parquet",
}


def load_all() -> pd.DataFrame:
    frames = []
    for name, p in TRAJECTORY_PATHS.items():
        df = pd.read_parquet(PROJECT_ROOT / p).copy()
        df["agent"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def normalized_levenshtein(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    n, m = len(ta), len(tb)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if ta[i - 1] == tb[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m] / max(n, m, 1)


def per_task_diversity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iid, sub in df.groupby("instance_id"):
        if len(sub) < 2:
            continue
        steps = sub["n_steps"].astype(float).values
        edits = sub["n_edits"].astype(float).values
        action_seqs = sub["action_sequence"].tolist()
        passed_flags = sub["passed"].astype(bool).tolist()

        pair_lev = [
            normalized_levenshtein(a, b)
            for a, b in combinations(action_seqs, 2)
        ]
        rows.append({
            "instance_id": iid,
            "n_agents": int(len(sub)),
            "n_resolved": int(sum(passed_flags)),
            "all_resolved": bool(all(passed_flags)),
            "n_steps_mean": float(np.mean(steps)),
            "n_steps_range": float(np.max(steps) - np.min(steps)),
            "n_steps_cv": float(np.std(steps) / np.mean(steps)) if np.mean(steps) > 0 else 0.0,
            "n_edits_range": float(np.max(edits) - np.min(edits)),
            "mean_pairwise_levenshtein": float(np.mean(pair_lev)) if pair_lev else 0.0,
            "max_pairwise_levenshtein": float(np.max(pair_lev)) if pair_lev else 0.0,
        })
    return pd.DataFrame(rows)


def plot_diversity_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(df["mean_pairwise_levenshtein"], bins=24, color="#0072B2", edgecolor="white")
    ax.axvline(df["mean_pairwise_levenshtein"].median(), color="#E69F00", lw=1.5, ls="--",
               label=f"median = {df['mean_pairwise_levenshtein'].median():.2f}")
    ax.set_xlabel("Mean pairwise Levenshtein across agents on this task", fontsize=9)
    ax.set_ylabel("Number of tasks", fontsize=9)
    ax.set_title("How much do agents diverge per task?\n(procedural permissiveness)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.hist(df["n_steps_cv"].clip(upper=2.0), bins=24, color="#009E73", edgecolor="white")
    ax.axvline(df["n_steps_cv"].median(), color="#E69F00", lw=1.5, ls="--",
               label=f"median = {df['n_steps_cv'].median():.2f}")
    ax.set_xlabel("Coefficient of variation on n_steps (across agents)", fontsize=9)
    ax.set_ylabel("Number of tasks", fontsize=9)
    ax.set_title("Step-count diversity per task\n(higher = more agent disagreement on effort)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_by_outcome(df: pd.DataFrame, out_path: Path) -> None:
    """Split tasks into all-resolved vs mixed-outcome vs none-resolved.

    Is diversity higher on permissive tasks (multiple paths to success) or
    constrained tasks (one right approach)?
    """
    categories = {
        "All resolved": df[df["all_resolved"]],
        "Mixed outcome": df[(~df["all_resolved"]) & (df["n_resolved"] > 0)],
        "None resolved": df[df["n_resolved"] == 0],
    }
    fig, ax = plt.subplots(figsize=(7.5, 4))
    positions = []
    data = []
    labels = []
    colors = ["#009E73", "#0072B2", "#E69F00"]
    for i, (name, sub) in enumerate(categories.items()):
        if len(sub) == 0:
            continue
        positions.append(i)
        data.append(sub["mean_pairwise_levenshtein"].values)
        labels.append(f"{name}\n(n={len(sub)})")
    parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=True)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean pairwise Levenshtein (procedural divergence)", fontsize=9)
    ax.set_title("Procedural divergence by task outcome category\n"
                 "Does success-multiplicity correlate with diverse paths?", fontsize=10)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading trajectories...")
    df = load_all()
    print(f"  {len(df)} trajectories across {df['instance_id'].nunique()} tasks × {df['agent'].nunique()} agents")

    print("\nComputing per-task diversity (this takes ~30s for Levenshtein on all pairs)...")
    diversity = per_task_diversity(df)
    print(f"  tasks with >=2 agents: {len(diversity)}")

    # Headline stats
    permissive_thresh = 0.6
    constrained_thresh = 0.3
    n_permissive = int((diversity["mean_pairwise_levenshtein"] > permissive_thresh).sum())
    n_constrained = int((diversity["mean_pairwise_levenshtein"] < constrained_thresh).sum())
    n_middle = len(diversity) - n_permissive - n_constrained

    print(f"\nTask categories by pairwise Levenshtein:")
    print(f"  permissive (>{permissive_thresh}): {n_permissive} / {len(diversity)} "
          f"({100*n_permissive/len(diversity):.1f}%)")
    print(f"  constrained (<{constrained_thresh}): {n_constrained} / {len(diversity)} "
          f"({100*n_constrained/len(diversity):.1f}%)")
    print(f"  middle: {n_middle} ({100*n_middle/len(diversity):.1f}%)")

    print(f"\nMean pairwise Levenshtein: {diversity['mean_pairwise_levenshtein'].mean():.3f} "
          f"(median {diversity['mean_pairwise_levenshtein'].median():.3f})")

    # Split by outcome
    by_outcome = (
        diversity.groupby("all_resolved")["mean_pairwise_levenshtein"]
        .agg(["count", "mean", "median"])
        .round(3)
    )
    print(f"\nDivergence by outcome:")
    print(by_outcome.to_string())

    # Save
    diversity.round(4).to_csv(OUT_DIR / "task_diversity.csv", index=False)
    json_path = OUT_DIR / "task_diversity.json"
    json_path.write_text(json.dumps({
        "n_tasks": int(len(diversity)),
        "n_permissive": n_permissive,
        "n_constrained": n_constrained,
        "n_middle": n_middle,
        "permissive_threshold": permissive_thresh,
        "constrained_threshold": constrained_thresh,
        "mean_pairwise_levenshtein": float(diversity["mean_pairwise_levenshtein"].mean()),
        "median_pairwise_levenshtein": float(diversity["mean_pairwise_levenshtein"].median()),
        "by_outcome": by_outcome.to_dict(),
    }, indent=2, default=str))

    plot_diversity_distribution(diversity, OUT_DIR / "task_diversity_distribution.png")
    plot_diversity_by_outcome(diversity, OUT_DIR / "task_diversity_by_resolved.png")

    print(f"\nSaved:\n  {OUT_DIR / 'task_diversity.csv'}\n  {json_path}\n"
          f"  {OUT_DIR / 'task_diversity_distribution.png'}\n"
          f"  {OUT_DIR / 'task_diversity_by_resolved.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
