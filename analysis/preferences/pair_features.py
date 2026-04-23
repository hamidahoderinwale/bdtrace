"""Pair-level feature analysis on tied-outcome pairs (same task, both agents resolved).

For each (instance, agent_A, agent_B) tied-outcome pair, compute per-feature
differences. This is the data-shape the atlas will consume — task is held
constant by construction.

Outputs:
  output/paper2_pilot/pair_features.json
  output/paper2_pilot/pair_features.csv
  output/paper2_pilot/pair_feature_distributions.png

Usage:
  python -m analysis.preferences.pair_features
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
    "GPT-4 (SWE-agent)": "output/trajectories/lite_20240402_sweagent_gpt4.parquet",
    "Claude 3.5 Sonnet (SWE-agent)": "output/trajectories/lite_20240620_sweagent_claude3.5sonnet.parquet",
    "GPT-4o (SWE-agent)": "output/trajectories/lite_20240728_sweagent_gpt4o.parquet",
}
FEATURES = [
    "n_steps", "n_edits", "n_searches", "n_opens", "n_runs", "n_nav",
    "edit_retries", "edit_retry_rate", "n_files_opened", "n_files_edited",
]
COLORS = {
    "GPT-4 (SWE-agent)": "#0072B2",
    "Claude 3.5 Sonnet (SWE-agent)": "#009E73",
    "GPT-4o (SWE-agent)": "#E69F00",
}


def load_resolved_by_agent() -> dict[str, pd.DataFrame]:
    """Return per-agent dataframes filtered to resolved trajectories only."""
    out = {}
    for name, p in TRAJECTORY_PATHS.items():
        df = pd.read_parquet(PROJECT_ROOT / p)
        df = df[df["passed"]].copy().reset_index(drop=True)
        out[name] = df
    return out


def enumerate_tied_outcome_pairs(by_agent: dict[str, pd.DataFrame]) -> list[dict]:
    """Build one row per (instance, agent_A, agent_B) tied-outcome pair."""
    agent_instance_idx = {name: df.set_index("instance_id") for name, df in by_agent.items()}
    pairs = []
    for agent_a, agent_b in combinations(sorted(by_agent.keys()), 2):
        a_set = set(agent_instance_idx[agent_a].index)
        b_set = set(agent_instance_idx[agent_b].index)
        shared = sorted(a_set & b_set)
        for iid in shared:
            row_a = agent_instance_idx[agent_a].loc[iid]
            row_b = agent_instance_idx[agent_b].loc[iid]
            entry = {
                "instance_id": iid,
                "agent_a": agent_a,
                "agent_b": agent_b,
            }
            for feat in FEATURES:
                entry[f"{feat}_a"] = float(row_a[feat])
                entry[f"{feat}_b"] = float(row_b[feat])
                entry[f"{feat}_diff"] = float(row_a[feat]) - float(row_b[feat])
                entry[f"{feat}_absdiff"] = abs(entry[f"{feat}_diff"])
            entry["action_seq_a"] = str(row_a["action_sequence"])
            entry["action_seq_b"] = str(row_b["action_sequence"])
            pairs.append(entry)
    return pairs


def normalized_levenshtein(a: str, b: str) -> float:
    """Normalized Levenshtein on space-separated tokens."""
    ta = a.split()
    tb = b.split()
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


def compute_pair_dataframe(pairs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(pairs)
    df["action_levenshtein"] = [
        normalized_levenshtein(row["action_seq_a"], row["action_seq_b"])
        for _, row in df.iterrows()
    ]
    return df


def summarize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (a, b), sub in df.groupby(["agent_a", "agent_b"]):
        row = {"agent_a": a, "agent_b": b, "n_pairs": len(sub)}
        for feat in FEATURES:
            row[f"mean_absdiff_{feat}"] = float(sub[f"{feat}_absdiff"].mean())
            row[f"mean_diff_{feat}"] = float(sub[f"{feat}_diff"].mean())
        row["mean_action_levenshtein"] = float(sub["action_levenshtein"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_levenshtein_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    agent_pairs = df.groupby(["agent_a", "agent_b"])
    positions = []
    labels = []
    data = []
    for i, ((a, b), sub) in enumerate(agent_pairs):
        positions.append(i)
        labels.append(f"{a.split(' (')[0]}\nvs\n{b.split(' (')[0]}")
        data.append(sub["action_levenshtein"].values)
    parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#0072B2")
        pc.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Normalized Levenshtein distance on action sequences", fontsize=10)
    ax.set_title(
        "Pairwise action-sequence divergence (tied-outcome pairs only)\n"
        "Higher = agents took more different procedural paths on the same task",
        fontsize=10,
    )
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """For each feature, a boxplot of |diff| grouped by agent-pair."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5), sharey=False)
    axes = axes.flatten()
    agent_pairs = sorted(df[["agent_a", "agent_b"]].apply(tuple, axis=1).unique())
    pair_labels = [f"{a.split(' (')[0]} vs {b.split(' (')[0]}" for a, b in agent_pairs]
    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        data = [df[(df["agent_a"] == a) & (df["agent_b"] == b)][f"{feat}_absdiff"].values
                for a, b in agent_pairs]
        bp = ax.boxplot(data, labels=[""] * len(data), patch_artist=True, widths=0.5)
        for patch in bp["boxes"]:
            patch.set_facecolor("#0072B2")
            patch.set_alpha(0.6)
        ax.set_title(feat, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xticklabels([lbl for lbl in pair_labels], rotation=35, ha="right", fontsize=7)
    fig.suptitle(
        "Per-feature |difference| between agents on tied-outcome pairs\n"
        "(same task, both resolved — what varies within a matched pair)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading resolved trajectories...")
    by_agent = load_resolved_by_agent()
    for name, df in by_agent.items():
        print(f"  {name}: {len(df)} resolved")

    print("\nEnumerating tied-outcome pairs...")
    pairs = enumerate_tied_outcome_pairs(by_agent)
    pair_df = compute_pair_dataframe(pairs)
    print(f"  total pairs: {len(pair_df)}")

    summary = summarize_pairs(pair_df)
    print("\nSummary (mean |diff| per agent-pair × feature):")
    print(
        summary[
            ["agent_a", "agent_b", "n_pairs", "mean_absdiff_n_steps",
             "mean_absdiff_n_edits", "mean_action_levenshtein"]
        ].round(2).to_string(index=False)
    )

    # Save the pair table — this is what the atlas pipeline will consume
    pair_table_path = OUT_DIR / "tied_outcome_pairs.csv"
    keep_cols = [
        "instance_id", "agent_a", "agent_b", "action_levenshtein",
        *[f"{f}_diff" for f in FEATURES],
        *[f"{f}_absdiff" for f in FEATURES],
        "action_seq_a", "action_seq_b",
    ]
    # Keep a minimal JSON-serializable version too
    pair_df[keep_cols].to_csv(pair_table_path, index=False)

    json_path = OUT_DIR / "pair_features.json"
    json_path.write_text(
        json.dumps(
            {
                "n_pairs_total": int(len(pair_df)),
                "pair_counts_per_agent_pair": summary[["agent_a", "agent_b", "n_pairs"]].to_dict(orient="records"),
                "feature_summary": summary.round(4).to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )

    # Plots
    plot_levenshtein_distribution(pair_df, OUT_DIR / "pair_action_levenshtein.png")
    plot_feature_distributions(pair_df, OUT_DIR / "pair_feature_distributions.png")

    print(f"\nSaved:")
    print(f"  {pair_table_path}")
    print(f"  {json_path}")
    print(f"  {OUT_DIR / 'pair_action_levenshtein.png'}")
    print(f"  {OUT_DIR / 'pair_feature_distributions.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
