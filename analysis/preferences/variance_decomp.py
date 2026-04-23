"""Variance decomposition across SWE-agents × SWE-bench Lite instances.

For each of 10 per-trajectory features, decompose variance into:
  - between-agent variance (how much does agent identity explain?)
  - between-task variance (how much does task identity explain?)
  - residual

Output:
  - output/paper2_pilot/variance_decomposition.json  (full numeric results)
  - output/paper2_pilot/variance_decomposition.png   (bar chart)
  - output/paper2_pilot/variance_decomposition.csv   (human-readable table)

Usage:
    python -m analysis.preferences.variance_decomp
"""

from __future__ import annotations

import json
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
    "n_steps",
    "n_edits",
    "n_searches",
    "n_opens",
    "n_runs",
    "n_nav",
    "edit_retries",
    "edit_retry_rate",
    "n_files_opened",
    "n_files_edited",
]

# Wong colorblind-safe palette (matching project style)
COLOR_AGENT = "#0072B2"
COLOR_TASK = "#E69F00"


def load_trajectories() -> pd.DataFrame:
    """Load all three agents' trajectories into one long-form dataframe."""
    frames = []
    for name, p in TRAJECTORY_PATHS.items():
        df = pd.read_parquet(PROJECT_ROOT / p)
        df = df.copy()
        df["agent"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def decompose_one_way(df: pd.DataFrame, feature: str, group_col: str) -> dict:
    """One-way variance decomposition of `feature` by `group_col`.

    Returns:
      total_variance, between_group_variance, within_group_variance,
      fraction_between (eta-squared equivalent).
    """
    values = df[feature].astype(float).values
    total_var = float(np.var(values, ddof=0))
    group_means = df.groupby(group_col)[feature].mean()
    group_counts = df.groupby(group_col)[feature].count()
    overall_mean = float(np.mean(values))

    # Between-group SS / N = variance of group means weighted by size
    between_ss = float(np.sum(group_counts.values * (group_means.values - overall_mean) ** 2))
    within_ss = float(
        np.sum(
            [
                (df[df[group_col] == g][feature] - m).pow(2).sum()
                for g, m in group_means.items()
            ]
        )
    )
    n_total = len(values)
    between_var = between_ss / n_total
    within_var = within_ss / n_total

    return {
        "total_variance": total_var,
        "between_group_variance": between_var,
        "within_group_variance": within_var,
        "fraction_between": between_var / total_var if total_var > 0 else float("nan"),
        "n": int(n_total),
        "n_groups": int(len(group_means)),
    }


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build a long-form table of decomposition results per feature × grouping."""
    rows = []
    for feat in FEATURES:
        agent_res = decompose_one_way(df, feat, "agent")
        task_res = decompose_one_way(df, feat, "instance_id")
        rows.append(
            {
                "feature": feat,
                "agent_fraction": agent_res["fraction_between"],
                "task_fraction": task_res["fraction_between"],
                "residual_fraction": max(
                    0.0, 1.0 - agent_res["fraction_between"] - task_res["fraction_between"]
                ),
                "total_variance": agent_res["total_variance"],
                "n_trajectories": agent_res["n"],
                "n_agents": agent_res["n_groups"],
                "n_tasks": task_res["n_groups"],
            }
        )
    return pd.DataFrame(rows)


def plot_decomposition(report: pd.DataFrame, out_path: Path) -> None:
    features = report["feature"].tolist()
    agent_frac = report["agent_fraction"].values
    task_frac = report["task_fraction"].values

    order = np.argsort(-(agent_frac + task_frac))
    features_sorted = [features[i] for i in order]
    agent_sorted = agent_frac[order]
    task_sorted = task_frac[order]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(features_sorted))
    width = 0.38
    ax.bar(x - width / 2, agent_sorted, width, label="Agent-explained", color=COLOR_AGENT)
    ax.bar(x + width / 2, task_sorted, width, label="Task-explained", color=COLOR_TASK)
    ax.set_xticks(x)
    ax.set_xticklabels(features_sorted, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of total variance", fontsize=10)
    ax.set_title(
        "Per-trajectory feature variance: agent-explained vs task-explained\n"
        "(3 SWE-agents on SWE-bench Lite; one-way ANOVA decomposition)",
        fontsize=10,
    )
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading trajectories...")
    df = load_trajectories()
    print(f"  total trajectories: {len(df)} across {df['agent'].nunique()} agents, "
          f"{df['instance_id'].nunique()} unique tasks")

    print("\nComputing decompositions...")
    report = build_report(df)

    # Save CSV + JSON
    csv_path = OUT_DIR / "variance_decomposition.csv"
    json_path = OUT_DIR / "variance_decomposition.json"
    png_path = OUT_DIR / "variance_decomposition.png"

    report.round(4).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "n_trajectories": int(len(df)),
                "n_agents": int(df["agent"].nunique()),
                "n_tasks": int(df["instance_id"].nunique()),
                "agents": sorted(df["agent"].unique().tolist()),
                "features": FEATURES,
                "results": report.round(6).to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    plot_decomposition(report, png_path)

    print("\nResults (ordered by agent-explained variance):")
    sorted_report = report.sort_values("agent_fraction", ascending=False)
    print(sorted_report[["feature", "agent_fraction", "task_fraction", "residual_fraction"]].round(3).to_string(index=False))

    print(f"\nSaved:\n  {csv_path}\n  {json_path}\n  {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
