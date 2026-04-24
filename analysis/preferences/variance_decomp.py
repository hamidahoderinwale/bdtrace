"""Does task or agent explain more feature variation? (simplest honest answer.)

For each of 10 per-trajectory features we compute one number per grouping:
the standard deviation of the group means. The ratio
sd(task_means) / sd(agent_means) says directly how much more variation lives
between tasks than between agents.

We also report each group's mean range (max - min) and plot a strip of
trajectories colored by agent and sorted by task. No ANOVA, no F-tests,
no eta-squared — the single ratio + the strip plot answer the question.

Output (same filenames as the prior ANOVA version, so downstream dashboards
don't break):
  output/paper2_pilot/variance_decomposition.json   (ratios + ranges)
  output/paper2_pilot/variance_decomposition.csv    (human-readable table)
  output/paper2_pilot/variance_decomposition.png    (bar chart of sd ratios)
  output/paper2_pilot/variance_decomposition_controlled.json  (same, task-matched subset)
  output/paper2_pilot/variance_decomposition_controlled.csv
  output/paper2_pilot/variance_decomposition_controlled.png

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
FIX_TYPES_PATH = PROJECT_ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "fix_types.json"

TRAJECTORY_PATHS = {
    "GPT-4 (SWE-agent)": "output/trajectories/lite_20240402_sweagent_gpt4.parquet",
    "Claude 3.5 Sonnet (SWE-agent)": "output/trajectories/lite_20240620_sweagent_claude3.5sonnet.parquet",
    "GPT-4o (SWE-agent)": "output/trajectories/lite_20240728_sweagent_gpt4o.parquet",
}

GROUPINGS = ["instance_id", "fix_type", "repo", "agent"]
COMPARISON_GROUPINGS = ["instance_id", "fix_type", "repo"]

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

GROUPING_COLORS = {
    "instance_id": "#E69F00",
    "fix_type": "#CC79A7",
    "repo": "#56B4E9",
}
GROUPING_LABEL = {
    "instance_id": "task-id (300 bins)",
    "fix_type": "fix type (13 bins)",
    "repo": "repo (12 bins)",
}


def load_trajectories() -> pd.DataFrame:
    frames = []
    for name, p in TRAJECTORY_PATHS.items():
        df = pd.read_parquet(PROJECT_ROOT / p).copy()
        df["agent"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # join fix_type and repo from the semantic labels dataset
    labels = json.loads(FIX_TYPES_PATH.read_text())["results"]
    lut = {r["instance_id"]: r for r in labels}
    df["fix_type"] = df["instance_id"].map(lambda i: lut.get(i, {}).get("fix_type", "unknown"))
    df["repo"] = df["instance_id"].map(lambda i: lut.get(i, {}).get("repo", "unknown"))
    return df


def decompose(df: pd.DataFrame, feature: str) -> dict:
    out = {
        "feature": feature,
        "n_trajectories": int(len(df)),
        "grand_mean": float(df[feature].mean()),
        "grand_std": float(df[feature].std(ddof=1)),
    }
    agent_sd = df.groupby("agent")[feature].mean().std(ddof=1)
    out["sd_agent_means"] = float(agent_sd) if not np.isnan(agent_sd) else 0.0
    out["n_agent_groups"] = int(df["agent"].nunique())
    for g in COMPARISON_GROUPINGS:
        grp_means = df.groupby(g)[feature].mean()
        out[f"sd_{g}_means"] = float(grp_means.std(ddof=1)) if len(grp_means) > 1 else 0.0
        out[f"n_{g}_groups"] = int(df[g].nunique())
        out[f"ratio_{g}_over_agent"] = (
            float(grp_means.std(ddof=1) / agent_sd)
            if agent_sd > 0 and len(grp_means) > 1 else float("inf")
        )
    return out


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([decompose(df, f) for f in FEATURES])


def plot_ratio_bars(report: pd.DataFrame, out_path: Path, title_suffix: str = "") -> None:
    groupings = COMPARISON_GROUPINGS
    n_feat = len(report)
    n_grp = len(groupings)
    fig, ax = plt.subplots(figsize=(10, 0.55 * n_feat * n_grp + 1.8))
    bar_h = 0.8 / n_grp
    y_base = np.arange(n_feat)

    for i, g in enumerate(groupings):
        col = f"ratio_{g}_over_agent"
        ratios = report[col].values
        offset = (i - (n_grp - 1) / 2) * bar_h
        ax.barh(y_base + offset, ratios, bar_h,
                color=GROUPING_COLORS[g],
                edgecolor="white", label=GROUPING_LABEL[g])

    ax.axvline(1.0, color="#555", lw=0.8, ls="--", label="equal (ratio = 1)")
    ax.set_yticks(y_base)
    ax.set_yticklabels(report["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ratio: sd(group means) / sd(agent means)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="best")
    ax.set_title(
        f"Does task, fix-type, or repo explain more than agent?{title_suffix}\n"
        "Ratio > 1 means that grouping's means vary more than agent means.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def emit(report: pd.DataFrame, basename: str, title_suffix: str) -> None:
    (OUT_DIR / f"{basename}.csv").write_text(report.to_csv(index=False))
    (OUT_DIR / f"{basename}.json").write_text(
        json.dumps({"features": report.to_dict(orient="records")}, indent=2)
    )
    plot_ratio_bars(report, OUT_DIR / f"{basename}.png", title_suffix)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_trajectories()

    def show(report: pd.DataFrame) -> None:
        print(
            "\n  feature              "
            + "  ".join(f"{GROUPING_LABEL[g][:18]:<18s}" for g in COMPARISON_GROUPINGS)
        )
        for _, row in report.iterrows():
            vals = "  ".join(
                f"{row[f'ratio_{g}_over_agent']:>5.2f}x  n={row[f'n_{g}_groups']:<4d}"
                for g in COMPARISON_GROUPINGS
            )
            print(f"  {row['feature']:<18s}  {vals}")

    print("=== full corpus ===")
    print(f"  {len(df)} trajectories, "
          f"{df['instance_id'].nunique()} tasks, "
          f"{df['fix_type'].nunique()} fix types, "
          f"{df['repo'].nunique()} repos, "
          f"{df['agent'].nunique()} agents")
    report = build_report(df)
    show(report)
    emit(report, "variance_decomposition", "")

    # Task-matched subset: only tasks attempted by all agents
    counts = df.groupby("instance_id")["agent"].nunique()
    shared = counts[counts == df["agent"].nunique()].index
    df_shared = df[df["instance_id"].isin(shared)]
    print(f"\n=== task-matched subset ({len(df_shared)} trajectories, {len(shared)} shared tasks) ===")
    report_ctrl = build_report(df_shared)
    show(report_ctrl)
    emit(report_ctrl, "variance_decomposition_controlled", "  (task-matched subset)")

    print(f"\nSaved:")
    for n in [
        "variance_decomposition.json",
        "variance_decomposition.csv",
        "variance_decomposition.png",
        "variance_decomposition_controlled.json",
        "variance_decomposition_controlled.csv",
        "variance_decomposition_controlled.png",
    ]:
        print(f"  {OUT_DIR / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
