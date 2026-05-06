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

import sys
import altair as alt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.theme import register, BLUE, COPPER, GREEN, MAGENTA
register()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "paper2_pilot"
FIX_TYPES_PATH = PROJECT_ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "fix_types.json"

TRAJECTORY_PATHS = {
    "Claude-3":   "output/trajectories/lite_20240402_sweagent_claude3opus.parquet",
    "GPT-4":      "output/trajectories/lite_20240402_sweagent_gpt4.parquet",
    "Claude-3.5": "output/trajectories/lite_20240620_sweagent_claude3.5sonnet.parquet",
    "GPT-4o":     "output/trajectories/lite_20240728_sweagent_gpt4o.parquet",
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

FEATURE_LABELS = {
    "n_steps":        "Total steps",
    "n_edits":        "File edits",
    "n_searches":     "Searches",
    "n_opens":        "File opens",
    "n_runs":         "Script runs",
    "n_nav":          "Navigation",
    "edit_retries":   "Edit retries",
    "edit_retry_rate":"Edit retry rate",
    "n_files_opened": "Files opened",
    "n_files_edited": "Files edited",
}

GROUPING_COLORS = {
    "instance_id": BLUE,
    "fix_type":    COPPER,
    "repo":        GREEN,
}
GROUPING_LABEL = {
    "instance_id": "Task identity",
    "fix_type":    "Fix type",
    "repo":        "Repository",
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
    n_feat = len(report)

    rows = []
    for _, row in report.iterrows():
        for g in COMPARISON_GROUPINGS:
            rows.append({
                "feature":        FEATURE_LABELS.get(row["feature"], row["feature"]),
                "feature_raw":    row["feature"],
                "grouping_label": GROUPING_LABEL[g],
                "ratio":          row[f"ratio_{g}_over_agent"],
            })
    df_plot = pd.DataFrame(rows)

    # Sort features by mean ratio descending (largest task-dominance at top)
    feat_order = (
        df_plot[df_plot["grouping_label"] == GROUPING_LABEL["instance_id"]]
        .sort_values("ratio", ascending=True)["feature"]
        .tolist()
    )

    color_scale = alt.Scale(
        domain=[GROUPING_LABEL[g] for g in COMPARISON_GROUPINGS],
        range=[GROUPING_COLORS[g] for g in COMPARISON_GROUPINGS],
    )

    bars = (
        alt.Chart(df_plot)
        .mark_bar(height=10)
        .encode(
            y=alt.Y(
                "feature:N",
                sort=feat_order,
                axis=alt.Axis(title=None, ticks=False, domain=False, labelFontSize=10),
            ),
            x=alt.X(
                "ratio:Q",
                title="SD of group means / SD of agent means",
                axis=alt.Axis(ticks=False, domain=False, labelFontSize=10),
            ),
            yOffset=alt.YOffset(
                "grouping_label:N",
                sort=[GROUPING_LABEL[g] for g in COMPARISON_GROUPINGS],
                scale=alt.Scale(range=[-8, 8]),
            ),
            color=alt.Color(
                "grouping_label:N",
                scale=color_scale,
                legend=alt.Legend(orient="bottom", title=None),
            ),
        )
    )

    chart = (
        bars
        .properties(
            width=380,
            height=max(260, n_feat * 34),
            title=alt.TitleParams(
                text=f"Behavioral feature variance by grouping (ratio to agent){title_suffix}",
                fontSize=13,
                color="#111111",
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    chart.save(str(out_path), scale_factor=2)


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
