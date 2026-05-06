"""Stage allocation: fraction of steps per behavioral stage, by agent.

Maps raw action counts (n_searches, n_opens, n_nav, n_edits, n_runs)
to five stages. Shows mean fraction per stage per agent as a stacked
horizontal bar. Passing and failing trajectories shown separately.

Stage mapping:
    Explore = SEARCH steps
    Browse  = OPEN + NAV steps
    Edit    = EDIT + CREATE steps
    Test    = RUN steps
    Other   = remaining steps (shell, submit, unknown)

Reads:  output/trajectories/lite_all_models.parquet
Writes: output/figures/fig_stage_allocation.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, STAGE_COLORS, STAGE_ORDER, AGENT_ORDER, AGENT_SHORT
register()

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOME_ORDER = ["pass", "fail"]


def main() -> None:
    df = pd.read_parquet(ROOT / "output/trajectories/lite_all_models.parquet")
    df["agent"] = df["model_id"].map(AGENT_SHORT)
    df = df[df["agent"].notna()].copy()
    df["passed_label"] = df["passed"].map({True: "pass", False: "fail"})

    # Stage fractions per trajectory
    df["f_explore"] = df["n_searches"] / df["n_steps"].clip(lower=1)
    df["f_browse"]  = (df["n_opens"] + df["n_nav"]) / df["n_steps"].clip(lower=1)
    df["f_edit"]    = df["n_edits"] / df["n_steps"].clip(lower=1)
    df["f_test"]    = df["n_runs"] / df["n_steps"].clip(lower=1)
    df["f_other"]   = (1 - df["f_explore"] - df["f_browse"] - df["f_edit"] - df["f_test"]).clip(lower=0)

    stage_cols = {
        "Explore": "f_explore",
        "Browse":  "f_browse",
        "Edit":    "f_edit",
        "Test":    "f_test",
        "Other":   "f_other",
    }

    rows = []
    for agent in AGENT_ORDER:
        for outcome in OUTCOME_ORDER:
            sub = df[(df["agent"] == agent) & (df["passed_label"] == outcome)]
            if len(sub) == 0:
                continue
            for stage, col in stage_cols.items():
                rows.append({
                    "agent":   agent,
                    "outcome": outcome,
                    "stage":   stage,
                    "frac":    float(sub[col].mean()),
                    "n":       len(sub),
                })

    plot_df = pd.DataFrame(rows)

    # Sort stages in display order (Other last)
    stage_display_order = ["Explore", "Browse", "Edit", "Test", "Other"]
    color_domain = stage_display_order
    color_range = [
        STAGE_COLORS.get("Explore", "#56B4E9"),
        STAGE_COLORS.get("Browse",  "#009E73"),
        STAGE_COLORS.get("Edit",    "#E69F00"),
        STAGE_COLORS.get("Test",    "#D55E00"),
        "#CCCCCC",  # Other
    ]
    # Explicit sort index for stacking order
    stage_sort_idx = {s: i for i, s in enumerate(stage_display_order)}
    plot_df["stage_order"] = plot_df["stage"].map(stage_sort_idx)

    # y-axis: agent x outcome combos, ordered by agent then outcome
    y_order = [
        f"{a} ({o})"
        for a in AGENT_ORDER
        for o in OUTCOME_ORDER
        if len(plot_df[(plot_df["agent"] == a) & (plot_df["outcome"] == o)]) > 0
    ]
    plot_df["group"] = plot_df["agent"] + " (" + plot_df["outcome"] + ")"

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("group:N", sort=y_order,
                    axis=alt.Axis(title=None, labelFontSize=11)),
            x=alt.X("frac:Q",
                    title="Mean fraction of steps",
                    stack="normalize",
                    axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0])),
            color=alt.Color("stage:N",
                            sort=stage_display_order,
                            scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title=None, orient="bottom",
                                              columns=len(stage_display_order))),
            order=alt.Order("stage_order:Q", sort="ascending"),
        )
        .properties(
            width=360, height=260,
            title=alt.TitleParams(
                "Stage allocation by agent and outcome",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = OUT / "fig_stage_allocation.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")

    # Print summary stats
    print("\nMean stage fractions (pass trajectories):")
    for agent in AGENT_ORDER:
        sub = plot_df[(plot_df["agent"] == agent) & (plot_df["outcome"] == "pass")]
        if len(sub) == 0:
            continue
        fracs = {r["stage"]: r["frac"] for _, r in sub.iterrows()}
        n = sub["n"].iloc[0]
        print(f"  {agent:12s} (n={n:3d}): "
              + "  ".join(f"{s}={fracs.get(s, 0):.2f}" for s in stage_display_order))

    print("\nMean stage fractions (fail trajectories):")
    for agent in AGENT_ORDER:
        sub = plot_df[(plot_df["agent"] == agent) & (plot_df["outcome"] == "fail")]
        if len(sub) == 0:
            continue
        fracs = {r["stage"]: r["frac"] for _, r in sub.iterrows()}
        n = sub["n"].iloc[0]
        print(f"  {agent:12s} (n={n:3d}): "
              + "  ".join(f"{s}={fracs.get(s, 0):.2f}" for s in stage_display_order))


if __name__ == "__main__":
    main()
