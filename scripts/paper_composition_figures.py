"""Paper figures for the compositional generalization finding.

Loads pre-computed JSON from output/compositional_generalization/ and
produces two paper-ready figures:

  figures/gap/b_failure_classification.png
      Bar chart: novel primitive / novel composition / familiar, mean across 84 agents.
      Key number (43.8%) annotated.

  figures/gap/c_composition_scatter.png
      Scatter: min primitive frequency vs ease.
      Top-left quadrant (common parts, hard to combine) highlighted.

Usage:
    uv run python scripts/paper_composition_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, GRAY
register()

OUT     = ROOT / "figures" / "gap"
OUT.mkdir(parents=True, exist_ok=True)
DATA    = ROOT / "output" / "compositional_generalization"

_W = 480
_H = 220


def fig_b():
    stats = json.loads((DATA / "summary_stats.json").read_text())
    mean_fracs = stats["per_agent_mean_fractions"]

    cat_order  = ["Novel primitive", "Novel composition", "Familiar"]
    cat_keys   = ["novel_primitive",  "novel_composition",  "familiar"]
    colors     = [BLUE, ORANGE, GREEN]

    rows = [
        {"category": label, "fraction": mean_fracs[key]}
        for label, key in zip(cat_order, cat_keys)
    ]
    df = pd.DataFrame(rows)

    cscale = alt.Scale(domain=cat_order, range=colors)

    bars = (
        alt.Chart(df)
        .mark_bar(width=72)
        .encode(
            x=alt.X("category:N", sort=cat_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=11)),
            y=alt.Y("fraction:Q",
                    scale=alt.Scale(domain=[0, 1.0]),
                    axis=alt.Axis(title="Mean fraction of agent failures",
                                  domain=False, ticks=False,
                                  format=".0%",
                                  values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
            color=alt.Color("category:N", scale=cscale, legend=None),
        )
    )

    # annotate only the composition bar
    comp_row = df[df["category"] == "Novel composition"]
    ann = (
        alt.Chart(comp_row)
        .mark_text(dy=-10, fontSize=12, color="#333333")
        .encode(
            x=alt.X("category:N", sort=cat_order),
            y=alt.Y("fraction:Q"),
            text=alt.Text("fraction:Q", format=".1%"),
        )
    )

    chart = (
        (bars + ann)
        .properties(
            title=alt.TitleParams(
                text="Composition failure types",
                fontSize=14, color="#111111", anchor="start",
            ),
            width=340, height=280,
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT / "b_failure_classification.png"), scale_factor=2)
    print("  B done")


def fig_c():
    gap_data = json.loads((DATA / "composition_gap.json").read_text())
    df = pd.DataFrame(gap_data)

    base_pts = (
        alt.Chart(df)
        .mark_circle(size=28, opacity=0.45)
        .encode(
            x=alt.X("min_primitive_freq:Q",
                    title="Rarest primitive frequency across agents",
                    scale=alt.Scale(domain=[0, 1.05]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.25, 0.5, 0.75, 1.0])),
            y=alt.Y("ease:Q",
                    title="Ease (fraction of agents solving)",
                    scale=alt.Scale(domain=[-0.02, 1.02]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.25, 0.5, 0.75, 1.0])),
            color=alt.value(BLUE),
        )
    )

    # highlight top-left quadrant: common primitives, hard tasks
    hard_comp = df[(df["min_primitive_freq"] > 0.5) & (df["ease"] < 0.1)]
    highlight = (
        alt.Chart(hard_comp)
        .mark_circle(size=55, opacity=0.85)
        .encode(
            x="min_primitive_freq:Q",
            y="ease:Q",
            color=alt.value(ORANGE),
        )
    )

    # annotation: n instances in quadrant
    n = len(hard_comp)
    ann_df = pd.DataFrame([{"x": 0.75, "y": 0.055,
                            "t": f"{n} instances: familiar parts, hard to combine"}])
    ann = (
        alt.Chart(ann_df)
        .mark_text(align="left", fontSize=10, color=ORANGE, fontStyle="italic")
        .encode(x="x:Q", y="y:Q", text="t:N")
    )

    chart = (
        (base_pts + highlight + ann)
        .properties(
            title=alt.TitleParams(
                text="Novel composition vs primitive novelty",
                fontSize=14, color="#111111", anchor="start",
            ),
            width=_W, height=_H + 60,
        )
        .configure_view(strokeWidth=0)
        .configure_axis(domain=False, ticks=False)
    )
    chart.save(str(OUT / "c_composition_scatter.png"), scale_factor=2)
    print("  C done")


if __name__ == "__main__":
    print("Generating composition figures...")
    fig_b()
    fig_c()
    print(f"\nAll saved to {OUT}/")
