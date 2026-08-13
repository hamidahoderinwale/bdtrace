"""Five standalone figures for the five core findings.

Design principles:
- No redundant legends — direct labels or color alone carries the group
- Only annotate the one key number that is the takeaway
- Titles are findings, not descriptions
- Tufte data-ink: no grid on the categorical axis, very light grid on quantitative
- Zoomed extents — don't waste space outside the data range

Usage:
    uv run python scripts/paper_gap_figures.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.theme import (
    register,
    BLUE, ORANGE, GREEN, VERMILLION, GRAY, NEAR_BLACK,
)

register()

OUT = ROOT / "figures" / "gap"
OUT.mkdir(parents=True, exist_ok=True)

_W = 480   # default width
_H = 220   # default height


# ── A: FIM separates difficulty ──────────────────────────────────────────────
# Lollipop. Structural (blue) vs semantic baselines (gray).
# Key number: 4.6× — annotated at the gap between the two groups.

def fig_a():
    rows = [
        {"label": "FIM patterns",         "value": 0.0333, "group": "Structural"},
        {"label": "AST decision tree",     "value": 0.0257, "group": "Structural"},
        {"label": "Fix from traces",       "value": 0.0083, "group": "Semantic"},
        {"label": "Predicted fix",         "value": 0.0087, "group": "Semantic"},
        {"label": "Issue text",            "value": 0.0073, "group": "Semantic"},
    ]
    df = pd.DataFrame(rows)
    df["baseline"] = 0.0
    # Altair ordinal Y: sort[0] renders at top, sort[-1] at bottom.
    # Structural methods (FIM, AST) at top; semantic baselines below.
    label_order = [r["label"] for r in rows]

    cscale = alt.Scale(domain=["Structural", "Semantic"], range=[BLUE, ORANGE])

    rule = (
        alt.Chart(df)
        .mark_rule(strokeWidth=2.5, opacity=0.5)
        .encode(
            y=alt.Y("label:O", sort=label_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=11)),
            x=alt.X("baseline:Q",
                    scale=alt.Scale(domain=[0, 0.036]),
                    axis=alt.Axis(title="Difficulty variance",
                                  domain=False, ticks=False,
                                  values=[0, 0.01, 0.02, 0.03],
                                  labelFontSize=10)),
            x2="value:Q",
            color=alt.Color("group:N", scale=cscale,
                            legend=alt.Legend(orient="bottom-right", title=None)),
        )
    )

    pts = (
        alt.Chart(df)
        .mark_point(size=120, filled=True, strokeWidth=1.5)
        .encode(
            y=alt.Y("label:O", sort=label_order),
            x="value:Q",
            color=alt.Color("group:N", scale=cscale, legend=None),
            stroke=alt.value("white"),
        )
    )

    chart = (
        (rule + pts)
        .properties(
            title=alt.TitleParams(
                text="Difficulty variance by feature type",
                fontSize=14, color="#111111", anchor="start",
            ),
            width=_W, height=_H,
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT / "a_fim_separates_difficulty.png"), scale_factor=2)
    print("  A done")


# ── D: Localization bottleneck ────────────────────────────────────────────────
# Three bars, means labeled in matching color. No dots (6 is too few to show spread).
# Short x labels. No y-axis gridlines inside bars — clean.

def fig_d():
    pair_data = [
        {"layer": "File",      "mean": np.mean([0.76, 0.76, 0.65, 0.61, 0.69, 0.68])},
        {"layer": "Edit type", "mean": np.mean([0.45, 0.44, 0.46, 0.54, 0.51, 0.51])},
        {"layer": "Scope",     "mean": np.mean([0.26, 0.28, 0.25, 0.26, 0.28, 0.31])},
    ]
    df = pd.DataFrame(pair_data)
    df["baseline"] = 0.0
    layer_order = ["File", "Edit type", "Scope"]
    cscale = alt.Scale(domain=layer_order, range=[BLUE, ORANGE, GREEN])

    bars = (
        alt.Chart(df)
        .mark_bar(width=64)
        .encode(
            x=alt.X("layer:O", sort=layer_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=12)),
            y=alt.Y("baseline:Q",
                    scale=alt.Scale(domain=[0, 0.85]),
                    title="Mean pairwise agreement",
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.2, 0.4, 0.6, 0.8])),
            y2="mean:Q",
            color=alt.Color("layer:O", sort=layer_order, scale=cscale, legend=None),
        )
    )

    chart = (
        bars
        .properties(
            title=alt.TitleParams(
                text="Localization agreement by granularity",
                fontSize=14, color="#111111", anchor="start",
            ),
            width=340, height=280,
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT / "d_localization_bottleneck.png"), scale_factor=2)
    print("  D done")


# ── E: Semantic independence ──────────────────────────────────────────────────
# Line chart. ARI stays near 0 for all k.
# Zero rule only — no band. Fewer x ticks.

def fig_e():
    data = json.loads(
        (ROOT / "output" / "form_alignment" / "alignment_results.json").read_text()
    )
    df = pd.DataFrame([{"k": s["k"], "ari": s["ari"]} for s in data["sweep"]])

    line = (
        alt.Chart(df)
        .mark_line(color=BLUE, strokeWidth=1.8)
        .encode(
            x=alt.X("k:Q",
                    title="Number of semantic clusters (k)",
                    scale=alt.Scale(domain=[2, 26]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[5, 10, 15, 20, 25])),
            y=alt.Y("ari:Q",
                    title="ARI vs structural forms",
                    scale=alt.Scale(domain=[-0.016, 0.014]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[-0.01, 0, 0.01])),
        )
    )

    pts = (
        alt.Chart(df)
        .mark_point(color=BLUE, filled=True, size=35, opacity=0.7)
        .encode(x="k:Q", y="ari:Q")
    )

    # "ARI ≈ 0" label at right edge
    ann = (
        alt.Chart(pd.DataFrame([{"k": 25.5, "ari": 0.001, "t": "ARI ≈ 0"}]))
        .mark_text(align="left", fontSize=11, color="#999999", fontStyle="italic")
        .encode(x="k:Q", y="ari:Q", text="t:N")
    )

    chart = (
        (line + pts + ann)
        .properties(
            title=alt.TitleParams(
                text="Semantic vs structural independence",
                fontSize=14, color="#111111", anchor="start",
            ),
            width=_W, height=_H,
        )
        .configure_view(strokeWidth=0)
        .configure_axis(domain=False, ticks=False)
    )
    chart.save(str(OUT / "e_semantic_independence.png"), scale_factor=2)
    print("  E done")


if __name__ == "__main__":
    print("Generating gap figures...")
    fig_a()
    fig_d()
    fig_e()
    print(f"\nAll saved to {OUT}/")
