"""Static PNG version of the V-sweep stability figure (9-agent corpus).

Companion to fig_jsd_stability_interactive.py (which writes the HTML
with hover tooltips). This static variant collapses the 36 individual
agent-pair lines into a band view colored by cell-pair so the figure
remains readable on the dashboard.

Message: rank order of pairs (by motif JSD) is invariant across
V in {100, 125, 150, 175, 200, 225, 250, 300, 500}. Magnitudes scale
upward with V, lines never cross.

Reads:  output/paper2_pilot/bpe_mdl_sweep.json
Writes: output/figures/fig_jsd_stability.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import (
    register, BLUE, GREEN, MAGENTA, COPPER, OLIVE,
    INDIGO, VIOLET, SIENNA,
)
register()

DATA = ROOT / "output" / "paper2_pilot" / "bpe_mdl_sweep.json"
OUT  = ROOT / "output" / "figures" / "fig_jsd_stability.png"

# Map each agent to its paradigm-by-scaffold cell, matching the
# trajectory_clusters_extended.py / pattern_matcher_validation.py scheme.
AGENT_CELL = {
    "Claude-3":              "SWE-agent base",
    "Claude-3.5":            "SWE-agent base",
    "GPT-4":                 "SWE-agent base",
    "GPT-4o":                "SWE-agent base",
    "Claude-3.7-thinking":   "SWE-agent extended-thinking",
    "Claude-4":              "SWE-agent extended-thinking",
    "Agentless+Claude-3.5":  "Agentless",
    "DARS+R1":               "DARS",
    "Moatless+V3":           "Moatless",
}

CELL_ORDER = [
    "SWE-agent base", "SWE-agent extended-thinking",
    "Agentless", "DARS", "Moatless",
]


def cell_pair_label(a: str, b: str) -> str:
    ca, cb = AGENT_CELL[a], AGENT_CELL[b]
    if ca == cb:
        return f"within {ca}"
    pair = tuple(sorted([ca, cb]))
    return f"{pair[0]} × {pair[1]}"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sweep = json.loads(DATA.read_text())

    rows: list[dict] = []
    for r in sweep["results"]:
        V = r["V"]
        for pair, jsd in r["jsd_motifs"].items():
            a, b = pair.split("__")
            if a not in AGENT_CELL or b not in AGENT_CELL:
                continue
            rows.append({
                "V": V,
                "pair": f"{a} / {b}",
                "cell_pair": cell_pair_label(a, b),
                "jsd": jsd,
            })
    df = pd.DataFrame(rows)
    Vs = sorted(df["V"].unique())

    # Color by cell-pair: within-cell pairs in cool colors, cross-cell
    # pairs in warm colors, so the cluster of within-cell-base lines
    # reads at a glance.
    cell_pair_colors: dict[str, str] = {}
    cool_within = [BLUE, GREEN, INDIGO, VIOLET, OLIVE]
    warm_across = [COPPER, MAGENTA, SIENNA, "#A0784A", "#7D6B3F",
                   "#5D8B6B", "#8C4D72", "#2E6D6D", "#A03D18", "#6B4A8C"]
    sorted_cell_pairs = sorted(df["cell_pair"].unique(),
                               key=lambda s: (not s.startswith("within"), s))
    for cp in sorted_cell_pairs:
        if cp.startswith("within"):
            cell_pair_colors[cp] = cool_within.pop(0) if cool_within else "#888888"
        else:
            cell_pair_colors[cp] = warm_across.pop(0) if warm_across else "#888888"

    color_scale = alt.Scale(
        domain=list(cell_pair_colors.keys()),
        range=list(cell_pair_colors.values()),
    )

    base = alt.Chart(df).encode(
        x=alt.X("V:Q",
                title="BPE vocabulary size V",
                scale=alt.Scale(type="log", domain=[Vs[0] * 0.9, Vs[-1] * 1.1]),
                axis=alt.Axis(values=Vs, labelFontSize=10, domain=False, ticks=False)),
        y=alt.Y("jsd:Q",
                title="Pairwise JSD (motifs)",
                scale=alt.Scale(domain=[0, max(df["jsd"]) * 1.08]),
                axis=alt.Axis(labelFontSize=10, domain=False, ticks=False, format=".2f")),
        color=alt.Color(
            "cell_pair:N", scale=color_scale,
            legend=alt.Legend(orient="bottom", title=None, columns=3,
                              symbolSize=80, labelFontSize=9),
        ),
        detail="pair:N",
    )

    lines = base.mark_line(strokeWidth=1.4, opacity=0.85)
    points = base.mark_point(filled=True, size=24, opacity=0.85)

    chart = (
        alt.layer(lines, points)
        .properties(
            width=520, height=320,
            title=alt.TitleParams(
                "JSD vs vocabulary size (9-agent corpus)",
                subtitle=(
                    "36 agent-pair lines across V in {100..500}; "
                    "colored by cell-pair. Lines never cross: pair "
                    "ranking is invariant to V. Magnitudes amplify."
                ),
                fontSize=13, subtitleFontSize=10,
                color="#111111", subtitleColor="#666666", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(OUT), scale_factor=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
