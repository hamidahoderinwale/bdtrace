"""Per-difficulty-bin agent pass-rate distribution (9-agent corpus).

For each n_resolved bucket (0/9..9/9), shows the underlying pass-rate
distribution across the 9 agents directly, as a strip plot. One dot per
(agent, bin); dots are colored by canonical agent palette. A range
bracket per bin marks min--max so the spread reads pre-attentively.

This replaces the prior "MI(agent; pass/fail) per bin" bar chart. The
MI / pct fields remain in the JSON for backwards compatibility, but
the figure now shows the underlying distribution rather than the
information-theoretic summary. Spread per bin is the same quantity
MI was trying to compress; showing the dots makes the agent-by-agent
detail visible without collapsing it.

Reads:  output/paper2_pilot/per_bin_agent_mi.json
Writes: output/figures/fig_per_bin_agent_mi.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER

register()

FIG_OUT  = ROOT / "output" / "figures"
DATA_IN  = ROOT / "output" / "paper2_pilot" / "per_bin_agent_mi.json"


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA_IN.read_text())

    n_bins = len(data)
    denom = n_bins - 1  # 9 for the extended corpus

    dot_rows: list[dict] = []
    range_rows: list[dict] = []
    n_rows: list[dict] = []

    for bin_key, vals in data.items():
        bin_label = f"{bin_key}/{denom}"
        bin_int = int(bin_key)
        n_rows.append({
            "bin": bin_label, "bin_order": bin_int,
            "n": vals.get("n", 0),
        })
        per_agent = vals.get("per_agent", {})
        rates = []
        for agent, stats in per_agent.items():
            rate = stats.get("pass_rate")
            if rate is None:
                continue
            rates.append(rate)
            dot_rows.append({
                "bin": bin_label,
                "bin_order": bin_int,
                "agent": agent,
                "pass_rate": rate,
                "n_traj": stats.get("n", 0),
                "n_resolved": stats.get("passed", 0),
            })
        if rates:
            range_rows.append({
                "bin": bin_label,
                "bin_order": bin_int,
                "lo": min(rates),
                "hi": max(rates),
                "spread": round(max(rates) - min(rates), 3),
                "spread_label": f"{int(round((max(rates) - min(rates)) * 100))}-pt",
            })

    dots_df  = pd.DataFrame(dot_rows).sort_values(["bin_order", "agent"])
    range_df = pd.DataFrame(range_rows).sort_values("bin_order")
    n_df     = pd.DataFrame(n_rows).sort_values("bin_order")
    bin_order = n_df["bin"].tolist()

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    # Range bracket: thin vertical rule from min to max per bin.
    range_layer = (
        alt.Chart(range_df)
        .mark_rule(strokeWidth=1.4, color="#888888")
        .encode(
            x=alt.X("bin:N", sort=bin_order),
            y=alt.Y("lo:Q"),
            y2=alt.Y2("hi:Q"),
        )
    )

    # One dot per (agent, bin), colored by canonical agent palette.
    # No xOffset: dots stack on the same vertical line within each bin
    # and are separated by their (already-distinct) y=pass_rate values.
    # When several agents share a pass rate (most often at 0% or 100%
    # in the boundary bins), the dots overlap and the cluster color is
    # the readable signal. Transparency lets overlap remain visible.
    dots = (
        alt.Chart(dots_df)
        .mark_circle(size=110, opacity=0.78, stroke="white", strokeWidth=0.8)
        .encode(
            x=alt.X(
                "bin:N",
                sort=bin_order,
                axis=alt.Axis(
                    title="Number of the 9 agents that resolved the task",
                    labelAngle=0, domain=False, ticks=False, labelFontSize=10,
                ),
            ),
            y=alt.Y(
                "pass_rate:Q",
                title="Pass rate on tasks in this difficulty bin",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", domain=False, ticks=False, labelFontSize=10),
            ),
            color=alt.Color(
                "agent:N", scale=color_scale,
                legend=alt.Legend(orient="bottom", title=None, columns=5),
            ),
            tooltip=[
                "agent", "bin",
                alt.Tooltip("pass_rate:Q", format=".1%"),
                alt.Tooltip("n_traj:Q", title="trajectories"),
                alt.Tooltip("n_resolved:Q", title="resolved"),
            ],
        )
    )

    # Spread label centered above each bin's range bracket, anchored to
    # y = 1.05 so it sits above the y=100% gridline (no collision with
    # the dot cluster at 100% in 9/9).
    spread_labels = (
        alt.Chart(range_df[range_df["spread"] > 0])
        .mark_text(fontSize=10, color="#444444", baseline="bottom")
        .encode(
            x=alt.X("bin:N", sort=bin_order),
            y=alt.value(8),  # 8 px from top of plot area
            text="spread_label:N",
        )
    )

    chart = (
        alt.layer(range_layer, dots, spread_labels)
        .properties(
            width=alt.Step(56),
            height=300,
            title=alt.TitleParams(
                "Agent pass-rate distribution within difficulty bins",
                subtitle=(
                    "Each dot = one agent's pass rate in that bin; bracket = range "
                    "(top labels). Bracket size is the metric: how much does the "
                    "agent matter at this difficulty?"
                ),
                fontSize=12, subtitleFontSize=10,
                color="#111111", subtitleColor="#777777", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = FIG_OUT / "fig_per_bin_agent_mi.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
