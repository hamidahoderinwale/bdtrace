"""Cost per resolved task vs mean actions per task, per agent.

Shows that step count and dollar cost do not track each other. All 9 agents.
Direct cost data shown as filled markers; estimates as open markers.

Reads:
    output/paper2_pilot/cost_per_agent.json
    output/paper2_pilot/aggregate_metrics_extended.json
Writes:
    output/figures/fig_cost_vs_steps.png

Usage:
    uv run python scripts/figures/fig_cost_vs_steps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, MAGENTA, COPPER, OLIVE, GREEN_D, BLUE_D, MAGENTA_D
register()

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"

AGENT_COLORS = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "Claude-3.7-thinking":   GREEN_D,
    "Claude-4":              "#187860",
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "DARS+R1":               MAGENTA_D,
    "Agentless+Claude-3.5":  BLUE_D,
    "Moatless+V3":           OLIVE,
}
AGENT_ORDER = list(AGENT_COLORS)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    cost_rows = json.loads((OUT_DAT / "cost_per_agent.json").read_text())
    agg = json.loads((OUT_DAT / "aggregate_metrics_extended.json").read_text())

    rows = []
    for r in cost_rows:
        agent = r["agent"]
        m = agg.get("metrics", {}).get(agent, {})
        mean_atoms = (m.get("canonical_length_mean")
                      or m.get("mean_canonical_length")
                      or m.get("mean_atoms"))
        cost_per_resolved = r.get("cost_per_resolved_usd")
        if mean_atoms is None or cost_per_resolved is None:
            continue
        is_estimate = "estimate" in (r.get("source") or "")
        rows.append({
            "agent": agent,
            "mean_atoms": float(mean_atoms),
            "cost_per_resolved": float(cost_per_resolved),
            "cost_per_task": r.get("cost_mean_usd"),
            "n_resolved": r.get("n_resolved"),
            "kind": "estimated" if is_estimate else "measured",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    measured = (
        alt.Chart(df[df["kind"] == "measured"])
        .mark_point(size=150, filled=True, opacity=0.95, strokeWidth=0)
        .encode(
            x=alt.X("mean_atoms:Q",
                    scale=alt.Scale(domain=[0, df["mean_atoms"].max() * 1.1]),
                    axis=alt.Axis(title="Mean actions per task",
                                  domain=False, ticks=False, labelFontSize=10)),
            y=alt.Y("cost_per_resolved:Q",
                    scale=alt.Scale(type="log",
                                    domain=[df["cost_per_resolved"].min() * 0.5,
                                            df["cost_per_resolved"].max() * 2]),
                    axis=alt.Axis(title="Cost per resolved task (USD, log)",
                                  domain=False, ticks=False, labelFontSize=10)),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
            tooltip=["agent", "mean_atoms", "cost_per_resolved", "kind"],
        )
    )
    estimated = (
        alt.Chart(df[df["kind"] == "estimated"])
        .mark_point(size=150, filled=False, opacity=0.95, strokeWidth=2)
        .encode(
            x="mean_atoms:Q",
            y="cost_per_resolved:Q",
            color=alt.Color("agent:N", scale=color_scale, legend=None),
            tooltip=["agent", "mean_atoms", "cost_per_resolved", "kind"],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=8, dy=-4, fontSize=10, color="#333333")
        .encode(
            x="mean_atoms:Q",
            y="cost_per_resolved:Q",
            text="agent:N",
        )
    )

    chart = (
        (measured + estimated + labels)
        .properties(
            width=480, height=320,
            title=alt.TitleParams(
                text="Cost per resolved task vs actions per task",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    out_png = OUT_FIG / "fig_cost_vs_steps.png"
    chart.save(str(out_png), scale_factor=2)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
