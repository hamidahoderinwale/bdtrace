"""Per-difficulty-bin agent MI figure (R7).

Shows MI(agent; pass/fail) within each n_resolved bin.
Bins 0/4 and 4/4 are zero by construction (no outcome variance).
Uncertain bins 1/4–3/4 show 5–11% agent MI — 3–7x the corpus-level 1.5%.

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
from scripts.theme import register, BLUE, GRAY

register()

FIG_OUT  = ROOT / "output" / "figures"
DATA_IN  = ROOT / "output" / "paper2_pilot" / "per_bin_agent_mi.json"


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    data = json.loads(DATA_IN.read_text())

    rows = []
    for bin_key, vals in data.items():
        label = f"{bin_key}/4 agents passed"
        rows.append({
            "bin":        label,
            "bin_order":  int(bin_key),
            "pct":        vals["pct"],
            "n":          vals["n"],
            "uncertain":  int(bin_key) in {1, 2, 3},
        })

    df = pd.DataFrame(rows).sort_values("bin_order")
    df["label"] = df["pct"].apply(lambda v: f"{v:.0f}%")

    bin_order = df["bin"].tolist()

    color_scale = alt.Scale(
        domain=[True, False],
        range=[BLUE, GRAY],
    )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("bin:N", sort=bin_order, axis=alt.Axis(title="Number of agents that passed", labelAngle=0)),
            y=alt.Y("pct:Q", title="Outcome uncertainty removed by agent (%)",
                    scale=alt.Scale(domain=[0, 14])),
            color=alt.Color("uncertain:N", scale=color_scale, legend=None),
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(dy=-8, fontSize=11, color="#333333")
        .encode(
            x=alt.X("bin:N", sort=bin_order),
            y=alt.Y("pct:Q"),
            text="label:N",
        )
    )

    n_labels = (
        alt.Chart(df)
        .mark_text(dy=12, fontSize=9, color="#888888")
        .encode(
            x=alt.X("bin:N", sort=bin_order),
            y=alt.value(0),
            text=alt.Text("n:Q", format=",d"),
        )
    )

    chart = (
        alt.layer(bars, labels, n_labels)
        .properties(
            width=320,
            height=200,
            title=alt.TitleParams(
                "Agent MI within difficulty bins",
                fontSize=12, color="#111111", anchor="start",
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
