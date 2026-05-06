"""Paper figure for the grounding failure finding.

Loads output/grounding_validation/grounding_gpt_4o.parquet and produces:

  figures/grounding/f_grounding_failure.png
      Grouped bar chart: precision, recall, and F1 by condition.

Usage:
    uv run python scripts/paper_grounding_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GRAY
register()

OUT  = ROOT / "figures" / "grounding"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "output" / "grounding_validation" / "grounding_gpt_4o.parquet"

CONDITION_LABELS = {
    "no_context": "No context",
    "procedural": "Structural context",
    "raw_logs":   "Raw logs",
}
COND_ORDER   = ["No context", "Structural context", "Raw logs"]
METRIC_ORDER = ["Precision", "Recall", "F1"]
CSCALE = alt.Scale(domain=COND_ORDER, range=[GRAY, BLUE, ORANGE])


def main() -> None:
    df = pd.read_parquet(DATA)
    df["condition_label"] = df["condition"].map(CONDITION_LABELS)

    long = df.melt(
        id_vars="condition_label",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(
        {"precision": "Precision", "recall": "Recall", "f1": "F1"}
    )

    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", sort=METRIC_ORDER,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=11)),
            y=alt.Y("mean(score):Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Score", domain=False, ticks=False,
                                  values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
            xOffset=alt.XOffset("condition_label:N", sort=COND_ORDER),
            color=alt.Color("condition_label:N", sort=COND_ORDER, scale=CSCALE,
                            legend=alt.Legend(orient="bottom", title=None,
                                              symbolSize=80)),
        )
    )

    errors = (
        alt.Chart(long)
        .mark_errorbar(extent="stderr", ticks=True)
        .encode(
            x=alt.X("metric:N", sort=METRIC_ORDER),
            y=alt.Y("score:Q", title=None),
            xOffset=alt.XOffset("condition_label:N", sort=COND_ORDER),
            color=alt.Color("condition_label:N", sort=COND_ORDER, scale=CSCALE,
                            legend=None),
        )
    )

    chart = (
        (bars + errors)
        .properties(
            title=alt.TitleParams(
                text="Edit grounding metrics (GPT-4o)",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=360, height=260,
        )
        .configure_view(strokeWidth=0)
    )

    out_path = OUT / "f_grounding_failure.png"
    chart.save(str(out_path), scale_factor=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
