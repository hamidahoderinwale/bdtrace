"""Per-agent step-count (trajectory-length) distribution.

Manual boxplot: whiskers + IQR box + outliers, no median bar.
Two PNGs:
  - 4-agent baseline (bpe_sequences.jsonl)
  - 8-submission extended (bpe_sequences_extended.jsonl)

Outputs:
    output/figures/fig_step_count_distribution.png
    output/figures/fig_step_count_distribution_extended.png

Usage:
    python scripts/figures/fig_step_count_distribution.py
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
    register, GREEN, BLUE, MAGENTA, COPPER, OLIVE,
    GREEN_D, BLUE_D, MAGENTA_D,
    AGENT_COLORS, AGENT_ORDER,
)
register()

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"

EXTENDED_PALETTE = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "Claude-3.7-thinking":   GREEN_D,
    "Claude-4":              "#187860",  # darker GREEN_D variant for the second extended-thinking Claude
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "DARS+R1":               MAGENTA_D,
    "Agentless+Claude-3.5":  OLIVE,
    "Moatless+V3":           BLUE_D,
}
EXTENDED_ORDER = [
    # Standing rule (feedback_fullest_corpus_default): every analysis includes
    # all 9 agents; the deterministic Agentless+Claude-3.5 row reads as a
    # delta-function and that itself is the data.
    "Agentless+Claude-3.5",
    "Claude-3",
    "Claude-3.5",
    "Claude-3.7-thinking",
    "Claude-4",
    "DARS+R1",
    "GPT-4",
    "GPT-4o",
    "Moatless+V3",
]


def boxplot_stats(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo = max(float(s.min()), float(q1 - 1.5 * iqr))
    hi = min(float(s.max()), float(q3 + 1.5 * iqr))
    return pd.Series({"q1": float(q1), "q3": float(q3), "lo": lo, "hi": hi})


def render(jsonl_path: Path, agents_order: list[str], palette: dict[str, str],
           title: str, out_path: Path) -> None:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"agent": r["agent"], "length": r["canonical_length"]})
    df = pd.DataFrame(rows)
    df = df[df["agent"].isin(agents_order)]

    summary = (
        df.groupby("agent")["length"]
          .apply(boxplot_stats)
          .unstack()
          .reset_index()
    )
    outliers = (
        df.merge(summary[["agent", "lo", "hi"]], on="agent")
          .query("length < lo or length > hi")
    )

    color_scale = alt.Scale(
        domain=agents_order,
        range=[palette.get(a, OLIVE) for a in agents_order],
    )

    whiskers = (
        alt.Chart(summary)
        .mark_rule(strokeWidth=1.2)
        .encode(
            x=alt.X("lo:Q",
                    axis=alt.Axis(title="Step count per task  (canonical atoms)",
                                  domain=False, ticks=False, labelFontSize=10)),
            x2="hi:Q",
            y=alt.Y("agent:N", sort=agents_order, title=None,
                    axis=alt.Axis(domain=False, ticks=False, labelFontSize=10, labelLimit=240)),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )
    )
    box = (
        alt.Chart(summary)
        .mark_bar(height=14)
        .encode(
            x="q1:Q",
            x2="q3:Q",
            y=alt.Y("agent:N", sort=agents_order, title=None),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )
    )
    pts = (
        alt.Chart(outliers)
        .mark_point(opacity=0.4, size=18, filled=False, strokeWidth=1)
        .encode(
            x="length:Q",
            y=alt.Y("agent:N", sort=agents_order, title=None),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )
    )

    chart = (
        (whiskers + box + pts)
        .properties(
            width=460,
            height=max(180, 32 * len(agents_order)),
            title=alt.TitleParams(
                text=title,
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(out_path), scale_factor=2)
    print(f"Saved: {out_path}")


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    render(
        OUT_DAT / "bpe_sequences.jsonl",
        AGENT_ORDER, AGENT_COLORS,
        "Step count per task",
        OUT_FIG / "fig_step_count_distribution.png",
    )

    render(
        OUT_DAT / "bpe_sequences_extended.jsonl",
        EXTENDED_ORDER, EXTENDED_PALETTE,
        "Step count per task",
        OUT_FIG / "fig_step_count_distribution_extended.png",
    )


if __name__ == "__main__":
    main()
