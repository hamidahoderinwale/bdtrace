"""Barbell version of the CoT-action coverage comparison.

Each barbell = one (agent, matcher) combination. Endpoints are forward and
reverse coverage; the line between them is the asymmetry gap. Four barbells
total: two agents × two matchers.

Reads:
    output/paper2_pilot/cot_action_alignment.jsonl
    output/paper2_pilot/cot_action_alignment_embedding.jsonl
Writes:
    output/figures/fig_cot_matcher_barbell.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, OLIVE
register()

REGEX_JSONL = ROOT / "output" / "paper2_pilot" / "cot_action_alignment.jsonl"
EMBED_JSONL = ROOT / "output" / "paper2_pilot" / "cot_action_alignment_embedding.jsonl"
OUT_FIG = ROOT / "output" / "figures" / "fig_cot_matcher_barbell.png"
PRIMARY_THRESHOLD = 0.45


def load_data() -> pd.DataFrame:
    regex = [json.loads(line) for line in REGEX_JSONL.open()]
    regex_df = pd.DataFrame([r for r in regex if r.get("jaccard") is not None])
    regex_agg = regex_df.groupby("agent").agg(
        forward=("forward_coverage", "median"),
        reverse=("reverse_coverage", "median"),
    ).reset_index().assign(matcher="Regex")

    embed = [json.loads(line) for line in EMBED_JSONL.open()]
    embed_df = pd.DataFrame([r for r in embed if r["threshold"] == PRIMARY_THRESHOLD and r.get("jaccard") is not None])
    embed_agg = embed_df.groupby("agent").agg(
        forward=("forward_coverage", "median"),
        reverse=("reverse_coverage", "median"),
    ).reset_index().assign(matcher="Embedding")

    return pd.concat([regex_agg, embed_agg], ignore_index=True)


def main() -> None:
    df = load_data()
    # Compose y-axis label that groups by agent and matcher in a stable order
    df["row_label"] = df["agent"] + " · " + df["matcher"]
    row_order = [
        f"Claude-3.7-thinking · Regex",
        f"Claude-3.7-thinking · Embedding",
        f"Claude-4 · Regex",
        f"Claude-4 · Embedding",
    ]

    color_scale = alt.Scale(domain=["Regex", "Embedding"], range=[OLIVE, BLUE])

    base = alt.Chart(df).encode(
        y=alt.Y("row_label:N", sort=row_order, title=None, axis=alt.Axis(labelFontSize=10, labelLimit=200)),
    )

    # The connecting line (the asymmetry gap)
    line = base.mark_rule(strokeWidth=3).encode(
        x=alt.X("forward:Q", title="Coverage", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        x2="reverse:Q",
        color=alt.Color("matcher:N", scale=color_scale, legend=alt.Legend(title="Matcher", orient="bottom")),
    )

    # Endpoint markers — forward = open circle, reverse = filled circle
    forward_pt = base.mark_circle(size=120, fill="white", strokeWidth=2).encode(
        x="forward:Q",
        stroke=alt.Color("matcher:N", scale=color_scale, legend=None),
    )
    reverse_pt = base.mark_circle(size=120).encode(
        x="reverse:Q",
        color=alt.Color("matcher:N", scale=color_scale, legend=None),
    )

    # Value labels at each endpoint
    forward_lbl = base.mark_text(align="right", dx=-8, fontSize=9, color="#444444").encode(
        x="forward:Q",
        text=alt.Text("forward:Q", format=".2f"),
    )
    reverse_lbl = base.mark_text(align="left", dx=8, fontSize=9, color="#444444").encode(
        x="reverse:Q",
        text=alt.Text("reverse:Q", format=".2f"),
    )

    chart = (
        (line + forward_pt + reverse_pt + forward_lbl + reverse_lbl)
        .properties(
            width=420, height=180,
            title=alt.TitleParams(
                text="Stated → done (open) vs done → stated (filled), per matcher",
                fontSize=11, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT_FIG), scale_factor=2)
    print(f"Saved {OUT_FIG}")
    print("\nValues:")
    print(df[["agent", "matcher", "forward", "reverse"]].to_string(index=False))


if __name__ == "__main__":
    main()
