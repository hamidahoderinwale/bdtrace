"""Side-by-side comparison figure: regex baseline vs embedding at matched liberality.

Reads existing per-trajectory results from both matchers, computes per-agent
medians at the matched-liberality threshold (0.45), and produces a 2x2 grouped
bar chart: agents on rows, metrics (forward, reverse) on columns, matcher as
the grouping variable.

Reads:
    output/paper2_pilot/cot_action_alignment.jsonl
    output/paper2_pilot/cot_action_alignment_embedding.jsonl
Writes:
    output/figures/fig_cot_matcher_comparison.png
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
OUT_FIG = ROOT / "output" / "figures" / "fig_cot_matcher_comparison.png"

PRIMARY_THRESHOLD = 0.45  # matched-liberality with regex


def load_regex() -> pd.DataFrame:
    records = [json.loads(line) for line in REGEX_JSONL.open()]
    df = pd.DataFrame([r for r in records if r.get("jaccard") is not None])
    return df.groupby("agent").agg(
        forward=("forward_coverage", "median"),
        reverse=("reverse_coverage", "median"),
    ).reset_index().assign(matcher="Regex (baseline)")


def load_embed(threshold: float) -> pd.DataFrame:
    records = [json.loads(line) for line in EMBED_JSONL.open()]
    df = pd.DataFrame([r for r in records if r["threshold"] == threshold and r.get("jaccard") is not None])
    return df.groupby("agent").agg(
        forward=("forward_coverage", "median"),
        reverse=("reverse_coverage", "median"),
    ).reset_index().assign(matcher=f"Embedding (threshold {threshold})")


def main() -> None:
    regex_df = load_regex()
    embed_df = load_embed(PRIMARY_THRESHOLD)
    combined = pd.concat([regex_df, embed_df], ignore_index=True)

    long_df = pd.melt(
        combined, id_vars=["agent", "matcher"],
        value_vars=["forward", "reverse"],
        var_name="metric", value_name="coverage",
    )
    long_df["metric"] = long_df["metric"].map({"forward": "Forward (stated → done)", "reverse": "Reverse (done → stated)"})

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("matcher:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=9, labelLimit=200)),
            y=alt.Y("coverage:Q", title="Median coverage", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "matcher:N",
                scale=alt.Scale(domain=["Regex (baseline)", f"Embedding (threshold {PRIMARY_THRESHOLD})"], range=[OLIVE, BLUE]),
                legend=None,
            ),
            column=alt.Column("metric:N", title=None, header=alt.Header(labelFontSize=11)),
            row=alt.Row("agent:N", title=None, header=alt.Header(labelFontSize=11)),
        )
        .properties(width=180, height=120)
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT_FIG), scale_factor=2)
    print(f"Saved {OUT_FIG}")

    # Print the table for the explanation
    print("\nAsymmetry under both matchers (matched liberality, threshold 0.45):")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
