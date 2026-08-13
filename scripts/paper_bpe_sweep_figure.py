"""Regenerate bpe_vocab_sweep_compression.png from cached JSON using Altair.

Reads output/paper2_pilot/bpe_vocab_sweep.json (already computed by the
expensive bpe_vocab_sweep.py sweep) and produces an Altair figure.

Both metrics normalized to [0, 1] on a single panel so the elbow — where
compression gain flattens while merge cost keeps rising — is directly visible.

Usage:
    uv run python scripts/paper_bpe_sweep_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE
register()

OUT = ROOT / "output" / "paper2_pilot"
DATA = OUT / "bpe_vocab_sweep.json"


def main() -> None:
    with open(DATA) as f:
        d = json.load(f)

    raw = {r["V"]: r for r in d["results"]}
    Vs = sorted(raw)

    comp_vals  = [raw[v]["compression_ratio"] for v in Vs]
    merge_vals = [raw[v]["n_merges"]          for v in Vs]

    comp_min,  comp_max  = min(comp_vals),  max(comp_vals)
    merge_min, merge_max = min(merge_vals), max(merge_vals)

    rows = []
    for v, c, m in zip(Vs, comp_vals, merge_vals):
        # Invert compression: higher = more compressed = better
        rows.append({"V": v, "metric": "Compression gain", "value": (comp_max - c) / (comp_max - comp_min)})
        rows.append({"V": v, "metric": "Merge cost",       "value": (m - merge_min) / (merge_max - merge_min)})

    df = pd.DataFrame(rows)

    metric_order = ["Compression gain", "Merge cost"]
    cscale = alt.Scale(domain=metric_order, range=[BLUE, ORANGE])

    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("V:Q",
                    title="BPE target vocabulary size (V)",
                    scale=alt.Scale(domain=[90, 510]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[100, 200, 300, 400, 500])),
            y=alt.Y("value:Q",
                    title="Normalized value",
                    scale=alt.Scale(domain=[-0.02, 1.05]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.25, 0.5, 0.75, 1.0],
                                  format=".0%")),
            color=alt.Color("metric:N", scale=cscale,
                            legend=alt.Legend(title=None, orient="bottom",
                                              direction="horizontal")),
        )
    )

    pts = (
        alt.Chart(df)
        .mark_point(size=60, filled=True, strokeWidth=0)
        .encode(
            x="V:Q",
            y="value:Q",
            color=alt.Color("metric:N", scale=cscale, legend=None),
        )
    )

    chart = (
        (lines + pts)
        .properties(
            title=alt.TitleParams(
                text="BPE sweep: compression gain vs merge cost",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=440, height=260,
        )
        .configure_view(strokeWidth=0)
    )

    out_path = OUT / "bpe_vocab_sweep_compression.png"
    chart.save(str(out_path), scale_factor=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
