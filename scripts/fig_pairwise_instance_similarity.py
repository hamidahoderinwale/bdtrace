#!/usr/bin/env python3
"""
Distribution of pairwise structural similarity between task instances.

Computes the Jaccard similarity between every pair of oracle edit
certificates, then plots the ECDF of those values.

The key finding: over 95 percent of instance pairs share fewer than half
their edit operations, confirming that tasks are structurally diverse.
The rare high-similarity pairs (the tail > 0.6) correspond to instances
that belong to the same FIM canonical form.

Outputs:
    output/fig_pairwise_instance_similarity.png
    figures/gap/h_pairwise_instance_similarity.png
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _sp in (ROOT / ".venv" / "lib").glob("python*/site-packages"):
    if str(_sp) not in sys.path:
        sys.path.insert(0, str(_sp))

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, OLIVE, NEAR_BLACK

register()

OUT_MAIN = ROOT / "output"
OUT_GAP  = ROOT / "figures" / "gap"
OUT_GAP.mkdir(parents=True, exist_ok=True)

# ── Load and compute pairwise Jaccard ────────────────────────────────────────

certs = json.loads((ROOT / "output" / "oracle_edit_certs.json").read_text())
instances = list(certs.keys())
op_sets = {iid: set(certs[iid]["ops"]) for iid in instances}

jaccards: list[float] = []
for a, b in combinations(instances, 2):
    sa, sb = op_sets[a], op_sets[b]
    union = sa | sb
    if union:
        jaccards.append(len(sa & sb) / len(union))

jaccards_arr = np.sort(np.array(jaccards))
n = len(jaccards_arr)
print(f"Total pairs: {n:,}")
print(f"Mean Jaccard: {jaccards_arr.mean():.3f}, Median: {np.median(jaccards_arr):.3f}")
print(f"Pairs > 0.50: {(jaccards_arr > 0.50).mean():.1%}")
print(f"Pairs > 0.60: {(jaccards_arr > 0.60).mean():.1%}")
print(f"p90: {np.percentile(jaccards_arr, 90):.3f}")

# ── ECDF data ─────────────────────────────────────────────────────────────────

df_ecdf = pd.DataFrame({
    "jaccard": jaccards_arr,
    "ecdf": np.arange(1, n + 1) / n,
})

# Annotate the tail threshold (Jaccard = 0.5)
pct_below_half = float((jaccards_arr <= 0.5).mean())

ecdf_line = (
    alt.Chart(df_ecdf)
    .mark_line(color=BLUE, strokeWidth=2)
    .encode(
        x=alt.X(
            "jaccard:Q",
            title="Jaccard similarity between oracle fix operations",
            scale=alt.Scale(domain=[0, 1]),
        ),
        y=alt.Y(
            "ecdf:Q",
            title="Fraction of instance pairs",
            axis=alt.Axis(format=".0%"),
            scale=alt.Scale(domain=[0, 1]),
        ),
    )
)

# Vertical reference at 0.5 with label — use a single point rather than a rule
# (rule = reference line, which is excluded; use text annotation instead)
label_df = pd.DataFrame({"x": [0.50], "y": [pct_below_half]})

dot = (
    alt.Chart(label_df)
    .mark_point(color=OLIVE, size=60, strokeWidth=0)
    .encode(
        x="x:Q",
        y="y:Q",
    )
)

label_text_1 = (
    alt.Chart(
        pd.DataFrame({
            "x": [0.52],
            "y": [pct_below_half - 0.04],
            "text": [f"{pct_below_half:.0%} of pairs"],
        })
    )
    .mark_text(align="left", color=NEAR_BLACK, fontSize=11)
    .encode(x="x:Q", y="y:Q", text="text:N")
)

label_text_2 = (
    alt.Chart(
        pd.DataFrame({
            "x": [0.52],
            "y": [pct_below_half - 0.10],
            "text": ["have Jaccard below 0.5"],
        })
    )
    .mark_text(align="left", color=NEAR_BLACK, fontSize=11)
    .encode(x="x:Q", y="y:Q", text="text:N")
)

fig = (
    alt.layer(ecdf_line, dot, label_text_1, label_text_2)
    .properties(
        width=380,
        height=260,
        title="Most task pairs share fewer than half their edit operations",
    )
)

for dest in [OUT_MAIN / "fig_pairwise_instance_similarity.png",
             OUT_GAP  / "h_pairwise_instance_similarity.png"]:
    fig.save(str(dest), scale_factor=2.0)
    print(f"saved → {dest.relative_to(ROOT)}")
