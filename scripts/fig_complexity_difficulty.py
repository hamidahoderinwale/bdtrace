#!/usr/bin/env python3
"""
Structural complexity of the oracle fix vs task difficulty.

Two panels:
  Left  — histogram of edit-certificate cardinality, split into three
           difficulty buckets (easy / medium / hard by ease-score tertile).
  Right — scatter: ease score (y) vs cert cardinality (x), one dot per
           instance, colored by the same bucket.

Outputs:
    output/fig_complexity_difficulty.png
    figures/gap/g_complexity_difficulty.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _sp in (ROOT / ".venv" / "lib").glob("python*/site-packages"):
    if str(_sp) not in sys.path:
        sys.path.insert(0, str(_sp))

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT))
from scripts.theme import register, MAGENTA, OLIVE, GREEN, NEAR_BLACK

register()

OUT_MAIN = ROOT / "output"
OUT_GAP  = ROOT / "figures" / "gap"
OUT_GAP.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

certs = json.loads((ROOT / "output" / "oracle_edit_certs.json").read_text())
leaderboard = json.loads((ROOT / "output" / "leaderboard" / "lite_results.json").read_text())

agents = list(leaderboard.keys())
n_agents = len(agents)

ease: dict[str, float] = {}
for iid in certs:
    solved = sum(1 for a in agents if leaderboard[a].get(iid) is True)
    ease[iid] = solved / n_agents

# Build flat dataframe
rows = []
for iid, cert in certs.items():
    if iid not in ease:
        continue
    rows.append({"instance": iid, "n_ops": cert["n_ops"], "ease": ease[iid]})

df = pd.DataFrame(rows)

# Two-bucket split: hard = bottom ease quartile vs the rest
q25 = df["ease"].quantile(0.25)
df["bucket"] = df["ease"].apply(lambda e: "hard" if e < q25 else "not hard")

# Tertile split for scatter color gradient
q1, q2 = df["ease"].quantile([1 / 3, 2 / 3])
df["tertile"] = pd.cut(
    df["ease"],
    bins=[-0.001, q1, q2, 1.001],
    labels=["hard", "medium", "easy"],
)

# Histogram: two buckets (hard = MAGENTA, not hard = OLIVE)
hist_domain = ["hard", "not hard"]
hist_range  = [MAGENTA, OLIVE]

hist = (
    alt.Chart(df)
    .mark_bar(opacity=0.82, binSpacing=1)
    .encode(
        x=alt.X(
            "n_ops:Q",
            bin=alt.Bin(step=2),
            title="Edit operations in oracle fix",
        ),
        y=alt.Y("count():Q", title="Instances", stack="zero"),
        color=alt.Color(
            "bucket:N",
            scale=alt.Scale(domain=hist_domain, range=hist_range),
            title="Difficulty",
            sort=hist_domain,
        ),
        order=alt.Order("bucket:N", sort="descending"),
    )
    .properties(
        width=240,
        height=220,
        title="Complex fixes cluster in harder instances",
    )
)

# ── Right panel: scatter ─────────────────────────────────────────────────────

# Jitter x slightly to reduce overplotting
rng = np.random.default_rng(0)
df["n_ops_jitter"] = df["n_ops"] + rng.uniform(-0.3, 0.3, size=len(df))

# Scatter: tertile colors for gradient (hard=MAGENTA, medium=OLIVE, easy=GREEN)
scatter_domain = ["hard", "medium", "easy"]
scatter_range  = [MAGENTA, OLIVE, GREEN]

scatter = (
    alt.Chart(df)
    .mark_point(size=28, opacity=0.6, strokeWidth=0)
    .encode(
        x=alt.X(
            "n_ops_jitter:Q",
            axis=alt.Axis(title="Edit operations in oracle fix", tickMinStep=2),
        ),
        y=alt.Y(
            "ease:Q",
            axis=alt.Axis(
                title="Fraction of 84 agents that solved the instance",
                format=".0%",
            ),
        ),
        color=alt.Color(
            "tertile:N",
            scale=alt.Scale(domain=scatter_domain, range=scatter_range),
            title="Difficulty",
            sort=scatter_domain,
        ),
    )
    .properties(
        width=240,
        height=220,
        title="Agents fail more on structurally complex fixes",
    )
)

# ── Compose and save ─────────────────────────────────────────────────────────

fig = alt.hconcat(hist, scatter, spacing=32).configure_legend(
    orient="bottom",
    direction="horizontal",
    titleOrient="left",
)

for dest in [OUT_MAIN / "fig_complexity_difficulty.png",
             OUT_GAP  / "g_complexity_difficulty.png"]:
    fig.save(str(dest), scale_factor=2.0)
    print(f"saved → {dest.relative_to(ROOT)}")

# Quick summary to stdout
print(f"\nInstances: {len(df)}")
print(f"Bottom-quartile ease threshold (hard): {q25:.3f}")
for b in ["hard", "not hard"]:
    sub = df[df["bucket"] == b]["n_ops"]
    print(f"  {b:8s}: n={len(sub):3d}  median_ops={sub.median():.0f}  "
          f"mean={sub.mean():.1f}  range=[{sub.min():.0f},{sub.max():.0f}]")
