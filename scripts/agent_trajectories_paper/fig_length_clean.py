"""Redesigned resolution-vs-length figure (regression panel).

Models the binary outcome directly: P(resolve) ~ log10(trajectory length), so
length is NOT discretized into bins. Plots the fitted logistic curve with a
bootstrap 95% band, overlays the observed pass rate in coarse length groups as
sanity points, and direct-labels the fitted slope. Copper = teacher palette is
not used here; the curve is the benchmark-primary blue.

Clean styling, shared palette, no parentheticals or brackets in any plot text.

Run from repo root with the project venv:
    .venv/bin/python scripts/agent_trajectories_paper/fig_length_clean.py
"""
import json
import sys

import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from scripts.theme import register, BLUE, BLUE_D, OLIVE  # noqa: E402

register()

OUT = "docs/papers/figures"

rows = [json.loads(l) for l in open("output/paper2_pilot/bpe_sequences_extended.jsonl")]
pf = json.load(open("output/paper2_pilot/extended_pass_fail.json"))
res = {k: set(v.get("resolved", [])) for k, v in pf.items()}

L, y = [], []
for r in rows:
    s = res.get(r["submission"])
    if s is None:
        continue
    n = r.get("canonical_length", len(r["canonical"]))
    if n >= 1:
        L.append(n)
        y.append(int(r["instance_id"] in s))
L = np.array(L, float)
y = np.array(y)
X = np.log10(L).reshape(-1, 1)

clf = LogisticRegression().fit(X, y)
slope = clf.coef_[0, 0]

grid = np.linspace(np.log10(L.min()), np.log10(L.max()), 120)
p = clf.predict_proba(grid.reshape(-1, 1))[:, 1]

# bootstrap 95% band
rng = np.random.default_rng(42)
boot = []
for _ in range(400):
    idx = rng.integers(0, len(L), len(L))
    b = LogisticRegression().fit(X[idx], y[idx])
    boot.append(b.predict_proba(grid.reshape(-1, 1))[:, 1])
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
fit = pd.DataFrame({"length": 10 ** grid, "p": p, "lo": lo, "hi": hi})

# observed pass rate in coarse length groups, as sanity points (not the model)
edges = [(1, 10), (11, 20), (21, 30), (31, 50), (51, 90), (91, 9999)]
obs = []
for lo_e, hi_e in edges:
    m = (L >= lo_e) & (L <= hi_e)
    if m.sum() >= 20:
        obs.append({"length": float(np.median(L[m])), "rate": float(y[m].mean()), "n": int(m.sum())})
obs = pd.DataFrame(obs)

xscale = alt.Scale(type="log", domain=[L.min(), L.max()], nice=False)
yscale = alt.Scale(domain=[0, 0.55])

band = alt.Chart(fit).mark_area(opacity=0.18, color=BLUE).encode(
    x=alt.X("length:Q", scale=xscale,
            title="Trajectory length in canonical actions on a log scale",
            axis=alt.Axis(domain=False, ticks=False, values=[5, 10, 20, 50, 100, 200])),
    y=alt.Y("lo:Q", scale=yscale, title="Resolution probability",
            axis=alt.Axis(domain=False, ticks=False, format=".0%")),
    y2="hi:Q",
)
line = alt.Chart(fit).mark_line(color=BLUE_D, strokeWidth=2.5).encode(
    x=alt.X("length:Q", scale=xscale), y=alt.Y("p:Q", scale=yscale),
)
points = alt.Chart(obs).mark_point(color=OLIVE, size=46, filled=True, opacity=0.85).encode(
    x=alt.X("length:Q", scale=xscale), y=alt.Y("rate:Q", scale=yscale),
)

# direct labels in separated clear zones: the slope, which is the informative
# part, sits upper-right above the band; the observed-points label lower-left
slope_lab = alt.Chart(pd.DataFrame([
    {"length": 200, "p": 0.50, "t": f"slope {slope:+.2f} log-odds per decade"},
])).mark_text(color=BLUE_D, fontSize=12, align="right", fontWeight=500).encode(
    x=alt.X("length:Q", scale=xscale), y=alt.Y("p:Q", scale=yscale), text="t:N",
)
obs_lab = alt.Chart(pd.DataFrame([
    {"length": 6, "rate": 0.07, "t": "observed group rate"}
])).mark_text(color=OLIVE, fontSize=11, align="left").encode(
    x=alt.X("length:Q", scale=xscale), y=alt.Y("rate:Q", scale=yscale), text="t:N",
)

chart = (
    (band + line + points + slope_lab + obs_lab)
    .properties(width=440, height=270, title="Resolution probability by trajectory length")
    .configure_view(strokeWidth=0)
    .configure_title(fontSize=15, fontWeight=600, anchor="start", dy=-4)
    .configure_axis(labelFontSize=12, titleFontSize=13)
)
chart.save(f"{OUT}/fig_regression_length.png", scale_factor=2)
print(f"slope = {slope:+.3f} log-odds per decade")
print("wrote fig_regression_length.png")
