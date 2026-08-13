"""Pass rate vs trajectory length — continuous logistic fit (replaces the binned plot).

Models the binary outcome directly: P(resolve) ~ log10(length), so length is NOT
discretized. Plots the fitted probability curve with a bootstrap 95% band; reports
the log-odds slope and endpoint pass rates. Run from repo root.
"""
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
import altair as alt
from sklearn.linear_model import LogisticRegression
from scripts.theme import register, BLUE

register()
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
        L.append(n); y.append(int(r["instance_id"] in s))
L = np.array(L, float); y = np.array(y)
X = np.log10(L).reshape(-1, 1)

clf = LogisticRegression().fit(X, y)
slope = clf.coef_[0, 0]              # log-odds per decade of length
grid = np.linspace(np.log10(L.min()), np.log10(L.max()), 100)
p = clf.predict_proba(grid.reshape(-1, 1))[:, 1]

# bootstrap 95% band
rng = np.random.default_rng(42)
boot = []
for _ in range(300):
    idx = rng.integers(0, len(L), len(L))
    b = LogisticRegression().fit(X[idx], y[idx])
    boot.append(b.predict_proba(grid.reshape(-1, 1))[:, 1])
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

df = pd.DataFrame({"length": 10 ** grid, "p": p, "lo": lo, "hi": hi})
band = alt.Chart(df).mark_area(opacity=0.2, color=BLUE).encode(
    x=alt.X("length:Q", scale=alt.Scale(type="log"), title="Trajectory length (canonical actions, log scale)",
            axis=alt.Axis(domain=False, ticks=False)),
    y=alt.Y("lo:Q", title="P(resolve)", scale=alt.Scale(domain=[0, 0.6]), axis=alt.Axis(domain=False, ticks=False)),
    y2="hi:Q")
line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2).encode(
    x=alt.X("length:Q", scale=alt.Scale(type="log")), y="p:Q")
chart = (band + line).properties(width=400, height=240,
                                 title="Resolution probability vs trajectory length")
chart.save("docs/papers/figures/fig_regression_length.png", scale_factor=2)
# endpoint readouts
p10 = clf.predict_proba([[np.log10(10)]])[0, 1]
p60 = clf.predict_proba([[np.log10(60)]])[0, 1]
print(f"slope (log-odds per decade) = {slope:.3f}")
print(f"P(resolve): length 10 -> {p10:.3f} | length 60 -> {p60:.3f}")
print("wrote fig_regression_length.png (logistic, unbinned)")
