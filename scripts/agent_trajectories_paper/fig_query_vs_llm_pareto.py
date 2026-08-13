"""Quality-vs-cost Pareto for answering behavioural trajectory queries:
procgrep (exact structural query) vs LLM judges. y = mean F1 across the five
structural predicates; x = seconds per decision (log). procgrep sits at the
top-left corner (highest quality, ~10^6x lower latency, $0); the judges scatter
bottom-right. Reads query_vs_llm_full.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

sys.path.insert(0, ".")
from scripts.theme import BLUE, COPPER, GREEN, MAGENTA, OLIVE, register

register()
ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
d = json.loads((ROOT / "output/paper2_pilot/query_vs_llm_full.json").read_text())
pg_us = d["procgrep_us_per_decision"]

rows = [{"name": "procgrep", "f1": 1.0, "sec": pg_us / 1e6, "kind": "structural query"}]
for m, v in d["pareto"].items():
    if m == "procgrep" or v.get("mean_f1") is None:
        continue
    rows.append({"name": m.split("/")[-1], "f1": v["mean_f1"], "sec": v["mean_latency_s"],
                 "kind": "LLM judge"})
df = pd.DataFrame(rows)
cs = alt.Scale(domain=["structural query", "LLM judge"], range=[GREEN, COPPER])

pts = alt.Chart(df).mark_point(size=140, filled=True, opacity=0.9).encode(
    x=alt.X("sec:Q", scale=alt.Scale(type="log"),
            title="seconds per decision, log scale",
            axis=alt.Axis(domain=False, ticks=False)),
    y=alt.Y("f1:Q", scale=alt.Scale(domain=[0, 1]),
            title="mean F1 across structural predicates",
            axis=alt.Axis(domain=False, ticks=False, values=[0, 0.5, 1])),
    color=alt.Color("kind:N", scale=cs, legend=alt.Legend(title=None, orient="top-right")),
)
# Split labels into two vertical lanes so neighbours in the cluster don't collide.
_up = {"procgrep", "claude-sonnet-4-6", "claude-3.5-haiku", "gpt-4o"}
df["lane"] = df["name"].apply(lambda n: "up" if n in _up else "down")
lab_up = alt.Chart(df[df.lane == "up"]).mark_text(
    align="left", dx=8, dy=-9, fontSize=11, color="#333333").encode(x="sec:Q", y="f1:Q", text="name:N")
lab_dn = alt.Chart(df[df.lane == "down"]).mark_text(
    align="left", dx=8, dy=13, fontSize=11, color="#333333").encode(x="sec:Q", y="f1:Q", text="name:N")
chart = (pts + lab_up + lab_dn).properties(
    width=440, height=300,
    title=alt.TitleParams("Quality versus latency per behavioural-query decision",
                          fontWeight="normal", fontSize=14, anchor="start", dx=6),
)
out = ROOT / "docs/papers/figures/fig_query_vs_llm_pareto.png"
chart.save(str(out), scale_factor=2)
print("wrote", out)
