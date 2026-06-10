"""Two search paradigms head-to-head on the same retrieval task: procedural
search (k-NN by Jensen-Shannon distance between action-sequence fingerprints)
vs keyword search (cosine over TF-IDF of the raw trace text), plus chance.
Precision@5 on two retrieval axes -- find a trajectory by the same agent
(behaviour) vs from the same repo (topic). Reads retrieval_pk.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import altair as alt
import sys
sys.path.insert(0, ".")
from scripts.theme import register, BLUE, COPPER, OLIVE

register()
ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
raw = json.load(open(ROOT / "output/paper2_pilot/retrieval_pk.json"))

AXIS = {"agent": "same behaviour", "repo": "same topic"}
METHOD = {"fingerprint": "procedural", "bm25": "keyword", "chance": "chance"}
rows = []
for r in raw:
    if r["method"] == "chance":
        if r["k"] != 1:
            continue
        lo = hi = r["precision"]
    elif r["k"] == 5:
        lo, hi = r["lo"], r["hi"]
    else:
        continue
    rows.append({"axis": AXIS[r["label"]], "method": METHOD[r["method"]],
                 "p": r["precision"], "lo": lo, "hi": hi})
df = pd.DataFrame(rows)

morder = ["procedural", "keyword", "chance"]
cs = alt.Scale(domain=morder, range=[BLUE, COPPER, OLIVE])
xoff = alt.XOffset("method:N", sort=morder)

bars = alt.Chart(df).mark_bar(width=34).encode(
    x=alt.X("axis:N", title=None, sort=["same behaviour", "same topic"],
            axis=alt.Axis(domain=False, ticks=False, labelFontSize=12)),
    xOffset=xoff,
    y=alt.Y("p:Q", title="precision@5", scale=alt.Scale(domain=[0, 1.08], clamp=True),
            axis=alt.Axis(domain=False, ticks=False, values=[0, 0.5, 1])),
    color=alt.Color("method:N", scale=cs, sort=morder,
                    legend=alt.Legend(title=None, orient="top-left")),
)
ci = alt.Chart(df).mark_rule(strokeWidth=1.3, color="#444444").encode(
    x=alt.X("axis:N", sort=["same behaviour", "same topic"]), xOffset=xoff,
    y="lo:Q", y2="hi:Q",
)
labels = alt.Chart(df).mark_text(dy=-6, fontSize=10, color="#333333").encode(
    x=alt.X("axis:N", sort=["same behaviour", "same topic"]), xOffset=xoff,
    y="hi:Q", text=alt.Text("p:Q", format=".2f"),
)
chart = (bars + ci + labels).properties(
    width=300, height=240,
    title=alt.TitleParams("Retrieval precision by axis and search paradigm",
                          fontWeight="normal", fontSize=14, anchor="start", dx=5),
)
out = ROOT / "docs/papers/figures/fig_retrieval_pk.png"
chart.save(str(out), scale_factor=2)
print("wrote", out)
