"""How well the strongest text baseline recovers a behavioural predicate from the
raw trace, against chance (0.5) and against procgrep's exact declarative
computation (1.0). One horizontal bar per row on a shared AUC axis: chance and
exact are their own bars (reference rows); lexical vs structural predicates are
colour-coded. Data from predicate_recovery.json.
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
rec = json.loads((ROOT / "output/paper2_pilot/predicate_recovery.json").read_text())

rows = [
    {"row": "chance", "auc": 0.5, "kind": "reference"},
    {"row": "edit-streak ≥5 (count)", "auc": rec["edit-streak >=5 (structural)"]["text_auc"], "kind": "structural"},
    {"row": "searched before edit (order)", "auc": rec["searched before first edit (order)"]["text_auc"], "kind": "structural"},
    {"row": "ran a test (lexical)", "auc": rec["ran a test (lexical)"]["text_auc"], "kind": "lexical"},
    {"row": "procgrep (exact)", "auc": 1.0, "kind": "reference"},
]
df = pd.DataFrame(rows)
order = df.sort_values("auc")["row"].tolist()
cs = alt.Scale(domain=["reference", "structural", "lexical"], range=[OLIVE, BLUE, COPPER])

bars = alt.Chart(df).mark_bar(height=18).encode(
    x=alt.X("auc:Q", title="recovery (AUC)", scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(domain=False, ticks=False, values=[0, 0.5, 1])),
    y=alt.Y("row:N", title=None, sort=order, axis=alt.Axis(domain=False, ticks=False)),
    color=alt.Color("kind:N", scale=cs, legend=None),
)
labels = alt.Chart(df).mark_text(align="left", dx=5, fontSize=11, color="#333333").encode(
    x="auc:Q", y=alt.Y("row:N", sort=order),
    text=alt.Text("auc:Q", format=".2f"),
)
chart = (bars + labels).properties(
    width=380, height=190,
    title=alt.TitleParams("Behavioural-predicate recovery from the raw trace",
                          fontWeight="normal", fontSize=14, anchor="start", dx=5),
)
out = ROOT / "docs/papers/figures/fig_predicate_recovery.png"
chart.save(str(out), scale_factor=2)
print("wrote", out)
