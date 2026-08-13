"""Canonical-alphabet JSD matrix (10 agents incl. distilled child) + harness decomposition.

Matches make_figs.py's method (base-2 Jensen-Shannon DISTANCE over per-trajectory
mean canonical distributions) but ADDS the SWE-agent-LM-32B child, so the figure
matches its caption. Also computes the scaffold/era/lineage decomposition that
grounds the harness-decomposition paragraph. Backs up the old figure. Run from repo root.
"""
import json
import shutil
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import altair as alt
from scipy.spatial.distance import jensenshannon
import sys
sys.path.insert(0, ".")
from scripts.theme import register

register()
rows = [json.loads(l) for l in open("output/paper2_pilot/bpe_sequences_extended.jsonl")]
child = [json.loads(l) for l in open("distillation_run/fingerprints_child.jsonl")]

alphabet = sorted({a for r in rows for a in r["canonical"]} | {a for r in child for a in r.get("native", [])})
idx = {a: i for i, a in enumerate(alphabet)}


def td(seq):
    v = np.zeros(len(alphabet))
    for a in seq:
        if a in idx:
            v[idx[a]] += 1
    s = v.sum()
    return v / s if s else v


by = {}
for r in rows:
    by.setdefault(r["agent"], []).append(td(r["canonical"]))
by["SWE-agent-LM-32B"] = [td(r.get("native", [])) for r in child]
means = {a: np.mean(v, axis=0) for a, v in by.items()}


def D(a, b):
    j = float(jensenshannon(means[a], means[b], base=2))
    return 0.0 if np.isnan(j) else j


ORDER = ["Claude-3", "Claude-3.5", "Claude-3.7-thinking", "SWE-agent-LM-32B", "Claude-4",
         "GPT-4", "GPT-4o", "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3"]
ORDER = [a for a in ORDER if a in means]

recs = [{"a": a, "b": b, "jsd": D(a, b)} for a in ORDER for b in ORDER]
df = pd.DataFrame(recs)
heat = alt.Chart(df).mark_rect().encode(
    x=alt.X("a:N", sort=ORDER, title=None, axis=alt.Axis(labelAngle=-40, domain=False, ticks=False)),
    y=alt.Y("b:N", sort=ORDER, title=None, axis=alt.Axis(domain=False, ticks=False)),
    color=alt.Color("jsd:Q", title="JSD", scale=alt.Scale(scheme="teals")),
).properties(width=360, height=360, title="Pairwise procedural JSD (canonical alphabet, 10 agents)")
OUT = Path("docs/papers/figures/fig_jsd_matrix_full_canonical.png")
if OUT.exists():
    shutil.copy(OUT, OUT.with_name("fig_jsd_matrix_full_canonical_prev9.png"))
heat.save(str(OUT), scale_factor=2)
print("wrote fig_jsd_matrix_full_canonical.png (10 agents incl child; backed up 9-agent -> _prev9)")

# decomposition
sc = D("Agentless+Claude-3.5", "Claude-3.5")
lin = D("Claude-3.7-thinking", "SWE-agent-LM-32B")
claude = ["Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4"]
gpt = ["GPT-4", "GPT-4o"]
era = [D(a, b) for a, b in combinations(claude, 2)] + [D(a, b) for a, b in combinations(gpt, 2)]
cd = sorted((D("SWE-agent-LM-32B", a), a) for a in means if a != "SWE-agent-LM-32B")
allp = sorted((D(a, b), a, b) for a, b in combinations(means, 2))
dec = {"scaffold_claude35_agentless_vs_sweagent": round(sc, 3),
       "lineage_claude37_vs_child": round(lin, 3),
       "era_within_family_sweagent_mean": round(float(np.mean(era)), 3),
       "era_range": [round(min(era), 3), round(max(era), 3)],
       "child_nearest": [(a, round(d, 3)) for d, a in cd[:3]],
       "closest_pairs": [(a, b, round(d, 3)) for d, a, b in allp[:5]],
       "farthest_pair": [allp[-1][1], allp[-1][2], round(allp[-1][0], 3)]}
Path("output/paper2_pilot/jsd_decomposition.json").write_text(json.dumps(dec, indent=2))
print(json.dumps(dec, indent=2))
