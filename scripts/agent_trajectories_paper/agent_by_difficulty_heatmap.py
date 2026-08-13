"""Agent pass rate by INDEPENDENT difficulty — replaces the circular per_bin_agent_mi.

Difficulty is leave-that-agent-out: for agent a on instance i, difficulty =
fraction of the OTHER agents that solved i. So an agent's own outcome never
enters its own difficulty axis (fixes the n_resolved circularity). Heatmap:
rows = agents, cols = difficulty band, cell = that agent's pass rate. Run from repo root.
"""
import sys
import json
from collections import defaultdict
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import altair as alt
from scripts.theme import register

register()
rows = [json.loads(l) for l in open("output/paper2_pilot/bpe_sequences_extended.jsonl")]
pf = json.load(open("output/paper2_pilot/extended_pass_fail.json"))
res = {k: set(v.get("resolved", [])) for k, v in pf.items()}

# (agent, instance) -> resolved bool; and per-instance solver set
solved = {}
inst_solvers = defaultdict(set)
agents = set()
for r in rows:
    s = res.get(r["submission"])
    if s is None:
        continue
    a, iid = r["agent"], r["instance_id"]
    ok = iid in s
    solved[(a, iid)] = ok
    agents.add(a)
    if ok:
        inst_solvers[iid].add(a)

n_agents = len(agents)
BANDS = [("hard (0–33%)", 0.0, 1 / 3), ("medium (33–66%)", 1 / 3, 2 / 3), ("easy (66–100%)", 2 / 3, 1.01)]
cells = defaultdict(lambda: [0, 0])  # (agent, band) -> [passed, n]
for (a, iid), ok in solved.items():
    others = n_agents - 1
    loo = len(inst_solvers[iid] - {a}) / others if others else 0.0
    band = next(b for b, lo, hi in BANDS if lo <= loo < hi)
    cells[(a, band)][1] += 1
    cells[(a, band)][0] += int(ok)

recs = []
for a in agents:
    for band, _, _ in BANDS:
        p, n = cells[(a, band)]
        recs.append({"agent": a, "difficulty": band, "pass_rate": (p / n if n else np.nan), "n": n})
df = pd.DataFrame(recs)
# order agents by overall pass rate
order = df.groupby("agent")["pass_rate"].mean().sort_values(ascending=False).index.tolist()

heat = alt.Chart(df).mark_rect().encode(
    x=alt.X("difficulty:N", sort=[b for b, _, _ in BANDS], title="Task difficulty (leave-agent-out solve rate)",
            axis=alt.Axis(domain=False, ticks=False, labelAngle=0)),
    y=alt.Y("agent:N", sort=order, title=None, axis=alt.Axis(domain=False, ticks=False)),
    color=alt.Color("pass_rate:Q", title="Pass rate", scale=alt.Scale(scheme="blues")),
)
labels = alt.Chart(df).mark_text(baseline="middle", fontSize=11).encode(
    x=alt.X("difficulty:N", sort=[b for b, _, _ in BANDS]),
    y=alt.Y("agent:N", sort=order),
    text=alt.Text("pass_rate:Q", format=".0%"),
    color=alt.condition("datum.pass_rate > 0.5", alt.value("white"), alt.value("black")),
)
chart = (heat + labels).properties(width=300, height=26 * n_agents,
                                    title="Pass rate by agent and independent task difficulty")
chart.save("docs/papers/figures/fig_agent_by_difficulty.png", scale_factor=2)
print("wrote fig_agent_by_difficulty.png")
print(df.pivot(index="agent", columns="difficulty", values="pass_rate").round(2).to_string())
