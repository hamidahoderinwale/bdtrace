"""Token-cost divergence from step share, by stage.

For each (agent, stage), computes T% - S%:
  positive = stage consumes more of the token budget than its step share
  negative = stage is cheap relative to its step share

One dot per agent per stage; median line per stage. Answers:
  "Which stages are expensive per step and which are cheap?"

Reads:  output/paper2_pilot/bpe_sequences_extended.jsonl
        output/paper2_pilot/step_resources.json
        output/paper2_pilot/extended_pass_fail.json
Writes: output/figures/fig_cost_divergence.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.theme import (
    register, NEAR_BLACK,
    BLUE, GREEN, COPPER, MAGENTA, INDIGO, OLIVE,
    AGENT_COLORS, AGENT_ORDER,
)
from _extended_pass_fail_df import SUBMISSION_TO_AGENT
register()

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STAGES = ["Explore", "Browse", "Edit", "Test", "Shell", "Finish"]
STAGE_COLORS = {
    "Explore": BLUE, "Browse": GREEN, "Edit": COPPER,
    "Test": MAGENTA, "Shell": INDIGO, "Finish": OLIVE,
}
FAMILY_ORDER = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking",
    "Claude-4", "GPT-4", "GPT-4o",
    "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
]

def classify(atom: str) -> str:
    if atom.startswith("SEARCH"):               return "Explore"
    if atom.startswith(("OPEN","NAV","FIND")):  return "Browse"
    if atom.startswith(("EDIT","CREATE")):      return "Edit"
    if atom.startswith("RUN"):                  return "Test"
    if atom.startswith("SHELL_"):               return "Shell"
    if atom.startswith("SUBMIT"):               return "Finish"
    return "Other"


# ── Load step fractions ───────────────────────────────────────────────────────

step_counts: dict[str, Counter] = defaultdict(Counter)
total_steps: dict[str, int]     = defaultdict(int)

seq_path = ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl"
with seq_path.open() as f:
    for line in f:
        r = json.loads(line)
        ag, atoms = r["agent"], r["canonical"]
        n = max(len(atoms), 1)
        total_steps[ag] += n
        for a in atoms:
            step_counts[ag][classify(a)] += 1

# ── Load token fractions ──────────────────────────────────────────────────────

atoms_data = json.loads(
    (ROOT / "output/paper2_pilot/step_resources.json").read_text()
)["atoms"]

agent_stage_tok:  dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
agent_total_tok:  dict[str, float]            = defaultdict(float)

for atom, info in atoms_data.items():
    s = classify(atom)
    for ag, cnt in info.get("by_agent", {}).items():
        tok = cnt * info["mean_tokens_per_use"]
        agent_stage_tok[ag][s] += tok
        agent_total_tok[ag]    += tok


# ── Build divergence rows ─────────────────────────────────────────────────────

rows = []
for agent in FAMILY_ORDER:
    if agent not in total_steps:
        continue
    n_steps = total_steps[agent]
    n_tok   = agent_total_tok.get(agent, 0)
    if n_tok == 0:
        continue   # skip agents without token data (DARS, Agentless)
    for stage in STAGES:
        sf = step_counts[agent].get(stage, 0) / n_steps
        tf = agent_stage_tok[agent].get(stage, 0) / n_tok
        rows.append({
            "agent":      agent,
            "stage":      stage,
            "divergence": tf - sf,   # positive = more expensive than step share
        })

df = pd.DataFrame(rows)

# Median per stage for the reference tick
med_df = df.groupby("stage", as_index=False)["divergence"].median()

# ── Chart ─────────────────────────────────────────────────────────────────────

zero_rule = (
    alt.Chart(pd.DataFrame({"x": [0]}))
    .mark_rule(color="#BBBBBB", strokeWidth=1)
    .encode(x=alt.X("x:Q"))
)

# Add agent-family label for tooltip and shape encoding
AGENT_FAMILY = {
    "Claude-3": "Claude", "Claude-3.5": "Claude",
    "Claude-3.7-thinking": "Claude*", "Claude-4": "Claude*",
    "GPT-4": "GPT", "GPT-4o": "GPT",
    "DARS+R1": "Scaffold", "Agentless+Claude-3.5": "Scaffold",
    "Moatless+V3": "Scaffold",
}
df["family"] = df["agent"].map(AGENT_FAMILY)

x_scale = alt.Scale(domain=[-0.50, 0.35])

dots = (
    alt.Chart(df)
    .mark_point(size=70, opacity=0.9, strokeWidth=0.5, stroke="white", filled=True)
    .encode(
        y=alt.Y(
            "stage:N",
            sort=STAGES,
            axis=alt.Axis(title=None, labelFontSize=12),
        ),
        x=alt.X(
            "divergence:Q",
            scale=x_scale,
            axis=alt.Axis(
                title="Token budget share minus step share",
                format="+.0%",
                labelFontSize=10,
            ),
        ),
        color=alt.Color(
            "stage:N",
            sort=STAGES,
            scale=alt.Scale(
                domain=STAGES,
                range=[STAGE_COLORS[s] for s in STAGES],
            ),
            legend=None,
        ),
        shape=alt.Shape(
            "family:N",
            scale=alt.Scale(
                domain=["Claude", "Claude*", "GPT", "Scaffold"],
                range=["circle", "triangle-up", "square", "diamond"],
            ),
            legend=alt.Legend(
                title=None, orient="bottom",
                direction="horizontal",
                labelFontSize=10, symbolSize=70,
            ),
        ),
        tooltip=[
            alt.Tooltip("agent:N", title="Agent"),
            alt.Tooltip("stage:N", title="Stage"),
            alt.Tooltip("divergence:Q", title="T% − S%", format="+.1%"),
        ],
    )
)

median_ticks = (
    alt.Chart(med_df)
    .mark_tick(size=18, strokeWidth=3, opacity=1.0)
    .encode(
        y=alt.Y("stage:N", sort=STAGES),
        x=alt.X("divergence:Q", scale=x_scale),
        color=alt.Color(
            "stage:N",
            sort=STAGES,
            scale=alt.Scale(
                domain=STAGES,
                range=[STAGE_COLORS[s] for s in STAGES],
            ),
            legend=None,
        ),
    )
)

fig = (
    alt.layer(zero_rule, dots, median_ticks)
    .properties(
        width=400,
        height=220,
        title=alt.TitleParams(
            "Shell steps are cheap; Explore steps are expensive",
            fontSize=13,
            color=NEAR_BLACK,
            anchor="start",
        ),
    )
    .configure_view(strokeWidth=0)
    .configure_axis(grid=False)
)

out = OUT / "fig_cost_divergence.png"
fig.save(str(out), scale_factor=2)
print(f"Saved {out}")

# Print summary
print("\nMedian T% - S% per stage:")
for _, row in med_df.sort_values("divergence").iterrows():
    print(f"  {row['stage']:8s}: {row['divergence']:+.1%}")
