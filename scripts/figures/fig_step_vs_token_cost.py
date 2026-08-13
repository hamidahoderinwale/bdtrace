"""Step share vs token share — dumbbell chart, one panel per agent.

For each (agent, stage): a MAGENTA dot at step fraction and a BLUE dot
at token fraction, connected by a thin line. The gap shows where step
count overstates or understates token cost. Agents without token data
(DARS, Agentless) show only step dots.

Reads:  output/paper2_pilot/bpe_sequences_extended.jsonl
        output/paper2_pilot/step_resources.json
Writes: output/figures/fig_step_vs_token_cost.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

import altair as alt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.theme import (
    register, NEAR_BLACK, MAGENTA, BLUE, OLIVE,
)
from _extended_pass_fail_df import SUBMISSION_TO_AGENT
register()

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STAGES = ["Explore", "Browse", "Edit", "Test", "Shell", "Finish"]
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}
THRESHOLD = 0.010

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


# ── Load data ─────────────────────────────────────────────────────────────────

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


# ── Build wide-format dumbbell data ──────────────────────────────────────────

rows = []
for agent in FAMILY_ORDER:
    if agent not in total_steps:
        continue
    n = total_steps[agent]
    t = agent_total_tok.get(agent, 0)
    has_tokens = t > 0

    for s in STAGES:
        sf = step_counts[agent].get(s, 0) / n
        tf = agent_stage_tok[agent].get(s, 0) / t if has_tokens else None

        tok_val = tf if tf is not None else 0.0
        if sf < THRESHOLD and tok_val < THRESHOLD:
            continue  # skip negligible stages

        rows.append({
            "agent":      agent,
            "stage":      s,
            "stage_idx":  STAGE_IDX[s],
            "sf":         sf if sf >= THRESHOLD else np.nan,
            "tf":         tf  if (tf is not None and tf >= THRESHOLD) else np.nan,
            "has_tokens": has_tokens,
        })

df = pd.DataFrame(rows)

# Alternating band flag: even stage_idx rows get a light gray background
df["band"] = df["stage_idx"] % 2 == 0

# Long-format for dots
dots_rows = []
for _, r in df.iterrows():
    if not np.isnan(r["sf"]):
        dots_rows.append({**r, "type": "Steps",  "frac": r["sf"]})
    if not np.isnan(r["tf"]):
        dots_rows.append({**r, "type": "Tokens", "frac": r["tf"]})

dots_df = pd.DataFrame(dots_rows)

agents_in_data = [a for a in FAMILY_ORDER if a in total_steps]

# ── Chart ─────────────────────────────────────────────────────────────────────

PANEL_W = 200
PANEL_H = 140
LABEL_FONT = 11

x_enc = alt.X(
    "frac:Q",
    scale=alt.Scale(domain=[0, 0.65]),
    axis=alt.Axis(
        title=None,
        format=".0%",
        values=[0, 0.2, 0.4, 0.6],
        labelFontSize=LABEL_FONT,
        tickCount=4,
    ),
)

y_enc = alt.Y(
    "stage:N",
    sort=alt.EncodingSortField(field="stage_idx", order="ascending"),
    axis=alt.Axis(title=None, labelFontSize=LABEL_FONT),
)

# Altair requires data at the top level for faceted layered charts.
# Use an empty base chart and pass data to the facet call.
base = alt.Chart()

band_layer = (
    base
    .transform_filter("datum.band")
    .mark_rect(color="#f0f0f0", opacity=1.0)
    .encode(
        y=alt.Y(
            "stage:N",
            sort=alt.EncodingSortField(field="stage_idx", order="ascending"),
        ),
    )
)

line_layer = (
    base.mark_rule(color="#C0C0C0", strokeWidth=2.0)
    .encode(
        y=alt.Y(
            "stage:N",
            sort=alt.EncodingSortField(field="stage_idx", order="ascending"),
        ),
        x=alt.X("sf:Q", scale=alt.Scale(domain=[0, 0.65])),
        x2="tf:Q",
    )
    .transform_filter("isValid(datum.sf) && isValid(datum.tf)")
)

dot_layer = (
    base
    .transform_fold(["sf", "tf"], as_=["_type", "frac"])
    .transform_filter("isValid(datum.frac)")
    .transform_calculate(
        "type_label",
        "datum._type === 'sf' ? 'Steps' : 'Tokens'"
    )
    .mark_point(size=110, filled=True, stroke="white", strokeWidth=1.0, opacity=1.0)
    .encode(
        y=alt.Y(
            "stage:N",
            sort=alt.EncodingSortField(field="stage_idx", order="ascending"),
            axis=alt.Axis(title=None, labelFontSize=LABEL_FONT),
        ),
        x=alt.X(
            "frac:Q",
            scale=alt.Scale(domain=[0, 0.65]),
            axis=alt.Axis(
                title=None,
                format=".0%",
                values=[0, 0.2, 0.4, 0.6],
                labelFontSize=LABEL_FONT,
            ),
        ),
        color=alt.Color(
            "type_label:N",
            sort=["Steps", "Tokens"],
            scale=alt.Scale(
                domain=["Steps", "Tokens"],
                range=[MAGENTA, BLUE],
            ),
            legend=alt.Legend(
                title=None,
                orient="bottom",
                direction="horizontal",
                labelFontSize=13,
                symbolSize=110,
            ),
        ),
    )
)

spec = alt.layer(band_layer, line_layer, dot_layer).properties(
    width=PANEL_W,
    height=PANEL_H,
)

chart = (
    spec
    .facet(
        data=df,
        facet=alt.Facet(
            "agent:N",
            sort=agents_in_data,
            header=alt.Header(
                title=None,
                labelFontSize=13,
                labelColor=NEAR_BLACK,
                labelOrient="top",
            ),
        ),
        columns=3,
        spacing=20,
    )
    .properties(
        title=alt.TitleParams(
            "Step share vs token share, by stage and agent",
            fontSize=13,
            color=NEAR_BLACK,
            anchor="start",
            offset=10,
        ),
    )
    .resolve_scale(color="shared", x="shared", y="independent")
    .configure_view(strokeWidth=0)
    .configure_axis(grid=False)
)

out = OUT / "fig_step_vs_token_cost.png"
chart.save(str(out), scale_factor=2)
print(f"Saved {out}")
