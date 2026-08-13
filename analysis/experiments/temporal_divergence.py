"""Temporal divergence: JSD between agent action distributions vs step number.

At each step t, computes the Jensen-Shannon divergence between the cumulative
action type distributions of within-family (GPT-4 x GPT-4o) and cross-family
(Claude x GPT) agent pairs. Shows when agents commit to divergent strategies.

Outputs:
    output/experiments/temporal_divergence.json
    output/experiments/temporal_divergence.png
"""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, GREEN
register()

OUT = ROOT / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

AGENT_SHORT = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}
WITHIN_PAIR  = ("GPT-4", "GPT-4o")
CROSS_PAIRS  = [("Claude-3.5", "GPT-4"), ("Claude-3.5", "GPT-4o")]


def seq_to_dist(actions: list[str], vocab: list[str]) -> np.ndarray:
    c = Counter(actions)
    total = max(sum(c.values()), 1)
    return np.array([c.get(v, 0) / total for v in vocab]) + 1e-9


def compute_jsd_at_steps(
    seq_a: list[str], seq_b: list[str], vocab: list[str], max_t: int
) -> list[float]:
    jsds = []
    for t in range(1, max_t + 1):
        d_a = seq_to_dist(seq_a[:t], vocab)
        d_b = seq_to_dist(seq_b[:t], vocab)
        jsds.append(float(jensenshannon(d_a, d_b) ** 2))
    return jsds


def main():
    df = pd.read_parquet(ROOT / "output/trajectories/lite_all_models.parquet")
    df["agent_short"] = df["model_id"].map(AGENT_SHORT)
    df = df.dropna(subset=["agent_short"])

    # Build vocabulary
    all_acts: list[str] = []
    for seq in df["action_sequence"]:
        all_acts.extend(str(seq).split())
    vocab = sorted(set(all_acts) - {"SUBMIT", "OTHER"})

    # For each instance, compute per-step JSD for within and cross-family pairs
    MAX_T = 40
    within_by_step:  list[list[float]] = [[] for _ in range(MAX_T)]
    cross_by_step:   list[list[float]] = [[] for _ in range(MAX_T)]

    instances = df["instance_id"].unique()
    for iid in instances:
        sub = df[df["instance_id"] == iid].set_index("agent_short")
        agents_present = set(sub.index)

        # Within-family
        a, b = WITHIN_PAIR
        if a in agents_present and b in agents_present:
            sa = str(sub.loc[a, "action_sequence"]).split()
            sb = str(sub.loc[b, "action_sequence"]).split()
            t_max = min(len(sa), len(sb), MAX_T)
            jsds = compute_jsd_at_steps(sa, sb, vocab, t_max)
            for t, jsd in enumerate(jsds):
                within_by_step[t].append(jsd)

        # Cross-family (average over two cross pairs)
        for a, b in CROSS_PAIRS:
            if a in agents_present and b in agents_present:
                sa = str(sub.loc[a, "action_sequence"]).split()
                sb = str(sub.loc[b, "action_sequence"]).split()
                t_max = min(len(sa), len(sb), MAX_T)
                jsds = compute_jsd_at_steps(sa, sb, vocab, t_max)
                for t, jsd in enumerate(jsds):
                    cross_by_step[t].append(jsd)

    rows = []
    for t in range(MAX_T):
        if len(within_by_step[t]) >= 10:
            rows.append({"step": t + 1, "group": "Within GPT family",
                         "jsd": np.mean(within_by_step[t]),
                         "n": len(within_by_step[t])})
        if len(cross_by_step[t]) >= 10:
            rows.append({"step": t + 1, "group": "Cross family",
                         "jsd": np.mean(cross_by_step[t]),
                         "n": len(cross_by_step[t])})

    result_df = pd.DataFrame(rows)
    (OUT / "temporal_divergence.json").write_text(
        json.dumps(result_df.to_dict(orient="records"), indent=2)
    )

    group_order = ["Cross family", "Within GPT family"]
    cscale = alt.Scale(domain=group_order, range=[BLUE, GREEN])

    line = (
        alt.Chart(result_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="Step in trajectory",
                    scale=alt.Scale(domain=[1, MAX_T]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[1, 10, 20, 30, 40])),
            y=alt.Y("jsd:Q", title="Mean Jensen-Shannon divergence",
                    scale=alt.Scale(domain=[0, result_df["jsd"].max() * 1.15]),
                    axis=alt.Axis(domain=False, ticks=False)),
            color=alt.Color("group:N", sort=group_order, scale=cscale,
                            legend=alt.Legend(title=None, orient="bottom")),
        )
    )
    pts = (
        alt.Chart(result_df)
        .mark_point(size=40, filled=True, strokeWidth=0)
        .encode(
            x="step:Q", y="jsd:Q",
            color=alt.Color("group:N", sort=group_order, scale=cscale, legend=None),
        )
    )
    chart = (
        (line + pts)
        .properties(
            title=alt.TitleParams(
                "Behavioral divergence accumulates over trajectory steps",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=420, height=240,
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT / "temporal_divergence.png"), scale_factor=2)
    print("Saved temporal_divergence.png")


if __name__ == "__main__":
    main()
