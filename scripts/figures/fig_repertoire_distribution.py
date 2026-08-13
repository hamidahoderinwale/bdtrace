"""Per-trajectory repertoire distribution as a horizontal box plot.

For each trajectory, computes distinct_at_coverage: how many distinct BPE
motifs (by frequency, descending) are needed to cover 90% of that
trajectory's tokens. Displays one box per agent.

Reads:
    output/paper2_pilot/bpe_sequences.jsonl
Writes:
    output/figures/fig_agg_repertoire.png
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER

register()

BPE_FILE = ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
FIG_OUT  = ROOT / "output" / "figures"


def distinct_at_coverage(bpe_list: list[str], threshold: float = 0.90) -> int:
    """Return count of top motifs (by frequency) needed to cover >= threshold of tokens."""
    if not bpe_list:
        return 0
    counts = Counter(bpe_list)
    total = sum(counts.values())
    target = threshold * total
    cumulative = 0
    for i, (_, freq) in enumerate(counts.most_common(), start=1):
        cumulative += freq
        if cumulative >= target:
            return i
    return len(counts)


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # Load records
    per_traj: dict[str, list[int]] = {a: [] for a in AGENT_ORDER}
    with BPE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            agent = rec.get("agent")
            bpe   = rec.get("bpe", [])
            if agent not in per_traj:
                continue
            per_traj[agent].append(distinct_at_coverage(bpe))

    # Build summary dataframe with percentile stats per agent
    box_rows = []
    for agent in AGENT_ORDER:
        vals = per_traj[agent]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        box_rows.append({
            "agent":  agent,
            "q10":    float(np.percentile(arr, 10)),
            "q25":    float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "q75":    float(np.percentile(arr, 75)),
            "q90":    float(np.percentile(arr, 90)),
        })
    box_df = pd.DataFrame(box_rows)

    # Build box plot layers manually (whisker + IQR bar + median tick)
    # one layer set per agent so each is colored independently
    whisker_layers = [
        alt.Chart(box_df[box_df["agent"] == agent])
        .mark_rule(color=AGENT_COLORS[agent], strokeWidth=1.5)
        .encode(
            y=alt.Y("agent:N", sort=AGENT_ORDER, axis=alt.Axis(title=None)),
            x=alt.X("q10:Q", title="Distinct motifs at 90% coverage",
                    scale=alt.Scale(zero=True)),
            x2="q90:Q",
        )
        for agent in AGENT_ORDER if agent in box_df["agent"].values
    ]
    iqr_layers = [
        alt.Chart(box_df[box_df["agent"] == agent])
        .mark_bar(size=14, color=AGENT_COLORS[agent])
        .encode(
            y=alt.Y("agent:N", sort=AGENT_ORDER, axis=alt.Axis(title=None)),
            x=alt.X("q25:Q"),
            x2="q75:Q",
        )
        for agent in AGENT_ORDER if agent in box_df["agent"].values
    ]
    med_layers = [
        alt.Chart(box_df[box_df["agent"] == agent])
        .mark_tick(size=14, thickness=2, color="white")
        .encode(
            y=alt.Y("agent:N", sort=AGENT_ORDER, axis=alt.Axis(title=None)),
            x=alt.X("median:Q"),
        )
        for agent in AGENT_ORDER if agent in box_df["agent"].values
    ]

    chart = (
        alt.layer(*whisker_layers, *iqr_layers, *med_layers)
        .properties(
            width=260,
            height=130,
            title=alt.TitleParams(
                "Motifs needed at 90% coverage",
                fontSize=12,
                color="#111111",
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out_fig = FIG_OUT / "fig_agg_repertoire.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
