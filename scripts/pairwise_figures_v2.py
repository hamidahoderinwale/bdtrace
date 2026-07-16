#!/usr/bin/env python3
"""
Improved pairwise agent comparison figures.

Replaces the basic heatmap/histogram with richer representations
matching the project's existing figure style.

Figures:
  fig1_pairwise_strip.png       -- Jaccard distribution per agent pair (strip + box)
  fig2_divergence_scatter.png   -- per-instance: ease vs structural agreement
  fig3_agent_vocabulary.png     -- edit operation frequency per agent (ridgeline)
  fig4_instance_flow.png        -- history-flow: per-instance edit certs across agents

Usage:
  uv run python scripts/pairwise_figures_v2.py
"""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

# Add venv site-packages
_ROOT = Path(__file__).resolve().parent.parent
for _sp in (_ROOT / ".venv" / "lib").glob("python*/site-packages"):
    if str(_sp) not in sys.path:
        sys.path.insert(0, str(_sp))

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(_ROOT))

ROOT = _ROOT
PAIR_DIR = ROOT / "output" / "pairwise_agent_comparison"

# Wong palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
PINK   = "#CC79A7"
GRAY   = "#999999"
SKY    = "#56B4E9"
RED    = "#D55E00"
YELLOW = "#F0E442"

# Color cycle for any number of agents
_PALETTE = [BLUE, ORANGE, GREEN, PINK, SKY, RED, YELLOW, GRAY,
            "#332288", "#88CCEE", "#44AA99", "#117733", "#DDCC77",
            "#CC6677", "#AA4499", "#882255"]


def _agent_color(agents: list[str]) -> dict[str, str]:
    return {a: _PALETTE[i % len(_PALETTE)] for i, a in enumerate(agents)}


def _short_name(name: str) -> str:
    """Generate a compact label from full agent name."""
    return name


def load_data():
    with open(PAIR_DIR / "agent_patches.json") as f:
        patches = json.load(f)

    # Try loading ease from leaderboard msgpack
    ease = {}
    try:
        import msgpack
        with open(ROOT / "output" / "leaderboard" / "lite_results.msgpack", "rb") as f:
            lb = msgpack.unpack(f, raw=False)
        votes = {}
        for agent_data in lb.values():
            for iid, passed in agent_data.items():
                votes.setdefault(iid, []).append(passed)
        ease = {iid: float(np.mean(v)) for iid, v in votes.items()}
    except Exception:
        # Fallback: compute ease from the agents we have
        all_iids = set()
        for certs in patches.values():
            all_iids.update(certs.keys())
        n_agents = len(patches)
        for iid in all_iids:
            n_solving = sum(1 for certs in patches.values() if iid in certs)
            ease[iid] = n_solving / n_agents

    return patches, ease


def compute_pairwise_jaccards(patches):
    """Compute Jaccard for every (agent_pair, instance) combination."""
    agents = list(patches.keys())
    rows = []
    for a1, a2 in combinations(agents, 2):
        for iid in patches[a1]:
            if iid in patches[a2] and patches[a1][iid] and patches[a2][iid]:
                s1 = set(patches[a1][iid])
                s2 = set(patches[a2][iid])
                if s1 or s2:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                    pair_label = f"{_short_name(a1)} vs {_short_name(a2)}"
                    rows.append({
                        "agent_1": a1, "agent_2": a2,
                        "pair": pair_label,
                        "instance_id": iid,
                        "jaccard": jaccard,
                    })
    return pd.DataFrame(rows)


# ── Fig 1: Strip + boxplot per agent pair ────────────────────────────

def fig1_pairwise_strip(jdf, max_pairs=20):
    """Jaccard distribution per agent pair, strip marks + box."""
    # Keep top pairs by co-solved count for readability
    pair_counts = jdf.groupby("pair").size().to_dict()
    top_pairs = sorted(pair_counts, key=pair_counts.get, reverse=True)[:max_pairs]
    jdf = jdf[jdf["pair"].isin(top_pairs)].copy()
    pair_counts = {p: pair_counts[p] for p in top_pairs}
    jdf["pair_label"] = jdf["pair"].map(lambda p: f"{p}  (n={pair_counts[p]})")

    pair_order = (
        jdf.groupby("pair_label")["jaccard"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    n_pairs = len(pair_order)

    # Boxplot layer
    box = alt.Chart(jdf).mark_boxplot(
        color=GRAY, opacity=0.35, median=alt.MarkConfig(color="#333333"), size=20
    ).encode(
        y=alt.Y("pair_label:N", sort=pair_order,
                 axis=alt.Axis(title=None, labelFontSize=10, labelLimit=280)),
        x=alt.X("jaccard:Q", scale=alt.Scale(domain=[0, 1.05]),
                 axis=alt.Axis(title="Jaccard similarity of edit certificates",
                               titleFontSize=10, values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
    )

    # Strip (jittered dots)
    strip = alt.Chart(jdf).mark_circle(size=50, opacity=0.6).encode(
        y=alt.Y("pair_label:N", sort=pair_order),
        x=alt.X("jaccard:Q"),
        color=alt.value(BLUE),
        tooltip=["instance_id:N", "jaccard:Q", "pair:N"],
    )

    fig = (box + strip).properties(
        width=480, height=max(350, n_pairs * 38),
        title=alt.TitleParams(
            "Structural similarity of fixes on co-solved instances",
            fontSize=12, fontWeight="normal", anchor="start",
        )
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(PAIR_DIR / "fig1_pairwise_strip.png"), scale_factor=2)
    print("Saved fig1_pairwise_strip.png")


# ── Fig 2: Per-instance divergence scatter ───────────────────────────

def fig2_divergence_scatter(patches, ease):
    """For each instance solved by 2+ agents: x = ease, y = mean structural agreement."""
    agents = list(patches.keys())
    instance_data = {}

    for iid in set().union(*[set(p.keys()) for p in patches.values()]):
        agent_certs = {}
        for a in agents:
            if iid in patches[a] and patches[a][iid]:
                agent_certs[a] = set(patches[a][iid])

        if len(agent_certs) < 2:
            continue

        jaccards = []
        for a1, a2 in combinations(agent_certs.keys(), 2):
            s1, s2 = agent_certs[a1], agent_certs[a2]
            if s1 or s2:
                jaccards.append(len(s1 & s2) / len(s1 | s2))

        instance_data[iid] = {
            "instance_id": iid,
            "ease": ease.get(iid, np.nan),
            "mean_jaccard": np.mean(jaccards),
            "n_agents": len(agent_certs),
        }

    df = pd.DataFrame(instance_data.values())

    points = alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X("ease:Q", scale=alt.Scale(domain=[0, 1]),
                 axis=alt.Axis(title="Agent ease (fraction of 84 agents solving)", titleFontSize=10)),
        y=alt.Y("mean_jaccard:Q", scale=alt.Scale(domain=[0, 1]),
                 axis=alt.Axis(title="Structural agreement among agents that solved it", titleFontSize=10)),
        size=alt.Size("n_agents:Q", scale=alt.Scale(range=[30, 200]),
                       legend=alt.Legend(title="Agents solving")),
        color=alt.value(BLUE),
        tooltip=["instance_id:N", "ease:Q", "mean_jaccard:Q", "n_agents:Q"],
    )

    # Regression trend line
    trend = alt.Chart(df).mark_line(color=ORANGE, strokeWidth=2, strokeDash=[6, 3]).transform_regression(
        "ease", "mean_jaccard"
    ).encode(
        x="ease:Q",
        y="mean_jaccard:Q",
    )

    # Correlation annotation
    r = df[["ease", "mean_jaccard"]].corr().iloc[0, 1]
    r_text = alt.Chart(pd.DataFrame({
        "x": [0.05], "y": [0.05], "text": [f"r = {r:.2f}"]
    })).mark_text(fontSize=11, color=ORANGE, fontWeight="normal", align="left").encode(
        x=alt.X("x:Q"), y=alt.Y("y:Q"), text="text:N"
    )

    fig = (points + trend + r_text).properties(
        width=400, height=350,
        title=alt.TitleParams(
            "Do easy instances have more structural agreement?",
            fontSize=12, fontWeight="normal", anchor="start",
        )
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(PAIR_DIR / "fig2_divergence_scatter.png"), scale_factor=2)
    print(f"Saved fig2_divergence_scatter.png  (r={r:.3f}, n={len(df)})")
    return r


# ── Fig 3: Agent vocabulary ridgeline ────────────────────────────────

def fig3_agent_vocabulary(patches, max_agents=8):
    """Edit operation frequency per agent, shown as dot/bar chart."""
    # Limit to top agents by number of certs for readability
    all_agents = sorted(patches.keys(),
                        key=lambda a: sum(1 for v in patches[a].values() if v),
                        reverse=True)
    agents = all_agents[:max_agents]
    colors = _agent_color(agents)

    # Count edit ops per agent
    agent_op_counts = {}
    for agent in agents:
        counter = Counter()
        n_instances = 0
        for iid, cert in patches[agent].items():
            if cert:
                counter.update(cert)
                n_instances += 1
        agent_op_counts[agent] = {op: count / max(n_instances, 1) for op, count in counter.items()}

    # Find top 15 ops by total frequency across agents
    total = Counter()
    for counts in agent_op_counts.values():
        for op, freq in counts.items():
            total[op] += freq
    top_ops = [op for op, _ in total.most_common(15)]

    rows = []
    for agent in agents:
        for op in top_ops:
            rows.append({
                "agent": _short_name(agent),
                "operation": op.replace("_", " "),
                "frequency": agent_op_counts[agent].get(op, 0),
                "color": colors[agent],
            })
    top_ops = [op.replace("_", " ") for op in top_ops]

    df = pd.DataFrame(rows)

    # Faceted dot plot — one row per agent, dots for each operation
    chart = alt.Chart(df).mark_circle(size=80, opacity=0.85).encode(
        x=alt.X("operation:N", sort=top_ops,
                 axis=alt.Axis(labelAngle=-45, title=None, labelFontSize=8)),
        y=alt.Y("frequency:Q",
                 axis=alt.Axis(title="Frequency per solved instance", titleFontSize=9)),
        color=alt.Color("agent:N",
                         scale=alt.Scale(
                             domain=[_short_name(a) for a in agents],
                             range=[colors[a] for a in agents]),
                         legend=alt.Legend(title=None, orient="top")),
        tooltip=["agent:N", "operation:N", "frequency:Q"],
    )

    # Connect dots per agent with lines
    lines = alt.Chart(df).mark_line(opacity=0.3, strokeWidth=1).encode(
        x=alt.X("operation:N", sort=top_ops),
        y=alt.Y("frequency:Q"),
        color=alt.Color("agent:N",
                         scale=alt.Scale(
                             domain=[_short_name(a) for a in agents],
                             range=[colors[a] for a in agents]),
                         legend=None),
    )

    fig = (lines + chart).properties(
        width=550, height=280,
        title=alt.TitleParams(
            "Edit operation vocabulary per agent, top 15 operations",
            fontSize=12, fontWeight="normal", anchor="start",
        )
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(PAIR_DIR / "fig3_agent_vocabulary.png"), scale_factor=2)
    print("Saved fig3_agent_vocabulary.png")


# ── Fig 4: Instance-level history flow ───────────────────────────────

def fig4_instance_flow(patches, ease, max_agents=8):
    """
    For a selection of interesting instances, show each agent's edit certificate
    as a row of colored blocks. Shared operations align vertically.
    """
    # Limit to top agents for readability
    all_agents = sorted(patches.keys(),
                        key=lambda a: sum(1 for v in patches[a].values() if v),
                        reverse=True)
    agents = all_agents[:max_agents]
    colors = _agent_color(agents)

    # Find instances solved by the most agents
    instance_agent_count = {}
    for iid in set().union(*[set(p.keys()) for p in patches.values()]):
        n = sum(1 for a in agents if iid in patches[a] and patches[a][iid])
        if n >= 3:
            instance_agent_count[iid] = n

    # Pick 8 instances: mix of high/low ease, high agent overlap
    sorted_instances = sorted(instance_agent_count.keys(),
                               key=lambda iid: (-instance_agent_count[iid], ease.get(iid, 0)))
    selected = sorted_instances[:8]

    # Build data: for each instance x agent x operation, is it present?
    rows = []
    all_ops = set()
    for iid in selected:
        for agent in agents:
            if iid in patches[agent] and patches[agent][iid]:
                for op in patches[agent][iid]:
                    all_ops.add(op)

    # Order operations by frequency across selected instances
    op_counter = Counter()
    for iid in selected:
        for agent in agents:
            if iid in patches[agent] and patches[agent][iid]:
                op_counter.update(patches[agent][iid])
    op_order = [op for op, _ in op_counter.most_common()]

    for iid in selected:
        iid_short = iid.split("__")[1] if "__" in iid else iid
        e = ease.get(iid, 0)
        for agent in agents:
            if iid in patches[agent] and patches[agent][iid]:
                for op in op_order:
                    rows.append({
                        "instance": f"{iid_short} (ease={e:.0%})",
                        "agent": _short_name(agent),
                        "operation": op.replace("_", " "),
                        "present": 1 if op in patches[agent][iid] else 0,
                        "ease": e,
                    })
    op_order = [op.replace("_", " ") for op in op_order]

    df = pd.DataFrame(rows)
    df = df[df["present"] == 1]

    instance_order = [f"{iid.split('__')[1] if '__' in iid else iid} (ease={ease.get(iid, 0):.0%})"
                      for iid in selected]
    agent_order = [_short_name(a) for a in agents]

    chart = alt.Chart(df).mark_rect(stroke="white", strokeWidth=0.5).encode(
        x=alt.X("operation:N", sort=op_order,
                 axis=alt.Axis(labelAngle=-45, title=None, labelFontSize=7)),
        y=alt.Y("agent:N", sort=agent_order,
                 axis=alt.Axis(title=None, labelFontSize=9)),
        color=alt.Color("agent:N",
                         scale=alt.Scale(
                             domain=agent_order,
                             range=[colors[a] for a in agents]),
                         legend=None),
        tooltip=["instance:N", "agent:N", "operation:N"],
    ).properties(
        width=400, height=80,
    ).facet(
        row=alt.Row("instance:N", sort=instance_order,
                     header=alt.Header(labelFontSize=9, labelAngle=0, labelAlign="left",
                                       title=None)),
    ).resolve_scale(
        x="shared",
    ).properties(
        title=alt.TitleParams(
            "Edit operations per agent across 8 co-solved instances",
            fontSize=12, fontWeight="normal", anchor="start",
        )
    )

    chart.save(str(PAIR_DIR / "fig4_instance_flow.png"), scale_factor=2)
    print("Saved fig4_instance_flow.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    patches, ease = load_data()

    print("\nComputing pairwise Jaccards...")
    jdf = compute_pairwise_jaccards(patches)
    print(f"  {len(jdf)} pairwise comparisons")

    print("\nBuilding figures...")
    fig1_pairwise_strip(jdf)
    r = fig2_divergence_scatter(patches, ease)
    fig3_agent_vocabulary(patches)
    fig4_instance_flow(patches, ease)

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Median Jaccard: {jdf['jaccard'].median():.2f}")
    print(f"  Identical (Jaccard=1.0): {(jdf['jaccard'] == 1.0).mean():.1%}")
    print(f"  Very different (Jaccard<0.3): {(jdf['jaccard'] < 0.3).mean():.1%}")
    print(f"  Ease vs agreement correlation: r={r:.3f}")
    print(f"\nFigures in {PAIR_DIR}")


if __name__ == "__main__":
    main()
