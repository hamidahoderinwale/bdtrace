"""Per-agent aggregate metrics.

For each of the four agents, compute:
    - Motif-distribution entropy (spread of usage)
    - Distinct motifs to cover 90% of the agent's tokens (repertoire width)
    - Mean canonical trajectory length (atoms)
    - Mean BPE-expressed length (motifs)
    - Mean compression ratio (bpe_len / canonical_len)
    - Novelty index: per-motif log-odds of this-agent's use relative to corpus
      baseline. Top positive = motifs this agent over-uses; top negative =
      motifs this agent under-uses.

Also cross-cut length by difficulty (does procedure length track the number
of agents that resolve the task?).

Outputs:
    output/paper2_pilot/aggregate_metrics.json
    output/paper2_pilot/aggregate_metrics.png          (4-panel combined)
    output/figures/fig_agg_entropy.png                 (individual)
    output/figures/fig_agg_repertoire.png              (individual)
    output/figures/fig_agg_length.png                  (individual)
    output/figures/fig_agg_compression.png             (individual)
    output/paper2_pilot/length_by_difficulty.png       (length cross-cut)
    output/paper2_pilot/novelty_top_motifs.png         (distinctive motifs per agent)

Usage:
    python -m analysis.preferences.aggregate_metrics
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import sys
import altair as alt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.theme import register, BLUE, ORANGE, GREEN, NEAR_BLACK, GRAY, AGENT_COLORS, AGENT_ORDER
register()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
FIG_OUT = PROJECT_ROOT / "output" / "figures"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

AGENTS = AGENT_ORDER  # ["Claude-3", "Claude-3.5", "GPT-4", "GPT-4o"]


def load_records() -> list[dict]:
    out = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_difficulty() -> dict[str, int]:
    out = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["instance_id"]] = int(row["n_resolved"])
    return out


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total) for c in counter.values() if c > 0
    )


def distinct_at_coverage(counter: Counter, coverage: float = 0.9) -> int:
    total = sum(counter.values())
    if total == 0:
        return 0
    sorted_counts = sorted(counter.values(), reverse=True)
    cum = 0
    for i, c in enumerate(sorted_counts, 1):
        cum += c
        if cum / total >= coverage:
            return i
    return len(sorted_counts)


def novelty_index(
    agent_counts: Counter,
    corpus_counts: Counter,
    motifs_only: bool = True,
    min_count: int = 5,
) -> list[tuple[str, float]]:
    """Log-odds per motif: log2( (agent_freq + eps) / (corpus_freq + eps) ).

    Positive = over-used by this agent; negative = under-used.
    """
    eps = 1.0
    agent_total = sum(agent_counts.values())
    corpus_total = sum(corpus_counts.values())
    if agent_total == 0 or corpus_total == 0:
        return []

    out = []
    for t, c_corp in corpus_counts.items():
        if motifs_only and "+" not in t:
            continue
        if c_corp < min_count:
            continue
        p_agent = (agent_counts.get(t, 0) + eps) / (agent_total + eps)
        p_corpus = (c_corp + eps) / (corpus_total + eps)
        logodds = math.log2(p_agent / p_corpus)
        out.append((t, logodds))
    out.sort(key=lambda kv: -kv[1])
    return out


def per_agent_metrics(records: list[dict]) -> dict:
    by_agent: dict[str, list[dict]] = {a: [] for a in AGENTS}
    for r in records:
        if r["agent"] in by_agent:
            by_agent[r["agent"]].append(r)

    counts_by_agent: dict[str, Counter] = {a: Counter() for a in AGENTS}
    for a, rs in by_agent.items():
        for r in rs:
            counts_by_agent[a].update(r["bpe"])

    corpus_counts: Counter = Counter()
    for c in counts_by_agent.values():
        corpus_counts.update(c)

    agent_metrics: dict[str, dict] = {}
    for a, rs in by_agent.items():
        if not rs:
            continue
        c_full = counts_by_agent[a]
        c_motifs = Counter({t: cnt for t, cnt in c_full.items() if "+" in t})
        canonical_lens = [r["canonical_length"] for r in rs]
        bpe_lens = [r["bpe_length"] for r in rs]
        compressions = [r["compression"] for r in rs]

        novelty = novelty_index(c_full, corpus_counts, motifs_only=True, min_count=5)

        agent_metrics[a] = {
            "n_trajectories": len(rs),
            "n_tokens_full": sum(c_full.values()),
            "n_tokens_motifs": sum(c_motifs.values()),
            "entropy_full_bits": entropy_bits(c_full),
            "entropy_motifs_bits": entropy_bits(c_motifs),
            "distinct_motifs_at_90pct": distinct_at_coverage(c_motifs, 0.9),
            "distinct_motifs_at_50pct": distinct_at_coverage(c_motifs, 0.5),
            "mean_canonical_length": float(np.mean(canonical_lens)),
            "median_canonical_length": float(np.median(canonical_lens)),
            "mean_bpe_length": float(np.mean(bpe_lens)),
            "median_bpe_length": float(np.median(bpe_lens)),
            "mean_compression": float(np.mean(compressions)),
            "median_compression": float(np.median(compressions)),
            "top_over_used_motifs": novelty[:8],
            "top_under_used_motifs": novelty[-8:][::-1],
        }
    return agent_metrics


def plot_metrics_summary(metrics: dict, out_path: Path) -> None:
    agents = [a for a in AGENTS if a in metrics]
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    color_scale = alt.Scale(
        domain=list(AGENT_COLORS.keys()),
        range=list(AGENT_COLORS.values()),
    )

    def _bar_panel(df_panel, title, y_title, width=180, height=200, y_domain=None):
        y_enc = alt.Y(
            "value:Q",
            axis=alt.Axis(title=y_title, domain=False, ticks=False, labelFontSize=10),
            **({"scale": alt.Scale(domain=y_domain)} if y_domain else {}),
        )
        base = alt.Chart(df_panel).encode(
            x=alt.X("agent:N", sort=AGENT_ORDER,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=10, labelAngle=-30)),
            y=y_enc,
            color=alt.Color("agent:N", scale=color_scale,
                            legend=alt.Legend(orient="bottom", title=None, symbolSize=80)),
        )
        bars = base.mark_bar(size=28)
        text = base.mark_text(dy=-8, fontSize=9).encode(
            text=alt.Text("value:Q", format=".2g"),
        )
        return (
            alt.layer(bars, text)
            .properties(
                width=width, height=height,
                title=alt.TitleParams(text=title, fontSize=11),
            )
        )

    # Panel 1: entropy
    df1 = pd.DataFrame([
        {"agent": a, "value": metrics[a]["entropy_motifs_bits"]} for a in agents
    ])
    panel1 = _bar_panel(
        df1,
        title="Motif entropy by agent (bits)",
        y_title="Entropy (bits)",
    )

    # Panel 2: distinct motifs at 90%
    df2 = pd.DataFrame([
        {"agent": a, "value": metrics[a]["distinct_motifs_at_90pct"]} for a in agents
    ])
    panel2 = _bar_panel(
        df2,
        title="Distinct motifs at 90% coverage",
        y_title="Number of motifs",
    )

    # Panel 3: grouped bar - canonical vs bpe
    rows3 = []
    for a in agents:
        rows3.append({"agent": a, "group": "Individual actions",
                      "value": metrics[a]["mean_canonical_length"]})
        rows3.append({"agent": a, "group": "Grouped patterns",
                      "value": metrics[a]["mean_bpe_length"]})
    df3 = pd.DataFrame(rows3)
    group_order = ["Individual actions", "Grouped patterns"]
    base3 = alt.Chart(df3).encode(
        x=alt.X("agent:N", sort=AGENT_ORDER,
                axis=alt.Axis(title=None, domain=False, ticks=False,
                              labelFontSize=10, labelAngle=-30)),
        xOffset=alt.XOffset("group:N", sort=group_order),
        y=alt.Y("value:Q",
                axis=alt.Axis(title="Mean steps per task",
                              domain=False, ticks=False, labelFontSize=10)),
        color=alt.Color("agent:N", scale=color_scale,
                        legend=alt.Legend(orient="bottom", title=None, symbolSize=80)),
        opacity=alt.Opacity("group:N",
                            scale=alt.Scale(domain=group_order, range=[1.0, 0.55]),
                            legend=None),
    )
    bars3 = base3.mark_bar(size=14)
    text3 = base3.mark_text(dy=-8, fontSize=9).encode(
        text=alt.Text("value:Q", format=".0f"),
    )
    panel3 = (
        alt.layer(bars3, text3)
        .properties(
            width=220, height=200,
            title=alt.TitleParams(
                text="Trajectory lengths",
                fontSize=11, color="#111111", anchor="start",
            ),
        )
    )

    # Panel 4: compression
    df4 = pd.DataFrame([
        {"agent": a, "value": metrics[a]["mean_compression"]} for a in agents
    ])
    panel4 = _bar_panel(
        df4,
        title="Mean compression ratio",
        y_title="BPE length / atom length",
        y_domain=[0, 1],
    )

    cfg = dict(strokeWidth=0)

    # Individual PNGs
    (panel1.configure_view(**cfg).configure_axis(grid=False)
     .save(str(FIG_OUT / "fig_agg_entropy.png"), scale_factor=2))
    (panel2.configure_view(**cfg).configure_axis(grid=False)
     .save(str(FIG_OUT / "fig_agg_repertoire.png"), scale_factor=2))
    (panel3.configure_view(**cfg).configure_axis(grid=False)
     .save(str(FIG_OUT / "fig_agg_length.png"), scale_factor=2))
    (panel4.configure_view(**cfg).configure_axis(grid=False)
     .save(str(FIG_OUT / "fig_agg_compression.png"), scale_factor=2))

    # Combined figure
    chart = (
        alt.hconcat(panel1, panel2, panel3, panel4, spacing=48)
        .properties(
            title=alt.TitleParams(
                text="Agent summary statistics (n = 1,162)",
                fontSize=13, color="#111111", anchor="start",
            )
        )
        .configure_view(**cfg)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def plot_length_by_difficulty(
    records: list[dict], difficulty: dict[str, int], out_path: Path
) -> None:
    diff_label = {0: "0/4", 1: "1/4", 2: "2/4", 3: "3/4"}
    panel_specs = [
        ("canonical_length", "Individual actions"),
        ("bpe_length", "Grouped action patterns"),
    ]

    rows = []
    for key, panel in panel_specs:
        for a in AGENTS:
            for d in [0, 1, 2, 3]:
                vals = [
                    r[key] for r in records
                    if r["agent"] == a and difficulty.get(r["instance_id"]) == d
                ]
                if vals:
                    rows.append({
                        "difficulty": d,
                        "difficulty_label": diff_label[d],
                        "agent": a,
                        "panel": panel,
                        "value": float(np.mean(vals)),
                    })
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=list(AGENT_COLORS.keys()),
        range=list(AGENT_COLORS.values()),
    )
    diff_order = ["0/4", "1/4", "2/4", "3/4"]
    panel_order = ["Individual actions", "Grouped action patterns"]

    base = alt.Chart(df).encode(
        x=alt.X("difficulty_label:O", sort=diff_order,
                axis=alt.Axis(title="Number of agents that solved the task",
                              domain=False, ticks=False, labelFontSize=10)),
        y=alt.Y("value:Q", axis=alt.Axis(title=None, domain=False, ticks=False,
                                         labelFontSize=10)),
        color=alt.Color("agent:N", scale=color_scale,
                        legend=alt.Legend(orient="bottom", title=None, symbolSize=80)),
        detail="agent:N",
    )

    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=60)

    chart = (
        alt.layer(lines, points)
        .properties(width=200, height=200)
        .facet(
            column=alt.Column(
                "panel:N",
                sort=panel_order,
                title=None,
                header=alt.Header(titleFontSize=11, labelFontSize=11,
                                  labelOrient="bottom"),
            )
        )
        .properties(
            title=alt.TitleParams(
                text="Trajectory length by task difficulty",
                fontSize=13, subtitleFontSize=11,
                color="#111111", subtitleColor="#888888", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_novelty_top(metrics: dict, out_path: Path, top_n: int = 6) -> None:
    agents = [a for a in AGENTS if a in metrics]

    def abbrev(m: str) -> str:
        parts = m.split("+")
        if len(parts) == 1:
            return parts[0]
        if len(set(parts)) == 1:
            return f"{parts[0]} x{len(parts)}"
        if len(parts) <= 2:
            return " -> ".join(parts)
        return f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)} steps)"

    color_scale = alt.Scale(
        domain=list(AGENT_COLORS.keys()),
        range=list(AGENT_COLORS.values()),
    )

    panels = []
    for a in agents:
        over = metrics[a]["top_over_used_motifs"][:top_n]
        if not over:
            continue
        rows = [
            {"motif": abbrev(m), "logodds": lo, "agent": a, "rank": i}
            for i, (m, lo) in enumerate(over)
        ]
        df = pd.DataFrame(rows)
        motif_order = [r["motif"] for r in rows]

        bars = alt.Chart(df).mark_bar().encode(
            y=alt.Y("motif:N", sort=motif_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=9)),
            x=alt.X("logodds:Q",
                    axis=alt.Axis(title="log2 odds vs corpus",
                                  domain=False, ticks=False, labelFontSize=10)),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )

        panel = (
            bars
            .properties(
                width=220, height=top_n * 24,
                title=alt.TitleParams(text=a, fontSize=11,
                                      color="#111111", anchor="start"),
            )
        )
        panels.append(panel)

    chart = (
        alt.hconcat(*panels, spacing=36)
        .properties(
            title=alt.TitleParams(
                text="Signature action patterns by agent",
                fontSize=13, color="#111111", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    records = load_records()
    difficulty = load_difficulty()

    metrics = per_agent_metrics(records)

    print("Per-agent metrics:")
    for a in AGENTS:
        if a not in metrics:
            continue
        m = metrics[a]
        print(f"\n{a}  (n={m['n_trajectories']})")
        print(f"  entropy (motifs)         = {m['entropy_motifs_bits']:.3f} bits")
        print(f"  distinct motifs @90%     = {m['distinct_motifs_at_90pct']}")
        print(f"  distinct motifs @50%     = {m['distinct_motifs_at_50pct']}")
        print(f"  mean canonical length    = {m['mean_canonical_length']:.1f} atoms")
        print(f"  mean BPE length          = {m['mean_bpe_length']:.1f} motifs")
        print(f"  mean compression         = {m['mean_compression']:.3f}")
        print(f"  top 3 over-used motifs:  "
              f"{[(mot, round(lo, 2)) for mot, lo in m['top_over_used_motifs'][:3]]}")

    (OUT / "aggregate_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics_summary(metrics, OUT / "aggregate_metrics.png")
    plot_length_by_difficulty(records, difficulty, OUT / "length_by_difficulty.png")
    plot_novelty_top(metrics, OUT / "novelty_top_motifs.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'aggregate_metrics.json'}")
    print(f"  {OUT / 'aggregate_metrics.png'}")
    print(f"  {FIG_OUT / 'fig_agg_entropy.png'}")
    print(f"  {FIG_OUT / 'fig_agg_repertoire.png'}")
    print(f"  {FIG_OUT / 'fig_agg_length.png'}")
    print(f"  {FIG_OUT / 'fig_agg_compression.png'}")
    print(f"  {OUT / 'length_by_difficulty.png'}")
    print(f"  {OUT / 'novelty_top_motifs.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
