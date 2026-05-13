"""Per-agent aggregate metrics on the extended 8-submission corpus.

Mirrors aggregate_metrics.py but reads bpe_sequences_extended.jsonl. Computes
per-submission entropy, repertoire@90%, mean lengths, mean compression. No
difficulty bucketing here (length-by-difficulty extension is gated on the
difficulty-definition decision).

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
Writes:
    output/paper2_pilot/aggregate_metrics_extended.json
    output/figures/fig_agg_entropy_extended.png
    output/figures/fig_agg_repertoire_extended.png
    output/figures/fig_agg_length_extended.png
    output/figures/fig_agg_compression_extended.png
"""
from __future__ import annotations
import json, math, sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import (
    register, GREEN, BLUE, MAGENTA, COPPER, OLIVE,
    GREEN_D, BLUE_D, MAGENTA_D,
    INDIGO, VIOLET, SIENNA,
)
register()

SEQ = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
DIVERSITY_PATH = ROOT / "output" / "paper2_pilot" / "task_diversity_extended.jsonl"
OUT_JSON = ROOT / "output" / "paper2_pilot" / "aggregate_metrics_extended.json"
FIG_OUT = ROOT / "output" / "figures"
OUT_DIR = ROOT / "output" / "paper2_pilot"

AGENT_ORDER_EXT = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4", "GPT-4", "GPT-4o",
    "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
]
# Canonical palette assignment matching scripts/theme.py AGENT_COLORS so
# Claude-3.7-thinking / Claude-4 / Moatless render identically across all
# dashboard figures.
AGENT_COLORS_EXT = {
    "Claude-3":              COPPER,
    "Claude-3.5":             GREEN,
    "Claude-3.7-thinking":    INDIGO,
    "Claude-4":               VIOLET,
    "GPT-4":                  BLUE,
    "GPT-4o":                 MAGENTA,
    "DARS+R1":                MAGENTA_D,
    "Agentless+Claude-3.5":   BLUE_D,
    "Moatless+V3":            SIENNA,
}


def load_records() -> list[dict]:
    out = []
    with SEQ.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


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
    """log2((agent_freq + eps) / (corpus_freq + eps)) per motif.
    Positive = over-used by this agent; negative = under-used. Sorted desc."""
    eps = 1.0
    agent_total = sum(agent_counts.values())
    corpus_total = sum(corpus_counts.values())
    if agent_total == 0 or corpus_total == 0:
        return []
    out: list[tuple[str, float]] = []
    for t, c_corp in corpus_counts.items():
        if motifs_only and "+" not in t:
            continue
        if c_corp < min_count:
            continue
        p_agent = (agent_counts.get(t, 0) + eps) / (agent_total + eps)
        p_corpus = (c_corp + eps) / (corpus_total + eps)
        out.append((t, math.log2(p_agent / p_corpus)))
    out.sort(key=lambda kv: -kv[1])
    return out


def per_agent_metrics(records: list[dict]) -> dict:
    by_agent: dict[str, list[dict]] = {a: [] for a in AGENT_ORDER_EXT}
    for r in records:
        a = r.get("agent")
        if a in by_agent:
            by_agent[a].append(r)

    counts_by_agent: dict[str, Counter] = {a: Counter() for a in AGENT_ORDER_EXT}
    for a, rs in by_agent.items():
        for r in rs:
            counts_by_agent[a].update(r["bpe"])

    corpus_counts: Counter = Counter()
    for c in counts_by_agent.values():
        corpus_counts.update(c)

    metrics: dict[str, dict] = {}
    for a, rs in by_agent.items():
        if not rs:
            continue
        c_full = counts_by_agent[a]
        c_motifs = Counter({t: cnt for t, cnt in c_full.items() if "+" in t})
        canonical_lens = [r["canonical_length"] for r in rs]
        bpe_lens = [r["bpe_length"] for r in rs]
        compressions = [r["compression"] for r in rs]
        novelty = novelty_index(c_full, corpus_counts, motifs_only=True, min_count=5)
        metrics[a] = {
            "n_trajectories":           len(rs),
            "n_tokens_full":            int(sum(c_full.values())),
            "n_tokens_motifs":          int(sum(c_motifs.values())),
            "entropy_full_bits":        round(entropy_bits(c_full), 4),
            "entropy_motifs_bits":      round(entropy_bits(c_motifs), 4),
            "distinct_motifs_at_90pct": distinct_at_coverage(c_motifs, 0.9),
            "distinct_motifs_at_50pct": distinct_at_coverage(c_motifs, 0.5),
            "mean_canonical_length":    round(float(np.mean(canonical_lens)), 3),
            "median_canonical_length":  round(float(np.median(canonical_lens)), 3),
            "mean_bpe_length":          round(float(np.mean(bpe_lens)), 3),
            "median_bpe_length":        round(float(np.median(bpe_lens)), 3),
            "mean_compression":         round(float(np.mean(compressions)), 4),
            "median_compression":       round(float(np.median(compressions)), 4),
            "top_over_used_motifs":     [(m, round(lo, 4)) for m, lo in novelty[:8]],
            "top_under_used_motifs":    [(m, round(lo, 4)) for m, lo in novelty[-8:][::-1]],
        }
    return metrics


def _bar_panel(df, title, y_title, y_domain=None, width=320, height=220):
    color_scale = alt.Scale(
        domain=list(AGENT_COLORS_EXT.keys()),
        range=list(AGENT_COLORS_EXT.values()),
    )
    y_enc = alt.Y(
        "value:Q",
        axis=alt.Axis(title=y_title, domain=False, ticks=False, labelFontSize=10),
        **({"scale": alt.Scale(domain=y_domain)} if y_domain else {}),
    )
    base = alt.Chart(df).encode(
        x=alt.X("agent:N", sort=AGENT_ORDER_EXT,
                axis=alt.Axis(title=None, domain=False, ticks=False,
                              labelFontSize=9, labelAngle=-35)),
        y=y_enc,
        color=alt.Color("agent:N", scale=color_scale, legend=None),
    )
    bars = base.mark_bar(size=22)
    text = base.mark_text(dy=-8, fontSize=9).encode(
        text=alt.Text("value:Q", format=".2g"),
    )
    return (
        alt.layer(bars, text)
        .properties(
            width=width, height=height,
            title=alt.TitleParams(text=title, fontSize=11,
                                  color="#111111", anchor="start"),
        )
    )


def load_difficulty_extended() -> dict[str, int]:
    """instance_id -> n_resolved (out of 9 agents that attempted it).

    Built by analysis.preferences.task_diversity_extended. If the JSONL
    isn't present, length_by_difficulty is skipped with a warning."""
    if not DIVERSITY_PATH.exists():
        return {}
    out: dict[str, int] = {}
    for line in DIVERSITY_PATH.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["instance_id"]] = int(r["n_resolved"])
    return out


def plot_length_by_difficulty(
    records: list[dict],
    difficulty: dict[str, int],
    out_path: Path,
) -> None:
    """Trajectory length (canonical and BPE) vs how many of 9 agents resolved
    the task. x-axis bucketed into 6 bins to keep the line plot readable
    with 9 series; 0/9 and 9/9 are kept as singletons because they carry
    the strongest signal.
    """
    if not difficulty:
        print("  WARN: task_diversity_extended.jsonl missing; "
              "skipping length_by_difficulty")
        return

    def bucket(n: int) -> str:
        if n == 0:
            return "0/9"
        if n == 9:
            return "9/9"
        if n <= 2:
            return "1-2/9"
        if n <= 4:
            return "3-4/9"
        if n <= 6:
            return "5-6/9"
        return "7-8/9"

    bucket_order = ["0/9", "1-2/9", "3-4/9", "5-6/9", "7-8/9", "9/9"]
    panel_specs = [
        ("canonical_length", "Individual actions"),
        ("bpe_length", "Grouped action patterns"),
    ]

    rows: list[dict] = []
    for key, panel in panel_specs:
        for a in AGENT_ORDER_EXT:
            for b in bucket_order:
                vals = [
                    r[key] for r in records
                    if r.get("agent") == a
                    and bucket(difficulty.get(r["instance_id"], -1)) == b
                    and difficulty.get(r["instance_id"], -1) >= 0
                ]
                if vals:
                    rows.append({
                        "difficulty_label": b,
                        "agent": a,
                        "panel": panel,
                        "value": float(np.mean(vals)),
                        "n": len(vals),
                    })
    if not rows:
        print("  WARN: no overlap between bpe sequences and difficulty data")
        return
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=list(AGENT_COLORS_EXT.keys()),
        range=list(AGENT_COLORS_EXT.values()),
    )

    base = alt.Chart(df).encode(
        x=alt.X("difficulty_label:O", sort=bucket_order,
                axis=alt.Axis(
                    title="Number of the 9 agents that resolved the task",
                    domain=False, ticks=False, labelFontSize=10,
                )),
        y=alt.Y("value:Q",
                axis=alt.Axis(title=None, domain=False, ticks=False,
                              labelFontSize=10)),
        color=alt.Color(
            "agent:N", scale=color_scale,
            legend=alt.Legend(orient="bottom", title=None, symbolSize=80,
                              columns=5),
        ),
        detail="agent:N",
        tooltip=["agent", "difficulty_label", alt.Tooltip("value:Q", format=".1f"), "n"],
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=50, filled=True)

    chart = (
        alt.layer(lines, points)
        .properties(width=220, height=200)
        .facet(
            column=alt.Column(
                "panel:N",
                sort=[p[1] for p in panel_specs],
                title=None,
                header=alt.Header(titleFontSize=11, labelFontSize=11,
                                  labelOrient="bottom"),
            )
        )
        .properties(
            title=alt.TitleParams(
                text="Trajectory length by task difficulty",
                subtitle="9-agent extended corpus; buckets coarsened for readability",
                fontSize=13, subtitleFontSize=10,
                color="#111111", subtitleColor="#888888", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def plot_novelty_top(metrics: dict, out_path: Path, top_n: int = 6) -> None:
    """Signature action patterns per agent: top-N over-used motifs by log2-odds.
    Wrapped to 5 panels per row at 9 agents to keep the figure scannable."""
    agents = [a for a in AGENT_ORDER_EXT if a in metrics]
    if not agents:
        return

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
        domain=list(AGENT_COLORS_EXT.keys()),
        range=list(AGENT_COLORS_EXT.values()),
    )

    panels = []
    for a in agents:
        over = metrics[a].get("top_over_used_motifs", [])[:top_n]
        if not over:
            continue
        rows = [
            {"motif": abbrev(m), "logodds": lo, "agent": a, "rank": i}
            for i, (m, lo) in enumerate(over)
        ]
        df = pd.DataFrame(rows)
        motif_order = [r["motif"] for r in rows]
        bars = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y("motif:N", sort=motif_order,
                        axis=alt.Axis(title=None, domain=False, ticks=False,
                                      labelFontSize=9, labelLimit=160)),
                x=alt.X("logodds:Q",
                        axis=alt.Axis(title="log2 odds vs corpus",
                                      domain=False, ticks=False, labelFontSize=10)),
                color=alt.Color("agent:N", scale=color_scale, legend=None),
            )
        )
        panel = bars.properties(
            width=200, height=top_n * 24,
            title=alt.TitleParams(text=a, fontSize=11,
                                  color="#111111", anchor="start"),
        )
        panels.append(panel)

    # Wrap into rows of 5 to keep readable.
    per_row = 5
    rows_of_panels = [panels[i:i + per_row] for i in range(0, len(panels), per_row)]
    chart = (
        alt.vconcat(
            *[alt.hconcat(*row, spacing=32) for row in rows_of_panels],
            spacing=24,
        )
        .properties(
            title=alt.TitleParams(
                text="Signature action patterns by agent (9-agent corpus)",
                fontSize=13, color="#111111", anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def save_panels(metrics: dict) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    agents_present = [a for a in AGENT_ORDER_EXT if a in metrics]

    # "distinct motifs at 90% coverage" was dropped: Pearson r = 0.892 with
    # entropy on the 9-agent corpus and PERFECT Spearman rank-order match,
    # so it added an extra panel without adding distinct information.
    # Entropy is the principled measure (information-theoretic, no
    # arbitrary 90% threshold). distinct_motifs_at_90pct stays in the
    # JSON for completeness; the panel is gone.
    panels = {
        "fig_agg_entropy_extended.png":     ("entropy_motifs_bits",
            "Motif entropy by agent", "Entropy (bits)", None),
        "fig_agg_length_extended.png":      ("mean_canonical_length",
            "Mean trajectory length", "Mean atoms per trajectory", None),
        "fig_agg_compression_extended.png": ("mean_compression",
            "Mean BPE compression ratio", "BPE length / atom length", [0, 1]),
    }
    for fname, (key, title, ylabel, ydomain) in panels.items():
        df = pd.DataFrame([{"agent": a, "value": metrics[a][key]} for a in agents_present])
        chart = (_bar_panel(df, title, ylabel, y_domain=ydomain)
                 .configure_view(strokeWidth=0)
                 .configure_axis(grid=False))
        chart.save(str(FIG_OUT / fname), scale_factor=2)
        print(f"  saved {fname}")


def main() -> int:
    records = load_records()
    print(f"Loaded {len(records)} extended-corpus trajectories across "
          f"{len(set(r.get('agent') for r in records))} agents.")
    metrics = per_agent_metrics(records)

    for a in AGENT_ORDER_EXT:
        if a not in metrics:
            continue
        m = metrics[a]
        print(f"\n  {a}  (n={m['n_trajectories']})")
        print(f"    entropy(motifs)        = {m['entropy_motifs_bits']:.3f} bits")
        print(f"    distinct@90%           = {m['distinct_motifs_at_90pct']}")
        print(f"    mean canonical length  = {m['mean_canonical_length']:.1f}")
        print(f"    mean BPE length        = {m['mean_bpe_length']:.1f}")
        print(f"    mean compression       = {m['mean_compression']:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_records": len(records),
        "agent_order": AGENT_ORDER_EXT,
        "metrics": metrics,
    }, indent=2))
    print(f"\nSaved {OUT_JSON}")

    save_panels(metrics)

    print("\nLength vs difficulty:")
    difficulty = load_difficulty_extended()
    if difficulty:
        out_lbd = OUT_DIR / "length_by_difficulty.png"
        plot_length_by_difficulty(records, difficulty, out_lbd)
        print(f"  saved {out_lbd.name}")

    print("\nNovelty (signature motifs per agent):")
    out_nov = OUT_DIR / "novelty_top_motifs.png"
    plot_novelty_top(metrics, out_nov)
    print(f"  saved {out_nov.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
