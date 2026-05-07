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
)
register()

SEQ = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
OUT_JSON = ROOT / "output" / "paper2_pilot" / "aggregate_metrics_extended.json"
FIG_OUT = ROOT / "output" / "figures"

AGENT_ORDER_EXT = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking", "GPT-4", "GPT-4o",
    "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
]
AGENT_COLORS_EXT = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "Claude-3.7-thinking":   GREEN_D,
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "DARS+R1":               MAGENTA_D,
    "Agentless+Claude-3.5":  BLUE_D,
    "Moatless+V3":           OLIVE,
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

    metrics: dict[str, dict] = {}
    for a, rs in by_agent.items():
        if not rs:
            continue
        c_full = counts_by_agent[a]
        c_motifs = Counter({t: cnt for t, cnt in c_full.items() if "+" in t})
        canonical_lens = [r["canonical_length"] for r in rs]
        bpe_lens = [r["bpe_length"] for r in rs]
        compressions = [r["compression"] for r in rs]
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


def save_panels(metrics: dict) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    agents_present = [a for a in AGENT_ORDER_EXT if a in metrics]

    panels = {
        "fig_agg_entropy_extended.png":     ("entropy_motifs_bits",
            "Motif entropy by agent (bits) — extended corpus", "Entropy (bits)", None),
        "fig_agg_repertoire_extended.png":  ("distinct_motifs_at_90pct",
            "Distinct motifs at 90% coverage — extended corpus", "Number of motifs", None),
        "fig_agg_length_extended.png":      ("mean_canonical_length",
            "Mean trajectory length (atoms) — extended corpus", "Mean atoms per trajectory", None),
        "fig_agg_compression_extended.png": ("mean_compression",
            "Mean BPE compression ratio — extended corpus", "BPE length / atom length", [0, 1]),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
