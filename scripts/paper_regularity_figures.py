"""Procedural regularity analyses connecting trajectory structure to composition failures.

Three linked analyses:
  1. Motif surprise by composition failure type (strip chart)
  2. Per-agent regularity and composition failure rate (horizontal bars)
  3. Trigram perplexity vs task ease (scatter)

Outputs (output/paper2_pilot/):
  regularity_1_surprise_by_class.png
  regularity_2_agent_profile.png
  regularity_3_perplexity_vs_ease.png
  regularity_data.json

Usage:
    python -m scripts.paper_regularity_figures
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, GRAY, VERMILLION
register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"

AGENT_SHORT_TO_LONG = {
    "GPT-4":      "20240402_sweagent_gpt4",
    "Claude-3.5": "20240620_sweagent_claude3.5sonnet",
    "GPT-4o":     "20240728_sweagent_gpt4o",
}

AGENT_COLORS = {"GPT-4": BLUE, "Claude-3.5": GREEN, "GPT-4o": ORANGE}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bpe_sequences() -> list[dict]:
    path = PROJECT_ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
    seqs = []
    with open(path) as f:
        for line in f:
            seqs.append(json.loads(line))
    return seqs


def load_leaderboard() -> dict[str, dict[str, bool]]:
    return json.load(open(PROJECT_ROOT / "output" / "leaderboard" / "lite_results.json"))


def load_instance_classification() -> dict[str, dict[str, str]]:
    return json.load(open(
        PROJECT_ROOT / "output" / "compositional_generalization" / "instance_classification.json"
    ))


def load_ease() -> dict[str, float]:
    gap = json.load(open(
        PROJECT_ROOT / "output" / "compositional_generalization" / "composition_gap.json"
    ))
    return {r["instance_id"]: r["ease"] for r in gap}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def build_corpus_motif_counts(seqs: list[dict]) -> Counter:
    counts: Counter = Counter()
    for s in seqs:
        counts.update(s["bpe"])
    return counts


def mean_motif_surprise(bpe_tokens: list[str], counts: Counter, total: int) -> float:
    if not bpe_tokens:
        return 0.0
    surprises = []
    for tok in bpe_tokens:
        p = counts.get(tok, 0) / total
        surprises.append(-math.log2(p) if p > 0 else -math.log2(1 / (total + 1)))
    return float(np.mean(surprises))


def build_trigram_model(seqs: list[dict], smoothing: float = 0.5) -> dict:
    vocab: set[str] = set()
    unigram: Counter = Counter()
    bigram: Counter  = Counter()
    trigram: Counter = Counter()

    for s in seqs:
        toks = s["bpe"]
        vocab.update(toks)
        for t in toks:
            unigram[t] += 1
        for i in range(len(toks) - 1):
            bigram[(toks[i], toks[i+1])] += 1
        for i in range(len(toks) - 2):
            trigram[(toks[i], toks[i+1], toks[i+2])] += 1

    return {
        "vocab": vocab, "unigram": unigram, "bigram": bigram, "trigram": trigram,
        "smoothing": smoothing, "total_unigram": sum(unigram.values()),
    }


def trigram_perplexity(bpe_tokens: list[str], model: dict) -> float | None:
    toks = bpe_tokens
    if len(toks) < 3:
        return None
    V = len(model["vocab"])
    k = model["smoothing"]
    log_prob = 0.0
    n = 0
    for i in range(2, len(toks)):
        t0, t1, t2 = toks[i-2], toks[i-1], toks[i]
        tri_count = model["trigram"].get((t0, t1, t2), 0)
        bi_count  = model["bigram"].get((t0, t1), 0)
        p = (tri_count + k) / (bi_count + k * V)
        log_prob += math.log2(p)
        n += 1
    return 2 ** (-log_prob / n) if n > 0 else None


def build_analysis_rows(
    seqs, leaderboard, instance_classification, ease_map, corpus_counts, trigram_model
) -> list[dict]:
    total_tokens = sum(corpus_counts.values())
    rows = []
    for s in seqs:
        agent_short = s["agent"]
        agent_long  = AGENT_SHORT_TO_LONG.get(agent_short)
        if agent_long is None:
            continue
        instance_id = s["instance_id"]
        passed = leaderboard.get(agent_long, {}).get(instance_id, False)
        classification = None
        if not passed:
            inst_map = instance_classification.get(instance_id, {})
            classification = inst_map.get(agent_long)
        rows.append({
            "agent":          agent_short,
            "instance_id":    instance_id,
            "passed":         bool(passed),
            "classification": classification,
            "mean_surprise":  mean_motif_surprise(s["bpe"], corpus_counts, total_tokens),
            "perplexity":     trigram_perplexity(s["bpe"], trigram_model),
            "ease":           ease_map.get(instance_id),
            "compression":    s["compression"],
        })
    return rows


# ---------------------------------------------------------------------------
# Figure 1: Motif surprise by composition failure type
# ---------------------------------------------------------------------------

CLASS_ORDER  = ["familiar", "novel_primitive", "novel_composition"]
CLASS_LABELS = {
    "familiar":          "Familiar",
    "novel_primitive":   "Novel primitive",
    "novel_composition": "Novel composition",
}
CLASS_COLORS = [GREEN, BLUE, ORANGE]


def plot_surprise_by_class(rows: list[dict], out_path: Path) -> None:
    data = [
        {"class_label": CLASS_LABELS[r["classification"]], "mean_surprise": r["mean_surprise"]}
        for r in rows
        if r["classification"] in CLASS_LABELS
    ]
    df = pd.DataFrame(data)

    label_order = [CLASS_LABELS[c] for c in CLASS_ORDER]
    cscale = alt.Scale(domain=label_order, range=CLASS_COLORS)

    rng = np.random.default_rng(42)
    df["jitter"] = rng.uniform(-0.3, 0.3, size=len(df))

    medians = (
        df.groupby("class_label")["mean_surprise"]
        .median()
        .reset_index()
        .rename(columns={"mean_surprise": "median_surprise"})
    )

    strip = (
        alt.Chart(df)
        .mark_point(size=22, filled=True, opacity=0.45, strokeWidth=0)
        .encode(
            x=alt.X("class_label:N", sort=label_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=11)),
            xOffset=alt.XOffset("jitter:Q", scale=alt.Scale(domain=[-1, 1])),
            y=alt.Y("mean_surprise:Q",
                    axis=alt.Axis(title="Mean motif surprise (bits)",
                                  domain=False, ticks=False)),
            color=alt.Color("class_label:N", sort=label_order,
                            scale=cscale, legend=None),
        )
    )

    ticks = (
        alt.Chart(medians)
        .mark_tick(thickness=2.5, size=44, color="#2B2D42")
        .encode(
            x=alt.X("class_label:N", sort=label_order),
            y=alt.Y("median_surprise:Q"),
        )
    )

    chart = (
        (strip + ticks)
        .properties(
            width=300, height=240,
            title=alt.TitleParams(
                text="Motif surprise by failure type",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )

    chart.save(str(out_path), scale_factor=2)
    print(f"  Saved: {out_path.name}")
    for c in CLASS_ORDER:
        vals = df[df["class_label"] == CLASS_LABELS[c]]["mean_surprise"].values
        if len(vals):
            print(f"    {CLASS_LABELS[c]}: n={len(vals)}, median={np.median(vals):.3f}")


# ---------------------------------------------------------------------------
# Figure 2: Per-agent regularity and composition failure rate
# ---------------------------------------------------------------------------

def plot_agent_profile(rows: list[dict], out_path: Path) -> None:
    agents = ["GPT-4", "Claude-3.5", "GPT-4o"]

    comp_by_agent: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        comp_by_agent[r["agent"]].append(r["compression"])
    mean_compression = {a: float(np.mean(comp_by_agent[a])) for a in agents}

    fail_class_counts: dict[str, Counter] = {a: Counter() for a in agents}
    for r in rows:
        if not r["passed"] and r["classification"]:
            fail_class_counts[r["agent"]][r["classification"]] += 1

    comp_fail_frac = {}
    for a in agents:
        total_fail = sum(fail_class_counts[a].values())
        comp_fail_frac[a] = (
            fail_class_counts[a]["novel_composition"] / total_fail
            if total_fail > 0 else 0.0
        )

    df_comp = pd.DataFrame([
        {"agent": a, "value": mean_compression[a], "metric": "BPE compression ratio"}
        for a in agents
    ])
    df_fail = pd.DataFrame([
        {"agent": a, "value": comp_fail_frac[a], "metric": "Novel composition fraction of failures"}
        for a in agents
    ])
    df = pd.concat([df_comp, df_fail], ignore_index=True)

    agent_order = agents[::-1]
    cscale = alt.Scale(domain=agents, range=[AGENT_COLORS[a] for a in agents])

    chart = (
        alt.Chart(df)
        .mark_bar(height=18)
        .encode(
            y=alt.Y("agent:N", sort=agent_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=10)),
            x=alt.X("value:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  format=".2f")),
            color=alt.Color("agent:N", sort=agents, scale=cscale, legend=None),
            column=alt.Column(
                "metric:N",
                sort=["BPE compression ratio", "Novel composition fraction of failures"],
                header=alt.Header(
                    title=None, labelFontSize=10,
                    labelOrient="bottom",
                ),
            ),
        )
        .properties(
            width=200, height=100,
            title=alt.TitleParams(
                text="Compression ratio and failure rate by agent",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(stroke="#DDDDDD", strokeWidth=0.5)
    )

    chart.save(str(out_path), scale_factor=2)
    print(f"  Saved: {out_path.name}")
    for a in agents:
        print(f"    {a}: compression={mean_compression[a]:.3f}, "
              f"novel_comp_frac={comp_fail_frac[a]:.3f}")


# ---------------------------------------------------------------------------
# Figure 3: Trigram perplexity vs task ease
# ---------------------------------------------------------------------------

def plot_perplexity_vs_ease(rows: list[dict], out_path: Path) -> None:
    data = [
        {
            "ease":       r["ease"],
            "perplexity": r["perplexity"],
            "outcome":    "Passed" if r["passed"] else "Failed",
        }
        for r in rows
        if r["perplexity"] is not None and r["ease"] is not None
    ]
    df = pd.DataFrame(data)

    outcome_order = ["Passed", "Failed"]
    cscale = alt.Scale(domain=outcome_order, range=[GREEN, ORANGE])

    chart = (
        alt.Chart(df)
        .mark_point(size=25, filled=True, opacity=0.4, strokeWidth=0)
        .encode(
            x=alt.X("ease:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Task ease (fraction of agents that pass)",
                                  domain=False, ticks=False,
                                  values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
            y=alt.Y("perplexity:Q",
                    scale=alt.Scale(type="log"),
                    axis=alt.Axis(title="Trajectory perplexity (bits, log scale)",
                                  domain=False, ticks=False)),
            color=alt.Color("outcome:N", sort=outcome_order, scale=cscale,
                            legend=alt.Legend(orient="bottom", title=None,
                                              symbolSize=80)),
        )
        .properties(
            title=alt.TitleParams(
                text="Trajectory perplexity vs task ease",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=360, height=260,
        )
        .configure_view(strokeWidth=0)
    )

    chart.save(str(out_path), scale_factor=2)
    print(f"  Saved: {out_path.name}")

    for outcome in outcome_order:
        vals = df[df["outcome"] == outcome]["perplexity"].values
        if len(vals):
            print(f"    {outcome}: n={len(vals)}, median perplexity={np.median(vals):.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    seqs        = load_bpe_sequences()
    leaderboard = load_leaderboard()
    inst_class  = load_instance_classification()
    ease_map    = load_ease()
    print(f"  {len(seqs)} BPE sequences loaded")

    print("Building corpus features...")
    corpus_counts = build_corpus_motif_counts(seqs)
    trigram_model = build_trigram_model(seqs)

    print("Building analysis rows...")
    rows = build_analysis_rows(seqs, leaderboard, inst_class, ease_map, corpus_counts, trigram_model)
    n_pass = sum(1 for r in rows if r["passed"])
    n_fail = sum(1 for r in rows if not r["passed"])
    print(f"  {len(rows)} rows ({n_pass} passed, {n_fail} failed)")

    print("\nFigure 1: Motif surprise by classification...")
    plot_surprise_by_class(rows, OUT / "regularity_1_surprise_by_class.png")

    print("\nFigure 2: Agent regularity profile...")
    plot_agent_profile(rows, OUT / "regularity_2_agent_profile.png")

    print("\nFigure 3: Perplexity vs ease...")
    plot_perplexity_vs_ease(rows, OUT / "regularity_3_perplexity_vs_ease.png")

    summary = {
        "n_rows": len(rows),
        "n_passed": n_pass,
        "n_failed": n_fail,
        "corpus_vocab_size": len(corpus_counts),
        "trigram_vocab_size": len(trigram_model["vocab"]),
    }
    (OUT / "regularity_data.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
