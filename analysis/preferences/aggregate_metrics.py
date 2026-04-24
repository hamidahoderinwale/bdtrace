"""Per-agent aggregate metrics.

For each of the three agents, compute:
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
    output/paper2_pilot/aggregate_metrics.png          (4-panel summary)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

AGENTS = ["Claude-3.5", "GPT-4", "GPT-4o"]
AGENT_COLORS = {"Claude-3.5": "#009E73", "GPT-4": "#0072B2", "GPT-4o": "#E69F00"}


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
    x = np.arange(len(agents))
    colors = [AGENT_COLORS[a] for a in agents]

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8))

    ax = axes[0]
    vals = [metrics[a]["entropy_motifs_bits"] for a in agents]
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("variety score (bits)")
    ax.set_title("How varied is each agent's action mix?\nhigher = spreads usage across more patterns", fontsize=10)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    vals = [metrics[a]["distinct_motifs_at_90pct"] for a in agents]
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("number of action patterns")
    ax.set_title("How many patterns does each agent use?\n(number needed to cover 90% of its actions)", fontsize=10)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.5, f"{v}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    vals = [metrics[a]["mean_canonical_length"] for a in agents]
    bpe_vals = [metrics[a]["mean_bpe_length"] for a in agents]
    w = 0.4
    ax.bar(x - w / 2, vals, w, color=colors, label="individual actions", edgecolor="white")
    ax.bar(
        x + w / 2, bpe_vals, w,
        color=colors, alpha=0.55, label="learned patterns",
        edgecolor="white", hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("mean steps per task")
    ax.set_title("How long is each agent's typical run?\n(individual actions vs grouped patterns)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[3]
    vals = [metrics[a]["mean_compression"] for a in agents]
    ax.bar(x, vals, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("fraction left after grouping")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "How much do agents repeat themselves?\n(fraction of steps after grouping repeats;\nlower = more repetition)",
        fontsize=10,
    )
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Four ways the three agents differ (867 trajectories total)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_length_by_difficulty(
    records: list[dict], difficulty: dict[str, int], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    for ax, key, title, ylab in [
        (axes[0], "canonical_length", "Individual actions", "mean number of actions"),
        (axes[1], "bpe_length", "Grouped action patterns", "mean number of patterns"),
    ]:
        for a in AGENTS:
            points_x, points_y = [], []
            for d in [0, 1, 2, 3]:
                vals = [
                    r[key] for r in records
                    if r["agent"] == a and difficulty.get(r["instance_id"]) == d
                ]
                if vals:
                    points_x.append(d)
                    points_y.append(float(np.mean(vals)))
            ax.plot(
                points_x, points_y,
                marker="o", color=AGENT_COLORS[a],
                label=a, linewidth=2, markersize=7,
            )
        ax.set_xlabel("number of agents that solved the task")
        ax.set_ylabel(f"{ylab} per task")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["0 (nobody)", "1", "2", "3 (everyone)"])
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "How long is each agent's run, and does task difficulty change it?\n"
        "All three shorten as tasks get easier; GPT-4o runs longest when nobody solves.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_novelty_top(metrics: dict, out_path: Path, top_n: int = 6) -> None:
    agents = [a for a in AGENTS if a in metrics]

    def abbrev(m: str) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            return m.replace("+", " -> ")
        return f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)} atoms)"

    fig, axes = plt.subplots(1, len(agents), figsize=(4.5 * len(agents), 4.5), sharex=True)
    if len(agents) == 1:
        axes = [axes]

    for ax, a in zip(axes, agents):
        over = metrics[a]["top_over_used_motifs"][:top_n]
        motifs, logodds = zip(*over)
        y = np.arange(len(motifs))
        ax.barh(y, logodds, color=AGENT_COLORS[a], edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([abbrev(m) for m in motifs], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("log2 odds vs corpus")
        ax.set_title(a, fontsize=10)
        ax.axvline(0, color="#888", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Top {top_n} distinctively over-used motifs per agent\n"
        "(positive log-odds = this agent uses the motif more than the corpus mean)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        print(f"  top 3 over-used motifs:  {[(mot, round(lo, 2)) for mot, lo in m['top_over_used_motifs'][:3]]}")

    (OUT / "aggregate_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics_summary(metrics, OUT / "aggregate_metrics.png")
    plot_length_by_difficulty(records, difficulty, OUT / "length_by_difficulty.png")
    plot_novelty_top(metrics, OUT / "novelty_top_motifs.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'aggregate_metrics.json'}")
    print(f"  {OUT / 'aggregate_metrics.png'}")
    print(f"  {OUT / 'length_by_difficulty.png'}")
    print(f"  {OUT / 'novelty_top_motifs.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
