"""Per-agent motif distributions + JSD matrix.

For each of the 3 agents:
  1. Aggregate motif-usage counts across all its BPE-expressed trajectories
  2. Normalize → per-agent probability distribution over vocabulary
  3. Compute pairwise JSD between agent distributions

Reports two versions:
  - Full vocabulary (atoms + motifs): JSD captures total procedural distribution
  - Motifs only (length ≥ 2): JSD captures multi-step-pattern usage specifically

Outputs:
  output/paper2_pilot/agent_motif_distributions.png  (top motifs per agent, stacked)
  output/paper2_pilot/agent_jsd_matrix.png           (pairwise JSD heatmap)
  output/paper2_pilot/agent_motif_distributions.json (full numeric output)

Usage:
    python -m analysis.preferences.motif_distributions
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"

AGENT_COLORS = {
    "GPT-4": "#0072B2",
    "Claude-3.5": "#009E73",
    "GPT-4o": "#E69F00",
}


def load_sequences() -> list[dict]:
    records = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def per_agent_token_counts(records: list[dict]) -> dict[str, Counter]:
    """Aggregate token counts per agent across all their trajectories."""
    counts: dict[str, Counter] = {}
    for r in records:
        agent = r["agent"]
        counts.setdefault(agent, Counter()).update(r["bpe"])
    return counts


def normalize(counter: Counter, vocab: list[str]) -> np.ndarray:
    """Turn a counter into a probability distribution over vocab (in order)."""
    total = sum(counter.values())
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([counter.get(v, 0) / total for v in vocab])


def compute_jsd_matrix(
    distributions: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Pairwise Jensen-Shannon distance between agent distributions.

    Returns JSD (not sqrt), bounded [0, 1] with log base 2.
    scipy's jensenshannon returns sqrt(JSD), so we square it.
    """
    out = {}
    agents = list(distributions.keys())
    for a, b in combinations(agents, 2):
        # scipy uses natural log by default; convert to log-base-2 by multiplying
        # by (1/ln(2))^2 — but easier: use base=2 explicitly
        d = float(jensenshannon(distributions[a], distributions[b], base=2)) ** 2
        out[(a, b)] = d
    return out


def plot_top_motifs_per_agent(
    records: list[dict],
    per_agent_counts: dict[str, Counter],
    top_n: int = 15,
    motifs_only: bool = True,
    out_path: Path = None,
) -> None:
    """Horizontal grouped bar chart: top N motifs by overall frequency,
    showing per-agent usage fraction side-by-side."""
    all_counts: Counter = Counter()
    for c in per_agent_counts.values():
        all_counts.update(c)

    # Pick top N
    items = list(all_counts.items())
    if motifs_only:
        items = [(t, c) for t, c in items if "+" in t]
    items.sort(key=lambda x: -x[1])
    top_motifs = [t for t, _ in items[:top_n]]

    # For each agent, fraction of their tokens that is each motif
    agent_totals = {a: sum(c.values()) for a, c in per_agent_counts.items()}
    agent_fractions = {
        a: [per_agent_counts[a].get(m, 0) / agent_totals[a] for m in top_motifs]
        for a in per_agent_counts
    }

    agents = sorted(per_agent_counts.keys())
    y = np.arange(len(top_motifs))
    height = 0.25

    fig, ax = plt.subplots(figsize=(11, 0.35 * len(top_motifs) + 1.8))
    for i, agent in enumerate(agents):
        offset = (i - (len(agents) - 1) / 2) * height
        ax.barh(y + offset, agent_fractions[agent], height,
                color=AGENT_COLORS.get(agent, "gray"),
                label=agent, edgecolor="white")

    # Abbreviate motif labels for readability
    labels = []
    for m in top_motifs:
        parts = m.split("+")
        if len(parts) <= 2:
            labels.append(m.replace("+", " -> "))
        else:
            labels.append(f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)} atoms)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of agent's procedural vocabulary")
    title = "Agents use different procedural motifs"
    subtitle = f"Top {top_n} {'motifs' if motifs_only else 'vocabulary items'}; bars = fraction of each agent's BPE tokens"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v*100:.1f}%"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_jsd_matrix(
    jsd_full: dict[tuple[str, str], float],
    jsd_motifs: dict[tuple[str, str], float],
    out_path: Path,
) -> None:
    """Two small JSD tables side by side: full vocab vs motifs-only."""
    agents = sorted({a for pair in jsd_full for a in pair})
    n = len(agents)

    def make_matrix(d):
        M = np.zeros((n, n))
        for (a, b), v in d.items():
            i, j = agents.index(a), agents.index(b)
            M[i, j] = v
            M[j, i] = v
        return M

    M_full = make_matrix(jsd_full)
    M_motifs = make_matrix(jsd_motifs)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    for ax, M, title in [
        (axes[0], M_full, "Full vocabulary (atoms + motifs)"),
        (axes[1], M_motifs, "Motifs only (length ≥ 2)"),
    ]:
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M_full.max(), M_motifs.max()) * 1.05)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(agents, fontsize=9)
        ax.set_yticklabels(agents, fontsize=9)
        ax.set_title(title, fontsize=10)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                color = "white" if M[i, j] > 0.15 else "black"
                text = "0" if i == j else f"{M[i, j]:.3f}"
                ax.text(j, i, text, ha="center", va="center",
                        color=color, fontsize=9)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    fig.suptitle(
        "Pairwise Jensen-Shannon distance between agent motif distributions\n"
        "Lower = more similar distribution of procedural practice",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading BPE-expressed sequences...")
    records = load_sequences()
    print(f"  {len(records)} records across {len({r['agent'] for r in records})} agents")

    per_agent_counts = per_agent_token_counts(records)

    # Build vocab from union of counters
    full_vocab = sorted({t for c in per_agent_counts.values() for t in c})
    motif_vocab = [t for t in full_vocab if "+" in t]
    print(f"  full vocabulary: {len(full_vocab)} items ({len(motif_vocab)} motifs)")

    # Per-agent distributions
    dist_full = {a: normalize(c, full_vocab) for a, c in per_agent_counts.items()}
    dist_motifs = {
        a: normalize(Counter({t: c[t] for t in motif_vocab if t in c}), motif_vocab)
        for a, c in per_agent_counts.items()
    }

    # JSD matrices
    jsd_full = compute_jsd_matrix(dist_full)
    jsd_motifs = compute_jsd_matrix(dist_motifs)

    print("\nPairwise JSD (full vocabulary):")
    for (a, b), v in sorted(jsd_full.items()):
        print(f"  {a:12s} x {b:12s}: {v:.4f}")
    print("\nPairwise JSD (motifs only):")
    for (a, b), v in sorted(jsd_motifs.items()):
        print(f"  {a:12s} x {b:12s}: {v:.4f}")

    # Figures
    plot_top_motifs_per_agent(
        records, per_agent_counts,
        top_n=15, motifs_only=True,
        out_path=OUT / "agent_motif_distributions.png",
    )
    plot_jsd_matrix(jsd_full, jsd_motifs, OUT / "agent_jsd_matrix.png")

    # Numeric output
    summary = {
        "n_records": len(records),
        "full_vocab_size": len(full_vocab),
        "n_motifs": len(motif_vocab),
        "per_agent_total_tokens": {a: sum(c.values()) for a, c in per_agent_counts.items()},
        "jsd_full_vocab": {f"{a}__{b}": v for (a, b), v in jsd_full.items()},
        "jsd_motifs_only": {f"{a}__{b}": v for (a, b), v in jsd_motifs.items()},
        "interpretation": {
            "jsd_range": "[0, 1] with log2; 0 = identical distributions, 1 = disjoint",
            "heritability_check": "lower JSD between same-family agents (GPT-4 x GPT-4o) relative to cross-family supports heritability",
        },
    }
    (OUT / "agent_motif_distributions.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print(f"\nSaved:")
    for n in ["agent_motif_distributions.png", "agent_jsd_matrix.png",
              "agent_motif_distributions.json"]:
        print(f"  {OUT / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
