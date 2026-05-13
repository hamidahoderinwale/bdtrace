"""Finer BPE sweep + MDL-based principled V selection.

Extends the earlier coarse sweep with additional V values in the elbow
region (120, 175, 225, 250) and computes MDL-style description length
for each V to identify the principled optimum.

MDL definition used:
    description_length(V) = corpus_bits(V) + vocab_bits(V)

where:
    corpus_bits = sum over tokens in compressed corpus of its self-information
                = sum_t count(t) * -log2(p(t))    (entropy-optimal coding)
    vocab_bits  = sum_i 2 * log2(|vocab| at merge i)  (merge-pointer cost)

Outputs:
    output/paper2_pilot/bpe_mdl_sweep.json
    output/paper2_pilot/bpe_mdl_curve.png

Usage:
    python -m analysis.preferences.bpe_mdl
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.preferences.bpe import train_bpe
from analysis.preferences.canonicalize import canonicalize_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences_extended.jsonl"

V_SWEEP = [100, 125, 150, 175, 200, 225, 250, 300, 500]


def load_canonical() -> tuple[list[list[str]], list[str]]:
    """Read already-canonicalized atom sequences for all 9 agents from
    bpe_sequences_extended.jsonl. Avoids per-scaffold re-canonicalization
    and covers Moatless / DARS / Agentless / Claude-4 etc."""
    sequences: list[list[str]] = []
    agents: list[str] = []
    with SEQ_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seq = r.get("canonical") or []
            if not seq:
                continue
            sequences.append(seq)
            agents.append(r["agent"])
    return sequences, agents


def compute_mdl(
    expressed: list[list[str]],
    n_merges: int,
    initial_vocab_size: int,
) -> dict:
    """Compute entropy-coded corpus bits + merge-pointer vocab bits.

    corpus_bits: Shannon coding of the expressed corpus under observed token
      distribution. Lower bound on encoding cost.
    vocab_bits: each of the n_merges merges needs to point to two vocab
      entries; vocab grows from initial_vocab_size to initial_vocab_size +
      n_merges. Per-merge cost = 2 * log2(vocab_size_at_that_merge).
    """
    counter: Counter = Counter()
    for seq in expressed:
        counter.update(seq)
    total = sum(counter.values())
    # entropy (bits per token)
    H = -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
    corpus_bits = total * H

    # vocab encoding cost: 2 * log2(V_at_merge_i) per merge
    vocab_bits = 0.0
    for i in range(n_merges):
        V_at = initial_vocab_size + i + 1  # vocab size after i-th merge
        vocab_bits += 2 * math.log2(V_at)

    return {
        "n_tokens": total,
        "entropy_bits_per_token": H,
        "corpus_bits": corpus_bits,
        "vocab_bits": vocab_bits,
        "total_bits": corpus_bits + vocab_bits,
    }


def agent_distributions(
    expressed: list[list[str]], agents: list[str], vocab: list[str], motifs_only: bool
) -> dict[str, np.ndarray]:
    per_agent_counts: dict[str, Counter] = {}
    for seq, agent in zip(expressed, agents):
        per_agent_counts.setdefault(agent, Counter()).update(seq)

    use_vocab = [v for v in vocab if "+" in v] if motifs_only else vocab
    out = {}
    for agent, counter in per_agent_counts.items():
        total = sum(counter[v] for v in use_vocab)
        if total == 0:
            out[agent] = np.zeros(len(use_vocab))
        else:
            out[agent] = np.array([counter.get(v, 0) / total for v in use_vocab])
    return out


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a, b, base=2)) ** 2


def run_mdl_sweep(
    sequences: list[list[str]], agents: list[str], V_values: list[int]
) -> list[dict]:
    initial_vocab_size = len({t for s in sequences for t in s})
    results = []
    for V in V_values:
        model, expressed = train_bpe(sequences, target_size=V, verbose=False)
        mdl = compute_mdl(expressed, len(model.merges), initial_vocab_size)

        # JSD at this V
        dist_full = agent_distributions(expressed, agents, model.vocab, motifs_only=False)
        dist_mot = agent_distributions(expressed, agents, model.vocab, motifs_only=True)
        pairs = list(combinations(sorted(dist_full.keys()), 2))
        jsd_full = {f"{a}__{b}": jsd(dist_full[a], dist_full[b]) for a, b in pairs}
        jsd_mot = {f"{a}__{b}": jsd(dist_mot[a], dist_mot[b]) for a, b in pairs}

        compression = sum(len(s) for s in expressed) / sum(len(s) for s in sequences)

        print(f"V={V:>4}: merges={len(model.merges):>4}, "
              f"compression={compression:.3f}, "
              f"corpus_bits={mdl['corpus_bits']:>10.0f}, "
              f"vocab_bits={mdl['vocab_bits']:>8.0f}, "
              f"total={mdl['total_bits']:>10.0f}, "
              f"GPT-family JSD (motifs)={jsd_mot['GPT-4__GPT-4o']:.4f}")

        results.append({
            "V": V,
            "n_merges": len(model.merges),
            "actual_vocab": len(model.vocab),
            "compression_ratio": compression,
            "mdl": mdl,
            "jsd_full": jsd_full,
            "jsd_motifs": jsd_mot,
        })
    return results


def plot_mdl_curve(results: list[dict], out_path: Path) -> None:
    Vs = [r["V"] for r in results]
    total_bits = [r["mdl"]["total_bits"] for r in results]
    corpus_bits = [r["mdl"]["corpus_bits"] for r in results]
    vocab_bits = [r["mdl"]["vocab_bits"] for r in results]
    compressions = [r["compression_ratio"] for r in results]

    min_V_idx = int(np.argmin(total_bits))
    optimal_V = Vs[min_V_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: MDL curve
    ax = axes[0]
    ax.plot(Vs, total_bits, "o-", color="#0072B2", linewidth=2, label="Total bits (MDL)")
    ax.plot(Vs, corpus_bits, "s--", color="#009E73", linewidth=1.5, alpha=0.7, label="Corpus bits (entropy-coded)")
    ax.plot(Vs, vocab_bits, "^--", color="#E69F00", linewidth=1.5, alpha=0.7, label="Vocab bits (merge pointers)")
    ax.axvline(optimal_V, color="black", lw=0.8, ls=":", alpha=0.7)
    ax.annotate(f"MDL min at V={optimal_V}",
                xy=(optimal_V, total_bits[min_V_idx]),
                xytext=(optimal_V + 20, total_bits[min_V_idx]),
                fontsize=9, color="black")
    ax.set_xlabel("BPE target vocabulary size (V)")
    ax.set_ylabel("Bits")
    ax.set_title("Description length vs vocabulary size\nMDL-optimal V minimizes total", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: compression ratio with marginal-gain annotations
    ax = axes[1]
    ax.plot(Vs, compressions, "o-", color="#0072B2", linewidth=2)
    # annotate gain per unit V
    for i in range(1, len(Vs)):
        dV = Vs[i] - Vs[i - 1]
        dC = compressions[i] - compressions[i - 1]
        rate = dC / dV if dV > 0 else 0
        mid_V = (Vs[i] + Vs[i - 1]) / 2
        mid_C = (compressions[i] + compressions[i - 1]) / 2
        ax.annotate(f"{rate*1000:.2f}",
                    xy=(mid_V, mid_C),
                    fontsize=7, color="gray",
                    ha="center", va="bottom")
    ax.axvline(optimal_V, color="black", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("BPE target vocabulary size (V)")
    ax.set_ylabel("Compression ratio (BPE tokens / canonical tokens)")
    ax.set_title("Compression ratio vs V\n(small numbers = marginal Δratio per V, ×1000)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("BPE: principled V via MDL", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading canonical sequences...")
    sequences, agents = load_canonical()
    print(f"  {len(sequences)} sequences, "
          f"{len({t for s in sequences for t in s})} atomic tokens")

    print(f"\nRunning MDL sweep at V = {V_SWEEP}")
    results = run_mdl_sweep(sequences, agents, V_SWEEP)

    total_bits = [r["mdl"]["total_bits"] for r in results]
    min_idx = int(np.argmin(total_bits))
    optimal_V = results[min_idx]["V"]
    optimal_result = results[min_idx]

    print(f"\n=== MDL-optimal V = {optimal_V} ===")
    print(f"  total_bits = {optimal_result['mdl']['total_bits']:.0f}")
    print(f"  compression = {optimal_result['compression_ratio']:.3f}")
    print(f"  GPT-4 x GPT-4o JSD (motifs) = {optimal_result['jsd_motifs']['GPT-4__GPT-4o']:.4f}")
    print(f"  Claude x GPT-4 JSD (motifs) = {optimal_result['jsd_motifs']['Claude-3.5__GPT-4']:.4f}")
    print(f"  Claude x GPT-4o JSD (motifs) = {optimal_result['jsd_motifs']['Claude-3.5__GPT-4o']:.4f}")

    (OUT / "bpe_mdl_sweep.json").write_text(json.dumps({
        "V_sweep": V_SWEEP,
        "mdl_optimal_V": optimal_V,
        "results": results,
    }, indent=2, default=str))

    plot_mdl_curve(results, OUT / "bpe_mdl_curve.png")

    print(f"\nSaved:\n  {OUT / 'bpe_mdl_sweep.json'}\n  {OUT / 'bpe_mdl_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
