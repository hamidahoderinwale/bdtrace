"""BPE vocabulary-size sensitivity sweep.

Retrain BPE at several target vocabulary sizes and rerun the agent JSD
analysis at each. Checks whether the heritability finding depends on the
V=200 choice or holds robustly across V ∈ [100, 500].

Outputs:
  output/paper2_pilot/bpe_vocab_sweep.json - full numeric results
  output/paper2_pilot/bpe_vocab_sweep_jsd.png - JSD vs V per agent-pair
  output/paper2_pilot/bpe_vocab_sweep_compression.png - vocab/compression curves

Usage:
    python -m analysis.preferences.bpe_vocab_sweep
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.preferences.bpe import train_bpe
from analysis.preferences.canonicalize import canonicalize_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "output" / "trajectories" / ".cache"
OUT = PROJECT_ROOT / "output" / "paper2_pilot"

AGENT_SHORT = {
    "20240402_sweagent_gpt4": "GPT-4",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20240728_sweagent_gpt4o": "GPT-4o",
}

SWEEP_V = [100, 150, 200, 300, 500]


def load_all_records() -> list[dict]:
    records = []
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        for traj_file in sorted(agent_dir.glob("*.json")):
            with open(traj_file) as f:
                raw = json.load(f)
            seq = canonicalize_trajectory(raw.get("trajectory", []))
            if not seq:
                continue
            records.append({
                "agent": AGENT_SHORT.get(agent_dir.name, agent_dir.name),
                "instance_id": traj_file.stem,
                "canonical": seq,
            })
    return records


def distributions(
    records_with_bpe: list[dict], vocab: list[str], motifs_only: bool = False
) -> dict[str, np.ndarray]:
    """Per-agent probability distribution over vocab."""
    per_agent_counts: dict[str, Counter] = {}
    for r in records_with_bpe:
        per_agent_counts.setdefault(r["agent"], Counter()).update(r["bpe"])

    if motifs_only:
        use_vocab = [v for v in vocab if "+" in v]
    else:
        use_vocab = vocab

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


def run_sweep(records: list[dict], sweep_V: list[int]) -> list[dict]:
    """Train BPE at each V and compute JSD matrices."""
    canonical = [r["canonical"] for r in records]
    n_canonical_tokens = sum(len(s) for s in canonical)

    results = []
    for V in sweep_V:
        print(f"\n=== V = {V} ===")
        model, expressed = train_bpe(canonical, target_size=V, verbose=False)
        n_merges = len(model.merges)
        n_bpe_tokens = sum(len(s) for s in expressed)
        compression = n_bpe_tokens / n_canonical_tokens

        records_bpe = [{"agent": r["agent"], "bpe": e} for r, e in zip(records, expressed)]

        dist_full = distributions(records_bpe, model.vocab, motifs_only=False)
        dist_motifs = distributions(records_bpe, model.vocab, motifs_only=True)

        jsd_full = {f"{a}__{b}": jsd(dist_full[a], dist_full[b])
                    for a, b in combinations(sorted(dist_full), 2)}
        jsd_motifs = {f"{a}__{b}": jsd(dist_motifs[a], dist_motifs[b])
                      for a, b in combinations(sorted(dist_motifs), 2)}

        print(f"  merges={n_merges}, vocab={len(model.vocab)}, compression={compression:.3f}")
        print(f"  JSD full: {jsd_full}")
        print(f"  JSD motifs: {jsd_motifs}")

        results.append({
            "V": V,
            "n_merges": n_merges,
            "actual_vocab": len(model.vocab),
            "compression_ratio": compression,
            "length_distribution": model.summary()["length_distribution"],
            "jsd_full": jsd_full,
            "jsd_motifs": jsd_motifs,
        })
    return results


def plot_jsd_sweep(results: list[dict], out_path: Path) -> None:
    Vs = [r["V"] for r in results]
    pair_names = list(results[0]["jsd_full"].keys())

    pair_colors = {
        "Claude-3.5__GPT-4": "#0072B2",
        "Claude-3.5__GPT-4o": "#E69F00",
        "GPT-4__GPT-4o": "#009E73",
    }

    def label(name: str) -> str:
        a, b = name.split("__")
        return f"{a} x {b}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, key, title in [
        (axes[0], "jsd_full", "Full vocabulary (atoms + motifs)"),
        (axes[1], "jsd_motifs", "Motifs only (length ≥ 2)"),
    ]:
        for pair in pair_names:
            ys = [r[key][pair] for r in results]
            color = pair_colors.get(pair, "gray")
            ax.plot(Vs, ys, marker="o", color=color, label=label(pair),
                    linewidth=2, markersize=6)
        ax.set_xlabel("BPE target vocabulary size (V)")
        ax.set_ylabel("Jensen-Shannon distance")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Heritability signal across BPE vocabulary size\n"
        "GPT-family pair should stay lowest if finding is V-robust",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compression_curve(results: list[dict], out_path: Path) -> None:
    Vs = [r["V"] for r in results]
    compression = [r["compression_ratio"] for r in results]
    merges = [r["n_merges"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(Vs, compression, "o-", color="#0072B2", linewidth=2, label="Compression ratio")
    ax1.set_xlabel("BPE target vocabulary size (V)")
    ax1.set_ylabel("Compression ratio (BPE / canonical token count)", color="#0072B2")
    ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(Vs, merges, "s-", color="#E69F00", linewidth=2, label="Merges performed")
    ax2.set_ylabel("Number of merges", color="#E69F00")
    ax2.tick_params(axis="y", labelcolor="#E69F00")

    ax1.set_title(
        "BPE compression saturates as vocabulary grows\n"
        "Where the curve flattens = diminishing returns on added merges",
        fontsize=10,
    )
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading canonical sequences...")
    records = load_all_records()
    print(f"  {len(records)} trajectories")

    results = run_sweep(records, SWEEP_V)

    # Save numeric results
    (OUT / "bpe_vocab_sweep.json").write_text(json.dumps({
        "sweep_V": SWEEP_V,
        "n_records": len(records),
        "results": results,
    }, indent=2, default=str))

    plot_jsd_sweep(results, OUT / "bpe_vocab_sweep_jsd.png")
    plot_compression_curve(results, OUT / "bpe_vocab_sweep_compression.png")

    print("\n=== SUMMARY TABLE ===")
    print(f"{'V':>5} {'merges':>7} {'compr':>8} "
          f"{'GPT4xGPT4o(full)':>20} {'GPT4xGPT4o(mot)':>20}")
    for r in results:
        gpt_family_full = r["jsd_full"].get("GPT-4__GPT-4o", float("nan"))
        gpt_family_mot = r["jsd_motifs"].get("GPT-4__GPT-4o", float("nan"))
        print(f"{r['V']:>5} {r['n_merges']:>7} {r['compression_ratio']:>8.3f} "
              f"{gpt_family_full:>20.4f} {gpt_family_mot:>20.4f}")

    print("\nHeritability ordering check (GPT-family should be lowest JSD at all V):")
    for r in results:
        pairs_full = r["jsd_full"]
        pairs_mot = r["jsd_motifs"]
        min_pair_full = min(pairs_full, key=pairs_full.get)
        min_pair_mot = min(pairs_mot, key=pairs_mot.get)
        print(f"  V={r['V']:>3}: min JSD pair (full) = {min_pair_full}; "
              f"(motifs) = {min_pair_mot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
