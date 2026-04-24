"""Difficulty-sliced motif analysis.

Bucket the 867 BPE-expressed trajectories by task difficulty (how many of the
three agents resolved it: 0/1-2/3), then recompute per-agent motif
distributions and pairwise Jensen-Shannon divergence within each bucket.

Hypothesis: the same-family vs cross-family JSD gap widens on harder tasks
(more exploration room = more style signal) and narrows on easy tasks
(convergent short paths). Either widening or narrowing is a finding.

Inputs:
    output/paper2_pilot/bpe_sequences.jsonl    (agent, instance_id, bpe)
    output/paper2_pilot/task_diversity.csv     (instance_id, n_resolved)

Outputs:
    output/paper2_pilot/slice_difficulty.json  (per-bucket JSD + top motifs)
    output/paper2_pilot/slice_difficulty_jsd.png
    output/paper2_pilot/slice_difficulty_motifs.png

Usage:
    python -m analysis.preferences.slice_difficulty
"""

from __future__ import annotations

import csv
import json
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
DIVERSITY_PATH = OUT / "task_diversity.csv"

BUCKET_LABEL = {0: "0/3", 1: "1/3", 2: "2/3", 3: "3/3"}
BUCKET_ORDER = ["0/3", "1/3", "2/3", "3/3"]

PAIR_COLORS = {
    "Claude-3.5__GPT-4": "#0072B2",
    "Claude-3.5__GPT-4o": "#E69F00",
    "GPT-4__GPT-4o": "#009E73",
}


def load_sequences() -> list[dict]:
    records = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_difficulty_map() -> dict[str, int]:
    difficulty = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            difficulty[row["instance_id"]] = int(row["n_resolved"])
    return difficulty


def bucket_records(records: list[dict], difficulty: dict[str, int]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {b: [] for b in BUCKET_ORDER}
    for r in records:
        n_res = difficulty.get(r["instance_id"])
        if n_res is None:
            continue
        label = BUCKET_LABEL[n_res]
        buckets[label].append(r)
    return buckets


def per_agent_counts(records: list[dict]) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for r in records:
        out.setdefault(r["agent"], Counter()).update(r["bpe"])
    return out


def normalize(counter: Counter, vocab: list[str]) -> np.ndarray:
    total = sum(counter[v] for v in vocab)
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([counter.get(v, 0) / total for v in vocab])


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a, b, base=2)) ** 2


def analyze_bucket(records: list[dict], bucket_name: str) -> dict:
    counts = per_agent_counts(records)
    all_counts: Counter = Counter()
    for c in counts.values():
        all_counts.update(c)

    full_vocab = sorted(all_counts.keys())
    motif_vocab = [t for t in full_vocab if "+" in t]

    dist_full = {a: normalize(c, full_vocab) for a, c in counts.items()}
    dist_motifs = {a: normalize(c, motif_vocab) for a, c in counts.items()}

    agents = sorted(dist_full.keys())
    pairs = list(combinations(agents, 2))

    jsd_full = {f"{a}__{b}": jsd(dist_full[a], dist_full[b]) for a, b in pairs}
    jsd_motifs = {f"{a}__{b}": jsd(dist_motifs[a], dist_motifs[b]) for a, b in pairs}

    per_agent_totals = {a: sum(c.values()) for a, c in counts.items()}
    top_motifs = [
        {"motif": t, "count": c, "n_atoms": t.count("+") + 1}
        for t, c in all_counts.most_common()
        if "+" in t
    ][:10]

    return {
        "bucket": bucket_name,
        "n_trajectories": len(records),
        "per_agent_trajectories": Counter(r["agent"] for r in records),
        "per_agent_total_tokens": per_agent_totals,
        "jsd_full": jsd_full,
        "jsd_motifs": jsd_motifs,
        "top_motifs": top_motifs,
    }


def plot_jsd_by_bucket(results: list[dict], out_path: Path) -> None:
    buckets = [r["bucket"] for r in results]
    pair_names = sorted(PAIR_COLORS.keys())

    def pair_label(name: str) -> str:
        a, b = name.split("__")
        return f"{a} x {b}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, key, title in [
        (axes[0], "jsd_full", "All action tokens (single steps + repeated sequences)"),
        (axes[1], "jsd_motifs", "Repeated sequences only (two or more steps)"),
    ]:
        for pair in pair_names:
            ys = [r[key][pair] for r in results]
            ax.plot(
                buckets, ys,
                marker="o", color=PAIR_COLORS[pair],
                label=pair_label(pair),
                linewidth=2, markersize=7,
            )
        ax.set_xlabel("Number of agents that solved the task (0 = nobody, 3 = everyone)")
        ax.set_ylabel("How different the action mix is\n(0 = identical, 1 = no overlap)")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "How similar are agents' action patterns, by task difficulty?\n"
        "Lower = agents use a more similar mix of actions. Green = GPT-4 x GPT-4o (same family).",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_motifs_by_bucket(results: list[dict], out_path: Path, top_n: int = 8) -> None:
    all_motifs: Counter = Counter()
    for r in results:
        for item in r["top_motifs"][:top_n]:
            all_motifs[item["motif"]] += item["count"]
    panel_motifs = [m for m, _ in all_motifs.most_common(top_n)]

    def abbrev(m: str) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            return m.replace("+", " -> ")
        return f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)} atoms)"

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.5), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        motif_counts = {item["motif"]: item["count"] for item in r["top_motifs"]}
        total = sum(r["per_agent_total_tokens"].values())
        fractions = [motif_counts.get(m, 0) / total if total else 0 for m in panel_motifs]
        y = np.arange(len(panel_motifs))
        ax.barh(y, fractions, color="#5d90e0", edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([abbrev(m) for m in panel_motifs], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("share of this bucket's actions")
        ax.set_title(f"{r['bucket']} solved\n({r['n_trajectories']} trajectories)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v*100:.1f}%"))

    fig.suptitle(
        f"Top {top_n} repeated action sequences by task difficulty\n"
        "Which patterns dominate when nobody / some / everyone solves the task?",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    records = load_sequences()
    difficulty = load_difficulty_map()
    buckets = bucket_records(records, difficulty)

    print("Difficulty bucket sizes:")
    for b in BUCKET_ORDER:
        rs = buckets[b]
        per_agent = Counter(r["agent"] for r in rs)
        print(f"  {b}: {len(rs)} trajectories  ({dict(per_agent)})")

    results = []
    for b in BUCKET_ORDER:
        rs = buckets[b]
        if not rs:
            continue
        r = analyze_bucket(rs, b)
        results.append(r)
        print(f"\n{b}:")
        print(f"  n={r['n_trajectories']}, per-agent tokens={r['per_agent_total_tokens']}")
        print(f"  JSD (full):    {r['jsd_full']}")
        print(f"  JSD (motifs):  {r['jsd_motifs']}")
        print(f"  top 3 motifs:  {[(m['motif'], m['count']) for m in r['top_motifs'][:3]]}")

    print("\nHeritability ordering check (GPT-family pair should have lowest JSD):")
    for r in results:
        min_pair_m = min(r["jsd_motifs"], key=r["jsd_motifs"].get)
        heritability_gap = min(
            r["jsd_motifs"][k] for k in r["jsd_motifs"] if "Claude" in k
        ) - r["jsd_motifs"].get("GPT-4__GPT-4o", float("nan"))
        print(f"  {r['bucket']}: min pair (motifs)={min_pair_m}, "
              f"heritability gap = {heritability_gap:.4f}")

    serializable = [
        {
            **r,
            "per_agent_trajectories": dict(r["per_agent_trajectories"]),
        }
        for r in results
    ]
    (OUT / "slice_difficulty.json").write_text(
        json.dumps(serializable, indent=2, default=str)
    )

    plot_jsd_by_bucket(results, OUT / "slice_difficulty_jsd.png")
    plot_top_motifs_by_bucket(results, OUT / "slice_difficulty_motifs.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'slice_difficulty.json'}")
    print(f"  {OUT / 'slice_difficulty_jsd.png'}")
    print(f"  {OUT / 'slice_difficulty_motifs.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
