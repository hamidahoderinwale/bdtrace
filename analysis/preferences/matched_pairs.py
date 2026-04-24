"""Matched-pairs per-motif analysis on tied-outcome pairs.

For each tied-outcome pair (agent_A, agent_B, task) where both agents resolved
the same task, compute per-motif frequency difference (A - B). Aggregate
across tasks per agent-pair. Wilcoxon signed-rank test per motif + BH FDR
correction at 5%.

This is the clean per-motif agent-effect estimate: task is controlled, so
what's left is agent-attributable.

Outputs:
    output/paper2_pilot/matched_pairs.json
    output/paper2_pilot/matched_pairs_volcano.png
    output/paper2_pilot/matched_pairs_top_motifs.png

Usage:
    python -m analysis.preferences.matched_pairs
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
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
PAIRS_PATH = OUT / "tied_outcome_pairs.csv"

AGENTS = ["Claude-3.5", "GPT-4", "GPT-4o"]
AGENT_FROM_CSV = {
    "Claude 3.5 Sonnet (SWE-agent)": "Claude-3.5",
    "GPT-4 (SWE-agent)": "GPT-4",
    "GPT-4o (SWE-agent)": "GPT-4o",
}
PAIR_COLORS = {
    ("Claude-3.5", "GPT-4"): "#0072B2",
    ("Claude-3.5", "GPT-4o"): "#E69F00",
    ("GPT-4", "GPT-4o"): "#009E73",
}
PAIR_IS_SAME_FAMILY = {
    ("Claude-3.5", "GPT-4"): False,
    ("Claude-3.5", "GPT-4o"): False,
    ("GPT-4", "GPT-4o"): True,
}
MIN_NONZERO_PAIRS = 3


def load_sequences() -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    with open(SEQ_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["agent"], r["instance_id"])] = r["bpe"]
    return out


def load_tied_pairs() -> list[tuple[str, str, str]]:
    """Return list of (agent_a, agent_b, instance_id), normalized to our short names."""
    pairs = []
    with open(PAIRS_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            a = AGENT_FROM_CSV.get(row["agent_a"], row["agent_a"])
            b = AGENT_FROM_CSV.get(row["agent_b"], row["agent_b"])
            if a in AGENTS and b in AGENTS:
                pairs.append((a, b, row["instance_id"]))
    return pairs


def freq_vector(motif_seq: list[str], vocab: list[str]) -> np.ndarray:
    c = Counter(motif_seq)
    total = sum(c.values())
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([c.get(v, 0) / total for v in vocab])


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return (adjusted_pvals, reject_mask) using Benjamini-Hochberg FDR."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    adjusted = np.empty(n)
    adjusted[order] = adj
    reject = adjusted < alpha
    return adjusted, reject


def analyze_agent_pair(
    pair: tuple[str, str],
    pair_instances: list[str],
    seqs: dict[tuple[str, str], list[str]],
    motifs_only: bool = True,
) -> dict:
    a, b = pair
    vocab_set: set[str] = set()
    for inst in pair_instances:
        vocab_set.update(seqs.get((a, inst), []))
        vocab_set.update(seqs.get((b, inst), []))
    vocab = sorted(vocab_set)
    if motifs_only:
        vocab = [v for v in vocab if "+" in v]

    deltas = np.zeros((len(pair_instances), len(vocab)))
    for i, inst in enumerate(pair_instances):
        va = freq_vector(seqs[(a, inst)], vocab)
        vb = freq_vector(seqs[(b, inst)], vocab)
        deltas[i] = va - vb

    n_pairs = len(pair_instances)
    results = []
    for j, motif in enumerate(vocab):
        col = deltas[:, j]
        n_nonzero = int(np.sum(col != 0))
        mean_delta = float(col.mean())
        median_delta = float(np.median(col))
        if n_nonzero < MIN_NONZERO_PAIRS:
            p = float("nan")
            stat = float("nan")
        else:
            try:
                res = wilcoxon(col, alternative="two-sided", zero_method="wilcox")
                stat = float(res.statistic)
                p = float(res.pvalue)
            except ValueError:
                stat = float("nan")
                p = float("nan")
        results.append({
            "motif": motif,
            "n_pairs": n_pairs,
            "n_nonzero_pairs": n_nonzero,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "wilcoxon_stat": stat,
            "p_value": p,
        })

    valid_idx = [i for i, r in enumerate(results) if not np.isnan(r["p_value"])]
    valid_pvals = np.array([results[i]["p_value"] for i in valid_idx])
    if len(valid_pvals) > 0:
        adjusted, reject = benjamini_hochberg(valid_pvals, alpha=0.05)
        for idx, adj, rej in zip(valid_idx, adjusted, reject):
            results[idx]["p_adjusted"] = float(adj)
            results[idx]["significant_fdr_5"] = bool(rej)
    for r in results:
        r.setdefault("p_adjusted", float("nan"))
        r.setdefault("significant_fdr_5", False)

    return {
        "pair": f"{a}__{b}",
        "agent_a": a,
        "agent_b": b,
        "same_family": PAIR_IS_SAME_FAMILY[pair],
        "n_tied_outcome_tasks": n_pairs,
        "n_motifs_tested": sum(1 for r in results if not np.isnan(r["p_value"])),
        "n_significant_fdr_5": sum(1 for r in results if r["significant_fdr_5"]),
        "motif_results": results,
    }


def plot_volcano(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4.2), sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        deltas = np.array([mr["mean_delta"] for mr in r["motif_results"]])
        pvals = np.array([mr["p_value"] for mr in r["motif_results"]])
        sig = np.array([mr["significant_fdr_5"] for mr in r["motif_results"]])
        valid = ~np.isnan(pvals)
        deltas_v = deltas[valid]
        pvals_v = pvals[valid]
        sig_v = sig[valid]

        ax.scatter(
            deltas_v[~sig_v],
            -np.log10(pvals_v[~sig_v] + 1e-12),
            s=22, color="#bbbbbb", edgecolor="none", alpha=0.6,
            label="not significant",
        )
        ax.scatter(
            deltas_v[sig_v],
            -np.log10(pvals_v[sig_v] + 1e-12),
            s=36, color=PAIR_COLORS[(r["agent_a"], r["agent_b"])],
            edgecolor="white", linewidth=0.4,
            label=f"significant (FDR 5%): {int(sig_v.sum())}",
        )
        ax.axvline(0, color="#555", lw=0.6)
        ax.set_xlabel(f"mean freq({r['agent_a']}) - freq({r['agent_b']})\n(per task, averaged)")
        if ax is axes[0]:
            ax.set_ylabel("-log10(p-value), Wilcoxon signed-rank")
        family_tag = "same family" if r["same_family"] else "cross family"
        ax.set_title(
            f"{r['agent_a']} vs {r['agent_b']}  ({family_tag})\n"
            f"{r['n_tied_outcome_tasks']} tied-outcome tasks, "
            f"{r['n_significant_fdr_5']}/{r['n_motifs_tested']} motifs significant",
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.suptitle(
        "Matched-pairs analysis: which motifs differ when agents solve the same task?\n"
        "Each dot is one motif. Right = used more by first agent; left = used more by second.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_motifs(results: list[dict], out_path: Path, top_n: int = 8) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.8))
    if len(results) == 1:
        axes = [axes]

    def abbrev(m: str, max_len: int = 26) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            s = m.replace("+", " -> ")
        else:
            s = f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)} atoms)"
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    for ax, r in zip(axes, results):
        all_tested = [mr for mr in r["motif_results"] if not np.isnan(mr["p_value"])]
        all_tested.sort(key=lambda mr: abs(mr["mean_delta"]), reverse=True)
        top = all_tested[:top_n]
        if not top:
            ax.text(0.5, 0.5, "no testable tokens",
                    ha="center", va="center", fontsize=11, transform=ax.transAxes)
            ax.set_title(f"{r['agent_a']} vs {r['agent_b']}", fontsize=10)
            ax.axis("off")
            continue

        names = [abbrev(mr["motif"]) for mr in top]
        deltas = [mr["mean_delta"] for mr in top]
        sigs = [mr["significant_fdr_5"] for mr in top]
        pair_color = PAIR_COLORS[(r["agent_a"], r["agent_b"])]
        colors = [
            pair_color if d > 0 else "#cc6666"
            for d in deltas
        ]
        edges = ["#111111" if s else "white" for s in sigs]
        widths = [1.8 if s else 0.4 for s in sigs]
        y = np.arange(len(top))
        for yi, d, c, e, w in zip(y, deltas, colors, edges, widths):
            ax.barh(yi, d, color=c, edgecolor=e, linewidth=w)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_yticks(y)
        labels = [f"{n}  *" if s else n for n, s in zip(names, sigs)]
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"mean freq({r['agent_a']}) - freq({r['agent_b']})")
        family_tag = "same family" if r["same_family"] else "cross family"
        n_sig = sum(sigs)
        ax.set_title(
            f"{r['agent_a']} vs {r['agent_b']}  ({family_tag})\n"
            f"{n_sig} of top {len(top)} significant at FDR 5%",
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Top {top_n} most-differentiating tokens per agent pair (by mean effect size)\n"
        "Star + dark border = passes FDR 5% threshold. Bars in agent-pair color favor the first agent; red bars favor the second.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    seqs = load_sequences()
    pairs = load_tied_pairs()
    print(f"Loaded {len(seqs)} trajectories and {len(pairs)} tied-outcome pairs")

    # Group pair instances by sorted agent pair
    by_pair: dict[tuple[str, str], list[str]] = {}
    for a, b, inst in pairs:
        key = tuple(sorted([a, b]))
        by_pair.setdefault(key, []).append(inst)

    results = []
    for pair_key in sorted(by_pair.keys()):
        pair_instances = by_pair[pair_key]
        missing = [inst for inst in pair_instances if (pair_key[0], inst) not in seqs or (pair_key[1], inst) not in seqs]
        if missing:
            print(f"  {pair_key}: dropping {len(missing)} pairs missing bpe sequences")
            pair_instances = [inst for inst in pair_instances if inst not in missing]
        print(f"\n=== {pair_key[0]} vs {pair_key[1]}  (n={len(pair_instances)} tasks) ===")
        r = analyze_agent_pair(pair_key, pair_instances, seqs, motifs_only=False)
        print(f"  motifs tested: {r['n_motifs_tested']}")
        print(f"  significant (FDR 5%): {r['n_significant_fdr_5']}")
        top = sorted(
            [mr for mr in r["motif_results"] if mr["significant_fdr_5"]],
            key=lambda x: abs(x["mean_delta"]),
            reverse=True,
        )[:5]
        for mr in top:
            sign = "+" if mr["mean_delta"] > 0 else "-"
            print(f"    {sign} {mr['motif']:<60s}  Δ={mr['mean_delta']:+.4f}  p_adj={mr['p_adjusted']:.3g}")
        results.append(r)

    (OUT / "matched_pairs.json").write_text(json.dumps(results, indent=2, default=str))
    plot_volcano(results, OUT / "matched_pairs_volcano.png")
    plot_top_motifs(results, OUT / "matched_pairs_top_motifs.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'matched_pairs.json'}")
    print(f"  {OUT / 'matched_pairs_volcano.png'}")
    print(f"  {OUT / 'matched_pairs_top_motifs.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
