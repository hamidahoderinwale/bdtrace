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
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
import altair as alt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, VERMILLION, SKY, GRAY, NEAR_BLACK
register()
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
    rows = []
    for r in results:
        pair_label = (
            f"{r['agent_a']} vs {r['agent_b']}  "
            f"({'same family' if r['same_family'] else 'cross family'})"
        )
        pair_color = PAIR_COLORS.get((r["agent_a"], r["agent_b"]), GRAY)
        for mr in r["motif_results"]:
            pval = mr["p_value"]
            if np.isnan(pval):
                continue
            rows.append({
                "pair_label": pair_label,
                "mean_delta": mr["mean_delta"],
                "neg_log_p": -np.log10(pval + 1e-12),
                "significant": mr["significant_fdr_5"],
                "point_color": pair_color if mr["significant_fdr_5"] else GRAY,
                "point_size": 50 if mr["significant_fdr_5"] else 20,
            })
    df = pd.DataFrame(rows)

    # Unified y-axis across all panels for visual comparability.
    y_max = max(2.0, float(df["neg_log_p"].max()) * 1.10)

    def slug(s: str) -> str:
        return s.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

    for r in results:
        pair_label = (
            f"{r['agent_a']} vs {r['agent_b']}  "
            f"({'same family' if r['same_family'] else 'cross family'})"
        )
        panel_caption = (
            f"{r['n_tied_outcome_tasks']} tied tasks  ·  "
            f"{r['n_significant_fdr_5']}/{r['n_motifs_tested']} motifs FDR<0.05"
        )
        sub = df[df["pair_label"] == pair_label]

        points = (
            alt.Chart(sub)
            .mark_point(opacity=0.6, filled=True)
            .encode(
                x=alt.X(
                    "mean_delta:Q",
                    axis=alt.Axis(title="mean motif-frequency difference  (agent_a − agent_b)",
                                  domain=False, ticks=False, labelFontSize=10),
                ),
                y=alt.Y(
                    "neg_log_p:Q",
                    scale=alt.Scale(domain=[0, y_max]),
                    axis=alt.Axis(title="−log10(p-value)",
                                  domain=False, ticks=False, labelFontSize=10),
                ),
                color=alt.Color("point_color:N", scale=None, legend=None),
                size=alt.Size("point_size:Q", scale=None, legend=None),
            )
            .properties(
                width=380,
                height=280,
                title=alt.TitleParams(
                    text=[pair_label, panel_caption],
                    fontSize=12,
                    color="#111111",
                    anchor="start",
                    subtitleFontSize=10,
                    subtitleColor="#666666",
                ),
            )
            .configure_view(strokeWidth=0)
        )

        panel_path = out_path.parent / (out_path.stem + "_" + slug(f"{r['agent_a']}_vs_{r['agent_b']}") + ".png")
        points.save(str(panel_path), scale_factor=2)
        print(f"  Saved: {panel_path.name}")


def plot_top_motifs(results: list[dict], out_path: Path, top_n: int = 8) -> None:
    # No truncation: render full motif names with extra label width.
    def fmt(m: str) -> str:
        parts = m.split("+")
        if len(parts) <= 3:
            return m.replace("+", " → ")
        return f"{parts[0]} → ... → {parts[-1]}  ({len(parts)} atoms)"

    def slug(a: str, b: str) -> str:
        return f"{a}_vs_{b}".replace(" ", "_").replace("(", "").replace(")", "")

    for r in results:
        pair_color = PAIR_COLORS.get((r["agent_a"], r["agent_b"]), GRAY)
        family_tag = "same family" if r["same_family"] else "cross family"

        all_tested = [mr for mr in r["motif_results"] if not np.isnan(mr["p_value"])]
        all_tested.sort(key=lambda mr: abs(mr["mean_delta"]), reverse=True)
        top = all_tested[:top_n]

        panel_path = out_path.parent / (out_path.stem + "_" + slug(r["agent_a"], r["agent_b"]) + ".png")

        if not top:
            chart = (
                alt.Chart(pd.DataFrame({"x": [0], "y": [0], "label": ["no testable tokens"]}))
                .mark_text(fontSize=11, color=GRAY)
                .encode(
                    x=alt.X("x:Q", axis=None),
                    y=alt.Y("y:Q", axis=None),
                    text="label:N",
                )
                .properties(
                    width=440,
                    height=top_n * 32,
                    title=alt.TitleParams(
                        text=f"{r['agent_a']} vs {r['agent_b']}  ({family_tag})",
                        fontSize=12,
                        color="#111111",
                        anchor="start",
                    ),
                )
                .configure_view(strokeWidth=0)
            )
            chart.save(str(panel_path), scale_factor=2)
            print(f"  Saved: {panel_path.name}")
            continue

        rows = []
        for pos, mr in enumerate(top):
            label = fmt(mr["motif"])
            if mr["significant_fdr_5"]:
                label = label + "  *"
            direction = "favors_a" if mr["mean_delta"] > 0 else "favors_b"
            rows.append({
                "motif_label": label,
                "mean_delta": mr["mean_delta"],
                "significant": mr["significant_fdr_5"],
                "direction": direction,
                "sort_pos": pos,
            })
        df = pd.DataFrame(rows)
        motif_order = df.sort_values("sort_pos")["motif_label"].tolist()

        bars = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "motif_label:N",
                    sort=motif_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=10, labelLimit=420),
                ),
                x=alt.X(
                    "mean_delta:Q",
                    axis=alt.Axis(title="mean motif-frequency difference  (agent_a − agent_b)",
                                  domain=False, ticks=False, labelFontSize=10),
                ),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["favors_a", "favors_b"],
                        range=[pair_color, VERMILLION],
                    ),
                    legend=None,
                ),
            )
            .properties(
                width=440,
                height=top_n * 32,
                title=alt.TitleParams(
                    text=f"{r['agent_a']} vs {r['agent_b']}  ({family_tag})",
                    fontSize=12,
                    color="#111111",
                    anchor="start",
                ),
            )
            .configure_view(strokeWidth=0)
        )

        bars.save(str(panel_path), scale_factor=2)
        print(f"  Saved: {panel_path.name}")


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
