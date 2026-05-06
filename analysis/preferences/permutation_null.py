"""Permutation null for the heritability gap.

Observed: same-family (GPT-4 x GPT-4o) motif-JSD is lower than cross-family
motif-JSD, with a gap of ~0.06 on the full corpus and ~0.12 on 1/3-resolved
tasks. Is the gap above chance?

Null: agent labels don't carry real information. Shuffle agent labels
across trajectories, recompute same-family vs cross-family gap, repeat.
p-value = fraction of permutations where shuffled-gap >= observed-gap.

Computed at:
  - aggregate (all 867 trajectories)
  - per difficulty bucket (0/3, 1/3, 2/3, 3/3)

Outputs:
  output/paper2_pilot/permutation_null.json
  output/paper2_pilot/permutation_null.png

Usage:
    python -m analysis.preferences.permutation_null [--n-perm 1000]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import sys
import altair as alt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.theme import register, BLUE, ORANGE, GREEN, VERMILLION, SKY, GRAY, NEAR_BLACK
register()
import numpy as np
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

BUCKET_LABEL = {0: "0/3", 1: "1/3", 2: "2/3", 3: "3/3"}
BUCKET_ORDER = ["0/3", "1/3", "2/3", "3/3"]


def load_records() -> list[dict]:
    records = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_difficulty() -> dict[str, int]:
    out = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["instance_id"]] = int(row["n_resolved"])
    return out


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a, b, base=2)) ** 2


def heritability_gap(records: list[dict], labels: list[str]) -> float:
    """Given per-record agent labels, compute same-family vs cross-family gap (motifs only).

    Same-family = GPT-4 x GPT-4o. Cross-family = pairs including Claude-3.5.
    Gap = mean(cross) - same.
    """
    per_agent: dict[str, Counter] = {}
    for r, lbl in zip(records, labels):
        per_agent.setdefault(lbl, Counter()).update(r["bpe"])

    agents_present = [a for a in ["Claude-3.5", "GPT-4", "GPT-4o"] if a in per_agent]
    if len(agents_present) < 3:
        return float("nan")

    vocab_motifs = sorted({t for c in per_agent.values() for t in c if "+" in t})
    if not vocab_motifs:
        return float("nan")

    dist = {}
    for a in agents_present:
        c = per_agent[a]
        total = sum(c[t] for t in vocab_motifs)
        if total == 0:
            return float("nan")
        dist[a] = np.array([c.get(t, 0) / total for t in vocab_motifs])

    pairs = {
        f"{a}__{b}": jsd(dist[a], dist[b])
        for a, b in combinations(sorted(agents_present), 2)
    }
    same = pairs.get("GPT-4__GPT-4o", float("nan"))
    cross = [v for k, v in pairs.items() if "Claude" in k]
    if not cross or np.isnan(same):
        return float("nan")
    return float(np.mean(cross) - same)


def run_permutation(
    records: list[dict], n_perm: int, rng: np.random.Generator
) -> dict:
    true_labels = [r["agent"] for r in records]
    observed = heritability_gap(records, true_labels)

    labels_arr = np.array(true_labels)
    null_gaps = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = rng.permutation(labels_arr).tolist()
        null_gaps[i] = heritability_gap(records, shuffled)

    valid = ~np.isnan(null_gaps)
    null_valid = null_gaps[valid]
    p_value = float((null_valid >= observed).sum() + 1) / (len(null_valid) + 1)
    return {
        "n_records": len(records),
        "observed_gap": observed,
        "n_permutations": int(valid.sum()),
        "null_mean": float(null_valid.mean()) if len(null_valid) else float("nan"),
        "null_std": float(null_valid.std()) if len(null_valid) else float("nan"),
        "null_q95": float(np.percentile(null_valid, 95)) if len(null_valid) else float("nan"),
        "p_value": p_value,
        "null_distribution": null_valid.tolist(),
    }


def plot_null(results: dict, out_path: Path) -> None:
    keys = ["aggregate"] + BUCKET_ORDER
    valid_keys = [k for k in keys if k in results and not np.isnan(results[k]["observed_gap"])]

    panels = []
    for key in valid_keys:
        r = results[key]
        null = np.array(r["null_distribution"])
        obs = r["observed_gap"]
        q95 = r["null_q95"]
        n_records = r["n_records"]
        p_value = r["p_value"]

        # Pre-compute histogram bins
        counts, bin_edges = np.histogram(null, bins=30)
        bin_df = pd.DataFrame({
            "x0": bin_edges[:-1],
            "x1": bin_edges[1:],
            "count": counts,
        })

        bars = (
            alt.Chart(bin_df)
            .mark_bar(color="#cccccc", stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X("x0:Q", axis=alt.Axis(
                    title="same-family similarity advantage",
                    domain=False, ticks=False, labelFontSize=10,
                )),
                x2="x1:Q",
                y=alt.Y("count:Q", axis=alt.Axis(
                    title=None, domain=False, ticks=False, labelFontSize=10,
                )),
            )
        )

        rule_obs = (
            alt.Chart(pd.DataFrame({"x": [obs], "label": [f"observed = {obs:.3f}"]}))
            .mark_rule(color=NEAR_BLACK, strokeWidth=2)
            .encode(x="x:Q", tooltip="label:N")
        )

        rule_q95 = (
            alt.Chart(pd.DataFrame({"x": [q95], "label": [f"chance 95th pct = {q95:.3f}"]}))
            .mark_rule(color=GRAY, strokeWidth=1, opacity=0.8, strokeDash=[4, 4])
            .encode(x="x:Q", tooltip="label:N")
        )

        panel = (
            (bars + rule_obs + rule_q95)
            .properties(
                width=160,
                height=180,
                title=alt.TitleParams(
                    text=f"{key} (n={n_records}, p={p_value:.4f})",
                    fontSize=10,
                    color="#111111",
                    anchor="start",
                ),
            )
        )
        panels.append(panel)

    chart = (
        alt.hconcat(*panels, spacing=20)
        .properties(
            title=alt.TitleParams(
                text="Permutation null: same-family divergence",
                fontSize=13,
                color="#111111",
                anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axisY(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    records = load_records()
    difficulty = load_difficulty()

    print(f"Loaded {len(records)} records; {args.n_perm} permutations per slice")

    results: dict = {}

    print("\n=== aggregate ===")
    results["aggregate"] = run_permutation(records, args.n_perm, rng)
    r = results["aggregate"]
    print(f"  observed gap = {r['observed_gap']:.4f}")
    print(f"  null mean    = {r['null_mean']:.4f} (sd {r['null_std']:.4f})")
    print(f"  null 95th    = {r['null_q95']:.4f}")
    print(f"  p-value      = {r['p_value']:.4f}")

    for b in BUCKET_ORDER:
        subset = [r for r in records if BUCKET_LABEL.get(difficulty.get(r["instance_id"], -1)) == b]
        if not subset:
            print(f"\n=== {b} === (empty, skipping)")
            continue
        print(f"\n=== {b} === (n={len(subset)})")
        results[b] = run_permutation(subset, args.n_perm, rng)
        r = results[b]
        print(f"  observed gap = {r['observed_gap']:.4f}")
        print(f"  null mean    = {r['null_mean']:.4f} (sd {r['null_std']:.4f})")
        print(f"  null 95th    = {r['null_q95']:.4f}")
        print(f"  p-value      = {r['p_value']:.4f}")

    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "null_distribution"}
        for k, v in results.items()
    }
    (OUT / "permutation_null.json").write_text(json.dumps(serializable, indent=2))
    plot_null(results, OUT / "permutation_null.png")

    print(f"\nSaved:\n  {OUT / 'permutation_null.json'}\n  {OUT / 'permutation_null.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
