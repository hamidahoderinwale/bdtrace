#!/usr/bin/env python3
"""
Build fix mechanism sequences for all resolved traces.

Each patch → sequence of mechanism labels, one per diff hunk.
Then run FIM on the sequences to find frequent mechanism patterns.

This is the motif-sequence analogy applied to fix intent:
  events → motif tokens → frequent sequences
  diff hunks → mechanism labels → frequent sequences

Outputs:
  output/intent_sequences/sequences.json   — per-instance label sequences
  output/intent_sequences/fim_patterns.json — frequent mechanism patterns
  output/intent_sequences/fig_*.png        — distribution and pattern figures

Usage:
  uv run python scripts/build_intent_sequences.py
  uv run python scripts/build_intent_sequences.py --limit 50   # quick test
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _env in [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
        except ImportError:
            pass

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "intent_sequences"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def run_fim(sequences: list[list[str]], min_support: float = 0.05) -> list[dict]:
    """
    FIM on mechanism sequences: find frequent itemsets of mechanism labels.
    Each sequence is treated as a set (order-agnostic, like edit certs).
    """
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    te_array = te.fit(sequences).transform(sequences)
    df = pd.DataFrame(te_array, columns=te.columns_)
    fi = fpgrowth(df, min_support=min_support, use_colnames=True)
    patterns = [
        {"itemset": sorted(row["itemsets"]), "support": float(row["support"])}
        for _, row in fi.iterrows()
    ]
    return sorted(patterns, key=lambda p: -p["support"])


def fig_label_distribution(all_labels: list[str], output_dir: Path):
    counts = Counter(all_labels)
    labels = [k for k, _ in counts.most_common()]
    vals = [counts[k] for k in labels]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.35)
    style_panel(ax)

    xs = np.arange(len(labels))
    ax.bar(xs, [v / total for v in vals], color=TEAL, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Fraction of all chunks", fontsize=9)
    ax.set_title("Fix mechanism label distribution across all diff hunks",
                 fontsize=11, pad=6, fontweight="normal")

    for xi, (v, t) in enumerate(zip(vals, [v / total for v in vals])):
        ax.text(xi, t + 0.003, str(v), ha="center", va="bottom", fontsize=7, color=NAVY)

    fig.savefig(output_dir / "fig1_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_label_distribution.png")


def fig_sequence_length(seq_lengths: list[int], output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.15)
    style_panel(ax)

    counts = Counter(seq_lengths)
    xs = sorted(counts)
    ax.bar(xs, [counts[x] for x in xs], color=TEAL, alpha=0.85)
    ax.set_xlabel("Chunks per patch (sequence length)", fontsize=9)
    ax.set_ylabel("Number of instances", fontsize=9)
    ax.set_title("Distribution of patch complexity (mechanism sequence length)",
                 fontsize=11, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig2_sequence_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_sequence_length.png")


def fig_fim_patterns(patterns: list[dict], top_n: int = 20, output_dir: Path = None):
    top = patterns[:top_n]
    labels = [" + ".join(p["itemset"]) for p in top]
    supports = [p["support"] for p in top]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
    fig.subplots_adjust(left=0.45, right=0.97)
    style_panel(ax)

    ys = np.arange(len(top))
    colors = [TEAL if len(p["itemset"]) == 1 else ORANGE if len(p["itemset"]) == 2
              else NAVY for p in top]
    ax.barh(ys, supports, color=colors, alpha=0.85)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Support (fraction of instances)", fontsize=9)
    ax.set_title(f"Top {top_n} frequent mechanism patterns (FIM)",
                 fontsize=11, pad=6, fontweight="normal")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TEAL, alpha=0.85, label="Single mechanism"),
        Patch(facecolor=ORANGE, alpha=0.85, label="2-mechanism pattern"),
        Patch(facecolor=NAVY, alpha=0.85, label="3+ mechanism pattern"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False)

    fig.savefig(output_dir / "fig3_fim_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_fim_patterns.png")


def fig_pass_rate_by_mechanism(
    sequences: dict[str, list[str]],
    pass_fail: dict[str, bool],
    output_dir: Path,
):
    """Pass rate for instances that contain each mechanism label."""
    from representations.inferred.fix_type.chunk_intent import MECHANISM_LABELS

    rows = []
    for label in MECHANISM_LABELS:
        instances = [iid for iid, seq in sequences.items() if label in seq]
        if not instances:
            continue
        pf = [pass_fail[iid] for iid in instances if iid in pass_fail]
        if not pf:
            continue
        rows.append({
            "label": label,
            "n": len(instances),
            "pass_rate": np.mean(pf),
        })
    rows.sort(key=lambda r: -r["pass_rate"])

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.35)
    style_panel(ax)

    xs = np.arange(len(rows))
    colors = [TEAL if r["pass_rate"] >= 0.25 else ORANGE if r["pass_rate"] >= 0.15
              else GRAY for r in rows]
    ax.bar(xs, [r["pass_rate"] for r in rows], color=colors, alpha=0.85)

    for xi, r in enumerate(rows):
        ax.text(xi, r["pass_rate"] + 0.01, f"n={r['n']}",
                ha="center", va="bottom", fontsize=7, color=NAVY)

    ax.axhline(0.23, color=NAVY, linewidth=0.8, linestyle=":", label="Baseline (23%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_title("Pass rate for instances containing each fix mechanism",
                 fontsize=11, pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False)

    fig.savefig(output_dir / "fig4_pass_rate_by_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_pass_rate_by_mechanism.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--traces", type=str,
        default="output/resolved_traces_lite_full.jsonl",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from configs.dspy_config import configure_dspy
    configure_dspy(model=args.model)

    from representations.inferred.fix_type.chunk_intent import (
        ChunkIntentModule, MECHANISM_LABELS,
    )

    sequences_path = OUTPUT_DIR / "sequences.json"
    existing: dict[str, list[str]] = {}
    if args.resume and sequences_path.exists():
        with open(sequences_path) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already labeled")

    classifier = ChunkIntentModule()

    traces_path = ROOT / args.traces
    print(f"Loading traces from {traces_path}...")

    results: dict[str, list[str]] = dict(existing)
    n_processed = 0

    with open(traces_path) as f:
        for line in f:
            if args.limit and n_processed >= args.limit:
                break
            trace = json.loads(line)
            iid = trace["instance_id"]
            if iid in existing:
                continue
            seq = classifier.label_trace(trace)
            results[iid] = seq
            n_processed += 1
            if n_processed % 10 == 0:
                print(f"  {n_processed} processed, last: {iid} → {seq}")
                with open(sequences_path, "w") as f2:
                    json.dump(results, f2, indent=2)

    with open(sequences_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sequences.json ({len(results)} instances)")

    # Stats
    all_labels = [label for seq in results.values() for label in seq]
    seq_lengths = [len(seq) for seq in results.values()]
    print(f"\nTotal chunks labeled: {len(all_labels)}")
    print(f"Mean sequence length: {np.mean(seq_lengths):.1f}")
    print(f"Label distribution: {Counter(all_labels).most_common()}")

    # FIM on sequences (treat each sequence as a set)
    print("\nRunning FIM on mechanism sequences...")
    sequences_as_sets = [list(set(seq)) for seq in results.values() if seq]
    patterns = run_fim(sequences_as_sets, min_support=0.05)
    print(f"Found {len(patterns)} frequent patterns at support >= 0.05")
    for p in patterns[:15]:
        print(f"  {p['itemset']}: support={p['support']:.3f}")

    with open(OUTPUT_DIR / "fim_patterns.json", "w") as f:
        json.dump({"n_instances": len(results), "patterns": patterns}, f, indent=2)
    print("Saved fim_patterns.json")

    # Load pass/fail for pass rate figure
    try:
        fix_df = pd.read_parquet(
            ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
        )[["instance_id", "passed"]]
        pass_fail = dict(zip(fix_df["instance_id"], fix_df["passed"]))
    except Exception:
        pass_fail = {}

    print("\nGenerating figures...")
    fig_label_distribution(all_labels, OUTPUT_DIR)
    fig_sequence_length(seq_lengths, OUTPUT_DIR)
    fig_fim_patterns(patterns, top_n=min(20, len(patterns)), output_dir=OUTPUT_DIR)
    if pass_fail:
        fig_pass_rate_by_mechanism(results, pass_fail, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
