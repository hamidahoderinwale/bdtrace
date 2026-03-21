#!/usr/bin/env python3
"""
Compositionality analysis: edit pattern reuse across SWE-bench Lite instances.

Measures whether agents solve problems by recomposing a small vocabulary of
recurring edit n-grams (bigrams + trigrams on AST token sequences).

Semantic enrichment: lift-based discriminative pattern mining (Brin et al. 1997).
A pattern is semantically specialized if lift(pattern → fix_type) > min_lift,
i.e. it appears significantly more often in one fix type than base rate predicts.
Cross-model overlap is computed over high-lift patterns only, separating
discriminative primitives from structural boilerplate.

Produces:
  1. compositionality_curve.png        — coverage vs. vocabulary size (reuse curve)
  2. model_vocab_overlap.png           — Jaccard overlap: all top-K patterns
  3. model_vocab_overlap_semantic.png  — Jaccard overlap: high-lift patterns only
  4. top_patterns.csv                  — patterns ranked by frequency + lift
  5. semantic_patterns.csv             — high-lift patterns with dominant fix type

Usage:
    uv run python scripts/analyze_compositionality.py
    uv run python scripts/analyze_compositionality.py --topk 30 --ngram 3 --min-lift 2.0
"""

import argparse
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DS_DIR    = ROOT / "output" / "datasets" / "swe_bench_lite_resolved"
TRAJS     = ROOT / "output" / "datasets" / "cross_agent_all" / "all_trajectories.jsonl"
PLOTS_OUT = ROOT / "notebooks" / "plots" / "cross_agent_all"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

# Wong colorblind-safe
MODEL_COLORS = {
    "GPT-4":         "#0072B2",
    "Claude 3.5":    "#E69F00",
    "GPT-4o":        "#009E73",
    "Claude 3 Opus": "#CC79A7",
}
SHORT = {"Claude 3.5": "C3.5", "GPT-4": "G4", "GPT-4o": "G4o", "Claude 3 Opus": "C3O"}


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------

def _ngrams(seq: list[str], n: int):
    it = iter(seq)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for tok in it:
        window = window[1:] + (tok,)
        yield window


def extract_ngrams(tokens: list[str], ns: tuple[int, ...] = (2, 3)) -> list[tuple]:
    out = []
    for n in ns:
        out.extend(_ngrams(tokens, n))
    return out


def parse_tokens(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Compositionality curve (overall)
# ---------------------------------------------------------------------------

def build_reuse_curve(
    all_ngrams_per_instance: list[list[tuple]],
    topk_values: list[int],
) -> pd.DataFrame:
    """
    For each top-K vocabulary size, compute mean compositional coverage per instance.
    Coverage = fraction of an instance's n-grams that appear in the global top-K.
    """
    global_counts: Counter = Counter()
    for ngrams in all_ngrams_per_instance:
        global_counts.update(set(ngrams))  # count instances, not occurrences

    rows = []
    for k in topk_values:
        vocab = set(p for p, _ in global_counts.most_common(k))
        coverages = []
        for ngrams in all_ngrams_per_instance:
            if not ngrams:
                continue
            unique = set(ngrams)
            coverages.append(len(unique & vocab) / len(unique))
        rows.append({"topk": k, "mean_coverage": np.mean(coverages), "std_coverage": np.std(coverages)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-model vocabulary overlap
# ---------------------------------------------------------------------------

def model_vocab_overlap(
    model_ngrams: dict[str, list[list[tuple]]],
    k: int = 20,
) -> pd.DataFrame:
    """
    For each model, compute its top-K global vocabulary (patterns appearing
    across the most instances). Return pairwise Jaccard overlap.
    """
    model_vocabs: dict[str, set] = {}
    for model, per_instance in model_ngrams.items():
        counts: Counter = Counter()
        for ngrams in per_instance:
            counts.update(set(ngrams))
        model_vocabs[model] = set(p for p, _ in counts.most_common(k))

    models = list(model_vocabs.keys())
    matrix = np.zeros((len(models), len(models)))
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            a, b = model_vocabs[ma], model_vocabs[mb]
            matrix[i, j] = len(a & b) / len(a | b) if (a | b) else 0.0

    return pd.DataFrame(matrix, index=models, columns=models)


# ---------------------------------------------------------------------------
# Lift-based semantic specialization (Brin et al. 1997)
# ---------------------------------------------------------------------------

def compute_lift(
    all_ngrams_per_instance: list[list[tuple]],
    fix_types: list[str],
    min_support: int = 5,
    min_lift: float = 2.0,
) -> pd.DataFrame:
    """
    For each (pattern, fix_type) pair compute lift:
        lift = P(fix_type | pattern) / P(fix_type)

    A pattern is semantically specialized if max_ft(lift) > min_lift AND
    it appears in at least min_support instances.

    Returns a DataFrame of high-lift patterns with their dominant fix type,
    lift score, support, and conditional probability.
    """
    n = len(all_ngrams_per_instance)
    assert len(fix_types) == n

    # Base rates P(fix_type)
    ft_counts: Counter = Counter(fix_types)
    base_rate = {ft: c / n for ft, c in ft_counts.items()}

    # Pattern co-occurrence with fix_type: count instances
    # pattern_ft[(pattern, ft)] = number of instances with this pattern and fix_type
    pattern_support: Counter = Counter()
    pattern_ft: Counter = Counter()

    for ngrams, ft in zip(all_ngrams_per_instance, fix_types):
        unique = set(ngrams)
        for pat in unique:
            pattern_support[pat] += 1
            pattern_ft[(pat, ft)] += 1

    # Compute lift for each (pattern, fix_type) pair
    rows = []
    for (pat, ft), co_count in pattern_ft.items():
        supp = pattern_support[pat]
        if supp < min_support:
            continue
        p_ft_given_pat = co_count / supp
        lift = p_ft_given_pat / base_rate[ft]
        rows.append({
            "pattern":      " → ".join(pat),
            "pattern_tuple": pat,
            "fix_type":     ft,
            "support":      supp,
            "co_count":     co_count,
            "p_ft_given_pat": round(p_ft_given_pat, 3),
            "base_rate":    round(base_rate[ft], 3),
            "lift":         round(lift, 3),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep only the dominant fix_type per pattern (highest lift)
    df = df.sort_values("lift", ascending=False)
    df = df.drop_duplicates(subset="pattern_tuple", keep="first")

    # Filter to high-lift patterns
    df = df[df["lift"] >= min_lift].copy()
    return df.sort_values("lift", ascending=False).reset_index(drop=True)


def cosine_model_overlap(
    model_ngrams: dict[str, list[list[tuple]]],
    model_fix_types: dict[str, list[str]],
    min_lift: float = 2.0,
    min_support: int = 5,
) -> tuple[pd.DataFrame, dict[str, Counter]]:
    """
    Cosine similarity over frequency-weighted semantic pattern vectors.
    Only high-lift patterns are included; frequency = number of instances
    in which the pattern appears (support count, not raw occurrences).
    """
    # Build per-model frequency vectors over high-lift patterns
    model_freq: dict[str, Counter] = {}
    for model, per_instance in model_ngrams.items():
        fts = model_fix_types.get(model, [])
        if len(fts) != len(per_instance):
            continue
        lift_df = compute_lift(per_instance, fts,
                               min_support=min_support, min_lift=min_lift)
        if lift_df.empty:
            model_freq[model] = Counter()
            continue
        high_lift = set(lift_df["pattern_tuple"].tolist())
        freq: Counter = Counter()
        for ngrams in per_instance:
            for pat in set(ngrams):
                if pat in high_lift:
                    freq[pat] += 1
        model_freq[model] = freq

    # Shared vocabulary
    vocab = sorted({p for freq in model_freq.values() for p in freq})
    if not vocab:
        models = list(model_freq.keys())
        empty = np.zeros((len(models), len(models)))
        return pd.DataFrame(empty, index=models, columns=models), model_freq

    vocab_idx = {p: i for i, p in enumerate(vocab)}
    models = list(model_freq.keys())
    vecs = np.zeros((len(models), len(vocab)))
    for i, model in enumerate(models):
        for pat, cnt in model_freq[model].items():
            if pat in vocab_idx:
                vecs[i, vocab_idx[pat]] = cnt

    # Cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vecs / norms
    matrix = normed @ normed.T

    return pd.DataFrame(matrix, index=models, columns=models), model_freq


def semantic_model_overlap(
    model_ngrams: dict[str, list[list[tuple]]],
    model_fix_types: dict[str, list[str]],
    min_lift: float = 2.0,
    min_support: int = 5,
) -> tuple[pd.DataFrame, dict[str, set]]:
    """
    For each model, compute its high-lift (semantically specialized) pattern set.
    Returns pairwise Jaccard overlap over those sets.
    """
    model_semantic_vocab: dict[str, set] = {}
    for model, per_instance in model_ngrams.items():
        fts = model_fix_types.get(model, [])
        if len(fts) != len(per_instance):
            continue
        lift_df = compute_lift(per_instance, fts, min_support=min_support, min_lift=min_lift)
        if lift_df.empty:
            model_semantic_vocab[model] = set()
        else:
            model_semantic_vocab[model] = set(lift_df["pattern_tuple"].tolist())

    models = list(model_semantic_vocab.keys())
    matrix = np.zeros((len(models), len(models)))
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            a, b = model_semantic_vocab[ma], model_semantic_vocab[mb]
            matrix[i, j] = len(a & b) / len(a | b) if (a | b) else 0.0

    return pd.DataFrame(matrix, index=models, columns=models), model_semantic_vocab


# ---------------------------------------------------------------------------
# Top patterns table
# ---------------------------------------------------------------------------

def top_patterns_table(
    all_ngrams_per_instance: list[list[tuple]],
    instance_ids: list[str],
    k: int = 20,
) -> pd.DataFrame:
    global_counts: Counter = Counter()
    pattern_examples: dict[tuple, str] = {}
    for iid, ngrams in zip(instance_ids, all_ngrams_per_instance):
        for ng in set(ngrams):
            global_counts[ng] += 1
            if ng not in pattern_examples:
                pattern_examples[ng] = iid

    rows = []
    for pattern, count in global_counts.most_common(k):
        rows.append({
            "pattern":  " → ".join(pattern),
            "n_instances": count,
            "pct_instances": round(100 * count / len(all_ngrams_per_instance), 1),
            "example_instance": pattern_examples[pattern],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reuse_curve(curve: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["topk"], curve["mean_coverage"], color="#0072B2", linewidth=2)
    ax.fill_between(
        curve["topk"],
        curve["mean_coverage"] - curve["std_coverage"],
        curve["mean_coverage"] + curve["std_coverage"],
        alpha=0.15, color="#0072B2",
    )
    ax.set_xlabel("Vocabulary size (top-K edit patterns)", fontsize=10)
    ax.set_ylabel("Mean compositional coverage", fontsize=10)
    ax.set_title("Edit pattern reuse across SWE-bench Lite instances", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.7, color="gray", linewidth=0.8, linestyle="--")
    ax.text(curve["topk"].max() * 0.98, 0.715, "70%", ha="right", fontsize=8, color="gray")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


def plot_model_overlap(overlap: pd.DataFrame, out_path: Path, title: str = "Top-K edit vocabulary overlap by model pair") -> None:
    models = list(overlap.index)
    n = len(models)
    short_labels = [SHORT.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(overlap.values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(short_labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            v = overlap.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if v > 0.6 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Jaccard overlap", fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk",     type=int,   default=20,  help="Top-K for overlap heatmap")
    parser.add_argument("--ngram",    type=int,   default=3,   help="Max n-gram size (2 or 3)")
    parser.add_argument("--min-lift", type=float, default=2.0, help="Minimum lift for semantic specialization")
    parser.add_argument("--min-sup",  type=int,   default=5,   help="Minimum instance support for lift filter")
    args = parser.parse_args()

    ns = tuple(range(2, args.ngram + 1))

    # ── Load fix types ───────────────────────────────────────────────────────
    ft_path = DS_DIR / "fix_types.json"
    fix_type_map: dict[str, str] = {}
    if ft_path.exists():
        with open(ft_path) as f:
            ft_data = json.load(f)
        fix_type_map = {r["instance_id"]: r["fix_type"] for r in ft_data["results"]}

    # ── Overall compositionality from test.parquet ──────────────────────────
    print("Loading token sequences from test.parquet...")
    df = pd.read_parquet(DS_DIR / "test.parquet")
    instance_ids = df["instance_id"].tolist()
    all_ngrams: list[list[tuple]] = []
    for raw in df["tokens"]:
        toks = parse_tokens(raw)
        all_ngrams.append(extract_ngrams(toks, ns))

    fix_types_all = [fix_type_map.get(iid, "unknown") for iid in instance_ids]

    print(f"  {len(all_ngrams)} instances, {sum(len(x) for x in all_ngrams):,} total n-grams")

    topk_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    curve = build_reuse_curve(all_ngrams, topk_values)
    print("\nReuse curve:")
    print(curve.to_string(index=False))
    plot_reuse_curve(curve, PLOTS_OUT / "compositionality_curve.png")

    # Top patterns table
    top = top_patterns_table(all_ngrams, instance_ids, k=args.topk)
    top.to_csv(PLOTS_OUT / "top_patterns.csv", index=False)
    print(f"\nTop-{args.topk} patterns:")
    print(top[["pattern", "n_instances", "pct_instances"]].to_string(index=False))

    # ── Lift-based semantic specialization (overall) ─────────────────────────
    if fix_type_map:
        print(f"\nComputing lift-based semantic patterns (min_lift={args.min_lift}, min_sup={args.min_sup})...")
        lift_df = compute_lift(all_ngrams, fix_types_all,
                               min_support=args.min_sup, min_lift=args.min_lift)
        if not lift_df.empty:
            lift_df.drop(columns="pattern_tuple").to_csv(
                PLOTS_OUT / "semantic_patterns.csv", index=False)
            print(f"  {len(lift_df)} high-lift patterns found")
            print(lift_df[["pattern", "fix_type", "support", "lift"]].head(20).to_string(index=False))
        else:
            print("  No patterns met lift threshold.")

    # ── Cross-model overlap from all_trajectories.jsonl ─────────────────────
    if not TRAJS.exists():
        print(f"\nSkipping cross-model overlap: {TRAJS.name} not found.")
        return

    from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

    print("\nLoading per-model patches from all_trajectories.jsonl...")
    model_ngrams: dict[str, list[list[tuple]]] = {}
    model_instance_ids: dict[str, list[str]] = {}

    with open(TRAJS) as f:
        for line in f:
            rec = json.loads(line)
            iid = rec["instance_id"]
            for model, traj in rec["models"].items():
                if not traj.get("resolved") or not traj.get("patch"):
                    continue
                toks = patch_to_ast_sequence(traj["patch"])
                ngrams = extract_ngrams(toks, ns)
                model_ngrams.setdefault(model, []).append(ngrams)
                model_instance_ids.setdefault(model, []).append(iid)

    print("  Resolved instances per model:")
    for m, v in model_ngrams.items():
        print(f"    {m}: {len(v)}")

    # All-patterns overlap
    overlap = model_vocab_overlap(model_ngrams, k=args.topk)
    print(f"\nTop-{args.topk} vocabulary overlap (Jaccard, all patterns):")
    print(overlap.round(3).to_string())
    plot_model_overlap(overlap, PLOTS_OUT / "model_vocab_overlap.png",
                       title=f"Top-{args.topk} edit vocabulary overlap (all patterns)")

    # Semantic (high-lift) overlap
    if fix_type_map:
        model_fix_types = {
            model: [fix_type_map.get(iid, "unknown") for iid in iids]
            for model, iids in model_instance_ids.items()
        }
        cos_overlap, model_freq = cosine_model_overlap(
            model_ngrams, model_fix_types,
            min_lift=args.min_lift, min_support=args.min_sup,
        )
        print(f"\nSemantic vocabulary overlap (cosine, lift≥{args.min_lift}):")
        print(cos_overlap.round(3).to_string())
        print("\nSemantic vocabulary size per model (high-lift patterns):")
        for m, freq in model_freq.items():
            print(f"  {SHORT.get(m, m)}: {len(freq)} patterns")
        plot_model_overlap(
            cos_overlap, PLOTS_OUT / "model_vocab_overlap_semantic.png",
            title=f"Semantic edit vocabulary — cosine similarity (lift≥{args.min_lift})",
        )

        # Top semantic patterns per model
        print(f"\nTop-5 high-lift patterns per model:")
        for model, per_instance in model_ngrams.items():
            fts = model_fix_types.get(model, [])
            ldf = compute_lift(per_instance, fts,
                               min_support=args.min_sup, min_lift=args.min_lift)
            if ldf.empty:
                print(f"  {SHORT.get(model, model)}: (none)")
                continue
            top5 = ldf[["pattern", "fix_type", "lift"]].head(5)
            print(f"  {SHORT.get(model, model)}:")
            for _, row in top5.iterrows():
                print(f"    {row['pattern']}  [{row['fix_type']}, lift={row['lift']}]")


if __name__ == "__main__":
    main()
