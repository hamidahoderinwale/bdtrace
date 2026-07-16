#!/usr/bin/env python3
"""
Recompute d_motifs using corpus-level PrefixSpan and patch distances.parquet.

Two sequence sources:
  --source trajectory  (default)
      Typed action sequences from agent .traj files (OPEN_SRC, EDIT_TEST, ...).
      Requires: output/trajectories/.cache/<model>/
  --source ast_edits
      Hunk-local AST node sequences from the canonical patch (DEL_If, ADD_Return, ...).
      Alphabet and pattern lengths emerge entirely from support — nothing is hardcoded.
      Requires: HuggingFace princeton-nlp/SWE-bench_Lite (downloaded on first run).

In both modes, patterns extend until no extension meets --min-support.
There is no max-length cap.

Usage:
    uv run python scripts/recompute_motif_distances.py --source ast_edits
    uv run python scripts/recompute_motif_distances.py --source trajectory --model 20240620_sweagent_claude3.5sonnet
    uv run python scripts/recompute_motif_distances.py --source ast_edits --min-support 0.1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.procedures.corpus_motifs import (
    compress_sequence,
    cosine_distance,
    encode_sequences,
    jaccard_distance,
    mine_corpus_patterns,
    mine_corpus_sequitur,
    tfidf_vectors,
    typed_action_sequence,
)
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence, vocabulary_stats

ROOT = Path(__file__).resolve().parent.parent


def load_ast_edit_sequences(instance_ids: list[str]) -> list[list[str]]:
    """Load raw patches from HF and extract AST edit sequences in canonical order."""
    from datasets import load_dataset
    print("Loading patches from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    patch_map = {row["instance_id"]: row["patch"] for row in ds}
    sequences = []
    missing = 0
    for iid in instance_ids:
        patch = patch_map.get(iid, "")
        if not patch:
            missing += 1
        sequences.append(patch_to_ast_sequence(patch))
    if missing:
        print(f"  Warning: {missing} instances had no patch (empty sequence)")
    return sequences


def load_trajectory_sequences(
    instance_ids: list[str],
    model: str,
) -> list[list[str]]:
    """Load raw trajectory JSONs from .cache and extract typed action sequences."""
    cache_dir = ROOT / "output" / "trajectories" / ".cache" / model
    if not cache_dir.exists():
        print(f"Cache dir not found: {cache_dir}")
        sys.exit(1)
    print(f"Loading trajectories from {cache_dir}")
    sequences = []
    missing = []
    for iid in instance_ids:
        path = cache_dir / f"{iid}.json"
        if not path.exists():
            missing.append(iid)
            sequences.append([])
            continue
        with open(path) as f:
            raw = json.load(f)
        steps = raw.get("trajectory", [])
        sequences.append(typed_action_sequence(steps))
    if missing:
        print(f"  Warning: {len(missing)} instances not in cache (empty sequences)")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="ast_edits",
                        choices=["ast_edits", "trajectory"],
                        help="Sequence source: hunk-local AST edits or agent trajectory actions")
    parser.add_argument("--model", default="20240620_sweagent_claude3.5sonnet",
                        help="Model ID for trajectory source (ignored for ast_edits)")
    parser.add_argument("--min-support", type=float, default=0.08,
                        help="Min fraction of instances a pattern must appear in. "
                             "Patterns extend until no extension meets this — no length cap.")
    parser.add_argument("--use-sequitur", action="store_true",
                        help="Also add Sequitur bigrams/trigrams to the pattern vocabulary")
    parser.add_argument("--weighting", default="tfidf",
                        choices=["tfidf", "binary"],
                        help="tfidf (default): IDF down-weights high-frequency noise patterns. "
                             "binary: raw presence vectors with Jaccard distance.")
    parser.add_argument("--compress", action="store_true", default=True,
                        help="Run-length encode sequences before mining (default: True)")
    parser.add_argument("--no-compress", dest="compress", action="store_false")
    parser.add_argument("--distances", type=Path,
                        default=ROOT / "output" / "distances.parquet")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "output" / "labels.parquet")
    args = parser.parse_args()

    ## Load canonical instance order
    labels = pd.read_parquet(args.labels)
    instance_ids: list[str] = labels["instance_id"].tolist()
    N = len(instance_ids)
    print(f"Canonical index: {N} instances")

    ## Build sequences from chosen source
    if args.source == "ast_edits":
        print("\nSource: hunk-local AST edit sequences")
        sequences = load_ast_edit_sequences(instance_ids)
    else:
        print(f"\nSource: agent trajectory sequences (model={args.model})")
        sequences = load_trajectory_sequences(instance_ids, args.model)

    non_empty = sum(1 for s in sequences if s)
    print(f"Non-empty sequences: {non_empty}/{N}")

    ## Report emergent vocabulary
    vocab = vocabulary_stats(sequences)
    print(f"\nEmergent vocabulary: {len(vocab)} unique tokens")
    print("Top 15 by frequency:")
    for tok, cnt in vocab[:15]:
        print(f"  {tok:30s} {cnt:5d}  ({100*cnt/N:.0f}% of instances)")

    ## Mine corpus patterns — length grows until support drops, no cap
    print(f"\nMining corpus patterns (min_support={args.min_support})...")
    patterns = mine_corpus_patterns(
        sequences,
        min_support=args.min_support,
        compress=args.compress,
    )
    print(f"  PrefixSpan patterns: {len(patterns)}")
    if patterns:
        lengths = [len(p) for p, _ in patterns]
        print(f"  Pattern lengths: min={min(lengths)} mean={np.mean(lengths):.1f} max={max(lengths)}")

    if args.use_sequitur:
        sq_patterns = mine_corpus_sequitur(
            sequences, min_support=args.min_support / 2, compress=args.compress
        )
        seen: set[tuple] = {tuple(p) for p, _ in patterns}
        for p, s in sq_patterns:
            k = tuple(p)
            if k not in seen:
                seen.add(k)
                patterns.append((list(p), s))
        print(f"  Combined (PrefixSpan + Sequitur): {len(patterns)} patterns")

    if not patterns:
        print("No patterns found. Lower --min-support and retry.")
        sys.exit(1)

    print("\nTop 20 patterns by support:")
    for pat, sup in patterns[:20]:
        pct = 100 * sup / N
        print(f"  [{sup:3d} / {pct:.0f}%] {' → '.join(pat)}")

    ## Encode and compute distances
    if args.weighting == "tfidf":
        vectors = tfidf_vectors(sequences, patterns, compress=args.compress)
        dist_fn = cosine_distance
        weighting_label = "TF-IDF + cosine"
    else:
        vectors = encode_sequences(sequences, patterns, compress=args.compress)
        dist_fn = jaccard_distance
        weighting_label = "binary + Jaccard"

    vec_sums = [sum(v) for v in vectors]
    print(f"\nEncoded {len(vectors)} instances × {len(patterns)} patterns ({weighting_label})")
    print(f"  Active patterns per instance: min={min(vec_sums):.2f} mean={np.mean(vec_sums):.2f} max={max(vec_sums):.2f}")

    print("Computing pairwise distances...")
    dists_new: dict[tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i + 1, N):
            dists_new[(i, j)] = dist_fn(vectors[i], vectors[j])

    vals = list(dists_new.values())
    print(f"  {len(dists_new)} pairs — range [{min(vals):.3f}, {max(vals):.3f}]  "
          f"mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  unique={len(set(vals))}")

    ## Patch distances.parquet
    if args.distances.exists():
        df = pd.read_parquet(args.distances)
        print(f"\nPatching {args.distances} ({len(df)} rows)...")
        df["d_motifs"] = df.apply(
            lambda row: dists_new.get(
                (int(row["i"]), int(row["j"])),
                dists_new.get((int(row["j"]), int(row["i"])), np.nan),
            ),
            axis=1,
        )
    else:
        print("\nBuilding distances.parquet from scratch...")
        df = pd.DataFrame(
            [{"i": i, "j": j, "d_motifs": d} for (i, j), d in dists_new.items()]
        )

    df.to_parquet(args.distances, index=False)
    print(f"Saved → {args.distances}")
    print(f"  d_motifs unique values: {df['d_motifs'].nunique()}")

    ## Save pattern vocabulary for inspection
    source_tag = args.source if args.source == "ast_edits" else args.model
    vocab_path = ROOT / "output" / f"motif_vocab_{source_tag}.json"
    with open(vocab_path, "w") as f:
        json.dump(
            [{"pattern": pat, "support": sup, "pct": round(100 * sup / N, 1)}
             for pat, sup in patterns],
            f, indent=2,
        )
    print(f"Vocabulary → {vocab_path}")


if __name__ == "__main__":
    main()
