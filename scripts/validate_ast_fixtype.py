#!/usr/bin/env python3
"""
Validate AST patterns against LLM fix-type labels.

For each fix type, finds the AST patterns that are most discriminative —
patterns that appear significantly more often within that fix type than
across the rest of the corpus. Uses log-odds ratio as the discrimination
score (no threshold, fully data-driven).

If the structural patterns align with the semantic labels, we gain
confidence that both representations are measuring the same underlying
phenomenon. Mismatches reveal either labeling errors or granularity gaps.

Outputs:
  - Console: top discriminative patterns per fix type with interpretation
  - output/analysis/ast_fixtype_validation.json: full results for inspection

Usage:
    uv run python scripts/validate_ast_fixtype.py
    uv run python scripts/validate_ast_fixtype.py --min-fix-type-n 10 --top-k 8
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

ROOT = Path(__file__).resolve().parent.parent


def log_odds(k_in: int, n_in: int, k_out: int, n_out: int) -> float:
    """
    Log-odds ratio: how much more likely is this pattern in the fix type
    compared to outside it.

    Positive = enriched in this fix type.
    Uses +0.5 smoothing to avoid log(0).
    """
    p_in = (k_in + 0.5) / (n_in + 1)
    p_out = (k_out + 0.5) / (n_out + 1)
    return math.log(p_in / (1 - p_in)) - math.log(p_out / (1 - p_out))


def pattern_str(pat: list[str]) -> str:
    return " → ".join(pat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-types", type=Path,
                        default=ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "fix_types.json")
    parser.add_argument("--min-fix-type-n", type=int, default=10,
                        help="Skip fix types with fewer than this many instances")
    parser.add_argument("--min-pattern-support", type=int, default=3,
                        help="Skip patterns appearing fewer than this many times within a fix type")
    parser.add_argument("--top-k", type=int, default=6,
                        help="Top K discriminative patterns to show per fix type")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "output" / "analysis" / "ast_fixtype_validation.json")
    args = parser.parse_args()

    ## Load fix-type labels
    with open(args.fix_types) as f:
        ft_data = json.load(f)
    ft_map = {r["instance_id"]: r["fix_type"] for r in ft_data["results"]}
    print(f"Loaded {len(ft_map)} fix-type labels")

    ## Load patches from HF
    from datasets import load_dataset
    print("Loading patches from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    patch_map = {row["instance_id"]: row["patch"] for row in ds}

    ## Build per-instance AST sequences
    instances = []
    for iid, ft in ft_map.items():
        patch = patch_map.get(iid, "")
        seq = patch_to_ast_sequence(patch)
        instances.append({"instance_id": iid, "fix_type": ft, "sequence": seq})

    ## Group by fix type
    by_type: dict[str, list[list[str]]] = defaultdict(list)
    for inst in instances:
        by_type[inst["fix_type"]].append(inst["sequence"])

    N = len(instances)
    fix_types = [ft for ft, seqs in by_type.items() if len(seqs) >= args.min_fix_type_n]
    print(f"\nFix types with n >= {args.min_fix_type_n}: {fix_types}")

    ## For each fix type, compute pattern frequencies inside vs outside
    ## Use the corpus vocab (all patterns of length >= 2 that appear in >= 2 instances)
    ## but compute from scratch per-group rather than loading the full 11k vocab

    ## First collect all unique patterns (length >= 2) across corpus
    from collections import Counter

    def all_ngrams(seq: list[str], max_n: int = 5) -> set[tuple[str, ...]]:
        """All subsequences of length 2..max_n (contiguous, for speed)."""
        out = set()
        for n in range(2, min(max_n + 1, len(seq) + 1)):
            for i in range(len(seq) - n + 1):
                out.add(tuple(seq[i:i + n]))
        return out

    # Count pattern support per fix type (contiguous n-grams, not arbitrary subsequences,
    # for tractability — the discriminative signal is similar and much faster)
    print("\nComputing per-fix-type pattern frequencies...")

    # Global pattern document frequency (how many instances contain it)
    global_df: Counter = Counter()
    instance_patterns: dict[str, set[tuple]] = {}
    for inst in instances:
        pats = all_ngrams(inst["sequence"])
        instance_patterns[inst["instance_id"]] = pats
        global_df.update(pats)

    # Only keep patterns appearing in >= 2 instances globally
    valid_patterns = {p for p, cnt in global_df.items() if cnt >= 2}
    print(f"Valid patterns (global support >= 2): {len(valid_patterns)}")

    results = {}
    for ft in fix_types:
        in_ids = {inst["instance_id"] for inst in instances if inst["fix_type"] == ft}
        out_ids = {inst["instance_id"] for inst in instances if inst["fix_type"] != ft}
        n_in, n_out = len(in_ids), len(out_ids)

        # Pattern frequencies: how many instances in/out contain each pattern
        pat_in: Counter = Counter()
        for iid in in_ids:
            for p in instance_patterns[iid]:
                if p in valid_patterns:
                    pat_in[p] += 1

        pat_out: Counter = Counter()
        for iid in out_ids:
            for p in instance_patterns[iid]:
                if p in valid_patterns:
                    pat_out[p] += 1

        # Compute log-odds for patterns meeting min support inside fix type
        scored = []
        for pat, k_in in pat_in.items():
            if k_in < args.min_pattern_support:
                continue
            k_out = pat_out.get(pat, 0)
            lo = log_odds(k_in, n_in, k_out, n_out)
            pct_in = 100 * k_in / n_in
            pct_out = 100 * k_out / n_out if n_out else 0
            scored.append({
                "pattern": list(pat),
                "pattern_str": pattern_str(list(pat)),
                "k_in": k_in,
                "n_in": n_in,
                "pct_in": round(pct_in, 1),
                "k_out": k_out,
                "pct_out": round(pct_out, 1),
                "log_odds": round(lo, 3),
            })

        scored.sort(key=lambda x: -x["log_odds"])
        results[ft] = {
            "n": n_in,
            "top_enriched": scored[:args.top_k],
            "top_depleted": sorted(scored, key=lambda x: x["log_odds"])[:args.top_k],
        }

    ## Print results
    for ft in fix_types:
        r = results[ft]
        print(f"\n{'='*60}")
        print(f"[{ft}]  n={r['n']}")
        print(f"  Top {args.top_k} enriched patterns (high log-odds = specific to this fix type):")
        for p in r["top_enriched"]:
            print(f"    lo={p['log_odds']:+.2f}  {p['pct_in']:4.0f}% in / {p['pct_out']:3.0f}% out  |  {p['pattern_str']}")
        print(f"  Top {args.top_k} depleted patterns (low log-odds = avoided by this fix type):")
        for p in r["top_depleted"]:
            print(f"    lo={p['log_odds']:+.2f}  {p['pct_in']:4.0f}% in / {p['pct_out']:3.0f}% out  |  {p['pattern_str']}")

    ## Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
