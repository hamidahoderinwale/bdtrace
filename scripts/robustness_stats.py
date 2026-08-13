"""Robustness statistics for all paper findings.

Computes the statistical support needed to report each finding confidently:
  - Wilson CIs on composition gap proportions
  - Kruskal-Wallis test on motif surprise by failure type (null result confirmation)
  - Bootstrap CIs on heritability statistic
  - Spearman correlation (perplexity vs. ease) with CI
  - Random baseline F1 for representation comparison

Outputs:
  output/paper2_pilot/robustness_stats.json   full numeric results
  output/paper2_pilot/robustness_stats.md     human-readable summary table

Usage:
    python -m scripts.robustness_stats
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, spearmanr, norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "output" / "paper2_pilot"


# ---------------------------------------------------------------------------
# Wilson confidence interval for proportions
# ---------------------------------------------------------------------------

def wilson_ci(count: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - alpha / 2)
    p_hat = count / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - margin), float(center + margin)


# ---------------------------------------------------------------------------
# 1. Composition gap CIs
# ---------------------------------------------------------------------------

def composition_gap_cis() -> dict:
    ic = json.load(open(
        PROJECT_ROOT / "output" / "compositional_generalization" / "instance_classification.json"
    ))
    ss = json.load(open(
        PROJECT_ROOT / "output" / "compositional_generalization" / "summary_stats.json"
    ))
    lb = json.load(open(PROJECT_ROOT / "output" / "leaderboard" / "lite_results.json"))

    AGENT_LONG = {
        "20240402_sweagent_gpt4":           "GPT-4",
        "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
        "20240728_sweagent_gpt4o":          "GPT-4o",
    }

    counts: Counter = Counter()
    for inst_id, agent_map in ic.items():
        for agent_long, cls in agent_map.items():
            if agent_long not in AGENT_LONG:
                continue
            passed = lb.get(agent_long, {}).get(inst_id, False)
            if not passed:
                counts[cls] += 1

    total = sum(counts.values())
    results = {}
    for cls in ["novel_primitive", "novel_composition", "familiar"]:
        n = counts[cls]
        lo, hi = wilson_ci(n, total)
        results[cls] = {
            "count": n,
            "total": total,
            "proportion": round(n / total, 4) if total > 0 else 0.0,
            "wilson_ci_95": [round(lo, 4), round(hi, 4)],
        }

    print("\n=== COMPOSITION GAP WILSON CIs ===")
    for cls, v in results.items():
        print(f"  {cls:<22}: {v['proportion']:.3f} "
              f"[{v['wilson_ci_95'][0]:.3f}, {v['wilson_ci_95'][1]:.3f}]  "
              f"(n={v['count']}/{v['total']})")
    return results


# ---------------------------------------------------------------------------
# 2. Kruskal-Wallis on motif surprise by failure type
# ---------------------------------------------------------------------------

def motif_surprise_kruskal() -> dict:
    seqs = [json.loads(l) for l in open(
        PROJECT_ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
    )]
    ic = json.load(open(
        PROJECT_ROOT / "output" / "compositional_generalization" / "instance_classification.json"
    ))
    lb = json.load(open(PROJECT_ROOT / "output" / "leaderboard" / "lite_results.json"))

    AGENT_LONG = {
        "GPT-4":     "20240402_sweagent_gpt4",
        "Claude-3.5":"20240620_sweagent_claude3.5sonnet",
        "GPT-4o":    "20240728_sweagent_gpt4o",
    }

    counts_all: Counter = Counter()
    for s in seqs:
        counts_all.update(s["bpe"])
    total = sum(counts_all.values())

    def mean_surprise(bpe_tokens):
        surprises = []
        for tok in bpe_tokens:
            p = counts_all.get(tok, 0) / total
            surprises.append(-math.log2(p) if p > 0 else -math.log2(1 / (total + 1)))
        return float(np.mean(surprises)) if surprises else 0.0

    by_class: dict[str, list[float]] = {
        "familiar": [], "novel_primitive": [], "novel_composition": []
    }
    for s in seqs:
        agent_long = AGENT_LONG.get(s["agent"])
        if not agent_long:
            continue
        passed = lb.get(agent_long, {}).get(s["instance_id"], False)
        if passed:
            continue
        cls = ic.get(s["instance_id"], {}).get(agent_long)
        if cls in by_class:
            by_class[cls].append(mean_surprise(s["bpe"]))

    groups = [v for v in by_class.values() if v]
    if len(groups) < 2:
        return {"error": "insufficient data"}

    H, p = kruskal(*groups)

    # Effect size: eta-squared approximation from H
    n_total = sum(len(g) for g in groups)
    eta_sq = (H - len(groups) + 1) / (n_total - len(groups))

    # Median and IQR per class
    per_class_stats = {}
    for cls, vals in by_class.items():
        arr = np.array(vals)
        per_class_stats[cls] = {
            "n": len(vals),
            "median": float(np.median(arr)),
            "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "mean": float(np.mean(arr)),
        }

    result = {
        "test": "Kruskal-Wallis H",
        "H": float(H),
        "p_value": float(p),
        "eta_squared": float(eta_sq),
        "n_total": n_total,
        "interpretation": "null confirmed (p > 0.05)" if p > 0.05 else "significant difference detected",
        "per_class": per_class_stats,
    }

    print("\n=== KRUSKAL-WALLIS: MOTIF SURPRISE BY FAILURE TYPE ===")
    print(f"  H = {H:.4f}, p = {p:.4f}, eta^2 = {eta_sq:.4f}")
    print(f"  Interpretation: {result['interpretation']}")
    for cls, v in per_class_stats.items():
        print(f"  {cls:<22}: n={v['n']}, median={v['median']:.4f}")
    return result


# ---------------------------------------------------------------------------
# 3. Bootstrap CIs on heritability statistic
# ---------------------------------------------------------------------------

def heritability_bootstrap_cis() -> dict:
    seqs = [json.loads(l) for l in open(
        PROJECT_ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
    )]
    perm = json.load(open(PROJECT_ROOT / "output" / "paper2_pilot" / "permutation_null.json"))

    observed = perm.get("aggregate", {}).get("observed_gap", None)
    if observed is None:
        keys = list(perm.get("aggregate", {}).keys())
        print(f"  permutation_null.json aggregate keys: {keys[:8]}")
        return {"error": "could not find observed_gap statistic"}

    # Bootstrap the observed statistic: resample trajectories with replacement
    # and recompute same-family similarity advantage
    AGENT_LONG = {
        "GPT-4":     "20240402_sweagent_gpt4",
        "Claude-3.5":"20240620_sweagent_claude3.5sonnet",
        "GPT-4o":    "20240728_sweagent_gpt4o",
    }
    GPT_FAMILY = {"GPT-4", "GPT-4o"}

    # Build pairwise compression distances as proxy for similarity
    by_instance: dict[str, dict[str, float]] = {}
    for s in seqs:
        by_instance.setdefault(s["instance_id"], {})[s["agent"]] = s["compression"]

    # Same-family advantage: mean(compression diff within GPT family) vs. across
    def same_family_advantage(instances):
        within, across = [], []
        for inst in instances:
            agents = by_instance.get(inst, {})
            gpt_comps   = [v for k, v in agents.items() if k in GPT_FAMILY]
            other_comps = [v for k, v in agents.items() if k not in GPT_FAMILY]
            if len(gpt_comps) == 2:
                within.append(abs(gpt_comps[0] - gpt_comps[1]))
            for gc in gpt_comps:
                for oc in other_comps:
                    across.append(abs(gc - oc))
        if not within or not across:
            return float("nan")
        return float(np.mean(across) - np.mean(within))

    instance_ids = list(by_instance.keys())
    rng = np.random.default_rng(42)
    n_bootstrap = 2000
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(instance_ids, size=len(instance_ids), replace=True).tolist()
        stat = same_family_advantage(sample)
        if not math.isnan(stat):
            bootstrap_stats.append(stat)

    bs = np.array(bootstrap_stats)
    ci_lo, ci_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    se = float(np.std(bs))

    result = {
        "observed_statistic": float(observed),
        "bootstrap_n": n_bootstrap,
        "bootstrap_mean": float(bs.mean()),
        "bootstrap_se": se,
        "ci_95": [ci_lo, ci_hi],
        "permutation_p": perm.get("aggregate", {}).get("p_value", None),
    }

    print("\n=== HERITABILITY BOOTSTRAP CIs ===")
    print(f"  Observed: {observed:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Bootstrap SE: {se:.4f}")
    return result


# ---------------------------------------------------------------------------
# 4. Spearman correlation: perplexity vs. ease
# ---------------------------------------------------------------------------

def perplexity_ease_correlation() -> dict:
    seqs = [json.loads(l) for l in open(
        PROJECT_ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
    )]
    lb  = json.load(open(PROJECT_ROOT / "output" / "leaderboard" / "lite_results.json"))
    gap = json.load(open(
        PROJECT_ROOT / "output" / "compositional_generalization" / "composition_gap.json"
    ))
    ease_map = {r["instance_id"]: r["ease"] for r in gap}

    AGENT_LONG = {
        "GPT-4":     "20240402_sweagent_gpt4",
        "Claude-3.5":"20240620_sweagent_claude3.5sonnet",
        "GPT-4o":    "20240728_sweagent_gpt4o",
    }

    # Rebuild perplexity (order-3 n-gram, same as regularity script)
    counts_all: Counter = Counter()
    bigram: Counter  = Counter()
    trigram: Counter = Counter()
    for s in seqs:
        toks = s["bpe"]
        counts_all.update(toks)
        for i in range(len(toks) - 1):
            bigram[(toks[i], toks[i+1])] += 1
        for i in range(len(toks) - 2):
            trigram[(toks[i], toks[i+1], toks[i+2])] += 1

    V = len(counts_all)
    k = 0.5

    def ppl(toks):
        if len(toks) < 3:
            return None
        log_prob = 0.0
        n = 0
        for i in range(2, len(toks)):
            t0, t1, t2 = toks[i-2], toks[i-1], toks[i]
            p = (trigram.get((t0,t1,t2), 0) + k) / (bigram.get((t0,t1), 0) + k*V)
            log_prob += math.log2(p)
            n += 1
        return 2 ** (-log_prob / n) if n > 0 else None

    paired = []
    for s in seqs:
        agent_long = AGENT_LONG.get(s["agent"])
        if not agent_long:
            continue
        ease = ease_map.get(s["instance_id"])
        p_ = ppl(s["bpe"])
        if ease is not None and p_ is not None:
            passed = lb.get(agent_long, {}).get(s["instance_id"], False)
            paired.append({"ease": ease, "perplexity": p_, "passed": passed})

    ease_vals = [r["ease"] for r in paired]
    ppl_vals  = [r["perplexity"] for r in paired]

    rho, p_val = spearmanr(ease_vals, ppl_vals)

    # CI via Fisher z-transform approximation for Spearman
    n = len(paired)
    se_z = 1 / math.sqrt(n - 3)
    z = math.atanh(rho)
    ci_lo = math.tanh(z - 1.96 * se_z)
    ci_hi = math.tanh(z + 1.96 * se_z)

    result = {
        "n": n,
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "ci_95_approx": [float(ci_lo), float(ci_hi)],
        "interpretation": (
            "no meaningful correlation" if abs(rho) < 0.1 else
            "weak correlation" if abs(rho) < 0.3 else
            "moderate correlation"
        ),
    }

    print("\n=== SPEARMAN: PERPLEXITY vs EASE ===")
    print(f"  rho = {rho:.4f}, p = {p_val:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Interpretation: {result['interpretation']}")
    return result


# ---------------------------------------------------------------------------
# 5. Random baseline F1 for representation comparison
# ---------------------------------------------------------------------------

def representation_random_baseline() -> dict:
    import pandas as pd
    df = pd.read_csv(
        PROJECT_ROOT / "output" / "representation_comparison" / "knn_f1_results.csv"
    )
    print("\n=== REPRESENTATION COMPARISON: RANDOM BASELINE ===")
    print("  knn_f1_results columns:", list(df.columns))

    # Class balance from BPE sequences + leaderboard
    seqs = [json.loads(l) for l in open(
        PROJECT_ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
    )]
    lb = json.load(open(PROJECT_ROOT / "output" / "leaderboard" / "lite_results.json"))
    AGENT_LONG = {
        "GPT-4":     "20240402_sweagent_gpt4",
        "Claude-3.5":"20240620_sweagent_claude3.5sonnet",
        "GPT-4o":    "20240728_sweagent_gpt4o",
    }
    n_pass = sum(
        1 for s in seqs
        if lb.get(AGENT_LONG.get(s["agent"], ""), {}).get(s["instance_id"], False)
    )
    n_total = len(seqs)
    p_pos = n_pass / n_total

    # The knn_f1_results.csv reports positive-class F1 (binary, pos_label=1).
    # The baseline must use the same metric — not macro-F1, which always ≈ 0.5
    # regardless of class balance and is therefore a misleadingly high comparison.
    # Positive-class F1 for a stratified random classifier ≈ pass_rate (p_pos).
    rng = np.random.default_rng(42)
    y_true = np.array([1]*n_pass + [0]*(n_total - n_pass))

    from sklearn.metrics import f1_score as sklearn_f1
    random_f1s = []
    majority_f1s = []
    for _ in range(1000):
        y_rand = rng.choice([0, 1], size=n_total, p=[1-p_pos, p_pos])
        random_f1s.append(sklearn_f1(y_true, y_rand, average="binary", pos_label=1, zero_division=0))
        y_maj = np.zeros(n_total, dtype=int)
        majority_f1s.append(sklearn_f1(y_true, y_maj, average="binary", pos_label=1, zero_division=0))

    result = {
        "n_total": n_total,
        "n_pass": n_pass,
        "pass_rate": float(p_pos),
        "random_baseline_positive_f1_mean": float(np.mean(random_f1s)),
        "random_baseline_positive_f1_std": float(np.std(random_f1s)),
        "majority_class_positive_f1": float(np.mean(majority_f1s)),
        "note": "All F1 values are positive-class (binary) F1, matching knn_f1_results.csv metric.",
    }

    print(f"  Pass rate: {p_pos:.3f} ({n_pass}/{n_total})")
    print(f"  Random baseline positive-class F1: {result['random_baseline_positive_f1_mean']:.4f} "
          f"+/- {result['random_baseline_positive_f1_std']:.4f}")
    print(f"  Majority class positive-class F1:  {result['majority_class_positive_f1']:.4f}")
    print(f"  (Compare: FIM best F1 at k=5 ~0.265, edit cert ~0.205 — same metric now)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    results = {}

    results["composition_gap_cis"]       = composition_gap_cis()
    results["motif_surprise_kruskal"]    = motif_surprise_kruskal()
    results["heritability_bootstrap"]    = heritability_bootstrap_cis()
    results["perplexity_ease_spearman"]  = perplexity_ease_correlation()
    results["representation_baseline"]  = representation_random_baseline()

    (OUT / "robustness_stats.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    # Human-readable markdown summary
    lines = [
        "# Robustness statistics summary\n",
        "## Composition gap (Wilson 95% CIs)",
    ]
    for cls, v in results["composition_gap_cis"].items():
        lo, hi = v["wilson_ci_95"]
        lines.append(
            f"- **{cls}**: {v['proportion']:.3f} [{lo:.3f}, {hi:.3f}]  (n={v['count']})"
        )

    kw = results["motif_surprise_kruskal"]
    lines += [
        "\n## Motif surprise by failure type (Kruskal-Wallis)",
        f"- H = {kw['H']:.4f}, p = {kw['p_value']:.4f}, eta^2 = {kw['eta_squared']:.4f}",
        f"- **{kw['interpretation']}**",
    ]

    hb = results["heritability_bootstrap"]
    lines += [
        "\n## Heritability bootstrap CIs",
        f"- Observed statistic: {hb.get('observed_statistic', 'N/A')}",
        f"- 95% CI: {hb.get('ci_95', 'N/A')}",
        f"- Permutation p: {hb.get('permutation_p', 'N/A')}",
    ]

    sp = results["perplexity_ease_spearman"]
    lines += [
        "\n## Perplexity vs ease (Spearman)",
        f"- rho = {sp['spearman_rho']:.4f}, p = {sp['p_value']:.4f}",
        f"- 95% CI: {sp['ci_95_approx']}",
        f"- {sp['interpretation']}",
    ]

    rb = results["representation_baseline"]
    lines += [
        "\n## Representation comparison random baselines (positive-class F1)",
        f"- Pass rate: {rb['pass_rate']:.3f}",
        f"- Random positive-class F1: {rb['random_baseline_positive_f1_mean']:.4f}",
        f"- Majority class positive-class F1: {rb['majority_class_positive_f1']:.4f}",
        f"- Note: {rb['note']}",
    ]

    md_path = OUT / "robustness_stats.md"
    md_path.write_text("\n".join(lines))
    print(f"\nSaved: robustness_stats.json and robustness_stats.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
