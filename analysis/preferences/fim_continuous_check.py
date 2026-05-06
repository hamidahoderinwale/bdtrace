"""Continuous-correlation sanity check for FIM forms.

The 15 named FIM forms are an interpretive grouping over per-instance
AST-edit token vectors. This script computes the underlying continuous
signal — Spearman correlation between AST-edit complexity (sum of
ADD/DEL token counts per oracle patch) and 84-agent ease — to confirm
the form-level finding is not an artifact of the form-level binning.

One number, one bootstrap CI. That's the whole sanity check.

Reads:
    HF dataset princeton-nlp/SWE-bench_Lite (oracle patches)
    output/leaderboard/lite_results.json    (84-agent pass/fail)
Writes:
    output/paper2_pilot/fim_continuous_check.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

LEADERBOARD = ROOT / "output" / "leaderboard" / "lite_results.json"
OUT_JSON    = ROOT / "output" / "paper2_pilot" / "fim_continuous_check.json"
N_BOOTSTRAP = 2000


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("Loading SWE-bench Lite oracle patches...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    patches = {row["instance_id"]: row["patch"] for row in ds}
    print(f"  {len(patches)} oracle patches")

    print("Computing 84-agent ease per instance...")
    leaderboard = json.loads(LEADERBOARD.read_text())
    instances = set()
    for ag in leaderboard.values():
        instances |= set(ag.keys())
    ease = {}
    n_agents = len(leaderboard)
    for iid in instances:
        n_pass = sum(1 for ag in leaderboard.values() if ag.get(iid) is True)
        ease[iid] = n_pass / n_agents
    print(f"  {len(ease)} instances; {n_agents} agents in leaderboard")

    print("Tokenising oracle patches into AST-edit token vectors...")
    complexity, ease_vec = [], []
    skipped = 0
    for iid, patch in patches.items():
        if iid not in ease:
            skipped += 1
            continue
        try:
            tokens = patch_to_ast_sequence(patch)
        except Exception:
            skipped += 1
            continue
        if not tokens:
            skipped += 1
            continue
        complexity.append(len(tokens))
        ease_vec.append(ease[iid])
    n = len(complexity)
    print(f"  {n} usable instances ({skipped} skipped — missing leaderboard or empty tokens)")

    complexity = np.asarray(complexity)
    ease_vec   = np.asarray(ease_vec)
    rho, p     = spearmanr(complexity, ease_vec)
    print(f"\nSpearman rho(complexity, ease) = {rho:.4f}  (p = {p:.2e}, n = {n})")

    boot_rhos = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(complexity[idx], ease_vec[idx])
        boot_rhos.append(r)
    boot_rhos = np.asarray(boot_rhos)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"95% bootstrap CI (B={N_BOOTSTRAP}): [{ci_lo:.4f}, {ci_hi:.4f}]")

    distinct = []
    distinct_ease = []
    for iid, patch in patches.items():
        if iid not in ease:
            continue
        try:
            tokens = patch_to_ast_sequence(patch)
        except Exception:
            continue
        if not tokens:
            continue
        distinct.append(len(set(tokens)))
        distinct_ease.append(ease[iid])
    distinct = np.asarray(distinct)
    distinct_ease = np.asarray(distinct_ease)
    rho_d, p_d = spearmanr(distinct, distinct_ease)
    print(f"\nAlt: rho(distinct AST-edit token TYPES, ease) = {rho_d:.4f}  (p = {p_d:.2e})")

    OUT_JSON.write_text(json.dumps({
        "n_instances":           int(n),
        "n_agents_in_leaderboard": int(n_agents),
        "complexity_metric":     "len(patch_to_ast_sequence(patch))",
        "spearman_rho":          float(rho),
        "p_value":               float(p),
        "bootstrap_n":           N_BOOTSTRAP,
        "bootstrap_ci_95":       [float(ci_lo), float(ci_hi)],
        "alt_distinct_types_rho": float(rho_d),
        "alt_distinct_types_p":   float(p_d),
    }, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
