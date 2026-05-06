"""Problem-side difficulty features (pre-patch, pre-agent).

Computes MI on the same scale as the n_resolved (60.2%) / FIM-form (19.4%) /
fix_type (1.3%) headline numbers, but using only features extracted from the
issue text — features that exist before any patch is written and before any
agent runs.

Features computed (all from problem_statement / issue text):
  has_reproducer      — does the issue contain a fenced code block with import/assert/raise?
  has_failing_test    — does the issue mention test_* or def test_?
  issue_length_words  — word count
  n_code_blocks       — count of fenced code blocks

Reads:
    HF princeton-nlp/SWE-bench_Lite             (problem_statement)
    output/trajectories/lite_all_models.parquet (per-trajectory pass/fail)
Writes:
    output/paper2_pilot/problem_side_features.json
"""
from __future__ import annotations
import json, sys, math, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
LITE_FILE = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
OUT_JSON  = ROOT / "output" / "paper2_pilot" / "problem_side_features.json"


def entropy_bits(values: list) -> float:
    n = len(values)
    if n == 0: return 0.0
    counts = Counter(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def mi_bits(xs: list, ys: list) -> tuple[float, float, float]:
    n = len(xs)
    h_y = entropy_bits(ys)
    df = pd.DataFrame({"x": xs, "y": ys})
    h_y_given_x = 0.0
    for _, sub in df.groupby("x", observed=True):
        h_y_given_x += (len(sub) / n) * entropy_bits(sub["y"].tolist())
    mi = max(0.0, h_y - h_y_given_x)
    pct = (mi / h_y * 100) if h_y > 0 else 0.0
    return mi, h_y, pct


CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)
TEST_NAME_RE  = re.compile(r"\bdef\s+test_\w+|\btest_\w+\s*\(", re.IGNORECASE)
RUNNABLE_RE   = re.compile(r"\b(import|from|assert|raise|print|>>>)\b")


def extract_features(problem_statement: str) -> dict:
    if not problem_statement:
        return {"has_reproducer": False, "has_failing_test": False,
                "issue_length_words": 0, "n_code_blocks": 0}
    blocks = CODE_BLOCK_RE.findall(problem_statement)
    has_runnable_block = any(RUNNABLE_RE.search(b) for b in blocks)
    return {
        "has_reproducer":     has_runnable_block,
        "has_failing_test":   bool(TEST_NAME_RE.search(problem_statement)),
        "issue_length_words": len(problem_statement.split()),
        "n_code_blocks":      len(blocks),
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Loading SWE-bench Lite issues…")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    feats = {row["instance_id"]: extract_features(row["problem_statement"]) for row in ds}
    print(f"  {len(feats)} issues processed")

    print("Loading per-trajectory pass/fail…")
    lite = pd.read_parquet(LITE_FILE)[["instance_id", "passed"]].copy()

    df = lite.merge(
        pd.DataFrame.from_dict(feats, orient="index").reset_index().rename(columns={"index": "instance_id"}),
        on="instance_id", how="inner",
    )
    print(f"Merged corpus: {len(df)} trajectories")
    print()

    print("Feature distribution (per-instance, deduplicated):")
    inst = df.drop_duplicates("instance_id")
    print(f"  has_reproducer:    {inst['has_reproducer'].sum():3d} / {len(inst)}  ({inst['has_reproducer'].mean()*100:.1f}%)")
    print(f"  has_failing_test:  {inst['has_failing_test'].sum():3d} / {len(inst)}  ({inst['has_failing_test'].mean()*100:.1f}%)")
    print(f"  issue_length_words: median={inst['issue_length_words'].median():.0f}, IQR=[{inst['issue_length_words'].quantile(0.25):.0f}, {inst['issue_length_words'].quantile(0.75):.0f}]")
    print(f"  n_code_blocks:     median={inst['n_code_blocks'].median():.0f}, max={inst['n_code_blocks'].max()}")
    print()

    df["length_quartile"] = pd.qcut(df["issue_length_words"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    df["blocks_bucket"]   = pd.cut(
        df["n_code_blocks"], bins=[-0.5, 0.5, 1.5, 3.5, 100],
        labels=["0", "1", "2-3", "4+"],
    )

    rows = []

    print("--- Investigation: MI(problem-side feature; pass/fail) ---")
    for feat_col, label in [
        ("has_reproducer",     "has_reproducer (binary)"),
        ("has_failing_test",   "has_failing_test (binary)"),
        ("length_quartile",    "issue_length_quartile (Q1-Q4)"),
        ("blocks_bucket",      "n_code_blocks (0/1/2-3/4+)"),
    ]:
        sub = df.dropna(subset=[feat_col])
        mi, h_y, pct = mi_bits(sub[feat_col].astype(str).tolist(), sub["passed"].astype(str).tolist())
        rows.append({"feature": label, "mi_bits": round(mi, 4), "pct_of_H": round(pct, 1), "n": int(len(sub))})
        print(f"  {label:42s}  MI = {mi:.4f} bits = {pct:.1f}% of H  (n={len(sub)})")

    print("\n--- Spearman ρ (continuous) ---")
    inst_passed = df.groupby("instance_id")["passed"].mean()
    inst_feats = inst.set_index("instance_id")
    common = inst_feats.index.intersection(inst_passed.index)
    rho_len, p_len = spearmanr(inst_feats.loc[common, "issue_length_words"], inst_passed.loc[common])
    rho_blocks, p_blocks = spearmanr(inst_feats.loc[common, "n_code_blocks"], inst_passed.loc[common])
    print(f"  ρ(issue_length_words, mean_pass_rate) = {rho_len:.4f}  (p = {p_len:.2e}, n={len(common)})")
    print(f"  ρ(n_code_blocks,      mean_pass_rate) = {rho_blocks:.4f}  (p = {p_blocks:.2e})")

    print("\n--- Reference points (same scale) ---")
    print(f"  MI(n_resolved; pass/fail)   = 60.2%  (from fim_investigations.json)")
    print(f"  MI(FIM-form;   pass/fail)   = 19.4%  (from fim_investigations.json)")
    print(f"  MI(fix_type;   pass/fail)   =  1.3%  (existing reference)")

    OUT_JSON.write_text(json.dumps({
        "n_trajectories": int(len(df)),
        "n_instances": int(len(inst)),
        "feature_distribution": {
            "has_reproducer_count": int(inst["has_reproducer"].sum()),
            "has_failing_test_count": int(inst["has_failing_test"].sum()),
            "issue_length_words_median": float(inst["issue_length_words"].median()),
            "n_code_blocks_median": float(inst["n_code_blocks"].median()),
        },
        "mi_per_feature": rows,
        "spearman_continuous": {
            "issue_length_words":  {"rho": round(float(rho_len), 4),    "p": float(p_len)},
            "n_code_blocks":       {"rho": round(float(rho_blocks), 4), "p": float(p_blocks)},
        },
        "scale_reference": {
            "mi_n_resolved_pct": 60.2,
            "mi_FIM_form_pct":  19.4,
            "mi_fix_type_pct":   1.3,
        },
    }, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
