"""Three FIM investigations on the same scale as the headline MI numbers.

Computes:
  (1) MI(FIM-form; pass/fail)          — direct predictor on the same scale as 60% MI
  (2) ρ(LOC, ease) vs ρ(FIM, ease)    — does structure beat surface size?
  (3) MI(FIM-form; pass/fail | n_resolved) — does FIM add signal beyond agent-derived difficulty?

Reads:
    HF princeton-nlp/SWE-bench_Lite             (oracle patches)
    output/canonical_forms/instance_assignments.parquet  (FIM form per instance)
    output/trajectories/lite_all_models.parquet (per-trajectory pass/fail)
    output/leaderboard/lite_results.json        (84-agent ease)
Writes:
    output/paper2_pilot/fim_investigations.json
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

LITE_FILE   = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
FORMS_FILE  = ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"
LEADERBOARD = ROOT / "output" / "leaderboard" / "lite_results.json"
OUT_JSON    = ROOT / "output" / "paper2_pilot" / "fim_investigations.json"


def entropy_bits(values: list) -> float:
    n = len(values)
    if n == 0: return 0.0
    counts = Counter(values)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def mi_bits(xs: list, ys: list) -> tuple[float, float, float]:
    """Return (MI in bits, H(Y), MI/H(Y))."""
    assert len(xs) == len(ys)
    n = len(xs)
    h_y = entropy_bits(ys)
    df = pd.DataFrame({"x": xs, "y": ys})
    h_y_given_x = 0.0
    for _, sub in df.groupby("x", observed=True):
        h_y_given_x += (len(sub) / n) * entropy_bits(sub["y"].tolist())
    mi = max(0.0, h_y - h_y_given_x)
    pct = (mi / h_y * 100) if h_y > 0 else 0.0
    return mi, h_y, pct


def conditional_mi_bits(xs: list, ys: list, zs: list) -> tuple[float, float]:
    """MI(X; Y | Z) = sum_z P(Z=z) * MI_z(X; Y).
    Returns (cond MI bits, weighted avg H(Y|z) so we can express as a percentage)."""
    n = len(xs)
    df = pd.DataFrame({"x": xs, "y": ys, "z": zs})
    cond_mi = 0.0
    weighted_hy = 0.0
    for z_val, sub in df.groupby("z", observed=True):
        n_z = len(sub)
        if n_z < 2:
            continue
        h_y_z = entropy_bits(sub["y"].tolist())
        h_y_given_xz = 0.0
        for _, ssub in sub.groupby("x", observed=True):
            h_y_given_xz += (len(ssub) / n_z) * entropy_bits(ssub["y"].tolist())
        mi_z = max(0.0, h_y_z - h_y_given_xz)
        cond_mi   += (n_z / n) * mi_z
        weighted_hy += (n_z / n) * h_y_z
    pct = (cond_mi / weighted_hy * 100) if weighted_hy > 0 else 0.0
    return cond_mi, pct


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Loading oracle patches from HF…")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    patches = {row["instance_id"]: row["patch"] for row in ds}

    print("Computing AST-edit complexity and LOC delta per instance…")
    ast_complexity, loc_delta = {}, {}
    for iid, patch in patches.items():
        try:
            tokens = patch_to_ast_sequence(patch)
        except Exception:
            tokens = []
        ast_complexity[iid] = len(tokens)
        n_added   = sum(1 for L in patch.splitlines() if L.startswith("+") and not L.startswith("+++"))
        n_removed = sum(1 for L in patch.splitlines() if L.startswith("-") and not L.startswith("---"))
        loc_delta[iid] = n_added + n_removed
    print(f"  {len(ast_complexity)} patches processed")

    print("Loading 84-agent ease…")
    leaderboard = json.loads(LEADERBOARD.read_text())
    ease = {}
    n_agents = len(leaderboard)
    for iid in patches:
        n_pass = sum(1 for ag in leaderboard.values() if ag.get(iid) is True)
        ease[iid] = n_pass / n_agents

    print("Loading per-trajectory pass/fail…")
    lite = pd.read_parquet(LITE_FILE)[["instance_id", "passed"]].copy()
    n_resolved = lite.groupby("instance_id")["passed"].sum().astype(int).to_dict()

    print("Loading FIM forms…")
    forms = pd.read_parquet(FORMS_FILE)[["instance_id", "form_name"]].drop_duplicates("instance_id")
    form_of = dict(zip(forms["instance_id"], forms["form_name"]))

    df = lite.merge(forms, on="instance_id", how="inner")
    df["n_resolved"] = df["instance_id"].map(n_resolved)

    print(f"\nMerged corpus: {len(df)} trajectories across {df['form_name'].nunique()} FIM forms")

    print("\n--- Investigation 1: MI(FIM-form; pass/fail) ---")
    mi1, h_y, pct1 = mi_bits(df["form_name"].tolist(), df["passed"].astype(str).tolist())
    print(f"  MI = {mi1:.4f} bits   H(pass/fail) = {h_y:.4f} bits   = {pct1:.1f}% of H")

    print("\n--- Reference points (same scale) ---")
    ref_n_resolved, _, ref_pct_n = mi_bits(df["n_resolved"].astype(str).tolist(),
                                           df["passed"].astype(str).tolist())
    print(f"  MI(n_resolved; pass/fail)   = {ref_n_resolved:.4f} bits = {ref_pct_n:.1f}% of H")

    print("\n--- Investigation 2: structural beats surface? ---")
    instances = sorted(set(ast_complexity) & set(ease) & set(loc_delta))
    ast_arr  = np.array([ast_complexity[i] for i in instances])
    loc_arr  = np.array([loc_delta[i] for i in instances])
    ease_arr = np.array([ease[i] for i in instances])
    keep = (ast_arr > 0) & (loc_arr > 0)
    ast_arr, loc_arr, ease_arr = ast_arr[keep], loc_arr[keep], ease_arr[keep]
    rho_ast, p_ast = spearmanr(ast_arr, ease_arr)
    rho_loc, p_loc = spearmanr(loc_arr, ease_arr)
    rho_ast_loc, _ = spearmanr(ast_arr, loc_arr)
    print(f"  ρ(AST complexity, ease)     = {rho_ast:.4f}   (p = {p_ast:.2e}, n = {len(ast_arr)})")
    print(f"  ρ(LOC delta,    ease)        = {rho_loc:.4f}   (p = {p_loc:.2e})")
    print(f"  ρ(AST complexity, LOC delta) = {rho_ast_loc:.4f}   (collinearity check)")

    print("\n--- Investigation 3: MI(FIM-form; pass/fail | n_resolved) ---")
    cond_mi, cond_pct = conditional_mi_bits(
        df["form_name"].tolist(),
        df["passed"].astype(str).tolist(),
        df["n_resolved"].astype(str).tolist(),
    )
    print(f"  MI(FIM | n_resolved) = {cond_mi:.4f} bits = {cond_pct:.1f}% of H(Y|n_resolved)")

    out = {
        "n_trajectories": int(len(df)),
        "n_forms": int(df["form_name"].nunique()),
        "investigation_1_mi_form_passfail": {
            "mi_bits": round(mi1, 4),
            "pct_of_H": round(pct1, 1),
            "h_y_bits": round(h_y, 4),
        },
        "reference_mi_n_resolved": {
            "mi_bits": round(ref_n_resolved, 4),
            "pct_of_H": round(ref_pct_n, 1),
        },
        "investigation_2_structure_vs_surface": {
            "n_instances": int(len(ast_arr)),
            "rho_ast_complexity_ease": round(float(rho_ast), 4),
            "p_ast": float(p_ast),
            "rho_loc_delta_ease": round(float(rho_loc), 4),
            "p_loc": float(p_loc),
            "rho_ast_vs_loc": round(float(rho_ast_loc), 4),
        },
        "investigation_3_conditional_mi": {
            "mi_bits": round(cond_mi, 4),
            "pct_of_H_y_given_z": round(cond_pct, 1),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
