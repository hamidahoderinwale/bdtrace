"""Joint difficulty predictors with feature ranking.

Single-feature MI numbers report each feature in isolation, missing
redundancy and complementarity between features. This script:

  1. Combines all candidate features into one feature matrix
  2. Fits a gradient-boosted classifier with grouped 5-fold cross-validation
     (folds split on instance_id so the same instance never appears in train+test)
  3. Reports cross-validated AUC for the joint model
  4. Reports permutation feature importance to rank features
  5. Reports forward-selection order to show marginal contributions

Reads:
    HF princeton-nlp/SWE-bench_Lite             (problem_statement, oracle patch)
    output/canonical_forms/instance_assignments.parquet  (FIM form)
    output/trajectories/lite_all_models.parquet (per-trajectory pass/fail, agent)
    output/leaderboard/lite_results.json        (84-agent ease)
Writes:
    output/paper2_pilot/joint_difficulty.json
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

LITE_FILE   = ROOT / "output" / "trajectories" / "lite_all_models.parquet"
FORMS_FILE  = ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"
LEADERBOARD = ROOT / "output" / "leaderboard" / "lite_results.json"
OUT_JSON    = ROOT / "output" / "paper2_pilot" / "joint_difficulty.json"

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)
TEST_NAME_RE  = re.compile(r"\bdef\s+test_\w+|\btest_\w+\s*\(", re.IGNORECASE)
RUNNABLE_RE   = re.compile(r"\b(import|from|assert|raise|print|>>>)\b")


def issue_features(text: str) -> dict:
    if not text:
        return {"has_reproducer": 0, "has_failing_test": 0, "issue_length_words": 0, "n_code_blocks": 0}
    blocks = CODE_BLOCK_RE.findall(text)
    return {
        "has_reproducer":     int(any(RUNNABLE_RE.search(b) for b in blocks)),
        "has_failing_test":   int(bool(TEST_NAME_RE.search(text))),
        "issue_length_words": len(text.split()),
        "n_code_blocks":      len(blocks),
    }


def patch_features(patch: str) -> dict:
    if not patch:
        return {"ast_complexity": 0, "ast_distinct_types": 0, "loc_delta": 0, "n_hunks": 0, "n_files_touched": 0}
    try:
        tokens = patch_to_ast_sequence(patch)
    except Exception:
        tokens = []
    n_added   = sum(1 for L in patch.splitlines() if L.startswith("+") and not L.startswith("+++"))
    n_removed = sum(1 for L in patch.splitlines() if L.startswith("-") and not L.startswith("---"))
    n_hunks   = patch.count("@@ ") // 2
    n_files   = patch.count("\ndiff --git ") + (1 if patch.startswith("diff --git ") else 0)
    return {
        "ast_complexity":     len(tokens),
        "ast_distinct_types": len(set(tokens)),
        "loc_delta":          n_added + n_removed,
        "n_hunks":            n_hunks,
        "n_files_touched":    n_files,
    }


def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("Loading SWE-bench Lite issues + oracle patches…")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_iid = {}
    for r in ds:
        iid = r["instance_id"]
        by_iid[iid] = {**issue_features(r["problem_statement"]), **patch_features(r["patch"])}

    print("Loading FIM-form labels…")
    forms = pd.read_parquet(FORMS_FILE)[["instance_id", "form_name"]].drop_duplicates("instance_id")
    form_map = dict(zip(forms["instance_id"], forms["form_name"]))

    print("Loading 84-agent ease as a continuous covariate…")
    leaderboard = json.loads(LEADERBOARD.read_text())
    n_agents_lb = len(leaderboard)
    ease_84 = {iid: sum(1 for ag in leaderboard.values() if ag.get(iid) is True) / n_agents_lb
               for iid in by_iid}

    print("Loading per-trajectory pass/fail…")
    lite = pd.read_parquet(LITE_FILE)[["instance_id", "model_id", "passed"]].copy()

    rows = []
    for _, r in lite.iterrows():
        iid = r["instance_id"]
        if iid not in by_iid:
            continue
        feats = dict(by_iid[iid])
        feats["fim_form"]   = form_map.get(iid, "unknown")
        feats["ease_84"]    = ease_84.get(iid, np.nan)
        feats["instance_id"] = iid
        feats["agent"]      = r["model_id"]
        feats["passed"]     = int(r["passed"])
        rows.append(feats)
    df = pd.DataFrame(rows).dropna(subset=["ease_84"])
    print(f"Joined corpus: {len(df)} trajectories, {df['instance_id'].nunique()} instances")

    fim_dummies = pd.get_dummies(df["fim_form"], prefix="fim").astype(float)
    df = pd.concat([df, fim_dummies], axis=1)

    feature_cols = [
        "has_reproducer", "has_failing_test", "issue_length_words", "n_code_blocks",
        "ast_complexity", "ast_distinct_types", "loc_delta", "n_hunks", "n_files_touched",
    ] + [c for c in df.columns if c.startswith("fim_") and c != "fim_form"]
    X = df[feature_cols].astype(float).values
    y = df["passed"].values
    groups = df["instance_id"].values

    print(f"\nFeature matrix: {X.shape}; {len(feature_cols)} features ({sum(c.startswith('fim_') for c in feature_cols)} are FIM-form one-hot)")

    print("\n--- Joint model: gradient-boosted classifier, 5-fold grouped CV ---")
    gkf = GroupKFold(n_splits=5)
    aucs = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=fold)
        clf.fit(X[tr], y[tr])
        pred = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], pred))
    aucs = np.array(aucs)
    print(f"  Joint AUC = {aucs.mean():.4f} ± {aucs.std():.4f}  (folds: {[f'{a:.3f}' for a in aucs]})")

    print("\n--- Single-feature reference baselines ---")
    baseline_aucs = {}
    for feat in ["ease_84"]:
        ref = df[feat].values.reshape(-1, 1)
        ref_aucs = []
        for tr, te in gkf.split(ref, y, groups=groups):
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0)
            clf.fit(ref[tr], y[tr])
            ref_aucs.append(roc_auc_score(y[te], clf.predict_proba(ref[te])[:, 1]))
        baseline_aucs[feat] = float(np.mean(ref_aucs))
        print(f"  {feat:30s} alone:  AUC = {np.mean(ref_aucs):.4f}")

    print("\n--- Permutation feature importance (drop in AUC when feature shuffled) ---")
    clf_full = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf_full.fit(X, y)
    pi = permutation_importance(clf_full, X, y, n_repeats=20, random_state=0,
                                scoring="roc_auc", n_jobs=-1)
    importances = sorted(zip(feature_cols, pi.importances_mean, pi.importances_std),
                         key=lambda r: -r[1])

    print(f"\n  {'feature':35s}  {'importance':>10s}  {'std':>8s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*8}")
    for name, imp, std in importances[:15]:
        print(f"  {name:35s}  {imp:>10.4f}  {std:>8.4f}")

    print("\n--- Forward feature selection (greedy, by AUC gain) ---")
    remaining = list(range(X.shape[1]))
    selected, history = [], []
    while remaining:
        best_gain, best_idx = -np.inf, None
        for idx in remaining:
            cols = selected + [idx]
            sub_aucs = []
            for tr, te in gkf.split(X[:, cols], y, groups=groups):
                clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
                clf.fit(X[tr][:, cols], y[tr])
                sub_aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te][:, cols])[:, 1]))
            mean_auc = float(np.mean(sub_aucs))
            if mean_auc > best_gain:
                best_gain, best_idx = mean_auc, idx
        prev_auc = history[-1]["auc"] if history else 0.5
        gain = best_gain - prev_auc
        if gain < 0.002 and len(selected) >= 5:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        history.append({
            "step":     len(selected),
            "feature":  feature_cols[best_idx],
            "auc":      round(best_gain, 4),
            "delta":    round(gain, 4),
        })
        print(f"  step {len(selected):2d}: +{feature_cols[best_idx]:35s}  AUC = {best_gain:.4f}  Δ = +{gain:.4f}")

    OUT_JSON.write_text(json.dumps({
        "n_trajectories": int(len(df)),
        "n_instances":    int(df["instance_id"].nunique()),
        "n_features":     int(len(feature_cols)),
        "joint_cv_auc_mean": round(float(aucs.mean()), 4),
        "joint_cv_auc_std":  round(float(aucs.std()),  4),
        "joint_cv_auc_folds": [round(float(a), 4) for a in aucs],
        "baseline_aucs":   {k: round(v, 4) for k, v in baseline_aucs.items()},
        "permutation_importance": [
            {"feature": n, "importance_mean": round(float(i), 4), "importance_std": round(float(s), 4)}
            for n, i, s in importances
        ],
        "forward_selection": history,
    }, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
