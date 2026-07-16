#!/usr/bin/env python3
"""
Cross-model transfer: does a structural library from one model predict success for another?

For each task T and source model S:
  coverage(T, S, repr, K) = fraction of T's K nearest structural neighbors solved by S

We use coverage as a predictor of whether a *different* target model solves T.
If AUC(coverage → target_pass) > 0.5 across (source, target) pairs, the structural
signal is model-agnostic — evidence of genuine structural regularity.

Usage:
  uv run python scripts/run_cross_model_transfer.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("output/datasets/swe_bench_lite_resolved")
RESULTS_DIR = Path("output/swebench_results")
PLOTS_DIR = Path("notebooks/plots/multi_benchmark")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_MODELS = {
    "lite_20240402_sweagent_gpt4.json": "GPT-4 (SWE-agent)",
    "lite_20240620_sweagent_claude3.5sonnet.json": "Claude 3.5 (SWE-agent)",
    "lite_20240728_sweagent_gpt4o.json": "GPT-4o (SWE-agent)",
    "lite_20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128.json": "Qwen2.5-72b (SWE-Fixer)",
}

_REPR_ORDER = ["modules", "edits_set_diff", "motifs", "edits"]

_WONG = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


def load_pass_fail(path: Path) -> dict[str, bool]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {k: bool(v) for k, v in data.items()}
    return {r["instance_id"]: bool(r.get("resolved", r.get("pass", False))) for r in data}


def load_matrices() -> tuple[list[str], dict[str, np.ndarray]]:
    labels = pd.read_parquet(DATA_DIR / "labels.parquet")
    instance_ids = labels["instance_id"].tolist()
    mats = np.load(DATA_DIR / "matrices.npz")
    return instance_ids, {k: mats[k] for k in mats}


def coverage_scores(
    dist_matrix: np.ndarray,
    pass_fail: list[bool],
    k: int,
) -> np.ndarray:
    """For each task i, coverage = fraction of its K nearest neighbors (excl. self) solved."""
    n = len(pass_fail)
    solved_mask = np.array(pass_fail, dtype=float)
    scores = np.zeros(n)
    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf  # exclude self
        nn_idx = np.argsort(row)[:k]
        scores[i] = solved_mask[nn_idx].mean()
    return scores


def compute_auc(y_score: np.ndarray, y_true: list[bool]) -> float | None:
    """Return AUC or None if only one class present."""
    y_true_arr = np.array(y_true)
    if y_true_arr.sum() == 0 or y_true_arr.sum() == len(y_true_arr):
        return None
    return roc_auc_score(y_true_arr, y_score)


def main() -> None:
    instance_ids, matrices = load_matrices()

    ## Load pass/fail for each model
    model_pf: dict[str, list[bool]] = {}
    for fname, label in _MODELS.items():
        pf_path = RESULTS_DIR / fname
        if not pf_path.exists():
            print(f"  [skip] {fname} not found")
            continue
        pf_map = load_pass_fail(pf_path)
        model_pf[label] = [pf_map.get(iid, False) for iid in instance_ids]
        print(f"  {label}: {sum(model_pf[label])}/{len(instance_ids)} solved")

    models = list(model_pf.keys())
    K_VALUES = [5, 10, 20]
    repr_names = [r for r in _REPR_ORDER if r in matrices]

    ## Compute coverage scores for every (repr, K, source_model)
    print("\nComputing coverage scores...")
    coverage: dict[tuple[str, int, str], np.ndarray] = {}
    for repr_name in repr_names:
        D = matrices[repr_name]
        for k in K_VALUES:
            for src in models:
                coverage[(repr_name, k, src)] = coverage_scores(D, model_pf[src], k)

    ## AUC table: (source_model, target_model, repr, K) → AUC
    print("Computing AUCs...")
    rows = []
    for repr_name in repr_names:
        for k in K_VALUES:
            for src in models:
                for tgt in models:
                    cov = coverage[(repr_name, k, src)]
                    auc = compute_auc(cov, model_pf[tgt])
                    if auc is not None:
                        rows.append(
                            {
                                "source": src,
                                "target": tgt,
                                "repr": repr_name,
                                "K": k,
                                "auc": auc,
                                "cross_model": src != tgt,
                            }
                        )

    df = pd.DataFrame(rows)

    ## Save full results
    df.to_parquet(DATA_DIR / "cross_model_transfer.parquet", index=False)
    print(f"Saved cross_model_transfer.parquet ({len(df)} rows)")

    ## ── Plot 1: AUC heatmap (K=10, edits_set_diff) ─────────────────────────
    pivot_df = (
        df[(df["K"] == 10) & (df["repr"] == "edits_set_diff")]
        .groupby(["source", "target"])["auc"]
        .mean()
        .reset_index()
    )

    heatmap = (
        alt.Chart(pivot_df)
        .mark_rect()
        .encode(
            alt.X("source:N", title="source model (library built from)"),
            alt.Y("target:N", title="target model (prediction for)"),
            alt.Color(
                "auc:Q",
                scale=alt.Scale(scheme="blues", domain=[0.4, 0.75]),
                title="AUC",
            ),
            alt.Tooltip(["source:N", "target:N", "auc:Q"]),
        )
        .properties(width=360, height=280)
    )

    text = (
        alt.Chart(pivot_df)
        .mark_text(fontSize=11)
        .encode(
            alt.X("source:N"),
            alt.Y("target:N"),
            alt.Text("auc:Q", format=".2f"),
            color=alt.condition(
                "datum.auc > 0.62",
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    (heatmap + text).properties(
        title=alt.TitleParams(
            text="Cross-model transfer AUC (K=10, edit-operation distance)",
            subtitle="Diagonal = self-prediction. Off-diagonal = cross-model transfer. Random = 0.50.",
            anchor="start",
        )
    ).save(PLOTS_DIR / "cross_model_transfer_heatmap.png")
    print("Saved cross_model_transfer_heatmap.png")

    ## ── Plot 2: AUC by repr and K for cross-model pairs only ────────────────
    cross_df = df[df["cross_model"]].copy()
    agg = (
        cross_df.groupby(["repr", "K"])["auc"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    base = alt.Chart(agg).encode(
        alt.X("K:O", title="K (neighbors)"),
        alt.Color(
            "repr:N",
            scale=alt.Scale(domain=repr_names, range=_WONG[: len(repr_names)]),
            title="representation",
        ),
    )

    line = base.mark_line(point=True).encode(
        alt.Y("mean:Q", title="mean AUC (cross-model pairs)", scale=alt.Scale(domain=[0.40, 0.70])),
    )
    band = base.mark_area(opacity=0.2).encode(
        alt.Y("y_lo:Q"),
        alt.Y2("y_hi:Q"),
    ).transform_calculate(
        y_lo="datum.mean - datum.std",
        y_hi="datum.mean + datum.std",
    )

    (band + line).properties(
        width=300,
        height=220,
        title=alt.TitleParams(
            text="Cross-model transfer AUC by representation and K",
            subtitle="Mean ± SD over all cross-model (source→target) pairs.",
            anchor="start",
        ),
    ).save(PLOTS_DIR / "cross_model_transfer_by_repr.png")
    print("Saved cross_model_transfer_by_repr.png")

    ## ── Plot 3: Coverage distribution by outcome (K=10, modules, best cross pair) ──
    best_row = (
        df[(df["K"] == 10) & (df["repr"] == "edits_set_diff") & df["cross_model"]]
        .sort_values("auc", ascending=False)
        .iloc[0]
    )
    best_src = best_row["source"]
    best_tgt = best_row["target"]
    print(f"\nBest cross-model pair: {best_src} → {best_tgt}, AUC={best_row['auc']:.3f}")

    cov_arr = coverage[("edits_set_diff", 10, best_src)]
    tgt_pass = model_pf[best_tgt]

    dist_df = pd.DataFrame(
        {
            "coverage": cov_arr,
            "outcome": ["solved" if p else "failed" for p in tgt_pass],
        }
    )

    cov_plot = (
        alt.Chart(dist_df)
        .mark_bar(opacity=0.7, binSpacing=1)
        .encode(
            alt.X("coverage:Q", bin=alt.Bin(step=0.1), title="coverage score (fraction of K=10 neighbors solved by source model)"),
            alt.Y("count()", title="tasks"),
            alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=["solved", "failed"], range=["#0072B2", "#E69F00"]),
            ),
        )
        .properties(
            width=340,
            height=200,
            title=alt.TitleParams(
                text=f"Coverage predicts target success: {best_src.split(' ')[0]} → {best_tgt.split(' ')[0]}",
                subtitle=f"AUC = {best_row['auc']:.3f}. K=10 nearest neighbors in modules space.",
                anchor="start",
            ),
        )
    )

    cov_plot.save(PLOTS_DIR / "cross_model_coverage_dist.png")
    print("Saved cross_model_coverage_dist.png")

    ## Summary
    print("\n── Summary ──")
    for repr_name in repr_names:
        cross_k10 = df[(df["K"] == 10) & (df["repr"] == repr_name) & df["cross_model"]]
        self_k10 = df[(df["K"] == 10) & (df["repr"] == repr_name) & ~df["cross_model"]]
        print(f"{repr_name:20s}: cross-model AUC mean={cross_k10['auc'].mean():.3f} [{cross_k10['auc'].min():.3f}–{cross_k10['auc'].max():.3f}],  self={self_k10['auc'].mean():.3f}")


if __name__ == "__main__":
    main()
