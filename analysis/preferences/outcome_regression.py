"""Logistic regression: which motifs predict task resolution?

Predicts binary outcome (resolved yes/no) from per-trajectory motif
frequency vectors, controlling for agent identity and task difficulty.
This controls for the confounders in the unconditional step_resources
associations — some motifs look wasteful simply because they appear in
harder tasks or because a specific agent uses them.

Model:
    y = resolved (1/0)
    X = [motif_freqs] + [agent_dummies (8, one-hot vs reference)] + [n_resolved_loo]
    Logistic regression, L2 regularization, class_weight='balanced'
    5-fold stratified cross-validation

Operates on the 9-agent extended corpus. Agent dummies span all nine
submissions; reference is Claude-3.5. Difficulty bin n_resolved_loo
counts other agents (out of 8) that solved each task, so it doesn't
leak the regressed agent's own outcome.

Outputs:
    output/paper2_pilot/outcome_regression.json  — coefficients + CV scores
    output/paper2_pilot/outcome_regression.png   — top-20 motif coefficients
    output/paper2_pilot/outcome_regression_cv.png — CV ROC curves

Usage:
    python -m analysis.preferences.outcome_regression
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, VERMILLION, SKY, GRAY
register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences_extended.jsonl"
DIVERSITY_PATH = OUT / "task_diversity_extended.jsonl"
LB_PATH = OUT / "extended_pass_fail.json"

# Nine submissions, agent short names; Claude-3.5 is the dummy-encoding
# reference (its column is dropped from the agent one-hot block).
AGENT_LONG = {
    "Claude-3":              "20240402_sweagent_claude3opus",
    "GPT-4":                 "20240402_sweagent_gpt4",
    "Claude-3.5":            "20240620_sweagent_claude3.5sonnet",
    "GPT-4o":                "20240728_sweagent_gpt4o",
    "Claude-3.7-thinking":   "20250226_sweagent_claude-3-7-sonnet-20250219",
    "Claude-4":              "20250526_sweagent_claude-4-sonnet-20250514",
    "Agentless+Claude-3.5":  "20241202_agentless-1.5_claude-3.5-sonnet-20241022",
    "DARS+R1":               "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1",
    "Moatless+V3":           "20250111_moatless_deepseek_v3",
}
AGENT_DUMMY_REF = "Claude-3.5"
AGENT_DUMMY_COLS = [a for a in AGENT_LONG if a != AGENT_DUMMY_REF]


def load_data() -> pd.DataFrame:
    seqs = [json.loads(l) for l in open(SEQ_PATH) if l.strip()]
    lb = json.load(open(LB_PATH))

    # extended_pass_fail.json shape: {sub_id: {"resolved": [iid, ...], ...}}.
    # Convert to the per-instance per-agent True/False map the rest of
    # load_data expects.
    per_instance: dict[str, dict[str, bool]] = {}
    seq_instance_ids: set[str] = {s["instance_id"] for s in seqs}
    for short, long in AGENT_LONG.items():
        resolved_set = set(lb.get(long, {}).get("resolved", []))
        for inst in seq_instance_ids:
            per_instance.setdefault(inst, {})[short] = inst in resolved_set

    vocab: list[str] = sorted({t for s in seqs for t in s["bpe"]})

    rows = []
    for s in seqs:
        agent = s["agent"]
        inst  = s["instance_id"]
        inst_outcomes = per_instance.get(inst, {})
        passed = inst_outcomes.get(agent, False)
        # Leave-one-out difficulty: how many OTHER agents solved this task
        n_resolved_loo = sum(
            1 for a, ok in inst_outcomes.items() if a != agent and ok
        )
        total = len(s["bpe"])
        freqs = Counter(s["bpe"])
        row = {
            "agent": agent,
            "instance_id": inst,
            "resolved": int(passed),
            "n_resolved_loo": n_resolved_loo,  # 0, 1, or 2 — does NOT encode this agent's outcome
        }
        for v in vocab:
            row[v] = freqs.get(v, 0) / total if total > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows), vocab


def _agent_dummy_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot encoding for the 8 non-reference agents (reference is
    AGENT_DUMMY_REF, encoded implicitly by all-zeros)."""
    cols = []
    names = []
    for a in AGENT_DUMMY_COLS:
        cols.append((df["agent"] == a).astype(float).values)
        names.append(f"agent_{a}")
    return np.column_stack(cols), names


def build_features(
    df: pd.DataFrame,
    vocab: list[str],
    include_difficulty: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    agent_mat, agent_names = _agent_dummy_matrix(df)
    motif_mat = StandardScaler().fit_transform(df[vocab].values)
    y = df["resolved"].values

    if include_difficulty:
        n_res_loo = df["n_resolved_loo"].values.astype(float)
        n_res_scaled = (n_res_loo - n_res_loo.mean()) / max(n_res_loo.std(), 1e-9)
        X = np.column_stack([motif_mat, agent_mat, n_res_scaled.reshape(-1, 1)])
        feature_names = vocab + agent_names + ["n_resolved_loo"]
    else:
        X = np.column_stack([motif_mat, agent_mat])
        feature_names = vocab + agent_names
    return X, y, feature_names


def build_ablation_features(df: pd.DataFrame, vocab: list[str], mode: str) -> tuple[np.ndarray, np.ndarray]:
    agent_mat, _ = _agent_dummy_matrix(df)
    motif_mat = StandardScaler().fit_transform(df[vocab].values)
    y = df["resolved"].values
    if mode == "agent_only":
        return agent_mat, y
    if mode == "motifs_only":
        return motif_mat, y
    raise ValueError(mode)


def fit_model(X: np.ndarray, y: np.ndarray) -> LogisticRegressionCV:
    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty="l2",
        class_weight="balanced",
        scoring="roc_auc",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, aps = [], []
    fold_curves = []
    oof_scores = []
    for fold_idx, (train, test) in enumerate(cv.split(X, y)):
        model = LogisticRegressionCV(
            Cs=10, cv=3, penalty="l2",
            class_weight="balanced", scoring="roc_auc",
            solver="lbfgs", max_iter=1000, random_state=42,
        )
        model.fit(X[train], y[train])
        proba = model.predict_proba(X[test])[:, 1]
        auc = roc_auc_score(y[test], proba)
        ap  = average_precision_score(y[test], proba)
        aucs.append(auc)
        aps.append(ap)
        fpr, tpr, _ = roc_curve(y[test], proba)
        fold_curves.append((fpr, tpr, auc))
        for yt, ys in zip(y[test], proba):
            oof_scores.append({"y_true": int(yt), "score": float(ys)})
        print(f"  Fold {fold_idx+1}: AUC={auc:.3f}  AP={ap:.3f}")
    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "mean_ap":  float(np.mean(aps)),
        "std_ap":   float(np.std(aps)),
        "fold_aucs": [float(a) for a in aucs],
        "fold_curves": [(fpr.tolist(), tpr.tolist(), float(auc)) for fpr, tpr, auc in fold_curves],
        "oof_scores": oof_scores,
    }


def plot_coefficients(coef: np.ndarray, feature_names: list[str], out_path: Path, top_n: int = 20, title_suffix: str = "") -> None:
    motif_coefs = [(name, c) for name, c in zip(feature_names, coef) if "+" in name]
    motif_coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = motif_coefs[:top_n]
    top.sort(key=lambda x: x[1])  # ascending so largest positive is at top in horizontal bar

    def abbrev(m: str) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            return m.replace("+", " -> ")
        return f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)} steps)"

    rows = [
        {
            "motif": abbrev(n),
            "coef": c,
            "direction": "predicts resolution" if c > 0 else "predicts failure",
        }
        for n, c in top
    ]
    df_coef = pd.DataFrame(rows)
    motif_order = [r["motif"] for r in rows]

    color_scale = alt.Scale(
        domain=["predicts resolution", "predicts failure"],
        range=[BLUE, VERMILLION],
    )

    chart = (
        alt.Chart(df_coef)
        .mark_bar()
        .encode(
            y=alt.Y("motif:N", sort=motif_order,
                    axis=alt.Axis(title=None, labelLimit=400, labelFontSize=9,
                                  domain=False, ticks=False)),
            x=alt.X("coef:Q",
                    axis=alt.Axis(title="Logistic regression coefficient",
                                  domain=False, ticks=False, format=".2f")),
            color=alt.Color("direction:N", scale=color_scale,
                            legend=alt.Legend(orient="bottom", title=None)),
        )
        .properties(
            width=480,
            height=max(280, top_n * 22),
            title=alt.TitleParams(
                text=f"Top {top_n} action patterns by association with task resolution",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axisX(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_oof_roc(cv_result: dict, out_path: Path) -> None:
    """Single ROC curve from out-of-fold predicted scores across all 5 folds."""
    scores_df = pd.DataFrame(cv_result["oof_scores"])
    y_true = scores_df["y_true"].values
    y_score = scores_df["score"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    diag_df  = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})

    roc_line = (
        alt.Chart(curve_df)
        .mark_line(color=BLUE, strokeWidth=2)
        .encode(
            x=alt.X("fpr:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(
                        title="False positive rate",
                        domain=False, ticks=False, format=".0%",
                        grid=True, gridColor="#F0F0F0", gridWidth=0.3,
                    )),
            y=alt.Y("tpr:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(
                        title="True positive rate",
                        domain=False, ticks=False, format=".0%",
                        grid=True, gridColor="#F0F0F0", gridWidth=0.3,
                    )),
        )
    )

    annotation_df = pd.DataFrame([{
        "fpr": 0.97, "tpr": 0.06,
        "text": f"Area under the curve = {auc:.2f}",
    }])
    annotation = (
        alt.Chart(annotation_df)
        .mark_text(align="right", baseline="bottom", fontSize=11, color=BLUE)
        .encode(
            x=alt.X("fpr:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", scale=alt.Scale(domain=[0, 1])),
            text=alt.Text("text:N"),
        )
    )

    chart = (
        alt.layer(roc_line, annotation)
        .properties(
            width=380,
            height=320,
            title=alt.TitleParams(
                text="Task resolution predicted from action patterns",
                fontSize=13,
                fontWeight="normal",
                color="#111111",
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_cv_roc(cv_result: dict, out_path: Path, title: str = "") -> None:
    n_folds = len(cv_result["fold_curves"])
    fold_colors = [BLUE, ORANGE, GREEN, VERMILLION, SKY][:n_folds]

    rows = []
    fold_order = []
    for i, (fpr, tpr, auc) in enumerate(cv_result["fold_curves"]):
        label = f"Fold {i + 1}  (AUC = {auc:.3f})"
        fold_order.append(label)
        for x, y in zip(fpr, tpr):
            rows.append({"fpr": x, "tpr": y, "fold": label})

    df_roc = pd.DataFrame(rows)
    mean_auc = cv_result["mean_auc"]
    std_auc  = cv_result["std_auc"]

    roc_lines = (
        alt.Chart(df_roc)
        .mark_line(opacity=0.75, strokeWidth=1.8)
        .encode(
            x=alt.X("fpr:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(
                        title="Share of failed trajectories incorrectly predicted as resolved",
                        domain=False, ticks=False, format=".0%",
                    )),
            y=alt.Y("tpr:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(
                        title="Share of resolved trajectories correctly identified",
                        domain=False, ticks=False, format=".0%",
                    )),
            color=alt.Color("fold:N",
                            sort=fold_order,
                            scale=alt.Scale(domain=fold_order, range=fold_colors),
                            legend=alt.Legend(orient="bottom", title=None,
                                              columns=3, symbolStrokeWidth=2)),
        )
    )

    chart = (
        roc_lines
        .properties(
            width=460,
            height=320,
            title=alt.TitleParams(
                text="Outcome prediction from action patterns",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axisY(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def run_model(X: np.ndarray, y: np.ndarray, feature_names: list[str], label: str) -> dict:
    print(f"\n--- {label} ---")
    print(f"  Feature matrix: {X.shape}, resolved: {y.sum()} / {len(y)}")
    print("  Cross-validation (5 folds):")
    cv_result = cross_validate(X, y)
    print(f"  Mean AUC = {cv_result['mean_auc']:.3f} ± {cv_result['std_auc']:.3f}")
    print(f"  Mean AP  = {cv_result['mean_ap']:.3f} ± {cv_result['std_ap']:.3f}")

    print("  Fitting full model...")
    model = fit_model(X, y)
    coef = model.coef_[0]
    best_C = float(model.C_[0])
    print(f"  Best C = {best_C:.4f}")

    motif_coefs = sorted(
        [(name, float(c)) for name, c in zip(feature_names, coef) if "+" in name],
        key=lambda x: x[1], reverse=True,
    )
    ctrl_coefs = {n: float(c) for n, c in zip(feature_names, coef) if "+" not in n}

    print(f"  Top 5 motifs predicting resolution:")
    for name, c in motif_coefs[:5]:
        print(f"    {name[:55]:<55s}  β={c:+.4f}")
    print(f"  Top 5 motifs predicting failure:")
    for name, c in motif_coefs[-5:]:
        print(f"    {name[:55]:<55s}  β={c:+.4f}")
    print(f"  Controls: {ctrl_coefs}")

    return {
        "label": label,
        "best_C": best_C,
        "cv": cv_result,
        "top_positive_motifs": motif_coefs[:10],
        "top_negative_motifs": motif_coefs[-10:],
        "control_coefficients": ctrl_coefs,
        "all_motif_coefs": motif_coefs,
        "_model": model,
        "_coef": coef,
        "_feature_names": feature_names,
    }


def plot_auc_comparison(comparisons: list[dict], out_path: Path) -> None:
    """Horizontal dot plot comparing AUC across feature ablations."""
    df = pd.DataFrame(comparisons)
    label_order = [r["label"] for r in comparisons]

    band_df = pd.DataFrame([
        {"label": lbl, "band": i % 2}
        for i, lbl in enumerate(label_order)
    ])

    bands = (
        alt.Chart(band_df)
        .mark_rect(opacity=0.06)
        .encode(
            y=alt.Y("label:N", sort=label_order),
            color=alt.condition(
                alt.datum.band == 0,
                alt.value("#000000"),
                alt.value("transparent"),
            ),
        )
    )

    dots = (
        alt.Chart(df)
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X("auc:Q",
                    scale=alt.Scale(domain=[0.0, 1.0]),
                    axis=alt.Axis(title="Area under the curve", domain=False, ticks=False,
                                  grid=True, gridColor="#F0F0F0", gridWidth=0.3,
                                  values=[0.0, 0.25, 0.5, 0.75, 1.0], format=".2f")),
            y=alt.Y("label:N", sort=label_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False, labelFontSize=11)),
            color=alt.value(BLUE),
        )
    )

    labels = dots.mark_text(dx=10, align="left", fontSize=10, color="#555555").encode(
        text=alt.Text("auc:Q", format=".2f")
    )

    chart = (
        alt.layer(bands, dots, labels)
        .properties(
            width=340,
            height=120,
            title=alt.TitleParams(
                text="Task resolution predicted from action patterns",
                fontSize=13, fontWeight="normal", color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
        .configure_axisX(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    print("Loading data...")
    df, vocab = load_data()
    print(f"  {len(df)} trajectories, {len(vocab)} motif features, "
          f"{df['resolved'].sum()} resolved ({df['resolved'].mean():.1%})")
    print(f"  n_resolved_loo distribution: "
          f"{df['n_resolved_loo'].value_counts().sort_index().to_dict()}")

    # Ablations
    print("\n--- Ablation: agent identity only ---")
    X_ag, y = build_ablation_features(df, vocab, "agent_only")
    cv_ag = cross_validate(X_ag, y)
    auc_agent = roc_auc_score(
        [s["y_true"] for s in cv_ag["oof_scores"]],
        [s["score"]  for s in cv_ag["oof_scores"]],
    )
    print(f"  OOF AUC = {auc_agent:.3f}")

    print("\n--- Ablation: motifs only ---")
    X_mo, y = build_ablation_features(df, vocab, "motifs_only")
    cv_mo = cross_validate(X_mo, y)
    auc_motifs = roc_auc_score(
        [s["y_true"] for s in cv_mo["oof_scores"]],
        [s["score"]  for s in cv_mo["oof_scores"]],
    )
    print(f"  OOF AUC = {auc_motifs:.3f}")

    # Model A: motif + agent only (no difficulty control)
    Xa, y, fna = build_features(df, vocab, include_difficulty=False)
    res_a = run_model(Xa, y, fna, "Model A: motifs + agent (no difficulty)")
    auc_a = roc_auc_score(
        [s["y_true"] for s in res_a["cv"]["oof_scores"]],
        [s["score"]  for s in res_a["cv"]["oof_scores"]],
    )

    # Model B: motif + agent + LOO difficulty (clean covariate)
    Xb, y, fnb = build_features(df, vocab, include_difficulty=True)
    res_b = run_model(Xb, y, fnb, "Model B: motifs + agent + n_resolved_loo")

    result = {
        "n_trajectories": len(df),
        "n_resolved": int(df["resolved"].sum()),
        "pass_rate": float(df["resolved"].mean()),
        "n_motif_features": len(vocab),
        "ablations": {
            "agent_only_auc": auc_agent,
            "motifs_only_auc": auc_motifs,
            "motifs_plus_agent_auc": auc_a,
        },
        "model_a": {k: v for k, v in res_a.items() if not k.startswith("_")},
        "model_b": {k: v for k, v in res_b.items() if not k.startswith("_")},
    }

    (OUT / "outcome_regression.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    plot_coefficients(
        res_b["_coef"], res_b["_feature_names"],
        OUT / "outcome_regression.png",
        title_suffix="controlling for agent identity and task difficulty (leave-one-out)",
    )
    plot_oof_roc(res_a["cv"], OUT / "outcome_regression_cv.png")

    comparisons = [
        {"label": "Agent identity",           "auc": auc_agent},
        {"label": "Action patterns",          "auc": auc_motifs},
        {"label": "Action patterns + agent",  "auc": auc_a},
    ]
    plot_auc_comparison(comparisons, OUT / "outcome_regression_ablation.png")

    print(f"\nSaved:")
    for n in ["outcome_regression.json", "outcome_regression.png",
              "outcome_regression_cv.png", "outcome_regression_ablation.png"]:
        print(f"  {OUT / n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
