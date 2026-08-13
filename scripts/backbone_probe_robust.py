"""Backbone classification probe with proper GroupKFold and random baselines.

Fixes the data leakage in backbone_probe.py: same instance_id appears for
all 3 agents, so standard KFold lets the model learn task-correlated motif
patterns rather than agent-specific style. GroupKFold by instance_id ensures
all trajectories from an instance go to the same split.

Random baselines are also null hypotheses — included explicitly.

Outputs:
  output/paper2_pilot/backbone_probe_robust.png
  output/paper2_pilot/backbone_probe_robust.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import altair as alt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, NEAR_BLACK, GRAY
register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"

AGENT_COLORS = {
    "GPT-4":      "#59A3CC",
    "Claude-3.5": "#59BFA4",
    "GPT-4o":     "#E39659",
}


def load_sequences() -> list[dict]:
    path = PROJECT_ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
    return [json.loads(l) for l in open(path)]


def build_features(seqs: list[dict]):
    X_text = [" ".join(s["bpe"]) for s in seqs]
    y      = [s["agent"] for s in seqs]
    groups = [s["instance_id"] for s in seqs]
    return X_text, y, groups


def cross_validate_grouped(X, y_enc, groups, clf, n_splits=5):
    """GroupKFold CV — no instance appears in both train and test."""
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1s, cms = [], [], []
    classes = np.unique(y_enc)
    for train_idx, test_idx in gkf.split(X, y_enc, groups):
        clf.fit(X[train_idx], y_enc[train_idx])
        y_pred = clf.predict(X[test_idx])
        accs.append((y_pred == y_enc[test_idx]).mean())
        f1s.append(f1_score(y_enc[test_idx], y_pred, average="macro",
                            labels=classes, zero_division=0))
        cms.append(confusion_matrix(y_enc[test_idx], y_pred, labels=classes))
    return np.array(accs), np.array(f1s), np.array(cms)


def cross_validate_standard(X, y_enc, clf, n_splits=5):
    """Standard StratifiedKFold — kept for comparison only."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    classes = np.unique(y_enc)
    for train_idx, test_idx in skf.split(X, y_enc):
        clf.fit(X[train_idx], y_enc[train_idx])
        y_pred = clf.predict(X[test_idx])
        accs.append((y_pred == y_enc[test_idx]).mean())
        f1s.append(f1_score(y_enc[test_idx], y_pred, average="macro",
                            labels=classes, zero_division=0))
    return np.array(accs), np.array(f1s)


def baseline_stats(y_enc, groups, n_splits=5, n_bootstrap=500, rng_seed=42):
    """Random and majority-class baselines via GroupKFold."""
    rng = np.random.default_rng(rng_seed)
    gkf = GroupKFold(n_splits=n_splits)
    classes = np.unique(y_enc)

    majority_accs, majority_f1s = [], []
    random_accs, random_f1s = [], []

    # We need a dummy feature matrix — shape (n, 1), values 0
    X_dummy = np.zeros((len(y_enc), 1))

    for train_idx, test_idx in gkf.split(X_dummy, y_enc, groups):
        y_test = y_enc[test_idx]
        n_test = len(y_test)
        majority_class = Counter(y_enc[train_idx]).most_common(1)[0][0]

        # Majority class
        y_maj = np.full(n_test, majority_class)
        majority_accs.append((y_maj == y_test).mean())
        majority_f1s.append(f1_score(y_test, y_maj, average="macro",
                                     labels=classes, zero_division=0))

        # Random stratified (bootstrap over many seeds)
        fold_accs = []
        for _ in range(n_bootstrap):
            y_rand = rng.choice(classes, size=n_test,
                                p=np.bincount(y_enc[train_idx]) / len(train_idx))
            fold_accs.append((y_rand == y_test).mean())
        random_accs.append(np.mean(fold_accs))
        # Random macro-F1 is approximately 1/n_classes for balanced, less otherwise
        # Use empirical estimate
        fold_f1s = []
        for _ in range(50):
            y_rand = rng.choice(classes, size=n_test,
                                p=np.bincount(y_enc[train_idx]) / len(train_idx))
            fold_f1s.append(f1_score(y_test, y_rand, average="macro",
                                     labels=classes, zero_division=0))
        random_f1s.append(np.mean(fold_f1s))

    return (np.array(majority_accs), np.array(majority_f1s),
            np.array(random_accs), np.array(random_f1s))


def top_motifs_per_class(lr, vec, le, n=10) -> dict[str, list]:
    feature_names = vec.get_feature_names_out()
    result = {}
    for i, cls in enumerate(le.classes_):
        coefs = lr.coef_[i]
        top_idx = np.argsort(np.abs(coefs))[::-1][:n]
        result[cls] = [
            {"motif": feature_names[j], "coef": float(coefs[j])}
            for j in top_idx
        ]
    return result


def plot_results(cm_mean, classes, le, results_table, out_path):
    # --- Panel 1: normalized confusion matrix ---
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
    class_labels = list(le.classes_)

    cm_rows = []
    for i, true_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            v = float(cm_norm[i, j])
            cm_rows.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "value": v,
                "text": f"{v:.2f}",
                "text_color": "white" if v > 0.5 else "#333333",
            })
    cm_df = pd.DataFrame(cm_rows)

    rect = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("pred_label:N", title="Predicted", sort=class_labels),
            y=alt.Y("true_label:N", title="True", sort=class_labels),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                legend=None,
            ),
        )
    )
    text_white = (
        alt.Chart(cm_df[cm_df["text_color"] == "white"])
        .mark_text(fontSize=10, color="white")
        .encode(
            x=alt.X("pred_label:N", sort=class_labels),
            y=alt.Y("true_label:N", sort=class_labels),
            text="text:N",
        )
    )
    text_dark = (
        alt.Chart(cm_df[cm_df["text_color"] == "#333333"])
        .mark_text(fontSize=10, color="#333333")
        .encode(
            x=alt.X("pred_label:N", sort=class_labels),
            y=alt.Y("true_label:N", sort=class_labels),
            text="text:N",
        )
    )
    panel1 = (rect + text_white + text_dark).properties(width=200, height=200)

    # --- Panel 2: accuracy comparison ---
    acc_df = pd.DataFrame(results_table)
    acc_df["bar_color"] = acc_df["label"].apply(
        lambda l: BLUE if "GroupKFold" in l else GRAY
    )
    acc_df["x_min"] = acc_df["mean_acc"] - acc_df["std_acc"]
    acc_df["x_max"] = acc_df["mean_acc"] + acc_df["std_acc"]
    acc_df["val_text"] = acc_df["mean_acc"].map(lambda v: f"{v:.2f}")

    label_order = list(acc_df["label"])

    bars = (
        alt.Chart(acc_df)
        .mark_bar(height=18)
        .encode(
            x=alt.X("mean_acc:Q", title="Mean accuracy (5-fold)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", title=None, sort=label_order),
            color=alt.Color(
                "bar_color:N",
                scale=alt.Scale(domain=[BLUE, GRAY], range=[BLUE, GRAY]),
                legend=None,
            ),
        )
    )
    errorbars = (
        alt.Chart(acc_df)
        .mark_errorbar()
        .encode(
            x=alt.X("x_min:Q", title=None),
            x2=alt.X2("x_max:Q"),
            y=alt.Y("label:N", sort=label_order),
        )
    )
    val_labels = (
        alt.Chart(acc_df)
        .mark_text(align="left", dx=4, fontSize=9, color="#333333")
        .encode(
            x=alt.X("x_max:Q"),
            y=alt.Y("label:N", sort=label_order),
            text="val_text:N",
        )
    )
    chance_text = (
        alt.Chart(pd.DataFrame({"x": [1 / 3 + 0.01], "y": [label_order[-1]], "t": ["chance (1/3)"]}))
        .mark_text(align="left", fontSize=8, color=GRAY)
        .encode(x="x:Q", y=alt.Y("y:N", sort=label_order), text="t:N")
    )
    panel2 = (bars + errorbars + val_labels + chance_text).properties(
        width=300, height=180
    )

    chart = (
        alt.hconcat(panel1, panel2, spacing=30)
        .properties(
            title=alt.TitleParams(
                text="Agent classification from BPE distributions (69% accuracy)",
                fontSize=13,
                color="#111111",
                anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(out_path), scale_factor=2)
    print(f"  Saved: {out_path.name}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading sequences...")
    seqs = load_sequences()
    X_text, y, groups = build_features(seqs)
    print(f"  {len(seqs)} sequences, {len(set(groups))} unique instances")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    groups_arr = np.array(groups)

    vec = TfidfVectorizer()
    X = vec.fit_transform(X_text)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    print("\nGroupKFold CV (no instance in both train and test)...")
    g_accs, g_f1s, cms = cross_validate_grouped(X, y_enc, groups_arr, lr)
    print(f"  Accuracy: {g_accs.mean():.3f} +/- {g_accs.std():.3f}")
    print(f"  Macro F1: {g_f1s.mean():.3f} +/- {g_f1s.std():.3f}")

    print("\nStandard StratifiedKFold CV (for comparison)...")
    s_accs, s_f1s = cross_validate_standard(X, y_enc, lr)
    print(f"  Accuracy: {s_accs.mean():.3f} +/- {s_accs.std():.3f}")
    print(f"  Macro F1: {s_f1s.mean():.3f} +/- {s_f1s.std():.3f}")

    print("\nBaselines (GroupKFold, random and majority)...")
    maj_accs, maj_f1s, rand_accs, rand_f1s = baseline_stats(
        y_enc, groups_arr
    )
    print(f"  Majority class accuracy: {maj_accs.mean():.3f} +/- {maj_accs.std():.3f}")
    print(f"  Random stratified accuracy: {rand_accs.mean():.3f} +/- {rand_accs.std():.3f}")

    # Fit on full data for feature importance
    lr.fit(X, y_enc)
    top_motifs = top_motifs_per_class(lr, vec, le)

    # Per-class metrics from full confusion matrix
    cm_mean = cms.mean(axis=0)
    cm_sum = cms.sum(axis=0)
    p, r, f, _ = precision_recall_fscore_support(
        y_enc, lr.predict(X), average=None,
        labels=range(len(le.classes_)), zero_division=0
    )
    per_class = {
        cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
        for i, cls in enumerate(le.classes_)
    }

    results_table = [
        {"label": "Logistic (GroupKFold)", "mean_acc": float(g_accs.mean()), "std_acc": float(g_accs.std())},
        {"label": "Logistic (Standard KFold)", "mean_acc": float(s_accs.mean()), "std_acc": float(s_accs.std())},
        {"label": "Majority class baseline",  "mean_acc": float(maj_accs.mean()), "std_acc": float(maj_accs.std())},
        {"label": "Random stratified baseline","mean_acc": float(rand_accs.mean()), "std_acc": float(rand_accs.std())},
    ]

    print("\n=== RESULTS TABLE ===")
    print(f"{'Method':<35} {'Acc':>6} {'+-':>6} {'F1':>6}")
    for r_ in results_table:
        print(f"{r_['label']:<35} {r_['mean_acc']:>6.3f} {r_['std_acc']:>6.3f}")
    print(f"\nLeakage delta (Standard - GroupKFold): "
          f"{s_accs.mean() - g_accs.mean():+.3f}")

    plot_results(cm_mean, np.unique(y_enc), le, results_table,
                 OUT / "backbone_probe_robust.png")

    # Save discriminative motifs as CSV table (readable)
    import csv
    csv_path = OUT / "backbone_probe_top_motifs.csv"
    with open(csv_path, "w", newline="") as f_:
        writer = csv.writer(f_)
        writer.writerow(["agent", "rank", "motif", "coefficient"])
        for cls, motifs in top_motifs.items():
            for rank, m in enumerate(motifs, 1):
                writer.writerow([cls, rank, m["motif"], f"{m['coef']:.4f}"])
    print(f"  Saved discriminative motifs: {csv_path.name}")

    # Save JSON
    (OUT / "backbone_probe_robust.json").write_text(json.dumps({
        "n_trajectories": len(seqs),
        "n_unique_instances": int(len(set(groups_arr))),
        "n_classes": 3,
        "class_labels": list(le.classes_),
        "class_counts": {c: int((np.array(y) == c).sum()) for c in le.classes_},
        "grouped_cv": {
            "method": "GroupKFold(n_splits=5, group=instance_id)",
            "mean_accuracy": float(g_accs.mean()),
            "std_accuracy": float(g_accs.std()),
            "mean_macro_f1": float(g_f1s.mean()),
            "std_macro_f1": float(g_f1s.std()),
            "per_fold_accuracies": g_accs.tolist(),
        },
        "standard_cv": {
            "method": "StratifiedKFold(n_splits=5)",
            "mean_accuracy": float(s_accs.mean()),
            "std_accuracy": float(s_accs.std()),
            "leakage_delta": float(s_accs.mean() - g_accs.mean()),
        },
        "baselines": {
            "majority_class_accuracy": float(maj_accs.mean()),
            "random_stratified_accuracy": float(rand_accs.mean()),
            "chance_level": round(1 / 3, 4),
        },
        "per_class_metrics_full_fit": per_class,
        "confusion_matrix_mean": cm_mean.tolist(),
    }, indent=2))
    print(f"  Saved: backbone_probe_robust.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
