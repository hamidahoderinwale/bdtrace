"""Phase 5: backbone probe on the 8-submission extended corpus.

Extends backbone_probe_robust.py to the 8-class problem (4 legacy SWE-agent
submissions + 4 new scaffolds/backbones). Class label = short agent name from
build_extended_bpe.SUBMISSION_LABEL.

Pipeline:
  - Read output/paper2_pilot/bpe_sequences_extended.jsonl
  - TF-IDF over BPE motifs (joined with spaces, default 1-gram)
  - Logistic regression (multinomial, L2)
  - GroupKFold(n_splits=5, group=instance_id) to prevent task leakage
  - Per-class F1, macro-F1, confusion matrix, top motif coefficients per class

Outputs:
  output/paper2_pilot/backbone_probe_extended.json
  output/paper2_pilot/backbone_probe_extended.png  (confusion matrix + accuracy bars)
  output/paper2_pilot/backbone_probe_extended_top_motifs.csv

Chance baseline = 1/8 = 12.5%.

Usage:
  python -m scripts.backbone_probe_extended
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import altair as alt
import pandas as pd
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
from scripts.theme import register, BLUE, GRAY
register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_FILE = OUT / "bpe_sequences_extended.jsonl"


def load_sequences() -> list[dict]:
    return [json.loads(l) for l in SEQ_FILE.open()]


def build_features(seqs: list[dict]):
    X_text = [" ".join(s["bpe"]) for s in seqs]
    y = [s["agent"] for s in seqs]
    groups = [s["instance_id"] for s in seqs]
    return X_text, y, groups


def cross_validate_grouped(X, y_enc, groups, clf, n_splits=5):
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


def baseline_stats(y_enc, groups, n_splits=5, n_bootstrap=200, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    gkf = GroupKFold(n_splits=n_splits)
    classes = np.unique(y_enc)
    majority_accs, random_accs = [], []
    X_dummy = np.zeros((len(y_enc), 1))
    for train_idx, test_idx in gkf.split(X_dummy, y_enc, groups):
        y_test = y_enc[test_idx]
        n_test = len(y_test)
        majority_class = Counter(y_enc[train_idx]).most_common(1)[0][0]
        y_maj = np.full(n_test, majority_class)
        majority_accs.append((y_maj == y_test).mean())
        rates = np.bincount(y_enc[train_idx], minlength=len(classes)) / max(len(train_idx), 1)
        rates = rates[:len(classes)] / rates.sum() if rates.sum() > 0 else None
        fold_accs = []
        for _ in range(n_bootstrap):
            y_rand = rng.choice(classes, size=n_test, p=rates)
            fold_accs.append((y_rand == y_test).mean())
        random_accs.append(np.mean(fold_accs))
    return np.array(majority_accs), np.array(random_accs)


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


def plot_confusion(cm_mean, le, mean_acc, out_path: Path) -> None:
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
    class_labels = list(le.classes_)
    rows = []
    for i, true_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            v = float(cm_norm[i, j])
            rows.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "value": v,
                "text": f"{v:.2f}",
            })
    df = pd.DataFrame(rows)
    rect = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("pred_label:N", sort=class_labels,
                    axis=alt.Axis(title="Predicted", labelAngle=-30)),
            y=alt.Y("true_label:N", sort=class_labels,
                    axis=alt.Axis(title="True")),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                            legend=None),
        )
    )
    text_white = (
        alt.Chart(df[df["value"] > 0.5])
        .mark_text(fontSize=10, color="white")
        .encode(
            x=alt.X("pred_label:N", sort=class_labels),
            y=alt.Y("true_label:N", sort=class_labels),
            text="text:N",
        )
    )
    text_dark = (
        alt.Chart(df[df["value"] <= 0.5])
        .mark_text(fontSize=10, color="#333333")
        .encode(
            x=alt.X("pred_label:N", sort=class_labels),
            y=alt.Y("true_label:N", sort=class_labels),
            text="text:N",
        )
    )
    chart = (
        (rect + text_white + text_dark)
        .properties(
            width=360, height=360,
            title=alt.TitleParams(
                f"8-class agent confusion ({mean_acc:.0%} acc)",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading sequences...")
    seqs = load_sequences()
    X_text, y, groups = build_features(seqs)
    print(f"  {len(seqs)} sequences  unique instances: {len(set(groups))}")
    cls_counts = Counter(y)
    for c, n in sorted(cls_counts.items()):
        print(f"    {c:30s}  n={n}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    groups_arr = np.array(groups)

    # token_pattern \S+ keeps multi-token BPE motifs (which contain '+') intact
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
    X = vec.fit_transform(X_text)
    print(f"  TF-IDF shape: {X.shape}")

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    print("\nGroupKFold (no instance in both train and test)...")
    g_accs, g_f1s, cms = cross_validate_grouped(X, y_enc, groups_arr, lr)
    print(f"  Acc:      {g_accs.mean():.3f} +/- {g_accs.std():.3f}")
    print(f"  Macro F1: {g_f1s.mean():.3f} +/- {g_f1s.std():.3f}")

    print("\nStandard StratifiedKFold (for leakage comparison)...")
    s_accs, s_f1s = cross_validate_standard(X, y_enc, lr)
    print(f"  Acc:      {s_accs.mean():.3f} +/- {s_accs.std():.3f}")
    print(f"  Macro F1: {s_f1s.mean():.3f} +/- {s_f1s.std():.3f}")

    print("\nBaselines (GroupKFold)...")
    maj_accs, rand_accs = baseline_stats(y_enc, groups_arr)
    print(f"  Majority class acc:   {maj_accs.mean():.3f}")
    print(f"  Random stratified:    {rand_accs.mean():.3f}")
    print(f"  Chance level (1/8):   {1 / 8:.3f}")

    # Fit on full data for feature importance
    lr.fit(X, y_enc)
    top_motifs = top_motifs_per_class(lr, vec, le, n=5)

    cm_mean = cms.mean(axis=0)
    p, r, f, _ = precision_recall_fscore_support(
        y_enc, lr.predict(X), average=None,
        labels=range(len(le.classes_)), zero_division=0,
    )
    per_class = {
        cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
        for i, cls in enumerate(le.classes_)
    }

    print("\n=== PER-CLASS F1 (full-fit, in-sample) ===")
    for cls, m in per_class.items():
        print(f"  {cls:30s}  F1={m['f1']:.3f}  (P={m['precision']:.3f}  R={m['recall']:.3f})")

    plot_confusion(cm_mean, le, g_accs.mean(), OUT / "backbone_probe_extended.png")
    print(f"  Saved: {OUT / 'backbone_probe_extended.png'}")

    # Save discriminative motifs CSV
    csv_path = OUT / "backbone_probe_extended_top_motifs.csv"
    with csv_path.open("w", newline="") as f_:
        writer = csv.writer(f_)
        writer.writerow(["agent", "rank", "motif", "coefficient"])
        for cls, motifs in top_motifs.items():
            for rank, m in enumerate(motifs, 1):
                writer.writerow([cls, rank, m["motif"], f"{m['coef']:.4f}"])
    print(f"  Saved: {csv_path.name}")

    # JSON summary
    (OUT / "backbone_probe_extended.json").write_text(json.dumps({
        "n_trajectories": len(seqs),
        "n_unique_instances": int(len(set(groups_arr))),
        "n_classes": len(le.classes_),
        "class_labels": list(le.classes_),
        "class_counts": {c: int(cls_counts[c]) for c in le.classes_},
        "grouped_cv": {
            "method": "GroupKFold(n_splits=5, group=instance_id)",
            "mean_accuracy": float(g_accs.mean()),
            "std_accuracy": float(g_accs.std()),
            "mean_macro_f1": float(g_f1s.mean()),
            "std_macro_f1": float(g_f1s.std()),
            "per_fold_accuracies": g_accs.tolist(),
            "per_fold_macro_f1":   g_f1s.tolist(),
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
            "chance_level": round(1 / len(le.classes_), 4),
        },
        "per_class_metrics_full_fit": per_class,
        "confusion_matrix_mean": cm_mean.tolist(),
        "top_5_motifs_per_class": top_motifs,
    }, indent=2))
    print(f"  Saved: backbone_probe_extended.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
