"""
backbone_probe.py

Can we predict which backbone (GPT-4, Claude-3.5, GPT-4o) produced a BPE
trajectory sequence?  Tests whether procedural signatures are recoverable
from motif unigram TF features via logistic regression (L2).

Outputs
-------
output/paper2_pilot/backbone_probe.png   -- confusion matrix + top motifs figure
output/paper2_pilot/backbone_probe.json  -- numeric results
"""

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
DATA  = ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
OUT_DIR = ROOT / "output" / "paper2_pilot"
OUT_PNG  = OUT_DIR / "backbone_probe.png"
OUT_JSON = OUT_DIR / "backbone_probe.json"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "GPT-4":     "#59A3CC",
    "Claude-3.5": "#59BFA4",
    "GPT-4o":    "#E39659",
}
BG      = "#f5f5f5"
BORDER  = "#dddddd"
CLASSES = ["GPT-4", "Claude-3.5", "GPT-4o"]  # fixed order

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
records = []
with open(DATA) as fh:
    for line in fh:
        records.append(json.loads(line))

agents     = [r["agent"] for r in records]
bpe_lists  = [r["bpe"]   for r in records]   # list of motif strings per trajectory

print(f"Loaded {len(records)} trajectories")
print(f"  per agent: {dict(Counter(agents))}")

# ---------------------------------------------------------------------------
# 2. Build vocabulary and TF matrix  (raw counts, then divide by row sum)
# ---------------------------------------------------------------------------
vocab_set = set()
for bpe in bpe_lists:
    vocab_set.update(bpe)
vocab    = sorted(vocab_set)
motif2idx = {m: i for i, m in enumerate(vocab)}
V        = len(vocab)

print(f"Vocabulary size: {V} unique motifs")

# Build raw count matrix (dense float32 is fine for V~few-thousand)
N = len(records)
X_counts = np.zeros((N, V), dtype=np.float32)
for i, bpe in enumerate(bpe_lists):
    for motif in bpe:
        X_counts[i, motif2idx[motif]] += 1

# TF: divide each row by its total motif count
row_sums = X_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1          # guard against empty (shouldn't happen)
X_tf = X_counts / row_sums

# TF-IDF version
tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
X_tfidf = tfidf_transformer.fit_transform(csr_matrix(X_counts)).toarray()

# Encode labels
le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(agents)           # integer labels aligned to CLASSES order

# ---------------------------------------------------------------------------
# 3. Cross-validated evaluation (5-fold stratified)
# ---------------------------------------------------------------------------
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def _make_clf():
    return LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42
    )


def cv_evaluate(X, label):
    """Run 5-fold CV, return predictions and fold accuracies."""
    y_pred = cross_val_predict(_make_clf(), X, y, cv=CV)
    fold_accs = []
    for train_idx, test_idx in CV.split(X, y):
        clf = _make_clf()
        clf.fit(X[train_idx], y[train_idx])
        fold_accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    macro_f1 = f1_score(y, y_pred, average="macro")
    per_class_f1 = f1_score(y, y_pred, average=None, labels=le.transform(CLASSES))
    cm = confusion_matrix(y, y_pred, labels=le.transform(CLASSES))
    print(f"\n--- {label} ---")
    print(f"  Fold accuracies: {[round(a, 4) for a in fold_accs]}")
    print(f"  Mean accuracy  : {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    for cls, f1v in zip(CLASSES, per_class_f1):
        print(f"    {cls}: F1 = {f1v:.4f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print("  Classes:", CLASSES)
    print(f"  {cm}")
    return {
        "fold_accuracies": [round(a, 6) for a in fold_accs],
        "mean_accuracy":   round(mean_acc, 6),
        "std_accuracy":    round(std_acc, 6),
        "macro_f1":        round(macro_f1, 6),
        "per_class_f1":    {cls: round(f1v, 6) for cls, f1v in zip(CLASSES, per_class_f1)},
        "confusion_matrix": cm.tolist(),
        "y_pred":          y_pred,
    }

res_tf    = cv_evaluate(X_tf,    "TF (unigram)")
res_tfidf = cv_evaluate(X_tfidf, "TF-IDF")

# Choose the better model for figure + JSON
if res_tf["mean_accuracy"] >= res_tfidf["mean_accuracy"]:
    best_name  = "motif_tf"
    best_res   = res_tf
    X_best     = X_tf
    print(f"\nBest features: TF (acc={res_tf['mean_accuracy']:.4f} vs {res_tfidf['mean_accuracy']:.4f})")
else:
    best_name  = "motif_tfidf"
    best_res   = res_tfidf
    X_best     = X_tfidf
    print(f"\nBest features: TF-IDF (acc={res_tfidf['mean_accuracy']:.4f} vs {res_tf['mean_accuracy']:.4f})")

# ---------------------------------------------------------------------------
# 4. Fit final model on full data for feature importance
# ---------------------------------------------------------------------------
clf_final = _make_clf()
clf_final.fit(X_best, y)

# coef_ shape: (n_classes, n_features)
# Top 10 motifs by absolute coefficient per class
TOP_K = 10
top_motifs_per_class = {}
for ci, cls in enumerate(CLASSES):
    coefs = clf_final.coef_[ci]              # shape (V,)
    top_idx = np.argsort(np.abs(coefs))[::-1][:TOP_K]
    top_motifs_per_class[cls] = [
        {"motif": vocab[idx], "coef": round(float(coefs[idx]), 6)}
        for idx in top_idx
    ]
    print(f"\nTop-{TOP_K} motifs for {cls}:")
    for item in top_motifs_per_class[cls]:
        print(f"  {item['coef']:+.4f}  {item['motif']}")

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# -- Panel A: Confusion matrix (normalized by true class) --
ax_cm = axes[0]
ax_cm.set_facecolor(BG)
for spine in ax_cm.spines.values():
    spine.set_edgecolor(BORDER)
    spine.set_linewidth(0.8)

cm = np.array(best_res["confusion_matrix"], dtype=float)
cm_norm = cm / cm.sum(axis=1, keepdims=True)   # row-normalize

im = ax_cm.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")

# Annotate each cell
for i in range(3):
    for j in range(3):
        val = cm_norm[i, j]
        text_color = "white" if val > 0.55 else "#333333"
        ax_cm.text(j, i, f"{val:.2f}", ha="center", va="center",
                   fontsize=10, color=text_color, fontweight="normal")

ax_cm.set_xticks(range(3))
ax_cm.set_yticks(range(3))
ax_cm.set_xticklabels(CLASSES, fontsize=9)
ax_cm.set_yticklabels(CLASSES, fontsize=9)
ax_cm.set_xlabel("Predicted", fontsize=9, labelpad=6)
ax_cm.set_ylabel("True", fontsize=9, labelpad=6)
ax_cm.set_title(f"Backbone classification (5-fold CV, acc={best_res['mean_accuracy']:.2f})",
                fontsize=10, pad=8)

# -- Panel B: Top discriminative motifs per agent --
ax_m = axes[1]
ax_m.set_facecolor(BG)
for spine in ax_m.spines.values():
    spine.set_edgecolor(BORDER)
    spine.set_linewidth(0.8)

# Lay out bars: group by class, interleaved
y_pos_all   = []
bar_vals    = []
bar_colors  = []
bar_labels  = []
group_sep   = 0.6           # extra gap between classes
bar_height  = 0.35
current_y   = 0.0

for cls in CLASSES:
    items = top_motifs_per_class[cls]
    # sort by coef value (most positive to most negative for readability)
    items_sorted = sorted(items, key=lambda x: x["coef"])
    for item in items_sorted:
        y_pos_all.append(current_y)
        bar_vals.append(item["coef"])
        bar_colors.append(COLORS[cls])
        bar_labels.append(item["motif"])
        current_y += bar_height + 0.06
    current_y += group_sep

y_pos_all = np.array(y_pos_all)
bar_vals  = np.array(bar_vals)

ax_m.barh(y_pos_all, bar_vals, height=bar_height,
          color=bar_colors, edgecolor="none")
ax_m.axvline(0, color="#888888", linewidth=0.6, linestyle="--")

ax_m.set_yticks(y_pos_all)
ax_m.set_yticklabels(bar_labels, fontsize=6.5)
ax_m.set_xlabel("Logistic regression coefficient", fontsize=9, labelpad=6)
ax_m.set_title(f"Top {TOP_K} discriminative motifs per agent",
               fontsize=10, pad=8)
ax_m.tick_params(axis="x", labelsize=8)

# Legend for agent colors
patches = [mpatches.Patch(color=COLORS[cls], label=cls) for cls in CLASSES]
ax_m.legend(handles=patches, loc="lower right", fontsize=8,
            frameon=True, framealpha=0.85, edgecolor=BORDER)

plt.tight_layout(rect=[0, 0, 1, 1])
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\nFigure saved: {OUT_PNG}")

# ---------------------------------------------------------------------------
# 6. JSON output
# ---------------------------------------------------------------------------
output = {
    "n_trajectories": N,
    "n_classes":      3,
    "model":          "logistic_regression_l2",
    "features":       best_name,
    "cv_folds":       5,
    "mean_accuracy":  best_res["mean_accuracy"],
    "std_accuracy":   best_res["std_accuracy"],
    "macro_f1":       best_res["macro_f1"],
    "per_class_f1":   best_res["per_class_f1"],
    "confusion_matrix": best_res["confusion_matrix"],
    "top_motifs_per_class": {
        cls: [item["motif"] for item in items]
        for cls, items in top_motifs_per_class.items()
    },
    "tf_result": {
        "mean_accuracy": res_tf["mean_accuracy"],
        "std_accuracy":  res_tf["std_accuracy"],
        "macro_f1":      res_tf["macro_f1"],
    },
    "tfidf_result": {
        "mean_accuracy": res_tfidf["mean_accuracy"],
        "std_accuracy":  res_tfidf["std_accuracy"],
        "macro_f1":      res_tfidf["macro_f1"],
    },
}

with open(OUT_JSON, "w") as fh:
    json.dump(output, fh, indent=2)
print(f"JSON saved:   {OUT_JSON}")
print("\nDone.")
