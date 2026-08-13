"""Top-motif ablation on the 9-class agent probe.

Trains a 9-class logistic regression probe on TF-IDF motif features (the same
setup as backbone_probe_extended), ranks motifs by their discriminative power
for the classifier, then iteratively drops the top-K most discriminative
motifs and retrains. Reports how probe accuracy degrades as a function of K.

Question answered: is the 86% probe accuracy concentrated in a small set of
diagnostic motifs, or distributed across the 124-motif vocabulary?

Discriminative power ranked by sum of absolute coefficients across classes
in a one-vs-rest logistic regression — the standard interpretability metric
for linear classifiers.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
Writes:
    output/paper2_pilot/motif_ablation_probe.json
    output/figures/fig_motif_ablation_probe.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, MAGENTA, OLIVE
register()

SEQ = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
OUT_DAT = ROOT / "output" / "paper2_pilot" / "motif_ablation_probe.json"
OUT_FIG = ROOT / "output" / "figures" / "fig_motif_ablation_probe.png"

ABLATION_LEVELS = [0, 1, 2, 5, 10, 20, 50, 100]  # K = number of motifs dropped
N_FOLDS = 5


def load_records() -> list[dict]:
    with SEQ.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_motifs_by_coef(X, y, vec) -> list[tuple[str, float]]:
    """Train LR on the full corpus, return motifs ranked by sum-of-abs-coefs.

    For multi-class LR, sklearn produces (n_classes, n_features) coefficient
    matrix. We rank features by sum of absolute coefficients across classes —
    the standard "this feature is discriminative for many classes" measure.
    """
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    clf.fit(X, y)
    abs_coefs = np.abs(clf.coef_).sum(axis=0)
    feature_names = vec.get_feature_names_out()
    ranked = sorted(zip(feature_names, abs_coefs), key=lambda kv: -kv[1])
    return ranked


def cv_accuracy(records, drop_motifs: set[str]) -> dict:
    """Train + evaluate the 9-class probe with `drop_motifs` removed.

    Standard setup: TF-IDF on motif sequences (drop_motifs removed via tokenizer
    filter), 5-fold GroupKFold by instance_id, return mean accuracy + macro F1.
    """
    # Pre-filter the bpe sequences to remove dropped motifs
    texts = []
    labels = []
    groups = []
    for r in records:
        filtered = [m for m in r["bpe"] if m not in drop_motifs]
        if not filtered:
            continue  # no motifs left after ablation
        texts.append(" ".join(filtered))
        labels.append(r["agent"])
        groups.append(r["instance_id"])

    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
    X = vec.fit_transform(texts)
    y = np.array(labels)
    g = np.array(groups)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    gkf = GroupKFold(n_splits=N_FOLDS)
    accs, f1s = [], []
    for train_idx, test_idx in gkf.split(X, y_enc, groups=g):
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        clf.fit(X[train_idx], y_enc[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append((pred == y_enc[test_idx]).mean())

        from sklearn.metrics import f1_score
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "n_features_kept": X.shape[1],
        "n_trajectories": len(texts),
    }


def main() -> None:
    records = load_records()
    print(f"Loaded {len(records)} trajectories")

    # Train once on the full corpus to get the motif ranking
    full_texts = [" ".join(r["bpe"]) for r in records]
    full_y = [r["agent"] for r in records]
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
    X = vec.fit_transform(full_texts)
    le = LabelEncoder()
    y_enc = le.fit_transform(full_y)
    print("Ranking motifs by discriminative power ...")
    ranked = rank_motifs_by_coef(X, y_enc, vec)
    print(f"Top 20 motifs by sum-of-abs-coefs:")
    for motif, score in ranked[:20]:
        print(f"  {score:8.3f}  {motif}")

    # Ablation sweep
    print(f"\nRunning ablation at K in {ABLATION_LEVELS} ...")
    ablation_results = []
    for k in ABLATION_LEVELS:
        drop = set(m for m, _ in ranked[:k])
        result = cv_accuracy(records, drop)
        result["k"] = k
        result["dropped"] = sorted(drop)
        ablation_results.append(result)
        print(f"  K={k:3d}: acc={result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}  F1={result['mean_macro_f1']:.4f}")

    out = {
        "ablation_levels": ABLATION_LEVELS,
        "results": ablation_results,
        "top_motifs": [{"motif": m, "score": float(s)} for m, s in ranked[:50]],
        "vocab_size": len(ranked),
    }
    OUT_DAT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DAT}")

    # Figure
    df = pd.DataFrame(ablation_results)
    chart = (
        alt.Chart(df)
        .mark_line(point=True, color=BLUE, strokeWidth=2)
        .encode(
            x=alt.X("k:Q", title="Top-K motifs dropped"),
            y=alt.Y("mean_accuracy:Q", title="Probe accuracy", scale=alt.Scale(domain=[0, 1])),
        )
    )
    # Add chance reference line at 1/9 = 0.111
    chance_df = pd.DataFrame({"x": [0, max(ABLATION_LEVELS)], "y": [1/9, 1/9]})
    chance = (
        alt.Chart(chance_df)
        .mark_line(color=OLIVE, strokeDash=[4, 4], opacity=0.5)
        .encode(x="x:Q", y="y:Q")
    )
    final = (
        (chart + chance)
        .properties(
            width=400, height=280,
            title=alt.TitleParams(
                text="Probe accuracy under top-K motif ablation",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    final.save(str(OUT_FIG), scale_factor=2)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
