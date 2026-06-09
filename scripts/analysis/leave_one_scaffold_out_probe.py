"""Leave-one-scaffold-out probe: does backbone signature transfer across scaffolds?

Trains a 6-class backbone classifier on the 6 SWE-agent backbone trajectories,
then applies it to the 3 cross-scaffold held-out submissions. Tests whether
"the same backbone" remains procedurally recognizable when the scaffold
changes — the load-bearing predictive-holdout test for the scaffold-dominance
headline.

Plus a within-scaffold leave-one-backbone-out control: hold out each of the 6
SWE-agent backbones in turn, train on 5, predict the held-out's labels. This
gives a same-scaffold backbone-recognition baseline for comparison.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
Writes:
    output/paper2_pilot/leave_one_scaffold_out_probe.json
    output/figures/fig_leave_one_scaffold_out.png

Usage:
    uv run python scripts/analysis/leave_one_scaffold_out_probe.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, MAGENTA, GREEN, COPPER, OLIVE
register()

SEQ = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
OUT_DAT = ROOT / "output" / "paper2_pilot"
OUT_FIG = ROOT / "output" / "figures"

SWE_AGENT_BACKBONES = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking", "Claude-4", "GPT-4", "GPT-4o",
]
CROSS_SCAFFOLD = {
    "Agentless+Claude-3.5":  "Claude-3.5",   # same backbone in different scaffold
    "DARS+R1":               None,           # no SWE-agent twin (R1 not on SWE-agent)
    "Moatless+V3":           None,           # no SWE-agent twin (V3 not on SWE-agent)
}


def load_records() -> list[dict]:
    with SEQ.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_predict(X_text_train, y_train, X_text_test):
    """Train TF-IDF + LR; return predictions + per-class probabilities."""
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
    X_train = vec.fit_transform(X_text_train)
    X_test  = vec.transform(X_text_test)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_enc)
    pred_idx = clf.predict(X_test)
    pred_labels = le.inverse_transform(pred_idx)
    pred_probs = clf.predict_proba(X_test)
    return pred_labels, pred_probs, le.classes_


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    records = load_records()
    swe_records   = [r for r in records if r["agent"] in SWE_AGENT_BACKBONES]
    cross_records = {sub: [r for r in records if r["agent"] == sub] for sub in CROSS_SCAFFOLD}

    swe_text = [" ".join(r["bpe"]) for r in swe_records]
    swe_y    = [r["agent"] for r in swe_records]

    # ===== Cross-scaffold transfer =====
    print("=== Cross-scaffold backbone transfer ===")
    transfer_results = {}
    cross_text_all = []
    cross_label_all = []
    for sub, recs in cross_records.items():
        if not recs:
            continue
        cross_text_all.extend(" ".join(r["bpe"]) for r in recs)
        cross_label_all.extend([sub] * len(recs))

    pred_labels, pred_probs, classes_ = fit_predict(swe_text, swe_y, cross_text_all)

    idx = 0
    for sub, recs in cross_records.items():
        if not recs:
            continue
        n = len(recs)
        sub_preds = pred_labels[idx:idx + n]
        sub_probs = pred_probs[idx:idx + n]
        idx += n
        true_backbone = CROSS_SCAFFOLD[sub]
        cnt = Counter(sub_preds)
        if true_backbone is not None:
            n_correct = sum(1 for p in sub_preds if p == true_backbone)
            accuracy = n_correct / n
        else:
            n_correct = None
            accuracy = None
        # Mean predicted-class confidence
        mean_max_prob = float(np.mean(np.max(sub_probs, axis=1)))
        # Entropy of the predicted-class distribution (in nats; high = scattered, low = concentrated)
        cls_freq = np.array([cnt.get(c, 0) for c in classes_]) / n
        cls_freq = cls_freq[cls_freq > 0]
        pred_entropy = float(-(cls_freq * np.log2(cls_freq)).sum())
        transfer_results[sub] = {
            "n": n,
            "true_backbone": true_backbone,
            "predicted_class_distribution": dict(cnt),
            "accuracy_vs_true_backbone": accuracy,
            "mean_max_prob": mean_max_prob,
            "predicted_entropy_bits": pred_entropy,
        }
        print(f"\n  {sub}  (n={n}, true_backbone={true_backbone})")
        for cls, c in cnt.most_common():
            print(f"    -> predicted as {cls:24s}  {c:4d}  ({c/n:5.1%})")
        if accuracy is not None:
            print(f"    accuracy (true backbone match): {accuracy:.1%}")
        print(f"    mean max-class probability: {mean_max_prob:.3f}")
        print(f"    predicted-class entropy:    {pred_entropy:.2f} bits")

    # ===== Within-SWE-agent leave-one-backbone-out =====
    print("\n=== Within-SWE-agent leave-one-backbone-out (control) ===")
    within_results = {}
    for held_out in SWE_AGENT_BACKBONES:
        train_text = [t for t, y in zip(swe_text, swe_y) if y != held_out]
        train_y    = [y for y in swe_y if y != held_out]
        test_text  = [t for t, y in zip(swe_text, swe_y) if y == held_out]
        if not test_text:
            continue
        pred_labels, pred_probs, classes_ = fit_predict(train_text, train_y, test_text)
        n = len(test_text)
        cnt = Counter(pred_labels)
        # When held-out is held out, no class match is possible by definition; report distribution.
        mean_max_prob = float(np.mean(np.max(pred_probs, axis=1)))
        cls_freq = np.array([cnt.get(c, 0) for c in classes_]) / n
        cls_freq = cls_freq[cls_freq > 0]
        pred_entropy = float(-(cls_freq * np.log2(cls_freq)).sum())
        within_results[held_out] = {
            "n": n,
            "predicted_class_distribution": dict(cnt),
            "mean_max_prob": mean_max_prob,
            "predicted_entropy_bits": pred_entropy,
        }
        print(f"  hold out {held_out:24s}  (n={n})")
        for cls, c in cnt.most_common(3):
            print(f"    -> as {cls:24s}  {c:4d}  ({c/n:5.1%})")

    # ===== Save JSON =====
    payload = {
        "design": (
            "6-class TF-IDF + L2 logistic regression, trained on SWE-agent backbone trajectories. "
            "Transfer test: classifier applied to cross-scaffold submissions. "
            "Within-scaffold control: leave-one-backbone-out within SWE-agent."
        ),
        "training_set": {
            "n_classes": len(SWE_AGENT_BACKBONES),
            "classes":   SWE_AGENT_BACKBONES,
            "n_records": sum(1 for r in swe_records),
        },
        "cross_scaffold_transfer": transfer_results,
        "within_scaffold_holdout": within_results,
        "headline": {
            "agentless_to_claude35_accuracy": transfer_results.get("Agentless+Claude-3.5", {}).get("accuracy_vs_true_backbone"),
            "agentless_predicted_concentration": transfer_results.get("Agentless+Claude-3.5", {}).get("predicted_entropy_bits"),
        },
        "interpretation": (
            "If backbone signature transfers across scaffolds, Agentless+Claude-3.5 should be predicted "
            "as Claude-3.5 most of the time (high accuracy). If scaffold dominates, the prediction either "
            "concentrates on a non-Claude-3.5 class (systematic mis-mapping) or scatters (high entropy). "
            "Compare cross-scaffold transfer accuracy to within-scaffold leave-one-out as the baseline."
        ),
    }
    out_json = OUT_DAT / "leave_one_scaffold_out_probe.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved {out_json}")

    # ===== Figure: predicted-class distribution per held-out submission =====
    rows = []
    for sub, info in transfer_results.items():
        n = info["n"]
        for cls, c in info["predicted_class_distribution"].items():
            rows.append({
                "submission": sub,
                "predicted_class": cls,
                "share": c / n,
                "count": c,
            })
    df = pd.DataFrame(rows)
    sub_order = list(transfer_results.keys())
    cls_order = SWE_AGENT_BACKBONES

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("submission:N", sort=sub_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False, labelFontSize=10, labelLimit=240)),
            x=alt.X("share:Q",
                    stack="normalize",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Predicted class share",
                                  domain=False, ticks=False, format=".0%", labelFontSize=10)),
            color=alt.Color("predicted_class:N", sort=cls_order,
                            scale=alt.Scale(domain=cls_order,
                                            range=[COPPER, GREEN, "#187860", "#0d4a3a",
                                                   BLUE, MAGENTA]),
                            legend=alt.Legend(orient="bottom", title=None, columns=3)),
            order=alt.Order("predicted_class:N", sort="ascending"),
            tooltip=["submission", "predicted_class", "count", "share"],
        )
        .properties(
            width=440, height=180,
            title=alt.TitleParams(
                text="Cross-scaffold backbone-transfer: SWE-agent classifier predictions on held-out scaffolds",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    out_png = OUT_FIG / "fig_leave_one_scaffold_out.png"
    chart.save(str(out_png), scale_factor=2)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
