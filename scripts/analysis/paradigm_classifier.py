"""Binary paradigm classifier: extended-thinking vs base on SWE-agent,
applied to cross-scaffold submissions.

Different falsifier from the leave-one-scaffold-out probe:
  - leave-one-scaffold-out asks "does BACKBONE signature transfer across scaffolds?"
  - this script asks "does PARADIGM signature transfer across scaffolds?"

Train: binary classifier on the 6 SWE-agent backbones, with
  positive class = extended-thinking (Claude-3.7-thinking, Claude-4)
  negative class = base RLHF       (Claude-3, Claude-3.5, GPT-4, GPT-4o)

Test 1 (held-out within-SWE-agent backbone):
  Hold out one of the 6 SWE-agent backbones at a time, train on the other 5,
  predict held-out trajectories. Reports per-trajectory P(extended-thinking).

Test 2 (cross-scaffold transfer):
  Apply the full SWE-agent classifier to the 3 cross-scaffold submissions
  (DARS+R1, Moatless+V3, Agentless+Claude-3.5). Reports the predicted-paradigm
  distribution per submission. Specific question: does DARS+R1 (RL-reasoning)
  read as extended-thinking, base, or neither?

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
Writes:
    output/paper2_pilot/paradigm_classifier.json
    output/figures/fig_paradigm_classifier.png

Usage:
    uv run python scripts/analysis/paradigm_classifier.py
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN_D, MAGENTA, OLIVE, COPPER, BLUE
register()

SEQ = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
OUT_DAT = ROOT / "output" / "paper2_pilot"
OUT_FIG = ROOT / "output" / "figures"

EXTENDED_THINKING = {"Claude-3.7-thinking", "Claude-4"}
BASE_RLHF = {"Claude-3", "Claude-3.5", "GPT-4", "GPT-4o"}
SWE_AGENT_BACKBONES = EXTENDED_THINKING | BASE_RLHF
CROSS_SCAFFOLD = ["DARS+R1", "Moatless+V3", "Agentless+Claude-3.5"]


def load_records() -> list[dict]:
    with SEQ.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_classifier(X_text_train, y_train):
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
    X_train = vec.fit_transform(X_text_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return vec, clf


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    records = load_records()
    swe_records = [r for r in records if r["agent"] in SWE_AGENT_BACKBONES]
    swe_text = [" ".join(r["bpe"]) for r in swe_records]
    swe_y = [int(r["agent"] in EXTENDED_THINKING) for r in swe_records]

    print(f"SWE-agent training: {len(swe_records)} trajectories, "
          f"{sum(swe_y)} extended-thinking + {len(swe_y) - sum(swe_y)} base")

    # ===== Test 2 first (cross-scaffold transfer): full SWE-agent classifier =====
    vec, clf = fit_classifier(swe_text, swe_y)
    print(f"\n=== Cross-scaffold paradigm-transfer test ===")
    cross_results = {}
    for sub in CROSS_SCAFFOLD:
        sub_records = [r for r in records if r["agent"] == sub]
        if not sub_records:
            continue
        X_test = vec.transform(" ".join(r["bpe"]) for r in sub_records)
        probs = clf.predict_proba(X_test)[:, 1]  # P(extended-thinking)
        preds = (probs >= 0.5).astype(int)
        cross_results[sub] = {
            "n":              len(sub_records),
            "mean_p_thinking": float(np.mean(probs)),
            "median_p_thinking": float(np.median(probs)),
            "n_predicted_thinking": int(preds.sum()),
            "n_predicted_base":     int((1 - preds).sum()),
            "share_predicted_thinking": float(preds.mean()),
        }
        print(f"\n  {sub}  (n={len(sub_records)})")
        print(f"    mean P(extended-thinking) = {np.mean(probs):.3f}")
        print(f"    median P(extended-thinking) = {np.median(probs):.3f}")
        print(f"    share predicted extended-thinking = {preds.mean():.1%}")
        print(f"    -> reads as: "
              f"{'extended-thinking' if preds.mean() > 0.6 else ('base' if preds.mean() < 0.4 else 'mixed')}")

    # ===== Test 1: within-SWE-agent leave-one-backbone-out =====
    print(f"\n=== Within-SWE-agent leave-one-backbone-out (control) ===")
    holdout_results = {}
    for held in sorted(SWE_AGENT_BACKBONES):
        train_text = [t for t, r in zip(swe_text, swe_records) if r["agent"] != held]
        train_y    = [y for y, r in zip(swe_y, swe_records) if r["agent"] != held]
        test_text  = [t for t, r in zip(swe_text, swe_records) if r["agent"] == held]
        if not test_text:
            continue
        vec_h, clf_h = fit_classifier(train_text, train_y)
        X_test = vec_h.transform(test_text)
        probs = clf_h.predict_proba(X_test)[:, 1]
        true_label = int(held in EXTENDED_THINKING)
        accuracy = float(np.mean((probs >= 0.5) == bool(true_label)))
        holdout_results[held] = {
            "n":                 len(test_text),
            "true_label":        "extended-thinking" if true_label else "base",
            "mean_p_thinking":   float(np.mean(probs)),
            "accuracy":          accuracy,
            "share_correct":     accuracy,
        }
        tag = "extended-thinking" if true_label else "base"
        print(f"  hold out {held:24s} ({tag:18s}, n={len(test_text)})  "
              f"mean P(thinking)={np.mean(probs):.3f}  accuracy={accuracy:.1%}")

    # ===== Save =====
    payload = {
        "design": (
            "Binary logistic regression: P(extended-thinking | trajectory) on SWE-agent. "
            "Pos class = Claude-3.7-thinking + Claude-4. Neg class = Claude-3 + Claude-3.5 + GPT-4 + GPT-4o. "
            "Tests: (1) within-SWE-agent leave-one-backbone-out; (2) cross-scaffold paradigm transfer."
        ),
        "training_set": {
            "n":               len(swe_records),
            "n_extended":      sum(swe_y),
            "n_base":          len(swe_y) - sum(swe_y),
            "extended_class":  sorted(EXTENDED_THINKING),
            "base_class":      sorted(BASE_RLHF),
        },
        "leave_one_out_holdout": holdout_results,
        "cross_scaffold_transfer": cross_results,
        "interpretation": (
            "If paradigm signature transfers across scaffolds, cross-scaffold trajectories should be "
            "classified consistent with the agent's actual paradigm. RL-reasoning (DARS+R1) is the most "
            "interesting cell: does it read as extended-thinking (paradigm-similar to Claude-3.7/4) or as "
            "base (paradigm-different)? The probability gives a continuous answer."
        ),
    }
    out_json = OUT_DAT / "paradigm_classifier.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved {out_json}")

    # ===== Figure: P(extended-thinking) distribution per submission =====
    rows = []
    # Cross-scaffold
    for sub, info in cross_results.items():
        rows.append({"submission": sub, "mean_p": info["mean_p_thinking"],
                     "share_thinking": info["share_predicted_thinking"],
                     "category": "cross-scaffold"})
    # Within-scaffold leave-one-out
    for held, info in holdout_results.items():
        rows.append({"submission": held, "mean_p": info["mean_p_thinking"],
                     "share_thinking": info["share_predicted_thinking"]
                                          if "share_predicted_thinking" in info else 0,
                     "category": "within-SWE-agent " + info["true_label"]})

    df = pd.DataFrame(rows)
    sub_order = (sorted(BASE_RLHF) + sorted(EXTENDED_THINKING) + CROSS_SCAFFOLD)
    df = df[df["submission"].isin(sub_order)]
    df["sort_idx"] = df["submission"].map(lambda s: sub_order.index(s))
    df = df.sort_values("sort_idx")

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("mean_p:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Mean P(extended-thinking)",
                                  domain=False, ticks=False, format=".2f", labelFontSize=10)),
            y=alt.Y("submission:N", sort=sub_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=10, labelLimit=240)),
            color=alt.Color("category:N",
                            scale=alt.Scale(
                                domain=["within-SWE-agent base",
                                        "within-SWE-agent extended-thinking",
                                        "cross-scaffold"],
                                range=[BLUE, GREEN_D, COPPER]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["submission", "category", "mean_p"],
        )
        .properties(
            width=440, height=max(260, 28 * len(df)),
            title=alt.TitleParams(
                text="P(extended-thinking) per submission",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    out_png = OUT_FIG / "fig_paradigm_classifier.png"
    bars.save(str(out_png), scale_factor=2)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
