"""Can the BEST keyword/text baseline recover a behavioural predicate from the
raw trace? Baseline = TF-IDF over the full serialized .traj + L2 logistic
regression, 5-fold cross-validated ROC-AUC (the strongest a bag-of-words method
can do). Procgrep computes each predicate exactly from the action sequence
(AUC 1.0 by construction).

Predicates (ground truth from the canonical action sequence):
  lexical control : "ran a test"          -- the command text is present, so a
                    text model should recover it.
  structural      : "edit-streak >= 5"    -- a count + contiguity fact
                    "tested before first edit"  -- an ordering fact
                    "searched before first edit"
                  -- order/contiguity/counts have no bag-of-words correlate, so
                     text recovery should collapse toward chance (AUC 0.5).

  .venv/bin/python eval_predicate_recovery.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
rows = [json.loads(l) for l in open(ROOT / "output/paper2_pilot/local_rawtext.jsonl")]
rows = [r for r in rows if r.get("text")]
texts = [r["text"] for r in rows]
atoms = [r["atoms"] for r in rows]


def first(seq, a):
    return seq.index(a) if a in seq else 10**9


def streak(seq, a, k):
    c = 0
    for x in seq:
        c = c + 1 if x == a else 0
        if c >= k:
            return True
    return False


preds = {
    "ran a test (lexical)": [("run_test" in s) for s in atoms],
    "edit-streak >=5 (structural)": [streak(s, "edit", 5) for s in atoms],
    "tested before first edit (order)": [(first(s, "run_test") < first(s, "edit")) and ("run_test" in s) for s in atoms],
    "searched before first edit (order)": [(first(s, "search_repo") < first(s, "edit")) and ("search_repo" in s) for s in atoms],
}

X = TfidfVectorizer(max_features=8000).fit_transform(texts)
out = {}
for name, y in preds.items():
    y = np.array(y, int)
    pos = float(y.mean())
    if min(pos, 1 - pos) < 0.03:
        out[name] = {"pos_rate": round(pos, 3), "text_auc": None, "note": "too imbalanced"}
        continue
    auc = float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc").mean())
    out[name] = {"pos_rate": round(pos, 3), "text_auc": round(auc, 3)}
    print(f"{name:38s} pos={pos:.2f}  best-text AUC={auc:.3f}")

(ROOT / "output/paper2_pilot/predicate_recovery.json").write_text(json.dumps(out, indent=2))
print("\nwrote predicate_recovery.json (procgrep computes each exactly: AUC 1.0)")
