"""② Early-abort / prefix-k failure prediction (the cost case study).

From the first k canonical actions of a trajectory, predict resolve/fail; then
sweep an abort threshold and report the compute-saved vs resolves-retained
frontier. Leakage-controlled (GroupKFold by instance_id), same as the attribution
probe. Local, no GPU. Run from repo root.

Honest confounds (printed): failure correlates with trajectory length, and at
small k the prefix rarely contains outcome-laden actions (SUBMIT/EXIT_ERROR).
"""
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

rows = [json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
pf = json.load(open('output/paper2_pilot/extended_pass_fail.json'))
res = {k: set(v.get('resolved', [])) for k, v in pf.items()}

# keep trajectories with a known label; record full length (compute proxy)
data = []
for r in rows:
    s = res.get(r['submission'])
    if s is None:
        continue
    data.append({
        'canonical': r['canonical'],
        'len': r.get('canonical_length', len(r['canonical'])),
        'resolved': int(r['instance_id'] in s),
        'instance': r['instance_id'],
    })
y = np.array([d['resolved'] for d in data])
groups = np.array([d['instance'] for d in data])
lengths = np.array([d['len'] for d in data])
base = y.mean()
print(f"n={len(data)}  base resolve rate={base:.3f}  (chance AUC=0.50)")
print(f"length: resolved median={np.median(lengths[y==1]):.0f}  unresolved median={np.median(lengths[y==0]):.0f}  "
      "(failure–length correlation caveat)\n")


def prefix_docs(k):
    # first-k actions as a space-joined doc of unigrams (bigrams added by vectorizer)
    return [" ".join(d['canonical'][:k]) for d in data]


def oof_proba(X, y, groups, n_splits=5):
    """Out-of-fold P(resolve) under GroupKFold."""
    proba = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    return proba


print("=== failure-prediction AUC by prefix length k ===")
best = None
for k in (3, 5, 8, 10, 15, 20):
    vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False)
    X = vec.fit_transform(prefix_docs(k))
    proba = oof_proba(X, y, groups)
    auc = roc_auc_score(y, proba)
    print(f"  k={k:2d}   AUC={auc:.3f}")
    if best is None or auc > best[1]:
        best = (k, auc, proba)

total_steps = lengths.sum()
total_resolved = y.sum()


def frontier(k_op):
    """Build the abort frontier at operating prefix k_op (leakage-clean: decisions
    fire only on trajectories still running at step k_op, len>k_op)."""
    vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False)
    X = vec.fit_transform(prefix_docs(k_op))
    proba = oof_proba(X, y, groups)
    auc = roc_auc_score(y, proba)
    still_running = (lengths > k_op).sum()
    print(f"\n=== cost frontier: abort if P(resolve) < tau at step k={k_op} "
          f"(AUC={auc:.3f}; {still_running} of {len(y)} trajectories still running at step {k_op}) ===")
    print(f"{'tau':>5}{'aborted':>9}{'steps_saved%':>14}{'resolves_kept%':>16}{'resolves_lost':>15}")
    for tau in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        abort = (proba < tau) & (lengths > k_op)
        steps_saved = (lengths[abort] - k_op).sum()
        resolves_lost = y[abort].sum()
        kept = total_resolved - resolves_lost
        print(f"{tau:>5.2f}{abort.sum():>9}{100*steps_saved/total_steps:>13.1f}%"
              f"{100*kept/total_resolved:>15.1f}%{resolves_lost:>15}")


for k_op in (8, 10, 20):
    frontier(k_op)
