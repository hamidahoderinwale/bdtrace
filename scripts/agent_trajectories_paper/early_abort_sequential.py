"""Sequential (any-step) early-abort policy — the principled, non-arbitrary version.

Instead of a hand-picked decision step k, re-score the prefix-so-far at every step
t and abort the first time P(resolve) < tau. Sweeps tau only. Per-step probes are
out-of-fold (GroupKFold by instance) so every score is leakage-clean. Overlays the
resulting frontier against the best fixed-k (step 8) to show the gain.

Run from repo root. Writes docs/papers/figures/fig_early_abort_frontier.png
(backs up the previous fixed-k version to *_kfixed.png).
"""
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from scripts.theme import register, GREEN, OLIVE

register()

rows = [json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
pf = json.load(open('output/paper2_pilot/extended_pass_fail.json'))
res = {k: set(v.get('resolved', [])) for k, v in pf.items()}

data = []
for r in rows:
    s = res.get(r['submission'])
    if s is None:
        continue
    data.append({'canonical': r['canonical'],
                 'len': r.get('canonical_length', len(r['canonical'])),
                 'resolved': int(r['instance_id'] in s),
                 'instance': r['instance_id']})
n = len(data)
y = np.array([d['resolved'] for d in data])
groups = np.array([d['instance'] for d in data])
lengths = np.array([d['len'] for d in data])
total_steps = lengths.sum()
total_resolved = y.sum()

T_GRID = list(range(3, 41))   # decision steps considered
print(f"n={n}  base resolve={y.mean():.3f}  scoring prefixes t={T_GRID[0]}..{T_GRID[-1]}")


def oof_proba_at(t):
    docs = [" ".join(d['canonical'][:t]) for d in data]
    X = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False).fit_transform(docs)
    proba = np.zeros(n)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    return proba


CACHE = Path('/tmp/ea_seq_frontier.json')
if CACHE.exists():
    df = pd.read_json(CACHE)
    print("loaded cached frontier (delete /tmp/ea_seq_frontier.json to recompute)")
else:
    # proba[i, j] = leakage-clean P(resolve) for trajectory i on its first T_GRID[j] actions
    P = np.column_stack([oof_proba_at(t) for t in T_GRID])
    print("per-step probes done")

    def seq_frontier(tau):
        """Abort each trajectory at the first decision step t<len where P<tau."""
        saved = 0
        lost = 0
        for i in range(n):
            Li = lengths[i]
            for j, t in enumerate(T_GRID):
                if t >= Li:
                    break                   # trajectory already finished; never aborted
                if P[i, j] < tau:
                    saved += Li - t         # cut the remaining steps
                    lost += y[i]            # a resolve only if it would have resolved
                    break
        kept = total_resolved - lost
        return 100 * saved / total_steps, 100 * kept / total_resolved

    def fixed_frontier(tau, k=8):
        proba_k = P[:, T_GRID.index(k)]
        abort = (proba_k < tau) & (lengths > k)
        saved = (lengths[abort] - k).sum()
        kept = total_resolved - y[abort].sum()
        return 100 * saved / total_steps, 100 * kept / total_resolved

    taus = np.linspace(0.0, 0.6, 61)
    recs = []
    print("\n=== sequential (any-step) frontier, selected thresholds ===")
    print(f"{'tau':>5}{'saved%':>9}{'kept%':>9}")
    for tau in taus:
        s, k = seq_frontier(tau)
        recs.append({'policy': 'Sequential (any step)', 'saved': s, 'kept': k})
        if round(tau, 2) in (0.10, 0.15, 0.20, 0.30):
            print(f"{tau:>5.2f}{s:>9.1f}{k:>9.1f}")
    for tau in taus:
        s, k = fixed_frontier(tau)
        recs.append({'policy': 'Fixed (step 8)', 'saved': s, 'kept': k})
    df = pd.DataFrame(recs)
    df.to_json(CACHE, orient='records')

# crop to the operating region (drop the uninteresting low-retention tail)
df = df[df['kept'] >= 70]
chart = (
    alt.Chart(df).mark_line(strokeWidth=2, clip=True).encode(
        x=alt.X('saved:Q', title='Compute saved (% of steps)',
                scale=alt.Scale(domain=[0, 44]), axis=alt.Axis(domain=False, ticks=False)),
        y=alt.Y('kept:Q', title='Resolves retained (%)',
                scale=alt.Scale(domain=[70, 100]), axis=alt.Axis(domain=False, ticks=False)),
        color=alt.Color('policy:N', title=None,
                        scale=alt.Scale(domain=['Sequential (any step)', 'Fixed (step 8)'],
                                        range=[GREEN, OLIVE])),
    ).properties(width=420, height=300, title='Early-abort frontier: sequential vs fixed-step')
    .configure_legend(orient='bottom')
)
out = Path('docs/papers/figures/fig_early_abort_frontier.png')
if out.exists():
    shutil.copy(out, out.with_name('fig_early_abort_frontier_kfixed.png'))
chart.save(str(out), scale_factor=2)
print(f"\nwrote {out} (backed up old fixed-k version -> fig_early_abort_frontier_kfixed.png)")
