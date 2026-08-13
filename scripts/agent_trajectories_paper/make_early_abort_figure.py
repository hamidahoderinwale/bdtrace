"""(a) Cost-frontier figure for the early-abort probe — Tufte-minimal Altair.

x = compute saved (% of steps), y = resolves retained (%), one line per operating
prefix k in {8,10,20}, swept over a fine abort-threshold grid. Run from repo root.
"""
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from scripts.theme import register, GREEN, BLUE, MAGENTA

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
y = np.array([d['resolved'] for d in data])
groups = np.array([d['instance'] for d in data])
lengths = np.array([d['len'] for d in data])
total_steps = lengths.sum()
total_resolved = y.sum()


def oof_proba(k):
    docs = [" ".join(d['canonical'][:k]) for d in data]
    X = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), lowercase=False).fit_transform(docs)
    proba = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    return proba


curves = []
for k, col in ((8, GREEN), (10, BLUE), (20, MAGENTA)):
    proba = oof_proba(k)
    for tau in np.linspace(0.0, 0.6, 61):
        abort = (proba < tau) & (lengths > k)
        saved = (lengths[abort] - k).sum()
        kept = total_resolved - y[abort].sum()
        curves.append({'k': f'abort at step {k}', 'saved': 100 * saved / total_steps,
                       'kept': 100 * kept / total_resolved})

df = pd.DataFrame(curves)
chart = (
    alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X('saved:Q', title='Compute saved (% of steps)',
                scale=alt.Scale(domain=[0, 70]), axis=alt.Axis(domain=False, ticks=False)),
        y=alt.Y('kept:Q', title='Resolves retained (%)',
                scale=alt.Scale(domain=[20, 100]), axis=alt.Axis(domain=False, ticks=False)),
        color=alt.Color('k:N', title=None,
                        scale=alt.Scale(domain=['abort at step 8', 'abort at step 10', 'abort at step 20'],
                                        range=[GREEN, BLUE, MAGENTA])),
    ).properties(width=420, height=300, title='Early-abort: compute saved vs resolves retained')
    .configure_legend(orient='bottom')
)
out = 'docs/papers/figures/fig_early_abort_frontier.png'
chart.save(out, scale_factor=2)
print('wrote', out)
