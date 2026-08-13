"""Canonical-alphabet V-measure vs vocabulary size (grounds fig:vmeasure).

Sweeps BPE target size V; clusters trajectories on motif-frequency vectors
(KMeans, k = #agents); scores clusters against agent labels with V-measure.
Replaces the unsupported 0.290/0.410 caption; native half is dropped (no native
field). Run from repo root.
"""
import sys
import json
from collections import Counter
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import altair as alt
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from analysis.preferences.bpe import train_bpe, apply_bpe
from scripts.theme import register, BLUE

register()
rows = [json.loads(l) for l in open("output/paper2_pilot/bpe_sequences_extended.jsonl")]
seqs = [r["canonical"] for r in rows]
labels = np.array([r["agent"] for r in rows])
k = len(set(labels))


def vmeasure_of(vectors, seed=0):
    X = np.asarray(vectors, dtype=float)
    s = X.sum(1, keepdims=True); s[s == 0] = 1; X = X / s
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    return v_measure_score(labels, km.labels_)


def bpe_vectors(V):
    model = train_bpe(seqs, target_size=V, verbose=False)
    if isinstance(model, tuple):
        model = model[0]
    tok = apply_bpe(seqs, model)
    vocab = [t for t, _ in Counter(t for s in tok for t in s).most_common(V)]
    idx = {t: i for i, t in enumerate(vocab)}
    out = []
    for s in tok:
        v = [0] * len(vocab)
        for t in s:
            if t in idx:
                v[idx[t]] += 1
        out.append(v)
    return out, len(vocab)


rec = []
for V in (32, 64, 128, 256, 512):
    vec, nb = bpe_vectors(V)
    vm = vmeasure_of(vec)
    print(f"V={V:4d} (vocab {nb:4d}): V-measure={vm:.3f}", flush=True)
    rec.append({"V": V, "vmeasure": round(vm, 4), "vocab": nb})

df = pd.DataFrame(rec)
df.to_json("output/paper2_pilot/vmeasure_canonical_sweep.json", orient="records")
chart = (
    alt.Chart(df).mark_line(point=True, color=BLUE, strokeWidth=2).encode(
        x=alt.X("V:Q", title="Vocabulary size (V)", scale=alt.Scale(type="log", base=2),
                axis=alt.Axis(domain=False, ticks=False)),
        y=alt.Y("vmeasure:Q", title="V-measure vs agent labels",
                axis=alt.Axis(domain=False, ticks=False)),
    ).properties(width=380, height=240, title="Canonical-alphabet vocabulary stability")
)
chart.save("docs/papers/figures/fig_vmeasure_sweep.png", scale_factor=2)
print("wrote fig_vmeasure_sweep.png", flush=True)
