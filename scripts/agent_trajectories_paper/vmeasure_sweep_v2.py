"""Canonical V-measure vs vocabulary size — denser log sweep with seed band.

Improves on vmeasure_sweep.py (5 points, single seed, line-interpolated peak):
  - denser log grid so the plateau edge and degradation are actually sampled
  - multiple KMeans seeds per V -> mean + uncertainty band (init variance)
  - x-axis = REALIZED vocab size (BPE may not reach target)
  - shaded V=64-256 band (the range used throughout the paper)
Backs up the old 5-point figure first; does not fabricate. Run from repo root.
"""
import sys
import json
import shutil
from collections import Counter
from pathlib import Path

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

GRID = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
SEEDS = [0, 1, 2, 3, 4]


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
    X = np.asarray(out, dtype=float)
    sm = X.sum(1, keepdims=True); sm[sm == 0] = 1
    return X / sm, len(vocab)


runs = []
agg = []
for V in GRID:
    X, nb = bpe_vectors(V)
    vms = []
    for seed in SEEDS:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        vm = v_measure_score(labels, km.labels_)
        vms.append(vm)
        runs.append({"V": V, "vocab": nb, "seed": seed, "vmeasure": round(vm, 4)})
    vms = np.array(vms)
    agg.append({"V": V, "vocab": nb, "mean": float(vms.mean()), "sd": float(vms.std()),
                "lo": float(vms.mean() - vms.std()), "hi": float(vms.mean() + vms.std())})
    print(f"V={V:4d} (vocab {nb:4d}): V-measure {vms.mean():.3f} +/- {vms.std():.3f}", flush=True)

Path("output/paper2_pilot/vmeasure_canonical_sweep_v2.json").write_text(
    json.dumps({"runs": runs, "agg": agg}, indent=2))
df = pd.DataFrame(agg)

# realized-vocab x of the V=64 and V=256 targets -> shaded used range
used = df[df.V.isin([64, 256])]["vocab"].tolist()
band_df = pd.DataFrame({"x0": [min(used)], "x1": [max(used)]})

xscale = alt.Scale(type="log", base=2, nice=False)
shade = alt.Chart(band_df).mark_rect(opacity=0.07, color=BLUE).encode(
    x=alt.X("x0:Q", scale=xscale), x2="x1:Q")
area = alt.Chart(df).mark_area(opacity=0.18, color=BLUE).encode(
    x=alt.X("vocab:Q", scale=xscale, title="Realized vocabulary size",
            axis=alt.Axis(domain=False, ticks=False, values=GRID)),
    y=alt.Y("lo:Q", title="V-measure vs agent labels",
            scale=alt.Scale(domain=[0.3, 0.75]), axis=alt.Axis(domain=False, ticks=False)),
    y2="hi:Q")
line = alt.Chart(df).mark_line(point=True, color=BLUE, strokeWidth=2).encode(
    x=alt.X("vocab:Q", scale=xscale), y="mean:Q")
chart = (shade + area + line).properties(
    width=400, height=240, title="Canonical-alphabet vocabulary stability")

OUT = Path("docs/papers/figures/fig_vmeasure_sweep.png")
if OUT.exists():
    shutil.copy(OUT, OUT.with_name("fig_vmeasure_sweep_prev_5pt.png"))
chart.save(str(OUT), scale_factor=2)
print("wrote fig_vmeasure_sweep.png (backed up old -> fig_vmeasure_sweep_prev_5pt.png)", flush=True)
print(df.round(3).to_string(index=False), flush=True)
