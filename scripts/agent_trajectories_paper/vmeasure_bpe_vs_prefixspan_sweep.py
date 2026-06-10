"""Merged V-measure sweep: BPE vs PrefixSpan over vocabulary size (one method).

Replaces the standalone 3-bar fig_bpe_vs_prefixspan and reconciles it with the
V-measure sweep: a single figure, V-measure (y) vs vocabulary/pattern size (x),
two lines (BPE, PrefixSpan), 5-seed mean +/- sd. Same computation as
vmeasure_sweep_v2 / bpe_vs_prefixspan (KMeans into #agents clusters, v_measure
vs agent labels, L1-normalized frequency vectors). Backs up the old figure.
Run from repo root.
"""
import sys, json, shutil
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, altair as alt
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
sys.path.insert(0, ".")
from analysis.preferences.bpe import train_bpe, apply_bpe
from analysis.procedures.corpus_motifs import mine_corpus_patterns, encode_sequences
from scripts.theme import register, BLUE, COPPER

register()
rows = [json.loads(l) for l in open("output/paper2_pilot/bpe_sequences_extended.jsonl")]
seqs = [r["canonical"] for r in rows]
labels = np.array([r["agent"] for r in rows])
k = len(set(labels))
GRID = [16, 24, 32, 48, 64, 96, 128, 192, 256]
SEEDS = [0, 1, 2, 3, 4]

def vmeas(vectors):
    X = np.asarray(vectors, dtype=float)
    s = X.sum(1, keepdims=True); s[s == 0] = 1; X = X / s
    out = []
    for seed in SEEDS:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        out.append(v_measure_score(labels, km.labels_))
    a = np.array(out); return float(a.mean()), float(a.std())

def bpe_vectors(V):
    model = train_bpe(seqs, target_size=V, verbose=False)
    if isinstance(model, tuple): model = model[0]
    tok = apply_bpe(seqs, model)
    vocab = [t for t, _ in Counter(t for s in tok for t in s).most_common(V)]
    idx = {t: i for i, t in enumerate(vocab)}
    out = [[0] * len(vocab) for _ in tok]
    for j, s in enumerate(tok):
        for t in s:
            if t in idx: out[j][idx[t]] += 1
    return out, len(vocab)

# mine PrefixSpan ONCE at low support -> pattern pool sorted by support; slice top-K
pool = None
for ms in (0.03, 0.05, 0.08, 0.10):
    pool = mine_corpus_patterns(seqs, min_support=ms, compress=True)
    if len(pool) >= max(GRID): break
print(f"PrefixSpan pool: {len(pool)} patterns (min_support reached)")

recs = []
for V in GRID:
    bv, nb = bpe_vectors(V)
    m, sd = vmeas(bv)
    recs.append({"method": "BPE", "K": V, "size": nb, "mean": round(m, 4), "sd": round(sd, 4)})
    print(f"BPE        K={V:4d} (vocab {nb:4d}): {m:.3f} +/- {sd:.3f}", flush=True)
    if len(pool) >= V:
        pv = encode_sequences(seqs, pool[:V], compress=True)
        m2, sd2 = vmeas(pv)
        recs.append({"method": "PrefixSpan", "K": V, "size": V, "mean": round(m2, 4), "sd": round(sd2, 4)})
        print(f"PrefixSpan K={V:4d} (pats  {V:4d}): {m2:.3f} +/- {sd2:.3f}", flush=True)

Path("output/paper2_pilot/vmeasure_bpe_vs_prefixspan_sweep.json").write_text(json.dumps(recs, indent=2))
df = pd.DataFrame(recs)
df["lo"] = df["mean"] - df["sd"]; df["hi"] = df["mean"] + df["sd"]
xs = alt.Scale(type="log", base=2, nice=False)
cs = alt.Scale(domain=["BPE", "PrefixSpan"], range=[BLUE, COPPER])
band = alt.Chart(df).mark_area(opacity=0.15).encode(
    x=alt.X("size:Q", scale=xs, title="Vocabulary size", axis=alt.Axis(domain=False, ticks=False, values=GRID)),
    y=alt.Y("lo:Q", scale=alt.Scale(domain=[0.3, 0.75]), title="V-measure vs agent labels",
            axis=alt.Axis(domain=False, ticks=False)),
    y2="hi:Q", color=alt.Color("method:N", scale=cs, legend=None))
line = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X("size:Q", scale=xs), y="mean:Q",
    color=alt.Color("method:N", scale=cs, legend=None))
# direct labels at the right end of each line (clearer than a legend)
ends = df[df.K == max(GRID)].copy()
labtext = alt.Chart(ends).mark_text(align="left", dx=8, fontSize=12, fontWeight="bold").encode(
    x=alt.X("size:Q", scale=xs), y="mean:Q", text="method:N",
    color=alt.Color("method:N", scale=cs, legend=None))
chart = (band + line + labtext).properties(width=440, height=240,
    title="V-measure against agent labels by vocabulary size")
OUT = Path("docs/papers/figures/fig_bpe_vs_prefixspan.png")
# preserve the ORIGINAL bar figure once; don't clobber it on re-runs
if OUT.exists() and not OUT.with_name("fig_bpe_vs_prefixspan_prev_bars.png").exists():
    shutil.copy(OUT, OUT.with_name("fig_bpe_vs_prefixspan_prev_bars.png"))
chart.save(str(OUT), scale_factor=2)
print("wrote fig_bpe_vs_prefixspan.png (backed up old bars -> _prev_bars)", flush=True)
