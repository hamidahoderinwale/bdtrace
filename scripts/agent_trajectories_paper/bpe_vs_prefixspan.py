import sys, json
sys.path.insert(0, '.')
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from analysis.preferences.bpe import train_bpe, apply_bpe
from analysis.procedures.corpus_motifs import mine_corpus_patterns, encode_sequences

rows = [json.loads(l) for l in open('output/paper2_pilot/bpe_sequences_extended.jsonl')]
seqs = [r['canonical'] for r in rows]
labels = np.array([r['agent'] for r in rows])
n_clusters = len(set(labels))
print(f"n={len(seqs)} sequences, {n_clusters} agents")

def vmeasure_of(vectors, seed=0):
    X = np.asarray(vectors, dtype=float)
    # L1-normalize rows (frequency vectors); guard zero rows
    s = X.sum(1, keepdims=True); s[s==0]=1; X = X/s
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    return v_measure_score(labels, km.labels_)

def bpe_vectors(K):
    model = train_bpe(seqs, target_size=K, verbose=False)
    if isinstance(model, tuple): model = model[0]
    tok = apply_bpe(seqs, model)
    vocab = [t for t,_ in Counter(t for s in tok for t in s).most_common(K)]
    idx = {t:i for i,t in enumerate(vocab)}
    V=[]
    for s in tok:
        v=[0]*len(vocab)
        for t in s:
            if t in idx: v[idx[t]]+=1
        V.append(v)
    return V, len(vocab)

def prefixspan_vectors(K):
    # lower min_support until >=K patterns, then take top-K by support
    for ms in (0.20,0.15,0.10,0.08,0.05):
        pats = mine_corpus_patterns(seqs, min_support=ms, compress=True)
        if len(pats) >= K: break
    pats = pats[:K]
    return encode_sequences(seqs, pats, compress=True), len(pats)

for K in (64,128):
    bv,nb = bpe_vectors(K)
    print(f"BPE        K={K:3d} (vocab {nb}):  V-measure = {vmeasure_of(bv):.3f}")
pv,npat = prefixspan_vectors(64)
print(f"PrefixSpan K= 64 (patterns {npat}): V-measure = {vmeasure_of(pv):.3f}")
