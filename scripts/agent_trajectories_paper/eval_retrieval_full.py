"""Behavioral-retrieval double dissociation, full methodology.

Protocol
--------
Corpus: the 9-agent SWE-bench-Verified trajectory set, restricted to instances
whose raw .traj text is available (so fingerprint and keyword retrieval are
compared on an identical document set).

Query set: a fixed random sample (seed 0) of Q=300 trajectories. Each query
retrieves its neighbours from the rest of the corpus (self excluded).

Retrieval methods:
  * fingerprint -- rank by Jensen-Shannon distance between L1-normalized
    coarse-atom distributions (the procedural fingerprint).
  * bm25        -- rank by cosine similarity of TF-IDF vectors over the raw
    serialized .traj text (a keyword index over the trace).
  * chance      -- expected precision of a random ranking = the label base rate
    P(two random trajectories share the label).

Ground-truth labels (both EXTERNAL to the atoms -> non-circular):
  * agent -- model+scaffold provenance (the behaviour class).
  * repo  -- repository (the topic).

Metric: precision@k for k in 1..20, averaged over queries; 90% bootstrap CIs
(1000 resamples over the query set). The dissociation predicts fingerprint >>
chance and >> bm25 on AGENT, while bm25 >> chance and >> fingerprint on REPO.

  .venv/bin/python eval_retrieval_full.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.insert(0, ".")
from scripts.theme import register, BLUE, COPPER

register()
ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
COARSE = ["search_repo", "read_file", "edit", "create_file", "run_test", "submit", "other"]
CIX = {a: i for i, a in enumerate(COARSE)}
KS = list(range(1, 21))
Q = 300
RNG = np.random.RandomState(0)


def fp(atoms):
    v = np.zeros(len(COARSE))
    for a in atoms:
        v[CIX[a]] += 1
    s = v.sum()
    return v / s if s else v


def jsd_all(P, q):
    # JSD (base 2) of every row of P against q; P,q are L1-normalized
    m = 0.5 * (P + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pq = np.where(P > 0, P * np.log2(P / m), 0.0).sum(1)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0).sum()
    return 0.5 * kl_pq + 0.5 * kl_qm


def base_rate(labels):
    from collections import Counter
    n = len(labels)
    return sum(c * (c - 1) for c in Counter(labels).values()) / (n * (n - 1))


def per_query_prec(order_excl_self, labels, qi):
    same = np.array([labels[j] == labels[qi] for j in order_excl_self])
    return [float(same[:k].mean()) for k in KS]


def boot_ci(mat):  # mat: [n_queries, len(KS)] -> mean, lo, hi per k
    mean = mat.mean(0)
    idx = RNG.randint(0, mat.shape[0], size=(1000, mat.shape[0]))
    samples = mat[idx].mean(1)  # [1000, len(KS)]
    lo, hi = np.percentile(samples, [5, 95], axis=0)
    return mean, lo, hi


def main():
    rows = [json.loads(l) for l in open(ROOT / "output/paper2_pilot/local_rawtext.jsonl")]
    rows = [r for r in rows if r.get("text")]            # identical doc set for both methods
    n = len(rows)
    print(f"corpus with text: {n}")
    FP = np.stack([fp(r["atoms"]) for r in rows])
    tfidf = TfidfVectorizer(max_features=8000).fit_transform([r["text"] for r in rows])
    qidx = RNG.choice(n, size=min(Q, n), replace=False)

    recs = []
    for label in ("agent", "repo"):
        labels = [r[label] for r in rows]
        br = base_rate(labels)
        fp_mat, bm_mat = [], []
        sims = cosine_similarity(tfidf[qidx], tfidf)
        for r_, i in enumerate(qidx):
            fo = np.argsort(jsd_all(FP, FP[i]))
            fo = [j for j in fo if j != i]
            fp_mat.append(per_query_prec(fo, labels, i))
            bo = np.argsort(-sims[r_])
            bo = [j for j in bo if j != i]
            bm_mat.append(per_query_prec(bo, labels, i))
        for name, mat in (("fingerprint", np.array(fp_mat)), ("bm25", np.array(bm_mat))):
            mean, lo, hi = boot_ci(mat)
            for ki, k in enumerate(KS):
                recs.append({"label": label, "method": name, "k": k,
                             "precision": mean[ki], "lo": lo[ki], "hi": hi[ki]})
        recs.append({"label": label, "method": "chance", "k": 1, "precision": br, "lo": br, "hi": br})
        recs.append({"label": label, "method": "chance", "k": 20, "precision": br, "lo": br, "hi": br})
        print(f"{label}: base={br:.3f}  fp@5={np.array(fp_mat).mean(0)[4]:.3f}  bm25@5={np.array(bm_mat).mean(0)[4]:.3f}")

    df = pd.DataFrame(recs)
    df.to_json(ROOT / "output/paper2_pilot/retrieval_pk.json", orient="records")
    cs = alt.Scale(domain=["fingerprint", "bm25", "chance"], range=[BLUE, COPPER, "#999999"])
    base = alt.Chart(df)
    band = base.transform_filter("datum.method != 'chance'").mark_area(opacity=0.15).encode(
        x="k:Q", y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 1])), y2="hi:Q",
        color=alt.Color("method:N", scale=cs, legend=None))
    lines = base.transform_filter("datum.method != 'chance'").mark_line(point=True).encode(
        x=alt.X("k:Q", title="k", axis=alt.Axis(domain=False, ticks=False)),
        y=alt.Y("precision:Q", title="precision@k", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(domain=False, ticks=False)),
        color=alt.Color("method:N", scale=cs, legend=alt.Legend(title=None)))
    chance = base.transform_filter("datum.method == 'chance'").mark_line(
        strokeDash=[4, 4], color="#999999").encode(x="k:Q", y="precision:Q")
    chart = alt.layer(band, lines, chance).properties(width=300, height=230).facet(
        column=alt.Column("label:N", title=None, sort=["agent", "repo"],
                          header=alt.Header(labelFontSize=12)))
    out = ROOT / "docs/papers/figures/fig_retrieval_pk.png"
    chart.save(str(out), scale_factor=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
