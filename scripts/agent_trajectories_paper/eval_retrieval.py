"""Behavioral-retrieval eval: does procedural-fingerprint retrieval capture a
different axis than keyword (BM25/TF-IDF) retrieval? Non-circular ground truths:

  (1) Local 9-agent corpus: fingerprint k-NN should recover AGENT (an external
      provenance label = behavior class) far above base rate -> behavioral
      retrieval works. (No raw text here, so no BM25 -- agent is the point.)
  (2) OpenHands SWE-Zero (single agent, has raw text): TF-IDF retrieval should
      recover REPO (topic) well; fingerprint should NOT (~base) -> the two axes
      are orthogonal, i.e. keyword search cannot serve behavioral retrieval.

Run with the local venv (sklearn+numpy):
  .venv/bin/python eval_retrieval.py
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
COARSE = ["search_repo", "read_file", "edit", "create_file", "run_test", "submit", "other"]
CIX = {a: i for i, a in enumerate(COARSE)}
K = 5
RNG = np.random.RandomState(0)


def to_canon(a):
    if a.startswith("EDIT"): return "edit"
    if a.startswith("CREATE"): return "create_file"
    if a.startswith("RUN"): return "run_test"
    if a.startswith(("OPEN", "NAV")): return "read_file"
    if a.startswith(("SEARCH", "FIND")): return "search_repo"
    if "SUBMIT" in a: return "submit"
    return "other"


def repo_of(iid):
    return re.sub(r"-\d+$", "", iid)


def fp(atoms):
    v = np.zeros(len(COARSE))
    for a in atoms:
        v[CIX[a]] += 1
    s = v.sum()
    return v / s if s else v


def jsd(p, q):
    p = p / (p.sum() or 1); q = q / (q.sum() or 1)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def precision_at_k(dists, labels, qidx, k=K):
    order = np.argsort(dists)
    order = [j for j in order if j != qidx][:k]
    return np.mean([labels[j] == labels[qidx] for j in order])


def eval_corpus(items, label_key, text=None, sample=200):
    fps = [fp(it["atoms"]) for it in items]
    labels = [it[label_key] for it in items]
    n = len(items)
    base = sum(c * (c - 1) for c in Counter(labels).values()) / (n * (n - 1))  # P(same label) for a random pair
    q = RNG.choice(n, size=min(sample, n), replace=False)

    # fingerprint retrieval (JSD)
    fp_p = []
    for i in q:
        d = np.array([jsd(fps[i], fps[j]) for j in range(n)])
        fp_p.append(precision_at_k(d, labels, i))
    out = {"n": n, "label": label_key, "base_rate": round(base, 3),
           "fingerprint_p@k": round(float(np.mean(fp_p)), 3)}

    if text is not None:
        X = TfidfVectorizer(max_features=5000).fit_transform(text)
        sim = cosine_similarity(X[q], X)
        bm_p = []
        for r, i in enumerate(q):
            d = 1 - sim[r]
            bm_p.append(precision_at_k(d, labels, i))
        out["bm25_p@k"] = round(float(np.mean(bm_p)), 3)
    return out


def main():
    # local multi-agent corpus (SWE-bench Verified: agents AND repos both repeat)
    local = []
    for line in open(ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl"):
        r = json.loads(line)
        local.append({"atoms": [to_canon(a) for a in r["canonical"]],
                      "agent": r["agent"], "repo": repo_of(r["instance_id"])})
    print("=== behavioral fingerprint retrieval: behavior axis vs topic axis (local corpus) ===")
    print("recover AGENT (behavior class):")
    print(json.dumps(eval_corpus(local, "agent"), indent=2))
    print("recover REPO (topic) -- should be near base if retrieval is behavioral, not topical:")
    print(json.dumps(eval_corpus(local, "repo"), indent=2))


if __name__ == "__main__":
    main()
