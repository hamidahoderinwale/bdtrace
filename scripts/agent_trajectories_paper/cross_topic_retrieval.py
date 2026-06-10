"""Decisive test: is procedural search useful for anything keyword search isn't?

The only scenario where procedural retrieval should win is CROSS-TOPIC behaviour
retrieval -- "find the same procedure applied in a different repo." Keyword
similarity tracks topic, so when the candidate pool is restricted to a DIFFERENT
repo than the query, keyword loses its main cue; procedural is topic-blind so it
should be unaffected.

For each query we rank only candidates from a different repo (self excluded) and
measure precision@k that the retrieved trajectory is the SAME AGENT (the external
behaviour label). Compared against the unrestricted retrieval, against keyword
(TF-IDF cosine), and against chance (P(same agent | different repo)).

Honest null: an agent's scaffold boilerplate is repo-independent, so keyword may
still recover the agent cross-repo -> procedural shows no advantage.

  .venv/bin/python cross_topic_retrieval.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")
COARSE = ["search_repo", "read_file", "edit", "create_file", "run_test", "submit", "other"]
CIX = {a: i for i, a in enumerate(COARSE)}
KS = [1, 5, 10]
RNG = np.random.RandomState(0)


def fp(atoms):
    v = np.zeros(len(COARSE))
    for a in atoms:
        v[CIX[a]] += 1
    s = v.sum()
    return v / s if s else v


def jsd_all(P, q):
    m = 0.5 * (P + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pq = np.where(P > 0, P * np.log2(P / m), 0.0).sum(1)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0).sum()
    return 0.5 * kl_pq + 0.5 * kl_qm


def boot(mat):
    mat = np.array(mat)
    mean = mat.mean(0)
    idx = RNG.randint(0, mat.shape[0], size=(1000, mat.shape[0]))
    s = mat[idx].mean(1)
    lo, hi = np.percentile(s, [5, 95], axis=0)
    return mean, lo, hi


def main():
    rows = [json.loads(l) for l in open(ROOT / "output/paper2_pilot/local_rawtext.jsonl")]
    rows = [r for r in rows if r.get("text")]
    n = len(rows)
    agent = [r["agent"] for r in rows]
    repo = [r["repo"] for r in rows]
    FP = np.stack([fp(r["atoms"]) for r in rows])
    tfidf = TfidfVectorizer(max_features=8000).fit_transform([r["text"] for r in rows])
    repo_arr = np.array(repo)

    # chance baselines over pairs
    def chance(restrict_diff_repo):
        same = tot = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if restrict_diff_repo and repo[i] == repo[j]:
                    continue
                tot += 1
                same += agent[i] == agent[j]
        return same / tot
    ch_all = sum(c * (c - 1) for c in Counter(agent).values()) / (n * (n - 1))
    ch_xr = chance(True)
    print(f"n={n}  chance same-agent (any pair)={ch_all:.3f}  chance same-agent | different-repo={ch_xr:.3f}\n")

    q = RNG.choice(n, size=min(300, n), replace=False)
    sims = cosine_similarity(tfidf[q], tfidf)

    results = {}
    for cond, diff_repo in (("unrestricted", False), ("cross-repo only", True)):
        for method in ("procedural", "keyword"):
            mat = []
            for r_, i in enumerate(q):
                if method == "procedural":
                    score = -jsd_all(FP, FP[i])      # higher = closer
                else:
                    score = sims[r_].copy()
                order = np.argsort(-score)
                cand = [j for j in order if j != i and (not diff_repo or repo[j] != repo[i])]
                hits = np.array([agent[j] == agent[i] for j in cand])
                mat.append([float(hits[:k].mean()) for k in KS])
            mean, lo, hi = boot(mat)
            results[(cond, method)] = (mean, lo, hi)
            p5 = mean[1]
            print(f"{cond:16s} {method:11s} p@5={p5:.3f} [{lo[1]:.3f},{hi[1]:.3f}]  "
                  f"(p@1={mean[0]:.3f}, p@10={mean[2]:.3f})")
        print()

    out = {f"{c}|{m}": {"k": KS, "precision": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
           for (c, m), (mean, lo, hi) in results.items()}
    out["chance"] = {"any_pair": ch_all, "cross_repo": ch_xr}
    (ROOT / "output/paper2_pilot/cross_topic_retrieval.json").write_text(json.dumps(out, indent=2))
    print("wrote cross_topic_retrieval.json")


if __name__ == "__main__":
    main()
