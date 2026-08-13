"""N-gram (n=3) baseline for the BPE-motif fingerprinting pipeline.

Reviewer-defensive robustness: if the three-regime JSD structure (Agentless ~1.0
/ within-SWE-agent / cross-scaffold) appears under fixed-length n-grams over the
76 canonical atoms, the fingerprint is not BPE-specific.

For each trajectory: extract overlapping atom-level 3-grams. Per agent: build a
probability distribution over the 3-gram vocabulary. Compute pairwise JSD and
run a backbone probe (TF-IDF + multinomial logistic regression, GroupKFold by
instance_id).

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/jsd_matrix_extended.json   (BPE motif baseline for comparison)
Writes:
    output/paper2_pilot/ngram_baseline.json
    output/figures/fig_ngram_jsd_scatter.png
    output/figures/fig_ngram_jsd_matrix.png
"""
from __future__ import annotations
import json, sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, MAGENTA, OLIVE
register()

SEQ = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
BPE_JSD = ROOT / "output" / "paper2_pilot" / "jsd_matrix_extended.json"
OUT_JSON = ROOT / "output" / "paper2_pilot" / "ngram_baseline.json"
FIG_OUT = ROOT / "output" / "figures"

N = 3  # n-gram size

AGENT_ORDER_EXT = [
    "Claude-3", "Claude-3.5", "Claude-3.7-thinking", "GPT-4", "GPT-4o",
    "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
]

SWE_AGENTS = {"Claude-3", "Claude-3.5", "Claude-3.7-thinking", "GPT-4", "GPT-4o"}


def pair_category(a: str, b: str) -> str:
    if "Agentless" in a or "Agentless" in b:
        return "Agentless vs other"
    if a in SWE_AGENTS and b in SWE_AGENTS:
        return "Within SWE-agent"
    return "Cross-scaffold"


def load_records() -> list[dict]:
    rows = []
    with SEQ.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.append({
                    "agent":       r["agent"],
                    "instance_id": r["instance_id"],
                    "canonical":   r["canonical"],
                })
    return rows


def trajectory_to_ngrams(seq: list[str], n: int = N) -> list[str]:
    if len(seq) < n:
        return []
    return ["+".join(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def per_agent_distribution(records: list[dict], vocab: list[str]) -> dict[str, np.ndarray]:
    counters: dict[str, Counter] = {}
    for r in records:
        ng = trajectory_to_ngrams(r["canonical"])
        counters.setdefault(r["agent"], Counter()).update(ng)
    out = {}
    for agent, c in counters.items():
        total = sum(c[v] for v in vocab)
        if total == 0:
            out[agent] = np.zeros(len(vocab))
        else:
            out[agent] = np.array([c.get(v, 0) / total for v in vocab])
    return out


def squared_jsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(jensenshannon(p, q, base=2)) ** 2


def compute_jsd_matrix(records: list[dict]) -> tuple[list[str], np.ndarray, list[str]]:
    full_vocab: Counter = Counter()
    for r in records:
        full_vocab.update(trajectory_to_ngrams(r["canonical"]))
    vocab = sorted(full_vocab.keys())
    dist = per_agent_distribution(records, vocab)
    agents = [a for a in AGENT_ORDER_EXT if a in dist]
    mat = np.zeros((len(agents), len(agents)))
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i == j:
                continue
            mat[i, j] = squared_jsd(dist[a], dist[b])
    return agents, mat, vocab


def backbone_probe(records: list[dict]) -> dict:
    docs, labels, groups = [], [], []
    for r in records:
        ng = trajectory_to_ngrams(r["canonical"])
        if not ng:
            continue
        docs.append(" ".join(ng))
        labels.append(r["agent"])
        groups.append(r["instance_id"])

    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, max_features=5000)
    X = vec.fit_transform(docs)
    y = np.array(labels)
    g = np.array(groups)

    gkf = GroupKFold(n_splits=5)
    f1s_macro, f1s_micro = [], []
    per_class_f1: dict[str, list[float]] = {a: [] for a in AGENT_ORDER_EXT}
    for tr, te in gkf.split(X, y, groups=g):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        f1s_macro.append(f1_score(y[te], pred, average="macro", zero_division=0))
        f1s_micro.append(f1_score(y[te], pred, average="micro", zero_division=0))
        for a in AGENT_ORDER_EXT:
            mask = (y[te] == a)
            if mask.sum() == 0:
                continue
            per_class_f1[a].append(
                f1_score(y[te] == a, pred == a, zero_division=0)
            )
    return {
        "macro_f1_mean":  round(float(np.mean(f1s_macro)), 4),
        "macro_f1_std":   round(float(np.std(f1s_macro)), 4),
        "micro_f1_mean":  round(float(np.mean(f1s_micro)), 4),
        "per_class_f1":   {a: round(float(np.mean(v)), 4) for a, v in per_class_f1.items() if v},
        "n_features":     int(X.shape[1]),
        "n_samples":      int(X.shape[0]),
    }


def load_bpe_jsd_pairs() -> dict[tuple[str, str], float]:
    d = json.load(BPE_JSD.open())
    agents = d["agents"]
    mat = d["matrix_array"]
    out = {}
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i < j:
                out[(a, b)] = mat[i][j]
    return out


def plot_jsd_scatter(agents: list[str], ngram_mat: np.ndarray,
                     bpe_pairs: dict, out_path: Path) -> None:
    rows = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i >= j:
                continue
            bpe_key = (a, b) if (a, b) in bpe_pairs else (b, a)
            if bpe_key not in bpe_pairs:
                continue
            rows.append({
                "pair":     f"{a} / {b}",
                "category": pair_category(a, b),
                "ngram":    float(ngram_mat[i, j]),
                "bpe":      float(bpe_pairs[bpe_key]),
            })
    df = pd.DataFrame(rows)

    cscale = alt.Scale(
        domain=["Within SWE-agent", "Cross-scaffold", "Agentless vs other"],
        range=[GREEN, BLUE, MAGENTA],
    )

    chart = (
        alt.Chart(df)
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("bpe:Q",
                    title="JSD (BPE motifs, V=200)",
                    scale=alt.Scale(domain=[0, 1.05])),
            y=alt.Y("ngram:Q",
                    title="JSD (3-grams over canonical atoms)",
                    scale=alt.Scale(domain=[0, 1.05])),
            color=alt.Color("category:N", scale=cscale,
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["pair", "category", "bpe", "ngram"],
        )
        .properties(
            width=380, height=380,
            title=alt.TitleParams(
                "Pair-similarity rank stable across BPE and 3-gram tokenization",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def plot_ngram_matrix(agents: list[str], mat: np.ndarray, out_path: Path) -> None:
    rows = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            rows.append({"row": a, "col": b, "jsd": float(mat[i, j])})
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("col:N", sort=agents,
                    axis=alt.Axis(title=None, labelAngle=-35, labelFontSize=9)),
            y=alt.Y("row:N", sort=agents,
                    axis=alt.Axis(title=None, labelFontSize=9)),
            color=alt.Color("jsd:Q",
                            scale=alt.Scale(scheme="blues", domain=[0, 1]),
                            legend=alt.Legend(title="JSD")),
            tooltip=["row", "col", "jsd"],
        )
        .properties(
            width=380, height=380,
            title=alt.TitleParams(
                "Pairwise agent JSD on 3-grams over canonical atoms",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    records = load_records()
    print(f"Loaded {len(records)} trajectories.")

    print(f"\nComputing pairwise JSD on {N}-grams over canonical atoms...")
    agents, mat, vocab = compute_jsd_matrix(records)
    print(f"  vocabulary size: {len(vocab)} distinct {N}-grams")
    print(f"  agents: {agents}")

    bpe_pairs = load_bpe_jsd_pairs()
    spearman_data = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i >= j:
                continue
            bpe_key = (a, b) if (a, b) in bpe_pairs else (b, a)
            if bpe_key in bpe_pairs:
                spearman_data.append((bpe_pairs[bpe_key], float(mat[i, j])))
    if spearman_data:
        from scipy.stats import spearmanr, pearsonr
        bpe_vals, ng_vals = zip(*spearman_data)
        rho, p_rho = spearmanr(bpe_vals, ng_vals)
        r, p_r = pearsonr(bpe_vals, ng_vals)
        print(f"\n  Spearman ρ between BPE-pair JSD and {N}-gram-pair JSD: {rho:.3f} (p={p_rho:.2g})")
        print(f"  Pearson r:                                              {r:.3f} (p={p_r:.2g})")

    print(f"\nRunning backbone probe on {N}-grams...")
    probe = backbone_probe(records)
    print(f"  macro F1: {probe['macro_f1_mean']:.3f} ± {probe['macro_f1_std']:.3f}")
    print(f"  micro F1: {probe['micro_f1_mean']:.3f}")
    print(f"  per-class F1: {probe['per_class_f1']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n":            N,
        "n_records":    len(records),
        "agents":       agents,
        "vocab_size":   len(vocab),
        "matrix_array": mat.tolist(),
        "spearman_vs_bpe": {"rho": round(rho, 4), "p": float(p_rho)} if spearman_data else None,
        "pearson_vs_bpe":  {"r":   round(r,   4), "p": float(p_r)}   if spearman_data else None,
        "backbone_probe": probe,
    }, indent=2))
    print(f"\nSaved {OUT_JSON}")

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plot_jsd_scatter(agents, mat, bpe_pairs, FIG_OUT / "fig_ngram_jsd_scatter.png")
    plot_ngram_matrix(agents, mat, FIG_OUT / "fig_ngram_jsd_matrix.png")
    print(f"Saved {FIG_OUT / 'fig_ngram_jsd_scatter.png'}")
    print(f"Saved {FIG_OUT / 'fig_ngram_jsd_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
