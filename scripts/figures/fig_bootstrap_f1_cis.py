"""Bootstrap F1 confidence intervals for the representation horse race.

Re-runs the kNN pass/fail evaluation from scripts.compare_representations on
the same five representations (edit_cert, fim_jaccard, staged_embed,
cot_embed, fix_type), but adds B=1000 bootstrap resamples on the instance
axis at k=5 to produce honest 95% CIs and pairwise difference CIs.

Reads:
    output/resolved_traces_lite_full.jsonl
    output/datasets/swe_bench_lite_resolved/fix_types.json
    output/leaderboard/lite_results.json
    output/staged_descriptions.json
    output/prompting_study/gpt_4o/records.json
    output/strategy_forms/frequent_itemsets.json

Writes:
    output/paper2_pilot/bootstrap_f1_cis.png
    docs/paper2_pilot/bootstrap_f1_cis.png      (mirror)
    output/paper2_pilot/bootstrap_f1_cis.json   (numeric results)

Usage:
    .venv/bin/python -m scripts.figures.fig_bootstrap_f1_cis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.theme import (
    register, GREEN, BLUE, MAGENTA, COPPER, OLIVE, MAGENTA_D,
)
register()

from scripts.compare_representations import (
    load_certs, pairwise_jaccard, fixtype_distance, embed_texts,
    cosine_dist_matrix,
)

OUT_DAT = ROOT / "output" / "paper2_pilot"
OUT_DOC = ROOT / "docs" / "paper2_pilot"
OUT_FIG = OUT_DAT / "bootstrap_f1_cis.png"
OUT_JSON = OUT_DAT / "bootstrap_f1_cis.json"
OUT_DOC_FIG = OUT_DOC / "bootstrap_f1_cis.png"

# Five representations in the same panel as the original figure.
# motif is excluded because it sits on a smaller instance set
# (common_mapped), and the figure compares apples-to-apples on the
# 275-instance shared support.
REP_ORDER = ["edit_cert", "fim_jaccard", "staged_embed", "cot_embed", "fix_type"]
REP_LABELS = {
    "edit_cert":    "Edit certificate (Jaccard)",
    "fim_jaccard":  "FIM pattern overlap",
    "staged_embed": "Staged narrative (embedding)",
    "cot_embed":    "Free-form plan (embedding)",
    "fix_type":     "Fix type (13 classes)",
}
REP_COLORS = {
    "edit_cert":    BLUE,
    "fim_jaccard":  GREEN,
    "staged_embed": COPPER,
    "cot_embed":    OLIVE,
    "fix_type":     MAGENTA,
}

K = 5
B = 1000
SEED = 42


def knn_f1_loo_indices(
    dist_matrix: np.ndarray, labels: np.ndarray, indices: np.ndarray, k: int
) -> float:
    """kNN F1 with leave-one-out, restricted to a subset of indices.

    For each query in indices, neighbors are also drawn from indices
    (resampled with replacement). Self is excluded.
    """
    sub_d = dist_matrix[np.ix_(indices, indices)]
    sub_y = labels[indices]
    n = len(indices)
    preds = np.empty(n, dtype=int)
    for i in range(n):
        row = sub_d[i].copy()
        row[i] = np.inf
        nbrs = np.argsort(row)[:k]
        preds[i] = 1 if sub_y[nbrs].mean() >= 0.5 else 0
    return f1_score(sub_y, preds, zero_division=0)


def build_distance_matrices() -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    print("Loading edit certificates...")
    certs = load_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")

    print("Loading fix types...")
    ft = json.load(open(ROOT / "output" / "datasets"
                        / "swe_bench_lite_resolved" / "fix_types.json"))
    type_map = {r["instance_id"]: r["fix_type"] for r in ft["results"]}

    print("Loading pass/fail labels (leaderboard)...")
    lb = json.load(open(ROOT / "output" / "leaderboard" / "lite_results.json"))
    AGENT_LONG = {
        "GPT-4":      "20240402_sweagent_gpt4",
        "Claude-3.5": "20240620_sweagent_claude3.5sonnet",
        "GPT-4o":     "20240728_sweagent_gpt4o",
    }
    pass_map: dict[str, int] = {}
    for inst in certs:
        anyp = False
        for _, long in AGENT_LONG.items():
            if lb.get(long, {}).get(inst, False):
                anyp = True
                break
        pass_map[inst] = int(anyp)

    print("Loading staged narratives...")
    sd = json.load(open(ROOT / "output" / "staged_descriptions.json"))
    staged_map = {r["instance_id"]: r["staged_narrative"] for r in sd["results"]}

    print("Loading CoT free-form plans...")
    rec = json.load(open(ROOT / "output" / "prompting_study" / "gpt_4o" / "records.json"))
    cot_map = {r["instance_id"]: r["conditions"]["no_context"].get("response", "")
               for r in rec}

    common = sorted(
        set(certs) & set(type_map) & set(staged_map)
        & {k for k, v in cot_map.items() if v}
    )
    print(f"  shared support n = {len(common)}")

    labels = np.array([pass_map.get(iid, 0) for iid in common], dtype=int)
    print(f"  pass: {labels.sum()}, fail: {(1-labels).sum()}")

    dist: dict[str, np.ndarray] = {}

    print("Building edit-cert Jaccard matrix...")
    dist["edit_cert"] = pairwise_jaccard(
        common, {iid: certs[iid] for iid in common}
    )

    print("Building FIM-pattern Jaccard matrix...")
    fim_data = json.load(open(ROOT / "output" / "strategy_forms" / "frequent_itemsets.json"))
    fim_patterns = [frozenset(p["itemset"]) for p in fim_data["patterns"]]
    fim_sets = {
        iid: frozenset(i for i, pat in enumerate(fim_patterns)
                       if pat.issubset(certs[iid]))
        for iid in common
    }
    dist["fim_jaccard"] = pairwise_jaccard(common, fim_sets)

    print("Building fix-type distance matrix...")
    dist["fix_type"] = fixtype_distance(common, type_map)

    print(f"Embedding staged narratives (n={len(common)})...")
    staged_emb = embed_texts([staged_map[iid] for iid in common])
    dist["staged_embed"] = cosine_dist_matrix(staged_emb)

    print(f"Embedding CoT plans (n={len(common)})...")
    cot_emb = embed_texts([cot_map[iid] for iid in common])
    dist["cot_embed"] = cosine_dist_matrix(cot_emb)

    return dist, labels, common


def bootstrap_cis(
    dist: dict[str, np.ndarray], labels: np.ndarray, B: int = 1000, k: int = 5,
) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(labels)
    indices_full = np.arange(n)

    point: dict[str, float] = {}
    for rep in REP_ORDER:
        point[rep] = float(knn_f1_loo_indices(dist[rep], labels, indices_full, k))
    print("\nPoint F1@k=5 (full sample):")
    for rep in REP_ORDER:
        print(f"  {REP_LABELS[rep]:35s}  {point[rep]:.4f}")

    # Bootstrap: same instance resample applied across all reps for paired diff
    boot = {rep: np.empty(B) for rep in REP_ORDER}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        for rep in REP_ORDER:
            boot[rep][b] = knn_f1_loo_indices(dist[rep], labels, idx, k)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{B}")

    per_rep: dict[str, dict] = {}
    for rep in REP_ORDER:
        arr = boot[rep]
        per_rep[rep] = {
            "point":   point[rep],
            "ci_low":  float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "mean":    float(arr.mean()),
        }

    # Paired differences: enumerate ordered pairs and rank by |point diff|
    pairs = []
    for i in range(len(REP_ORDER)):
        for j in range(i + 1, len(REP_ORDER)):
            a, b_ = REP_ORDER[i], REP_ORDER[j]
            diffs = boot[a] - boot[b_]
            point_diff = point[a] - point[b_]
            ci_lo = float(np.percentile(diffs, 2.5))
            ci_hi = float(np.percentile(diffs, 97.5))
            sig = (ci_lo > 0) or (ci_hi < 0)
            # orient so point_diff >= 0 for readability
            if point_diff < 0:
                a, b_ = b_, a
                point_diff = -point_diff
                ci_lo, ci_hi = -ci_hi, -ci_lo
            pairs.append({
                "left": a, "right": b_,
                "point_diff": point_diff,
                "ci_low": ci_lo, "ci_high": ci_hi,
                "significant": sig,
            })
    pairs.sort(key=lambda r: -r["point_diff"])

    return {"per_rep": per_rep, "pairs": pairs, "n": int(len(labels)), "B": B, "k": k}


def render(results: dict, n: int, out_path: Path) -> None:
    """Two-panel restyle. Theme palette, alternating row bands, no grid."""
    per_rep = results["per_rep"]
    pairs = results["pairs"]

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.32},
    )

    # ------------------------------------------------------------------
    # Panel A: per-representation F1 with CI bars, sorted by point estimate
    # ------------------------------------------------------------------
    ax = axes[0]
    sorted_reps = sorted(REP_ORDER, key=lambda r: per_rep[r]["point"])
    y = np.arange(len(sorted_reps))

    # Alternating row bands
    for i, _ in enumerate(sorted_reps):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F6F6F4", zorder=0)

    for i, rep in enumerate(sorted_reps):
        d = per_rep[rep]
        color = REP_COLORS[rep]
        ax.hlines(i, d["ci_low"], d["ci_high"],
                  color=color, linewidth=2.2, alpha=0.55, zorder=2)
        # Whisker caps
        for x in (d["ci_low"], d["ci_high"]):
            ax.vlines(x, i - 0.16, i + 0.16, color=color,
                      linewidth=1.6, alpha=0.55, zorder=2)
        ax.scatter([d["point"]], [i], s=58, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([REP_LABELS[r] for r in sorted_reps], fontsize=10)
    ax.set_ylim(-0.5, len(sorted_reps) - 0.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.65)
    ax.set_xlabel(f"F1 at k = {results['k']} (positive-class, 95% bootstrap CI)",
                  fontsize=10)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8, zorder=1)
    ax.set_title(
        f"Per-representation pass/fail prediction  (n = {n}, B = {results['B']})",
        fontsize=11, color="#111111", loc="left", pad=8,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
    ax.tick_params(colors="#444444")

    # ------------------------------------------------------------------
    # Panel B: paired differences, ranked by point estimate
    # ------------------------------------------------------------------
    ax = axes[1]
    y = np.arange(len(pairs))

    # Alternating row bands
    for i, _ in enumerate(pairs):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F6F6F4", zorder=0)

    for i, p in enumerate(pairs):
        sig_color = OLIVE if p["significant"] else MAGENTA_D
        ax.hlines(i, p["ci_low"], p["ci_high"],
                  color=sig_color, linewidth=2.0,
                  alpha=0.85 if p["significant"] else 0.45, zorder=2)
        for x in (p["ci_low"], p["ci_high"]):
            ax.vlines(x, i - 0.16, i + 0.16, color=sig_color,
                      linewidth=1.5,
                      alpha=0.85 if p["significant"] else 0.45, zorder=2)
        ax.scatter([p["point_diff"]], [i], s=48, color=sig_color,
                   edgecolor="white", linewidth=1.0, zorder=3)

    ax.set_yticks(y)
    short = {
        "edit_cert": "edit cert", "fim_jaccard": "FIM",
        "staged_embed": "staged emb.", "cot_embed": "free-form emb.",
        "fix_type": "fix type",
    }
    ax.set_yticklabels(
        [f"{short[p['left']]}  −  {short[p['right']]}" for p in pairs],
        fontsize=10,
    )
    ax.set_ylim(-0.5, len(pairs) - 0.5)
    ax.invert_yaxis()
    ax.axvline(0, color="#666666", linewidth=0.8, zorder=1)
    ax.set_xlim(-0.35, 0.50)
    ax.set_xlabel("F1 difference (point estimate, 95% bootstrap CI)", fontsize=10)
    ax.set_title(
        "Paired differences  (CI crosses zero ⇒ not significant, crimson)",
        fontsize=11, color="#111111", loc="left", pad=8,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
    ax.tick_params(colors="#444444")

    fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    OUT_DAT.mkdir(parents=True, exist_ok=True)
    OUT_DOC.mkdir(parents=True, exist_ok=True)

    cache = OUT_DAT / "_bootstrap_f1_cache.npz"
    if cache.exists():
        print(f"Loading cached distance matrices from {cache}")
        z = np.load(cache, allow_pickle=True)
        dist = {rep: z[rep] for rep in REP_ORDER}
        labels = z["labels"]
        common = list(z["common"])
    else:
        dist, labels, common = build_distance_matrices()
        np.savez_compressed(
            cache,
            **dist, labels=labels, common=np.array(common, dtype=object),
        )
        print(f"Cached distance matrices to {cache}")

    results = bootstrap_cis(dist, labels, B=B, k=K)

    OUT_JSON.write_text(json.dumps({
        "n":     int(len(labels)),
        "B":     B,
        "k":     K,
        "per_representation": results["per_rep"],
        "paired_differences": results["pairs"],
    }, indent=2))
    print(f"Saved {OUT_JSON}")

    render(results, n=int(len(labels)), out_path=OUT_FIG)
    # Mirror into docs/
    import shutil
    shutil.copyfile(OUT_FIG, OUT_DOC_FIG)
    print(f"Mirrored to {OUT_DOC_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
