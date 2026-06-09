"""Paired permutation null + baseline check for representation usefulness.

Reuses the cached distance matrices from fig_bootstrap_f1_cis (no
re-embedding needed). For each subsample (m = 200 out of n = 275,
drawn WITHOUT replacement to avoid duplicate-leakage in kNN), computes:
  - F1 with the real pass/fail labels (signal)
  - F1 with shuffled labels (per-resample null floor)
  - paired difference (signal - null) — the statistic of interest
Also reports stratified-random and majority-class baselines for
context.

Why subsampling, not with-replacement bootstrap: with replacement,
~37% of resample positions are duplicates of other positions. Each
duplicate's nearest neighbor is its own copy (distance 0), and the
kNN trivially predicts the duplicate's true label. This artifact
inflates F1 by ~30-40 percentage points equally in the signal and
the null run, swamping the comparison. Subsampling without
replacement eliminates duplicates entirely.

The question this answers: does each representation beat its own
noise floor significantly? Stricter than the bootstrap_f1_cis
pairwise test because it controls for the kNN-on-imbalanced-data
inherent F1 inflation.

Reads:
    output/paper2_pilot/_bootstrap_f1_cache.npz

Writes:
    output/paper2_pilot/representation_usefulness.json
    output/paper2_pilot/fig_representation_usefulness.png
    docs/paper2_pilot/fig_representation_usefulness.png

Usage:
    .venv/bin/python -m scripts.figures.fig_representation_usefulness
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.theme import register, GREEN, BLUE, MAGENTA, COPPER, OLIVE, MAGENTA_D
register()

CACHE = ROOT / "output" / "paper2_pilot" / "_bootstrap_f1_cache.npz"
OUT_DAT = ROOT / "output" / "paper2_pilot"
OUT_DOC = ROOT / "docs" / "paper2_pilot"
OUT_JSON = OUT_DAT / "representation_usefulness.json"
OUT_FIG = OUT_DAT / "fig_representation_usefulness.png"
OUT_DOC_FIG = OUT_DOC / "fig_representation_usefulness.png"

REP_ORDER = ["fim_jaccard", "cot_embed", "edit_cert", "fix_type", "staged_embed"]
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
SUBSAMPLE_FRAC = 0.75   # m = 0.75 × n; without replacement; no duplicates
SEED = 42


def knn_f1_loo_indices(d: np.ndarray, y: np.ndarray, idx: np.ndarray, k: int) -> float:
    sub_d = d[np.ix_(idx, idx)]
    sub_y = y[idx]
    n = len(idx)
    preds = np.empty(n, dtype=int)
    for i in range(n):
        row = sub_d[i].copy()
        row[i] = np.inf
        nbrs = np.argsort(row)[:k]
        preds[i] = 1 if sub_y[nbrs].mean() >= 0.5 else 0
    return f1_score(sub_y, preds, zero_division=0)


def paired_signal_vs_null(
    dist: dict[str, np.ndarray], labels: np.ndarray, B: int, k: int,
) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(labels)
    m = int(round(SUBSAMPLE_FRAC * n))
    print(f"  Subsampling without replacement: m = {m} of n = {n}")

    signal_full = {rep: knn_f1_loo_indices(dist[rep], labels, np.arange(n), k)
                   for rep in REP_ORDER}

    # Per-resample paired signal and null (subsample-without-replacement)
    sig = {rep: np.empty(B) for rep in REP_ORDER}
    null = {rep: np.empty(B) for rep in REP_ORDER}

    for b in range(B):
        idx = rng.choice(n, size=m, replace=False)   # no duplicates
        perm = rng.permutation(n)                     # shuffle full labels
        y_shuf = labels[perm]
        for rep in REP_ORDER:
            sig[rep][b] = knn_f1_loo_indices(dist[rep], labels, idx, k)
            null[rep][b] = knn_f1_loo_indices(dist[rep], y_shuf, idx, k)
        if (b + 1) % 100 == 0:
            print(f"  subsample {b+1}/{B}")

    out = {}
    for rep in REP_ORDER:
        diff = sig[rep] - null[rep]
        out[rep] = {
            "signal_point":        float(signal_full[rep]),
            "signal_mean":         float(sig[rep].mean()),
            "signal_ci":           [float(np.percentile(sig[rep], 2.5)),
                                    float(np.percentile(sig[rep], 97.5))],
            "null_mean":           float(null[rep].mean()),
            "null_ci":             [float(np.percentile(null[rep], 2.5)),
                                    float(np.percentile(null[rep], 97.5))],
            "diff_mean":           float(diff.mean()),
            "diff_ci":             [float(np.percentile(diff, 2.5)),
                                    float(np.percentile(diff, 97.5))],
            "p_diff_gt_zero":      float((diff > 0).mean()),
            "useful":              bool(np.percentile(diff, 2.5) > 0),
        }
    return out


def naive_baselines(labels: np.ndarray, B: int) -> dict:
    rng = np.random.default_rng(SEED + 1)
    n = len(labels)
    p_pos = labels.mean()
    y_true = labels

    rand_f1, maj_f1 = [], []
    for _ in range(B):
        y_rand = rng.choice([0, 1], size=n, p=[1 - p_pos, p_pos])
        rand_f1.append(f1_score(y_true, y_rand, average="binary",
                                pos_label=1, zero_division=0))
        maj_f1.append(0.0)  # majority class is "fail" → 0 by definition
    return {
        "pass_rate":           float(p_pos),
        "random_mean":         float(np.mean(rand_f1)),
        "random_ci":           [float(np.percentile(rand_f1, 2.5)),
                                float(np.percentile(rand_f1, 97.5))],
        "majority_class_f1":   0.0,
    }


def render(result: dict, baselines: dict, out_path: Path, n: int, B: int) -> None:
    sig_null = result
    fig, ax = plt.subplots(figsize=(12, 5.4))
    sorted_reps = sorted(REP_ORDER, key=lambda r: sig_null[r]["diff_mean"])
    y = np.arange(len(sorted_reps))

    # Alternating row bands
    for i, _ in enumerate(sorted_reps):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F6F6F4", zorder=0)

    # Vertical zero line + baselines
    ax.axvline(0, color="#666666", linewidth=0.8, zorder=1)
    rand_offset = baselines["random_mean"] - 0.0  # informational only
    ax.axvline(rand_offset, color=MAGENTA_D, linewidth=0.8,
               linestyle="--", alpha=0.5, zorder=1,
               label=f"Stratified-random F1 ≈ {baselines['random_mean']:.2f}")

    for i, rep in enumerate(sorted_reps):
        d = sig_null[rep]
        color = REP_COLORS[rep]
        useful = d["useful"]
        alpha = 0.95 if useful else 0.45

        # Signal CI (solid)
        ax.hlines(i + 0.18, d["signal_ci"][0], d["signal_ci"][1],
                  color=color, linewidth=2.4, alpha=alpha, zorder=2)
        ax.scatter([d["signal_point"]], [i + 0.18], s=60, color=color,
                   edgecolor="white", linewidth=1.0, zorder=3,
                   label="signal F1 (point + CI)" if i == 0 else None)

        # Null CI (lighter, lower)
        ax.hlines(i - 0.18, d["null_ci"][0], d["null_ci"][1],
                  color=OLIVE, linewidth=2.0, alpha=0.5, zorder=2)
        ax.scatter([d["null_mean"]], [i - 0.18], s=42, color=OLIVE,
                   edgecolor="white", linewidth=1.0, alpha=0.7, zorder=3,
                   label="permuted-label null F1 (CI)" if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels([REP_LABELS[r] for r in sorted_reps], fontsize=10)
    ax.set_ylim(-0.6, len(sorted_reps) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, max(0.7,
                           max(d["signal_ci"][1] for d in sig_null.values()) + 0.05))
    ax.set_xlabel(
        f"Positive-class F1 at k = {K}  (paired bootstrap, n = {n}, B = {B})",
        fontsize=10,
    )
    ax.set_title(
        "Does each representation beat its own permuted-label noise floor?",
        fontsize=11, color="#111111", loc="left", pad=8,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
    ax.tick_params(colors="#444444")

    fig.subplots_adjust(left=0.22, right=0.97, top=0.90, bottom=0.13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    if not CACHE.exists():
        print(f"ERROR: cache missing at {CACHE}. Run fig_bootstrap_f1_cis first.")
        return 1
    z = np.load(CACHE, allow_pickle=True)
    dist = {rep: z[rep] for rep in REP_ORDER}
    labels = z["labels"]
    n = len(labels)
    print(f"Loaded cache: n={n}, pass={int(labels.sum())}, fail={int((1-labels).sum())}")

    baselines = naive_baselines(labels, B=B)
    print(f"\nBaselines: pass_rate={baselines['pass_rate']:.3f}, "
          f"stratified-random F1 = {baselines['random_mean']:.4f} "
          f"[{baselines['random_ci'][0]:.4f}, {baselines['random_ci'][1]:.4f}]")
    print(f"           majority-class positive-F1 = 0.0000")

    print(f"\nRunning paired signal-vs-null bootstrap (B={B})...")
    result = paired_signal_vs_null(dist, labels, B=B, k=K)

    print("\nPer-representation signal vs permuted-label null:")
    for rep in REP_ORDER:
        d = result[rep]
        marker = "USEFUL" if d["useful"] else " null "
        print(f"  [{marker}] {REP_LABELS[rep]:35s}  "
              f"signal={d['signal_point']:.3f}  null_mean={d['null_mean']:.3f}  "
              f"diff={d['diff_mean']:+.3f}  CI=[{d['diff_ci'][0]:+.3f}, "
              f"{d['diff_ci'][1]:+.3f}]  P(diff>0)={d['p_diff_gt_zero']:.3f}")

    OUT_JSON.write_text(json.dumps({
        "n": int(n), "B": B, "k": K,
        "baselines":              baselines,
        "per_representation":     result,
    }, indent=2))
    print(f"\nSaved {OUT_JSON}")

    render(result, baselines, OUT_FIG, n=n, B=B)
    import shutil
    OUT_DOC.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_FIG, OUT_DOC_FIG)
    print(f"Mirrored to {OUT_DOC_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
