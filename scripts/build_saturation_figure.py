"""
Structural saturation curves across benchmarks.
70% of SWE-bench Lite tasks are structurally represented in ~16 tasks (d_edits, t=0.28).
Contrast with HumanEval and MBPP which saturate even faster.

Produces: notebooks/plots/saturation_coverage_curve.pdf + .png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "notebooks" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Wong colorblind-safe palette
COLORS = {
    "SWE-bench Lite": "#0072B2",
    "HumanEval":      "#E69F00",
    "MBPP":           "#009E73",
}

THRESHOLD = 0.28  # reproduces "16 tasks" claim for SWE-bench Lite

def saturation_curve(df: pd.DataFrame, dist_col: str, threshold: float):
    n = max(df["i"].max(), df["j"].max()) + 1
    mat = np.ones((n, n))
    for row in df.itertuples(index=False):
        v = getattr(row, dist_col)
        mat[row.i, row.j] = v
        mat[row.j, row.i] = v
    np.fill_diagonal(mat, 0.0)
    covered_by = {i: set(np.where(mat[i] < threshold)[0]) for i in range(n)}
    covered, remaining = set(), set(range(n))
    curve = []
    while remaining:
        best = max(remaining, key=lambda i: len(covered_by[i] - covered))
        covered |= covered_by[best] - covered
        curve.append(len(covered) / n)
        remaining.remove(best)
        if len(covered) >= n:
            break
    return np.array(curve), n

print("Loading distance matrices...")
df_swe = pd.read_parquet(ROOT / "output/datasets/swe_bench_lite_resolved/distances.parquet")
df_he  = pd.read_parquet(ROOT / "output/datasets/humaneval/distances.parquet")
df_mb  = pd.read_parquet(ROOT / "output/datasets/mbpp/distances.parquet")

# SWE-bench uses d_edits; HumanEval/MBPP also have d_edits
print("Computing saturation curves...")
swe_curve, swe_n = saturation_curve(df_swe, "d_edits", THRESHOLD)
he_curve,  he_n  = saturation_curve(df_he,  "d_edits", THRESHOLD)
mb_curve,  mb_n  = saturation_curve(df_mb,  "d_edits", THRESHOLD)

def knee70(curve):
    idx = np.argmax(curve >= 0.70)
    return int(idx) + 1 if curve[idx] >= 0.70 else len(curve)

swe_k = knee70(swe_curve)
he_k  = knee70(he_curve)
mb_k  = knee70(mb_curve)

print(f"SWE-bench Lite: n={swe_n}, 70% at rank {swe_k} ({100*swe_k/swe_n:.1f}%)")
print(f"HumanEval:      n={he_n},  70% at rank {he_k}  ({100*he_k/he_n:.1f}%)")
print(f"MBPP:           n={mb_n}, 70% at rank {mb_k}  ({100*mb_k/mb_n:.1f}%)")

# --- Plot ---
fig, ax = plt.subplots(figsize=(5.5, 3.8))

datasets = [
    ("SWE-bench Lite", swe_curve, swe_n, swe_k),
    ("HumanEval",      he_curve,  he_n,  he_k),
    ("MBPP",           mb_curve,  mb_n,  mb_k),
]

for label, curve, n, k in datasets:
    xs = np.arange(1, len(curve) + 1) / n * 100   # % of tasks on x-axis
    ys = curve * 100                                # % coverage on y-axis
    ax.plot(xs, ys, color=COLORS[label], lw=2, label=label)

    # Mark 70% knee
    kx = k / n * 100
    ax.axvline(kx, color=COLORS[label], lw=0.8, ls="--", alpha=0.6)
    ax.scatter([kx], [70], color=COLORS[label], s=40, zorder=5)
    offset = 1.5
    ax.text(kx + offset, 70 - 5, f"{k} tasks\n({kx:.0f}%)",
            color=COLORS[label], fontsize=7.5, va="top")

# 70% reference line
ax.axhline(70, color="gray", lw=0.8, ls=":", alpha=0.7)
ax.text(0.5, 71, "70% coverage", color="gray", fontsize=7.5, va="bottom")

ax.set_xlabel("Tasks selected (% of benchmark)", fontsize=10)
ax.set_ylabel("Structural coverage (%)", fontsize=10)
ax.set_title("Saturation of structural diversity\nacross benchmarks (AST-edit distance, $t=0.28$)", fontsize=10)
ax.set_xlim(0, 30)
ax.set_ylim(0, 105)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax.legend(fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out_png = OUT_DIR / "saturation_coverage_curve.png"
out_pdf = OUT_DIR / "saturation_coverage_curve.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out_png}")
print(f"Saved {out_pdf}")
