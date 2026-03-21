#!/usr/bin/env python3
"""
Altair version of the structural saturation curves figure.

Produces: notebooks/plots/saturation_coverage_curve_altair.png
"""
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "notebooks" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "SWE-bench Lite": "#0072B2",
    "HumanEval":      "#E69F00",
    "MBPP":           "#009E73",
}
THRESHOLD = 0.28


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


def knee70(curve):
    idx = np.argmax(curve >= 0.70)
    return int(idx) + 1 if curve[idx] >= 0.70 else len(curve)


print("Loading distance matrices...")
df_swe = pd.read_parquet(ROOT / "output/datasets/swe_bench_lite_resolved/distances.parquet")
df_he  = pd.read_parquet(ROOT / "output/datasets/humaneval/distances.parquet")
df_mb  = pd.read_parquet(ROOT / "output/datasets/mbpp/distances.parquet")

print("Computing saturation curves...")
swe_curve, swe_n = saturation_curve(df_swe, "d_edits", THRESHOLD)
he_curve,  he_n  = saturation_curve(df_he,  "d_edits", THRESHOLD)
mb_curve,  mb_n  = saturation_curve(df_mb,  "d_edits", THRESHOLD)

swe_k = knee70(swe_curve)
he_k  = knee70(he_curve)
mb_k  = knee70(mb_curve)

print(f"SWE-bench Lite: n={swe_n}, 70% at rank {swe_k} ({100*swe_k/swe_n:.1f}%)")
print(f"HumanEval:      n={he_n},  70% at rank {he_k}  ({100*he_k/he_n:.1f}%)")
print(f"MBPP:           n={mb_n}, 70% at rank {mb_k}  ({100*mb_k/mb_n:.1f}%)")

# --- Build tidy DataFrames ---
line_rows = []
for label, curve, n in [
    ("SWE-bench Lite", swe_curve, swe_n),
    ("HumanEval",      he_curve,  he_n),
    ("MBPP",           mb_curve,  mb_n),
]:
    for i, y in enumerate(curve):
        x_pct = (i + 1) / n * 100
        if x_pct <= 30:
            line_rows.append({"benchmark": label, "x": x_pct, "y": y * 100})

df_lines = pd.DataFrame(line_rows)

knee_rows = []
for label, curve, n, k in [
    ("SWE-bench Lite", swe_curve, swe_n, swe_k),
    ("HumanEval",      he_curve,  he_n,  he_k),
    ("MBPP",           mb_curve,  mb_n,  mb_k),
]:
    kx = k / n * 100
    knee_rows.append({
        "benchmark": label,
        "x": kx,
        "y": 70.0,
        "label": f"{k} tasks ({kx:.0f}%)",
    })

df_knees = pd.DataFrame(knee_rows)

# --- Altair chart ---
color_scale = alt.Scale(
    domain=list(COLORS.keys()),
    range=list(COLORS.values()),
)

lines = alt.Chart(df_lines).mark_line(strokeWidth=2).encode(
    x=alt.X("x:Q",
            title="Tasks selected (% of benchmark)",
            axis=alt.Axis(labelExpr="datum.value + '%'"),
            scale=alt.Scale(domain=[0, 30])),
    y=alt.Y("y:Q",
            title="Structural coverage (%)",
            axis=alt.Axis(labelExpr="datum.value + '%'"),
            scale=alt.Scale(domain=[0, 105])),
    color=alt.Color("benchmark:N", scale=color_scale, title=""),
)

hline = alt.Chart(pd.DataFrame([{"y": 70}])).mark_rule(
    strokeDash=[4, 4], color="gray", opacity=0.6, strokeWidth=0.8,
).encode(y="y:Q")

vrules = alt.Chart(df_knees).mark_rule(
    strokeDash=[4, 4], opacity=0.5, strokeWidth=0.8,
).encode(
    x="x:Q",
    color=alt.Color("benchmark:N", scale=color_scale, legend=None),
)

points = alt.Chart(df_knees).mark_point(size=60, filled=True).encode(
    x="x:Q",
    y="y:Q",
    color=alt.Color("benchmark:N", scale=color_scale, legend=None),
)

labels = alt.Chart(df_knees).mark_text(align="left", dx=5, dy=-12, fontSize=8).encode(
    x="x:Q",
    y="y:Q",
    text="label:N",
    color=alt.Color("benchmark:N", scale=color_scale, legend=None),
)

chart = (lines + hline + vrules + points + labels).properties(
    width=380,
    height=260,
).configure_axis(
    grid=False,
    labelFontSize=9,
    titleFontSize=10,
).configure_view(
    strokeWidth=0,
).configure_legend(
    orient="bottom-right",
    labelFontSize=9,
    titleFontSize=9,
)

out_path = OUT_DIR / "saturation_coverage_curve.png"
chart.save(str(out_path), scale_factor=2)
print(f"Saved {out_path}")
