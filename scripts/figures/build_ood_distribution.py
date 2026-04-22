#!/usr/bin/env python3
"""Figure 2: OOD distribution by benchmark (Lite vocab reference).

Histogram of per-trace OOD scores over fixed buckets: one row per
target benchmark (Verified, SWE-Smith), one column per reference
strictness (min_count=1 and min_count=2).

Reads output/pdiff_smoke_test/cross_benchmark_transfer.json and
writes figures/procedural-diff/fig2_ood_distribution.{png,svg}.

Usage:
    python -m scripts.figures.build_ood_distribution
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output" / "pdiff_smoke_test" / "cross_benchmark_transfer.json"
OUT_DIR = ROOT / "figures" / "procedural-diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Project palette (matches scripts/build_saturation_figure_altair.py).
COLORS = {
    "SWE-bench Lite": "#0072B2",
    "SWE-bench Verified": "#009E73",
    "SWE-Smith": "#E69F00",
}

BUCKETS = [
    ("{0}", "zero"),
    ("(0, 0.25]", "0_to_25"),
    ("(0.25, 0.5]", "25_to_50"),
    ("(0.5, 0.75]", "50_to_75"),
    ("(0.75, 1)", "75_to_100"),
    ("{1}", "one"),
]
BUCKET_LABELS = [b[0] for b in BUCKETS]

PANELS = [
    ("verified_vs_lite_mc1", "SWE-bench Verified", "min_count = 1"),
    ("verified_vs_lite_mc2", "SWE-bench Verified", "min_count = 2"),
    ("swe_smith_vs_lite_mc1", "SWE-Smith", "min_count = 1"),
    ("swe_smith_vs_lite_mc2", "SWE-Smith", "min_count = 2"),
]


def _rows_for_panel(panel_key: str, target: str, column: str, coverage: dict) -> list[dict]:
    edits = coverage[panel_key]["edits"]
    hist = edits["histogram"]
    n = edits["n"]
    rows = []
    for label, key in BUCKETS:
        count = hist.get(key, 0)
        rows.append({
            "target": target,
            "column": column,
            "bucket": label,
            "fraction": count / n if n else 0.0,
            "count": count,
            "n": n,
        })
    return rows


def _panel(df: pd.DataFrame, target: str, column: str, annot_label: str,
           show_y_axis: bool, show_x_axis: bool) -> alt.Chart:
    color = COLORS[target]
    sub = df[(df["target"] == target) & (df["column"] == column)]

    y_axis_args = {"format": ".0%"}
    if not show_y_axis:
        y_axis_args["title"] = None
    y = alt.Y(
        "fraction:Q",
        title="Fraction of traces" if show_y_axis else None,
        axis=alt.Axis(**y_axis_args),
        scale=alt.Scale(domain=[0, 1]),
    )

    x_args = {"labelAngle": -40, "labelFontSize": 8}
    x_title = "OOD score bucket" if show_x_axis else None
    if not show_x_axis:
        x_args["labels"] = False
    x = alt.X(
        "bucket:N",
        title=x_title,
        sort=BUCKET_LABELS,
        axis=alt.Axis(**x_args),
    )

    bars = alt.Chart(sub).mark_bar(color=color, size=22).encode(x=x, y=y)

    annot_df = pd.DataFrame([{"label": annot_label}])
    annot = alt.Chart(annot_df).mark_text(
        align="right", baseline="top", fontSize=9, color="#333333",
    ).encode(x=alt.value(195), y=alt.value(10), text="label:N")

    title = f"{target}  |  {column}"
    return (bars + annot).properties(width=200, height=140, title=alt.TitleParams(text=title, fontSize=10, anchor="start"))


def build_chart(df: pd.DataFrame, annots: dict[tuple[str, str], str]) -> alt.Chart:
    rows = []
    for i, target in enumerate(["SWE-bench Verified", "SWE-Smith"]):
        cols = []
        for j, column in enumerate(["min_count = 1", "min_count = 2"]):
            label = annots[(target, column)]
            show_y = (j == 0)
            show_x = (i == 1)
            cols.append(_panel(df, target, column, label, show_y, show_x))
        rows.append(alt.hconcat(*cols, spacing=18))
    chart = alt.vconcat(*rows, spacing=14)
    return chart.configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(
        strokeWidth=0,
    )


def main() -> int:
    data = json.loads(DATA.read_text())
    coverage = data["coverage"]

    rows: list[dict] = []
    annots: dict[tuple[str, str], str] = {}
    for key, target, column in PANELS:
        rows.extend(_rows_for_panel(key, target, column, coverage))
        edits = coverage[key]["edits"]
        pct = 100.0 * edits["frac_fully_covered"]
        annots[(target, column)] = f"{pct:.1f}% fully covered"

    df = pd.DataFrame(rows)
    chart = build_chart(df, annots)

    png = OUT_DIR / "fig2_ood_distribution.png"
    svg = OUT_DIR / "fig2_ood_distribution.svg"
    chart.save(str(png), scale_factor=2)
    chart.save(str(svg))
    print(f"Wrote {png}")
    print(f"Wrote {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
