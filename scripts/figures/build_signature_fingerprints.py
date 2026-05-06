#!/usr/bin/env python3
"""Figure 4: signature fingerprints per benchmark.

Three horizontal-bar facets (Lite, Verified, SWE-Smith) sharing the
same y-axis: the top-20 edit ops by Lite frequency. X-axis is the
normalized frequency (occurrences / total edit-op occurrences in
that benchmark). Ops kept in Lite order across panels so readers
can see where the head agrees and the tail diverges.

Rebuilds signatures from the full resolved corpora (same caps as
run_cross_benchmark_transfer.py so the numbers are consistent).

Writes:
  figures/procedural-diff/fig4_signature_fingerprints.{png,svg}

Usage:
    python -m scripts.figures.build_signature_fingerprints
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import sys

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, GREEN, ORANGE
register()

from analysis.pdiff import signature, view_from_trace

LITE = ROOT / "output" / "resolved_traces_lite_full.jsonl"
VERIFIED = ROOT / "output" / "resolved_traces_verified_full.jsonl"
SWE_SMITH = ROOT / "output" / "resolved_traces_swe_smith.jsonl"
OUT_DIR = ROOT / "figures" / "procedural-diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SWE_SMITH_CAP = 5000
TOP_K = 20

COLORS = {
    "SWE-bench Lite": BLUE,
    "SWE-bench Verified": GREEN,
    "SWE-Smith": ORANGE,
}


def _load_views(path: Path, cap: int | None = None) -> list:
    views = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            views.append(view_from_trace(trace))
            if cap is not None and len(views) >= cap:
                break
    return views


def _signature_freq(views: list) -> Counter:
    sig = signature(views)
    return sig.edit_freq


def _normalize(freq: Counter) -> dict[str, float]:
    total = sum(freq.values())
    if total == 0:
        return {}
    return {op: c / total for op, c in freq.items()}


def build_chart(df: pd.DataFrame, top_ops: list[str]) -> alt.Chart:
    color_scale = alt.Scale(
        domain=list(COLORS.keys()),
        range=list(COLORS.values()),
    )

    panels = []
    for bench in ["SWE-bench Lite", "SWE-bench Verified", "SWE-Smith"]:
        sub = df[df["benchmark"] == bench]
        chart = alt.Chart(sub).mark_bar().encode(
            y=alt.Y(
                "op:N",
                title=None,
                sort=top_ops,
                axis=alt.Axis(labelFontSize=9),
            ),
            x=alt.X(
                "fraction:Q",
                title="Share of edit ops",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 0.12]),
            ),
            color=alt.Color("benchmark:N", scale=color_scale, legend=None),
            tooltip=["benchmark:N", "op:N", "fraction:Q", "count:Q"],
        ).properties(
            width=180,
            height=340,
            title=alt.TitleParams(text=bench, fontSize=11, anchor="start"),
        )
        # Hide y-axis labels for non-leftmost panels so they share the axis.
        if bench != "SWE-bench Lite":
            chart = chart.encode(
                y=alt.Y(
                    "op:N",
                    title=None,
                    sort=top_ops,
                    axis=alt.Axis(labels=False, ticks=False),
                ),
            )
        panels.append(chart)

    chart = alt.hconcat(*panels, spacing=12).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(
        strokeWidth=0,
    )
    return chart


def main() -> int:
    print("Loading Lite views...")
    lite_views = _load_views(LITE)
    print(f"  lite: {len(lite_views)}")
    print("Loading Verified views...")
    verified_views = _load_views(VERIFIED)
    print(f"  verified: {len(verified_views)}")
    print(f"Loading SWE-Smith views (cap {SWE_SMITH_CAP})...")
    swe_smith_views = _load_views(SWE_SMITH, cap=SWE_SMITH_CAP)
    print(f"  swe_smith: {len(swe_smith_views)}")

    lite_freq = _signature_freq(lite_views)
    verified_freq = _signature_freq(verified_views)
    swe_freq = _signature_freq(swe_smith_views)

    # Top-K ops by Lite frequency; ordered high to low for y-axis sort top-down.
    top_ops = [op for op, _ in lite_freq.most_common(TOP_K)]

    lite_norm = _normalize(lite_freq)
    verified_norm = _normalize(verified_freq)
    swe_norm = _normalize(swe_freq)

    rows = []
    for op in top_ops:
        for bench, freq, norm in [
            ("SWE-bench Lite", lite_freq, lite_norm),
            ("SWE-bench Verified", verified_freq, verified_norm),
            ("SWE-Smith", swe_freq, swe_norm),
        ]:
            rows.append({
                "benchmark": bench,
                "op": op,
                "count": int(freq.get(op, 0)),
                "fraction": float(norm.get(op, 0.0)),
            })
    df = pd.DataFrame(rows)

    # Sanity print.
    print("\nTop-20 Lite ops (op, Lite share, Verified share, SWE-Smith share):")
    for op in top_ops:
        print(f"  {op:<22s} {lite_norm.get(op, 0):.3f}  "
              f"{verified_norm.get(op, 0):.3f}  {swe_norm.get(op, 0):.3f}")

    chart = build_chart(df, top_ops)
    png = OUT_DIR / "fig4_signature_fingerprints.png"
    svg = OUT_DIR / "fig4_signature_fingerprints.svg"
    chart.save(str(png), scale_factor=2)
    chart.save(str(svg))
    print(f"Wrote {png}")
    print(f"Wrote {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
