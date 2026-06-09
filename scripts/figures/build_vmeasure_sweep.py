#!/usr/bin/env python3
"""Figure: V-measure vs BPE vocabulary size, canonical vs native alphabet.

Two-line chart. X-axis is the BPE target vocabulary size V (log-spaced
sweep points). Y-axis is the V-measure of the induced motif clustering
against agent-identity labels. One line per alphabet:

  * canonical -- the small abstract action alphabet (read_file, edit,
    run_test, search_repo, ...). Coarse; saturates early.
  * native    -- the raw scaffold-specific tool names. Fine-grained;
    keeps gaining and peaks at a larger V.

The reading: finer atoms carry more agent-discriminative signal but
need a larger vocabulary to express it.

Reads:
  output/paper2_pilot/vmeasure_sweep.json
    schema: {"V": [16,32,64,128,256,512],
             "canonical": [<v_measure per V>],
             "native":    [<v_measure per V>]}

  If that file is absent, the sweep must be produced first by running
  the V-measure battery (analysis/pdiff/vmeasure.py) across vocabulary
  sizes for each alphabet. This script intentionally does NOT fabricate
  the curve -- it renders whatever sweep JSON it is given.

Writes:
  figures/procedural-diff/fig_vmeasure_sweep.{png,svg}

Usage:
    python -m scripts.figures.build_vmeasure_sweep
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import BLUE, COPPER, register  # noqa: E402

register()

SWEEP = ROOT / "output" / "paper2_pilot" / "vmeasure_sweep.json"
OUT_DIR = ROOT / "figures" / "procedural-diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHABET_COLOR = {"canonical": BLUE, "native": COPPER}
ALPHABET_LABEL = {"canonical": "Canonical alphabet", "native": "Native alphabet"}


def load_sweep(path: Path) -> pd.DataFrame:
    """Load the sweep JSON into long form: (V, alphabet, v_measure)."""
    if not path.exists():
        raise SystemExit(
            f"sweep file not found: {path}\n"
            "Produce it first by sweeping the V-measure battery "
            "(analysis/pdiff/vmeasure.py) over vocabulary sizes for the "
            "canonical and native alphabets, then write the JSON described "
            "in this module's docstring. This script does not fabricate the curve."
        )
    raw = json.loads(path.read_text())
    rows: list[dict] = []
    for alphabet in ("canonical", "native"):
        for v, score in zip(raw["V"], raw[alphabet], strict=True):
            rows.append(
                {"V": int(v), "alphabet": ALPHABET_LABEL[alphabet], "v_measure": float(score)}
            )
    return pd.DataFrame(rows)


def build_chart(df: pd.DataFrame) -> alt.Chart:
    color_domain = [ALPHABET_LABEL["canonical"], ALPHABET_LABEL["native"]]
    color_range = [ALPHABET_COLOR["canonical"], ALPHABET_COLOR["native"]]

    base = alt.Chart(df).encode(
        x=alt.X(
            "V:Q",
            scale=alt.Scale(type="log", base=2),
            title="BPE vocabulary size (V)",
            axis=alt.Axis(domain=False, ticks=False, grid=False),
        ),
        y=alt.Y(
            "v_measure:Q",
            title="V-measure vs agent labels",
            scale=alt.Scale(domainMin=0.0),
            axis=alt.Axis(domain=False, ticks=False, grid=True),
        ),
        color=alt.Color(
            "alphabet:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title=None, orient="top-left"),
        ),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=55, filled=True)
    chart = (lines + points).properties(
        width=420,
        height=240,
        title=alt.TitleParams(
            "Procedural vocabulary discriminates agents; "
            "native alphabet peaks higher at larger V",
            fontSize=12,
            anchor="start",
        ),
    )
    return chart


def main() -> None:
    df = load_sweep(SWEEP)
    chart = build_chart(df)
    png = OUT_DIR / "fig_vmeasure_sweep.png"
    svg = OUT_DIR / "fig_vmeasure_sweep.svg"
    chart.save(str(png), scale_factor=2)
    chart.save(str(svg))
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
