"""Permutation null: heritability gap vs null distribution.

Shows the aggregate panel only (cleaner for paper).
Null distribution as a histogram; observed gap shown via colored bars
(bars at or beyond observed value are highlighted) and a text annotation.
No reference lines.

Reads:  output/paper2_pilot/bpe_sequences.jsonl  (re-runs permutation)
        output/paper2_pilot/permutation_null.json (for summary stats)
Writes: output/figures/fig_permutation_null.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GRAY, VERMILLION
register()

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    summary = json.loads(
        (ROOT / "output/paper2_pilot/permutation_null.json").read_text()
    )
    agg = summary["aggregate"]
    observed = agg["observed_gap"]
    null_mean = agg["null_mean"]
    null_std = agg["null_std"]
    n_perm = agg["n_permutations"]
    p_value = agg["p_value"]

    # Regenerate null distribution by re-running permutations with fixed seed.
    # We use the saved null_mean/std as a sanity check only.
    sys.path.insert(0, str(ROOT))
    from analysis.preferences.permutation_null import load_records, heritability_gap
    records = load_records()
    rng = np.random.default_rng(0)
    true_labels = [r["agent"] for r in records]
    labels_arr = np.array(true_labels)
    null_samples = np.array([
        heritability_gap(records, rng.permutation(labels_arr).tolist())
        for _ in range(n_perm)
    ])
    null_samples = null_samples[~np.isnan(null_samples)]

    counts, edges = np.histogram(null_samples, bins=30)
    bin_df = pd.DataFrame({
        "x0": edges[:-1],
        "x1": edges[1:],
        "count": counts,
        "beyond": edges[1:] >= observed,
    })

    obs_label = (
        alt.Chart(pd.DataFrame({"x": [observed], "label": [f"observed = {observed:.3f}  p = {p_value:.3f}"]}))
        .mark_text(align="left", dx=5, dy=20, fontSize=10, color=VERMILLION)
        .encode(
            x=alt.X("x:Q"),
            text="label:N",
            y=alt.value(0),
        )
    )

    # Color the bars that fall at or beyond the observed gap VERMILLION to mark the tail
    bars_colored = (
        alt.Chart(bin_df)
        .mark_bar(stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("x0:Q",
                    title="Same-family similarity advantage",
                    bin=alt.BinParams(binned=True),
                    scale=alt.Scale(domain=[-0.04, observed + 0.015]),
                    axis=alt.Axis(values=[-0.03, 0, 0.03])),
            x2="x1:Q",
            y=alt.Y("count:Q",
                    title="Count",
                    axis=alt.Axis(values=[0, 50, 100, 150])),
            color=alt.condition(
                alt.datum.beyond,
                alt.value(VERMILLION),
                alt.value(GRAY),
            ),
        )
    )

    chart = (
        (bars_colored + obs_label)
        .properties(
            width=340, height=200,
            title=alt.TitleParams(
                "Permutation null for agent procedural similarity",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = OUT / "fig_permutation_null.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
