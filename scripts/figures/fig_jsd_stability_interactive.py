"""Interactive JSD-vs-V line chart with hover tooltips.

Standalone HTML version of the V-sweep stability figure. Hover over a
line to see per-pair JSD at every V plus the seq-share decomposition.
Demonstrates rank-order invariance + magnitude amplification visually.

Reads:  output/paper2_pilot/bpe_mdl_sweep.json
Writes: output/figures/fig_jsd_stability_interactive.html
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, GREEN, COPPER
register()

DATA = ROOT / "output" / "paper2_pilot" / "bpe_mdl_sweep.json"
OUT  = ROOT / "output" / "figures" / "fig_jsd_stability_interactive.html"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sweep = json.loads(DATA.read_text())

    rows = []
    for r in sweep["results"]:
        V = r["V"]
        for pair, jsd in r["jsd_motifs"].items():
            a, b = pair.split("__")
            rows.append({"V": V, "pair": f"{a} / {b}", "jsd": jsd})
    df = pd.DataFrame(rows)

    Vs = sorted(df["V"].unique())
    V_low, V_high = int(Vs[0]), int(Vs[-1])

    seq_share = {}
    for pair, sub in df.groupby("pair"):
        sub = sub.sort_values("V")
        low  = float(sub.iloc[0]["jsd"])
        high = float(sub.iloc[-1]["jsd"])
        seq_share[pair] = (high - low) / high if high > 0 else 0.0
    df["seq_share_pct"] = df["pair"].map(lambda p: f"{seq_share[p]*100:.0f}%")

    pairs_sorted = sorted(seq_share.keys(), key=lambda p: -seq_share[p])
    palette = [BLUE, GREEN, COPPER]
    color_scale = alt.Scale(domain=pairs_sorted, range=palette[:len(pairs_sorted)])

    hover = alt.selection_point(
        fields=["pair"],
        on="mouseover",
        nearest=True,
        empty=False,
    )

    base = alt.Chart(df).encode(
        x=alt.X("V:Q",
                title="BPE vocabulary size V",
                scale=alt.Scale(type="log", domain=[Vs[0]*0.9, Vs[-1]*1.1]),
                axis=alt.Axis(values=Vs)),
        y=alt.Y("jsd:Q",
                title="Pairwise JSD (bits)",
                scale=alt.Scale(domain=[0, max(df["jsd"]) * 1.1])),
        color=alt.Color("pair:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom")),
    )

    lines = base.mark_line(strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.35)),
    )
    points = base.mark_point(filled=True, size=80).encode(
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.35)),
        tooltip=[
            alt.Tooltip("pair:N", title="agent pair"),
            alt.Tooltip("V:Q", title="V"),
            alt.Tooltip("jsd:Q", title="JSD (bits)", format=".3f"),
            alt.Tooltip("seq_share_pct:N", title="sequence-level share"),
        ],
    ).add_params(hover)

    chart = (
        alt.layer(lines, points)
        .properties(
            width=520, height=320,
            title=alt.TitleParams(
                "JSD vs vocabulary size",
                subtitle="Rank order is invariant; magnitudes amplify with V. Hover a line for details.",
                fontSize=13, subtitleFontSize=11,
                color="#111111", subtitleColor="#666666", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    chart.save(str(OUT))
    print(f"Saved {OUT}")
    print()
    print("seq-share per pair:")
    for pair in pairs_sorted:
        print(f"  {pair:30s}  {seq_share[pair]*100:.0f}%")


if __name__ == "__main__":
    main()
