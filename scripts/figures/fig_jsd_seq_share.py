"""V-sweep abstraction-level decomposition (seq-share figure).

Decomposes pairwise JSD growth across vocabulary size V into atom-level
(present at smallest V) vs sequence-level (added by larger merges) shares.

For each agent pair (A, B):
    JSD_low  = JSD at V_min  (atom-dominated regime)
    JSD_high = JSD at V_max  (sequence-amplified regime)
    seq_share(A, B) = (JSD_high - JSD_low) / JSD_high

High seq-share = sequence-level distinctiveness; low seq-share = atom-level.

Reads:  output/paper2_pilot/bpe_mdl_sweep.json   (9 V values, motif JSDs)
Writes: output/figures/fig_jsd_seq_share.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, COPPER, OLIVE
register()

DATA  = ROOT / "output" / "paper2_pilot" / "bpe_mdl_sweep.json"
OUT   = ROOT / "output" / "figures" / "fig_jsd_seq_share.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sweep = json.loads(DATA.read_text())
    Vs = sorted(int(r["V"]) for r in sweep["results"])
    V_low, V_high = Vs[0], Vs[-1]

    # Pull motif-level JSDs at low and high V
    jsd_low_dict  = {r["V"]: r["jsd_motifs"] for r in sweep["results"]}[V_low]
    jsd_high_dict = {r["V"]: r["jsd_motifs"] for r in sweep["results"]}[V_high]

    rows = []
    for pair, jsd_high in jsd_high_dict.items():
        jsd_low = jsd_low_dict[pair]
        a, b = pair.split("__")
        seq_share = (jsd_high - jsd_low) / jsd_high if jsd_high > 0 else 0.0
        rows.append({
            "pair":      f"{a} / {b}",
            "jsd_low":   jsd_low,
            "jsd_high":  jsd_high,
            "atom_part": jsd_low,
            "seq_part":  jsd_high - jsd_low,
            "seq_share": seq_share,
            "label":     f"{seq_share*100:.0f}%",
        })
    df = pd.DataFrame(rows).sort_values("seq_share", ascending=False).reset_index(drop=True)
    pair_order = df["pair"].tolist()

    # Long form for stacked bar
    long_rows = []
    for _, r in df.iterrows():
        long_rows.append({"pair": r["pair"], "component": "atom-level (V=100)",      "value": r["atom_part"]})
        long_rows.append({"pair": r["pair"], "component": "sequence-level (V=500 added)", "value": r["seq_part"]})
    long_df = pd.DataFrame(long_rows)

    color_scale = alt.Scale(
        domain=["atom-level (V=100)", "sequence-level (V=500 added)"],
        range=[OLIVE, BLUE],
    )

    bars = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            y=alt.Y("pair:N", sort=pair_order, axis=alt.Axis(title=None)),
            x=alt.X("value:Q", title="Pairwise JSD (bits)",
                    stack="zero",
                    scale=alt.Scale(domain=[0, max(df["jsd_high"]) * 1.15])),
            color=alt.Color("component:N", scale=color_scale,
                            legend=alt.Legend(title=None, orient="bottom")),
            order=alt.Order("component:N"),
        )
    )

    seq_share_labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontSize=11, color="#333333")
        .encode(
            y=alt.Y("pair:N", sort=pair_order),
            x=alt.X("jsd_high:Q"),
            text="label:N",
        )
    )

    chart = (
        alt.layer(bars, seq_share_labels)
        .properties(
            width=380, height=130,
            title=alt.TitleParams(
                "Pairwise JSD by abstraction level",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    chart.save(str(OUT), scale_factor=2)
    print(f"Saved {OUT}")
    print(f"\nseq_share per pair (corpus state: 3-agent, n={sweep['results'][0]['mdl']['n_tokens']:,} tokens at V={V_low}):")
    for _, r in df.iterrows():
        print(f"  {r['pair']:30s}  JSD {V_low}={r['jsd_low']:.3f}  JSD {V_high}={r['jsd_high']:.3f}  seq-share={r['seq_share']*100:.0f}%")


if __name__ == "__main__":
    main()
