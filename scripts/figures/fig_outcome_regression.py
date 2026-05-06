"""Outcome regression ablation: AUC by feature set.

Tests whether agent identity, motifs, or both predict pass/fail.
Key finding: agent alone is at chance (AUC = 0.50); motifs alone reach AUC = 0.685;
adding agent to motifs does not improve over motifs alone.

This is the cleanest version of the "agent identity does not predict success"
finding — directly comparable to the 1.5% MI figure but with a held-out evaluation.

Reads:  output/paper2_pilot/outcome_regression.json
Writes: output/figures/fig_outcome_regression.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GRAY, BLUE, GREEN

register()

FIG_OUT  = ROOT / "output" / "figures"
DATA_IN  = ROOT / "output" / "paper2_pilot" / "outcome_regression.json"


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    data = json.loads(DATA_IN.read_text())
    abl  = data["ablations"]

    rows = [
        {"label": "Agent identity only",         "auc": abl["agent_only_auc"],         "color_key": "chance"},
        {"label": "Motif features only",         "auc": abl["motifs_only_auc"],        "color_key": "signal"},
        {"label": "Motifs + agent identity",     "auc": abl["motifs_plus_agent_auc"],  "color_key": "signal"},
    ]
    df = pd.DataFrame(rows)
    df["pct"] = (df["auc"] * 100).round(1)
    df["label_text"] = df["pct"].apply(lambda v: f"{v:.1f}%")

    color_scale = alt.Scale(
        domain=["chance", "signal"],
        range=[GRAY, BLUE],
    )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort=df["label"].tolist(), axis=alt.Axis(title=None)),
            x=alt.X("auc:Q",
                    title="AUC (held-out, 5-fold CV)",
                    scale=alt.Scale(domain=[0, 1.0]),
                    axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1.0])),
            color=alt.Color("color_key:N", scale=color_scale, legend=None),
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, fontSize=11, color="#333333")
        .encode(
            y=alt.Y("label:N", sort=df["label"].tolist()),
            x=alt.X("auc:Q"),
            text="label_text:N",
        )
    )

    chart = (
        alt.layer(bars, labels)
        .properties(
            width=320,
            height=110,
            title=alt.TitleParams(
                "Pass/fail prediction: agent vs motifs",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = FIG_OUT / "fig_outcome_regression.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
