"""Representation comparison: kNN pass/fail F1 at k=2.

Horizontal dot plot, globally sorted by F1 descending. Random baseline
shown as a natural row in its sorted position. Alternating row shading
for readability. Dot color encodes family: BLUE=structural, COPPER=semantic,
OLIVE=baseline.

Reads:  output/representation_comparison/knn_f1_results.csv
Writes: output/figures/fig_representation_comparison.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, COPPER, OLIVE
register()

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    "fim_jaccard":  "FIM pattern overlap",
    "edit_cert":    "Edit ops (Jaccard)",
    "motif":        "Motif distance (BPE)",
    "staged_embed": "Staged narrative",
    "cot_embed":    "Chain-of-thought",
    "fix_type":     "Fix type taxonomy",
}

STRUCTURAL_REPS = {"fim_jaccard", "edit_cert", "motif"}
ROW_A = "#F2F4F8"
ROW_B = "#FFFFFF"


def main():
    df_raw = pd.read_csv(ROOT / "output/representation_comparison/knn_f1_results.csv")
    row = df_raw[df_raw["k"] == 2].iloc[0]

    rows = []
    for key, label in LABELS.items():
        rows.append({
            "label": label,
            "f1": float(row[key]),
            "group": "Structural" if key in STRUCTURAL_REPS else "Semantic",
        })
    rows.append({"label": "Random baseline", "f1": 0.205, "group": "Baseline"})

    df = (
        pd.DataFrame(rows)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    df["row_bg"] = [ROW_A if i % 2 == 0 else ROW_B for i in range(len(df))]
    df["x0"] = 0.0
    df["x1"] = 0.42
    df["f1_label"] = df["f1"].map(lambda v: f"{v:.3f}")

    order = df["label"].tolist()
    x_domain = [0.0, 0.42]

    y_enc = alt.Y("label:N", sort=order,
                  axis=alt.Axis(title=None, labelFontSize=11, labelLimit=220,
                                ticks=False, domain=False))

    bands = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            y=y_enc,
            x=alt.X("x0:Q", scale=alt.Scale(domain=x_domain),
                    axis=alt.Axis(labels=False, ticks=False, domain=False, title=None)),
            x2=alt.X2("x1:Q"),
            color=alt.Color("row_bg:N", scale=None, legend=None),
        )
    )

    color_scale = alt.Scale(
        domain=["Structural", "Semantic", "Baseline"],
        range=[BLUE, COPPER, OLIVE],
    )

    dots = (
        alt.Chart(df)
        .mark_point(size=110, filled=True, strokeWidth=0)
        .encode(
            y=y_enc,
            x=alt.X("f1:Q",
                    title="Pass/fail prediction F1 (k=2)",
                    scale=alt.Scale(domain=x_domain),
                    axis=alt.Axis(values=[0.0, 0.10, 0.20, 0.30, 0.40], format=".2f")),
            color=alt.Color("group:N", scale=color_scale, legend=None),
        )
    )

    val_labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=9, dy=0, fontSize=10, color="#444444")
        .encode(
            y=y_enc,
            x=alt.X("f1:Q", scale=alt.Scale(domain=x_domain)),
            text=alt.Text("f1_label:N"),
        )
    )

    chart = (
        (bands + dots + val_labels)
        .properties(
            width=340, height=220,
            title=alt.TitleParams(
                "Prediction F1 by representation type",
                subtitle=f"Blue = structural  |  Orange = semantic  |  Gray = random baseline",
                subtitleFontSize=10, subtitleColor="#666666",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = OUT / "fig_representation_comparison.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
