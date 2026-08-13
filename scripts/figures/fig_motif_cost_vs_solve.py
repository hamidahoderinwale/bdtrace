"""Motif cost vs solve rate scatter.

Each dot is a motif. X = mean tokens per use. Y = resolve rate when used.
Key motifs labeled. Shows that expensive motifs (edit bursts) resolve below
base rate; cheap motifs in the upper-left dominate successful trajectories.

Reads:  output/paper2_pilot/step_resources.json
Writes: output/figures/fig_motif_cost_vs_solve.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GRAY
register()

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Motifs worth labeling explicitly
LABEL_MOTIFS = {
    "EDIT_SRC_PY+SUBMIT":
        "EDIT then SUBMIT",
    "EDIT_SRC_PY+RUN_PYTHON_REPRO_PY+SHELL_RM+SUBMIT":
        "EDIT, verify, clean, SUBMIT",
    "EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY+EDIT_SRC_PY":
        "EDIT x32 burst",
}

def main():
    sr = json.loads((ROOT / "output/paper2_pilot/step_resources.json").read_text())
    motifs = sr["motifs"]

    # Corpus base resolve rate
    ef = sr["efficiency_frontier"]
    total_resolved = sum(v["n_resolved"] for v in ef.values())
    total_traj = sum(v["n_total"] for v in ef.values())
    base_rate = total_resolved / total_traj

    rows = []
    for key, v in motifs.items():
        if key.startswith("__") or v.get("occurrences", 0) < 5:
            continue
        short = LABEL_MOTIFS.get(key, "")
        rows.append({
            "motif": key,
            "label": short,
            "tokens": v["mean_tokens_per_use"],
            "solve_rate": v["success_rate_when_used"],
            "occurrences": v["occurrences"],
            "above_base": v["success_rate_when_used"] >= base_rate,
        })
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=[True, False],
        range=[BLUE, ORANGE],
    )

    dots = (
        alt.Chart(df)
        .mark_point(filled=True, opacity=0.55, strokeWidth=0)
        .encode(
            x=alt.X("tokens:Q",
                    title="Mean tokens per use",
                    scale=alt.Scale(domain=[0, 480000]),
                    axis=alt.Axis(format=",.0f",
                                  values=[0, 100000, 200000, 300000, 400000])),
            y=alt.Y("solve_rate:Q",
                    title="Resolve rate when used",
                    scale=alt.Scale(domain=[-0.02, 0.45]),
                    axis=alt.Axis(format=".0%",
                                  values=[0, 0.1, 0.2, 0.3, 0.4])),
            size=alt.Size("occurrences:Q", scale=alt.Scale(range=[30, 200]),
                          legend=None),
            color=alt.Color("above_base:N", scale=color_scale, legend=None),
        )
    )

    label_df = df[df["label"] != ""].copy()
    labs = (
        alt.Chart(label_df)
        .mark_text(fontSize=9, color="#333333", align="left", dx=6)
        .encode(
            x=alt.X("tokens:Q"),
            y=alt.Y("solve_rate:Q"),
            text=alt.Text("label:N"),
        )
    )

    chart = (
        (dots + labs)
        .properties(
            width=380, height=260,
            title=alt.TitleParams(
                "Motif cost (tokens) vs resolve rate",
                subtitle=f"Each dot is a motif (min 5 uses). Base resolve rate = {base_rate:.1%}. "
                         f"Blue = above base rate.",
                subtitleFontSize=10, subtitleColor="#888888",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = OUT / "fig_motif_cost_vs_solve.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
