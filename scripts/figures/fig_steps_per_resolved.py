"""Steps per resolved task — step-budget efficiency, all 9 agents.

Companion to fig_step_count_distribution_extended. Step count alone
doesn't tell you how productive each step was; combining with resolve
rate gives a single per-agent number: total actions across all 300
attempts divided by tasks resolved.

Reads:
    output/paper2_pilot/aggregate_metrics_extended.json
    output/paper2_pilot/extended_pass_fail.json
Writes:
    output/figures/fig_steps_per_resolved.png
    output/paper2_pilot/steps_per_resolved.json

Usage:
    uv run python scripts/figures/fig_steps_per_resolved.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, MAGENTA, COPPER, OLIVE, GREEN_D, BLUE_D, MAGENTA_D
register()

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"

SUBMISSION_LABEL = {
    "20240402_sweagent_claude3opus":                          "Claude-3",
    "20240402_sweagent_gpt4":                                 "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                      "Claude-3.5",
    "20240728_sweagent_gpt4o":                                "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":           "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":             "Claude-4",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":      "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":      "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                          "Moatless+V3",
}
AGENT_ORDER = [
    "Agentless+Claude-3.5", "Claude-3", "Claude-3.5",
    "Claude-3.7-thinking", "Claude-4", "DARS+R1",
    "GPT-4", "GPT-4o", "Moatless+V3",
]
AGENT_COLORS = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "Claude-3.7-thinking":   GREEN_D,
    "Claude-4":              "#187860",
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "DARS+R1":               MAGENTA_D,
    "Agentless+Claude-3.5":  BLUE_D,
    "Moatless+V3":           OLIVE,
}
LITE_TOTAL = 300


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    agg = json.loads((OUT_DAT / "aggregate_metrics_extended.json").read_text())
    pf  = json.loads((OUT_DAT / "extended_pass_fail.json").read_text())

    rows = []
    for sub_id, agent in SUBMISSION_LABEL.items():
        info = pf.get(sub_id, {})
        n_resolved = len(set(info.get("resolved", [])))
        # canonical_length comes from agg metrics
        m = agg.get("metrics", {}).get(agent, {})
        mean_len = m.get("canonical_length_mean") or m.get("mean_canonical_length") or m.get("mean_atoms")
        if mean_len is None or n_resolved == 0:
            print(f"  skip {agent}: missing data ({mean_len=}, {n_resolved=})")
            continue
        steps_per_resolved = mean_len * LITE_TOTAL / n_resolved
        rows.append({
            "agent":              agent,
            "mean_atoms":         round(float(mean_len), 1),
            "n_resolved":         int(n_resolved),
            "resolve_rate":       round(n_resolved / LITE_TOTAL, 3),
            "steps_per_resolved": round(float(steps_per_resolved), 1),
        })

    df = pd.DataFrame(rows).sort_values("steps_per_resolved")
    print("\n=== steps per resolved task ===")
    print(df.to_string(index=False))

    out_json = OUT_DAT / "steps_per_resolved.json"
    out_json.write_text(df.to_json(orient="records", indent=2))
    print(f"\nSaved {out_json}")

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("steps_per_resolved:Q",
                    axis=alt.Axis(title="Actions per resolved task",
                                  domain=False, ticks=False, labelFontSize=10)),
            y=alt.Y("agent:N",
                    sort=alt.SortField(field="steps_per_resolved", order="ascending"),
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=10, labelLimit=240)),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
            tooltip=["agent", "mean_atoms", "n_resolved", "resolve_rate", "steps_per_resolved"],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, fontSize=10, color="#444444")
        .encode(
            x="steps_per_resolved:Q",
            y=alt.Y("agent:N", sort=alt.SortField(field="steps_per_resolved", order="ascending")),
            text=alt.Text("steps_per_resolved:Q", format=".0f"),
        )
    )

    chart = (
        (bars + labels)
        .properties(
            width=440,
            height=max(260, 32 * len(df)),
            title=alt.TitleParams(
                text="Actions per resolved task",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    out_png = OUT_FIG / "fig_steps_per_resolved.png"
    chart.save(str(out_png), scale_factor=2)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
