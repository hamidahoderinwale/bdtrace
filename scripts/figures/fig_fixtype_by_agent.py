"""Pass rate by fix type and agent (9-agent extended corpus).

Shows where agents systematically differ — the routing signal.
Only fix types with n >= 10 per agent included.

Reads:  output/datasets/swe_bench_lite_resolved/fix_types.json
        output/paper2_pilot/extended_pass_fail.json (via _extended_pass_fail_df)
        output/paper2_pilot/bpe_sequences_extended.jsonl
Writes: output/figures/fig_fixtype_by_agent.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER
register()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extended_pass_fail_df import load_extended_traj_pass_fail

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MIN_N = 10

FIX_TYPE_LABELS = {
    "logic_fix":           "Logic fix",
    "exception_handling":  "Exception handling",
    "api_change":          "API change",
    "config_fix":          "Config fix",
    "guard_clause":        "Guard clause",
    "type_coercion":       "Type coercion",
    "refactor":            "Refactor",
    "string_fix":          "String fix",
    "test_fix":            "Test fix",
}


def main():
    # Per-instance fix type from the canonical fix_types.json (300 of 300
    # extended-corpus instances are covered).
    ft_data = json.loads(
        (ROOT / "output/datasets/swe_bench_lite_resolved/fix_types.json").read_text()
    )
    fix_df = pd.DataFrame([
        {"instance_id": r["instance_id"], "fix_type_hand": r["fix_type"]}
        for r in ft_data["results"]
    ]).drop_duplicates()

    # Per-(agent, instance) pass/fail from the 9-agent extended corpus.
    traj_df = load_extended_traj_pass_fail()[["instance_id", "agent", "passed"]]

    df = traj_df.merge(fix_df, on="instance_id", how="inner")

    # Keep fix types with >= MIN_N per agent for ALL agents
    counts = df.groupby(["fix_type_hand", "agent"]).size().unstack(fill_value=0)
    valid_types = counts[counts.min(axis=1) >= MIN_N].index.tolist()
    df = df[df["fix_type_hand"].isin(valid_types)].copy()

    # Compute mean + 95% Wilson CI per fix_type x agent
    rows = []
    for ft in valid_types:
        for agent in AGENT_ORDER:
            sub = df[(df["fix_type_hand"] == ft) & (df["agent"] == agent)]
            n    = len(sub)
            if n == 0:
                continue
            k    = sub["passed"].sum()
            mean = k / n
            # Wilson score interval
            z = 1.96
            denom = 1 + z**2 / n
            centre = (mean + z**2 / (2 * n)) / denom
            half   = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denom
            rows.append({
                "fix_type":  FIX_TYPE_LABELS.get(ft, ft),
                "fix_type_raw": ft,
                "agent":     agent,
                "mean":      mean,
                "lo":        max(0.0, centre - half),
                "hi":        min(1.0, centre + half),
                "n":         n,
            })

    plot_df = pd.DataFrame(rows)

    # Order fix types by mean pass rate across all agents (ascending so best at top)
    ft_order = (
        plot_df.groupby("fix_type")["mean"].mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    # Alternating row bands for fix-type scan-tracking.
    band_df = pd.DataFrame([
        {"fix_type": ft} for i, ft in enumerate(ft_order) if i % 2 == 0
    ])
    bands = (
        alt.Chart(band_df)
        .mark_rect(fill="#F1F1EE", opacity=1.0, stroke=None)
        .encode(y=alt.Y("fix_type:N", sort=ft_order,
                        axis=alt.Axis(title=None, labelFontSize=11)))
    )

    base = alt.Chart(plot_df).encode(
        y=alt.Y("fix_type:N", sort=ft_order,
                axis=alt.Axis(title=None, labelFontSize=11)),
        color=alt.Color("agent:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom")),
        yOffset=alt.YOffset("agent:N", sort=AGENT_ORDER,
                            scale=alt.Scale(range=[-10, 10])),
    )

    points = base.mark_point(filled=True, size=70, strokeWidth=0).encode(
        x=alt.X("mean:Q",
                title="Pass rate",
                scale=alt.Scale(domain=[0, 0.55]),
                axis=alt.Axis(format=".0%", values=[0, 0.1, 0.2, 0.3, 0.4, 0.5])),
    )

    errors = base.mark_errorbar().encode(
        x=alt.X("lo:Q", title=""),
        x2=alt.X2("hi:Q"),
    )

    chart = (
        alt.layer(bands, points, errors)
        .properties(
            width=340, height=260,
            title=alt.TitleParams(
                "Pass rate by fix type and agent",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = OUT / "fig_fixtype_by_agent.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
