"""Per-agent pass rate by difficulty bucket (9-agent extended corpus).

Shows whether the difficulty effect is uniform across agents or whether
some agents are disproportionately affected at certain difficulty levels.
Each agent is a separate line; difficulty (0/9 to 9/9) is the x-axis.
Wilson 95% CIs shown as error bars.

Reads:  output/paper2_pilot/extended_pass_fail.json (via _extended_pass_fail_df)
        output/paper2_pilot/bpe_sequences_extended.jsonl (trajectory inventory)
Writes: output/figures/fig_difficulty_by_agent.png
        output/paper2_pilot/difficulty_by_agent.json
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

OUT     = ROOT / "output" / "paper2_pilot"
FIG_OUT = ROOT / "output" / "figures"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    df = load_extended_traj_pass_fail()
    df = df[df["agent"].notna()].copy()

    DIFFICULTY_LABELS = {n: f"{n} / 9" for n in range(10)}

    rows = []
    summary = {}
    for agent in AGENT_ORDER:
        summary[agent] = {}
        for nr in sorted(df["n_resolved"].unique()):
            sub = df[(df["agent"] == agent) & (df["n_resolved"] == nr)]
            if len(sub) == 0:
                continue
            k = int(sub["passed"].sum())
            n = len(sub)
            lo, hi = wilson_ci(k, n)
            label = DIFFICULTY_LABELS[int(nr)]
            rows.append({
                "agent":  agent,
                "bucket": int(nr),
                "label":  label,
                "mean":   k / n,
                "lo":     lo,
                "hi":     hi,
                "n":      n,
            })
            summary[agent][label] = {"pass_rate": k / n, "n": n, "lo": lo, "hi": hi}
            print(f"  {agent:12s}  {label}  n={n:3d}  rate={k/n:.0%}")

    plot_df = pd.DataFrame(rows)
    x_order = [DIFFICULTY_LABELS[i] for i in range(10)]

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    base = alt.Chart(plot_df).encode(
        x=alt.X("label:N", sort=x_order,
                axis=alt.Axis(title="Difficulty (agents that solved it / 9)",
                              labelAngle=0)),
        color=alt.Color("agent:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom")),
    )

    lines = base.mark_line(point=True, strokeWidth=2).encode(
        y=alt.Y("mean:Q",
                title="Pass rate",
                scale=alt.Scale(domain=[0, 1.0]),
                axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0])),
    )

    errors = base.mark_errorbar().encode(
        y=alt.Y("lo:Q", title="Pass rate",
                scale=alt.Scale(domain=[0, 1.0])),
        y2=alt.Y2("hi:Q"),
    )

    chart = (
        (lines + errors)
        .resolve_scale(y="shared")
        .properties(
            width=320, height=240,
            title=alt.TitleParams(
                "Pass rate by difficulty and agent",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = FIG_OUT / "fig_difficulty_by_agent.png"
    chart.save(str(out), scale_factor=2)
    (OUT / "difficulty_by_agent.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
