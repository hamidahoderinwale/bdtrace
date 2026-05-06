"""Efficiency: cost per resolved task and cost vs resolve-rate tradeoff.

Two-panel figure:
  Left  — horizontal bar of cost per resolved task (the primary efficiency metric)
  Right — scatter of cost-per-task vs resolve rate (the tradeoff space)

Reads:  output/paper2_pilot/step_resources.json (cost data)
        output/trajectories/lite_all_models.parquet (ground-truth pass rates)
Writes: output/figures/fig_efficiency_frontier.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER, AGENT_SHORT
register()

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    sr = json.loads((ROOT / "output/paper2_pilot/step_resources.json").read_text())
    ef = sr["efficiency_frontier"]

    traj = pd.read_parquet(ROOT / "output/trajectories/lite_all_models.parquet")
    traj["agent"] = traj["model_id"].map(AGENT_SHORT)
    pass_rates = traj.groupby("agent")["passed"].mean().to_dict()

    rows = []
    for agent, d in ef.items():
        rr = pass_rates.get(agent, d["resolve_rate"])
        cpr = d["mean_cost_per_task_usd"] / rr if rr > 0 else float("nan")
        rows.append({
            "agent":            agent,
            "cost_per_task":    d["mean_cost_per_task_usd"],
            "resolve_rate":     rr,
            "cost_per_resolved": cpr,
            "rr_label":         f"{rr:.1%} resolve",
            "cpr_label":        f"${cpr:.2f}",
        })
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    # ── Panel A: cost per resolved task (horizontal bar) ─────────────────────
    # Sort: cheapest (most efficient) at bottom → best resolve rate visible
    bar_order = (
        df.sort_values("cost_per_resolved", ascending=False)["agent"].tolist()
    )

    bar_base = alt.Chart(df).encode(
        y=alt.Y("agent:N", sort=bar_order,
                axis=alt.Axis(title=None, labelFontSize=11)),
        color=alt.Color("agent:N", scale=color_scale, legend=None),
    )

    bars = bar_base.mark_bar(height=18).encode(
        x=alt.X("cost_per_resolved:Q",
                title="Cost per resolved task (USD)",
                scale=alt.Scale(domain=[0, 35]),
                axis=alt.Axis(format="$,.0f", values=[0, 10, 20, 30])),
    )

    bar_labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=5, fontSize=10.5, color="#444444")
        .encode(
            y=alt.Y("agent:N", sort=bar_order),
            x=alt.X("cost_per_resolved:Q", scale=alt.Scale(domain=[0, 35])),
            text=alt.Text("cpr_label:N"),
        )
    )

    panel_a = (
        (bars + bar_labels)
        .properties(
            width=230, height=130,
            title=alt.TitleParams("Cost per resolved task (USD)",
                                  fontSize=12, color="#111111", anchor="start"),
        )
    )

    # ── Panel B: scatter — cost/task vs resolve rate ──────────────────────────
    x_min = max(0, df["cost_per_task"].min() * 0.80)
    x_max = df["cost_per_task"].max() * 1.12
    y_min = max(0, df["resolve_rate"].min() * 0.78)
    y_max = min(1.0, df["resolve_rate"].max() * 1.18)

    dots = (
        alt.Chart(df)
        .mark_point(size=160, filled=True, strokeWidth=0)
        .encode(
            x=alt.X("cost_per_task:Q",
                    title="Mean cost per task (USD)",
                    scale=alt.Scale(domain=[x_min, x_max]),
                    axis=alt.Axis(format="$.2f")),
            y=alt.Y("resolve_rate:Q",
                    title="Resolve rate",
                    scale=alt.Scale(domain=[y_min, y_max]),
                    axis=alt.Axis(format=".0%")),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )
    )

    agent_labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=9, dy=-5, fontSize=10.5)
        .encode(
            x=alt.X("cost_per_task:Q"),
            y=alt.Y("resolve_rate:Q"),
            text=alt.Text("agent:N"),
            color=alt.Color("agent:N", scale=color_scale, legend=None),
        )
    )

    panel_b = (
        (dots + agent_labels)
        .properties(
            width=210, height=130,
            title=alt.TitleParams("Cost per task vs resolve rate",
                                  fontSize=12, color="#111111", anchor="start"),
        )
    )

    for panel, name in [
        (panel_a, "fig_efficiency_frontier.png"),
        (panel_b, "fig_efficiency_scatter.png"),
    ]:
        out = OUT / name
        (
            panel
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        ).save(str(out), scale_factor=2)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
