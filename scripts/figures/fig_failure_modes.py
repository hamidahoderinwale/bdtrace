"""Failure mode decomposition for failing trajectories.

Splits failures into two types:
  A — never reached the gold file
  B — reached the gold file but still failed

For type B, shows the distribution of steps spent after localization.

Two-panel figure:
  Left:  stacked bar per agent — proportion of failures that are type A vs B
  Right: box plot of steps-after-localization for type B failures, by agent

Reads:
    output/trajectories/.cache/{agent}/*.json
    output/trajectories/lite_all_models.parquet
    output/resolved_traces_lite_full.jsonl
Writes:
    output/figures/fig_failure_modes.png
    output/paper2_pilot/failure_modes.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER, AGENT_SHORT, VERMILLION, GRAY, BLUE
from analysis.preferences.localization import (
    load_gold_files, load_pass_fail, first_localization_step, AGENT_MAP
)
register()

OUT     = ROOT / "output" / "paper2_pilot"
FIG_OUT = ROOT / "output" / "figures"
CACHE   = ROOT / "output" / "trajectories" / ".cache"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    gold_files = load_gold_files()
    pass_fail  = load_pass_fail()

    records = []
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_short = AGENT_MAP.get(agent_dir.name)
        if agent_short is None:
            continue
        for traj_file in sorted(agent_dir.glob("*.json")):
            iid    = traj_file.stem
            gold   = gold_files.get(iid)
            passed = pass_fail.get((agent_dir.name, iid))
            if gold is None or passed is None:
                continue
            raw  = json.loads(traj_file.read_text())
            traj = raw.get("trajectory", [])
            n    = len(traj)
            if n == 0:
                continue
            loc_step  = first_localization_step(traj, gold)
            localized = loc_step is not None
            records.append({
                "agent":        agent_short,
                "instance_id":  iid,
                "passed":       passed,
                "n_steps":      n,
                "localized":    localized,
                "loc_step":     loc_step,
                "steps_after":  (n - loc_step) if localized else None,
            })

    df = pd.DataFrame(records)
    fails = df[~df["passed"]].copy()

    # ── summary stats ─────────────────────────────────────────────────────────
    summary = {}
    for agent in AGENT_ORDER:
        sub = fails[fails["agent"] == agent]
        n_total       = len(sub)
        n_never       = int((~sub["localized"]).sum())
        n_reached     = int(sub["localized"].sum())
        steps_after   = sub[sub["localized"]]["steps_after"].dropna()
        summary[agent] = {
            "n_fail":           n_total,
            "n_never_reached":  n_never,
            "n_reached_failed": n_reached,
            "frac_never":       n_never / n_total if n_total else float("nan"),
            "frac_reached":     n_reached / n_total if n_total else float("nan"),
            "steps_after_median": float(steps_after.median()) if len(steps_after) else None,
            "steps_after_q25":    float(steps_after.quantile(0.25)) if len(steps_after) else None,
            "steps_after_q75":    float(steps_after.quantile(0.75)) if len(steps_after) else None,
        }
        print(f"{agent:12s}  n_fail={n_total}  "
              f"never={n_never} ({n_never/n_total:.0%})  "
              f"reached-but-failed={n_reached} ({n_reached/n_total:.0%})  "
              f"median steps after={steps_after.median():.0f}" if len(steps_after) else
              f"{agent:12s}  n_fail={n_total}  never={n_never}  reached={n_reached}")

    # ── Panel A: stacked bar ───────────────────────────────────────────────────
    bar_rows = []
    for agent in AGENT_ORDER:
        s = summary[agent]
        bar_rows.append({"agent": agent, "mode": "Never reached gold file",
                          "frac": s["frac_never"]})
        bar_rows.append({"agent": agent, "mode": "Reached but failed",
                          "frac": s["frac_reached"]})
    bar_df = pd.DataFrame(bar_rows)

    color_scale = alt.Scale(
        domain=["Never reached gold file", "Reached but failed"],
        range=[VERMILLION, GRAY],
    )
    agent_color = alt.Scale(
        domain=AGENT_ORDER,
        range=[AGENT_COLORS[a] for a in AGENT_ORDER],
    )

    panel_a = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            y=alt.Y("agent:N", sort=AGENT_ORDER,
                    axis=alt.Axis(title=None)),
            x=alt.X("frac:Q", title="Proportion of failing trajectories",
                    stack="normalize",
                    axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0])),
            color=alt.Color("mode:N", scale=color_scale,
                            legend=alt.Legend(title=None, orient="bottom")),
        )
        .properties(
            width=260, height=130,
            title=alt.TitleParams("Failure types by agent",
                                  fontSize=12, color="#111111", anchor="start"),
        )
    )

    # ── Panel B: steps after localization for type B failures ─────────────────
    b_df = fails[fails["localized"]].copy()
    b_df["agent_label"] = b_df["agent"]

    # Compute quartiles manually for errorbar
    box_rows = []
    for agent in AGENT_ORDER:
        sub = b_df[b_df["agent"] == agent]["steps_after"].dropna()
        if len(sub) == 0:
            continue
        box_rows.append({
            "agent":  agent,
            "median": float(sub.median()),
            "q25":    float(sub.quantile(0.25)),
            "q75":    float(sub.quantile(0.75)),
            "q10":    float(sub.quantile(0.10)),
            "q90":    float(sub.quantile(0.90)),
        })
    box_df = pd.DataFrame(box_rows)

    base_b = alt.Chart(box_df).encode(
        y=alt.Y("agent:N", sort=AGENT_ORDER,
                axis=alt.Axis(title=None)),
    )

    def colored_layer(mark_fn, **kwargs):
        layers = []
        for agent in AGENT_ORDER:
            sub = box_df[box_df["agent"] == agent]
            if len(sub) == 0:
                continue
            layers.append(
                alt.Chart(sub)
                .transform_filter(alt.datum.agent == agent)
                .__getattribute__(mark_fn)(**kwargs)
                .encode(
                    y=alt.Y("agent:N", sort=AGENT_ORDER,
                            axis=alt.Axis(title=None)),
                    color=alt.value(AGENT_COLORS[agent]),
                )
            )
        return layers

    whisker_layers = [
        alt.Chart(box_df[box_df["agent"] == agent])
        .mark_rule(color=AGENT_COLORS[agent])
        .encode(
            y=alt.Y("agent:N", sort=AGENT_ORDER, axis=alt.Axis(title=None)),
            x=alt.X("q10:Q", title="Steps after reaching gold file"),
            x2="q90:Q",
        )
        for agent in AGENT_ORDER if agent in box_df["agent"].values
    ]
    iqr_layers = [
        alt.Chart(box_df[box_df["agent"] == agent])
        .mark_bar(size=14, color=AGENT_COLORS[agent])
        .encode(
            y=alt.Y("agent:N", sort=AGENT_ORDER, axis=alt.Axis(title=None)),
            x=alt.X("q25:Q"),
            x2="q75:Q",
        )
        for agent in AGENT_ORDER if agent in box_df["agent"].values
    ]
    med_layers = [
        alt.Chart(box_df[box_df["agent"] == agent])
        .mark_tick(size=14, thickness=2, color="white")
        .encode(
            y=alt.Y("agent:N", sort=AGENT_ORDER, axis=alt.Axis(title=None)),
            x=alt.X("median:Q"),
        )
        for agent in AGENT_ORDER if agent in box_df["agent"].values
    ]

    panel_b = (
        alt.layer(*whisker_layers, *iqr_layers, *med_layers)
        .properties(
            width=220, height=130,
            title=alt.TitleParams("Steps after localization (type B failures)",
                                  fontSize=12, color="#111111", anchor="start"),
        )
    )

    for panel, name in [
        (panel_a, "fig_failure_types.png"),
        (panel_b, "fig_failure_steps.png"),
    ]:
        out_fig = FIG_OUT / name
        (
            panel
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        ).save(str(out_fig), scale_factor=2)
        print(f"\nSaved {out_fig}")

    (OUT / "failure_modes.json").write_text(
        json.dumps({"by_agent": summary}, indent=2, default=float)
    )
    print(f"Saved {OUT / 'failure_modes.json'}")


if __name__ == "__main__":
    main()
