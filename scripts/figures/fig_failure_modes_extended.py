"""Phase 8: failure-mode anatomy on the 8-submission extended corpus.

Extends fig_failure_modes.py to scaffolds beyond legacy SWE-agent. Type
A = "never reached gold file" / reached-but-failed = "reached but failed". Steps-after-
localization is reported in canonical-sequence units (same units across
scaffolds).

Per-submission applicability:
  SWE-agent legacy 4         never-reached/B applicable
  Claude-3.7 SWE-agent       never-reached/B applicable
  DARS+R1                    never-reached/B applicable (action regex over file paths)
  Agentless+Claude-3.5       NOT APPLICABLE — section-level only; localization
                              N/A. Reported with marker.
  Moatless+V3                never-reached/B applicable (walks action_steps tree)

Reads:
    output/trajectories/.cache/<sub>/<iid>.json
    output/paper2_pilot/extended_pass_fail.json
    output/resolved_traces_lite_full.jsonl   (gold-file lookup)
    output/paper2_pilot/bpe_sequences_extended.jsonl  (canonical-length per traj)
Writes:
    output/figures/fig_localization_outcome_extended.png   (renamed from fig_failure_types_extended.png)
    output/figures/fig_steps_after_localization_extended.png (renamed from fig_failure_steps_extended.png)
    output/paper2_pilot/failure_modes_extended.json

Usage:
    python -m scripts.figures.fig_failure_modes_extended
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, MAGENTA, OLIVE, COPPER, GREEN, BLUE
register()

from analysis.preferences.localization import load_gold_files
from analysis.preferences.localization_extended import first_localization_step_extended
from analysis.preferences.canonicalize_extended import canonicalize_envelope

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"
CACHE = ROOT / "output" / "trajectories" / ".cache"

PASS_FILE = OUT_DAT / "extended_pass_fail.json"
SEQ_FILE = OUT_DAT / "bpe_sequences_extended.jsonl"

SUBMISSION_LABEL = {
    "20240402_sweagent_claude3opus":                                   "Claude-3",
    "20240402_sweagent_gpt4":                                          "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                               "Claude-3.5",
    "20240728_sweagent_gpt4o":                                         "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":                    "Claude-3.7-thinking",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":               "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":               "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                                   "Moatless+V3",
    "20250526_sweagent_claude-4-sonnet-20250514":                      "Claude-4",
}

# Display order: legacy SWE-agent first, then new scaffolds
AGENT_ORDER = [
    "Claude-3", "Claude-3.5", "GPT-4", "GPT-4o",
    "Claude-3.7-thinking", "Claude-4", "DARS+R1", "Moatless+V3", "Agentless+Claude-3.5",
]
AGENT_COLORS = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "Claude-3.7-thinking":   "#187860",
    "DARS+R1":               "#3D7AD8",
    "Moatless+V3":           "#A03D18",
    "Agentless+Claude-3.5":  OLIVE,
}


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DAT.mkdir(parents=True, exist_ok=True)

    gold_files = load_gold_files()
    pass_data = json.loads(PASS_FILE.read_text())

    # Map (agent_short, instance_id) -> canonical sequence length
    seq_len: dict[tuple[str, str], int] = {}
    with SEQ_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            seq_len[(r["agent"], r["instance_id"])] = r["canonical_length"]

    records: list[dict] = []
    for sub_id, agent_short in SUBMISSION_LABEL.items():
        sub_dir = CACHE / sub_id
        if not sub_dir.is_dir():
            continue
        resolved = set(pass_data.get(sub_id, {}).get("resolved", []))
        for traj_file in sorted(sub_dir.glob("*.json")):
            if traj_file.name == "manifest.json":
                continue
            iid = traj_file.stem
            gold = gold_files.get(iid)
            if not gold:
                continue
            try:
                env = json.loads(traj_file.read_text())
            except Exception:
                continue
            n_canon = seq_len.get((agent_short, iid))
            if n_canon is None:
                # Skip if not in BPE corpus (canonicalization yielded empty)
                continue
            loc = first_localization_step_extended(env, gold)
            passed = iid in resolved
            records.append({
                "submission":   sub_id,
                "agent":        agent_short,
                "instance_id":  iid,
                "passed":       passed,
                "n_canonical":  n_canon,
                "loc_step":     loc,
                "localized":    isinstance(loc, int),
                "applicable":   loc != "n/a",
                "steps_after":  (n_canon - loc) if isinstance(loc, int) else None,
            })

    df = pd.DataFrame(records)
    fails = df[~df["passed"]].copy()
    print(f"\nTotal trajectories: {len(df)}  |  failures: {len(fails)}")

    summary: dict[str, dict] = {}
    for agent in AGENT_ORDER:
        sub = fails[fails["agent"] == agent]
        n_total = len(sub)
        if n_total == 0:
            summary[agent] = {"n_fail": 0, "applicable": False}
            continue
        if not sub["applicable"].any():
            summary[agent] = {
                "n_fail": int(n_total),
                "applicable": False,
                "note": "section-level scaffold; type A/B not applicable",
            }
            continue
        # only count among applicable failures
        sub_app = sub[sub["applicable"]]
        n_app = len(sub_app)
        n_never = int((~sub_app["localized"]).sum())
        n_reached = int(sub_app["localized"].sum())
        steps_after = sub_app[sub_app["localized"]]["steps_after"].dropna()
        summary[agent] = {
            "n_fail":              int(n_total),
            "n_fail_applicable":   int(n_app),
            "n_never_reached":     n_never,
            "n_reached_failed":    n_reached,
            "frac_never":          n_never / n_app if n_app else None,
            "frac_reached":        n_reached / n_app if n_app else None,
            "steps_after_median":  float(steps_after.median()) if len(steps_after) else None,
            "steps_after_q25":     float(steps_after.quantile(0.25)) if len(steps_after) else None,
            "steps_after_q75":     float(steps_after.quantile(0.75)) if len(steps_after) else None,
            "applicable":          True,
        }
        print(f"  {agent:23s}  n_fail={n_total}  applicable={n_app}  "
              f"never={n_never} ({n_never/n_app:.0%}) reached={n_reached} ({n_reached/n_app:.0%})  "
              f"median_steps_after={steps_after.median():.0f}" if len(steps_after) else
              f"  {agent:23s}  n_fail={n_total}  applicable={n_app}")

    # Panel A: never-reached vs B (only for applicable agents)
    bar_rows = []
    for agent in AGENT_ORDER:
        s = summary.get(agent, {})
        if not s.get("applicable"):
            continue
        bar_rows.append({"agent": agent, "mode": "Never reached gold file",
                         "frac": s["frac_never"]})
        bar_rows.append({"agent": agent, "mode": "Reached but failed",
                         "frac": s["frac_reached"]})
    bar_df = pd.DataFrame(bar_rows)
    applicable_agents = [a for a in AGENT_ORDER if summary.get(a, {}).get("applicable")]

    color_scale = alt.Scale(
        domain=["Never reached gold file", "Reached but failed"],
        range=[MAGENTA, OLIVE],
    )

    panel_a = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            y=alt.Y("agent:N", sort=applicable_agents, axis=alt.Axis(title=None)),
            x=alt.X("frac:Q",
                    title="Proportion of failing trajectories",
                    stack="normalize",
                    axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0])),
            color=alt.Color("mode:N", scale=color_scale,
                            legend=alt.Legend(title=None, orient="bottom")),
        )
        .properties(
            width=300, height=180,
            title=alt.TitleParams(
                "Failure breakdown per agent: never reached gold file vs reached but failed",
                fontSize=12, color="#111111", anchor="start"),
        )
    )

    out_a = OUT_FIG / "fig_localization_outcome_extended.png"
    (panel_a.configure_view(strokeWidth=0).configure_axis(grid=False)).save(
        str(out_a), scale_factor=2
    )
    print(f"\nSaved {out_a}")

    # Panel B: steps-after for reached-but-failed failures
    b_df = fails[fails["localized"]].copy()
    box_rows = []
    for agent in applicable_agents:
        sub = b_df[b_df["agent"] == agent]["steps_after"].dropna()
        if len(sub) == 0:
            continue
        box_rows.append({
            "agent": agent,
            "median": float(sub.median()),
            "q25": float(sub.quantile(0.25)),
            "q75": float(sub.quantile(0.75)),
            "q10": float(sub.quantile(0.10)),
            "q90": float(sub.quantile(0.90)),
            "n":   int(len(sub)),
        })
    box_df = pd.DataFrame(box_rows)

    if len(box_df) > 0:
        whisker_layers = []
        iqr_layers = []
        med_layers = []
        for agent in applicable_agents:
            r = box_df[box_df["agent"] == agent]
            if r.empty:
                continue
            color = AGENT_COLORS.get(agent, OLIVE)
            whisker_layers.append(
                alt.Chart(r).mark_rule(color=color).encode(
                    y=alt.Y("agent:N", sort=applicable_agents, axis=alt.Axis(title=None)),
                    x=alt.X("q10:Q",
                            title="Steps after reaching gold file (canonical units)"),
                    x2="q90:Q",
                )
            )
            iqr_layers.append(
                alt.Chart(r).mark_bar(size=14, color=color).encode(
                    y=alt.Y("agent:N", sort=applicable_agents, axis=alt.Axis(title=None)),
                    x="q25:Q", x2="q75:Q",
                )
            )
            med_layers.append(
                alt.Chart(r).mark_tick(size=14, thickness=2, color="white").encode(
                    y=alt.Y("agent:N", sort=applicable_agents, axis=alt.Axis(title=None)),
                    x="median:Q",
                )
            )
        panel_b = (
            alt.layer(*whisker_layers, *iqr_layers, *med_layers)
            .properties(
                width=280, height=180,
                title=alt.TitleParams(
                    "Steps after reaching the gold file (reached-but-failed trajectories)",
                    fontSize=12, color="#111111", anchor="start",
                ),
            )
        )
        out_b = OUT_FIG / "fig_steps_after_localization_extended.png"
        (panel_b.configure_view(strokeWidth=0).configure_axis(grid=False)).save(
            str(out_b), scale_factor=2
        )
        print(f"Saved {out_b}")

    out_json = OUT_DAT / "failure_modes_extended.json"
    out_json.write_text(json.dumps({
        "n_trajectories": int(len(df)),
        "n_failures":     int(len(fails)),
        "by_agent":       summary,
    }, indent=2, default=float))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
