"""Stage allocation: fraction of steps per behavioral stage, by agent.

Maps canonical atom counts to six behavioural stages. One facet panel
per agent; each panel shows pass and fail as two stacked horizontal bars.

Stage mapping:
    Explore = SEARCH*
    Browse  = OPEN* / NAV* / FIND*
    Edit    = EDIT* / CREATE*
    Test    = RUN*
    Shell   = SHELL_*                  (was buried in Other)
    Finish  = SUBMIT*
    Other   = EXIT_ERROR, EMPTY, UNKNOWN_*, residual

Reads:  output/paper2_pilot/bpe_sequences_extended.jsonl
        output/paper2_pilot/extended_pass_fail.json
Writes: output/figures/fig_stage_allocation.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import (
    register, STAGE_COLORS, AGENT_ORDER,
    BLUE, GREEN, COPPER, MAGENTA, OLIVE, INDIGO, NEAR_BLACK,
)
register()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extended_pass_fail_df import SUBMISSION_TO_AGENT

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

STAGE_DISPLAY = ["Explore", "Browse", "Edit", "Test", "Shell", "Finish", "Other"]
STAGE_COLORS_EXT = {
    "Explore": BLUE,
    "Browse":  GREEN,
    "Edit":    COPPER,
    "Test":    MAGENTA,
    "Shell":   INDIGO,
    "Finish":  OLIVE,
    "Other":   "#C8C8C8",
}
OUTCOME_ORDER = ["pass", "fail"]


def _classify_atom(atom: str) -> str:
    if atom.startswith("SEARCH"):
        return "Explore"
    if atom.startswith(("OPEN", "NAV", "FIND")):
        return "Browse"
    if atom.startswith(("EDIT", "CREATE")):
        return "Edit"
    if atom.startswith("RUN"):
        return "Test"
    if atom.startswith("SHELL_"):
        return "Shell"
    if atom.startswith("SUBMIT"):
        return "Finish"
    return "Other"


def _per_traj_stage_fractions() -> pd.DataFrame:
    pf = json.loads(
        (ROOT / "output/paper2_pilot/extended_pass_fail.json").read_text()
    )
    resolved_by_agent: dict[str, set[str]] = {}
    for sub, agent in SUBMISSION_TO_AGENT.items():
        resolved_by_agent[agent] = set(pf.get(sub, {}).get("resolved", []))

    rows: list[dict] = []
    seq_path = ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl"
    with seq_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            atoms = r["canonical"]
            n_steps = max(len(atoms), 1)
            counts = {s: 0 for s in STAGE_DISPLAY}
            for a in atoms:
                counts[_classify_atom(a)] += 1
            agent = r["agent"]
            iid = r["instance_id"]
            row = {
                "agent":   agent,
                "instance_id": iid,
                "passed":  iid in resolved_by_agent.get(agent, set()),
            }
            for s in STAGE_DISPLAY:
                row[f"f_{s.lower()}"] = counts[s] / n_steps
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = _per_traj_stage_fractions()
    df["outcome"] = df["passed"].map({True: "pass", False: "fail"})

    # Aggregate: mean fraction per (agent, outcome, stage)
    rows = []
    for agent in AGENT_ORDER:
        for outcome in OUTCOME_ORDER:
            sub = df[(df["agent"] == agent) & (df["outcome"] == outcome)]
            if len(sub) == 0:
                continue
            for stage in STAGE_DISPLAY:
                col = f"f_{stage.lower()}"
                rows.append({
                    "agent":   agent,
                    "outcome": outcome,
                    "stage":   stage,
                    "frac":    float(sub[col].mean()),
                    "n":       len(sub),
                })

    plot_df = pd.DataFrame(rows)

    # Drop Other if it's < 1% on average — keeps the legend clean
    other_mean = plot_df[plot_df["stage"] == "Other"]["frac"].mean()
    if other_mean < 0.01:
        plot_df = plot_df[plot_df["stage"] != "Other"]
        stages_used = [s for s in STAGE_DISPLAY if s != "Other"]
    else:
        stages_used = STAGE_DISPLAY

    stage_sort_idx = {s: i for i, s in enumerate(stages_used)}
    plot_df["stage_order"] = plot_df["stage"].map(stage_sort_idx)

    color_domain = stages_used
    color_range  = [STAGE_COLORS_EXT[s] for s in stages_used]

    # Family-grouped order: Claude (4) → GPT (2) → Scaffolds (3)
    # With columns=3 this gives:
    #   Row 1: Claude-3, Claude-3.5, Claude-3.7-thinking
    #   Row 2: Claude-4, GPT-4, GPT-4o
    #   Row 3: DARS+R1, Agentless+Claude-3.5, Moatless+V3
    FAMILY_ORDER = [
        "Claude-3", "Claude-3.5", "Claude-3.7-thinking",
        "Claude-4", "GPT-4", "GPT-4o",
        "DARS+R1", "Agentless+Claude-3.5", "Moatless+V3",
    ]
    # Only keep agents present in the data
    family_order_filtered = [a for a in FAMILY_ORDER if a in plot_df["agent"].unique()]

    base = (
        alt.Chart(plot_df)
        .mark_bar(height=14)
        .encode(
            y=alt.Y(
                "outcome:N",
                sort=OUTCOME_ORDER,
                axis=alt.Axis(title=None, labelFontSize=10),
            ),
            x=alt.X(
                "frac:Q",
                stack="normalize",
                axis=alt.Axis(
                    title=None,
                    format=".0%",
                    values=[0, 0.5, 1.0],
                    labelFontSize=9,
                ),
            ),
            color=alt.Color(
                "stage:N",
                sort=stages_used,
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(
                    title=None, orient="bottom",
                    columns=len(stages_used),
                    labelFontSize=10,
                    symbolSize=60,
                ),
            ),
            order=alt.Order("stage_order:Q", sort="ascending"),
        )
    )

    chart = (
        base
        .facet(
            facet=alt.Facet(
                "agent:N",
                sort=family_order_filtered,
                header=alt.Header(
                    title=None,
                    labelFontSize=11,
                    labelColor=NEAR_BLACK,
                    labelOrient="top",
                ),
            ),
            columns=3,
        )
        .properties(
            title=alt.TitleParams(
                "Stage allocation by agent",
                fontSize=13,
                color=NEAR_BLACK,
                anchor="start",
                offset=8,
            ),
        )
        .resolve_scale(color="shared")
    )

    out = OUT / "fig_stage_allocation.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")

    # Summary stats
    print("\nShell fraction by agent (mean across pass+fail):")
    for agent in AGENT_ORDER:
        sub = plot_df[
            (plot_df["agent"] == agent) & (plot_df["stage"] == "Shell")
        ]
        if len(sub):
            print(f"  {agent:25s}: {sub['frac'].mean():.1%}")


if __name__ == "__main__":
    main()
