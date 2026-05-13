"""Stage allocation: fraction of steps per behavioral stage, by agent.

Maps canonical atom counts (SEARCH, OPEN, NAV, EDIT, CREATE, RUN, ...)
to five behavioural stages. Shows mean fraction per stage per agent as
a stacked horizontal bar. Passing and failing trajectories shown
separately.

Stage mapping (from canonical atoms in bpe_sequences_extended.jsonl):
    Explore = atoms starting with SEARCH
    Browse  = atoms starting with OPEN / NAV / FIND
    Edit    = atoms starting with EDIT / CREATE
    Test    = atoms starting with RUN
    Other   = remaining atoms (SHELL_*, SUBMIT, EMPTY, UNKNOWN_*, etc.)

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
from scripts.theme import register, STAGE_COLORS, STAGE_ORDER, AGENT_ORDER
register()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extended_pass_fail_df import SUBMISSION_TO_AGENT

OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOME_ORDER = ["pass", "fail"]


def _classify_atom(atom: str) -> str:
    """Map a canonical atom to one of the five behavioural stages."""
    if atom.startswith("SEARCH"):
        return "f_explore"
    if atom.startswith(("OPEN", "NAV", "FIND")):
        return "f_browse"
    if atom.startswith(("EDIT", "CREATE")):
        return "f_edit"
    if atom.startswith("RUN"):
        return "f_test"
    return "f_other"


def _per_traj_stage_fractions() -> pd.DataFrame:
    """Build a (agent, instance_id, passed, f_*) DataFrame from the
    extended canonical-atom sequences plus extended_pass_fail.json."""
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
            counts = {"f_explore": 0, "f_browse": 0, "f_edit": 0,
                      "f_test": 0, "f_other": 0}
            for a in atoms:
                counts[_classify_atom(a)] += 1
            agent = r["agent"]
            iid = r["instance_id"]
            rows.append({
                "agent":   agent,
                "instance_id": iid,
                "passed":  iid in resolved_by_agent.get(agent, set()),
                "f_explore": counts["f_explore"] / n_steps,
                "f_browse":  counts["f_browse"]  / n_steps,
                "f_edit":    counts["f_edit"]    / n_steps,
                "f_test":    counts["f_test"]    / n_steps,
                "f_other":   counts["f_other"]   / n_steps,
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = _per_traj_stage_fractions()
    df["passed_label"] = df["passed"].map({True: "pass", False: "fail"})

    stage_cols = {
        "Explore": "f_explore",
        "Browse":  "f_browse",
        "Edit":    "f_edit",
        "Test":    "f_test",
        "Other":   "f_other",
    }

    rows = []
    for agent in AGENT_ORDER:
        for outcome in OUTCOME_ORDER:
            sub = df[(df["agent"] == agent) & (df["passed_label"] == outcome)]
            if len(sub) == 0:
                continue
            for stage, col in stage_cols.items():
                rows.append({
                    "agent":   agent,
                    "outcome": outcome,
                    "stage":   stage,
                    "frac":    float(sub[col].mean()),
                    "n":       len(sub),
                })

    plot_df = pd.DataFrame(rows)

    # Sort stages in display order (Other last)
    stage_display_order = ["Explore", "Browse", "Edit", "Test", "Other"]
    color_domain = stage_display_order
    color_range = [
        STAGE_COLORS.get("Explore", "#56B4E9"),
        STAGE_COLORS.get("Browse",  "#009E73"),
        STAGE_COLORS.get("Edit",    "#E69F00"),
        STAGE_COLORS.get("Test",    "#D55E00"),
        "#CCCCCC",  # Other
    ]
    # Explicit sort index for stacking order
    stage_sort_idx = {s: i for i, s in enumerate(stage_display_order)}
    plot_df["stage_order"] = plot_df["stage"].map(stage_sort_idx)

    # y-axis: agent x outcome combos, ordered by agent then outcome
    y_order = [
        f"{a} ({o})"
        for a in AGENT_ORDER
        for o in OUTCOME_ORDER
        if len(plot_df[(plot_df["agent"] == a) & (plot_df["outcome"] == o)]) > 0
    ]
    plot_df["group"] = plot_df["agent"] + " (" + plot_df["outcome"] + ")"

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y("group:N", sort=y_order,
                    axis=alt.Axis(title=None, labelFontSize=11)),
            x=alt.X("frac:Q",
                    title="Mean fraction of steps",
                    stack="normalize",
                    axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1.0])),
            color=alt.Color("stage:N",
                            sort=stage_display_order,
                            scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title=None, orient="bottom",
                                              columns=len(stage_display_order))),
            order=alt.Order("stage_order:Q", sort="ascending"),
        )
        .properties(
            width=360, height=260,
            title=alt.TitleParams(
                "Stage allocation by agent and outcome",
                fontSize=13, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out = OUT / "fig_stage_allocation.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out}")

    # Print summary stats
    print("\nMean stage fractions (pass trajectories):")
    for agent in AGENT_ORDER:
        sub = plot_df[(plot_df["agent"] == agent) & (plot_df["outcome"] == "pass")]
        if len(sub) == 0:
            continue
        fracs = {r["stage"]: r["frac"] for _, r in sub.iterrows()}
        n = sub["n"].iloc[0]
        print(f"  {agent:12s} (n={n:3d}): "
              + "  ".join(f"{s}={fracs.get(s, 0):.2f}" for s in stage_display_order))

    print("\nMean stage fractions (fail trajectories):")
    for agent in AGENT_ORDER:
        sub = plot_df[(plot_df["agent"] == agent) & (plot_df["outcome"] == "fail")]
        if len(sub) == 0:
            continue
        fracs = {r["stage"]: r["frac"] for _, r in sub.iterrows()}
        n = sub["n"].iloc[0]
        print(f"  {agent:12s} (n={n:3d}): "
              + "  ".join(f"{s}={fracs.get(s, 0):.2f}" for s in stage_display_order))


if __name__ == "__main__":
    main()
