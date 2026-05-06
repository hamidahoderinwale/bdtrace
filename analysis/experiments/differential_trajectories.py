"""Cross-agent differential trajectory analysis.

Finds instances where exactly one agent resolved the task and compares
the resolving agent's action sequence to the failing agents' sequences.
Reports which action types are over/under-represented in resolving trajectories.

Outputs:
    output/experiments/differential_trajectories.json
    output/experiments/differential_action_odds.png
"""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GRAY
register()

OUT = ROOT / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(ROOT / "output/trajectories/lite_all_models.parquet")
    print(f"Loaded {len(df)} trajectories, {df['instance_id'].nunique()} instances")

    # Count resolvers per instance
    per_instance = df.groupby("instance_id")["passed"].sum().reset_index()
    per_instance.columns = ["instance_id", "n_resolved"]

    # Differential: exactly 1 out of 3 resolved
    diff_ids = per_instance[per_instance["n_resolved"] == 1]["instance_id"].tolist()
    print(f"Differential instances (exactly 1 resolver): {len(diff_ids)}")

    diff_df = df[df["instance_id"].isin(diff_ids)].copy()

    # Collect action types from resolving vs failing trajectories
    resolve_actions: list[str] = []
    fail_actions:    list[str] = []

    for iid in diff_ids:
        sub = diff_df[diff_df["instance_id"] == iid]
        for _, row in sub.iterrows():
            acts = str(row["action_sequence"]).split()
            if row["passed"]:
                resolve_actions.extend(acts)
            else:
                fail_actions.extend(acts)

    # Action type counts and log2-odds
    all_types = sorted(set(resolve_actions) | set(fail_actions))
    rows = []
    n_res  = len(resolve_actions)
    n_fail = len(fail_actions)

    for at in all_types:
        if at in ("SUBMIT", "OTHER"):
            continue
        p_res  = (resolve_actions.count(at) + 0.5) / (n_res  + 1)
        p_fail = (fail_actions.count(at)   + 0.5) / (n_fail + 1)
        log_odds = float(np.log2(p_res / p_fail))
        rows.append({
            "action_type": at,
            "rate_resolving": p_res,
            "rate_failing": p_fail,
            "log2_odds": log_odds,
            "count_resolving": resolve_actions.count(at),
            "count_failing": fail_actions.count(at),
        })

    rows_df = pd.DataFrame(rows).sort_values("log2_odds", ascending=False)
    print(rows_df[["action_type", "log2_odds", "rate_resolving", "rate_failing"]].to_string(index=False))

    (OUT / "differential_trajectories.json").write_text(json.dumps({
        "n_differential_instances": len(diff_ids),
        "n_resolving_steps": n_res,
        "n_failing_steps": n_fail,
        "action_log_odds": rows,
    }, indent=2))

    # --- Plot: horizontal bar chart of log2-odds ---
    rows_df["color"] = rows_df["log2_odds"].apply(
        lambda x: "Overrepresented in resolving" if x > 0 else "Overrepresented in failing"
    )
    group_order = ["Overrepresented in resolving", "Overrepresented in failing"]
    cscale = alt.Scale(domain=group_order, range=[BLUE, ORANGE])
    act_order = rows_df.sort_values("log2_odds")["action_type"].tolist()

    bars = (
        alt.Chart(rows_df)
        .mark_bar(height=20)
        .encode(
            y=alt.Y("action_type:N", sort=act_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False, labelFontSize=12)),
            x=alt.X("log2_odds:Q",
                    title="Log2 odds ratio (resolving vs failing)",
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[-1, -0.5, 0, 0.5, 1.0])),
            color=alt.Color("color:N", scale=cscale,
                            legend=alt.Legend(title=None, orient="bottom")),
        )
    )
    pos_df = rows_df[rows_df["log2_odds"] > 0]
    neg_df = rows_df[rows_df["log2_odds"] <= 0]
    labels_pos = (
        alt.Chart(pos_df)
        .mark_text(align="left", dx=4, fontSize=10, color="#444444")
        .encode(
            y=alt.Y("action_type:N", sort=act_order),
            x=alt.X("log2_odds:Q"),
            text=alt.Text("log2_odds:Q", format="+.2f"),
        )
    )
    labels_neg = (
        alt.Chart(neg_df)
        .mark_text(align="right", dx=-4, fontSize=10, color="#444444")
        .encode(
            y=alt.Y("action_type:N", sort=act_order),
            x=alt.X("log2_odds:Q"),
            text=alt.Text("log2_odds:Q", format="+.2f"),
        )
    )
    chart = (
        (bars + labels_pos + labels_neg)
        .properties(
            title=alt.TitleParams(
                "Action types in resolving vs failing trajectories",
                subtitle=f"Differential instances only (n={len(diff_ids)}, exactly 1 of 3 agents resolved)",
                subtitleFontSize=10, subtitleColor="#888888",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=360, height=200,
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(OUT / "differential_action_odds.png"), scale_factor=2)
    print("Saved differential_action_odds.png")


if __name__ == "__main__":
    main()
