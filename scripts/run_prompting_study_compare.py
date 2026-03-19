#!/usr/bin/env python3
"""
Compare prompting study results across two task model tiers.

Loads records.json from two run directories and produces cross-model plots:
  - Score by condition × model tier (grouped bars)
  - Lift (procedural − no_context) per model tier (overlaid box plots)
  - Divergence level distribution per model tier
  - Per-instance structural distance between tiers on the same task

Usage:
  python scripts/run_prompting_study_compare.py \
    --a output/prompting_study/gpt_4o_mini \
    --b output/prompting_study/gpt_4o \
    --output-dir output/prompting_study/compare
"""

import argparse
import json
from pathlib import Path

# Wong colorblind-safe palette
CONDITION_COLORS = {
    "no_context": "#999999",
    "raw_logs": "#E69F00",
    "procedural": "#0072B2",
}
CONDITION_ORDER = ["no_context", "raw_logs", "procedural"]
SCORE_DIMS = ["localization", "edit_type", "plan_quality", "explanation"]
LEVEL_ORDER = ["surface", "compositional", "relational"]
LEVEL_COLORS = ["#CC79A7", "#F0E442", "#009E73"]
# Two model tiers: solid vs dashed, dark vs light blue
TIER_COLORS = ["#0072B2", "#56B4E9"]


def load_records(path: Path) -> list[dict]:
    with open(path / "records.json") as f:
        return json.load(f)


def _slug(path: Path) -> str:
    return path.name.replace("_", "-")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        print("Install with: uv sync --extra notebooks")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="First run dir (e.g. gpt-4o-mini)")
    parser.add_argument("--b", type=Path, required=True, help="Second run dir (e.g. gpt-4o)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/prompting_study/compare"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_a = _slug(args.a)
    label_b = _slug(args.b)

    records_a = load_records(args.a)
    records_b = load_records(args.b)
    print(f"Loaded {len(records_a)} ({label_a}) + {len(records_b)} ({label_b}) records")

    color_scale = alt.Scale(domain=[label_a, label_b], range=TIER_COLORS)

    # 1. Mean score per condition × dimension × model tier
    rows = []
    for label, records in [(label_a, records_a), (label_b, records_b)]:
        for rec in records:
            for condition in CONDITION_ORDER:
                scores = rec["conditions"].get(condition, {}).get("scores", {})
                for dim in SCORE_DIMS:
                    v = scores.get(dim)
                    if isinstance(v, (int, float)):
                        rows.append({
                            "model": label, "condition": condition,
                            "dimension": dim, "score": float(v),
                        })

    if rows:
        df = pd.DataFrame(rows)
        mean_df = df.groupby(["model", "condition", "dimension"])["score"].mean().reset_index()
        chart = (
            alt.Chart(mean_df)
            .mark_bar()
            .encode(
                alt.X("condition:N", title=None, sort=CONDITION_ORDER,
                      axis=alt.Axis(labelAngle=0)),
                alt.Y("score:Q", title="mean score (0–3)", scale=alt.Scale(domain=[0, 3])),
                alt.Color("model:N", scale=color_scale, title="model"),
                alt.XOffset("model:N"),
                alt.Column("dimension:N", title=None,
                           header=alt.Header(labelOrient="bottom", labelPadding=4)),
            )
            .properties(width=90, height=200,
                        title=f"Score by condition × dimension: {label_a} vs {label_b}")
        )
        chart.save(args.output_dir / "scores_by_model.png")
        print("Saved scores_by_model.png")

    # 2. Lift (procedural − no_context) per instance, overlaid by model tier
    lift_rows = []
    for label, records in [(label_a, records_a), (label_b, records_b)]:
        for rec in records:
            base = rec["conditions"].get("no_context", {}).get("scores", {})
            proc = rec["conditions"].get("procedural", {}).get("scores", {})
            for dim in SCORE_DIMS:
                b, p = base.get(dim), proc.get(dim)
                if isinstance(b, (int, float)) and isinstance(p, (int, float)):
                    lift_rows.append({
                        "model": label, "dimension": dim,
                        "lift": float(p) - float(b),
                    })

    if lift_rows:
        lift_df = pd.DataFrame(lift_rows)
        lift_chart = (
            alt.Chart(lift_df)
            .mark_boxplot(size=22)
            .encode(
                alt.X("dimension:N", title=None, axis=alt.Axis(labelAngle=0)),
                alt.Y("lift:Q", title="lift (procedural − no context)",
                      scale=alt.Scale(domain=[-3, 3])),
                alt.Color("model:N", scale=color_scale, title="model"),
                alt.XOffset("model:N"),
            )
            .properties(width=380, height=250,
                        title="Procedural lift by model tier and dimension")
        )
        zero = (
            alt.Chart(pd.DataFrame([{"y": 0}]))
            .mark_rule(color="#444444", strokeDash=[4, 2])
            .encode(alt.Y("y:Q"))
        )
        (lift_chart + zero).save(args.output_dir / "lift_by_model.png")
        print("Saved lift_by_model.png")

    # 3. Divergence level distribution per model tier × condition
    div_rows = []
    for label, records in [(label_a, records_a), (label_b, records_b)]:
        for rec in records:
            for condition in CONDITION_ORDER:
                scores = rec["conditions"].get(condition, {}).get("scores", {})
                level = scores.get("divergence_level", "none")
                if level and level != "none":
                    div_rows.append({"model": label, "condition": condition,
                                     "divergence_level": level})

    if div_rows:
        div_df = pd.DataFrame(div_rows)
        div_counts = (
            div_df.groupby(["model", "condition", "divergence_level"])
            .size()
            .reset_index(name="count")
        )
        div_chart = (
            alt.Chart(div_counts)
            .mark_bar()
            .encode(
                alt.X("condition:N", title=None, sort=CONDITION_ORDER,
                      axis=alt.Axis(labelAngle=0)),
                alt.Y("count:Q", title="failure count"),
                alt.Color("divergence_level:N", sort=LEVEL_ORDER,
                          scale=alt.Scale(domain=LEVEL_ORDER, range=LEVEL_COLORS),
                          title="divergence level"),
                alt.Order("divergence_level:N", sort="ascending"),
                alt.Column("model:N", title=None),
            )
            .properties(width=200, height=200,
                        title="Divergence level by condition and model tier")
        )
        div_chart.save(args.output_dir / "divergence_by_model.png")
        print("Saved divergence_by_model.png")

    # 4. Per-instance procedural score: model A vs model B scatter (procedural condition)
    # Match on instance_id
    id_to_a = {r["instance_id"]: r for r in records_a}
    id_to_b = {r["instance_id"]: r for r in records_b}
    shared_ids = set(id_to_a) & set(id_to_b)
    scatter_rows = []
    for iid in shared_ids:
        for dim in SCORE_DIMS:
            sa = id_to_a[iid]["conditions"].get("procedural", {}).get("scores", {}).get(dim)
            sb = id_to_b[iid]["conditions"].get("procedural", {}).get("scores", {}).get(dim)
            if isinstance(sa, (int, float)) and isinstance(sb, (int, float)):
                scatter_rows.append({
                    "instance_id": iid, "dimension": dim,
                    f"score_{label_a}": float(sa),
                    f"score_{label_b}": float(sb),
                })

    if scatter_rows:
        sc_df = pd.DataFrame(scatter_rows)
        scatter = (
            alt.Chart(sc_df)
            .mark_circle(size=40, opacity=0.5)
            .encode(
                alt.X(f"score_{label_a}:Q", title=label_a, scale=alt.Scale(domain=[-0.2, 3.2])),
                alt.Y(f"score_{label_b}:Q", title=label_b, scale=alt.Scale(domain=[-0.2, 3.2])),
                alt.Color("dimension:N", scale=alt.Scale(scheme="tableau10"), title="dimension"),
                alt.Tooltip(["instance_id", "dimension",
                             f"score_{label_a}", f"score_{label_b}"]),
                alt.Column("dimension:N", title=None),
            )
            .properties(width=130, height=130,
                        title=f"Per-instance procedural scores: {label_a} vs {label_b}")
        )
        diag_df = pd.DataFrame({"x": [0, 3], "y": [0, 3]})
        diag = (
            alt.Chart(diag_df)
            .mark_line(color="#888888", strokeDash=[3, 2])
            .encode(alt.X("x:Q"), alt.Y("y:Q"))
        )
        (scatter + diag).save(args.output_dir / "per_instance_scatter.png")
        print("Saved per_instance_scatter.png")

    print(f"\nCompare plots in {args.output_dir}")


if __name__ == "__main__":
    main()
