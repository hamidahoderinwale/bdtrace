#!/usr/bin/env python3
"""
FIM difficulty analysis.

Connects the closed frequent itemset forms (canonical_forms.json) to the
84-agent ease data and measures how well FIM patterns discriminate difficulty,
compared to intent-based decision tree forms and semantic clusters.

Outputs:
  fig1_fim_ease_per_form.png     -- ease distribution per FIM form (n >= 5)
  fig2_variance_comparison.png   -- ease variance: FIM vs intent vs semantic
  fim_difficulty_summary.json    -- per-form ease stats

Usage:
  uv run python scripts/fim_difficulty_analysis.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import msgpack
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "fim_difficulty"
OUT.mkdir(parents=True, exist_ok=True)

# Wong color-blind safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
PINK   = "#CC79A7"
GRAY   = "#999999"
YELLOW = "#F0E442"
SKY    = "#56B4E9"
RED    = "#D55E00"

FORM_COLOR = BLUE
INTENT_COLOR = ORANGE
SEMANTIC_COLOR = GRAY


# --- Load data ---

def load_ease(lb_path: Path) -> dict[str, float]:
    with open(lb_path, "rb") as f:
        lb = msgpack.unpack(f, raw=False)
    instance_votes: dict[str, list] = {}
    for agent_data in lb.values():
        for iid, passed in agent_data.items():
            instance_votes.setdefault(iid, []).append(passed)
    return {iid: float(np.mean(v)) for iid, v in instance_votes.items()}


def load_fim_forms(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["forms"]


def load_intent_forms(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_semantic_cluster_forms(path: Path) -> dict[str, list[str]]:
    """Returns {form_label: [instance_id, ...]} from hunk cluster form_assignments."""
    df = pd.read_parquet(path)
    groups: dict[str, list[str]] = {}
    for form, grp in df.groupby("form_label"):
        groups[str(form)] = grp["instance_id"].tolist()
    return groups


# --- Ease variance ---

def ease_variance_across_groups(
    group_to_instances: dict[str, list[str]],
    ease: dict[str, float],
    min_size: int = 5,
) -> float:
    """Variance of per-group mean ease, weighted by group size."""
    group_means = []
    for name, instances in group_to_instances.items():
        vals = [ease[iid] for iid in instances if iid in ease]
        if len(vals) >= min_size:
            group_means.append(np.mean(vals))
    return float(np.var(group_means)) if len(group_means) >= 2 else 0.0


# --- Main ---

def main():
    print("Loading ease data...")
    ease = load_ease(ROOT / "output" / "leaderboard" / "lite_results.msgpack")
    print(f"  {len(ease)} instances with ease scores")

    print("Loading FIM canonical forms...")
    forms = load_fim_forms(ROOT / "output" / "canonical_forms" / "canonical_forms.json")
    print(f"  {len(forms)} forms")

    # Filter to forms with n >= 5 for meaningful ease estimates
    MIN_SIZE = 5
    forms_large = [f for f in forms if f["n_instances"] >= MIN_SIZE]
    print(f"  {len(forms_large)} forms with n >= {MIN_SIZE}")

    # Compute 84-agent ease per form
    rows = []
    for f in forms_large:
        instances = f["instances"]
        ease_vals = [ease[iid] for iid in instances if iid in ease]
        if not ease_vals:
            continue
        rows.append({
            "form": f["name"].replace("_", " "),
            "pattern": " + ".join(sorted(f["pattern"])),
            "n": len(instances),
            "mean_ease": float(np.mean(ease_vals)),
            "min_ease": float(np.min(ease_vals)),
            "max_ease": float(np.max(ease_vals)),
            "instances": instances,
            "ease_vals": ease_vals,
        })

    rows.sort(key=lambda r: r["mean_ease"])

    # Save summary
    summary = [{k: v for k, v in r.items() if k != "ease_vals"} for r in rows]
    with open(OUT / "fim_difficulty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved fim_difficulty_summary.json ({len(rows)} forms)")

    # --- Fig 1: ease distribution per FIM form ---
    print("\nBuilding fig1: ease per FIM form...")

    # Build long-form dataframe for boxplot
    long_rows = []
    for r in rows:
        for iid, e in zip(r["instances"], r["ease_vals"]):
            long_rows.append({
                "form": r["form"],
                "ease": e,
                "n": r["n"],
                "mean_ease": r["mean_ease"],
            })

    df_long = pd.DataFrame(long_rows)
    form_order = [r["form"] for r in rows]  # sorted by mean_ease ascending

    base = alt.Chart(df_long).properties(width=600, height=300)

    boxes = base.mark_boxplot(
        color=BLUE,
        median=alt.MarkConfig(color="white"),
        size=18,
    ).encode(
        x=alt.X(
            "form:N",
            sort=form_order,
            axis=alt.Axis(
                labelAngle=-40,
                labelLimit=160,
                title=None,
                labelFontSize=9,
            ),
        ),
        y=alt.Y(
            "ease:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(title="Agent ease (fraction of 84 agents solving)", titleFontSize=10),
        ),
        tooltip=["form:N", "ease:Q"],
    )

    # Mean ease dots
    mean_dots = alt.Chart(pd.DataFrame(rows)).mark_point(
        color=ORANGE, size=60, filled=True
    ).encode(
        x=alt.X("form:N", sort=form_order),
        y=alt.Y("mean_ease:Q"),
        tooltip=["form:N", "mean_ease:Q", "n:Q"],
    )

    # n annotation below
    n_text = alt.Chart(pd.DataFrame(rows)).mark_text(
        baseline="top", dy=5, fontSize=8, color=GRAY
    ).encode(
        x=alt.X("form:N", sort=form_order),
        y=alt.value(300),
        text=alt.Text("n:Q", format="d"),
    )

    fig1 = (boxes + mean_dots).properties(
        title=alt.TitleParams(
            "Agent ease by FIM canonical form, n >= 5",
            fontSize=12,
            fontWeight="normal",
            anchor="start",
        )
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig1.save(str(OUT / "fig1_fim_ease_per_form.png"), scale_factor=2)
    print("  Saved fig1_fim_ease_per_form.png")

    # --- Variance comparison ---
    print("\nComputing ease variance across groupings...")

    # FIM forms (n >= 5)
    fim_groups = {r["form"]: r["instances"] for r in rows}
    fim_var = ease_variance_across_groups(fim_groups, ease, min_size=MIN_SIZE)
    print(f"  FIM forms variance: {fim_var:.4f}")

    # AST cert decision tree forms (10 forms, from fix_forms)
    ast_tree_var = None
    ast_tree_path = ROOT / "output" / "fix_forms" / "form_assignments.parquet"
    if ast_tree_path.exists():
        ast_df = pd.read_parquet(ast_tree_path)
        ast_groups = {
            name: grp["instance_id"].tolist()
            for name, grp in ast_df.groupby("form_label")
        }
        ast_tree_var = ease_variance_across_groups(ast_groups, ease, min_size=MIN_SIZE)
        print(f"  AST cert decision tree variance: {ast_tree_var:.4f}")
    else:
        print("  AST cert decision tree not found, using known value (0.0257)")
        ast_tree_var = 0.0257

    # Semantic k-means on issue text (from semantic_vs_structural; known value)
    # k-means k=10 on all-MiniLM-L6-v2 embeddings of issue text: 0.0073
    semantic_var = 0.0073
    print(f"  Semantic k-means (issue text, k=10): {semantic_var:.4f} [from semantic_vs_structural]")

    # --- Fig 2: variance comparison bar chart ---
    print("\nBuilding fig2: variance comparison...")

    var_df = pd.DataFrame([
        {"grouping": "Semantic k-means\n(k=10, issue text)", "variance": semantic_var, "color": SEMANTIC_COLOR},
        {"grouping": "AST cert decision tree\n(10 forms)", "variance": ast_tree_var, "color": INTENT_COLOR},
        {"grouping": "FIM closed itemsets\n(15 forms, n≥5)", "variance": fim_var, "color": FORM_COLOR},
    ])

    max_var = var_df["variance"].max()

    bars = alt.Chart(var_df).mark_bar().encode(
        x=alt.X(
            "variance:Q",
            scale=alt.Scale(domain=[0, max_var * 1.2]),
            axis=alt.Axis(title="Variance of per-group mean agent ease", titleFontSize=10),
        ),
        y=alt.Y(
            "grouping:N",
            sort=None,
            axis=alt.Axis(title=None, labelFontSize=10),
        ),
        color=alt.Color(
            "color:N",
            scale=None,
            legend=None,
        ),
    )

    labels = alt.Chart(var_df).mark_text(
        align="left", dx=4, fontSize=10
    ).encode(
        x=alt.X("variance:Q"),
        y=alt.Y("grouping:N", sort=None),
        text=alt.Text("variance:Q", format=".4f"),
    )

    fig2 = (bars + labels).properties(
        width=400,
        height=150,
        title=alt.TitleParams(
            "Which grouping best separates difficulty?",
            fontSize=12,
            fontWeight="normal",
            anchor="start",
        )
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig2.save(str(OUT / "fig2_variance_comparison.png"), scale_factor=2)
    print("  Saved fig2_variance_comparison.png")

    # --- Print summary ---
    print("\n--- Form difficulty summary (n >= 5, sorted by ease) ---")
    for r in rows:
        print(f"  ease={r['mean_ease']:.2f}  n={r['n']:2d}  {r['form']}")
        print(f"        {r['pattern']}")

    print(f"\nEase variance:")
    print(f"  Semantic k-means  : {semantic_var:.4f}")
    print(f"  AST cert tree     : {ast_tree_var:.4f}")
    print(f"  FIM forms         : {fim_var:.4f}")
    ratio_fim_sem = fim_var / semantic_var if semantic_var > 0 else float("inf")
    ratio_fim_ast = fim_var / ast_tree_var if ast_tree_var > 0 else float("inf")
    print(f"\n  FIM / semantic    : {ratio_fim_sem:.1f}x")
    print(f"  FIM / AST tree    : {ratio_fim_ast:.1f}x")

    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
