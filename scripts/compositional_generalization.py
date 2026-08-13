#!/usr/bin/env python3
"""
Compositional generalization analysis.

Do agents fail because they lack specific edit primitives, or because they
can't combine primitives they've individually demonstrated?

For each of the 84 agents:
  1. Build a primitive library (union of edit types from solved instances)
  2. Build a combination library (set of oracle certs from solved instances)
  3. Classify each unsolved instance as:
     - novel_primitive: oracle cert contains an op the agent has never produced
     - novel_composition: all ops are in library, but the exact combo is new
     - familiar: the oracle cert (or a superset) appears in the agent's solved set

Outputs:
  agent_libraries.json          -- per-agent primitive library stats
  instance_classification.json  -- per-instance, per-agent classification
  composition_gap.json          -- per-instance composition gap fraction
  summary_stats.json            -- aggregate numbers

Figures (Altair, Wong palette):
  fig1_failure_classification.png
  fig2_composition_gap_vs_ease.png
  fig3_primitive_freq_vs_ease.png

Usage:
  uv run python scripts/compositional_generalization.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import altair as alt
import msgpack
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, VERMILLION, PINK, GRAY
register()

from scripts.build_canonical_forms import load_certs, _NORMALIZE_OPS

OUT = ROOT / "output" / "compositional_generalization"
OUT.mkdir(parents=True, exist_ok=True)


# --- Data loading ---


def load_leaderboard(path: Path) -> dict[str, dict[str, bool]]:
    with open(path, "rb") as f:
        lb = msgpack.unpack(f, raw=False)
    return lb


def load_ease(lb: dict[str, dict[str, bool]]) -> dict[str, float]:
    instance_votes: dict[str, list[bool]] = {}
    for agent_data in lb.values():
        for iid, passed in agent_data.items():
            instance_votes.setdefault(iid, []).append(passed)
    return {iid: float(np.mean(v)) for iid, v in instance_votes.items()}


# --- Classification ---


def classify_failure(
    oracle_cert: frozenset[str],
    primitive_library: set[str],
    combination_library: set[frozenset[str]],
) -> str:
    """Classify why an agent failed on an instance.

    Returns one of: novel_primitive, novel_composition, familiar.
    """
    missing_ops = oracle_cert - primitive_library
    if missing_ops:
        return "novel_primitive"

    # Agent has all primitives. Check if it has demonstrated
    # the exact combination or a superset of it.
    for solved_cert in combination_library:
        if oracle_cert.issubset(solved_cert):
            return "familiar"

    return "novel_composition"


def build_agent_libraries(
    lb: dict[str, dict[str, bool]],
    certs: dict[str, frozenset[str]],
) -> dict[str, dict]:
    """Build primitive and combination libraries for each agent."""
    libraries = {}

    for agent, results in lb.items():
        solved_ids = [iid for iid, passed in results.items() if passed]
        solved_certs = [certs[iid] for iid in solved_ids if iid in certs]

        primitive_library = set()
        for cert in solved_certs:
            primitive_library.update(cert)

        combination_library = set(solved_certs)

        libraries[agent] = {
            "primitive_library": primitive_library,
            "combination_library": combination_library,
            "n_solved": len(solved_ids),
            "n_solved_with_cert": len(solved_certs),
            "n_unique_certs": len(combination_library),
            "n_primitives": len(primitive_library),
        }

    return libraries


def classify_all(
    lb: dict[str, dict[str, bool]],
    certs: dict[str, frozenset[str]],
    libraries: dict[str, dict],
) -> dict[str, dict[str, str]]:
    """Classify every (agent, unsolved instance) pair."""
    classifications: dict[str, dict[str, str]] = {}

    for agent, results in lb.items():
        lib = libraries[agent]
        agent_classes: dict[str, str] = {}

        for iid, passed in results.items():
            if passed:
                continue
            if iid not in certs:
                continue
            agent_classes[iid] = classify_failure(
                certs[iid],
                lib["primitive_library"],
                lib["combination_library"],
            )

        classifications[agent] = agent_classes

    return classifications


def compute_composition_gap(
    lb: dict[str, dict[str, bool]],
    certs: dict[str, frozenset[str]],
    libraries: dict[str, dict],
    ease: dict[str, float],
) -> list[dict]:
    """For each instance, compute how many failing agents have all primitives."""
    all_instances = set()
    for results in lb.values():
        all_instances.update(results.keys())

    rows = []
    for iid in sorted(all_instances):
        if iid not in certs:
            continue

        oracle_cert = certs[iid]

        n_fail = 0
        n_fail_have_primitives = 0

        for agent, results in lb.items():
            if iid not in results:
                continue
            if results[iid]:
                continue

            n_fail += 1
            missing = oracle_cert - libraries[agent]["primitive_library"]
            if not missing:
                n_fail_have_primitives += 1

        gap_fraction = n_fail_have_primitives / n_fail if n_fail > 0 else 0.0

        # Min primitive frequency: for each op in the cert, what fraction
        # of agents have it in their library?
        n_agents = len(lb)
        op_freqs = []
        for op in oracle_cert:
            count = sum(
                1 for lib in libraries.values()
                if op in lib["primitive_library"]
            )
            op_freqs.append(count / n_agents)

        min_prim_freq = min(op_freqs) if op_freqs else 0.0
        mean_prim_freq = float(np.mean(op_freqs)) if op_freqs else 0.0

        rows.append({
            "instance_id": iid,
            "ease": ease.get(iid, 0.0),
            "oracle_cert": sorted(oracle_cert),
            "cert_size": len(oracle_cert),
            "n_fail": n_fail,
            "n_fail_have_primitives": n_fail_have_primitives,
            "composition_gap_fraction": gap_fraction,
            "min_primitive_freq": min_prim_freq,
            "mean_primitive_freq": mean_prim_freq,
        })

    return rows


# --- Figures ---


def fig1_failure_classification(
    classifications: dict[str, dict[str, str]],
    out_path: Path,
):
    """Stacked bar: aggregate failure classification across all agents."""
    # Aggregate across all agents
    total = Counter()
    for agent_classes in classifications.values():
        total.update(agent_classes.values())

    grand_total = sum(total.values())
    print(f"\n  Aggregate failure classification ({grand_total} agent-instance failures)")
    for cat in ["novel_primitive", "novel_composition", "familiar"]:
        n = total.get(cat, 0)
        pct = 100 * n / grand_total if grand_total else 0
        print(f"    {cat}: {n} ({pct:.1f}%)")

    # Per-agent breakdown for the chart
    rows = []
    for agent, agent_classes in classifications.items():
        agent_total = len(agent_classes)
        if agent_total == 0:
            continue
        counts = Counter(agent_classes.values())
        for cat in ["novel_primitive", "novel_composition", "familiar"]:
            rows.append({
                "agent": agent,
                "category": cat,
                "count": counts.get(cat, 0),
                "fraction": counts.get(cat, 0) / agent_total,
            })

    df = pd.DataFrame(rows)

    # Compute mean fraction per category for ordering
    mean_fracs = df.groupby("category")["fraction"].mean().to_dict()

    # Sort agents by novel_composition fraction (most interesting)
    agent_nc = (
        df[df["category"] == "novel_composition"]
        .set_index("agent")["fraction"]
        .sort_values(ascending=False)
    )
    agent_order = agent_nc.index.tolist()

    # Aggregate bar chart (mean across agents)
    agg_rows = []
    for cat in ["novel_primitive", "novel_composition", "familiar"]:
        vals = df[df["category"] == cat]["fraction"]
        agg_rows.append({
            "category": cat,
            "mean_fraction": vals.mean(),
            "std_fraction": vals.std(),
        })
    agg_df = pd.DataFrame(agg_rows)

    color_map = {
        "novel_primitive": BLUE,
        "novel_composition": ORANGE,
        "familiar": GREEN,
    }
    cat_order = ["novel_primitive", "novel_composition", "familiar"]
    cat_labels = {
        "novel_primitive": "Novel primitive",
        "novel_composition": "Novel composition",
        "familiar": "Familiar",
    }

    agg_df["category_label"] = agg_df["category"].map(cat_labels)
    label_order = [cat_labels[c] for c in cat_order]

    bars = alt.Chart(agg_df).mark_bar().encode(
        x=alt.X(
            "category_label:N",
            sort=label_order,
            axis=alt.Axis(title=None, labelFontSize=10),
        ),
        y=alt.Y(
            "mean_fraction:Q",
            axis=alt.Axis(
                title="Mean fraction of failures",
                titleFontSize=10,
                format=".0%",
            ),
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color(
            "category_label:N",
            scale=alt.Scale(
                domain=label_order,
                range=[color_map[c] for c in cat_order],
            ),
            legend=None,
        ),
    )

    labels = alt.Chart(agg_df).mark_text(
        dy=-8, fontSize=11
    ).encode(
        x=alt.X("category_label:N", sort=label_order),
        y=alt.Y("mean_fraction:Q"),
        text=alt.Text("mean_fraction:Q", format=".1%"),
    )

    fig = (bars + labels).properties(
        width=350,
        height=300,
        title=alt.TitleParams(
            "Failure classification, mean across 84 agents",
            fontSize=12,
            fontWeight="normal",
            anchor="start",
        ),
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(out_path), scale_factor=2)
    print(f"  Saved {out_path.name}")


def fig2_composition_gap_vs_ease(
    gap_data: list[dict],
    out_path: Path,
):
    """Scatter: x = ease, y = composition gap fraction."""
    df = pd.DataFrame(gap_data)

    points = alt.Chart(df).mark_circle(
        size=30,
        opacity=0.6,
    ).encode(
        x=alt.X(
            "ease:Q",
            axis=alt.Axis(
                title="Ease (fraction of 84 agents solving)",
                titleFontSize=10,
            ),
            scale=alt.Scale(domain=[0, 1]),
        ),
        y=alt.Y(
            "composition_gap_fraction:Q",
            axis=alt.Axis(
                title="Composition gap (fraction of failing agents with all primitives)",
                titleFontSize=10,
            ),
            scale=alt.Scale(domain=[0, 1.05]),
        ),
        color=alt.value(BLUE),
        tooltip=["instance_id:N", "ease:Q", "composition_gap_fraction:Q", "cert_size:Q"],
    )

    # Highlight the hard instances (ease < 0.05) with high gap
    hard = df[(df["ease"] < 0.05) & (df["composition_gap_fraction"] > 0.5)]
    if not hard.empty:
        highlight = alt.Chart(hard).mark_circle(
            size=60,
            opacity=0.9,
            stroke=ORANGE,
            strokeWidth=1.5,
        ).encode(
            x="ease:Q",
            y="composition_gap_fraction:Q",
            color=alt.value(ORANGE),
            tooltip=["instance_id:N", "ease:Q", "composition_gap_fraction:Q"],
        )
        chart = points + highlight
    else:
        chart = points

    fig = chart.properties(
        width=500,
        height=350,
        title=alt.TitleParams(
            "Composition gap vs instance ease",
            fontSize=12,
            fontWeight="normal",
            anchor="start",
        ),
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(out_path), scale_factor=2)
    print(f"  Saved {out_path.name}")


def fig3_primitive_freq_vs_ease(
    gap_data: list[dict],
    out_path: Path,
):
    """Scatter: x = min primitive frequency, y = ease."""
    df = pd.DataFrame(gap_data)

    points = alt.Chart(df).mark_circle(
        size=30,
        opacity=0.6,
    ).encode(
        x=alt.X(
            "min_primitive_freq:Q",
            axis=alt.Axis(
                title="Min primitive frequency (fraction of agents with rarest op)",
                titleFontSize=10,
            ),
            scale=alt.Scale(domain=[0, 1.05]),
        ),
        y=alt.Y(
            "ease:Q",
            axis=alt.Axis(
                title="Ease (fraction of 84 agents solving)",
                titleFontSize=10,
            ),
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.value(BLUE),
        tooltip=["instance_id:N", "min_primitive_freq:Q", "ease:Q", "cert_size:Q"],
    )

    # Highlight composition failures: common primitives but low ease
    comp_fail = df[(df["min_primitive_freq"] > 0.5) & (df["ease"] < 0.1)]
    if not comp_fail.empty:
        highlight = alt.Chart(comp_fail).mark_circle(
            size=60,
            opacity=0.9,
            stroke=ORANGE,
            strokeWidth=1.5,
        ).encode(
            x="min_primitive_freq:Q",
            y="ease:Q",
            color=alt.value(ORANGE),
            tooltip=["instance_id:N", "min_primitive_freq:Q", "ease:Q"],
        )
        chart = points + highlight
    else:
        chart = points

    fig = chart.properties(
        width=500,
        height=350,
        title=alt.TitleParams(
            "Primitive frequency vs ease",
            subtitle="Top-left quadrant shows composition failures (common parts, hard to combine)",
            fontSize=12,
            fontWeight="normal",
            anchor="start",
            subtitleFontSize=9,
        ),
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(out_path), scale_factor=2)
    print(f"  Saved {out_path.name}")


# --- Main ---


def main():
    print("Loading leaderboard data...")
    lb = load_leaderboard(ROOT / "output" / "leaderboard" / "lite_results.msgpack")
    print(f"  {len(lb)} agents")

    ease = load_ease(lb)
    all_instances = set(ease.keys())
    print(f"  {len(all_instances)} instances with ease scores")

    print("\nLoading edit certificates...")
    certs = load_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")
    print(f"  {len(certs)} instances with oracle certs")

    # How many leaderboard instances have certs?
    covered = all_instances & set(certs.keys())
    print(f"  {len(covered)} instances have both ease and certs")

    # --- Step 1-2: Build agent libraries ---
    print("\nBuilding agent primitive and combination libraries...")
    libraries = build_agent_libraries(lb, certs)

    lib_sizes = [lib["n_primitives"] for lib in libraries.values()]
    cert_counts = [lib["n_unique_certs"] for lib in libraries.values()]
    solved_counts = [lib["n_solved"] for lib in libraries.values()]
    print(f"  Primitive library size: mean={np.mean(lib_sizes):.1f}, "
          f"median={np.median(lib_sizes):.0f}, range=[{min(lib_sizes)}, {max(lib_sizes)}]")
    print(f"  Unique certs per agent: mean={np.mean(cert_counts):.1f}, "
          f"range=[{min(cert_counts)}, {max(cert_counts)}]")
    print(f"  Solved instances: mean={np.mean(solved_counts):.1f}, "
          f"range=[{min(solved_counts)}, {max(solved_counts)}]")

    # --- Step 3: Classify failures ---
    print("\nClassifying failures...")
    classifications = classify_all(lb, certs, libraries)

    total_classified = sum(len(v) for v in classifications.values())
    print(f"  {total_classified} (agent, instance) failures classified")

    # --- Step 4: Failure rates ---
    print("\nComputing failure rate breakdown...")
    all_counts = Counter()
    for agent_classes in classifications.values():
        all_counts.update(agent_classes.values())

    grand = sum(all_counts.values())
    print(f"\n  Overall breakdown ({grand} agent-instance failures):")
    for cat in ["novel_primitive", "novel_composition", "familiar"]:
        n = all_counts.get(cat, 0)
        pct = 100 * n / grand if grand else 0
        print(f"    {cat}: {n:,} ({pct:.1f}%)")

    # Hard instances breakdown (ease < 0.05)
    hard_ids = {iid for iid, e in ease.items() if e < 0.05}
    hard_counts = Counter()
    for agent_classes in classifications.values():
        for iid, cat in agent_classes.items():
            if iid in hard_ids:
                hard_counts[cat] += 1

    hard_total = sum(hard_counts.values())
    if hard_total > 0:
        print(f"\n  Hard instances (ease < 0.05, n={len(hard_ids)}, "
              f"{hard_total} agent-instance failures):")
        for cat in ["novel_primitive", "novel_composition", "familiar"]:
            n = hard_counts.get(cat, 0)
            pct = 100 * n / hard_total if hard_total else 0
            print(f"    {cat}: {n:,} ({pct:.1f}%)")

    # --- Step 5: Key test ---
    print("\n--- Key test: compositional generalization bottleneck ---")

    # Instances where ALL failing agents have the primitives but none solve
    print("\nComputing composition gap per instance...")
    gap_data = compute_composition_gap(lb, certs, libraries, ease)
    gap_df = pd.DataFrame(gap_data)

    high_gap = gap_df[gap_df["composition_gap_fraction"] > 0.8]
    print(f"  Instances with composition gap > 0.8: {len(high_gap)}/{len(gap_df)}")

    hard_high_gap = gap_df[
        (gap_df["ease"] < 0.05) & (gap_df["composition_gap_fraction"] > 0.8)
    ]
    print(f"  Hard instances (ease < 0.05) with gap > 0.8: {len(hard_high_gap)}")

    # Correlation between composition gap and ease
    if len(gap_df) > 2:
        corr = gap_df["ease"].corr(gap_df["composition_gap_fraction"])
        print(f"  Correlation(ease, composition_gap): {corr:.3f}")

    # Mean composition gap by ease quartile
    gap_df["ease_bin"] = pd.qcut(gap_df["ease"], 4, labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"])
    for label, grp in gap_df.groupby("ease_bin", observed=False):
        mean_gap = grp["composition_gap_fraction"].mean()
        print(f"    {label}: mean composition gap = {mean_gap:.3f} (n={len(grp)})")

    # --- Step 6: Primitive frequency analysis ---
    print("\nPrimitive frequency analysis...")

    # Global primitive frequencies
    all_primitives = Counter()
    for lib in libraries.values():
        all_primitives.update(lib["primitive_library"])

    n_agents = len(lb)
    print(f"\n  Top 15 primitives by agent coverage:")
    for op, count in all_primitives.most_common(15):
        print(f"    {op}: {count}/{n_agents} agents ({100*count/n_agents:.0f}%)")

    # Instances with common primitives but low ease
    comp_failures = gap_df[
        (gap_df["min_primitive_freq"] > 0.5) & (gap_df["ease"] < 0.1)
    ]
    print(f"\n  Composition failure instances (min_prim_freq > 0.5, ease < 0.1): "
          f"{len(comp_failures)}")
    if not comp_failures.empty:
        print(f"    Mean ease: {comp_failures['ease'].mean():.3f}")
        print(f"    Mean composition gap: {comp_failures['composition_gap_fraction'].mean():.3f}")
        print(f"    Mean cert size: {comp_failures['cert_size'].mean():.1f}")

    # --- Save outputs ---
    print("\nSaving outputs...")

    # agent_libraries.json
    agent_lib_out = {}
    for agent, lib in libraries.items():
        agent_lib_out[agent] = {
            "primitive_library": sorted(lib["primitive_library"]),
            "n_solved": lib["n_solved"],
            "n_solved_with_cert": lib["n_solved_with_cert"],
            "n_unique_certs": lib["n_unique_certs"],
            "n_primitives": lib["n_primitives"],
        }
    with open(OUT / "agent_libraries.json", "w") as f:
        json.dump(agent_lib_out, f, indent=2)
    print(f"  Saved agent_libraries.json ({len(agent_lib_out)} agents)")

    # instance_classification.json (compact: per-instance -> {agent: category})
    instance_classes: dict[str, dict[str, str]] = defaultdict(dict)
    for agent, agent_classes in classifications.items():
        for iid, cat in agent_classes.items():
            instance_classes[iid][agent] = cat

    with open(OUT / "instance_classification.json", "w") as f:
        json.dump(dict(instance_classes), f, indent=2)
    print(f"  Saved instance_classification.json ({len(instance_classes)} instances)")

    # composition_gap.json
    with open(OUT / "composition_gap.json", "w") as f:
        json.dump(gap_data, f, indent=2)
    print(f"  Saved composition_gap.json ({len(gap_data)} instances)")

    # summary_stats.json
    per_agent_fractions = []
    for agent, agent_classes in classifications.items():
        if not agent_classes:
            continue
        counts = Counter(agent_classes.values())
        total = sum(counts.values())
        per_agent_fractions.append({
            "agent": agent,
            "n_classified": total,
            "novel_primitive_frac": counts.get("novel_primitive", 0) / total,
            "novel_composition_frac": counts.get("novel_composition", 0) / total,
            "familiar_frac": counts.get("familiar", 0) / total,
        })

    paf_df = pd.DataFrame(per_agent_fractions)

    summary = {
        "n_agents": len(lb),
        "n_instances_with_certs": len(certs),
        "n_instances_in_leaderboard": len(all_instances),
        "n_covered": len(covered),
        "total_agent_instance_failures": grand,
        "overall_breakdown": {
            cat: {
                "count": all_counts.get(cat, 0),
                "fraction": all_counts.get(cat, 0) / grand if grand else 0,
            }
            for cat in ["novel_primitive", "novel_composition", "familiar"]
        },
        "hard_instances_breakdown": {
            "n_hard_instances": len(hard_ids),
            "total_failures": hard_total,
            "breakdown": {
                cat: {
                    "count": hard_counts.get(cat, 0),
                    "fraction": hard_counts.get(cat, 0) / hard_total if hard_total else 0,
                }
                for cat in ["novel_primitive", "novel_composition", "familiar"]
            },
        },
        "per_agent_mean_fractions": {
            "novel_primitive": float(paf_df["novel_primitive_frac"].mean()),
            "novel_composition": float(paf_df["novel_composition_frac"].mean()),
            "familiar": float(paf_df["familiar_frac"].mean()),
        },
        "composition_gap_stats": {
            "n_instances_gap_gt_0.8": int(len(high_gap)),
            "n_hard_instances_gap_gt_0.8": int(len(hard_high_gap)),
            "mean_gap_all": float(gap_df["composition_gap_fraction"].mean()),
            "mean_gap_hard": float(
                gap_df[gap_df["ease"] < 0.05]["composition_gap_fraction"].mean()
            ) if len(gap_df[gap_df["ease"] < 0.05]) > 0 else None,
        },
        "composition_failure_instances": {
            "n_common_prims_low_ease": int(len(comp_failures)),
            "mean_ease": float(comp_failures["ease"].mean()) if not comp_failures.empty else None,
            "mean_gap": float(comp_failures["composition_gap_fraction"].mean()) if not comp_failures.empty else None,
        },
    }

    with open(OUT / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary_stats.json")

    # --- Figures ---
    print("\nGenerating figures...")
    fig1_failure_classification(classifications, OUT / "fig1_failure_classification.png")
    fig2_composition_gap_vs_ease(gap_data, OUT / "fig2_composition_gap_vs_ease.png")
    fig3_primitive_freq_vs_ease(gap_data, OUT / "fig3_primitive_freq_vs_ease.png")

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
