#!/usr/bin/env python3
"""
Scoped certificate figures: 4 Altair charts decomposing agent alignment.

  Fig A: File navigation accuracy (horizontal bars per agent)
  Fig B: Scope agreement decomposition (grouped bars per agent pair)
  Fig C: Patch minimality vs scope accuracy (scatter, faceted by agent)
  Fig D: Enhanced instance anatomy (edit ops colored by oracle alignment)

Requires output from:
  - scripts/build_scoped_certificates.py  (oracle_scoped_certs.json)
  - scripts/build_agent_scoped_certs.py   (agent_scoped_certs.json, oracle_alignment.json)

Usage:
    uv run python scripts/scoped_figures.py
"""

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.theme import (
    register,
    BLUE, ORANGE, GREEN, VERMILLION, SKY, PINK, GRAY, NEAR_BLACK,
)
register()

RED = VERMILLION  # backward-compat alias

OUTPUT_DIR = ROOT / "output" / "scoped_certificates"
ALIGNMENT_DIR = ROOT / "output" / "pairwise_agent_comparison"


def _configure(chart: alt.Chart) -> alt.Chart:
    """Apply common styling."""
    return chart.configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
        titleFontWeight="normal",
    ).configure_legend(
        labelFontSize=9,
        titleFontSize=10,
        titleFontWeight="normal",
    ).configure_title(
        fontSize=11,
        fontWeight="normal",
    ).configure_view(strokeWidth=0)


def _layered_similarity(cert_a: dict, cert_b: dict) -> dict:
    """Compute similarity at file, scope, and edit-type levels.

    Inlined from scoped_edit_ops to avoid triggering the heavy __init__.py
    import chain in analysis.procedures.
    """
    files_a = set(cert_a.get("file_paths") or [cert_a.get("file_path", "")])
    files_b = set(cert_b.get("file_paths") or [cert_b.get("file_path", "")])
    file_match = bool(files_a & files_b)

    scopes_a = set(cert_a.get("scopes_touched", []))
    scopes_b = set(cert_b.get("scopes_touched", []))
    scope_overlap = scopes_a & scopes_b
    scope_union = scopes_a | scopes_b
    scope_jaccard = len(scope_overlap) / len(scope_union) if scope_union else 0.0

    edits_a = set(cert_a.get("edit_cert", []))
    edits_b = set(cert_b.get("edit_cert", []))
    edit_union = edits_a | edits_b
    edit_jaccard = len(edits_a & edits_b) / len(edit_union) if edit_union else 0.0

    return {
        "file_match": file_match,
        "scope_jaccard": scope_jaccard,
        "edit_jaccard": edit_jaccard,
        "scope_overlap_count": len(scope_overlap),
    }


def _short(name: str) -> str:
    """Shorten agent names for display."""
    return (name
            .replace("SWE-agent ", "")
            .replace("Claude 3.5 Sonnet", "C3.5S")
            .replace("Claude 3.5", "C3.5")
            .replace("Claude 3 Opus", "C3O")
            .replace("GPT-4o", "G4o")
            .replace("GPT-4", "G4"))


def fig_a_file_navigation(alignment: list[dict]) -> alt.Chart:
    """Fig A: File navigation accuracy per agent.

    Horizontal stacked bar: correct file only, correct + extra, wrong file.
    """
    rows = []
    for agent in sorted(set(r["agent"] for r in alignment)):
        recs = [r for r in alignment if r["agent"] == agent]
        n = len(recs)
        cats = Counter(r["file_category"] for r in recs)
        for cat, label in [
            ("correct_only", "Correct file only"),
            ("correct_plus_extra", "Correct + extra files"),
            ("wrong_file", "Wrong file"),
        ]:
            count = cats.get(cat, 0)
            rows.append({
                "agent": agent,
                "category": label,
                "count": count,
                "fraction": count / n if n > 0 else 0,
                "n": n,
            })

    df = pd.DataFrame(rows)

    cat_order = ["Correct file only", "Correct + extra files", "Wrong file"]
    color_scale = alt.Scale(
        domain=cat_order,
        range=[GREEN, ORANGE, RED],
    )

    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y("agent:N", title=None,
                 sort=alt.EncodingSortField(field="fraction", order="descending")),
        x=alt.X("fraction:Q", title="Fraction of instances",
                 scale=alt.Scale(domain=[0, 1]),
                 axis=alt.Axis(format=".0%")),
        color=alt.Color("category:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom")),
        order=alt.Order("category:N", sort="ascending"),
        tooltip=["agent", "category",
                 alt.Tooltip("fraction:Q", format=".1%"), "count", "n"],
    ).properties(
        width=400,
        height=150,
        title="File navigation accuracy by agent",
    )

    # Count annotation to the right
    totals = df.groupby("agent")["n"].first().reset_index()
    text = alt.Chart(totals).mark_text(
        align="left", dx=5, fontSize=9, color="black",
    ).encode(
        y=alt.Y("agent:N",
                 sort=alt.EncodingSortField(field="n", order="descending")),
        x=alt.value(405),
        text=alt.Text("n:Q", format="d"),
    )

    chart = (bars + text)
    return _configure(chart)


def fig_b_scope_decomposition(alignment: list[dict]) -> alt.Chart:
    """Fig B: Scope agreement decomposition.

    For each agent pair with enough co-processed instances (n >= 10),
    show three grouped bars: file agreement, scope Jaccard, edit Jaccard.
    """
    agent_certs_path = ALIGNMENT_DIR / "agent_scoped_certs.json"

    with open(agent_certs_path) as f:
        agent_certs = json.load(f)

    agents = sorted(agent_certs.keys())
    rows = []

    for a1, a2 in combinations(agents, 2):
        certs1 = agent_certs[a1]
        certs2 = agent_certs[a2]
        co_instances = set(certs1.keys()) & set(certs2.keys())

        if len(co_instances) < 10:
            continue

        file_matches = []
        scope_jaccards = []
        edit_jaccards = []

        for iid in co_instances:
            sim = _layered_similarity(certs1[iid], certs2[iid])
            file_matches.append(float(sim["file_match"]))
            scope_jaccards.append(sim["scope_jaccard"])
            edit_jaccards.append(sim["edit_jaccard"])

        pair_label = f"{_short(a1)} / {_short(a2)}"
        for metric, values in [
            ("File agreement", file_matches),
            ("Scope Jaccard", scope_jaccards),
            ("Edit Jaccard", edit_jaccards),
        ]:
            rows.append({
                "pair": pair_label,
                "metric": metric,
                "value": np.mean(values),
                "n": len(co_instances),
            })

    if not rows:
        print("  No agent pairs with >= 10 co-processed instances for Fig B")
        return None

    df = pd.DataFrame(rows)

    metric_order = ["File agreement", "Scope Jaccard", "Edit Jaccard"]
    color_scale = alt.Scale(
        domain=metric_order,
        range=[BLUE, GREEN, ORANGE],
    )

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("pair:N", title=None,
                 axis=alt.Axis(labelAngle=-30, labelFontSize=8)),
        y=alt.Y("value:Q", title="Mean agreement",
                 scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom")),
        xOffset="metric:N",
        tooltip=["pair", "metric",
                 alt.Tooltip("value:Q", format=".3f"), "n"],
    ).properties(
        width=400,
        height=250,
        title="Agent pair agreement decomposed by file, scope, and edit type",
    )

    return _configure(chart)


def fig_c_minimality_vs_scope(alignment: list[dict]) -> alt.Chart:
    """Fig C: Patch minimality vs scope accuracy.

    Scatter: x = agent/oracle patch size (log), y = scope overlap.
    Color by resolved. Faceted by agent.
    """
    rows = []
    for r in alignment:
        if r["patch_minimality"] <= 0 or r["patch_minimality"] == float("inf"):
            continue
        rows.append({
            "agent": r["agent"],
            "instance_id": r["instance_id"],
            "minimality": r["patch_minimality"],
            "scope_jaccard": r["scope_jaccard"],
            "resolved": "resolved" if r["resolved"] else "failed",
            "agent_size": r["agent_patch_size"],
            "oracle_size": r["oracle_patch_size"],
            "ref_x": 1.0,  # reference line at oracle-size parity
        })

    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=["resolved", "failed"],
        range=[BLUE, GRAY],
    )

    points = alt.Chart(df).mark_circle(size=30, opacity=0.6).encode(
        x=alt.X("minimality:Q", title="Patch size ratio (agent / oracle)",
                 scale=alt.Scale(type="log", domain=[0.1, 100]),
                 axis=alt.Axis(format=".1f")),
        y=alt.Y("scope_jaccard:Q", title="Scope overlap (Jaccard)",
                 scale=alt.Scale(domain=[-0.02, 1.02])),
        color=alt.Color("resolved:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["instance_id", "agent",
                 alt.Tooltip("minimality:Q", format=".2f"),
                 alt.Tooltip("scope_jaccard:Q", format=".3f"),
                 "resolved"],
    )

    rule = alt.Chart(df).mark_rule(
        strokeDash=[4, 4], strokeWidth=1, color=RED,
    ).encode(x="ref_x:Q")

    combined = (points + rule).properties(
        width=200,
        height=180,
    ).facet(
        column=alt.Column("agent:N", title=None,
                          header=alt.Header(labelFontSize=9)),
    ).properties(
        title="Patch minimality vs scope accuracy",
    )

    return _configure(combined)


def fig_d_instance_anatomy(alignment: list[dict]) -> alt.Chart:
    """Fig D: Enhanced instance anatomy.

    For a sample of instances, show edit operations colored by oracle alignment.
    Green = in oracle cert, orange = agent-only. Include file match indicator.
    """
    # Pick instances that have data from multiple agents
    instance_agents = defaultdict(list)
    for r in alignment:
        instance_agents[r["instance_id"]].append(r)

    # Select instances with >= 3 agents and mixed file match results
    multi_agent = {iid: recs for iid, recs in instance_agents.items()
                   if len(recs) >= 3}

    # Sort by scope jaccard variance to get interesting cases
    scored = []
    for iid, recs in multi_agent.items():
        jaccards = [r["scope_jaccard"] for r in recs]
        scored.append((iid, np.std(jaccards), recs))
    scored.sort(key=lambda x: -x[1])

    # Take top 8 instances
    sample = scored[:8]

    # Load oracle certs for edit_cert info (once)
    with open(OUTPUT_DIR / "oracle_scoped_certs.json") as f:
        oracle_certs_list = json.load(f)
    oracle_certs = {c["instance_id"]: set(c["edit_cert"]) for c in oracle_certs_list}

    # Load agent certs (once)
    with open(ALIGNMENT_DIR / "agent_scoped_certs.json") as f:
        agent_certs_data = json.load(f)

    rows = []
    for iid, _, recs in sample:
        oracle_ops = oracle_certs.get(iid, set())
        # Truncate instance_id for display
        display_id = iid.split("__")[-1] if "__" in iid else iid[:25]
        for r in recs:
            agent_short = _short(r["agent"])
            agent_cert = agent_certs_data.get(r["agent"], {}).get(iid, {})
            agent_ops = set(agent_cert.get("edit_cert", []))

            shared = agent_ops & oracle_ops
            agent_only = agent_ops - oracle_ops

            file_label = "Y" if r["file_match"] else "N"

            for op in sorted(shared):
                rows.append({
                    "instance": display_id,
                    "agent": f"{agent_short} [{file_label}]",
                    "op": op.replace("_", " "),
                    "alignment": "In oracle cert",
                })
            for op in sorted(agent_only):
                rows.append({
                    "instance": display_id,
                    "agent": f"{agent_short} [{file_label}]",
                    "op": op.replace("_", " "),
                    "alignment": "Agent only",
                })

    if not rows:
        print("  No data for Fig D")
        return None

    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=["In oracle cert", "Agent only"],
        range=[GREEN, ORANGE],
    )

    ops_chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("op:N", title="Edit operation",
                 axis=alt.Axis(labelAngle=-45, labelFontSize=7)),
        y=alt.Y("agent:N", title=None,
                 axis=alt.Axis(labelFontSize=7)),
        color=alt.Color("alignment:N", scale=color_scale,
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["instance", "agent", "op", "alignment"],
    ).properties(
        width=350,
        height=70,
    ).facet(
        row=alt.Row("instance:N", title=None,
                     header=alt.Header(labelFontSize=8, labelAngle=0,
                                       labelAlign="left")),
    ).properties(
        title="Instance anatomy, edit ops colored by oracle alignment (file match Y/N in agent label)",
    )

    return _configure(ops_chart)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load alignment data
    alignment_path = ALIGNMENT_DIR / "oracle_alignment.json"
    if not alignment_path.exists():
        print(f"Missing {alignment_path}. Run build_agent_scoped_certs.py first.")
        sys.exit(1)

    with open(alignment_path) as f:
        alignment = json.load(f)
    print(f"Loaded {len(alignment)} alignment records")

    # Fig A
    print("\nBuilding Fig A: File navigation accuracy...")
    chart_a = fig_a_file_navigation(alignment)
    out_a = OUTPUT_DIR / "fig_a_file_navigation.png"
    chart_a.save(str(out_a), scale_factor=2)
    print(f"  Saved {out_a.relative_to(ROOT)}")

    # Fig B
    print("\nBuilding Fig B: Scope agreement decomposition...")
    chart_b = fig_b_scope_decomposition(alignment)
    if chart_b is not None:
        out_b = OUTPUT_DIR / "fig_b_scope_decomposition.png"
        chart_b.save(str(out_b), scale_factor=2)
        print(f"  Saved {out_b.relative_to(ROOT)}")

    # Fig C
    print("\nBuilding Fig C: Patch minimality vs scope accuracy...")
    chart_c = fig_c_minimality_vs_scope(alignment)
    out_c = OUTPUT_DIR / "fig_c_minimality_vs_scope.png"
    chart_c.save(str(out_c), scale_factor=2)
    print(f"  Saved {out_c.relative_to(ROOT)}")

    # Fig D
    print("\nBuilding Fig D: Instance anatomy...")
    chart_d = fig_d_instance_anatomy(alignment)
    if chart_d is not None:
        out_d = OUTPUT_DIR / "fig_d_instance_anatomy.png"
        chart_d.save(str(out_d), scale_factor=2)
        print(f"  Saved {out_d.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
