"""Scaffold-property decomposition: which scaffold properties co-vary with
behavioral divergence?

Goal (L2 -> L3 on the mechanistic-depth ladder): treat scaffold not as a
black box but as a multi-axis variable. Each scaffold gets values along
6 design dimensions. Then: for each pair of scaffolds, correlate property-
difference (Hamming distance) against between-scaffold mean pair-JSD.

This is observational not causal — we have 4 scaffolds, not a controlled
factorial — but it surfaces which axis explains the most divergence.

Reads:
    output/paper2_pilot/jsd_matrix_extended.json
Writes:
    output/paper2_pilot/scaffold_property_decomposition.json
    output/figures/fig_scaffold_property_decomposition.png

Usage:
    uv run python scripts/analysis/scaffold_property_decomposition.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, MAGENTA, OLIVE, COPPER, GREEN
register()

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"
JSD_FILE = OUT_DAT / "jsd_matrix_extended.json"

# --- Scaffold design properties (manual assignment from public docs/configs) ---
# Each property is binary 0/1 or ordinal {0, 1, 2}. Six dimensions chosen to
# span: control flow, tool affordance, memory, search, deliberation budget.
SCAFFOLD_PROPERTIES: dict[str, dict[str, int]] = {
    "SWE-agent": {
        "iterative_loop":      1,  # multi-step ReAct loop
        "tool_count_high":     1,  # ~15+ tools (file_editor, search, navigate, etc.)
        "step_cap_high":       1,  # default 50, often reaches 100+
        "structured_phases":   0,  # free-form ReAct, no explicit phases
        "search_tool_explicit": 1,  # has explicit `search` tool
        "rollback_on_failure": 0,  # no rollback / re-plan on failure
    },
    "Agentless": {
        "iterative_loop":      0,  # deterministic pipeline (one-shot per stage)
        "tool_count_high":     0,  # narrow tool set
        "step_cap_high":       0,  # short fixed number of stages
        "structured_phases":   2,  # explicit localization → repair → validation phases
        "search_tool_explicit": 0,  # no tool-style search; LLM-guided file selection
        "rollback_on_failure": 0,  # no rollback
    },
    "DARS": {
        "iterative_loop":      1,  # tree-search exploration
        "tool_count_high":     0,  # smaller tool set than SWE-agent
        "step_cap_high":       1,  # extended exploration budget for R1
        "structured_phases":   1,  # implicit phases (search / select / repair)
        "search_tool_explicit": 1,  # knowledge-graph search subsystem
        "rollback_on_failure": 1,  # tree-search rolls back from failed branches
    },
    "Moatless": {
        "iterative_loop":      1,  # tree of action_steps
        "tool_count_high":     0,  # narrow action vocabulary
        "step_cap_high":       0,  # tighter step budget than SWE-agent (~10-15)
        "structured_phases":   2,  # explicit search/select/edit/test action types
        "search_tool_explicit": 1,  # tree-search backed by codebase index
        "rollback_on_failure": 1,  # tree backtracks on failure
    },
}

PROPERTY_LABELS = {
    "iterative_loop":       "iterative loop",
    "tool_count_high":      "high tool count",
    "step_cap_high":        "high step cap",
    "structured_phases":    "structured phases",
    "search_tool_explicit": "explicit search tool",
    "rollback_on_failure":  "rollback on failure",
}

# Map agent -> scaffold (for between-scaffold pair construction)
AGENT_SCAFFOLD = {
    "Claude-3":              "SWE-agent",
    "Claude-3.5":            "SWE-agent",
    "Claude-3.7-thinking":   "SWE-agent",
    "Claude-4":              "SWE-agent",
    "GPT-4":                 "SWE-agent",
    "GPT-4o":                "SWE-agent",
    "Agentless+Claude-3.5":  "Agentless",
    "DARS+R1":               "DARS",
    "Moatless+V3":           "Moatless",
}


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    jsd = json.loads(JSD_FILE.read_text())
    matrix_rows = jsd.get("matrix", [])

    # Build pairwise mean JSD per (scaffold_a, scaffold_b)
    pair_jsds: dict[tuple[str, str], list[float]] = {}
    for row in matrix_rows:
        a, b, v = row["row"], row["col"], row["jsd"]
        if a >= b:
            continue
        sa, sb = AGENT_SCAFFOLD.get(a), AGENT_SCAFFOLD.get(b)
        if sa is None or sb is None or sa == sb:
            continue
        key = tuple(sorted([sa, sb]))
        pair_jsds.setdefault(key, []).append(v)

    pair_summary = []
    for (sa, sb), vs in sorted(pair_jsds.items()):
        props_a = SCAFFOLD_PROPERTIES[sa]
        props_b = SCAFFOLD_PROPERTIES[sb]
        prop_diffs = {p: abs(props_a[p] - props_b[p]) for p in props_a}
        hamming = sum(1 for v in prop_diffs.values() if v > 0)
        manhattan = sum(prop_diffs.values())
        pair_summary.append({
            "pair": f"{sa} × {sb}",
            "scaffold_a": sa,
            "scaffold_b": sb,
            "mean_jsd": round(float(np.mean(vs)), 3),
            "max_jsd": round(float(np.max(vs)), 3),
            "n_pairs": len(vs),
            "hamming_distance": hamming,
            "manhattan_distance": manhattan,
            **{f"diff_{p}": prop_diffs[p] for p in prop_diffs},
        })

    df = pd.DataFrame(pair_summary)
    print("=== between-scaffold JSD vs property distance ===")
    print(df.to_string(index=False))

    # Correlate property distance with mean JSD
    rho_h, p_h = spearmanr(df["hamming_distance"], df["mean_jsd"])
    rho_m, p_m = spearmanr(df["manhattan_distance"], df["mean_jsd"])
    print(f"\nSpearman ρ(Hamming, mean JSD)   = {rho_h:+.3f}  (p = {p_h:.4f})")
    print(f"Spearman ρ(Manhattan, mean JSD) = {rho_m:+.3f}  (p = {p_m:.4f})")

    # Per-property: when this property differs, what's the mean JSD vs when it doesn't?
    per_property = []
    for p in SCAFFOLD_PROPERTIES["SWE-agent"]:
        diff_col = f"diff_{p}"
        differ = df[df[diff_col] > 0]["mean_jsd"]
        same   = df[df[diff_col] == 0]["mean_jsd"]
        per_property.append({
            "property": p,
            "label": PROPERTY_LABELS[p],
            "n_pairs_differ": int(len(differ)),
            "n_pairs_same":   int(len(same)),
            "mean_jsd_differ": round(float(differ.mean()), 3) if len(differ) else None,
            "mean_jsd_same":   round(float(same.mean()), 3)   if len(same)   else None,
            "diff": round(float(differ.mean()) - float(same.mean()), 3)
                    if len(differ) and len(same) else None,
        })
    pp_df = pd.DataFrame(per_property).sort_values(
        "diff", ascending=False, na_position="last",
    )
    print("\n=== per-property JSD-shift when property differs ===")
    print(pp_df.to_string(index=False))

    payload = {
        "scaffold_properties": SCAFFOLD_PROPERTIES,
        "between_scaffold_pairs": pair_summary,
        "property_distance_correlation": {
            "spearman_hamming": float(rho_h),
            "spearman_hamming_p": float(p_h),
            "spearman_manhattan": float(rho_m),
            "spearman_manhattan_p": float(p_m),
        },
        "per_property_jsd_shift": pp_df.to_dict(orient="records"),
        "interpretation": (
            "With 4 scaffolds (n=6 pairs), Spearman correlations are observational not causal. "
            "Properties whose variation co-varies most with cross-scaffold JSD are surfaced as "
            "candidate mechanism-load-bearing axes; controlled scaffold-ablation (future work) "
            "would tighten."
        ),
    }
    out_json = OUT_DAT / "scaffold_property_decomposition.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nSaved {out_json}")

    # Figure: dumbbell of per-property mean JSD when differs vs when same.
    # Each property gets two dots (differs, same) connected by a rule that
    # makes the gap visually load-bearing. Alternating row backgrounds
    # provide horizontal scan-tracking across the chart.

    fig_rows = []
    for r in per_property:
        if r["mean_jsd_differ"] is None or r["mean_jsd_same"] is None:
            continue
        fig_rows.append({"property": r["label"], "condition": "differs",
                         "mean_jsd": r["mean_jsd_differ"], "n": r["n_pairs_differ"]})
        fig_rows.append({"property": r["label"], "condition": "same",
                         "mean_jsd": r["mean_jsd_same"], "n": r["n_pairs_same"]})
    fig_df = pd.DataFrame(fig_rows)

    sort_order = (
        pp_df.dropna(subset=["diff"]).sort_values("diff", ascending=False)["label"].tolist()
    )

    # Wide-form data for the connecting rule between the two dots.
    rule_rows = []
    for r in per_property:
        if r["mean_jsd_differ"] is None or r["mean_jsd_same"] is None:
            continue
        rule_rows.append({
            "property": r["label"],
            "x_lo": min(r["mean_jsd_differ"], r["mean_jsd_same"]),
            "x_hi": max(r["mean_jsd_differ"], r["mean_jsd_same"]),
        })
    rule_df = pd.DataFrame(rule_rows)

    # Alternating row-band data: only emit the shaded rows so the band
    # layer does not have a color encoding that competes with the dot
    # layer's "differs vs same" color scale.
    band_df = pd.DataFrame([
        {"property": prop} for i, prop in enumerate(sort_order) if i % 2 == 0
    ])

    y_axis = alt.Axis(
        title=None, domain=False, ticks=False,
        labelFontSize=10, labelLimit=200, labelPadding=8,
    )

    bands = (
        alt.Chart(band_df)
        .mark_rect(fill="#F1F1EE", opacity=1.0, stroke=None)
        .encode(
            y=alt.Y("property:N", sort=sort_order, axis=y_axis),
        )
    )
    connecting = (
        alt.Chart(rule_df)
        .mark_rule(color="#666666", strokeWidth=1.4, opacity=0.7)
        .encode(
            x=alt.X("x_lo:Q", scale=alt.Scale(domain=[0, 1])),
            x2="x_hi:Q",
            y=alt.Y("property:N", sort=sort_order, axis=y_axis),
        )
    )
    dots = (
        alt.Chart(fig_df)
        .mark_circle(size=180, opacity=1.0, strokeWidth=0)
        .encode(
            x=alt.X("mean_jsd:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Mean cross-scaffold pair JSD",
                                  domain=False, ticks=False, labelFontSize=10)),
            y=alt.Y("property:N", sort=sort_order, axis=y_axis),
            color=alt.Color(
                "condition:N",
                scale=alt.Scale(domain=["differs", "same"], range=[MAGENTA, OLIVE]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["property", "condition", "mean_jsd", "n"],
        )
    )

    chart = (
        alt.layer(bands, connecting, dots)
        .resolve_scale(color="independent")
        .properties(
            width=420, height=240,
            title=alt.TitleParams(
                text="Mean cross-scaffold JSD when property differs vs same",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    out_png = OUT_FIG / "fig_scaffold_property_decomposition.png"
    chart.save(str(out_png), scale_factor=2)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
