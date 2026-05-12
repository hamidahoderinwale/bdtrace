"""Within-agent EDIT_SRC_PY vs SHELL_CD correlation in long Type B
failures, the Simpson's-paradox check on the aggregate r = -0.40.

Aggregate across the long-Type-B cohort: Pearson r = -0.40, suggesting
the two signatures are alternative failure modes. The within-agent
check tests whether the relationship is structural (negative within
every agent) or driven by between-agent positioning (near zero within
each agent; agents cluster in different quadrants of the share space).

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/extended_pass_fail.json
    output/resolved_traces_lite_full.jsonl
    output/trajectories/.cache/<sub>/<iid>.json
Writes:
    output/paper2_pilot/r10_within_agent_correlation.json
    output/figures/fig_postloc_within_agent_corr.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.preferences.localization import load_gold_files
from analysis.preferences.localization_extended import first_localization_step_extended
from scripts.theme import BLUE, COPPER, GREEN, MAGENTA, OLIVE, register

register()

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"
CACHE = ROOT / "output" / "trajectories" / ".cache"

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

AGENT_CELL = {
    "Claude-3":             "SWE-agent base",
    "Claude-3.5":           "SWE-agent base",
    "GPT-4":                "SWE-agent base",
    "GPT-4o":               "SWE-agent base",
    "Claude-3.7-thinking":  "SWE-agent extended-thinking",
    "Claude-4":             "SWE-agent extended-thinking",
    "Agentless+Claude-3.5": "Agentless",
    "DARS+R1":              "DARS",
    "Moatless+V3":          "Moatless",
}

CELL_COLOR = {
    "SWE-agent base":               BLUE,
    "SWE-agent extended-thinking":  GREEN,
    "Agentless":                    COPPER,
    "DARS":                         MAGENTA,
    "Moatless":                     OLIVE,
}


def share(atoms: list[str], target: str) -> float:
    return sum(1 for a in atoms if a == target) / len(atoms) if atoms else 0.0


def main() -> None:
    canonical_idx: dict[tuple[str, str], list[str]] = {}
    with (OUT_DAT / "bpe_sequences_extended.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            canonical_idx[(r["agent"], r["instance_id"])] = r["canonical"]

    gold_files = load_gold_files()
    pass_data = json.loads((OUT_DAT / "extended_pass_fail.json").read_text())

    records = []
    for sub_id, agent in SUBMISSION_LABEL.items():
        sub_dir = CACHE / sub_id
        if not sub_dir.is_dir():
            continue
        resolved = set(pass_data.get(sub_id, {}).get("resolved", []))
        for tf in sorted(sub_dir.glob("*.json")):
            if tf.name == "manifest.json":
                continue
            iid = tf.stem
            gold = gold_files.get(iid)
            if not gold or iid in resolved:
                continue
            try:
                env = json.loads(tf.read_text())
            except Exception:
                continue
            loc = first_localization_step_extended(env, gold)
            if not isinstance(loc, int):
                continue
            c = canonical_idx.get((agent, iid))
            if c is None:
                continue
            records.append({
                "agent": agent,
                "post": c[loc:],
                "steps_after": len(c) - loc,
            })

    df = pd.DataFrame(records)
    median = float(df["steps_after"].median())
    long_df = df[df["steps_after"] > median].copy()
    long_df["edit"] = long_df["post"].apply(lambda x: share(x, "EDIT_SRC_PY"))
    long_df["cd"] = long_df["post"].apply(lambda x: share(x, "SHELL_CD"))

    # Aggregate correlation, for the reference line
    agg_pear, agg_pp = pearsonr(long_df["edit"], long_df["cd"])
    agg_spear, agg_sp = spearmanr(long_df["edit"], long_df["cd"])

    # Per-agent correlations
    rows = []
    for agent, g in long_df.groupby("agent"):
        n = len(g)
        if n < 10:
            continue
        if g["edit"].std() == 0 or g["cd"].std() == 0:
            rows.append({
                "agent": agent, "n": n,
                "pearson": None, "p_pear": None,
                "spearman": None, "p_spear": None,
                "note": "one variable constant; correlation undefined",
            })
            continue
        pr, pp = pearsonr(g["edit"], g["cd"])
        sr, sp = spearmanr(g["edit"], g["cd"])
        rows.append({
            "agent": agent, "n": int(n),
            "pearson": float(pr), "p_pear": float(pp),
            "spearman": float(sr), "p_spear": float(sp),
            "note": None,
        })

    payload = {
        "n_long_typeB": int(len(long_df)),
        "aggregate": {
            "pearson": float(agg_pear), "p_pear": float(agg_pp),
            "spearman": float(agg_spear), "p_spear": float(agg_sp),
        },
        "per_agent": rows,
    }
    out_json = OUT_DAT / "r10_within_agent_correlation.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved {out_json}")
    print(f"\nAggregate Pearson r = {agg_pear:+.3f}  (p = {agg_pp:.2e})")
    for r in rows:
        if r["note"]:
            print(f"  {r['agent']:25s} n={r['n']:3d}  ({r['note']})")
        else:
            print(f"  {r['agent']:25s} n={r['n']:3d}  r = {r['pearson']:+.3f}  (p = {r['p_pear']:.2e})")

    # Figure: dot plot of within-agent Pearson r per agent, colored by cell,
    # with the aggregate r marked as a vertical reference line.
    plot_rows = [r for r in rows if r["pearson"] is not None]
    for r in plot_rows:
        r["cell"] = AGENT_CELL[r["agent"]]
    plot_df = pd.DataFrame(plot_rows)
    plot_df = plot_df.sort_values("pearson")
    agent_order = plot_df["agent"].tolist()

    # Filter legend to only cells actually present in the plot so the
    # legend does not advertise cells (DARS, Moatless, Agentless) that
    # were excluded from the panel for honest reasons.
    present_cells = list(dict.fromkeys(plot_df["cell"].tolist()))
    domain = [c for c in CELL_COLOR if c in present_cells]
    range_ = [CELL_COLOR[k] for k in domain]

    y_axis = alt.Axis(title=None, domain=False, ticks=False,
                      labelFontSize=10, labelLimit=200, labelPadding=8)

    # Alternating row bands for scan-tracking (5 rows worth of bands).
    band_df = pd.DataFrame([
        {"agent": a} for i, a in enumerate(agent_order) if i % 2 == 0
    ])
    bands = (
        alt.Chart(band_df)
        .mark_rect(fill="#F1F1EE", opacity=1.0, stroke=None)
        .encode(y=alt.Y("agent:N", sort=agent_order, axis=y_axis))
    )

    # Vertical reference line at 0 (null effect).
    zero_line = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color="#999999", strokeDash=[4, 3])
        .encode(x="x:Q")
    )

    # Aggregate-r reference line.
    agg_line = (
        alt.Chart(pd.DataFrame({"x": [agg_pear]}))
        .mark_rule(color="#444444", strokeDash=[2, 2])
        .encode(x="x:Q")
    )

    dots = (
        alt.Chart(plot_df)
        .mark_circle(size=200, opacity=1.0, strokeWidth=0)
        .encode(
            x=alt.X("pearson:Q",
                    scale=alt.Scale(domain=[-0.5, 0.2]),
                    title="Within-agent Pearson r (edit_share vs cd_share)",
                    axis=alt.Axis(domain=False, ticks=False, labelFontSize=10)),
            y=alt.Y("agent:N", sort=agent_order, axis=y_axis),
            color=alt.Color("cell:N", scale=alt.Scale(domain=domain, range=range_),
                            legend=alt.Legend(orient="bottom", title=None, columns=3)),
            tooltip=["agent", "n", alt.Tooltip("pearson:Q", format="+.3f"),
                     alt.Tooltip("p_pear:Q", format=".2e")],
        )
    )

    chart = (
        alt.layer(bands, zero_line, agg_line, dots)
        .resolve_scale(color="independent")
        .properties(
            width=420, height=180,
            title=alt.TitleParams(
                f"Within-agent correlation (long Type B); aggregate r = {agg_pear:+.2f}",
                fontSize=11, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    out_png = OUT_FIG / "fig_postloc_within_agent_corr.png"
    chart.save(str(out_png), scale_factor=2)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
