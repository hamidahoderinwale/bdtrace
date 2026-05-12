"""Phase 9: R10 stuck-loop signature on the extended corpus.

Same as scripts/analysis/r10_post_localization_motifs.py but on the
8-submission cross-scaffold corpus. Splits Type B failures (reached gold,
still failed) into short vs long post-localization segments at the
median, and reports the most diverging canonical-atom motifs.

Excludes Agentless (Type A/B not applicable; sections are deterministic).

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/extended_pass_fail.json
    output/resolved_traces_lite_full.jsonl
    output/trajectories/.cache/<sub>/<iid>.json
Writes:
    output/paper2_pilot/r10_postloc_motifs_extended.json
    output/figures/fig_postloc_motifs_extended.png

Usage:
    python -m scripts.analysis.r10_post_localization_motifs_extended
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.preferences.localization import load_gold_files
from analysis.preferences.localization_extended import first_localization_step_extended
from scripts.theme import register, BLUE, MAGENTA
register()

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

TOP_N = 15


def load_canonical_index() -> dict[tuple[str, str], list[str]]:
    """Map (agent, instance_id) -> canonical atom sequence."""
    out: dict[tuple[str, str], list[str]] = {}
    with SEQ_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["agent"], r["instance_id"])] = r["canonical"]
    return out


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DAT.mkdir(parents=True, exist_ok=True)

    gold_files = load_gold_files()
    pass_data = json.loads(PASS_FILE.read_text())
    canonical_idx = load_canonical_index()

    # Collect Type B failures (reached gold, did NOT pass) across all
    # applicable scaffolds. Skip Agentless ('n/a' return).
    records = []
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
            if iid in resolved:
                continue  # passed → skip
            try:
                env = json.loads(traj_file.read_text())
            except Exception:
                continue
            loc = first_localization_step_extended(env, gold)
            if not isinstance(loc, int):
                continue  # never reached or n/a
            canonical = canonical_idx.get((agent_short, iid))
            if canonical is None:
                continue
            n = len(canonical)
            steps_after = n - loc
            records.append({
                "agent":          agent_short,
                "submission":     sub_id,
                "instance_id":    iid,
                "n_canonical":    n,
                "loc_step":       loc,
                "steps_after":    steps_after,
                "post_canonical": canonical[loc:],
                "pre_canonical":  canonical[:loc],
            })

    df = pd.DataFrame(records)
    print(f"Type B failures with canonical data: {len(df)}")
    if len(df) == 0:
        print("ERROR: no Type B failures matched")
        return

    print(df.groupby("agent").size().to_string())

    median_steps = float(df["steps_after"].median())
    q25 = float(df["steps_after"].quantile(0.25))
    q75 = float(df["steps_after"].quantile(0.75))
    print(f"\nsteps_after median={median_steps:.0f}  IQR=[{q25:.0f}, {q75:.0f}]")

    df["duration_group"] = df["steps_after"].apply(
        lambda x: "short" if x <= median_steps else "long"
    )

    def top_motifs(seqs, n=TOP_N):
        counts = Counter()
        total = 0
        for s in seqs:
            counts.update(s)
            total += len(s)
        if total == 0:
            return []
        return [(t, c / total) for t, c in counts.most_common(n)]

    short_df = df[df["duration_group"] == "short"]
    long_df = df[df["duration_group"] == "long"]
    short_motifs = top_motifs(short_df["post_canonical"].tolist())
    long_motifs = top_motifs(long_df["post_canonical"].tolist())

    print(f"\nShort post-loc (n={len(short_df)}, <= {median_steps:.0f} steps):")
    for tok, rate in short_motifs[:10]:
        print(f"  {tok:35s}  {rate:.3f}")
    print(f"\nLong post-loc (n={len(long_df)}, > {median_steps:.0f} steps):")
    for tok, rate in long_motifs[:10]:
        print(f"  {tok:35s}  {rate:.3f}")

    short_dict = dict(short_motifs)
    long_dict = dict(long_motifs)
    all_motifs = list(set(short_dict) | set(long_dict))
    diffs = [{"motif": m,
              "short_rate": short_dict.get(m, 0.0),
              "long_rate":  long_dict.get(m, 0.0),
              "diff":       long_dict.get(m, 0.0) - short_dict.get(m, 0.0),
              "abs_diff":   abs(long_dict.get(m, 0.0) - short_dict.get(m, 0.0))}
             for m in all_motifs]
    diff_df = pd.DataFrame(diffs).sort_values("abs_diff", ascending=False).head(20)

    # Sort by signed diff descending: motifs that grow most in long failures
    # appear at top; motifs that shrink most appear at the bottom.
    diff_df = diff_df.sort_values("diff", ascending=False).head(18)

    plot_rows = []
    for _, row in diff_df.iterrows():
        plot_rows.append({"motif": row["motif"], "group": "short", "rate": row["short_rate"]})
        plot_rows.append({"motif": row["motif"], "group": "long",  "rate": row["long_rate"]})
    plot_df = pd.DataFrame(plot_rows)
    motif_order = diff_df["motif"].tolist()

    # Connecting-rule data: span from the lower rate to the higher rate per motif.
    rule_rows = [
        {"motif": row["motif"],
         "x_lo": min(row["short_rate"], row["long_rate"]),
         "x_hi": max(row["short_rate"], row["long_rate"])}
        for _, row in diff_df.iterrows()
    ]
    rule_df = pd.DataFrame(rule_rows)

    # Alternating row bands: every other motif gets a subtle gray stripe so
    # the eye can scan a long category list without losing the row.
    band_df = pd.DataFrame([
        {"motif": m} for i, m in enumerate(motif_order) if i % 2 == 0
    ])

    color_scale = alt.Scale(domain=["short", "long"], range=[BLUE, MAGENTA])
    x_max = max(plot_df["rate"]) * 1.15
    y_axis = alt.Axis(
        title=None, domain=False, ticks=False,
        labelFontSize=10, labelLimit=200, labelPadding=8,
    )

    bands = (
        alt.Chart(band_df)
        .mark_rect(fill="#F1F1EE", opacity=1.0, stroke=None)
        .encode(y=alt.Y("motif:N", sort=motif_order, axis=y_axis))
    )
    connecting = (
        alt.Chart(rule_df)
        .mark_rule(color="#666666", strokeWidth=1.2, opacity=0.7)
        .encode(
            x=alt.X("x_lo:Q", scale=alt.Scale(domain=[0, x_max])),
            x2="x_hi:Q",
            y=alt.Y("motif:N", sort=motif_order, axis=y_axis),
        )
    )
    dots = (
        alt.Chart(plot_df)
        .mark_circle(size=140, opacity=1.0, strokeWidth=0)
        .encode(
            x=alt.X(
                "rate:Q",
                title="Token frequency in post-localization segment",
                scale=alt.Scale(domain=[0, x_max]),
            ),
            y=alt.Y("motif:N", sort=motif_order, axis=y_axis),
            color=alt.Color(
                "group:N", scale=color_scale,
                legend=alt.Legend(title="Post-localization duration", orient="bottom"),
            ),
            tooltip=["motif", "group", alt.Tooltip("rate:Q", format=".3f")],
        )
    )

    chart = (
        alt.layer(bands, connecting, dots)
        .resolve_scale(color="independent")
        .properties(
            width=420, height=380,
            title=alt.TitleParams(
                "Post-localization motif frequencies, short vs long Type B failures",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    out_fig = OUT_FIG / "fig_postloc_motifs_extended.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"\nSaved {out_fig}")

    payload = {
        "n_type_b":           int(len(df)),
        "n_short":            int(len(short_df)),
        "n_long":             int(len(long_df)),
        "median_steps_after": median_steps,
        "q25_steps_after":    q25,
        "q75_steps_after":    q75,
        "by_agent_n_typeB":   df.groupby("agent").size().to_dict(),
        "short_top_motifs":   [(t, float(r)) for t, r in short_motifs[:10]],
        "long_top_motifs":    [(t, float(r)) for t, r in long_motifs[:10]],
        "top_diverging_motifs": diff_df.to_dict(orient="records"),
    }
    out_json = OUT_DAT / "r10_postloc_motifs_extended.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
