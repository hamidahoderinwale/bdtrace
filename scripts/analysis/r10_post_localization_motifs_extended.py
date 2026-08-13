"""Phase 9: R10 stuck-loop signature on the extended corpus.

Post-localization motif rates as a function of steps_after, plotted
as the full distribution rather than a median split. For each top
diverging motif (selected by the same procedure as before), shows
per-trajectory proportion of that motif across the entire steps_after
distribution -- with per-decile median + IQR band and individual
trajectory dots overlaid. The reader sees the gradient (or threshold)
shape directly instead of trusting a two-number short/long summary.

Excludes Agentless (reach/no-reach not applicable; sections are
deterministic).

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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.preferences.localization import load_gold_files
from analysis.preferences.localization_extended import first_localization_step_extended
from scripts.theme import register, BLUE, COPPER, GREEN, MAGENTA, OLIVE
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

    # ── Motif selection ──────────────────────────────────────────────────────
    # Pick the top diverging motifs the same way the median-split version
    # did (used only to choose which curves to draw; the curves themselves
    # use no splits). Below-median = "short", above-median = "long".
    def top_atoms(seqs, n=TOP_N):
        counts: Counter = Counter()
        total = 0
        for s in seqs:
            counts.update(s)
            total += len(s)
        if total == 0:
            return []
        return [(t, c / total) for t, c in counts.most_common(n)]

    short_seqs = df[df["steps_after"] <= median_steps]["post_canonical"].tolist()
    long_seqs  = df[df["steps_after"]  > median_steps]["post_canonical"].tolist()
    short_top = dict(top_atoms(short_seqs))
    long_top  = dict(top_atoms(long_seqs))
    all_motifs = set(short_top) | set(long_top)
    diffs = sorted(
        ({"motif": m,
          "short_rate": short_top.get(m, 0.0),
          "long_rate":  long_top.get(m, 0.0),
          "diff":       long_top.get(m, 0.0) - short_top.get(m, 0.0)}
         for m in all_motifs),
        key=lambda d: abs(d["diff"]), reverse=True,
    )
    # Keep up to 8 motifs for small-multiples; signed-sort so growers and
    # shrinkers are both represented.
    selected = sorted(diffs[:8], key=lambda d: -d["diff"])
    motif_order = [d["motif"] for d in selected]
    print("\nSelected motifs for distribution panels (top |diff|, signed):")
    for d in selected:
        print(f"  {d['motif']:35s}  short={d['short_rate']:.3f}  long={d['long_rate']:.3f}  Δ={d['diff']:+.3f}")

    # ── Per-trajectory per-motif proportion ─────────────────────────────────
    rows: list[dict] = []
    for _, r in df.iterrows():
        post = r["post_canonical"]
        n_post = len(post)
        if n_post == 0:
            continue
        for motif in motif_order:
            rows.append({
                "instance_id": r["instance_id"],
                "agent":       r["agent"],
                "steps_after": r["steps_after"],
                "motif":       motif,
                "proportion":  post.count(motif) / n_post,
            })
    long_df = pd.DataFrame(rows)

    # ── Decile binning on steps_after ────────────────────────────────────────
    # 10 quantile bins; midpoint of each bin used as the x-position for the
    # per-decile summary line.
    decile_edges = np.quantile(df["steps_after"], np.linspace(0, 1, 11))
    decile_edges[0] = decile_edges[0] - 0.5
    decile_edges[-1] = decile_edges[-1] + 0.5
    decile_mids = [(decile_edges[i] + decile_edges[i + 1]) / 2 for i in range(10)]

    def decile_of(x: float) -> int:
        for i in range(10):
            if x <= decile_edges[i + 1]:
                return i
        return 9

    long_df["decile"] = long_df["steps_after"].map(decile_of)
    long_df["decile_mid"] = long_df["decile"].map(lambda i: decile_mids[i])

    summary_rows: list[dict] = []
    for motif in motif_order:
        for d in range(10):
            sub = long_df[(long_df["motif"] == motif) & (long_df["decile"] == d)]["proportion"]
            if len(sub) == 0:
                continue
            summary_rows.append({
                "motif":  motif,
                "decile": d,
                "decile_mid": decile_mids[d],
                "median": float(sub.median()),
                "q25":    float(sub.quantile(0.25)),
                "q75":    float(sub.quantile(0.75)),
                "n":      int(len(sub)),
            })
    summary_df = pd.DataFrame(summary_rows)

    # ── Small-multiples plot ────────────────────────────────────────────────
    # For each selected motif: scatter of per-trajectory proportions, an IQR
    # band per decile, and a median line connecting decile midpoints.
    # X-axis: steps_after (raw value, log scale to handle the long tail).
    x_min = max(1, float(df["steps_after"].min()))
    x_max = float(df["steps_after"].max())

    def short_label(m: str, n: int = 30) -> str:
        return m if len(m) <= n else m[: n - 2] + ".."

    # Build one panel per motif via hconcat (altair's facet can't combine
    # layered charts that pull from different DataFrames).
    panel_palette = [BLUE, COPPER, GREEN, MAGENTA, OLIVE, BLUE, COPPER, GREEN]

    def make_panel(motif: str, color: str) -> alt.LayerChart:
        traj = long_df[long_df["motif"] == motif]
        summ = summary_df[summary_df["motif"] == motif]
        scatter = (
            alt.Chart(traj)
            .mark_circle(size=14, opacity=0.18, color="#666666", strokeWidth=0)
            .encode(
                x=alt.X("steps_after:Q",
                        scale=alt.Scale(type="log", domain=[x_min, x_max * 1.1]),
                        axis=alt.Axis(title="steps after localization",
                                      domain=False, ticks=False, labelFontSize=8)),
                y=alt.Y("proportion:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(title="proportion",
                                      format=".0%", domain=False, ticks=False, labelFontSize=8)),
            )
        )
        band = (
            alt.Chart(summ)
            .mark_area(opacity=0.25, color=color)
            .encode(
                x=alt.X("decile_mid:Q",
                        scale=alt.Scale(type="log", domain=[x_min, x_max * 1.1])),
                y=alt.Y("q25:Q", scale=alt.Scale(domain=[0, 1])),
                y2=alt.Y2("q75:Q"),
            )
        )
        median_line = (
            alt.Chart(summ)
            .mark_line(strokeWidth=2, color=color)
            .encode(
                x=alt.X("decile_mid:Q",
                        scale=alt.Scale(type="log", domain=[x_min, x_max * 1.1])),
                y=alt.Y("median:Q", scale=alt.Scale(domain=[0, 1])),
            )
        )
        median_points = (
            alt.Chart(summ)
            .mark_point(size=40, filled=True, color=color)
            .encode(
                x=alt.X("decile_mid:Q",
                        scale=alt.Scale(type="log", domain=[x_min, x_max * 1.1])),
                y=alt.Y("median:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("decile:Q", title="decile"),
                         alt.Tooltip("decile_mid:Q", title="steps_after midpoint", format=".0f"),
                         alt.Tooltip("median:Q", format=".1%"),
                         alt.Tooltip("q25:Q", format=".1%"),
                         alt.Tooltip("q75:Q", format=".1%"),
                         alt.Tooltip("n:Q", title="n trajectories")],
            )
        )
        return (
            alt.layer(scatter, band, median_line, median_points)
            .properties(
                width=180, height=140,
                title=alt.TitleParams(short_label(motif), fontSize=10,
                                      color="#111111", anchor="start"),
            )
        )

    panels = [make_panel(m, panel_palette[i % len(panel_palette)])
              for i, m in enumerate(motif_order)]

    # Arrange in rows of 4 panels.
    per_row = 4
    rows_of_panels = [panels[i:i + per_row] for i in range(0, len(panels), per_row)]
    grid = alt.vconcat(
        *[alt.hconcat(*row, spacing=18) for row in rows_of_panels],
        spacing=22,
    )
    chart = (
        grid
        .properties(
            title=alt.TitleParams(
                "Post-localization motif proportion vs steps after first localization",
                subtitle=(
                    f"Per-trajectory proportions (grey dots), per-decile median "
                    f"(coloured line) and IQR (coloured band) across "
                    f"{len(df)} reached-but-failed trajectories. X log-scaled. "
                    "Replaces the prior short-vs-long median split."
                ),
                fontSize=12, subtitleFontSize=10,
                color="#111111", subtitleColor="#666666", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    out_fig = OUT_FIG / "fig_postloc_motifs_extended.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"\nSaved {out_fig}")

    payload = {
        "n_reached_but_failed": int(len(df)),
        "median_steps_after":   median_steps,
        "q25_steps_after":      q25,
        "q75_steps_after":      q75,
        "by_agent_n":           df.groupby("agent").size().to_dict(),
        "selected_motifs":      motif_order,
        "median_split_diffs":   selected,
        "per_decile_summary":   summary_df.to_dict(orient="records"),
    }
    out_json = OUT_DAT / "r10_postloc_motifs_extended.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
