"""R10: Post-localization motif split for Type B failures.

Type B failures: agent reached the gold file but still failed.
Splits each Type B trajectory at the localization step and computes
motif frequency distributions for the post-localization segment.
Compares short vs long post-localization segments (IQR split).

Reads:
    output/trajectories/.cache/{agent}/*.json
    output/paper2_pilot/bpe_sequences.jsonl
    output/trajectories/lite_all_models.parquet
Writes:
    output/paper2_pilot/r10_postloc_motifs.json
    output/figures/fig_postloc_motifs.png
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.preferences.localization import (
    load_gold_files, load_pass_fail, first_localization_step, AGENT_MAP
)
from scripts.theme import register, AGENT_COLORS, AGENT_ORDER, BLUE, VERMILLION, GRAY

register()

CACHE   = ROOT / "output" / "trajectories" / ".cache"
BPE_FILE = ROOT / "output" / "paper2_pilot" / "bpe_sequences.jsonl"
FIG_OUT  = ROOT / "output" / "figures"
OUT      = ROOT / "output" / "paper2_pilot"

TOP_N = 15  # top motifs to display


def load_bpe_index() -> dict[tuple[str, str], list[str]]:
    """Return {(agent, instance_id): canonical_sequence} from bpe_sequences.jsonl."""
    idx: dict[tuple[str, str], list[str]] = {}
    with BPE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            canonical = rec.get("canonical", [])
            if canonical:
                idx[(rec["agent"], rec["instance_id"])] = canonical
    return idx


def main() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    gold_files = load_gold_files()
    pass_fail  = load_pass_fail()
    bpe_idx    = load_bpe_index()

    # Collect Type B failures: reached gold file, did not pass
    records = []
    for agent_dir in sorted(CACHE.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_short = AGENT_MAP.get(agent_dir.name)
        if agent_short is None:
            continue
        for traj_file in sorted(agent_dir.glob("*.json")):
            iid    = traj_file.stem
            gold   = gold_files.get(iid)
            passed = pass_fail.get((agent_dir.name, iid))
            if gold is None or passed is None or passed:
                continue  # skip if passed or missing
            raw  = json.loads(traj_file.read_text())
            traj = raw.get("trajectory", [])
            n    = len(traj)
            if n == 0:
                continue
            loc_step = first_localization_step(traj, gold)
            if loc_step is None:
                continue  # Type A, skip

            canonical = bpe_idx.get((agent_short, iid))
            if canonical is None:
                continue

            steps_after = n - loc_step
            post_canonical = canonical[loc_step:]  # tokens after localization
            pre_canonical  = canonical[:loc_step]

            records.append({
                "agent":         agent_short,
                "instance_id":   iid,
                "n_steps":       n,
                "loc_step":      loc_step,
                "steps_after":   steps_after,
                "post_canonical": post_canonical,
                "pre_canonical":  pre_canonical,
            })

    df = pd.DataFrame(records)
    print(f"Type B failures with BPE data: {len(df)}")

    # IQR split: short = below median, long = above median
    median_steps = df["steps_after"].median()
    print(f"Median steps after: {median_steps:.0f}")
    q25 = df["steps_after"].quantile(0.25)
    q75 = df["steps_after"].quantile(0.75)
    print(f"IQR: [{q25:.0f}, {q75:.0f}]")

    df["duration_group"] = df["steps_after"].apply(
        lambda x: "short" if x <= median_steps else "long"
    )

    # Top motifs in post-localization segments
    def top_motifs(sequences: list[list[str]], n: int = TOP_N) -> list[tuple[str, float]]:
        counts: Counter = Counter()
        total = 0
        for seq in sequences:
            counts.update(seq)
            total += len(seq)
        if total == 0:
            return []
        return [(tok, cnt / total) for tok, cnt in counts.most_common(n)]

    short_df = df[df["duration_group"] == "short"]
    long_df  = df[df["duration_group"] == "long"]

    short_motifs = top_motifs(short_df["post_canonical"].tolist())
    long_motifs  = top_motifs(long_df["post_canonical"].tolist())

    print(f"\nShort post-localization (n={len(short_df)}, <= {median_steps:.0f} steps):")
    for tok, rate in short_motifs[:10]:
        print(f"  {tok:50s}  {rate:.3f}")

    print(f"\nLong post-localization (n={len(long_df)}, > {median_steps:.0f} steps):")
    for tok, rate in long_motifs[:10]:
        print(f"  {tok:50s}  {rate:.3f}")

    # Compute rate diff: long - short for shared motifs
    short_dict = dict(short_motifs)
    long_dict  = dict(long_motifs)
    all_motifs = list(set(short_dict) | set(long_dict))
    diffs = []
    for m in all_motifs:
        s = short_dict.get(m, 0.0)
        l = long_dict.get(m, 0.0)
        diffs.append({"motif": m, "short_rate": s, "long_rate": l, "diff": l - s, "abs_diff": abs(l - s)})

    diff_df = pd.DataFrame(diffs).sort_values("abs_diff", ascending=False).head(20)

    # Build figure: dot plot of motif rates by group for top diverging motifs
    plot_rows = []
    for _, row in diff_df.iterrows():
        plot_rows.append({"motif": row["motif"], "group": "short", "rate": row["short_rate"]})
        plot_rows.append({"motif": row["motif"], "group": "long",  "rate": row["long_rate"]})
    plot_df = pd.DataFrame(plot_rows)

    motif_order = diff_df["motif"].tolist()

    color_scale = alt.Scale(
        domain=["short", "long"],
        range=[BLUE, VERMILLION],
    )

    chart = (
        alt.Chart(plot_df)
        .mark_point(filled=True, size=80)
        .encode(
            y=alt.Y("motif:N", sort=motif_order, axis=alt.Axis(title=None)),
            x=alt.X("rate:Q", title="Token frequency in post-localization segment",
                    scale=alt.Scale(domain=[0, max(plot_df["rate"]) * 1.15])),
            color=alt.Color("group:N", scale=color_scale,
                            legend=alt.Legend(
                                title="Post-localization duration",
                                orient="bottom",
                            )),
        )
        .properties(
            width=380,
            height=320,
            title=alt.TitleParams(
                "Post-localization motifs: short vs long Type B failures",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    out_fig = FIG_OUT / "fig_postloc_motifs.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"\nSaved {out_fig}")

    # Save JSON
    result = {
        "n_type_b": len(df),
        "median_steps_after": float(median_steps),
        "q25_steps_after": float(q25),
        "q75_steps_after": float(q75),
        "n_short": int(len(short_df)),
        "n_long":  int(len(long_df)),
        "short_top_motifs":  [(t, float(r)) for t, r in short_motifs[:10]],
        "long_top_motifs":   [(t, float(r)) for t, r in long_motifs[:10]],
        "top_diverging_motifs": diff_df.to_dict(orient="records"),
    }

    (OUT / "r10_postloc_motifs.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    print(f"Saved {OUT / 'r10_postloc_motifs.json'}")


if __name__ == "__main__":
    main()
