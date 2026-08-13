"""Per-task procedural divergence across the 9-agent corpus.

For each task, compute the mean pairwise Jensen-Shannon divergence across
the agents that attempted it (over canonical-atom distributions, base 2).
This is the 9-agent action-stream analogue of the older
procedure_divergence_gap figure (which used patch-AST distances on
4 SWE-agent baseline agents).

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl

Writes:
    output/paper2_pilot/procedural_divergence_per_task_extended.json
    output/paper2_pilot/procedural_divergence_per_task_extended.png

Usage:
    uv run python scripts/figures/fig_procedural_divergence_per_task.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE
register()

OUT = ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences_extended.jsonl"
FIX_TYPES = ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "fix_types.json"


def load_records() -> list[dict]:
    with SEQ_PATH.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def per_task_pairwise_jsd(records: list[dict]) -> pd.DataFrame:
    by_task: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for r in records:
        by_task[r["instance_id"]].append((r["agent"], r["canonical"]))

    vocab: list[str] = sorted({a for r in records for a in r["canonical"]})
    vocab_idx = {a: i for i, a in enumerate(vocab)}

    def to_dist(seq: list[str]) -> np.ndarray:
        v = np.zeros(len(vocab))
        if not seq:
            return v
        for a in seq:
            v[vocab_idx[a]] += 1
        return v / v.sum()

    rows = []
    for iid, agent_seqs in by_task.items():
        if len(agent_seqs) < 2:
            continue
        dists = [(a, to_dist(seq)) for a, seq in agent_seqs]
        pairwise = []
        for (a, da), (b, db) in combinations(dists, 2):
            d = float(jensenshannon(da, db, base=2)) ** 2
            pairwise.append(d)
        rows.append({
            "instance_id":     iid,
            "n_agents":        len(agent_seqs),
            "mean_pairwise_jsd": float(np.mean(pairwise)),
            "max_pairwise_jsd":  float(np.max(pairwise)),
            "min_pairwise_jsd":  float(np.min(pairwise)),
        })
    return pd.DataFrame(rows)


def add_fix_type(df: pd.DataFrame) -> pd.DataFrame:
    labels = json.loads(FIX_TYPES.read_text())["results"]
    lut = {r["instance_id"]: r["fix_type"] for r in labels}
    df["fix_type"] = df["instance_id"].map(lambda i: lut.get(i, "unknown"))
    return df


def plot(df: pd.DataFrame, out_path: Path) -> None:
    df = df.sort_values("mean_pairwise_jsd", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    p10 = df["mean_pairwise_jsd"].quantile(0.10)
    p25 = df["mean_pairwise_jsd"].quantile(0.25)
    p50 = df["mean_pairwise_jsd"].median()
    p75 = df["mean_pairwise_jsd"].quantile(0.75)
    p90 = df["mean_pairwise_jsd"].quantile(0.90)

    main = (
        alt.Chart(df)
        .mark_area(opacity=0.85, color=BLUE, interpolate="step-after")
        .encode(
            x=alt.X("rank:Q",
                    axis=alt.Axis(title="Tasks ranked by procedural divergence",
                                  domain=False, ticks=False, labelFontSize=10)),
            y=alt.Y("mean_pairwise_jsd:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Mean pairwise JSD across agents (per task)",
                                  domain=False, ticks=False, labelFontSize=10)),
        )
    )

    chart = (
        main
        .properties(
            width=520, height=300,
            title=alt.TitleParams(
                text=(f"Per-task agent divergence is narrow: "
                      f"IQR {p25:.2f}–{p75:.2f}, median {p50:.2f}"),
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(out_path), scale_factor=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    df = per_task_pairwise_jsd(records)
    df = add_fix_type(df)
    print(f"per-task divergence on {len(df)} tasks "
          f"(median={df['mean_pairwise_jsd'].median():.3f}, "
          f"top-decile={df['mean_pairwise_jsd'].quantile(0.9):.3f}, "
          f"max={df['mean_pairwise_jsd'].max():.3f})")

    out_json = OUT / "procedural_divergence_per_task_extended.json"
    out_png = OUT / "procedural_divergence_per_task_extended.png"
    out_json.write_text(json.dumps({
        "n_tasks":       int(len(df)),
        "median":        float(df["mean_pairwise_jsd"].median()),
        "p90":           float(df["mean_pairwise_jsd"].quantile(0.9)),
        "p10":           float(df["mean_pairwise_jsd"].quantile(0.1)),
        "max":           float(df["mean_pairwise_jsd"].max()),
        "min":           float(df["mean_pairwise_jsd"].min()),
        "by_fix_type":   df.groupby("fix_type")["mean_pairwise_jsd"]
                           .agg(["count", "median", "max"])
                           .reset_index()
                           .to_dict(orient="records"),
        "rows":          df.to_dict(orient="records"),
    }, indent=2))
    plot(df, out_png)
    print(f"Saved {out_json}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
