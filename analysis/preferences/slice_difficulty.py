"""Difficulty-sliced motif analysis.

Bucket the 867 BPE-expressed trajectories by task difficulty (how many of the
three agents resolved it: 0/1-2/3), then recompute per-agent motif
distributions and pairwise Jensen-Shannon divergence within each bucket.

Hypothesis: the same-family vs cross-family JSD gap widens on harder tasks
(more exploration room = more style signal) and narrows on easy tasks
(convergent short paths). Either widening or narrowing is a finding.

Inputs:
    output/paper2_pilot/bpe_sequences.jsonl    (agent, instance_id, bpe)
    output/paper2_pilot/task_diversity.csv     (instance_id, n_resolved)

Outputs:
    output/paper2_pilot/slice_difficulty.json  (per-bucket JSD + top motifs)
    output/paper2_pilot/slice_difficulty_jsd.png
    output/paper2_pilot/slice_difficulty_motifs.png

Usage:
    python -m analysis.preferences.slice_difficulty
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.theme import register, BLUE, ORANGE, GREEN
register()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

BUCKET_LABEL = {0: "0/3", 1: "1/3", 2: "2/3", 3: "3/3"}
BUCKET_ORDER = ["0/3", "1/3", "2/3", "3/3"]

PAIR_COLORS = {
    "Claude-3.5__GPT-4": "#0072B2",
    "Claude-3.5__GPT-4o": "#E69F00",
    "GPT-4__GPT-4o": "#009E73",
}


def load_sequences() -> list[dict]:
    records = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_difficulty_map() -> dict[str, int]:
    difficulty = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            difficulty[row["instance_id"]] = int(row["n_resolved"])
    return difficulty


def bucket_records(records: list[dict], difficulty: dict[str, int]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {b: [] for b in BUCKET_ORDER}
    for r in records:
        n_res = difficulty.get(r["instance_id"])
        if n_res is None:
            continue
        label = BUCKET_LABEL[n_res]
        buckets[label].append(r)
    return buckets


def per_agent_counts(records: list[dict]) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for r in records:
        out.setdefault(r["agent"], Counter()).update(r["bpe"])
    return out


def normalize(counter: Counter, vocab: list[str]) -> np.ndarray:
    total = sum(counter[v] for v in vocab)
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([counter.get(v, 0) / total for v in vocab])


def jsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a, b, base=2)) ** 2


def analyze_bucket(records: list[dict], bucket_name: str) -> dict:
    counts = per_agent_counts(records)
    all_counts: Counter = Counter()
    for c in counts.values():
        all_counts.update(c)

    full_vocab = sorted(all_counts.keys())
    motif_vocab = [t for t in full_vocab if "+" in t]

    dist_full = {a: normalize(c, full_vocab) for a, c in counts.items()}
    dist_motifs = {a: normalize(c, motif_vocab) for a, c in counts.items()}

    agents = sorted(dist_full.keys())
    pairs = list(combinations(agents, 2))

    jsd_full = {f"{a}__{b}": jsd(dist_full[a], dist_full[b]) for a, b in pairs}
    jsd_motifs = {f"{a}__{b}": jsd(dist_motifs[a], dist_motifs[b]) for a, b in pairs}

    per_agent_totals = {a: sum(c.values()) for a, c in counts.items()}
    top_motifs = [
        {"motif": t, "count": c, "n_atoms": t.count("+") + 1}
        for t, c in all_counts.most_common()
        if "+" in t
    ][:10]

    return {
        "bucket": bucket_name,
        "n_trajectories": len(records),
        "per_agent_trajectories": Counter(r["agent"] for r in records),
        "per_agent_total_tokens": per_agent_totals,
        "jsd_full": jsd_full,
        "jsd_motifs": jsd_motifs,
        "top_motifs": top_motifs,
    }


def plot_jsd_by_bucket(results: list[dict], out_path: Path) -> None:
    # Collapse three pairs into two meaningful groups:
    # within-family = GPT-4 x GPT-4o (consistently lower)
    # cross-family  = mean(Claude x GPT-4, Claude x GPT-4o) (consistently higher)
    WITHIN = "GPT-4__GPT-4o"
    CROSS  = ["Claude-3.5__GPT-4", "Claude-3.5__GPT-4o"]

    group_order  = ["Cross family", "Within GPT family"]
    group_colors = ["#0072B2", "#009E73"]  # BLUE, GREEN

    rows = []
    for r in results:
        for key, panel in [("jsd_full", "All tokens"), ("jsd_motifs", "Repeated sequences only")]:
            within_jsd = r[key][WITHIN]
            cross_jsd  = sum(r[key][p] for p in CROSS) / len(CROSS)
            rows.append({"bucket": r["bucket"], "panel": panel, "group": "Within GPT family", "jsd": within_jsd})
            rows.append({"bucket": r["bucket"], "panel": panel, "group": "Cross family",       "jsd": cross_jsd})
    df = pd.DataFrame(rows)

    cscale = alt.Scale(domain=group_order, range=group_colors)

    panels = [
        ("All tokens",              out_path.parent / (out_path.stem + "_all_tokens.png")),
        ("Repeated sequences only", out_path.parent / (out_path.stem + "_repeated.png")),
    ]
    titles = {
        "All tokens":              "Behavioral divergence by task difficulty (all tokens)",
        "Repeated sequences only": "Behavioral divergence by task difficulty (repeated sequences)",
    }

    for panel_name, panel_path in panels:
        panel_df = df[df["panel"] == panel_name].copy()

        base = alt.Chart(panel_df).encode(
            x=alt.X("bucket:O",
                    sort=BUCKET_ORDER,
                    axis=alt.Axis(title="Task difficulty", domain=False, ticks=False,
                                  labelFontSize=11, labelPadding=8)),
            y=alt.Y("jsd:Q",
                    scale=alt.Scale(domain=[0, 0.65]),
                    axis=alt.Axis(title="Jensen-Shannon divergence",
                                  domain=False, ticks=False,
                                  values=[0, 0.2, 0.4, 0.6])),
            color=alt.Color("group:N", sort=group_order, scale=cscale,
                            legend=alt.Legend(orient="bottom", title=None,
                                              symbolSize=80)),
        )

        chart = (
            (base.mark_line(strokeWidth=2) + base.mark_point(size=60, filled=True, strokeWidth=0))
            .properties(
                title=alt.TitleParams(
                    text=titles[panel_name],
                    fontSize=13, color="#111111", anchor="start",
                ),
                width=320, height=220,
            )
            .configure_view(strokeWidth=0)
        )

        chart.save(str(panel_path), scale_factor=2)
        print(f"  Saved: {panel_path.name}")


def plot_top_motifs_by_bucket(results: list[dict], out_path: Path, top_n: int = 8) -> None:
    all_motifs: Counter = Counter()
    for r in results:
        for item in r["top_motifs"][:top_n]:
            all_motifs[item["motif"]] += item["count"]
    panel_motifs = [m for m, _ in all_motifs.most_common(top_n)]
    motif_order_top_first = list(reversed(panel_motifs))

    def fmt(m: str) -> str:
        parts = m.split("+")
        if len(parts) <= 3:
            return m.replace("+", " → ")
        return f"{parts[0]} → ... → {parts[-1]}  ({len(parts)} atoms)"

    abbrev_order = [fmt(m) for m in motif_order_top_first]

    # Compute global x-axis max so all per-bucket panels share the same scale.
    bucket_max_share = []
    for r in results:
        total = sum(r["per_agent_total_tokens"].values()) or 1
        motif_counts = {item["motif"]: item["count"] for item in r["top_motifs"]}
        for m in panel_motifs:
            bucket_max_share.append(motif_counts.get(m, 0) / total)
    x_max = max(0.005, max(bucket_max_share) * 1.10)

    for r in results:
        bucket_slug = r["bucket"].replace("/", "of")  # 0/3 -> 0of3
        panel_path = out_path.parent / (out_path.stem + "_" + bucket_slug + ".png")

        motif_counts = {item["motif"]: item["count"] for item in r["top_motifs"]}
        total = sum(r["per_agent_total_tokens"].values()) or 1
        rows = [
            {
                "motif": fmt(m),
                "motif_key": m,
                "share": motif_counts.get(m, 0) / total,
                "share_pct": f"{motif_counts.get(m, 0) / total * 100:.1f}%",
            }
            for m in panel_motifs
        ]
        df = pd.DataFrame(rows)
        panel_title = f"{r['bucket']} solved  ·  {r['n_trajectories']} trajectories"

        chart = (
            alt.Chart(df)
            .mark_bar(color=BLUE)
            .encode(
                y=alt.Y(
                    "motif:N",
                    sort=abbrev_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=10, labelLimit=480),
                ),
                x=alt.X(
                    "share:Q",
                    scale=alt.Scale(domain=[0, x_max]),
                    axis=alt.Axis(
                        title="share of bucket's actions",
                        domain=False,
                        ticks=False,
                        labelFontSize=10,
                        format=".1%",
                        values=[0, x_max / 2, x_max],
                    ),
                ),
                tooltip=["motif:N", "share_pct:N"],
            )
            .properties(
                width=420,
                height=top_n * 32,
                title=alt.TitleParams(
                    text=panel_title,
                    fontSize=12,
                    color="#111111",
                    anchor="start",
                ),
            )
            .configure_view(strokeWidth=0)
        )

        chart.save(str(panel_path), scale_factor=2)
        print(f"  Saved: {panel_path.name}")


def main() -> int:
    records = load_sequences()
    difficulty = load_difficulty_map()
    buckets = bucket_records(records, difficulty)

    print("Difficulty bucket sizes:")
    for b in BUCKET_ORDER:
        rs = buckets[b]
        per_agent = Counter(r["agent"] for r in rs)
        print(f"  {b}: {len(rs)} trajectories  ({dict(per_agent)})")

    results = []
    for b in BUCKET_ORDER:
        rs = buckets[b]
        if not rs:
            continue
        r = analyze_bucket(rs, b)
        results.append(r)
        print(f"\n{b}:")
        print(f"  n={r['n_trajectories']}, per-agent tokens={r['per_agent_total_tokens']}")
        print(f"  JSD (full):    {r['jsd_full']}")
        print(f"  JSD (motifs):  {r['jsd_motifs']}")
        print(f"  top 3 motifs:  {[(m['motif'], m['count']) for m in r['top_motifs'][:3]]}")

    print("\nHeritability ordering check (GPT-family pair should have lowest JSD):")
    for r in results:
        min_pair_m = min(r["jsd_motifs"], key=r["jsd_motifs"].get)
        heritability_gap = min(
            r["jsd_motifs"][k] for k in r["jsd_motifs"] if "Claude" in k
        ) - r["jsd_motifs"].get("GPT-4__GPT-4o", float("nan"))
        print(f"  {r['bucket']}: min pair (motifs)={min_pair_m}, "
              f"heritability gap = {heritability_gap:.4f}")

    serializable = [
        {
            **r,
            "per_agent_trajectories": dict(r["per_agent_trajectories"]),
        }
        for r in results
    ]
    (OUT / "slice_difficulty.json").write_text(
        json.dumps(serializable, indent=2, default=str)
    )

    plot_jsd_by_bucket(results, OUT / "slice_difficulty_jsd.png")
    plot_top_motifs_by_bucket(results, OUT / "slice_difficulty_motifs.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'slice_difficulty.json'}")
    print(f"  {OUT / 'slice_difficulty_jsd.png'}")
    print(f"  {OUT / 'slice_difficulty_motifs.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
