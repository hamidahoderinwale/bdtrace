"""Fix-type × motif heatmap.

For each semantic fix_type (13 categories from the SWE-bench Lite labels) and
each BPE motif, compute the usage ratio (motif frequency within that fix type
divided by motif frequency across the whole corpus). A cell value of 2 means
"this motif is used twice as often in this fix type as on average"; 0.5 means
half as often; 1 is average.

Shows which procedural patterns are fix-type-specific vs universal.

Outputs:
    output/paper2_pilot/fixtype_motif.json
    output/paper2_pilot/fixtype_motif.png

Usage:
    python -m analysis.preferences.fixtype_motif_heatmap
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
import numpy as np
import altair as alt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, NEAR_BLACK, GRAY, VERMILLION
register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
FIX_TYPES_PATH = PROJECT_ROOT / "output" / "datasets" / "swe_bench_lite_resolved" / "fix_types.json"

TOP_N_MOTIFS = 25
MIN_FIX_TYPE_N = 10  # drop fix types with fewer than N trajectories


def load_sequences() -> list[dict]:
    out = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_fix_types() -> dict[str, str]:
    d = json.loads(FIX_TYPES_PATH.read_text())
    return {r["instance_id"]: r["fix_type"] for r in d["results"]}


def abbrev(m: str, maxl: int = 28) -> str:
    parts = m.split("+")
    if len(parts) <= 2:
        s = m.replace("+", " -> ")
    else:
        s = f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)} atoms)"
    return s if len(s) <= maxl else s[: maxl - 1] + "..."


def main() -> int:
    records = load_sequences()
    fix_types = load_fix_types()

    # Group trajectories by fix_type and accumulate motif counts
    per_ft_counts: dict[str, Counter] = defaultdict(Counter)
    per_ft_n_trajectories: Counter = Counter()
    corpus_counts: Counter = Counter()

    for r in records:
        ft = fix_types.get(r["instance_id"], "unknown")
        per_ft_n_trajectories[ft] += 1
        for m in r["bpe"]:
            per_ft_counts[ft][m] += 1
            corpus_counts[m] += 1

    # Drop fix types with too few trajectories
    kept_fts = [ft for ft, n in per_ft_n_trajectories.items()
                if n >= MIN_FIX_TYPE_N]
    # Order by corpus frequency (largest fix types first)
    kept_fts.sort(key=lambda ft: -per_ft_n_trajectories[ft])

    # Pick top N motifs by corpus frequency (motifs only, length >= 2)
    top_motifs = [m for m, _ in corpus_counts.most_common() if "+" in m][:TOP_N_MOTIFS]

    # Compute usage ratio per (motif, fix_type): (freq in fix_type) / (freq in corpus).
    # Cell > 1 = over-used in this fix type; cell < 1 = under-used; cell = 1 = average.
    corpus_total = sum(corpus_counts.values())
    eps = 1.0
    matrix = np.zeros((len(top_motifs), len(kept_fts)))
    for j, ft in enumerate(kept_fts):
        ft_total = sum(per_ft_counts[ft].values())
        for i, m in enumerate(top_motifs):
            p_ft = (per_ft_counts[ft].get(m, 0) + eps) / (ft_total + eps)
            p_corpus = (corpus_counts.get(m, 0) + eps) / (corpus_total + eps)
            matrix[i, j] = p_ft / p_corpus

    # Sort motifs by how differentiating they are: max |ratio - 1| across fix types
    discrim = np.abs(matrix - 1.0).max(axis=1)
    order = np.argsort(-discrim)
    matrix = matrix[order]
    top_motifs = [top_motifs[i] for i in order]

    FIX_TYPE_LABELS = {
        "logic_fix":          "Logic fix",
        "exception_handling": "Exception handling",
        "api_change":         "API change",
        "config_fix":         "Config fix",
        "guard_clause":       "Guard clause",
        "type_coercion":      "Type coercion",
        "refactor":           "Refactor",
        "string_fix":         "String fix",
        "test_fix":           "Test fix",
        "import_fix":         "Import fix",
        "other":              "Other",
    }

    # Plot — diverging colormap centered at 1.0 using paper palette
    span = float(max(abs(matrix.min() - 1), abs(matrix.max() - 1)))
    span = max(span, 0.5)

    heatmap_rows = []
    motif_order = [abbrev(m) for m in top_motifs]
    for i, motif in enumerate(top_motifs):
        for j, ft in enumerate(kept_fts):
            v = float(matrix[i, j])
            n = per_ft_n_trajectories[ft]
            ft_label = FIX_TYPE_LABELS.get(ft, ft) + f"\n(n={n})"
            heatmap_rows.append({
                "motif":    abbrev(motif),
                "fix_type": ft_label,
                "ratio":    v,
                "text":     f"{v:.1f}x" if abs(v - 1.0) >= 0.4 else "",
            })
    hm_df = pd.DataFrame(heatmap_rows)

    ft_order = [
        FIX_TYPE_LABELS.get(ft, ft) + f"\n(n={per_ft_n_trajectories[ft]})"
        for ft in kept_fts
    ]

    chart_width  = max(420, len(kept_fts) * 60)
    chart_height = max(320, len(top_motifs) * 20)

    rect = (
        alt.Chart(hm_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "fix_type:N",
                title=None,
                sort=ft_order,
                axis=alt.Axis(labelAngle=-30, labelFontSize=10),
            ),
            y=alt.Y(
                "motif:N",
                title=None,
                sort=motif_order,
                axis=alt.Axis(labelFontSize=10, labelLimit=320),
            ),
            color=alt.Color(
                "ratio:Q",
                scale=alt.Scale(
                    range=["#D55E00", "#FFFFFF", "#0072B2"],
                    domainMid=1.0,
                    domain=[1 - span, 1 + span],
                ),
                legend=alt.Legend(title="Usage ratio vs corpus average",
                                  titleFontSize=10, labelFontSize=9,
                                  gradientLength=120),
            ),
        )
    )
    text_layer = (
        alt.Chart(hm_df[hm_df["text"] != ""])
        .mark_text(fontSize=9, fontWeight="normal")
        .encode(
            x=alt.X("fix_type:N", sort=ft_order),
            y=alt.Y("motif:N", sort=motif_order),
            text="text:N",
            color=alt.value("#111111"),
        )
    )
    chart = (
        (rect + text_layer)
        .properties(
            width=chart_width,
            height=chart_height,
            title=alt.TitleParams(
                text="Action patterns by fix type",
                fontSize=13,
                color="#111111",
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    chart.save(str(OUT / "fixtype_motif.png"), scale_factor=2)

    # Machine-readable output
    out_data = {
        "fix_types": kept_fts,
        "fix_type_trajectory_counts": {ft: per_ft_n_trajectories[ft] for ft in kept_fts},
        "motifs": top_motifs,
        "log_odds_matrix": matrix.tolist(),
    }
    (OUT / "fixtype_motif.json").write_text(json.dumps(out_data, indent=2))

    # Console summary: most fix-type-specific motifs
    print(f"Fix types kept: {kept_fts}")
    print(f"\nMost fix-type-specific motifs (top 10 by max deviation from 1x):")
    for i in range(min(10, len(top_motifs))):
        m = top_motifs[i]
        row = matrix[i]
        top_ft_idx = int(np.argmax(row))
        bot_ft_idx = int(np.argmin(row))
        print(f"  {m[:50]:<50s}  peaks at {kept_fts[top_ft_idx]:<18s} ({row[top_ft_idx]:.2f}x)  "
              f"rare on {kept_fts[bot_ft_idx]:<18s} ({row[bot_ft_idx]:.2f}x)")

    print(f"\nSaved:\n  {OUT / 'fixtype_motif.json'}\n  {OUT / 'fixtype_motif.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
