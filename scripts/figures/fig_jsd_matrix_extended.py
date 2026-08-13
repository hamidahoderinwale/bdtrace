"""Phase 6: JSD heatmap for the 8-submission extended corpus.

Reads the BPE-expressed sequences and computes pairwise Jensen-Shannon
divergence between agents on the BPE-vocab probability distribution.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl
    output/paper2_pilot/bpe_model_extended.json
Writes:
    output/figures/fig_jsd_matrix_extended.png
    output/paper2_pilot/jsd_matrix_extended.json

Usage:
    python -m scripts.figures.fig_jsd_matrix_extended
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register
register()

OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot"

SEQ_FILE = OUT_DAT / "bpe_sequences_extended.jsonl"
MODEL_FILE = OUT_DAT / "bpe_model_extended.json"


def jsd_squared(p: np.ndarray, q: np.ndarray) -> float:
    return float(jensenshannon(p, q, base=2)) ** 2


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DAT.mkdir(parents=True, exist_ok=True)

    model = json.loads(MODEL_FILE.read_text())
    vocab = model["vocab"]

    records = [json.loads(l) for l in SEQ_FILE.open()]
    print(f"Loaded {len(records)} sequences across {len(set(r['agent'] for r in records))} agents")

    per_agent: dict[str, Counter] = {}
    for r in records:
        per_agent.setdefault(r["agent"], Counter()).update(r["bpe"])

    agents = sorted(per_agent)
    distributions = {}
    for agent, counter in per_agent.items():
        total = sum(counter.get(v, 0) for v in vocab)
        if total == 0:
            distributions[agent] = np.zeros(len(vocab))
        else:
            distributions[agent] = np.array([counter.get(v, 0) / total for v in vocab])

    # Compute full pairwise matrix
    n = len(agents)
    matrix = np.zeros((n, n))
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i == j:
                continue
            matrix[i, j] = jsd_squared(distributions[a], distributions[b])

    print("\nJSD matrix (rounded):")
    print(f"{'':25s}  " + "  ".join(f"{a[:18]:>18s}" for a in agents))
    for i, a in enumerate(agents):
        row = "  ".join(f"{matrix[i, j]:>18.3f}" for j in range(n))
        print(f"{a[:25]:25s}  {row}")

    # Save JSON
    payload = {
        "n_records": len(records),
        "agents": agents,
        "vocab_size": len(vocab),
        "matrix": [
            {"row": agents[i], "col": agents[j],
             "jsd": float(matrix[i, j])}
            for i in range(n) for j in range(n)
        ],
        "matrix_array": matrix.tolist(),
    }
    (OUT_DAT / "jsd_matrix_extended.json").write_text(json.dumps(payload, indent=2))

    # Long-form for Altair
    rows = []
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            v = float(matrix[i, j])
            rows.append({
                "row":   a,
                "col":   b,
                "value": v,
                "label": "" if i == j else f"{v:.2f}",
            })
    df = pd.DataFrame(rows)
    vmax = df["value"].max() or 1.0

    rect = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("col:N", sort=agents,
                    axis=alt.Axis(title=None, labelAngle=-30, labelLimit=200)),
            y=alt.Y("row:N", sort=agents,
                    axis=alt.Axis(title=None, labelLimit=200)),
            color=alt.Color("value:Q",
                            scale=alt.Scale(scheme="blues", domain=[0, vmax]),
                            legend=alt.Legend(title="JSD",
                                              orient="right",
                                              titleFontSize=10,
                                              labelFontSize=9)),
        )
    )
    text_white = (
        alt.Chart(df[df["value"] > vmax / 2])
        .mark_text(fontSize=9, color="white")
        .encode(
            x=alt.X("col:N", sort=agents),
            y=alt.Y("row:N", sort=agents),
            text="label:N",
        )
    )
    text_dark = (
        alt.Chart(df[(df["value"] <= vmax / 2) & (df["value"] > 0)])
        .mark_text(fontSize=9, color="#222222")
        .encode(
            x=alt.X("col:N", sort=agents),
            y=alt.Y("row:N", sort=agents),
            text="label:N",
        )
    )

    chart = (
        (rect + text_white + text_dark)
        .properties(
            width=400, height=400,
            title=alt.TitleParams(
                "Pairwise agent JSD on extended corpus",
                fontSize=12, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    out_fig = OUT_FIG / "fig_jsd_matrix_extended.png"
    chart.save(str(out_fig), scale_factor=2)
    print(f"\nSaved {out_fig}")
    print(f"Saved {OUT_DAT / 'jsd_matrix_extended.json'}")


if __name__ == "__main__":
    main()
