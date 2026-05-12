"""9-agent UMAP visualizations of procedural space.

Two figures:
  - Per-trajectory UMAP from BPE motif distributions, colored by
    (paradigm x scaffold) cell. ~2,639 points, 5 colors.
  - Per-agent UMAP from agent-mean motif distributions. 9 points,
    cell-colored, labeled.

Reads:
    output/paper2_pilot/bpe_sequences_extended.jsonl

Writes:
    output/figures/fig_trajectory_umap_extended.png
    output/figures/fig_agent_umap_extended.png
    output/paper2_pilot/trajectory_clusters_extended.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, GREEN, BLUE, MAGENTA, COPPER, OLIVE
register()

SEQ_PATH = ROOT / "output" / "paper2_pilot" / "bpe_sequences_extended.jsonl"
OUT_FIG = ROOT / "output" / "figures"
OUT_DAT = ROOT / "output" / "paper2_pilot" / "trajectory_clusters_extended.json"

# Paradigm x scaffold cell assignment for each agent in the corpus
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

CELL_COLORS = {
    "SWE-agent base":         BLUE,
    "SWE-agent extended-thinking":  GREEN,
    "Agentless":                    COPPER,
    "DARS":                         MAGENTA,
    "Moatless":                     OLIVE,
}
CELL_ORDER = list(CELL_COLORS.keys())


def load_records() -> list[dict]:
    return [json.loads(line) for line in SEQ_PATH.open() if line.strip()]


def build_matrix(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    vocab_counter: Counter = Counter()
    for r in records:
        vocab_counter.update(r["bpe"])
    vocab = sorted(vocab_counter.keys())
    idx = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(records), len(vocab)), dtype=np.float32)
    for i, r in enumerate(records):
        for t in r["bpe"]:
            X[i, idx[t]] += 1
    X = normalize(X, norm="l1", axis=1)
    return X, vocab


def per_agent_means(X: np.ndarray, records: list[dict]) -> tuple[np.ndarray, list[str]]:
    agents = sorted({r["agent"] for r in records})
    means = np.zeros((len(agents), X.shape[1]), dtype=np.float32)
    for i, agent in enumerate(agents):
        idxs = [j for j, r in enumerate(records) if r["agent"] == agent]
        means[i] = X[idxs].mean(axis=0)
    means = normalize(means, norm="l1", axis=1)
    return means, agents


def umap_project(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int = 0) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        n_components=2,
    )
    return reducer.fit_transform(X)


def plot_trajectory_umap(coords: np.ndarray, records: list[dict], out_path: Path) -> None:
    rows = []
    for i, r in enumerate(records):
        rows.append({
            "umap1": float(coords[i, 0]),
            "umap2": float(coords[i, 1]),
            "agent": r["agent"],
            "cell": AGENT_CELL.get(r["agent"], "?"),
        })
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_point(size=18, filled=True, opacity=0.55, strokeWidth=0)
        .encode(
            x=alt.X("umap1:Q", axis=alt.Axis(title="UMAP-1", domain=False, ticks=False)),
            y=alt.Y("umap2:Q", axis=alt.Axis(title="UMAP-2", domain=False, ticks=False)),
            color=alt.Color(
                "cell:N",
                scale=alt.Scale(domain=CELL_ORDER, range=[CELL_COLORS[c] for c in CELL_ORDER]),
                legend=alt.Legend(orient="bottom", title=None, columns=3),
            ),
        )
        .properties(
            width=520, height=420,
            title=alt.TitleParams(
                text="Procedural space UMAP",
                fontSize=11, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(out_path), scale_factor=2)
    print(f"Saved {out_path}")


def plot_agent_umap(coords: np.ndarray, agents: list[str], out_path: Path) -> None:
    rows = []
    for i, a in enumerate(agents):
        rows.append({
            "umap1": float(coords[i, 0]),
            "umap2": float(coords[i, 1]),
            "agent": a,
            "cell": AGENT_CELL.get(a, "?"),
        })
    df = pd.DataFrame(rows)
    points = (
        alt.Chart(df)
        .mark_point(size=300, filled=True, opacity=0.9, strokeWidth=0)
        .encode(
            x=alt.X("umap1:Q", axis=alt.Axis(title="UMAP-1", domain=False, ticks=False)),
            y=alt.Y("umap2:Q", axis=alt.Axis(title="UMAP-2", domain=False, ticks=False)),
            color=alt.Color(
                "cell:N",
                scale=alt.Scale(domain=CELL_ORDER, range=[CELL_COLORS[c] for c in CELL_ORDER]),
                legend=alt.Legend(orient="bottom", title=None, columns=3),
            ),
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=10, dy=-4, fontSize=10, color="#333333")
        .encode(x="umap1:Q", y="umap2:Q", text="agent:N")
    )
    chart = (
        (points + labels)
        .properties(
            width=520, height=420,
            title=alt.TitleParams(
                text="Per-agent UMAP",
                fontSize=11, color="#111111", anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
    )
    chart.save(str(out_path), scale_factor=2)
    print(f"Saved {out_path}")


def main() -> int:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    print("Loading BPE sequences ...")
    records = load_records()
    print(f"  {len(records)} trajectories")

    print("Building motif-frequency matrix ...")
    X, vocab = build_matrix(records)
    print(f"  matrix shape {X.shape}; vocab size {len(vocab)}")

    print("Per-trajectory UMAP (n_neighbors=15, min_dist=0.25, metric=cosine) ...")
    traj_coords = umap_project(X, n_neighbors=15, min_dist=0.25, seed=0)
    plot_trajectory_umap(traj_coords, records, OUT_FIG / "fig_trajectory_umap_extended.png")

    print("Per-agent mean fingerprints ...")
    agent_means, agents = per_agent_means(X, records)
    print(f"  {len(agents)} agents")

    # Per-agent UMAP needs different hyperparameters since n=9
    # n_neighbors must be < n, use n-1=8
    print("Per-agent UMAP (n_neighbors=4, min_dist=0.5, metric=cosine) ...")
    agent_coords = umap_project(agent_means, n_neighbors=4, min_dist=0.5, seed=0)
    plot_agent_umap(agent_coords, agents, OUT_FIG / "fig_agent_umap_extended.png")

    # Persist coords for inspection
    OUT_DAT.write_text(json.dumps({
        "n_trajectories": len(records),
        "vocab_size": len(vocab),
        "n_agents": len(agents),
        "agents": agents,
        "agent_coords": {a: [float(c[0]), float(c[1])] for a, c in zip(agents, agent_coords)},
        "umap_params_trajectory": {"n_neighbors": 15, "min_dist": 0.25, "metric": "cosine", "seed": 0},
        "umap_params_agent": {"n_neighbors": 4, "min_dist": 0.5, "metric": "cosine", "seed": 0},
    }, indent=2))
    print(f"Saved {OUT_DAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
