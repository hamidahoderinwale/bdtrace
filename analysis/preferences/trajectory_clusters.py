"""Trajectory clustering.

Encode each trajectory as a motif-frequency vector over the BPE vocabulary,
reduce dimensions with UMAP, cluster with HDBSCAN. Ask: do trajectories
cluster by task, by agent, or by procedural phase?

Three possible outcomes, each informative:
  - Clusters dominated by task -> trajectory style is task-driven
  - Clusters dominated by agent -> heritability at whole-trajectory level
  - Clusters cross both -> agent-independent procedural types exist

Outputs:
    output/paper2_pilot/trajectory_clusters.json
    output/paper2_pilot/trajectory_clusters_umap.png     (UMAP by agent and by cluster)
    output/paper2_pilot/trajectory_clusters_profile.png  (per-cluster top motifs + composition)

Usage:
    python -m analysis.preferences.trajectory_clusters
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, VERMILLION, SKY, GRAY, NEAR_BLACK
register()

OUT = PROJECT_ROOT / "output" / "paper2_pilot"
SEQ_PATH = OUT / "bpe_sequences.jsonl"
DIVERSITY_PATH = OUT / "task_diversity.csv"

AGENT_COLORS = {
    "Claude-3.5": "#009E73",
    "GPT-4": "#0072B2",
    "GPT-4o": "#E69F00",
}

# Map a motif atom-sequence to a short English phrase.
# Checked longest-to-shortest so "CREATE_REPRO+EDIT+RUN" wins over "CREATE_REPRO+EDIT".
MOTIF_PHRASES: list[tuple[tuple[str, ...], str]] = [
    (("CREATE_REPRO_PY", "EDIT_REPRO_PY", "RUN_PYTHON_REPRO_PY"), "build reproducer"),
    (("EDIT_SRC_PY", "RUN_PYTHON_REPRO_PY", "SHELL_RM", "SUBMIT"), "edit-run-cleanup-submit"),
    (("EDIT_SRC_PY", "RUN_PYTHON_REPRO_PY", "SHELL_RM"), "edit-run-cleanup"),
    (("OPEN_SRC_PY", "SEARCH", "NAV_SRC_PY"), "open-search-navigate"),
    (("EDIT_SRC_PY", "RUN_PYTHON_TEST_PY"), "edit-test loop"),
    (("EDIT_REPRO_PY", "RUN_PYTHON_REPRO_PY"), "reproducer iteration"),
    (("CREATE_TEST_PY", "EDIT_TEST_PY", "RUN_PYTHON_TEST_PY"), "test authoring"),
    (("CREATE_CONFIG_PY", "EDIT_CONFIG_PY"), "config editing"),
    (("CREATE_DOC", "EDIT_DOC"), "doc editing"),
    (("FIND_FILE", "OPEN_SRC_PY"), "find-then-open"),
    (("SEARCH", "OPEN_SRC_PY"), "search-then-open"),
    (("OPEN_SRC_PY", "NAV_SRC_PY"), "open-navigate"),
    (("NAV_SRC_PY", "NAV_SRC_PY"), "in-file scrolling"),
    (("FIND_FILE", "FIND_FILE"), "repeated find-file"),
    (("SEARCH", "SEARCH"), "repeated search"),
    (("EDIT_SRC_PY", "EDIT_SRC_PY"), "edit burst"),
    (("EDIT_TEST_PY", "EDIT_TEST_PY"), "test-edit burst"),
    (("RUN_PYTHON_TEST_PY", "RUN_PYTHON_TEST_PY"), "test reruns"),
    (("SHELL_RM", "SUBMIT"), "cleanup-then-submit"),
]


def motif_phrase(motif: str) -> str:
    atoms = motif.split("+")
    for atom_pattern, phrase in MOTIF_PHRASES:
        if atoms == list(atom_pattern):
            return phrase
        if len(atoms) >= len(atom_pattern):
            if atoms[: len(atom_pattern)] == list(atom_pattern):
                return phrase
    # fallback: compress consecutive repeats, take first + last
    if len(atoms) <= 2:
        return " -> ".join(a.replace("_PY", "").replace("SHELL_", "").replace("_", " ").lower() for a in atoms)
    return f"{atoms[0].replace('_PY', '').lower()} -> ... -> {atoms[-1].replace('_PY', '').lower()}"


def derive_cluster_label(summary: dict) -> str:
    """Human-readable label from a cluster summary."""
    if summary["is_noise"]:
        return "generic (no dense cluster)"
    top_motifs = summary.get("top_motifs", [])
    if not top_motifs:
        return f"cluster {summary['cluster_id']}"
    top_phrase = motif_phrase(top_motifs[0]["motif"])
    dom = summary["dominant_agent"]
    frac = summary["dominant_agent_fraction"]
    if frac >= 0.7:
        return f"{top_phrase} ({dom} {int(frac * 100)}%)"
    if frac >= 0.55:
        return f"{top_phrase} ({dom}-lean {int(frac * 100)}%)"
    return f"{top_phrase} (mixed)"


def load_records() -> list[dict]:
    out = []
    with open(SEQ_PATH) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_difficulty() -> dict[str, int]:
    out = {}
    with open(DIVERSITY_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["instance_id"]] = int(row["n_resolved"])
    return out


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


def cluster_and_embed(
    X: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    # Cluster on the high-dim motif-frequency space, not on the UMAP projection.
    # Use cosine distance (via L2-normalized vectors + euclidean) to match the
    # intuition of "how aligned are these two agents' motif mixes".
    X_l2 = normalize(X, norm="l2", axis=1)
    clusterer = HDBSCAN(
        min_cluster_size=10, min_samples=3,
        metric="euclidean", cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_l2)

    # UMAP for visualization only — not used for clustering.
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.25, metric="cosine",
        random_state=seed, n_components=2,
    )
    coords = reducer.fit_transform(X)
    return coords, labels


def summarize_clusters(
    records: list[dict],
    labels: np.ndarray,
    difficulty: dict[str, int],
) -> list[dict]:
    clusters = {}
    for i, r in enumerate(records):
        lbl = int(labels[i])
        clusters.setdefault(lbl, []).append((i, r))

    summaries = []
    for lbl, items in sorted(clusters.items()):
        idxs = [i for i, _ in items]
        rs = [r for _, r in items]
        agent_comp = Counter(r["agent"] for r in rs)
        difficulty_comp = Counter(difficulty.get(r["instance_id"], -1) for r in rs)
        motif_counts: Counter = Counter()
        for r in rs:
            motif_counts.update(r["bpe"])
        total_tokens = sum(motif_counts.values())
        top_motifs = [
            {"motif": m, "count": c, "share": c / total_tokens if total_tokens else 0}
            for m, c in motif_counts.most_common()
            if "+" in m
        ][:10]

        agent_dominant = max(agent_comp, key=agent_comp.get)
        dom_agent_frac = agent_comp[agent_dominant] / len(rs)

        s = {
            "cluster_id": lbl,
            "is_noise": lbl == -1,
            "n_trajectories": len(rs),
            "n_unique_tasks": len({r["instance_id"] for r in rs}),
            "agent_composition": dict(agent_comp),
            "dominant_agent": agent_dominant,
            "dominant_agent_fraction": round(dom_agent_frac, 3),
            "difficulty_composition": {str(k): v for k, v in sorted(difficulty_comp.items())},
            "mean_canonical_length": float(np.mean([r["canonical_length"] for r in rs])),
            "mean_bpe_length": float(np.mean([r["bpe_length"] for r in rs])),
            "mean_compression": float(np.mean([r["compression"] for r in rs])),
            "top_motifs": top_motifs,
        }
        s["label"] = derive_cluster_label(s)
        summaries.append(s)
    return summaries


def plot_umap(
    coords: np.ndarray,
    records: list[dict],
    labels: np.ndarray,
    summaries: list[dict],
    out_path: Path,
) -> None:
    label_by_id = {s["cluster_id"]: s["label"] for s in summaries}

    # Build one row per trajectory
    rows = []
    for i, r in enumerate(records):
        cid = int(labels[i])
        rows.append({
            "umap1": float(coords[i, 0]),
            "umap2": float(coords[i, 1]),
            "agent": r["agent"],
            "cluster_id": cid,
            "cluster_label": label_by_id.get(cid, "noise") if cid != -1 else "noise",
            "is_noise": cid == -1,
        })
    df = pd.DataFrame(rows)

    # --- Panel 1: colored by agent ---
    agent_domain = sorted(AGENT_COLORS.keys())
    agent_range = [AGENT_COLORS[a] for a in agent_domain]

    panel1 = (
        alt.Chart(df)
        .mark_point(size=20, filled=True, opacity=0.55, strokeWidth=0)
        .encode(
            x=alt.X("umap1:Q", axis=alt.Axis(title="UMAP-1")),
            y=alt.Y("umap2:Q", axis=alt.Axis(title="UMAP-2")),
            color=alt.Color(
                "agent:N",
                scale=alt.Scale(domain=agent_domain, range=agent_range),
                legend=alt.Legend(orient="bottom", title=None),
            ),
        )
        .properties(
            width=360,
            height=300,
            title=alt.TitleParams(text="Colored by agent", fontSize=13, color="#111111", anchor="start"),
        )
    )

    # --- Panel 2: colored by cluster ---
    unique_real_cluster_ids = sorted(set(labels[labels != -1].tolist()))
    cluster_labels_ordered = [label_by_id.get(c, f"cluster {c}") for c in unique_real_cluster_ids]
    palette = [SKY, ORANGE, GREEN, VERMILLION, BLUE, "#CC79A7", "#F0E442", NEAR_BLACK]
    cluster_colors = palette[:len(unique_real_cluster_ids)]

    df_noise = df[df["is_noise"]].copy()
    df_cluster = df[~df["is_noise"]].copy()

    noise_layer = (
        alt.Chart(df_noise)
        .mark_point(size=10, opacity=0.2, color=GRAY, strokeWidth=0, filled=True)
        .encode(
            x=alt.X("umap1:Q", axis=alt.Axis(title="UMAP-1")),
            y=alt.Y("umap2:Q", axis=alt.Axis(title="UMAP-2")),
        )
    )

    cluster_layer = (
        alt.Chart(df_cluster)
        .mark_point(size=30, opacity=0.85, filled=True, strokeWidth=0.5)
        .encode(
            x=alt.X("umap1:Q", axis=alt.Axis(title="UMAP-1")),
            y=alt.Y("umap2:Q", axis=alt.Axis(title="UMAP-2")),
            color=alt.Color(
                "cluster_label:N",
                scale=alt.Scale(domain=cluster_labels_ordered, range=cluster_colors),
                legend=alt.Legend(orient="bottom", title=None),
            ),
        )
    )

    panel2 = (
        (noise_layer + cluster_layer)
        .properties(
            width=360,
            height=300,
            title=alt.TitleParams(text="Colored by cluster", fontSize=13, color="#111111", anchor="start"),
        )
    )

    chart = (
        alt.hconcat(panel1, panel2)
        .properties(
            title=alt.TitleParams(
                text="Trajectory clusters (UMAP + HDBSCAN)",
                fontSize=13,
                color="#111111",
                anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axisY(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def plot_cluster_profile(
    summaries: list[dict],
    out_path: Path,
    top_n: int = 5,
) -> None:
    real_clusters = sorted(
        [s for s in summaries if not s["is_noise"]],
        key=lambda s: -s["n_trajectories"],
    )[:6]

    if not real_clusters:
        chart = (
            alt.Chart(pd.DataFrame([{"text": "No dense clusters found"}]))
            .mark_text(fontSize=13, color="#111111")
            .encode(
                x=alt.value(200),
                y=alt.value(60),
                text="text:N",
            )
            .properties(width=400, height=120)
            .configure_view(strokeWidth=0)
        )
        chart.save(str(out_path), scale_factor=2)
        return

    def abbrev(m: str, maxl: int = 22) -> str:
        parts = m.split("+")
        if len(parts) <= 2:
            sstr = " -> ".join(parts)
        else:
            sstr = f"{parts[0]} -> ... -> {parts[-1]} ({len(parts)})"
        return sstr if len(sstr) <= maxl else sstr[: maxl - 1] + "..."

    # --- Chart 1: agent composition (stacked bar, one row per cluster) ---
    comp_rows = []
    cluster_order = [s["label"] for s in real_clusters]  # largest-first order
    for s in real_clusters:
        total = s["n_trajectories"] or 1
        for agent, count in s["agent_composition"].items():
            comp_rows.append({
                "cluster_label": s["label"],
                "agent": agent,
                "count": count,
                "fraction": count / total,
            })
    df_comp = pd.DataFrame(comp_rows)

    agent_domain = sorted(AGENT_COLORS.keys())
    agent_range = [AGENT_COLORS[a] for a in agent_domain]

    chart1 = (
        alt.Chart(df_comp)
        .mark_bar()
        .encode(
            y=alt.Y(
                "cluster_label:N",
                sort=cluster_order,
                axis=alt.Axis(title=None, labelLimit=300),
            ),
            x=alt.X(
                "fraction:Q",
                stack="normalize",
                axis=alt.Axis(title="agent share", format=".0%"),
            ),
            color=alt.Color(
                "agent:N",
                scale=alt.Scale(domain=agent_domain, range=agent_range),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            order=alt.Order("agent:N", sort="ascending"),
        )
        .properties(
            width=500,
            height=max(len(real_clusters) * 25, 60),
        )
    )

    # --- Chart 2: top motifs per cluster (faceted horizontal bar) ---
    motif_rows = []
    for s in real_clusters:
        for m in s["top_motifs"][:top_n]:
            motif_rows.append({
                "cluster_label": s["label"],
                "motif_label": abbrev(m["motif"]),
                "share": m["share"],
            })
    df_motifs = pd.DataFrame(motif_rows)

    if not df_motifs.empty:
        chart2 = (
            alt.Chart(df_motifs)
            .mark_bar(color=BLUE)
            .encode(
                y=alt.Y(
                    "motif_label:N",
                    sort=alt.EncodingSortField(field="share", op="sum", order="descending"),
                    axis=alt.Axis(title=None, labelLimit=200),
                ),
                x=alt.X(
                    "share:Q",
                    axis=alt.Axis(title="share of cluster actions", format=".1%"),
                ),
                facet=alt.Facet(
                    "cluster_label:N",
                    sort=cluster_order,
                    header=alt.Header(title=None, labelFontSize=10),
                ),
            )
            .properties(width=130, height=top_n * 22)
            .resolve_scale(y="independent")
        )
        combined = alt.vconcat(chart1, chart2)
    else:
        combined = chart1

    chart = (
        combined
        .properties(
            title=alt.TitleParams(
                text="Cluster composition by agent",
                fontSize=13,
                color="#111111",
                anchor="start",
            )
        )
        .configure_view(strokeWidth=0)
        .configure_axisY(grid=True, gridColor="#F0F0F0", gridWidth=0.3)
    )

    chart.save(str(out_path), scale_factor=2)


def main() -> int:
    records = load_records()
    difficulty = load_difficulty()
    print(f"Loaded {len(records)} trajectories")

    X, vocab = build_matrix(records)
    print(f"Matrix shape {X.shape}; vocab size {len(vocab)}")

    coords, labels = cluster_and_embed(X, seed=0)
    summaries = summarize_clusters(records, labels, difficulty)

    n_clusters = sum(1 for s in summaries if not s["is_noise"])
    n_noise = sum(s["n_trajectories"] for s in summaries if s["is_noise"])
    print(f"\n{n_clusters} clusters + {n_noise} noise points")
    for s in summaries:
        tag = "NOISE" if s["is_noise"] else f"cluster {s['cluster_id']}"
        print(f"  {tag}: n={s['n_trajectories']}, "
              f"agents={s['agent_composition']}, "
              f"dom={s['dominant_agent']} {s['dominant_agent_fraction']*100:.0f}%, "
              f"mean atoms={s['mean_canonical_length']:.1f}")
        if s["top_motifs"]:
            top3 = ", ".join(m["motif"][:40] for m in s["top_motifs"][:3])
            print(f"    top motifs: {top3}")

    (OUT / "trajectory_clusters.json").write_text(json.dumps({
        "n_trajectories": len(records),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "vocab_size": len(vocab),
        "clusters": summaries,
    }, indent=2, default=str))
    plot_umap(coords, records, labels, summaries, OUT / "trajectory_clusters_umap.png")
    plot_cluster_profile(summaries, OUT / "trajectory_clusters_profile.png")

    print(f"\nSaved:")
    print(f"  {OUT / 'trajectory_clusters.json'}")
    print(f"  {OUT / 'trajectory_clusters_umap.png'}")
    print(f"  {OUT / 'trajectory_clusters_profile.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
