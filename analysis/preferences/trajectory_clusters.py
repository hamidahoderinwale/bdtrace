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
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    agents = [r["agent"] for r in records]
    unique_agents = sorted(set(agents))
    for a in unique_agents:
        mask = np.array([ag == a for ag in agents])
        axes[0].scatter(
            coords[mask, 0], coords[mask, 1],
            s=14, color=AGENT_COLORS.get(a, "#888"), alpha=0.55,
            edgecolor="none", label=f"{a} (n={int(mask.sum())})",
        )
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].set_title("Colored by agent", fontsize=11)
    axes[0].legend(fontsize=9, frameon=False, loc="best")
    axes[0].spines[["top", "right"]].set_visible(False)

    unique_clusters = sorted(set(labels.tolist()))
    n_real = sum(1 for c in unique_clusters if c != -1)
    cmap = plt.cm.get_cmap("tab10" if n_real <= 10 else "tab20", max(n_real, 1))
    color_list = [
        cmap(i) if c != -1 else (0.7, 0.7, 0.7, 0.5)
        for i, c in enumerate([c for c in unique_clusters if c != -1])
    ]
    cluster_color_map = {c: color_list[i] for i, c in enumerate([cc for cc in unique_clusters if cc != -1])}
    cluster_color_map[-1] = (0.82, 0.82, 0.82, 0.55)

    label_by_id = {s["cluster_id"]: s["label"] for s in summaries}

    # draw noise points first (behind), then real clusters on top
    for c in unique_clusters:
        mask = labels == c
        if c == -1:
            axes[1].scatter(
                coords[mask, 0], coords[mask, 1],
                s=10, color=cluster_color_map[c], edgecolor="none",
                alpha=0.35, label=f"generic (n={int(mask.sum())})",
            )
    for c in unique_clusters:
        if c == -1:
            continue
        mask = labels == c
        axes[1].scatter(
            coords[mask, 0], coords[mask, 1],
            s=28, color=cluster_color_map[c], edgecolor="white", linewidth=0.6,
            alpha=0.95,
        )
        cx = float(coords[mask, 0].mean())
        cy = float(coords[mask, 1].mean())
        lbl_text = label_by_id.get(c, f"cluster {c}")
        axes[1].annotate(
            f"{lbl_text}\nn={int(mask.sum())}",
            xy=(cx, cy),
            xytext=(cx, cy),
            fontsize=7,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=cluster_color_map[c], alpha=0.9, linewidth=1.0),
            zorder=5,
        )

    axes[1].set_xlabel("UMAP-1")
    axes[1].set_title(f"Colored by HDBSCAN cluster ({n_real} clusters; labels are top-motif + dominant agent)", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False, loc="best")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Trajectory clusters: do trajectories group by agent, by task, or by procedural phase?\n"
        "Each dot is one trajectory. Left: colored by which agent ran it. Right: clusters found without using agent labels.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no real clusters found (all noise)",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(real_clusters)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 7), gridspec_kw={"height_ratios": [1.3, 2]})
    if n == 1:
        axes = axes.reshape(2, 1)

    agent_order = sorted({a for s in real_clusters for a in s["agent_composition"]})
    colors = [AGENT_COLORS.get(a, "#888") for a in agent_order]

    for i, s in enumerate(real_clusters):
        ax = axes[0, i]
        vals = [s["agent_composition"].get(a, 0) for a in agent_order]
        total = sum(vals) or 1
        fracs = [v / total for v in vals]
        bottom = 0
        for frac, col, a in zip(fracs, colors, agent_order):
            ax.bar(0, frac, bottom=bottom, color=col, edgecolor="white", width=0.6, label=a if i == 0 else None)
            if frac > 0.05:
                ax.text(0, bottom + frac / 2, f"{a}\n{int(frac*total)}", ha="center", va="center",
                        fontsize=8, color="white" if frac > 0.2 else "black")
            bottom += frac
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_title(
            f"{s['label']}\n(cluster {s['cluster_id']}, n={s['n_trajectories']})",
            fontsize=8,
        )
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        if i == 0:
            ax.set_ylabel("agent share")

        ax = axes[1, i]
        top = s["top_motifs"][:top_n]
        if not top:
            ax.axis("off")
            continue
        motifs = [m["motif"] for m in top]
        shares = [m["share"] for m in top]

        def abbrev(m: str, maxl: int = 22) -> str:
            parts = m.split("+")
            if len(parts) <= 2:
                sstr = m.replace("+", "->")
            else:
                sstr = f"{parts[0]}->...->{parts[-1]} ({len(parts)})"
            return sstr if len(sstr) <= maxl else sstr[: maxl - 1] + "..."

        y = np.arange(len(top))
        ax.barh(y, shares, color="#5d90e0", edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels([abbrev(m) for m in motifs], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("share of cluster actions", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v*100:.1f}%"))

    fig.suptitle(
        f"Per-cluster composition (top) and top {top_n} action patterns (bottom)\n"
        "If 'dom' fraction ~33%, cluster cuts across agents (type-driven). If >60%, cluster is agent-driven.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
