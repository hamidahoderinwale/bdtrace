#!/usr/bin/env python3
"""
Test: do convergent semantic descriptions cluster the same way FIM structural patterns do?

If yes: structure alone is sufficient for canonical forms.
If no: context matters and a richer representation is needed.

Method:
  1. Take instances with high cross-model description convergence
  2. Embed their descriptions (semantic space)
  3. Cluster embeddings at multiple k values
  4. Compare to FIM structural assignments via ARI and NMI
  5. UMAP visualization colored by both

Usage:
  uv run python scripts/test_form_alignment.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _env in [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    if _env.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_env)
        except ImportError:
            pass

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "form_alignment"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"

MODEL_DIRS = {
    "gpt_4o": "GPT-4o",
    "gpt_4o_mini": "GPT-4o mini",
    "qwen_2.5_72b_instruct": "Qwen 2.5 72B",
    "llama_3.3_70b_instruct": "Llama 3.3 70B",
}


def load_convergent_responses(study_dir: Path,
                               convergence_df: pd.DataFrame,
                               threshold: float,
                               condition: str = "no_context") -> dict[str, str]:
    convergent_ids = set(
        convergence_df[convergence_df["convergence_score"] >= threshold]["instance_id"]
    )
    # Use GPT-4o responses as representative (most coverage)
    with open(study_dir / "gpt_4o" / "records.json") as f:
        records = json.load(f)
    responses = {}
    for r in records:
        iid = r["instance_id"]
        if iid not in convergent_ids:
            continue
        resp = r["conditions"].get(condition, {}).get("response", "")
        if resp:
            responses[iid] = resp
    return responses


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig_ari_sweep(k_values, ari_scores, nmi_scores, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(bottom=0.15)
    style_panel(ax)

    ax.plot(k_values, ari_scores, color=TEAL, marker="o", markersize=4,
            linewidth=1.5, label="ARI")
    ax.plot(k_values, nmi_scores, color=ORANGE, marker="s", markersize=4,
            linewidth=1.5, linestyle="--", label="NMI")
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("k (semantic clusters)", fontsize=9)
    ax.set_ylabel("Alignment with FIM structural forms", fontsize=9)
    ax.set_title("Semantic vs structural cluster alignment across k", fontsize=11,
                 pad=6, fontweight="normal")
    ax.legend(fontsize=9, frameon=False)

    fig.savefig(output_dir / "fig1_ari_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_ari_sweep.png")


def fig_umap(embeddings, structural_labels, semantic_labels,
             instance_ids, output_dir: Path):
    try:
        from umap import UMAP
    except ImportError:
        print("umap not available, skipping UMAP plot")
        return

    print("  Running UMAP...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.3, bottom=0.1)

    unique_struct = sorted(set(structural_labels))
    unique_sem = sorted(set(semantic_labels))
    struct_cmap = plt.cm.get_cmap("tab20", len(unique_struct))
    sem_cmap = plt.cm.get_cmap("tab20", len(unique_sem))

    for ax, labels, unique_labels, cmap, title in [
        (axes[0], structural_labels, unique_struct, struct_cmap,
         "Colored by FIM structural form"),
        (axes[1], semantic_labels, unique_sem, sem_cmap,
         "Colored by semantic cluster (k-means)"),
    ]:
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(PANEL_EDGE)
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=cmap(i), s=18, alpha=0.7, label=str(label)[:20])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, pad=6, fontweight="normal")

    fig.suptitle("Description embedding space: structural vs semantic clustering",
                 fontsize=11, y=1.01, fontweight="normal")
    fig.savefig(output_dir / "fig2_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_umap.png")


def fig_confusion(structural_labels, semantic_labels,
                  k_best, output_dir: Path):
    struct_le = LabelEncoder().fit(structural_labels)
    sem_le = LabelEncoder().fit(semantic_labels)
    s = struct_le.transform(structural_labels)
    e = sem_le.transform(semantic_labels)

    n_struct = len(struct_le.classes_)
    n_sem = len(sem_le.classes_)
    mat = np.zeros((n_struct, n_sem), dtype=int)
    for si, ei in zip(s, e):
        mat[si, ei] += 1

    # Normalize by structural form size
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = mat / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_sem * 0.6), max(6, n_struct * 0.4)))
    fig.subplots_adjust(left=0.3, bottom=0.2)

    im = ax.imshow(mat_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_sem))
    ax.set_xticklabels([f"S{i}" for i in range(n_sem)], fontsize=8)
    ax.set_yticks(range(n_struct))
    ax.set_yticklabels([c[:30] for c in struct_le.classes_], fontsize=7)
    ax.set_xlabel(f"Semantic cluster (k={k_best})", fontsize=9)
    ax.set_ylabel("FIM structural form", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Fraction of form")
    ax.set_title("Alignment between structural forms and semantic clusters",
                 fontsize=10, pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig3_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_confusion.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load assignments
    assign_df = pd.read_parquet(
        ROOT / "output" / "canonical_forms" / "instance_assignments.parquet"
    )

    # Use convergence >= 0.80, assigned instances only
    threshold = 0.80
    working = assign_df[
        (assign_df["convergence_score"] >= threshold) &
        (assign_df["assigned"])
    ].copy()
    print(f"Working set: {len(working)} instances "
          f"(convergence >= {threshold}, assigned to a form)")

    # Load responses
    study_dir = ROOT / "output" / "prompting_study"
    responses = load_convergent_responses(study_dir, working, threshold=0.0)
    working = working[working["instance_id"].isin(responses)]
    print(f"With GPT-4o responses: {len(working)} instances")

    instance_ids = working["instance_id"].tolist()
    structural_labels = working.set_index("instance_id")["form_name"].tolist()

    # Coarsen structural labels: merge forms with n<5 into "other_structural"
    form_counts = working["form_name"].value_counts()
    structural_labels_coarse = [
        label if form_counts.get(label, 0) >= 5 else "other"
        for label in structural_labels
    ]
    n_struct_forms = len(set(structural_labels_coarse))
    print(f"Structural forms (n>=5 + other): {n_struct_forms}")
    print("  Form distribution:",
          dict(pd.Series(structural_labels_coarse).value_counts()))

    # Embed responses
    print("\nEmbedding responses...")
    texts = [responses[iid][:800] for iid in instance_ids]
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = normalize(embeddings)

    # Sweep k and compute ARI/NMI against coarsened structural labels
    k_values = list(range(3, 26))
    ari_scores = []
    nmi_scores = []

    print("\nSweeping k...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        sem_labels = km.fit_predict(embeddings)
        ari = adjusted_rand_score(structural_labels_coarse, sem_labels)
        nmi = normalized_mutual_info_score(structural_labels_coarse, sem_labels)
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        print(f"  k={k:2d}: ARI={ari:.3f}, NMI={nmi:.3f}")

    best_k_idx = np.argmax(ari_scores)
    best_k = k_values[best_k_idx]
    best_ari = ari_scores[best_k_idx]
    best_nmi = nmi_scores[best_k_idx]
    print(f"\nBest alignment at k={best_k}: ARI={best_ari:.3f}, NMI={best_nmi:.3f}")

    # Interpretation
    print("\nInterpretation:")
    if best_ari < 0.05:
        print("  ARI near zero: semantic and structural clusters are essentially independent.")
        print("  Structure alone is NOT sufficient. Context/semantics capture different variation.")
    elif best_ari < 0.15:
        print("  ARI low: weak alignment. Some shared structure but mostly independent.")
        print("  The representations capture different aspects of fix strategy.")
    elif best_ari < 0.30:
        print("  ARI moderate: partial alignment. Structure and semantics partially agree.")
        print("  Combined representation likely better than either alone.")
    else:
        print("  ARI high: strong alignment. Structure and semantics largely agree.")
        print("  FIM forms are a good proxy for semantic intent.")

    # Best k semantic labels for figures
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    sem_labels_best = km_best.fit_predict(embeddings).tolist()

    # Save results
    results = {
        "threshold": threshold,
        "n_instances": len(working),
        "n_structural_forms": n_struct_forms,
        "best_k": best_k,
        "best_ari": float(best_ari),
        "best_nmi": float(best_nmi),
        "sweep": [{"k": k, "ari": float(a), "nmi": float(n)}
                  for k, a, n in zip(k_values, ari_scores, nmi_scores)],
    }
    with open(OUTPUT_DIR / "alignment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved alignment_results.json")

    print("\nGenerating figures...")
    fig_ari_sweep(k_values, ari_scores, nmi_scores, OUTPUT_DIR)
    fig_umap(embeddings, structural_labels_coarse, sem_labels_best,
             instance_ids, OUTPUT_DIR)
    fig_confusion(structural_labels_coarse, sem_labels_best, best_k, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
