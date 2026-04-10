#!/usr/bin/env python3
"""
Horse race: which representation best captures the platonic form of a fix?

Compares six representations on a single criterion: do nearest neighbors
by this representation co-pass or co-fail more than chance?

Metric: kNN pass/fail prediction F1 (k=5, leave-one-out).
Also sweeps k=1..15 to show stability.

Representations:
  1. edit_cert     -- Jaccard on normalized edit op sets (from distances.parquet)
  2. motif         -- motif sequence distance (from distances.parquet)
  3. fix_type      -- one-hot 13-type taxonomy
  4. staged_embed  -- sentence embedding of grounded staged narrative
  5. cot_embed     -- sentence embedding of model's free-form plan (no_context)
  6. fim_jaccard   -- Jaccard on which FIM patterns an instance matches

Usage:
  uv run python scripts/compare_representations.py
"""

import difflib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "representation_comparison"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
REP_COLORS = {
    "edit_cert":    "#0C6583",
    "motif":        "#2B2D42",
    "fix_type":     "#AAAAAA",
    "staged_embed": "#EE7733",
    "cot_embed":    "#56B4E9",
    "fim_jaccard":  "#009E73",
}
REP_LABELS = {
    "edit_cert":    "Edit cert (Jaccard)",
    "motif":        "Motif distance",
    "fix_type":     "Fix type (13 classes)",
    "staged_embed": "Staged narrative (embedding)",
    "cot_embed":    "CoT free-form (embedding)",
    "fim_jaccard":  "FIM pattern overlap",
}

_NORMALIZE_OPS = {
    "ADD_if": "ADD_If", "DEL_if": "DEL_If",
    "ADD_for": "ADD_For", "DEL_for": "DEL_For",
    "ADD_return": "ADD_Return", "DEL_return": "DEL_Return",
    "ADD_raise": "ADD_Raise", "DEL_raise": "DEL_Raise",
    "ADD_try": "ADD_Try", "DEL_try": "DEL_Try",
    "ADD_while": "ADD_While", "DEL_while": "DEL_While",
    "ADD_with": "ADD_With", "DEL_with": "DEL_With",
    "ADD_def": "ADD_FunctionDef", "DEL_def": "DEL_FunctionDef",
    "ADD_class": "ADD_ClassDef", "DEL_class": "DEL_ClassDef",
    "ADD_elif": "ADD_If", "DEL_elif": "DEL_If",
    "ADD_else": "ADD_If", "DEL_else": "DEL_If",
    "ADD_except": "ADD_ExceptHandler", "DEL_except": "DEL_ExceptHandler",
    "ADD_assert": "ADD_Assert",
}


# --- Distance matrix builders ---

def load_certs(traces_path: Path) -> dict[str, frozenset[str]]:
    certs = {}
    with open(traces_path) as f:
        for line in f:
            trace = json.loads(line)
            ops = []
            for ev in trace["events"]:
                if ev["type"] != "code_change":
                    continue
                d = ev["details"]
                if not d["file_path"].endswith(".py"):
                    continue
                before = d["before_content"].splitlines(keepends=True)
                after = d["after_content"].splitlines(keepends=True)
                raw = "".join(difflib.unified_diff(
                    before, after, fromfile=d["file_path"], tofile=d["file_path"]
                ))
                if not raw:
                    continue
                diff = f"diff --git a/{d['file_path']} b/{d['file_path']}\n" + raw
                ops.extend(patch_to_ast_sequence(diff))
            if ops:
                norm = frozenset(_NORMALIZE_OPS.get(op, op) for op in ops)
                certs[trace["instance_id"]] = norm
    return certs


def jaccard_dist(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def pairwise_jaccard(instances: list[str], sets: dict[str, frozenset]) -> np.ndarray:
    n = len(instances)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_dist(sets[instances[i]], sets[instances[j]])
            mat[i, j] = mat[j, i] = d
    return mat


def fixtype_distance(instances: list[str], type_map: dict[str, str]) -> np.ndarray:
    n = len(instances)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if type_map.get(instances[i]) == type_map.get(instances[j]):
                mat[i, j] = 0.0
    np.fill_diagonal(mat, 0.0)
    return mat


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return normalize(embeddings)


def cosine_dist_matrix(embeddings: np.ndarray) -> np.ndarray:
    sim = embeddings @ embeddings.T
    return 1.0 - np.clip(sim, -1, 1)


def fim_jaccard_dist(instances: list[str], pattern_sets: dict[str, frozenset]) -> np.ndarray:
    return pairwise_jaccard(instances, pattern_sets)


# --- Evaluation ---

def knn_f1_loo(dist_matrix: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = len(labels)
    preds = []
    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf
        neighbors = np.argsort(row)[:k]
        neighbor_labels = labels[neighbors]
        pred = 1 if neighbor_labels.mean() >= 0.5 else 0
        preds.append(pred)
    return f1_score(labels, preds, zero_division=0)


def sweep_k(dist_matrix: np.ndarray, labels: np.ndarray,
            k_range: range) -> list[float]:
    return [knn_f1_loo(dist_matrix, labels, k) for k in k_range]


# --- Plotting ---

def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def fig1_k_sweep(results: dict[str, list[float]], k_range: range, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(bottom=0.15, right=0.72)
    style_panel(ax)

    ks = list(k_range)
    for rep, f1s in sorted(results.items(), key=lambda x: -max(x[1])):
        ax.plot(ks, f1s, color=REP_COLORS[rep], linewidth=1.8,
                label=REP_LABELS[rep], marker="o", markersize=3)

    ax.set_xlabel("k (nearest neighbors)", fontsize=9)
    ax.set_ylabel("Pass/fail prediction F1 (leave-one-out)", fontsize=9)
    ax.set_title("Representation comparison: kNN pass/fail prediction", fontsize=11,
                 pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))

    fig.savefig(output_dir / "fig1_k_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_k_sweep.png")


def fig2_at_k5(results: dict[str, list[float]], k_range: range, output_dir: Path):
    k5_idx = list(k_range).index(5)
    scores = {rep: f1s[k5_idx] for rep, f1s in results.items()}
    sorted_reps = sorted(scores, key=lambda r: -scores[r])

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(bottom=0.3)
    style_panel(ax)

    xs = np.arange(len(sorted_reps))
    colors = [REP_COLORS[r] for r in sorted_reps]
    ax.bar(xs, [scores[r] for r in sorted_reps], color=colors, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([REP_LABELS[r] for r in sorted_reps],
                       fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("F1 at k=5", fontsize=9)
    ax.set_title("Pass/fail prediction at k=5 nearest neighbors", fontsize=11,
                 pad=6, fontweight="normal")

    fig.savefig(output_dir / "fig2_at_k5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_at_k5.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ground truth pass/fail
    fix_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "fix_type", "passed"]]

    print("Loading edit certificates...")
    certs = load_certs(ROOT / "output" / "resolved_traces_lite_full.jsonl")

    # Align: use instances with both cert and pass/fail label
    common = sorted(set(certs) & set(fix_df["instance_id"]))
    print(f"  {len(common)} instances with both cert and pass/fail label")

    fix_sub = fix_df[fix_df["instance_id"].isin(common)].set_index("instance_id")
    labels = np.array([int(fix_sub.loc[iid, "passed"]) for iid in common])
    print(f"  Pass: {labels.sum()}, Fail: {(1-labels).sum()}")

    dist_matrices: dict[str, np.ndarray] = {}

    # 1. Edit cert Jaccard
    print("Building edit cert distance matrix...")
    dist_matrices["edit_cert"] = pairwise_jaccard(
        common, {iid: certs[iid] for iid in common}
    )

    # 2. Motif distance (from precomputed matrices)
    print("Loading motif distances...")
    all_labels = pd.read_parquet(ROOT / "output" / "labels.parquet")
    iid_to_idx = all_labels.set_index("instance_id")["index"].to_dict()
    dist_parquet = pd.read_parquet(ROOT / "output" / "distances.parquet")

    common_idxs = [iid_to_idx[iid] for iid in common if iid in iid_to_idx]
    common_mapped = [iid for iid in common if iid in iid_to_idx]
    n = len(common_mapped)
    motif_mat = np.ones((n, n))
    np.fill_diagonal(motif_mat, 0.0)
    idx_pos = {idx: pos for pos, idx in enumerate(common_idxs)}
    for _, row in dist_parquet.iterrows():
        i, j = int(row["i"]), int(row["j"])
        if i in idx_pos and j in idx_pos:
            pi, pj = idx_pos[i], idx_pos[j]
            motif_mat[pi, pj] = motif_mat[pj, pi] = row["d_motifs"]
    dist_matrices["motif"] = motif_mat
    labels_motif = np.array([int(fix_sub.loc[iid, "passed"]) for iid in common_mapped])

    # 3. Fix type distance
    print("Building fix type distance matrix...")
    type_map = fix_sub["fix_type"].to_dict()
    dist_matrices["fix_type"] = fixtype_distance(common, type_map)

    # 4. Staged narrative embedding
    print("Loading staged narratives...")
    with open(ROOT / "output" / "staged_descriptions.json") as f:
        sd = json.load(f)
    staged_map = {r["instance_id"]: r["staged_narrative"] for r in sd["results"]}
    staged_common = [iid for iid in common if iid in staged_map]
    staged_texts = [staged_map[iid] for iid in staged_common]
    staged_labels = np.array([int(fix_sub.loc[iid, "passed"]) for iid in staged_common])
    print(f"  Embedding {len(staged_texts)} staged narratives...")
    staged_emb = embed_texts(staged_texts)
    dist_matrices["staged_embed"] = cosine_dist_matrix(staged_emb)

    # 5. CoT free-form embedding (no_context responses)
    print("Loading CoT responses...")
    with open(ROOT / "output" / "prompting_study" / "gpt_4o" / "records.json") as f:
        records = json.load(f)
    cot_map = {r["instance_id"]: r["conditions"]["no_context"].get("response", "")
               for r in records}
    cot_common = [iid for iid in common if iid in cot_map and cot_map[iid]]
    cot_texts = [cot_map[iid] for iid in cot_common]
    cot_labels = np.array([int(fix_sub.loc[iid, "passed"]) for iid in cot_common])
    print(f"  Embedding {len(cot_texts)} CoT responses...")
    cot_emb = embed_texts(cot_texts)
    dist_matrices["cot_embed"] = cosine_dist_matrix(cot_emb)

    # 6. FIM pattern Jaccard
    print("Loading FIM patterns...")
    fim_path = ROOT / "output" / "strategy_forms" / "frequent_itemsets.json"
    with open(fim_path) as f:
        fim_data = json.load(f)
    fim_patterns = [frozenset(p["itemset"]) for p in fim_data["patterns"]]
    fim_sets = {
        iid: frozenset(i for i, pat in enumerate(fim_patterns)
                       if pat.issubset(certs[iid]))
        for iid in common
    }
    dist_matrices["fim_jaccard"] = pairwise_jaccard(common, fim_sets)

    # Evaluate all representations
    k_range = range(1, 16)
    results: dict[str, list[float]] = {}

    # Match label arrays to each representation's instance set
    rep_instances = {
        "edit_cert": (common, labels),
        "fix_type": (common, labels),
        "fim_jaccard": (common, labels),
        "motif": (common_mapped, labels_motif),
        "staged_embed": (staged_common, staged_labels),
        "cot_embed": (cot_common, cot_labels),
    }

    print("\nEvaluating representations (kNN F1 sweep)...")
    for rep, mat in dist_matrices.items():
        _, rep_labels = rep_instances[rep]
        f1s = sweep_k(mat, rep_labels, k_range)
        results[rep] = f1s
        print(f"  {REP_LABELS[rep]:35s}  F1@k=5={f1s[4]:.3f}  max={max(f1s):.3f}")

    # Save results
    results_df = pd.DataFrame(results, index=list(k_range))
    results_df.index.name = "k"
    results_df.to_csv(OUTPUT_DIR / "knn_f1_results.csv")
    print("\nSaved knn_f1_results.csv")

    print("\nGenerating figures...")
    fig1_k_sweep(results, k_range, OUTPUT_DIR)
    fig2_at_k5(results, k_range, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
