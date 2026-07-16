#!/usr/bin/env python3
"""
Cluster free-form hunk descriptions to discover mechanism categories from data.

Pipeline:
  1. Load hunk descriptions from build_hunk_descriptions.py output
  2. Embed all descriptions with sentence-transformers (all-MiniLM-L6-v2)
  3. HDBSCAN clustering on embeddings — discovers k without pre-specifying
  4. Label each cluster by TF-IDF top terms over member descriptions
  5. Map clusters back to instances: binary feature matrix (instance x cluster)
  6. Decision tree on feature matrix to discover fix forms
  7. Frontier analysis against leaderboard agents

Outputs:
  output/hunk_clusters/cluster_labels.json     — cluster definitions + top terms
  output/hunk_clusters/form_assignments.parquet — per-instance form assignments
  output/hunk_clusters/form_summaries.json      — form-level stats
  output/hunk_clusters/fig_*.png

Usage:
  uv run python scripts/cluster_hunk_descriptions.py
  uv run python scripts/cluster_hunk_descriptions.py --min-cluster-size 3
"""

import argparse
import json
import sys
from pathlib import Path

import dspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "hunk_clusters"

PANEL_BG = "#f5f5f5"
PANEL_EDGE = "#dddddd"
TEAL = "#0C6583"
ORANGE = "#EE7733"
NAVY = "#2B2D42"
GRAY = "#AAAAAA"
GREEN = "#009E73"


def style_panel(ax):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_EDGE)
    ax.tick_params(labelsize=9)


def embed_descriptions(descriptions: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(descriptions, show_progress_bar=True, batch_size=64)


def reduce_umap(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def centroid_representative(
    cluster_embeddings: np.ndarray,
    cluster_descriptions: list[str],
) -> str:
    """Return the description closest to the cluster centroid (cosine distance)."""
    centroid = cluster_embeddings.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True) + 1e-9
    cosine_sims = (cluster_embeddings / norms) @ centroid_norm
    best_idx = int(np.argmax(cosine_sims))
    return cluster_descriptions[best_idx]


class ClusterLabelSignature(dspy.Signature):
    """
    Name a cluster of code edit descriptions with a short mechanism label.
    The descriptions each describe what one diff hunk structurally accomplished.
    """
    sample_descriptions = dspy.InputField(
        desc="3-5 descriptions of code edits from the same cluster"
    )
    mechanism_name = dspy.OutputField(
        desc="2-4 word snake_case label naming the shared fix mechanism. "
             "Examples: add_guard_clause, replace_loop_with_comprehension, "
             "add_exception_handling, refactor_api_call. Be specific and structural."
    )


def label_clusters_with_llm(
    cluster_ids: list[int],
    descriptions_by_cluster: dict[int, list[str]],
    n_samples: int = 5,
) -> dict[int, str]:
    import dspy
    predictor = dspy.Predict(ClusterLabelSignature)
    labels = {}
    for cid in cluster_ids:
        samples = descriptions_by_cluster[cid][:n_samples]
        out = predictor(sample_descriptions="\n".join(f"- {s}" for s in samples))
        raw = (out.mechanism_name or "").strip().lower().replace(" ", "_")
        labels[cid] = raw or f"cluster_{cid}"
    return labels


def build_binary_features(
    instance_ids: list[str],
    instance_clusters: dict[str, set[int]],
    cluster_ids: list[int],
) -> np.ndarray:
    cid_idx = {c: i for i, c in enumerate(cluster_ids)}
    X = np.zeros((len(instance_ids), len(cluster_ids)), dtype=np.float32)
    for row, iid in enumerate(instance_ids):
        for cid in instance_clusters.get(iid, set()):
            if cid in cid_idx:
                X[row, cid_idx[cid]] = 1.0
    return X


def leaf_pass_rate_variance(tree, X, y):
    leaf_ids = tree.apply(X)
    rates = [y[leaf_ids == lid].mean()
             for lid in np.unique(leaf_ids)
             if (leaf_ids == lid).sum() >= 3]
    return float(np.var(rates)) if len(rates) > 1 else 0.0


def leaf_summary(tree, X, y, feature_names, instances, cluster_labels):
    leaf_ids = tree.apply(X)
    forms = []
    for lid in sorted(np.unique(leaf_ids)):
        mask = leaf_ids == lid
        members = [instances[i] for i in np.where(mask)[0]]
        y_sub = y[mask]
        X_sub = X[mask]
        pass_rate = float(y_sub.mean())
        n = int(mask.sum())
        op_freq = X_sub.mean(axis=0)
        dominant_idxs = [i for i in np.argsort(-op_freq) if op_freq[i] >= 0.5]
        dominant = [cluster_labels[feature_names[i]] for i in dominant_idxs[:3]]
        label = " + ".join(dominant) if dominant else "minimal"
        forms.append({
            "leaf_id": int(lid),
            "label": label,
            "n": n,
            "pass_rate": pass_rate,
            "dominant_cluster_ids": [int(feature_names[i]) for i in dominant_idxs[:4]],
            "dominant_labels": dominant[:4],
            "instance_ids": members,
        })
    return forms


def fig_cluster_sizes(cluster_ids, cluster_labels, descriptions_by_cluster, output_dir):
    sizes = {cid: len(descriptions_by_cluster[cid]) for cid in cluster_ids}
    sorted_ids = sorted(sizes, key=lambda c: -sizes[c])
    labels = [cluster_labels[cid] for cid in sorted_ids]
    vals = [sizes[cid] for cid in sorted_ids]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_ids) * 0.7), 4))
    fig.subplots_adjust(bottom=0.4)
    style_panel(ax)
    xs = np.arange(len(sorted_ids))
    ax.bar(xs, vals, color=TEAL, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Number of hunks", fontsize=9)
    ax.set_title(f"HDBSCAN cluster sizes ({len(cluster_ids)} clusters)", fontsize=11, pad=6, fontweight="normal")
    fig.savefig(output_dir / "fig1_cluster_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_cluster_sizes.png")


def fig_form_pass_rates(forms, depth, output_dir):
    forms_sorted = sorted(forms, key=lambda f: -f["pass_rate"])
    fig, ax = plt.subplots(figsize=(max(10, len(forms_sorted) * 0.8), 5))
    fig.subplots_adjust(bottom=0.45, left=0.07, right=0.97)
    style_panel(ax)
    xs = np.arange(len(forms_sorted))
    colors = [TEAL if f["pass_rate"] >= 0.3 else ORANGE if f["pass_rate"] >= 0.15
              else GRAY for f in forms_sorted]
    bars = ax.bar(xs, [f["pass_rate"] for f in forms_sorted], color=colors, alpha=0.85)
    for bar, f in zip(bars, forms_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={f['n']}", ha="center", va="bottom", fontsize=7, color=NAVY)
    ax.axhline(0.23, color=NAVY, linewidth=0.8, linestyle=":", label="Baseline (23%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f["label"] for f in forms_sorted], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Pass rate", fontsize=9)
    ax.set_title(f"Data-driven fix forms at depth={depth}: pass rate per form",
                 fontsize=11, pad=6, fontweight="normal")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(output_dir / "fig2_form_pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_form_pass_rates.png")


def fig_frontier(forms, agent_results, output_dir):
    n_agents = len(agent_results)
    form_order = sorted(forms, key=lambda f: -f["pass_rate"])
    frac_unsolved = []
    for form in form_order:
        members = set(form["instance_ids"])
        n_unsolved = sum(
            1 for iid in members
            if not any(res.get(iid, False) for res in agent_results.values())
        )
        frac_unsolved.append(n_unsolved / max(len(members), 1))

    fig, ax = plt.subplots(figsize=(max(10, len(form_order) * 0.8), 5))
    fig.subplots_adjust(bottom=0.45)
    style_panel(ax)
    xs = np.arange(len(form_order))
    colors = [GRAY if f >= 0.5 else ORANGE if f >= 0.2 else TEAL for f in frac_unsolved]
    ax.bar(xs, frac_unsolved, color=colors, alpha=0.85)
    for xi, (f, form) in enumerate(zip(frac_unsolved, form_order)):
        ax.text(xi, f + 0.01, f"n={form['n']}", ha="center", va="bottom",
                fontsize=7, color=NAVY)
    ax.set_xticks(xs)
    ax.set_xticklabels([f["label"] for f in form_order], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Fraction of instances unsolved by all agents", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Instance-level coverage gaps across {n_agents} leaderboard agents",
                 fontsize=11, pad=6, fontweight="normal")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GRAY, alpha=0.85, label=">=50% unsolved (structural frontier)"),
        Patch(facecolor=ORANGE, alpha=0.85, label="20-50% unsolved"),
        Patch(facecolor=TEAL, alpha=0.85, label="<20% unsolved"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False)
    fig.savefig(output_dir / "fig3_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_frontier.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="HDBSCAN min_cluster_size (default 5)")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--label-llm", action="store_true",
                        help="Use LLM to name clusters instead of TF-IDF")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    for _env in [
        ROOT / ".venv" / ".env",
        ROOT / ".env",
    ]:
        if _env.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(_env)
            except ImportError:
                pass

    from configs.dspy_config import configure_dspy
    configure_dspy(model=args.model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load descriptions
    desc_path = ROOT / "output" / "hunk_descriptions" / "descriptions.json"
    if not desc_path.exists():
        print("ERROR: Run build_hunk_descriptions.py first.")
        sys.exit(1)
    with open(desc_path) as f:
        all_descriptions: dict[str, list[str]] = json.load(f)
    print(f"Loaded descriptions for {len(all_descriptions)} instances")

    # Flatten to list of (instance_id, hunk_idx, description)
    records = [
        (iid, idx, desc)
        for iid, descs in all_descriptions.items()
        for idx, desc in enumerate(descs)
        if desc.strip()
    ]
    print(f"Total non-empty hunk descriptions: {len(records)}")

    flat_descs = [r[2] for r in records]

    # Embed
    emb_path = OUTPUT_DIR / "embeddings.npy"
    if emb_path.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(emb_path)
    else:
        print("Embedding descriptions...")
        embeddings = embed_descriptions(flat_descs)
        np.save(emb_path, embeddings)
        print(f"Saved embeddings ({embeddings.shape})")

    # UMAP → HDBSCAN (standard pipeline for sentence embedding clustering)
    umap_path = OUTPUT_DIR / "umap_reduced.npy"
    if umap_path.exists():
        print("Loading cached UMAP reduction...")
        reduced = np.load(umap_path)
    else:
        print("Reducing to 10D with UMAP (cosine)...")
        reduced = reduce_umap(embeddings, n_components=10)
        np.save(umap_path, reduced)
        print(f"Saved umap_reduced ({reduced.shape})")

    print(f"\nClustering with HDBSCAN (min_cluster_size={args.min_cluster_size})...")
    labels = cluster_hdbscan(reduced, args.min_cluster_size)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise}/{len(labels)} noise points")

    # Group descriptions by cluster
    descriptions_by_cluster: dict[int, list[str]] = {}
    for (iid, idx, desc), cid in zip(records, labels):
        descriptions_by_cluster.setdefault(cid, []).append(desc)

    valid_cluster_ids = sorted(c for c in descriptions_by_cluster if c != -1)
    cluster_labels: dict[int, str] = {}

    # Build per-cluster embedding arrays for centroid representative
    cluster_emb_map: dict[int, tuple[np.ndarray, list[str]]] = {}
    for (iid, idx, desc), cid, emb in zip(records, labels, embeddings):
        if cid == -1:
            continue
        if cid not in cluster_emb_map:
            cluster_emb_map[cid] = ([], [])
        cluster_emb_map[cid][0].append(emb)
        cluster_emb_map[cid][1].append(desc)
    for cid in cluster_emb_map:
        embs, descs = cluster_emb_map[cid]
        cluster_emb_map[cid] = (np.array(embs), descs)

    if args.label_llm:
        print("\nLabeling clusters with LLM...")
        cluster_labels = label_clusters_with_llm(valid_cluster_ids, descriptions_by_cluster)
    else:
        # Centroid representative — description closest to cluster centroid
        for cid in valid_cluster_ids:
            embs, descs = cluster_emb_map[cid]
            cluster_labels[cid] = centroid_representative(embs, descs)
    if -1 in descriptions_by_cluster:
        cluster_labels[-1] = "noise/other"

    print(f"\nCluster labels ({n_clusters} clusters):")
    for cid in valid_cluster_ids:
        n = len(descriptions_by_cluster[cid])
        print(f"  [{cid:3d}] n={n:4d}  {cluster_labels[cid]}")
    if -1 in descriptions_by_cluster:
        print(f"  [noise] n={len(descriptions_by_cluster[-1])} descriptions unassigned")

    # Save cluster definitions
    cluster_info = {
        "n_clusters": n_clusters,
        "n_noise": int(n_noise),
        "min_cluster_size": args.min_cluster_size,
        "clusters": [
            {
                "id": int(cid),
                "label": cluster_labels[cid],
                "n_hunks": int(len(descriptions_by_cluster[cid])),
                "sample_descriptions": descriptions_by_cluster[cid][:5],
            }
            for cid in valid_cluster_ids
        ],
    }
    with open(OUTPUT_DIR / "cluster_labels.json", "w") as f:
        json.dump(cluster_info, f, indent=2)
    print("\nSaved cluster_labels.json")

    # Map cluster assignments back to instances
    instance_clusters: dict[str, set[int]] = {}
    for (iid, idx, desc), cid in zip(records, labels):
        if cid == -1:
            continue
        instance_clusters.setdefault(iid, set()).add(cid)

    # Load pass/fail labels
    fix_df = pd.read_parquet(
        ROOT / "notebooks" / "plots" / "fix_type_analysis" / "merged_analysis.parquet"
    )[["instance_id", "passed"]]

    # Align: instances with >=1 cluster assignment and pass/fail label
    common = sorted(
        set(instance_clusters) & set(fix_df["instance_id"])
    )
    print(f"\nInstances with cluster assignments and pass/fail: {len(common)}")

    fix_sub = fix_df[fix_df["instance_id"].isin(common)].set_index("instance_id")
    y = np.array([int(fix_sub.loc[iid, "passed"]) for iid in common])
    print(f"Pass: {y.sum()}, Fail: {(1-y).sum()}")

    # Build binary feature matrix
    X = build_binary_features(common, instance_clusters, valid_cluster_ids)
    feature_names = valid_cluster_ids  # int cluster ids as feature names
    print(f"Feature matrix: {X.shape}")

    # Decision tree depth sweep
    from sklearn.tree import DecisionTreeClassifier, export_text
    depths = list(range(2, args.max_depth + 1))
    variances, n_leaves_list = [], []
    print("\nSweeping depth...")
    for d in depths:
        tree = DecisionTreeClassifier(
            max_depth=d, class_weight="balanced", random_state=42, min_samples_leaf=3
        )
        tree.fit(X, y)
        v = leaf_pass_rate_variance(tree, X, y)
        n = tree.get_n_leaves()
        variances.append(v)
        n_leaves_list.append(n)
        print(f"  depth={d}: {n} leaves, variance={v:.4f}")

    # Select depth by largest marginal variance gain
    deltas = [variances[i] - variances[i - 1] for i in range(1, len(variances))]
    best_depth = depths[np.argmax(deltas) + 1]
    print(f"\nSelected depth={best_depth} (largest gain: +{max(deltas):.4f})")

    tree_final = DecisionTreeClassifier(
        max_depth=best_depth, class_weight="balanced", random_state=42, min_samples_leaf=3
    )
    tree_final.fit(X, y)
    forms = leaf_summary(tree_final, X, y, feature_names, common, cluster_labels)

    print(f"\n{len(forms)} data-driven fix forms:")
    for fm in sorted(forms, key=lambda f: -f["pass_rate"]):
        print(f"  [{fm['label']:55s}] n={fm['n']:3d}  pass={fm['pass_rate']:.2f}")

    # Save form assignments
    rows = []
    for fm in forms:
        for iid in fm["instance_ids"]:
            rows.append({
                "instance_id": iid,
                "form_label": fm["label"],
                "form_leaf_id": fm["leaf_id"],
                "form_pass_rate": fm["pass_rate"],
                "form_n": fm["n"],
                "passed": bool(fix_sub.loc[iid, "passed"]),
                "descriptions": all_descriptions.get(iid, []),
                "clusters": sorted(instance_clusters.get(iid, set())),
            })
    pd.DataFrame(rows).to_parquet(OUTPUT_DIR / "form_assignments.parquet", index=False)
    print(f"\nSaved form_assignments.parquet ({len(rows)} rows)")

    with open(OUTPUT_DIR / "form_summaries.json", "w") as f:
        json.dump({
            "depth": best_depth,
            "n_forms": len(forms),
            "n_clusters": n_clusters,
            "forms": [{k: v for k, v in fm.items() if k != "instance_ids"} for fm in forms],
        }, f, indent=2)
    print("Saved form_summaries.json")

    # Frontier analysis
    agent_results = {}
    lb_path = ROOT / "output" / "leaderboard" / "lite_results.msgpack"
    if lb_path.exists():
        import msgpack
        with open(lb_path, "rb") as f:
            lb_data = msgpack.unpack(f, raw=False)
        for agent_id, pf in lb_data.items():
            agent_results[agent_id] = {iid: bool(v) for iid, v in pf.items()}
        print(f"\nLoaded {len(agent_results)} agents from leaderboard msgpack")

    if agent_results:
        n_agents = len(agent_results)
        print(f"Frontier analysis with {n_agents} agents:")
        for fm in sorted(forms, key=lambda f: -f["pass_rate"]):
            members = set(fm["instance_ids"])
            n_unsolved = sum(
                1 for iid in members
                if not any(res.get(iid, False) for res in agent_results.values())
            )
            pct = 100 * n_unsolved / max(len(members), 1)
            print(f"  [{fm['label']:55s}] n={fm['n']:3d}  pass={fm['pass_rate']:.2f}  unsolved={n_unsolved}/{len(members)} ({pct:.0f}%)")

        global_unsolved = sum(
            1 for iid in common
            if not any(res.get(iid, False) for res in agent_results.values())
        )
        print(f"\nGlobal: {global_unsolved}/{len(common)} instances unsolved by all {n_agents} agents")

    print("\nTree structure:")
    cluster_name_map = [cluster_labels.get(cid, str(cid)) for cid in feature_names]
    print(export_text(tree_final, feature_names=cluster_name_map, max_depth=best_depth))

    print("\nGenerating figures...")
    fig_cluster_sizes(valid_cluster_ids, cluster_labels, descriptions_by_cluster, OUTPUT_DIR)
    fig_form_pass_rates(forms, best_depth, OUTPUT_DIR)
    if agent_results:
        fig_frontier(forms, agent_results, OUTPUT_DIR)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
