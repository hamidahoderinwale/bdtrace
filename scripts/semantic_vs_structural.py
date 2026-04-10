#!/usr/bin/env python3
"""
Three analyses showing what semantic clustering misses vs. structural forms.

All figures use Altair + project color palette. No hard frontier cutoff —
agent ease is treated as a continuous variable throughout.

1. Nearest-neighbor scatter: for each instance, x=agent ease, y=fraction of
   k=10 semantic neighbors that are easy. If semantic space predicts difficulty,
   expect r>0. If orthogonal, expect r≈0.

2. UMAP comparison: side-by-side semantic vs structural 2D projections, points
   colored by continuous agent ease. Frontier cluster should be visible in
   structural space but scattered in semantic space.

3. Within-cluster ease distribution: strip/box of agent ease per semantic
   cluster (k-means) vs per structural form. Structural forms should separate
   the ease distribution; semantic clusters should not.

Outputs:
  output/semantic_vs_structural/fig1_nn_scatter.png
  output/semantic_vs_structural/fig2_umap_comparison.png
  output/semantic_vs_structural/fig3_ease_distributions.png
  output/semantic_vs_structural/nearest_neighbor_results.json

Usage:
  uv run python scripts/semantic_vs_structural.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "semantic_vs_structural"

# Project palette (Wong color-blind safe)
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
PINK   = "#CC79A7"
GRAY   = "#999999"
NAVY   = "#2B2D42"

FORM_PALETTE = [BLUE, ORANGE, GREEN, PINK, GRAY, "#332288", "#AA4499",
                "#DDCC77", "#88CCEE", "#117733"]


def _altair_base():
    return alt.themes.enable("default")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problem_statements() -> dict[str, str]:
    from datasets import load_dataset
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    return {row["instance_id"]: row["problem_statement"] for row in ds}


def embed_texts(texts: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path.name}...")
        return np.load(cache_path)
    from sentence_transformers import SentenceTransformer
    print("Embedding texts...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, show_progress_bar=True, batch_size=64)
    np.save(cache_path, embs)
    print(f"Saved {cache_path.name} ({embs.shape})")
    return embs


_NORMALIZE_OPS = {
    "ADD_arguments": "ADD_arg", "DEL_arguments": "DEL_arg",
    "ADD_keyword": "ADD_arg", "DEL_keyword": "DEL_arg",
}


def load_edit_cert_features(instance_ids: list[str]) -> np.ndarray:
    from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence
    import difflib

    iid_set = set(instance_ids)
    all_certs: dict[str, frozenset] = {}
    traces_path = ROOT / "output" / "resolved_traces_lite_full.jsonl"
    with open(traces_path) as f:
        for line in f:
            trace = json.loads(line)
            iid = trace["instance_id"]
            if iid not in iid_set:
                continue
            ops = []
            for ev in trace["events"]:
                if ev["type"] != "code_change":
                    continue
                d = ev["details"]
                if not d["file_path"].endswith(".py"):
                    continue
                before = (d["before_content"] or "").splitlines(keepends=True)
                after = (d["after_content"] or "").splitlines(keepends=True)
                if before == after:
                    continue
                raw = "".join(difflib.unified_diff(
                    before, after,
                    fromfile=d["file_path"], tofile=d["file_path"],
                ))
                if not raw:
                    continue
                patch = f"diff --git a/{d['file_path']} b/{d['file_path']}\n" + raw
                ops.extend(patch_to_ast_sequence(patch))
            if ops:
                all_certs[iid] = frozenset(_NORMALIZE_OPS.get(op, op) for op in ops)

    all_ops = sorted({op for cert in all_certs.values() for op in cert})
    op_idx = {op: i for i, op in enumerate(all_ops)}
    X = np.zeros((len(instance_ids), len(all_ops)), dtype=np.float32)
    for row, iid in enumerate(instance_ids):
        for op in all_certs.get(iid, frozenset()):
            if op in op_idx:
                X[row, op_idx[op]] = 1.0
    print(f"Edit cert matrix: {X.shape}")
    return X


def load_agent_ease(instance_ids: list[str]) -> dict[str, float]:
    import msgpack
    with open(ROOT / "output" / "leaderboard" / "lite_results.msgpack", "rb") as f:
        lb = msgpack.unpack(f, raw=False)
    return {
        iid: float(np.mean([float(res.get(iid, False)) for res in lb.values()]))
        for iid in instance_ids
    }


def umap_2d(X: np.ndarray, cache_path: Path, metric: str = "cosine") -> np.ndarray:
    if cache_path.exists():
        print(f"Loading cached 2D UMAP from {cache_path.name}...")
        return np.load(cache_path)
    import umap
    print(f"Computing 2D UMAP ({metric})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric=metric, random_state=42)
    coords = reducer.fit_transform(X)
    np.save(cache_path, coords)
    return coords


# ---------------------------------------------------------------------------
# Analysis 1: Nearest-neighbor scatter (distributional, no hard cutoff)
# ---------------------------------------------------------------------------

def nearest_neighbor_analysis(
    sem_embs: np.ndarray,
    instance_ids: list[str],
    ease: dict[str, float],
    k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(sem_embs)
    np.fill_diagonal(sim, -1)

    ease_arr = np.array([ease[iid] for iid in instance_ids])
    rows = []
    for i, iid in enumerate(instance_ids):
        top_k = np.argsort(-sim[i])[:k]
        neighbor_ease = float(ease_arr[top_k].mean())
        rows.append({
            "instance_id": iid,
            "ease": ease[iid],
            "neighbor_ease_mean": neighbor_ease,
        })
    df = pd.DataFrame(rows)

    r = float(np.corrcoef(df["ease"], df["neighbor_ease_mean"])[0, 1])
    print(f"\nNearest neighbor (k={k}): r(ease, semantic-neighbor-ease) = {r:.3f}")
    print(f"  (r≈0 means semantic space is blind to difficulty)")

    # Semantic classifier AUC on continuous ease
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    y_hard = (ease_arr <= 0.05).astype(int)
    if y_hard.sum() >= 5:
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=500))])
        proba = cross_val_predict(pipe, sem_embs, y_hard, cv=5, method="predict_proba")
        auc = float(roc_auc_score(y_hard, proba[:, 1]))
        print(f"  Semantic classifier AUC (hard vs easy): {auc:.3f}  (0.5=random)")
    else:
        auc = None

    results = {"k": k, "r_ease_neighbor_ease": r, "semantic_classifier_auc": auc}
    return results, df


def fig_nn_scatter(df: pd.DataFrame, nn_results: dict, output_dir: Path):
    r = nn_results["r_ease_neighbor_ease"]
    auc = nn_results.get("semantic_classifier_auc")

    # Add jitter to ease for visibility
    rng = np.random.default_rng(42)
    df = df.copy()
    df["ease_j"] = df["ease"] + rng.normal(0, 0.008, len(df))

    note = f"r = {r:.3f}"
    if auc:
        note += f"   AUC = {auc:.3f}"

    scatter = alt.Chart(df).mark_circle(size=40, opacity=0.55).encode(
        x=alt.X("ease_j:Q",
                title="Instance agent ease (fraction of 84 agents solving)",
                scale=alt.Scale(domain=[-0.05, 1.05]),
                axis=alt.Axis(format=".0%")),
        y=alt.Y("neighbor_ease_mean:Q",
                title=f"Mean ease of k={nn_results['k']} semantic neighbors",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%")),
        color=alt.value(BLUE),
        tooltip=["instance_id", "ease", "neighbor_ease_mean"],
    )

    # Regression line
    regression = scatter.transform_regression(
        "ease_j", "neighbor_ease_mean", method="linear",
    ).mark_line(color=ORANGE, strokeWidth=1.5, strokeDash=[4, 3])

    # Global mean reference
    mean_ease = float(df["ease"].mean())
    hline = alt.Chart(pd.DataFrame({"y": [mean_ease]})).mark_rule(
        color=GRAY, strokeDash=[2, 2], strokeWidth=1,
    ).encode(y="y:Q")

    annotation = alt.Chart(pd.DataFrame({"x": [0.7], "y": [0.05], "text": [note]})).mark_text(
        align="left", fontSize=9, color=NAVY,
    ).encode(x="x:Q", y="y:Q", text="text:N")

    chart = (scatter + regression + hline + annotation).properties(
        width=420, height=280,
        title=alt.TitleParams(
            "Semantic problem space is blind to structural difficulty",
            subtitle="Each point is one SWE-bench Lite instance. "
                     "Flat slope means hard instances have easy semantic neighbors.",
            fontSize=11, subtitleFontSize=9,
        ),
    ).configure_axis(
        grid=False, labelFontSize=9, titleFontSize=9,
    ).configure_view(stroke=None)

    out = output_dir / "fig1_nn_scatter.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out.name}")


# ---------------------------------------------------------------------------
# Analysis 2: UMAP comparison (continuous ease coloring)
# ---------------------------------------------------------------------------

def fig_umap_comparison(
    sem_coords: np.ndarray,
    struct_coords: np.ndarray,
    instance_ids: list[str],
    ease: dict[str, float],
    output_dir: Path,
):
    ease_vals = [ease[iid] for iid in instance_ids]
    df_sem = pd.DataFrame({
        "x": sem_coords[:, 0], "y": sem_coords[:, 1],
        "ease": ease_vals, "instance_id": instance_ids,
        "space": "Semantic\n(issue text)",
    })
    df_struct = pd.DataFrame({
        "x": struct_coords[:, 0], "y": struct_coords[:, 1],
        "ease": ease_vals, "instance_id": instance_ids,
        "space": "Structural\n(edit certs)",
    })
    df = pd.concat([df_sem, df_struct], ignore_index=True)

    chart = alt.Chart(df).mark_circle(size=35, opacity=0.7).encode(
        x=alt.X("x:Q", axis=None, title=None),
        y=alt.Y("y:Q", axis=None, title=None),
        color=alt.Color(
            "ease:Q",
            title="Agent ease",
            scale=alt.Scale(scheme="plasma", domain=[0, 1], reverse=True),
            legend=alt.Legend(
                orient="bottom", direction="horizontal",
                gradientLength=200, labelFontSize=8, titleFontSize=9,
                format=".0%",
            ),
        ),
        tooltip=["instance_id", alt.Tooltip("ease:Q", format=".2f")],
        facet=alt.Facet(
            "space:N",
            title=None,
            header=alt.Header(labelFontSize=10, labelFontWeight="normal"),
        ),
    ).properties(
        width=280, height=260,
        title=alt.TitleParams(
            "Same instances — hard cases (dark) scatter in semantic space, cluster in structural space",
            fontSize=10, subtitleFontSize=8,
        ),
    ).configure_axis(
        grid=False,
    ).configure_view(stroke=GRAY, strokeWidth=0.5)

    out = output_dir / "fig2_umap_comparison.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out.name}")


# ---------------------------------------------------------------------------
# Analysis 3: Ease distributions per cluster/form (distributional)
# ---------------------------------------------------------------------------

def ease_distribution_analysis(
    sem_embs: np.ndarray,
    instance_ids: list[str],
    ease: dict[str, float],
    form_labels: dict[str, str],
    k_semantic: int = 10,
) -> dict:
    ease_arr = np.array([ease[iid] for iid in instance_ids])

    # Semantic clusters
    kmeans = KMeans(n_clusters=k_semantic, random_state=42, n_init=10)
    sem_labels = kmeans.fit_predict(sem_embs)

    # Variance of per-cluster mean ease
    sem_means = [ease_arr[sem_labels == c].mean()
                 for c in range(k_semantic) if (sem_labels == c).sum() >= 3]
    sem_var = float(np.var(sem_means))

    form_arr = np.array([form_labels.get(iid, "unknown") for iid in instance_ids])
    unique_forms = [f for f in np.unique(form_arr) if (form_arr == f).sum() >= 3]
    struct_means = [ease_arr[form_arr == f].mean() for f in unique_forms]
    struct_var = float(np.var(struct_means))
    ratio = struct_var / max(sem_var, 1e-9)

    print(f"\nEase distribution variance:")
    print(f"  Semantic k-means (k={k_semantic}): {sem_var:.4f}")
    print(f"  Structural forms (n={len(unique_forms)}): {struct_var:.4f}")
    print(f"  Structural forms separate difficulty {ratio:.1f}x better")

    return {
        "semantic_k": k_semantic,
        "semantic_variance": sem_var,
        "structural_n_forms": len(unique_forms),
        "structural_variance": struct_var,
        "ratio": ratio,
        "sem_labels": sem_labels.tolist(),
        "unique_forms": unique_forms,
    }


def fig_ease_distributions(
    sem_embs: np.ndarray,
    instance_ids: list[str],
    ease: dict[str, float],
    form_labels: dict[str, str],
    dist_results: dict,
    output_dir: Path,
):
    ease_arr = np.array([ease[iid] for iid in instance_ids])
    sem_labels = np.array(dist_results["sem_labels"])
    form_arr = np.array([form_labels.get(iid, "unknown") for iid in instance_ids])

    # Semantic panel: sort clusters by mean ease
    sem_means = {c: ease_arr[sem_labels == c].mean() for c in range(dist_results["semantic_k"])}
    sem_order = [str(c) for c in sorted(sem_means, key=sem_means.get)]
    sem_rows = [{"cluster": str(c), "ease": float(e), "group": "semantic"}
                for c, e in zip(sem_labels, ease_arr)]
    df_sem = pd.DataFrame(sem_rows)
    df_sem["cluster_label"] = "C" + df_sem["cluster"].str.zfill(2)
    sem_label_order = ["C" + str(c).zfill(2) for c in sorted(sem_means, key=sem_means.get)]

    # Structural panel: sort forms by mean ease
    struct_means = {f: ease_arr[form_arr == f].mean() for f in dist_results["unique_forms"]}
    struct_rows = [{"cluster": form_labels.get(iid, "unknown"), "ease": float(e)}
                   for iid, e in zip(instance_ids, ease_arr)
                   if form_labels.get(iid, "unknown") in dist_results["unique_forms"]]
    df_struct = pd.DataFrame(struct_rows)
    struct_label_order = sorted(struct_means, key=struct_means.get)
    # Shorten form labels for axis
    df_struct["cluster"] = df_struct["cluster"].apply(lambda s: s[:30])
    struct_label_order_short = [s[:30] for s in struct_label_order]

    def strip_box(df, x_field, x_order, color, title, subtitle):
        base = alt.Chart(df)
        boxes = base.mark_boxplot(
            extent="min-max", size=14,
            median=alt.MarkConfig(color="white", strokeWidth=1.5),
            outliers=alt.MarkConfig(size=10, opacity=0.4),
        ).encode(
            x=alt.X(f"{x_field}:N", sort=x_order, title=None,
                    axis=alt.Axis(labelAngle=-40, labelFontSize=8)),
            y=alt.Y("ease:Q", title="Agent ease", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", labelFontSize=8, titleFontSize=9)),
            color=alt.value(color),
        )
        return boxes.properties(
            width=360, height=220,
            title=alt.TitleParams(title, subtitle=subtitle,
                                  fontSize=10, subtitleFontSize=8),
        )

    sem_var = dist_results["semantic_variance"]
    struct_var = dist_results["structural_variance"]
    ratio = dist_results["ratio"]

    panel_sem = strip_box(
        df_sem, "cluster_label", sem_label_order,
        color=GRAY,
        title=f"Semantic clusters (k={dist_results['semantic_k']})",
        subtitle=f"Ease variance = {sem_var:.4f}  — clusters mix all difficulty levels",
    )
    panel_struct = strip_box(
        df_struct, "cluster", struct_label_order_short,
        color=BLUE,
        title=f"Structural forms (n={dist_results['structural_n_forms']})",
        subtitle=f"Ease variance = {struct_var:.4f}  — {ratio:.1f}x more stratified",
    )

    chart = alt.hconcat(panel_sem, panel_struct, spacing=40).configure_axis(
        grid=False,
    ).configure_view(stroke=None)

    out = output_dir / "fig3_ease_distributions.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load structural form assignments
    form_df = pd.read_parquet(ROOT / "output" / "fix_forms" / "form_assignments.parquet")
    instance_ids = form_df["instance_id"].tolist()
    form_labels = dict(zip(form_df["instance_id"], form_df["form_label"]))
    print(f"Instances: {len(instance_ids)}")

    print("Loading agent ease (84 agents)...")
    ease = load_agent_ease(instance_ids)

    print("Loading problem statements...")
    problem_statements = load_problem_statements()
    texts = [problem_statements.get(iid, "") for iid in instance_ids]

    sem_embs = embed_texts(texts, OUTPUT_DIR / "semantic_embeddings.npy")

    struct_cache = OUTPUT_DIR / "structural_features.npy"
    if struct_cache.exists():
        print("Loading cached structural features...")
        struct_X = np.load(struct_cache)
    else:
        print("Computing structural edit cert features...")
        struct_X = load_edit_cert_features(instance_ids)
        np.save(struct_cache, struct_X)

    # -------------------------------------------------------------------
    print("\n" + "="*60)
    print("ANALYSIS 1: Nearest-neighbor scatter")
    print("="*60)
    nn_results, nn_df = nearest_neighbor_analysis(sem_embs, instance_ids, ease, k=10)
    with open(OUTPUT_DIR / "nearest_neighbor_results.json", "w") as f:
        json.dump(nn_results, f, indent=2)
    fig_nn_scatter(nn_df, nn_results, OUTPUT_DIR)

    # -------------------------------------------------------------------
    print("\n" + "="*60)
    print("ANALYSIS 2: UMAP comparison")
    print("="*60)
    sem_coords = umap_2d(sem_embs, OUTPUT_DIR / "umap_semantic_2d.npy", metric="cosine")
    struct_coords = umap_2d(struct_X, OUTPUT_DIR / "umap_structural_2d.npy", metric="jaccard")
    fig_umap_comparison(sem_coords, struct_coords, instance_ids, ease, OUTPUT_DIR)

    # -------------------------------------------------------------------
    print("\n" + "="*60)
    print("ANALYSIS 3: Ease distributions per cluster/form")
    print("="*60)
    dist_results = ease_distribution_analysis(sem_embs, instance_ids, ease, form_labels)
    fig_ease_distributions(sem_embs, instance_ids, ease, form_labels, dist_results, OUTPUT_DIR)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({"nearest_neighbor": nn_results,
                   "variance": {k: v for k, v in dist_results.items()
                                if k not in ("sem_labels", "unique_forms")}}, f, indent=2)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
