#!/usr/bin/env python3
"""
Cluster instances by fix-grounded semantic representations from the prompting study,
then compare ease variance to FIM structural forms and issue text clustering.

Two conditions:
  raw_logs   -- model describes the actual fix after seeing agent traces (fix-grounded)
  no_context -- model predicts what fix is needed from the issue alone (problem-grounded)

Both use GPT-4o responses (286 instances).

Outputs:
  fix_semantic_clusters.json         -- cluster assignments and ease stats per condition
  fig_variance_comparison.png        -- all groupings side by side

Usage:
  uv run python scripts/cluster_fix_descriptions.py
"""

import json
import sys
from pathlib import Path

import altair as alt
import msgpack
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "fix_semantic_clusters"
OUT.mkdir(parents=True, exist_ok=True)

# Wong palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
PINK   = "#CC79A7"
GRAY   = "#999999"
RED    = "#D55E00"
SKY    = "#56B4E9"


def load_ease() -> dict[str, float]:
    with open(ROOT / "output" / "leaderboard" / "lite_results.msgpack", "rb") as f:
        lb = msgpack.unpack(f, raw=False)
    votes: dict[str, list] = {}
    for agent_data in lb.values():
        for iid, passed in agent_data.items():
            votes.setdefault(iid, []).append(passed)
    return {iid: float(np.mean(v)) for iid, v in votes.items()}


def load_responses(condition: str, model: str = "gpt_4o") -> dict[str, str]:
    path = ROOT / "output" / "prompting_study" / model / "records.json"
    with open(path) as f:
        records = json.load(f)
    result = {}
    for r in records:
        resp = r["conditions"].get(condition, {}).get("response", "").strip()
        if resp:
            result[r["instance_id"]] = resp
    return result


def embed(texts: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path.name}...")
        return np.load(cache_path)
    print(f"  Embedding {len(texts)} texts...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, show_progress_bar=True, batch_size=32)
    np.save(cache_path, embs)
    return embs


def best_kmeans(X: np.ndarray, k_range: range) -> tuple[int, np.ndarray]:
    """Find k with highest silhouette score."""
    best_k, best_labels, best_score = k_range.start, None, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"    k={k}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    print(f"  Best k={best_k} (silhouette={best_score:.4f})")
    return best_k, best_labels


def ease_variance(group_to_instances: dict[str, list[str]],
                  ease: dict[str, float],
                  min_size: int = 5) -> float:
    means = []
    for instances in group_to_instances.values():
        vals = [ease[iid] for iid in instances if iid in ease]
        if len(vals) >= min_size:
            means.append(np.mean(vals))
    return float(np.var(means)) if len(means) >= 2 else 0.0


def run_condition(condition: str, ease: dict[str, float]) -> dict:
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"{'='*60}")

    responses = load_responses(condition)
    instance_ids = list(responses.keys())
    texts = [responses[iid] for iid in instance_ids]
    print(f"  {len(instance_ids)} instances with responses")

    # Embed
    cache = OUT / f"embeddings_{condition}.npy"
    X = embed(texts, cache)

    # k-means k=10 (matches issue text semantic baseline)
    print(f"\n  k-means k=10 (matching baseline)...")
    km10 = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels10 = km10.fit_predict(X)
    groups10 = {}
    for iid, label in zip(instance_ids, labels10):
        groups10.setdefault(str(label), []).append(iid)
    var10 = ease_variance(groups10, ease)
    sil10 = silhouette_score(X, labels10)
    print(f"  k=10  variance={var10:.4f}  silhouette={sil10:.4f}")

    # k-means sweep k=5..20 to find best k
    print(f"\n  k-means sweep k=5..20:")
    best_k, best_labels = best_kmeans(X, range(5, 21))
    groups_best = {}
    for iid, label in zip(instance_ids, best_labels):
        groups_best.setdefault(str(label), []).append(iid)
    var_best = ease_variance(groups_best, ease)
    sil_best = silhouette_score(X, best_labels)

    return {
        "condition": condition,
        "n_instances": len(instance_ids),
        "k10_variance": var10,
        "k10_silhouette": float(sil10),
        "best_k": best_k,
        "best_k_variance": var_best,
        "best_k_silhouette": float(sil_best),
        "instance_ids": instance_ids,
        "labels_k10": labels10.tolist(),
        "labels_best": best_labels.tolist(),
    }


def main():
    print("Loading ease data...")
    ease = load_ease()

    results = {}
    for condition in ["no_context", "raw_logs"]:
        results[condition] = run_condition(condition, ease)

    with open(OUT / "fix_semantic_clusters.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk not in ("instance_ids", "labels_k10", "labels_best")}
                   for k, v in results.items()}, f, indent=2)

    # --- Variance comparison figure ---
    print("\nBuilding variance comparison figure...")

    nc_best_k = results["no_context"]["best_k"]
    rl_best_k = results["raw_logs"]["best_k"]

    baselines = [
        {"grouping": "Issue text",                  "k_label": "k=10",           "variance": 0.0073,                                          "color": GRAY,   "order": 0},
        {"grouping": "GPT-4o predicted fix",        "k_label": "k=10",           "variance": results["no_context"]["k10_variance"],            "color": SKY,    "order": 1},
        {"grouping": "GPT-4o fix from traces",      "k_label": "k=10",           "variance": results["raw_logs"]["k10_variance"],              "color": GREEN,  "order": 2},
        {"grouping": "GPT-4o predicted fix",        "k_label": f"k={nc_best_k}", "variance": results["no_context"]["best_k_variance"],         "color": SKY,    "order": 3},
        {"grouping": "GPT-4o fix from traces",      "k_label": f"k={rl_best_k}", "variance": results["raw_logs"]["best_k_variance"],           "color": GREEN,  "order": 4},
        {"grouping": "AST cert decision tree",      "k_label": "10 forms",       "variance": 0.0257,                                          "color": ORANGE, "order": 5},
        {"grouping": "FIM closed itemsets",         "k_label": "15 forms",       "variance": 0.0333,                                          "color": BLUE,   "order": 6},
    ]

    df = pd.DataFrame(baselines)
    max_var = df["variance"].max()
    # Unique row key for display — grouping + k_label
    df["row_key"] = df["grouping"] + " (" + df["k_label"] + ")"
    sort_order = df.sort_values("order")["row_key"].tolist()

    bars = alt.Chart(df).mark_bar(height=18).encode(
        x=alt.X(
            "variance:Q",
            scale=alt.Scale(domain=[0, max_var * 1.35]),
            axis=alt.Axis(title="Variance of per-group mean agent ease", titleFontSize=10),
        ),
        y=alt.Y(
            "row_key:N",
            sort=sort_order,
            axis=alt.Axis(
                title=None,
                labelFontSize=9,
                labelLimit=280,
            ),
        ),
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=["grouping:N", "k_label:N", "variance:Q"],
    )

    # k label inside bar at right edge
    k_labels = alt.Chart(df).mark_text(
        align="right", dx=-5, fontSize=8, color="white", fontWeight="normal"
    ).encode(
        x=alt.X("variance:Q"),
        y=alt.Y("row_key:N", sort=sort_order),
        text=alt.Text("k_label:N"),
    )

    # Variance value just past end of bar
    val_labels = alt.Chart(df).mark_text(
        align="left", dx=4, fontSize=9, color="#444444"
    ).encode(
        x=alt.X("variance:Q"),
        y=alt.Y("row_key:N", sort=sort_order),
        text=alt.Text("variance:Q", format=".4f"),
    )

    fig = (bars + k_labels + val_labels).properties(
        width=430,
        height=240,
        title=alt.TitleParams(
            "Which grouping best separates difficulty?",
            fontSize=12,
            fontWeight="normal",
            anchor="start",
        )
    ).configure_axis(
        grid=False,
        labelFontSize=9,
        titleFontSize=10,
    ).configure_view(strokeWidth=0)

    fig.save(str(OUT / "fig_variance_comparison.png"), scale_factor=2)
    print("  Saved fig_variance_comparison.png")

    # Print summary
    print("\n--- Summary ---")
    print(f"Issue text k-means k=10          : 0.0073  (baseline)")
    for cond in ["no_context", "raw_logs"]:
        r = results[cond]
        print(f"Fix desc {cond:12s} k=10    : {r['k10_variance']:.4f}")
        print(f"Fix desc {cond:12s} k={r['best_k']:2d}   : {r['best_k_variance']:.4f}")
    print(f"AST cert decision tree           : 0.0257")
    print(f"FIM closed itemsets              : 0.0333")

    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
