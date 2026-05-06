"""Embed SWE-bench Lite problem statements into a semantic vector space.

Foundation for the three-axis evaluation framework:
  Problem space  <->  Procedure space  <->  Outcome space

Outputs (all in output/issue_embeddings/):
  embeddings.npy     (300, 384) float32 — sentence-transformer vectors
  instance_ids.json  ordered list of instance_ids matching row order
  umap_2d.npy        (300, 2) float32 — UMAP projection for visualization
  metadata.json      provenance: model, date, N, dim, umap params
  umap_scatter.html  Altair scatter colored by ease score (inspectable output)

Usage:
    python -m scripts.embed_issue_text
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.theme import register, BLUE, ORANGE, NEAR_BLACK

register()

OUT = PROJECT_ROOT / "output" / "issue_embeddings"
LB_PATH = PROJECT_ROOT / "output" / "leaderboard" / "lite_results.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
TARGET_AGENTS = {
    "20240402_sweagent_gpt4",
    "20240620_sweagent_claude3.5sonnet",
    "20240728_sweagent_gpt4o",
}

UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42}


def load_problem_statements() -> tuple[list[str], list[str], list[str]]:
    """Returns (instance_ids, repos, texts) for all 300 SWE-bench Lite instances."""
    from datasets import load_dataset
    print("Loading SWE-bench Lite problem statements...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    instance_ids = [str(r["instance_id"]) for r in ds]
    repos = [str(r["repo"]) for r in ds]
    texts = [str(r["problem_statement"]) for r in ds]
    print(f"  {len(texts)} instances loaded")
    return instance_ids, repos, texts


def load_ease_scores(instance_ids: list[str]) -> dict[str, float]:
    """Mean resolve rate across the 3 target agents."""
    with open(LB_PATH) as f:
        lb = json.load(f)
    target = {k: v for k, v in lb.items() if k in TARGET_AGENTS}
    scores: dict[str, float] = {}
    for iid in instance_ids:
        votes = [bool(agent.get(iid, False)) for agent in target.values()]
        scores[iid] = float(sum(votes) / len(votes)) if votes else 0.0
    return scores


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"Embedding {len(texts)} texts with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(texts, show_progress_bar=True, batch_size=32,
                        normalize_embeddings=True)
    return embs.astype(np.float32)


def umap_project(embs: np.ndarray) -> np.ndarray:
    import umap
    print("Running UMAP projection...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    return reducer.fit_transform(embs).astype(np.float32)


def make_scatter(
    instance_ids: list[str],
    repos: list[str],
    xy: np.ndarray,
    ease: dict[str, float],
    out_path: Path,
) -> None:
    import pandas as pd
    import altair as alt

    df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "instance_id": instance_ids,
        "repo": [r.split("/")[-1] for r in repos],
        "ease": [ease.get(iid, 0.0) for iid in instance_ids],
    })
    # Classify by whether any agent resolved it
    df["resolved_by_any"] = df["ease"] > 0

    chart = (
        alt.Chart(df)
        .mark_circle(size=55, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title=None, labels=False, ticks=False, domain=False)),
            y=alt.Y("y:Q", axis=alt.Axis(title=None, labels=False, ticks=False, domain=False)),
            color=alt.Color(
                "ease:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 1]),
                legend=alt.Legend(title="Ease (fraction of agents that resolved)"),
            ),
            tooltip=["instance_id", "repo", alt.Tooltip("ease:Q", format=".2f")],
        )
        .properties(
            title=alt.TitleParams(
                "Issue text: semantic space (UMAP)",
                fontSize=13, fontWeight="normal", anchor="start",
                color=NEAR_BLACK,
            ),
            width=520,
            height=420,
        )
    )

    chart.save(str(out_path))
    print(f"Saved scatter: {out_path}")


def make_scatter_by_repo(
    instance_ids: list[str],
    repos: list[str],
    xy: np.ndarray,
    out_path: Path,
) -> None:
    import pandas as pd
    import altair as alt
    from scripts.theme import CATEGORY

    repo_names = [r.split("/")[-1] for r in repos]
    unique_repos = sorted(set(repo_names))

    df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "instance_id": instance_ids,
        "repo": repo_names,
    })

    color_scale = alt.Scale(
        domain=unique_repos,
        range=CATEGORY[:len(unique_repos)],
    )

    chart = (
        alt.Chart(df)
        .mark_circle(size=55, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title=None, labels=False, ticks=False, domain=False)),
            y=alt.Y("y:Q", axis=alt.Axis(title=None, labels=False, ticks=False, domain=False)),
            color=alt.Color("repo:N", scale=color_scale,
                            legend=alt.Legend(title=None, labelFontSize=10)),
            tooltip=["instance_id", "repo"],
        )
        .properties(
            title=alt.TitleParams(
                "Issue text: semantic space colored by repository (UMAP)",
                fontSize=13, fontWeight="normal", anchor="start",
                color=NEAR_BLACK,
            ),
            width=520,
            height=420,
        )
    )

    html_path = out_path.with_suffix(".html")
    chart.save(str(html_path))
    print(f"Saved repo scatter: {html_path}")

    try:
        import vl_convert as vlc
        png = vlc.vegalite_to_png(chart.to_json(), scale=2)
        out_path.write_bytes(png)
        print(f"Saved repo scatter PNG: {out_path}")
    except ImportError:
        print("vl_convert not available, HTML only")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    instance_ids, repos, texts = load_problem_statements()
    ease = load_ease_scores(instance_ids)

    embs = embed(texts)
    np.save(OUT / "embeddings.npy", embs)
    print(f"Saved embeddings: {embs.shape}")

    xy = umap_project(embs)
    np.save(OUT / "umap_2d.npy", xy)
    print(f"Saved UMAP projection: {xy.shape}")

    (OUT / "instance_ids.json").write_text(json.dumps(instance_ids, indent=2))
    (OUT / "repos.json").write_text(json.dumps(repos, indent=2))

    ease_list = [ease.get(iid, 0.0) for iid in instance_ids]
    (OUT / "ease_scores.json").write_text(json.dumps(
        {iid: ease.get(iid, 0.0) for iid in instance_ids}, indent=2
    ))

    metadata = {
        "model": EMBED_MODEL,
        "date": datetime.utcnow().isoformat(),
        "n_instances": len(instance_ids),
        "embedding_dim": int(embs.shape[1]),
        "normalized": True,
        "umap_params": UMAP_PARAMS,
        "source": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
        "ease_agents": sorted(TARGET_AGENTS),
        "n_resolved_by_any": int(sum(1 for v in ease_list if v > 0)),
        "n_resolved_by_all": int(sum(1 for v in ease_list if v == 1.0)),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {metadata['n_instances']} instances, "
          f"{metadata['n_resolved_by_any']} resolved by at least one agent")

    make_scatter(instance_ids, repos, xy, ease, OUT / "umap_scatter.html")
    make_scatter_by_repo(instance_ids, repos, xy, OUT / "umap_by_repo.png")
    print("\nDone. Outputs in:", OUT)


if __name__ == "__main__":
    main()
