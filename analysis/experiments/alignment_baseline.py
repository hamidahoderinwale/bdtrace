"""Problem -> Procedure alignment with shuffled baseline.

Embeds hunk descriptions (fix side) and computes cosine similarity to
issue embeddings. Compares real matched pairs to shuffled pairs, and
correlates alignment score with task ease.

Outputs:
    output/experiments/alignment_scores.json
    output/experiments/alignment_distribution.png
    output/experiments/alignment_vs_ease.png
"""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, GRAY, ORANGE, NEAR_BLACK
register()

OUT  = ROOT / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)


def load_issue_embeddings():
    emb = np.load(ROOT / "output/issue_embeddings/embeddings.npy")
    ids = json.loads((ROOT / "output/issue_embeddings/instance_ids.json").read_text())
    ease = json.loads((ROOT / "output/issue_embeddings/ease_scores.json").read_text())
    return {iid: emb[i] for i, iid in enumerate(ids)}, ease


def embed_fix_descriptions():
    from sentence_transformers import SentenceTransformer
    desc_path = ROOT / "output/hunk_descriptions/descriptions.json"
    raw = json.loads(desc_path.read_text())

    model = SentenceTransformer("all-MiniLM-L6-v2")
    fix_embs = {}
    for iid, descs in raw.items():
        if not descs:
            continue
        vecs = model.encode(descs, normalize_embeddings=True, show_progress_bar=False)
        fix_embs[iid] = vecs.mean(axis=0)
    return fix_embs


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    print("Loading issue embeddings...")
    issue_embs, ease = load_issue_embeddings()

    print("Embedding fix descriptions...")
    fix_embs = embed_fix_descriptions()

    shared = sorted(set(issue_embs) & set(fix_embs))
    print(f"Matched instances: {len(shared)}")

    issue_mat = np.array([issue_embs[i] for i in shared])
    fix_mat   = np.array([fix_embs[i]   for i in shared])

    # Real similarities
    real_sims = (issue_mat * fix_mat).sum(axis=1)

    # Shuffled baseline (1000 random permutations, sample same N)
    rng = np.random.default_rng(42)
    shuffled = []
    for _ in range(1000):
        perm = rng.permutation(len(shared))
        shuffled.extend((issue_mat * fix_mat[perm]).sum(axis=1).tolist())
    shuffled = np.array(shuffled)

    ease_vals = np.array([ease.get(i, 0.0) for i in shared])

    scores = [
        {"instance_id": iid, "alignment": float(s), "ease": float(e)}
        for iid, s, e in zip(shared, real_sims, ease_vals)
    ]
    (OUT / "alignment_scores.json").write_text(json.dumps({
        "mean_real": float(real_sims.mean()),
        "mean_shuffled": float(shuffled.mean()),
        "n": len(shared),
        "pearson_r_ease": float(np.corrcoef(real_sims, ease_vals)[0, 1]),
        "scores": scores,
    }, indent=2))
    print(f"Mean real: {real_sims.mean():.3f}  Mean shuffled: {shuffled.mean():.3f}")
    print(f"Pearson r vs ease: {np.corrcoef(real_sims, ease_vals)[0, 1]:.3f}")

    # --- Plot 1: distribution real vs shuffled ---
    bins = np.linspace(0, 1, 41)
    real_counts, edges = np.histogram(real_sims, bins=bins, density=True)
    shuf_counts, _     = np.histogram(shuffled[:len(shared)], bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dist_rows = (
        [{"sim": c, "density": v, "group": "Issue-fix pairs"} for c, v in zip(centers, real_counts)] +
        [{"sim": c, "density": v, "group": "Shuffled"} for c, v in zip(centers, shuf_counts)]
    )
    dist_df = pd.DataFrame(dist_rows)
    cscale = alt.Scale(domain=["Issue-fix pairs", "Shuffled"], range=[BLUE, GRAY])

    dist_chart = (
        alt.Chart(dist_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("sim:Q", title="Cosine similarity",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(domain=False, ticks=False, values=[0, 0.25, 0.5, 0.75, 1.0])),
            y=alt.Y("density:Q", title="Density",
                    axis=alt.Axis(domain=False, ticks=False)),
            color=alt.Color("group:N", scale=cscale,
                            legend=alt.Legend(title=None, orient="bottom")),
        )
        .properties(
            title=alt.TitleParams("Problem to procedure alignment vs shuffled baseline",
                                  fontSize=13, color="#111111", anchor="start"),
            width=400, height=220,
        )
        .configure_view(strokeWidth=0)
    )
    dist_chart.save(str(OUT / "alignment_distribution.png"), scale_factor=2)
    print("Saved alignment_distribution.png")

    # --- Plot 2: alignment vs ease ---
    sc_df = pd.DataFrame({"alignment": real_sims, "ease": ease_vals})
    scatter = (
        alt.Chart(sc_df)
        .mark_circle(size=40, opacity=0.5, color=BLUE)
        .encode(
            x=alt.X("alignment:Q", title="Alignment score (cosine similarity)",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.25, 0.5, 0.75, 1.0])),
            y=alt.Y("ease:Q", title="Ease (fraction of agents resolving)",
                    scale=alt.Scale(domain=[-0.05, 1.05]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.33, 0.67, 1.0],
                                  format=".0%")),
        )
        .properties(
            title=alt.TitleParams("Alignment score vs task ease",
                                  fontSize=13, color="#111111", anchor="start"),
            width=380, height=260,
        )
        .configure_view(strokeWidth=0)
    )
    scatter.save(str(OUT / "alignment_vs_ease.png"), scale_factor=2)
    print("Saved alignment_vs_ease.png")


if __name__ == "__main__":
    main()
