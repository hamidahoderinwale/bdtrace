"""Navigational gap: semantic distance between issue text and fix location.

For each instance, computes the cosine distance between:
  - The issue embedding (what the problem says)
  - An embedding of the fix location (which module, which function, which code)

The gap is the distance the agent must traverse from problem description
to fix location in semantic space. Larger gap = harder to find the fix
by reading the issue alone.

Tests whether gap size:
  1. Negatively correlates with ease (larger gap → harder)
  2. Predicts whether agents need exploration-first strategies

Outputs:
    output/experiments/navigational_gap.json
    output/experiments/navigational_gap_vs_ease.png
    output/experiments/navigational_gap_by_outcome.png
"""
from __future__ import annotations
import json, re, sys
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.theme import register, BLUE, ORANGE, GREEN, GRAY
register()

OUT = ROOT / "output" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)


def parse_patch(patch: str) -> dict:
    """Extract file paths, function names, and context lines from a gold patch."""
    files = re.findall(r"diff --git a/(.+?) b/", patch)
    # Hunk headers: @@ -N,M +N,M @@ [optional context like def func_name]
    funcs = re.findall(r"@@[^@]+@@\s+(?:def |class )?(\w+)", patch)
    # Removed lines (what was there before the fix) as code context
    context_lines = [
        l[1:].strip() for l in patch.splitlines()
        if l.startswith("-") and not l.startswith("---") and l[1:].strip()
    ]
    return {"files": files, "funcs": funcs, "context": context_lines[:8]}


def file_to_module(path: str) -> str:
    """Convert file path to readable module description."""
    parts = Path(path).with_suffix("").parts
    return " ".join(parts).replace("_", " ")


def build_fix_location_text(files: list[str], funcs: list[str], context: list[str]) -> str:
    """Construct natural language description of fix location."""
    parts = []
    if files:
        modules = [file_to_module(f) for f in files[:2]]
        parts.append("Module: " + "; ".join(modules))
    if funcs:
        unique_funcs = list(dict.fromkeys(funcs))[:3]
        readable = [f.replace("_", " ") for f in unique_funcs]
        parts.append("Function: " + ", ".join(readable))
    if context:
        parts.append("Code context: " + " | ".join(context[:4]))
    return ". ".join(parts)


def main():
    # Load issue embeddings
    issue_embs = np.load(ROOT / "output/issue_embeddings/embeddings.npy")
    iids = json.loads((ROOT / "output/issue_embeddings/instance_ids.json").read_text())
    ease = json.loads((ROOT / "output/issue_embeddings/ease_scores.json").read_text())
    issue_map = {iid: issue_embs[i] for i, iid in enumerate(iids)}

    # Load gold patches from SWE-bench Lite
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    patch_map = {r["instance_id"]: r["patch"] for r in ds}

    # Build fix location texts
    print("Parsing gold patches...")
    fix_texts = {}
    for iid, patch in patch_map.items():
        parsed = parse_patch(patch)
        text = build_fix_location_text(
            parsed["files"], parsed["funcs"], parsed["context"]
        )
        if text.strip():
            fix_texts[iid] = text

    # Embed fix locations
    print(f"Embedding {len(fix_texts)} fix locations...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    shared = sorted(set(issue_map) & set(fix_texts))
    print(f"Matched instances: {len(shared)}")

    fix_loc_list = [fix_texts[i] for i in shared]
    fix_embs = model.encode(fix_loc_list, normalize_embeddings=True,
                             batch_size=64, show_progress_bar=True)
    issue_vecs = np.array([issue_map[i] for i in shared])

    # Cosine similarity then gap = 1 - similarity
    similarities = (issue_vecs * fix_embs).sum(axis=1)
    gaps = 1.0 - similarities

    ease_vals = np.array([ease.get(i, 0.0) for i in shared])

    # Shuffled baseline
    rng = np.random.default_rng(42)
    shuf_sims = []
    for _ in range(500):
        perm = rng.permutation(len(shared))
        shuf_sims.extend((issue_vecs * fix_embs[perm]).sum(axis=1).tolist())
    shuf_sims = np.array(shuf_sims)

    pearson_r = float(np.corrcoef(gaps, ease_vals)[0, 1])
    print(f"Mean gap (real):     {gaps.mean():.3f}")
    print(f"Mean gap (shuffled): {(1 - shuf_sims.mean()):.3f}")
    print(f"Pearson r (gap vs ease): {pearson_r:.3f}")

    scores = [
        {"instance_id": iid, "gap": float(g), "similarity": float(s),
         "ease": float(e), "fix_location_text": fix_texts[iid]}
        for iid, g, s, e in zip(shared, gaps, similarities, ease_vals)
    ]
    (OUT / "navigational_gap.json").write_text(json.dumps({
        "mean_gap_real": float(gaps.mean()),
        "mean_gap_shuffled": float(1 - shuf_sims.mean()),
        "mean_similarity_real": float(similarities.mean()),
        "mean_similarity_shuffled": float(shuf_sims.mean()),
        "pearson_r_ease": pearson_r,
        "n": len(shared),
        "scores": scores,
    }, indent=2))

    # --- Plot 1: gap vs ease scatter ---
    sc_df = pd.DataFrame({
        "gap": gaps,
        "ease": ease_vals,
        "resolved_any": (ease_vals > 0).astype(int),
    })
    cscale = alt.Scale(domain=[0, 1], range=[ORANGE, BLUE])

    scatter = (
        alt.Chart(sc_df)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X("gap:Q",
                    title="Navigational gap (1 - cosine similarity)",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.25, 0.5, 0.75, 1.0])),
            y=alt.Y("ease:Q",
                    title="Ease (fraction of agents resolving)",
                    scale=alt.Scale(domain=[-0.05, 1.05]),
                    axis=alt.Axis(domain=False, ticks=False,
                                  values=[0, 0.33, 0.67, 1.0], format=".0%")),
            color=alt.Color("ease:Q", scale=cscale, legend=None),
        )
        .properties(
            title=alt.TitleParams(
                "Navigational gap vs task ease",
                subtitle=f"Pearson r = {pearson_r:.2f}  (n = {len(shared)})",
                subtitleFontSize=10, subtitleColor="#888888",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=380, height=260,
        )
        .configure_view(strokeWidth=0)
    )
    scatter.save(str(OUT / "navigational_gap_vs_ease.png"), scale_factor=2)
    print("Saved navigational_gap_vs_ease.png")

    # --- Plot 2: gap distribution by outcome (resolved vs not) ---
    sc_df["outcome"] = sc_df["ease"].apply(
        lambda e: "All resolved" if e == 1.0
        else ("None resolved" if e == 0.0 else "Partially resolved")
    )
    outcome_order = ["All resolved", "Partially resolved", "None resolved"]
    outcome_colors = [BLUE, GREEN, ORANGE]
    oc_scale = alt.Scale(domain=outcome_order, range=outcome_colors)

    summary_rows = []
    for out in outcome_order:
        sub = sc_df[sc_df["outcome"] == out]["gap"]
        if len(sub) >= 5:
            summary_rows.append({
                "outcome": out,
                "mean_gap": float(sub.mean()),
                "zero": 0.0,
                "n": len(sub),
            })
    sum_df = pd.DataFrame(summary_rows)

    rule = (
        alt.Chart(sum_df)
        .mark_rule(strokeWidth=2.5, opacity=0.6)
        .encode(
            y=alt.Y("outcome:N", sort=outcome_order,
                    axis=alt.Axis(title=None, domain=False, ticks=False,
                                  labelFontSize=12)),
            x=alt.X("zero:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Mean navigational gap",
                                  domain=False, ticks=False,
                                  values=[0, 0.25, 0.5, 0.75, 1.0])),
            x2="mean_gap:Q",
            color=alt.Color("outcome:N", scale=oc_scale, legend=None),
        )
    )
    pts = (
        alt.Chart(sum_df)
        .mark_point(size=120, filled=True, strokeWidth=1.5, stroke="white")
        .encode(
            y=alt.Y("outcome:N", sort=outcome_order),
            x=alt.X("mean_gap:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("outcome:N", scale=oc_scale, legend=None),
        )
    )
    labels = (
        alt.Chart(sum_df)
        .mark_text(align="left", dx=10, fontSize=11, color="#444444")
        .encode(
            y=alt.Y("outcome:N", sort=outcome_order),
            x=alt.X("mean_gap:Q", scale=alt.Scale(domain=[0, 1])),
            text=alt.Text("mean_gap:Q", format=".3f"),
        )
    )
    gap_chart = (
        (rule + pts + labels)
        .properties(
            title=alt.TitleParams(
                "Navigational gap by resolution outcome",
                fontSize=13, color="#111111", anchor="start",
            ),
            width=340, height=120,
        )
        .configure_view(strokeWidth=0)
    )
    gap_chart.save(str(OUT / "navigational_gap_by_outcome.png"), scale_factor=2)
    print("Saved navigational_gap_by_outcome.png")


if __name__ == "__main__":
    main()
