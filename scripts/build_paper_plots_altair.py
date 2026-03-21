#!/usr/bin/env python3
"""
Altair versions of the paper plots:
  1. cross_agent_distances_by_pair.png
  2. localization_metrics.png
  3. compositionality_curve.png
  4. model_vocab_overlap_semantic.png

Usage:
    uv run python scripts/build_paper_plots_altair.py
"""
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PLOTS_OUT = ROOT / "notebooks" / "plots" / "cross_agent_all"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

DS_DIR  = ROOT / "output" / "datasets" / "swe_bench_lite_resolved"
CA_DIR  = ROOT / "output" / "datasets" / "cross_agent_all"
TRAJS   = CA_DIR / "all_trajectories.jsonl"

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
PINK   = "#CC79A7"
GRAY   = "#999999"

SHORT = {"Claude 3.5": "C3.5", "GPT-4": "G4", "GPT-4o": "G4o", "Claude 3 Opus": "C3O"}


# ---------------------------------------------------------------------------
# 1. Cross-agent distances by model pair
# ---------------------------------------------------------------------------

def build_cross_agent_distances():
    df = pd.read_parquet(DS_DIR / "cross_agent_distances.parquet")
    df["pair"] = df["agent_a"] + " / " + df["agent_b"]
    stage_labels = {
        "d_tokens":  "Tokens (Levenshtein)",
        "d_edits":   "Edits (sym diff)",
        "d_modules": "Modules (Jaccard)",
    }
    df_long = df.melt(
        id_vars=["pair"],
        value_vars=["d_tokens", "d_edits", "d_modules"],
        var_name="stage", value_name="distance",
    ).dropna()
    df_long["stage_label"] = df_long["stage"].map(stage_labels)

    pair_order = sorted(df_long["pair"].unique())
    stage_order = ["Tokens (Levenshtein)", "Edits (sym diff)", "Modules (Jaccard)"]

    chart = alt.Chart(df_long).mark_boxplot(
        extent="min-max", size=20, median=alt.MarkConfig(color="black", strokeWidth=1.5),
    ).encode(
        x=alt.X("pair:N", title=None, sort=pair_order,
                axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("distance:Q", title="Distance", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("pair:N",
                        scale=alt.Scale(scheme="tableau10"),
                        legend=alt.Legend(title=None, labelFontSize=8)),
    ).facet(
        column=alt.Column("stage_label:N", title=None, sort=stage_order,
                          header=alt.Header(labelFontSize=9)),
    ).configure_axis(
        grid=False, labelFontSize=8, titleFontSize=9,
    ).configure_legend(
        orient="bottom",
        columns=3,
        labelFontSize=8,
        symbolSize=60,
    ).configure_view(strokeWidth=0)

    out = PLOTS_OUT / "cross_agent_distances_by_pair.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 2. Localization metrics
# ---------------------------------------------------------------------------

def build_localization_metrics():
    df = pd.read_parquet(CA_DIR / "localization_metrics.parquet")
    df["outcome"] = df["resolved"].map({True: "resolved", False: "failed"})

    metric_labels = {
        "files_edited":        "Files edited",
        "step_of_first_edit":  "Step of first edit (normalised)",
        "early_patch_focus":   "Early patch focus",
    }
    df_long = df.melt(
        id_vars=["outcome"],
        value_vars=list(metric_labels.keys()),
        var_name="metric", value_name="value",
    ).dropna()
    df_long["metric_label"] = df_long["metric"].map(metric_labels)

    metric_order = list(metric_labels.values())
    color_scale = alt.Scale(
        domain=["resolved", "failed"],
        range=[BLUE, ORANGE],
    )

    chart = alt.Chart(df_long).mark_boxplot(
        extent="min-max", size=25, median=alt.MarkConfig(color="black", strokeWidth=1.5),
    ).encode(
        x=alt.X("outcome:N", title=None, axis=alt.Axis(labelFontSize=9)),
        y=alt.Y("value:Q", title="Value"),
        color=alt.Color("outcome:N", scale=color_scale, legend=None),
    ).properties(
        width=100, height=120,
    ).facet(
        column=alt.Column("metric_label:N", title=None, sort=metric_order,
                          header=alt.Header(labelFontSize=9)),
    ).configure_axis(
        grid=False, labelFontSize=8, titleFontSize=9,
    ).configure_view(strokeWidth=0)

    out = PLOTS_OUT / "localization_metrics.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 3. Compositionality curve
# ---------------------------------------------------------------------------

def _ngrams(seq, n):
    it = iter(seq)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for tok in it:
        window = window[1:] + (tok,)
        yield window


def extract_ngrams(tokens, ns=(2, 3)):
    out = []
    for n in ns:
        out.extend(_ngrams(tokens, n))
    return out


def build_reuse_curve(all_ngrams, topk_values):
    global_counts: Counter = Counter()
    for ngrams in all_ngrams:
        global_counts.update(set(ngrams))
    rows = []
    for k in topk_values:
        vocab = set(p for p, _ in global_counts.most_common(k))
        coverages = []
        for ngrams in all_ngrams:
            if not ngrams:
                continue
            unique = set(ngrams)
            coverages.append(len(unique & vocab) / len(unique))
        rows.append({"topk": k, "mean_coverage": np.mean(coverages), "std": np.std(coverages)})
    return pd.DataFrame(rows)


def build_compositionality_curve():
    df = pd.read_parquet(DS_DIR / "test.parquet")
    all_ngrams = []
    for raw in df["tokens"]:
        toks = json.loads(raw) if isinstance(raw, str) else (raw or [])
        all_ngrams.append(extract_ngrams(toks))

    topk_values = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    curve = build_reuse_curve(all_ngrams, topk_values)
    curve["mean_pct"] = curve["mean_coverage"] * 100
    curve["upper"] = (curve["mean_coverage"] + curve["std"]) * 100
    curve["lower"] = (curve["mean_coverage"] - curve["std"]) * 100

    band = alt.Chart(curve).mark_area(opacity=0.15, color=BLUE).encode(
        x=alt.X("topk:Q", title="Vocabulary size (top-K edit patterns)"),
        y=alt.Y("lower:Q", title="Mean compositional coverage (%)"),
        y2="upper:Q",
    )
    line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2).encode(
        x="topk:Q",
        y=alt.Y("mean_pct:Q", scale=alt.Scale(domain=[0, 102])),
    )
    hline = alt.Chart(pd.DataFrame([{"y": 70}])).mark_rule(
        strokeDash=[4, 4], color=GRAY, strokeWidth=0.8,
    ).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame([{"topk": 205, "y": 71.5, "text": "70%"}])).mark_text(
        align="left", fontSize=8, color=GRAY,
    ).encode(x="topk:Q", y="y:Q", text="text:N")

    chart = (band + line + hline + label).properties(
        width=320, height=220,
    ).configure_axis(
        grid=False, labelFontSize=9, titleFontSize=10,
    ).configure_view(strokeWidth=0)

    out = PLOTS_OUT / "compositionality_curve.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 4. Semantic model overlap (cosine similarity heatmap)
# ---------------------------------------------------------------------------

def compute_lift(all_ngrams, fix_types, min_support=5, min_lift=2.0):
    n = len(all_ngrams)
    ft_counts: Counter = Counter(fix_types)
    base_rate = {ft: c / n for ft, c in ft_counts.items()}
    pattern_support: Counter = Counter()
    pattern_ft: Counter = Counter()
    for ngrams, ft in zip(all_ngrams, fix_types):
        for pat in set(ngrams):
            pattern_support[pat] += 1
            pattern_ft[(pat, ft)] += 1
    rows = []
    for (pat, ft), co in pattern_ft.items():
        supp = pattern_support[pat]
        if supp < min_support:
            continue
        p_ft = co / supp
        lift = p_ft / base_rate[ft]
        rows.append({"pattern_tuple": pat, "lift": lift})
    if not rows:
        return set()
    df = pd.DataFrame(rows).sort_values("lift", ascending=False)
    df = df.drop_duplicates("pattern_tuple")
    return set(df[df["lift"] >= min_lift]["pattern_tuple"])


def build_semantic_overlap():
    if not TRAJS.exists():
        print(f"Skipping semantic overlap: {TRAJS.name} not found")
        return

    ft_path = DS_DIR / "fix_types.json"
    fix_type_map = {}
    if ft_path.exists():
        with open(ft_path) as f:
            ft_data = json.load(f)
        fix_type_map = {r["instance_id"]: r["fix_type"] for r in ft_data["results"]}

    from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

    model_ngrams: dict[str, list] = {}
    model_iids: dict[str, list] = {}
    with open(TRAJS) as f:
        for line in f:
            rec = json.loads(line)
            iid = rec["instance_id"]
            for model, traj in rec["models"].items():
                if not traj.get("resolved") or not traj.get("patch"):
                    continue
                toks = patch_to_ast_sequence(traj["patch"])
                model_ngrams.setdefault(model, []).append(extract_ngrams(toks))
                model_iids.setdefault(model, []).append(iid)

    model_fix_types = {
        m: [fix_type_map.get(iid, "unknown") for iid in iids]
        for m, iids in model_iids.items()
    }

    model_freq: dict[str, Counter] = {}
    for model, per_instance in model_ngrams.items():
        fts = model_fix_types.get(model, [])
        high_lift = compute_lift(per_instance, fts)
        freq: Counter = Counter()
        for ngrams in per_instance:
            for pat in set(ngrams):
                if pat in high_lift:
                    freq[pat] += 1
        model_freq[model] = freq

    vocab = sorted({p for freq in model_freq.values() for p in freq})
    models = list(model_freq.keys())
    vecs = np.zeros((len(models), len(vocab)))
    vocab_idx = {p: i for i, p in enumerate(vocab)}
    for i, m in enumerate(models):
        for pat, cnt in model_freq[m].items():
            if pat in vocab_idx:
                vecs[i, vocab_idx[pat]] = cnt

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = (vecs / norms) @ (vecs / norms).T

    short_models = [SHORT.get(m, m) for m in models]
    rows = []
    for i, ma in enumerate(short_models):
        for j, mb in enumerate(short_models):
            rows.append({"model_a": ma, "model_b": mb, "similarity": round(matrix[i, j], 2)})
    df_heat = pd.DataFrame(rows)

    chart = alt.Chart(df_heat).mark_rect().encode(
        x=alt.X("model_a:N", title=None, axis=alt.Axis(labelAngle=-30, labelFontSize=9)),
        y=alt.Y("model_b:N", title=None, axis=alt.Axis(labelFontSize=9)),
        color=alt.Color("similarity:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        title="Cosine similarity"),
    ) + alt.Chart(df_heat).mark_text(fontSize=9).encode(
        x="model_a:N",
        y="model_b:N",
        text=alt.Text("similarity:Q", format=".2f"),
        color=alt.condition("datum.similarity > 0.6", alt.value("white"), alt.value("black")),
    )

    chart = chart.properties(width=220, height=200).configure_axis(
        grid=False,
    ).configure_view(strokeWidth=0)

    out = PLOTS_OUT / "model_vocab_overlap_semantic.png"
    chart.save(str(out), scale_factor=2)
    print(f"Saved {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building cross-agent distances by pair...")
    build_cross_agent_distances()

    print("Building localization metrics...")
    build_localization_metrics()

    print("Building compositionality curve...")
    build_compositionality_curve()

    print("Building semantic model overlap...")
    build_semantic_overlap()
