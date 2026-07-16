#!/usr/bin/env python3
"""
Conditional procedural analysis: within fix-type cluster, does procedure predict pass?

Analyses:
1. Fix-type distribution overview (count, pass rate per fix type).
2. Within each fix type: pass rate vs corpus motif quantile (d_motifs).
3. Efficiency by fix type: pass rate vs edit-retry bucket, stratified.
4. Procedure template mining per fix type: most common typed action sequences.
5. Cross-fix-type confusion: where do agents use similar procedures for different fix types?

Inputs:
    output/datasets/swe_bench_lite_resolved/fix_types.json  (from label_fix_types.py)
    output/trajectory_features_lite_*.parquet               (from fetch_trajectories.py)
    output/swebench_results/lite_*.json                     (pass/fail)

Usage:
    uv run python scripts/analyze_fix_types.py
    uv run python scripts/analyze_fix_types.py --model 20240620_sweagent_claude3.5sonnet
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_paths = [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]
for _p in _env_paths:
    if _p.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_p)
        except ImportError:
            pass
        break


def load_fix_types(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {r["instance_id"]: r for r in data["results"]}


def load_pass_fail(path: Path) -> dict[str, bool]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {r["instance_id"]: bool(r.get("resolved", False)) for r in data}
    return {k: bool(v) for k, v in data.items()}


def load_trajectories(model_slug: str, split: str = "lite") -> pd.DataFrame | None:
    paths = list(Path("output").glob(f"trajectory_features_{split}_{model_slug}*.parquet"))
    if not paths:
        paths = list(Path("output").glob(f"trajectory_features_{split}*.parquet"))
    if not paths:
        return None
    return pd.read_parquet(sorted(paths)[-1])


def _pass_rate_table(group: pd.DataFrame, label: str) -> None:
    total = len(group)
    passed = group["passed"].sum()
    print(f"  {label:35s}: {passed:3d}/{total:3d} = {100*passed/total:.1f}%")


## Analysis 1: Fix-type overview

def analysis_fix_type_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Pass rate and count per fix type."""
    agg = (
        df.groupby("fix_type")
        .agg(n=("passed", "count"), n_pass=("passed", "sum"))
        .reset_index()
    )
    agg["pass_rate"] = agg["n_pass"] / agg["n"]
    return agg.sort_values("n", ascending=False)


def plot_fix_type_overview(agg: pd.DataFrame, out_dir: Path) -> None:
    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("fix_type:N", sort="-y", title="Fix Type"),
            y=alt.Y("pass_rate:Q", title="Pass Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), title="Count"),
            tooltip=["fix_type", "n", "n_pass", alt.Tooltip("pass_rate:Q", format=".1%")],
        )
        .properties(title="Pass Rate by Fix Type", width=600, height=300)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chart.save(str(out_dir / "fix_type_overview.html"))
    print(f"  Saved fix_type_overview.html")


## Analysis 2: Efficiency by fix type (edit_retries vs pass rate)

def analysis_efficiency_by_fix_type(df: pd.DataFrame, top_n: int = 4) -> None:
    top_types = df["fix_type"].value_counts().head(top_n).index.tolist()
    print("\n== Efficiency by fix type (pass rate vs edit-retry bucket) ==")
    for ft in top_types:
        sub = df[df["fix_type"] == ft]
        print(f"\n  [{ft}] n={len(sub)}")
        # edit-retry bucket: 0, 1-2, 3+
        sub = sub.copy()
        sub["retry_bucket"] = pd.cut(
            sub["edit_retries"].fillna(0),
            bins=[-1, 0, 2, 100],
            labels=["0", "1-2", "3+"],
        )
        for bucket, g in sub.groupby("retry_bucket", observed=True):
            _pass_rate_table(g, f"  retries={bucket}")


## Analysis 3: Corpus motif distance vs pass rate, within fix type

def analysis_motif_by_fix_type(
    df: pd.DataFrame,
    pass_fail: dict[str, bool],
    top_n: int = 4,
) -> None:
    if "d_motifs" not in df.columns:
        print("  d_motifs not in trajectory data, skipping motif analysis")
        return

    print("\n== Motif distance vs pass rate, within fix type ==")
    top_types = df["fix_type"].value_counts().head(top_n).index.tolist()
    for ft in top_types:
        sub = df[df["fix_type"] == ft].dropna(subset=["d_motifs"])
        if len(sub) < 10:
            continue
        quantiles = pd.qcut(sub["d_motifs"], q=3, labels=["near", "mid", "far"], duplicates="drop")
        sub = sub.copy()
        sub["quantile"] = quantiles
        print(f"\n  [{ft}] n={len(sub)}")
        for q, g in sub.groupby("quantile", observed=True):
            _pass_rate_table(g, f"  motif_dist={q}")


## Analysis 4: UMAP structural space colored by fix type

def plot_structural_space_by_fix_type(
    ft_map: dict[str, dict],
    data_dir: Path,
    out_dir: Path,
) -> None:
    """2D UMAP of structural distance matrices, colored by fix type.
    Shows whether fix type clusters in structural space (it doesn't)."""
    import numpy as np

    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed, skipping. Run: uv add umap-learn")
        return

    matrices_path = data_dir / "matrices.npz"
    if not matrices_path.exists():
        print("  matrices.npz not found, skipping")
        return

    mats = np.load(matrices_path)
    labels_df = pd.read_parquet(data_dir / "labels.parquet")
    instance_ids = labels_df["instance_id"].tolist()

    def norm(D):
        mx = float(D.max())
        return D / mx if mx > 0 else D

    # Combine the two independent structural dimensions
    available = [k for k in ["modules", "motifs"] if k in mats]
    if not available:
        available = list(mats.keys())[:2]
    D_combined = sum(norm(mats[k].astype(float)) for k in available) / len(available)

    reducer = UMAP(n_components=2, metric="precomputed", random_state=42,
                   n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(D_combined)

    df = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "instance_id": instance_ids,
        "fix_type": [ft_map.get(iid, {}).get("fix_type", "other") for iid in instance_ids],
    })

    top_types = df["fix_type"].value_counts().head(7).index.tolist()
    df["fix_type_display"] = df["fix_type"].apply(lambda x: x if x in top_types else "other")
    types_order = df["fix_type_display"].value_counts().index.tolist()
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]

    chart = (
        alt.Chart(df)
        .mark_circle(size=55, opacity=0.75)
        .encode(
            alt.X("x:Q", title=None, axis=None),
            alt.Y("y:Q", title=None, axis=None),
            alt.Color("fix_type_display:N", sort=types_order,
                      scale=alt.Scale(range=palette), title="fix type"),
            alt.Tooltip(["instance_id:N", "fix_type:N"]),
        )
        .properties(
            width=420, height=340,
            title=alt.TitleParams(
                text="Structural space of SWE-Bench Lite tasks",
                subtitle="2D UMAP of structural distances (files changed + edit patterns). Color = fix type.",
                anchor="start",
            ),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chart.save(str(out_dir / "structural_space_by_fix_type.png"))
    print("  Saved structural_space_by_fix_type.png")


## Analysis 5: Structural distance by fix type

def analysis_structural_by_fix_type(
    ft_map: dict[str, dict],
    data_dir: Path,
    out_dir: Path,
) -> None:
    """For each structural representation, compare mean pairwise distance
    within-fix-type vs. between-fix-type. A lower intra/inter ratio means
    the fix type clusters structurally."""
    import numpy as np

    labels_df = pd.read_parquet(data_dir / "labels.parquet")
    matrices_path = data_dir / "matrices.npz"
    if not matrices_path.exists():
        print("  matrices.npz not found, skipping structural analysis")
        return

    mats = np.load(matrices_path)
    repr_names = {"edits": "raw edits", "edits_set_diff": "edit set-diff",
                  "modules": "dependency graph", "motifs": "recurring patterns"}

    instance_ids = labels_df["instance_id"].tolist()
    fix_type_vec = [ft_map.get(iid, {}).get("fix_type", None) for iid in instance_ids]

    rows = []
    for repr_key, repr_label in repr_names.items():
        if repr_key not in mats:
            continue
        D = mats[repr_key]
        n = len(instance_ids)
        intra, inter = [], []
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = fix_type_vec[i], fix_type_vec[j]
                if fi is None or fj is None:
                    continue
                d = float(D[i, j])
                if fi == fj:
                    intra.append(d)
                else:
                    inter.append(d)
        if not intra or not inter:
            continue
        rows.append({
            "repr": repr_label,
            "intra": float(np.mean(intra)),
            "inter": float(np.mean(inter)),
            "ratio": float(np.mean(intra) / np.mean(inter)) if np.mean(inter) > 0 else None,
        })
        print(f"  {repr_label:22s}  intra={np.mean(intra):.3f}  inter={np.mean(inter):.3f}  ratio={np.mean(intra)/np.mean(inter):.3f}")

    if not rows:
        return

    df = pd.DataFrame(rows)
    melted = df.melt(id_vars="repr", value_vars=["intra", "inter"],
                     var_name="pair_type", value_name="mean_distance")

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            alt.X("repr:N", title=None, axis=alt.Axis(labelAngle=-20)),
            alt.Y("mean_distance:Q", title="mean pairwise distance"),
            alt.Color("pair_type:N",
                      scale=alt.Scale(domain=["intra", "inter"], range=["#0072B2", "#E69F00"]),
                      title="pair type"),
            alt.XOffset("pair_type:N"),
            alt.Tooltip(["repr:N", "pair_type:N", "mean_distance:Q"]),
        )
        .properties(
            width=340, height=220,
            title=alt.TitleParams(
                text="Structural distance: within vs. between fix type",
                subtitle="Lower intra than inter = fix type clusters structurally",
                anchor="start",
            ),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chart.save(str(out_dir / "fix_type_structural_distance.png"))
    print(f"  Saved fix_type_structural_distance.png")


## Analysis 5: Procedure templates per fix type

def analysis_procedure_templates(df: pd.DataFrame, top_n: int = 3) -> None:
    if "action_sequence" not in df.columns:
        print("  action_sequence not in trajectory data, skipping template analysis")
        return

    print("\n== Top procedure templates per fix type ==")
    top_types = df["fix_type"].value_counts().head(top_n).index.tolist()
    for ft in top_types:
        sub = df[df["fix_type"] == ft].dropna(subset=["action_sequence"])
        seqs = sub["action_sequence"].dropna().tolist()
        counts = Counter(seqs)
        print(f"\n  [{ft}] n={len(seqs)}, unique seqs={len(counts)}")
        for seq, cnt in counts.most_common(3):
            prate = sub[sub["action_sequence"] == seq]["passed"].mean()
            print(f"    [{cnt:3d}x, pass={prate:.2f}] {seq[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--fix-types", type=str, default="output/datasets/swe_bench_lite_resolved/fix_types.json")
    parser.add_argument("--pass-fail", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="notebooks/plots/fix_type_analysis")
    args = parser.parse_args()

    fix_types_path = Path(args.fix_types)
    if not fix_types_path.exists():
        print(f"Fix types not found at {fix_types_path}. Run label_fix_types.py first.")
        sys.exit(1)

    ft_map = load_fix_types(fix_types_path)
    print(f"Loaded {len(ft_map)} fix-type labels")

    # Locate pass/fail
    pf_path = Path(args.pass_fail) if args.pass_fail else None
    if pf_path is None:
        candidates = sorted(Path("output/swebench_results").glob("lite_*.json"))
        if candidates:
            pf_path = candidates[-1]
        else:
            candidates = sorted(Path("output/swebench_results").glob("verified_*.json"))
            if candidates:
                pf_path = candidates[-1]
    if pf_path is None or not pf_path.exists():
        print("Pass/fail file not found. Run fetch_swebench_results.py first.")
        sys.exit(1)

    pass_fail = load_pass_fail(pf_path)
    print(f"Loaded {len(pass_fail)} pass/fail labels from {pf_path.name}")

    # Build base dataframe from fix types + pass/fail
    rows: list[dict[str, Any]] = []
    for iid, ft in ft_map.items():
        if iid not in pass_fail:
            continue
        rows.append({
            "instance_id": iid,
            "repo": ft.get("repo", ""),
            "fix_type": ft["fix_type"],
            "confidence": ft.get("confidence", "medium"),
            "n_files_changed": ft.get("n_files_changed", 1),
            "net_lines": ft.get("net_lines", 0),
            "added_has_if": ft.get("added_has_if", False),
            "added_has_raise": ft.get("added_has_raise", False),
            "added_has_try": ft.get("added_has_try", False),
            "passed": pass_fail[iid],
        })
    df = pd.DataFrame(rows)
    print(f"Base df: {len(df)} instances with fix type + pass/fail overlap")

    # Merge trajectory features if available
    model_slug = args.model or ""
    traj_df = load_trajectories(model_slug)
    if traj_df is not None:
        before = len(df)
        df = df.merge(traj_df.drop(columns=["passed"], errors="ignore"), on="instance_id", how="left")
        print(f"Merged trajectory features ({before} -> {len(df)} rows, {traj_df.columns.tolist()[:8]}...)")
    else:
        print("No trajectory features found — skipping trajectory analyses")

    ## Run analyses
    print("\n" + "=" * 60)
    print("Analysis 1: Fix-type overview")
    agg = analysis_fix_type_overview(df)
    print(agg.to_string(index=False))

    out_dir = Path(args.out_dir)
    plot_fix_type_overview(agg, out_dir)

    print("\nAnalysis 2: Efficiency by fix type")
    if "edit_retries" in df.columns:
        analysis_efficiency_by_fix_type(df)
    else:
        print("  edit_retries not available")

    print("\nAnalysis 3: Motif distance vs pass rate within fix type")
    analysis_motif_by_fix_type(df, pass_fail)

    print("\nAnalysis 4: Structural space by fix type (UMAP)")
    plot_structural_space_by_fix_type(ft_map, fix_types_path.parent, out_dir)

    print("\nAnalysis 5: Structural distance by fix type")
    analysis_structural_by_fix_type(ft_map, fix_types_path.parent, out_dir)

    print("\nAnalysis 5: Procedure templates per fix type")
    analysis_procedure_templates(df)

    # Save merged df
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_dir / "merged_analysis.parquet"))
    print(f"\nSaved merged df to {out_dir}/merged_analysis.parquet")


if __name__ == "__main__":
    main()
