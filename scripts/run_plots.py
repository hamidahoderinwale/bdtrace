#!/usr/bin/env python3
"""
Generate analysis plots from existing pipeline outputs.

Run after: extraction, build_distance_matrices, run_diversity_analysis.
Optional: eval (for divergence_from_baseline), run_procedure_divergence (for gap).

Usage:
  python scripts/run_plots.py --output-dir notebooks/plots
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

try:
    import altair as alt
except ImportError:
    print("Install with: uv sync --extra notebooks")
    sys.exit(1)

# Structural representation display names (hierarchy: tokens → edits → modules)
REPR_LABELS = {
    "tokens": "Tokens",
    "edits_set_diff": "Edits (set diff)",
    "edits_tree": "Edits (tree)",
    "modules_graph": "Modules (graph)",
}


def _repr_label(k: str) -> str:
    return REPR_LABELS.get(k, k.replace("_", " ").title())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/plots"))
    parser.add_argument("--data-dir", type=Path, default=Path("output"))
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. swe_bench_lite_resolved). Default: prefer resolved if exists.")
    args = parser.parse_args()

    data = args.data_dir.resolve()
    if args.dataset:
        ds = data / "datasets" / args.dataset
        out = (args.output_dir / args.dataset).resolve()
    else:
        out = args.output_dir.resolve()
        if (data / "datasets" / "swe_bench_lite_resolved" / "test.parquet").exists():
            ds = data / "datasets" / "swe_bench_lite_resolved"
        else:
            ds = data / "datasets" / "swe_bench_lite"
    out.mkdir(parents=True, exist_ok=True)
    out_structural = out / "structural"  # Edits/structural rung only
    out_structural.mkdir(parents=True, exist_ok=True)

    # 1. Distributions (needs test.parquet)
    parquet_path = ds / "test.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        def _parse_json_col(ser):
            return ser.apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        for col in ["edits", "modules", "motifs"]:
            if col not in df.columns:
                continue
            # Parquet may store JSON as string (object or string dtype)
            sample = df[col].iloc[0]
            if isinstance(sample, str):
                df[col] = _parse_json_col(df[col])

        def edits_delta(e):
            return sum(c.get("delta", 0) for c in (e or []) if isinstance(c, dict))

        def edits_ops(e):
            return sum(len(c.get("operations", [])) for c in (e or []) if isinstance(c, dict))

        def modules_count(m):
            return len(m) if isinstance(m, list) else 0

        def motifs_len(m):
            return len(m.get("sequence", [])) if isinstance(m, dict) else 0

        stats = pd.DataFrame({
            "edits_delta": df["edits"].apply(edits_delta),
            "edits_ops": df["edits"].apply(edits_ops),
            "modules_count": df["modules"].apply(modules_count),
            "motifs_seq_len": df["motifs"].apply(motifs_len),
        })

        def _hist_chart(col: str, title: str, maxbins: int = 20) -> alt.Chart:
            """Histogram for numeric distributions; avoids overlapping x-labels."""
            return (
                alt.Chart(stats)
                .mark_bar(color="steelblue")
                .encode(
                    alt.X(
                        f"{col}:Q",
                        bin=alt.Bin(maxbins=maxbins),
                        title=title,
                        axis=alt.Axis(format="~s"),
                    ),
                    alt.Y("count():Q", title="instances"),
                )
                .properties(width=220, height=200, title=title)
            )

        def _value_count_chart(col: str, title: str) -> alt.Chart:
            """Bar chart for low-cardinality categorical (e.g. modules_count=0 only)."""
            vc = stats[col].value_counts().sort_index().reset_index()
            vc.columns = ["val", "count"]
            return (
                alt.Chart(vc)
                .mark_bar(color="steelblue")
                .encode(
                    alt.X("val:O", title=title),
                    alt.Y("count:Q", title="instances"),
                )
                .properties(width=220, height=200, title=title)
            )

        def _degenerate_note_chart(title: str, note: str) -> alt.Chart:
            """Text-only chart when a metric has no variation (avoids single-bar block)."""
            df_note = pd.DataFrame([{"note": note}])
            return (
                alt.Chart(df_note)
                .mark_text(size=14, align="center")
                .encode(text="note:N")
                .properties(width=220, height=200, title=title)
            )

        # Use histograms for high-cardinality numeric; value_counts for sparse
        n_unique = {c: stats[c].nunique() for c in stats.columns}
        if n_unique["modules_count"] == 1:
            val = int(stats["modules_count"].iloc[0])
            n_inst = len(stats)
            h3 = _degenerate_note_chart("modules_count", f"All {n_inst} instances: {val} (single-file)")
        else:
            h3 = _value_count_chart("modules_count", "modules_count")
        h1 = _hist_chart("edits_delta", "edits_delta") if n_unique["edits_delta"] > 15 else _value_count_chart("edits_delta", "edits_delta")
        h2 = _hist_chart("edits_ops", "edits_ops") if n_unique["edits_ops"] > 15 else _value_count_chart("edits_ops", "edits_ops")
        h4 = _hist_chart("motifs_seq_len", "motifs_seq_len") if n_unique["motifs_seq_len"] > 15 else _value_count_chart("motifs_seq_len", "motifs_seq_len")
        (alt.hconcat(h1, h2) & alt.hconcat(h3, h4)).save(out_structural / "distributions.png")
        print("Saved structural/distributions.png")

        repo_counts = stats.assign(repo=df.get("repo", "unknown")).groupby("repo").size().reset_index(name="n")
        alt.Chart(repo_counts).mark_bar(color="steelblue").encode(
            alt.X("repo:N", sort="-y", title="stratum (repo)"),
            alt.Y("n:Q", title="instances"),
        ).properties(
            title="Instances per stratum",
            width=400,
            height=250,
        ).save(out_structural / "stratum_counts.png")
        print("Saved structural/stratum_counts.png")

        # Complexity by number of AST stages (edit certificates per instance)
        n_stages = df["edits"].apply(lambda e: len(e) if isinstance(e, list) else 0)
        stage_stats = stats.assign(n_stages=n_stages)
        n_unique_stages = n_stages.nunique()
        if n_unique_stages >= 2:
            stage_dist = n_stages.value_counts().sort_index().reset_index()
            stage_dist.columns = ["n_stages", "count"]
            chart = alt.Chart(stage_dist).mark_bar(color="steelblue").encode(
                alt.X("n_stages:O", title="number of edit sites (AST stages)"),
                alt.Y("count:Q", title="instances"),
            )
            title = "Complexity: distribution by number of AST stages"
        else:
            # All instances share same n_stages; show edits_ops distribution instead
            ops_dist = stats["edits_ops"].value_counts().sort_index().reset_index()
            ops_dist.columns = ["edits_ops", "count"]
            n_ops_unique = stats["edits_ops"].nunique()
            if n_ops_unique > 15:
                chart = alt.Chart(stats).mark_bar(color="steelblue").encode(
                    alt.X("edits_ops:Q", bin=alt.Bin(maxbins=20), title="operations per instance"),
                    alt.Y("count():Q", title="instances"),
                )
            else:
                chart = alt.Chart(ops_dist).mark_bar(color="steelblue").encode(
                    alt.X("edits_ops:O", title="operations per instance"),
                    alt.Y("count:Q", title="instances"),
                )
            title = f"Complexity: operations per instance (all have {int(n_stages.iloc[0])} edit site)"
        chart.properties(title=title, width=350, height=220).save(out_structural / "complexity_by_n_stages.png")
        print("Saved structural/complexity_by_n_stages.png")

        # Mean complexity per n_stages bucket: ops per stage (informative) + modules if non-degenerate
        stage_stats["ops_per_stage"] = stage_stats["edits_ops"] / stage_stats["n_stages"].replace(0, 1)
        agg_by_stage = stage_stats.groupby("n_stages").agg(
            mean_ops_per_stage=("ops_per_stage", "mean"),
            mean_edits_ops=("edits_ops", "mean"),
            mean_modules=("modules_count", "mean"),
            n_instances=("edits_ops", "count"),
        ).reset_index()
        n_unique_stages = agg_by_stage["n_stages"].nunique()
        value_vars = ["mean_ops_per_stage"]
        if agg_by_stage["mean_modules"].sum() > 0:
            value_vars.append("mean_modules")
        agg_long = pd.melt(
            agg_by_stage,
            id_vars=["n_stages"],
            value_vars=value_vars,
            var_name="metric",
            value_name="mean_val",
        )
        agg_long["metric"] = agg_long["metric"].replace(
            {"mean_ops_per_stage": "ops per edit site", "mean_modules": "modules"}
        )
        if n_unique_stages >= 2:
            chart = alt.Chart(agg_long).mark_line(point=True).encode(
                alt.X("n_stages:O", title="number of AST stages (edit sites)"),
                alt.Y("mean_val:Q", title="mean value"),
                alt.Color("metric:N", title=""),
            )
        else:
            chart = alt.Chart(agg_long).mark_bar().encode(
                alt.X("metric:N", title=""),
                alt.Y("mean_val:Q", title="mean value"),
                alt.Color("metric:N", title=""),
            )
        chart.properties(
            title="Mean complexity by number of AST stages (ops per edit site)",
            width=350,
            height=220,
        ).save(out_structural / "complexity_by_stages.png")
        print("Saved structural/complexity_by_stages.png")

        # Saturation: op types per instance + top-k coverage
        def op_types_from_edits(edits):
            types = set()
            for cert in (edits or []):
                if not isinstance(cert, dict):
                    continue
                for op in cert.get("operations", []):
                    if isinstance(op, dict) and op.get("type"):
                        types.add(str(op["type"]))
            return types

        df["op_type_set"] = df["edits"].apply(op_types_from_edits)
        n_inst = len(df)
        n_with_ops = (df["op_type_set"].apply(len) > 0).sum()

        # Histogram: unique op types per instance
        op_counts = df["op_type_set"].apply(len)
        hist_df = op_counts.value_counts().sort_index().reset_index()
        hist_df.columns = ["n_op_types", "count"]
        alt.Chart(hist_df).mark_bar(color="steelblue").encode(
            alt.X("n_op_types:O", title="unique op types per instance"),
            alt.Y("count:Q", title="instances"),
        ).properties(
            title="Distribution of action-type diversity per instance",
            width=350,
            height=220,
        ).save(out_structural / "op_types_per_instance.png")
        print("Saved structural/op_types_per_instance.png")

        # Action-type coverage: how many action types cover N trajectories?
        type_to_indices = {}
        for idx, row in df.iterrows():
            for t in row["op_type_set"]:
                type_to_indices.setdefault(t, set()).add(idx)
        coverage_counts = [len(indices) for indices in type_to_indices.values()]
        cov_df = pd.Series(coverage_counts).value_counts().sort_index().reset_index()
        cov_df.columns = ["n_trajectories", "n_action_types"]
        alt.Chart(cov_df).mark_bar(color="steelblue").encode(
            alt.X("n_trajectories:O", title="trajectories covered"),
            alt.Y("n_action_types:Q", title="action types"),
        ).properties(
            title="Action types by trajectory coverage (overall dataset)",
            width=350,
            height=220,
        ).save(out_structural / "action_coverage.png")
        print("Saved structural/action_coverage.png")

        # Saturation curve: cumulative % of instances covered by top-k types
        sorted_types = sorted(
            type_to_indices, key=lambda t: len(type_to_indices[t]), reverse=True
        )

        coverage_data = []
        covered = set()
        for k, t in enumerate(sorted_types, 1):
            covered |= type_to_indices[t]
            pct = 100 * len(covered) / n_inst if n_inst else 0
            coverage_data.append({"k": k, "pct_covered": pct})

        if coverage_data:
            sat_df = pd.DataFrame(coverage_data)
            max_pct = sat_df["pct_covered"].max()
            k_90 = sat_df.loc[sat_df["pct_covered"] >= 90, "k"].min()
            k_95 = sat_df.loc[sat_df["pct_covered"] >= 95, "k"].min()
            k_90_s = f"{k_90:.0f}" if pd.notna(k_90) else "—"
            k_95_s = f"{k_95:.0f}" if pd.notna(k_95) else "—"
            subtitle = f"saturation {max_pct:.0f}% | k@90%: {k_90_s}, k@95%: {k_95_s}"
            alt.Chart(sat_df).mark_line(point=True, color="#0072B2").encode(
                alt.X("k:Q", title="top-k action types"),
                alt.Y("pct_covered:Q", title="% instances covered"),
            ).properties(
                title=alt.TitleParams(
                    text="Saturation: cumulative coverage by top-k action types",
                    subtitle=subtitle,
                ),
                width=350,
                height=220,
            ).save(out_structural / "action_type_saturation.png")
            print(f"Saved structural/action_type_saturation.png ({subtitle})")
        else:
            print(f"Skipped action_type_saturation.png (no op types: {n_with_ops}/{n_inst} instances have ops; run diff resolution for full-file data)")

    # 2. Diversity (needs distances + labels)
    # Prefer structural distances: token, set-diff (op count), tree, graph
    STRUCTURAL_REPRS = ["tokens", "edits_set_diff", "edits_tree", "modules_graph"]
    dist_path = ds / "distances.parquet" if (ds / "distances.parquet").exists() else data / "distances.parquet"
    lbl_path = ds / "labels.parquet" if (ds / "labels.parquet").exists() else data / "labels.parquet"
    if dist_path.exists() and lbl_path.exists():
        from analysis.diversity import run_diversity_analysis
        from analysis.io import load_labels, load_matrices

        all_matrices = load_matrices(dist_path)
        labels = load_labels(lbl_path)
        # Filter to structural reprs when available; else use all
        matrices = {k: all_matrices[k] for k in STRUCTURAL_REPRS if k in all_matrices}
        if len(matrices) < 2:
            matrices = all_matrices
        if len(matrices) >= 2 and len(labels) == list(matrices.values())[0].shape[0]:
            # Exclude degenerate (constant) matrices: all distances same → no signal
            import numpy as np
            non_degen = {}
            for k, D in matrices.items():
                idx = np.triu_indices(D.shape[0], k=1)
                vals = D[idx]
                if len(vals) > 0 and np.var(vals) > 0:
                    non_degen[k] = D
                else:
                    print(f"  Skipping {k} (degenerate: constant distances)")
            matrices = non_degen if non_degen else matrices
            if len(matrices) < 2:
                print("  Need ≥2 non-degenerate matrices for diversity plots")
            else:
                results = run_diversity_analysis(matrices, labels)
                names = list(matrices.keys())
                skipped = [k for k in STRUCTURAL_REPRS if k not in names]
                rho = results["rank_correlation"]
                heatmap_data = [
                    {"repr_i": _repr_label(names[i]), "repr_j": _repr_label(names[j]), "rho": float(rho[i, j])}
                    for i in range(len(names)) for j in range(len(names))
                ]
                subtitle = f"Excluded (degenerate): {', '.join(_repr_label(k) for k in skipped)}" if skipped else None
                chart = alt.Chart(pd.DataFrame(heatmap_data)).mark_rect().encode(
                    alt.X("repr_j:N", title=""),
                    alt.Y("repr_i:N", title=""),
                    alt.Color("rho:Q", scale=alt.Scale(scheme="redblue", domainMid=0), title="Spearman ρ"),
                ).properties(
                    title=alt.TitleParams(
                        text="Rank correlation: structural representations",
                        subtitle=subtitle,
                    ),
                    width=300,
                    height=280,
                )
                chart.save(out / "rank_correlation.png")
                print("Saved rank_correlation.png")

                sr_df = pd.DataFrame([
                    {"repr": _repr_label(k), "ratio": v} for k, v in results["stratum_ratios"].items()
                ])
                so_df = pd.DataFrame([
                    {"repr": _repr_label(k), "overlap": v} for k, v in results.get("stratum_overlaps", {}).items()
                ])
                sr_subtitle = f"Excluded: {', '.join(_repr_label(k) for k in skipped)}" if skipped else None
                ratio_chart = alt.Chart(sr_df).mark_bar(color="steelblue").encode(
                    alt.X("repr:N", sort="-y", title="representation"),
                    alt.Y("ratio:Q", title="mean within / mean across"),
                ).properties(
                    title=alt.TitleParams(
                        text="Stratum ratio (<1 = clusters by repo)",
                        subtitle=sr_subtitle,
                    ),
                    width=300,
                    height=220,
                )
                if len(so_df) > 0:
                    overlap_chart = alt.Chart(so_df).mark_bar(color="#0072B2").encode(
                        alt.X("repr:N", sort="-y", title="representation"),
                        alt.Y("overlap:Q", title="P(within < across)", scale=alt.Scale(domain=[0, 1])),
                    ).properties(
                        title=alt.TitleParams(
                            text="Stratum overlap (>0.5 = separable)",
                            subtitle=sr_subtitle,
                        ),
                        width=300,
                        height=220,
                    )
                    (ratio_chart | overlap_chart).save(out / "stratum_ratios.png")
                else:
                    ratio_chart.save(out / "stratum_ratios.png")
                print("Saved stratum_ratios.png")

                # Unique variance and stratum ratio only (no silhouette)
                var_df = pd.DataFrame([
                    {"repr": _repr_label(k), "score": v} for k, v in results["unique_variances"].items()
                ])
                var_chart = alt.Chart(var_df).mark_bar(color="#0072B2").encode(
                    alt.X("repr:N", sort="-y", title="representation"),
                    alt.Y("score:Q", title="unique variance"),
                ).properties(
                    title="Unique variance (info not in other reprs)",
                    width=300,
                    height=220,
                )
                sr_chart = alt.Chart(sr_df).mark_bar(color="steelblue").encode(
                    alt.X("repr:N", sort="-y", title="representation"),
                    alt.Y("ratio:Q", title="stratum ratio"),
                ).properties(
                    title="Stratum ratio (<1 = clusters)",
                    width=300,
                    height=220,
                )
                (var_chart | sr_chart).save(out / "diversity_scores.png")
                print("Saved diversity_scores.png")

                # Diversity by stage: unique_variance, stratum_ratio, stratum_overlap
                # Order representations by unique_variance descending (most informative first)
                uv = results.get("unique_variances", {})
                ordered = sorted(names, key=lambda r: uv.get(r, 0), reverse=True)
                ordered_labels = [_repr_label(r) for r in ordered]
                div_rows = []
                for r in ordered:
                    lbl = _repr_label(r)
                    if r in results["unique_variances"]:
                        v = results["unique_variances"][r]
                        if pd.notna(v):
                            div_rows.append({"repr": lbl, "metric": "unique_variance", "value": float(v)})
                    if r in results["stratum_ratios"]:
                        v = results["stratum_ratios"][r]
                        if pd.notna(v):
                            div_rows.append({"repr": lbl, "metric": "stratum_ratio", "value": float(v)})
                    if r in results.get("stratum_overlaps", {}):
                        v = results["stratum_overlaps"][r]
                        if pd.notna(v):
                            div_rows.append({"repr": lbl, "metric": "stratum_overlap", "value": float(v)})
                if div_rows:
                    div_df = pd.DataFrame(div_rows)
                    div_df["repr"] = pd.Categorical(div_df["repr"], categories=ordered_labels, ordered=True)
                    alt.Chart(div_df).mark_line(point=True).encode(
                        alt.X("repr:N", sort=ordered_labels, title="representation"),
                        alt.Y("value:Q", title="score"),
                        alt.Color("metric:N", title=""),
                    ).properties(
                        title="Diversity metrics: structural representations",
                        width=400,
                        height=250,
                    ).save(out / "diversity_by_stage.png")
                    print("Saved diversity_by_stage.png")

                # Leap summary: range (max - min) per metric across reprs
                leap_rows = []
                for metric, key in [
                    ("stratum_ratio", "stratum_ratios"),
                    ("stratum_overlap", "stratum_overlaps"),
                    ("unique_variance", "unique_variances"),
                ]:
                    vals = results.get(key, {})
                    valid = [v for v in vals.values() if pd.notna(v)]
                    if len(valid) >= 2:
                        leap_rows.append({
                            "metric": metric,
                            "min": min(valid),
                            "max": max(valid),
                            "leap": max(valid) - min(valid),
                        })
                if leap_rows:
                    leap_df = pd.DataFrame(leap_rows)
                    leap_chart = alt.Chart(leap_df).mark_bar(color="steelblue").encode(
                        alt.X("metric:N", title=""),
                        alt.Y("leap:Q", title="range (max − min)"),
                    ).properties(
                        title="Leap between representations (per metric)",
                        width=280,
                        height=180,
                    )
                    leap_chart.save(out / "diversity_leap.png")
                    print("Saved diversity_leap.png")

                # Full comparison table: all metrics × all reprs
                comp_rows = []
                for r in ordered:
                    row = {"repr": r}
                    for key in ["stratum_ratios", "stratum_overlaps", "unique_variances"]:
                        v = results.get(key, {}).get(r)
                        row[key] = float(v) if pd.notna(v) else None
                    comp_rows.append(row)
                if comp_rows:
                    comp_df = pd.DataFrame(comp_rows)
                    comp_path = ds / "diversity_comparison.parquet"
                    comp_df.to_parquet(comp_path, index=False)
                    print(f"Saved {comp_path.name}")

    # 3. Per-instance rho (value_counts on binned values, no altair binning)
    pi_path = ds / "per_instance_rep_correlation.parquet" if (ds / "per_instance_rep_correlation.parquet").exists() else data / "per_instance_rep_correlation.parquet"
    if pi_path.exists():
        pi_df = pd.read_parquet(pi_path)
        valid = pi_df.dropna(subset=["mean_rho"])
        if len(valid) > 0:
            # Bin to 0.1 steps for readable distribution
            binned = (valid["mean_rho"] * 10).round().astype(int) / 10
            vc = binned.value_counts().sort_index().reset_index()
            vc.columns = ["mean_rho", "count"]
            alt.Chart(vc).mark_bar(color="steelblue").encode(
                alt.X("mean_rho:O", title="mean ρ (representation agreement)"),
                alt.Y("count:Q", title="instances"),
            ).properties(
                title="Per-instance representation agreement",
                width=350,
                height=220,
            ).save(out / "per_instance_rho.png")
            print("Saved per_instance_rho.png")

    # 4. Retrieval agreement (dropna for pairs with NaN)
    pair_path = ds / "per_instance_pair_rho.parquet" if (ds / "per_instance_pair_rho.parquet").exists() else data / "per_instance_pair_rho.parquet"
    if pair_path.exists():
        pair_df = pd.read_parquet(pair_path)
        agg = pair_df.dropna(subset=["rho"]).groupby(["repr_i", "repr_j"])["rho"].mean().reset_index()
        if len(agg) > 0:
            agg["pair"] = agg["repr_i"].map(_repr_label) + " vs " + agg["repr_j"].map(_repr_label)
            alt.Chart(agg).mark_bar(color="#0072B2").encode(
                alt.X("pair:N", sort="-y", title="representation pair"),
                alt.Y("rho:Q", title="mean ρ"),
            ).properties(
                title="Retrieval agreement: structural representation pairs",
                width=350,
                height=220,
            ).save(out / "retrieval_agreement.png")
            print("Saved retrieval_agreement.png")

    # 5. Divergence from baseline (needs eval divergence_results.json)
    div_candidates = [
        ds / "eval" / "divergence_results.json",
        data / "eval_results.json",
        data / "divergence_results.json",
    ]
    for p in div_candidates:
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            per_proc = d.get("divergence_from_baseline", {}).get("per_procedure", {})
            if per_proc:
                df = pd.DataFrame([{"repr": k, "mean_dist": v} for k, v in per_proc.items()])
                alt.Chart(df).mark_bar(color="steelblue").encode(
                    alt.X("repr:N"), alt.Y("mean_dist:Q")
                ).properties(width=280, height=200).save(out / "divergence_from_baseline.png")
                print("Saved divergence_from_baseline.png")
            break

    # 6. Procedure divergence gap (needs run_procedure_divergence on eval records)
    proc_candidates = [
        ds / "eval" / "procedure_divergence.parquet",
        ds / "procedure_divergence.parquet",
        data / "procedure_divergence.parquet",
    ]
    proc_path = next((p for p in proc_candidates if p.exists()), None)
    if proc_path:
        df = pd.read_parquet(proc_path)
        df["pair"] = df["proc_a"] + " vs " + df["proc_b"]
        alt.Chart(df.dropna(subset=["gap"])).mark_boxplot().encode(
            alt.X("pair:N"), alt.Y("gap:Q")
        ).properties(width=350, height=220).save(out / "procedure_divergence_gap.png")
        print("Saved procedure_divergence_gap.png")

    # 7. Embedding ablation (structure vs code as basis of behavioral embedding)
    emb_path = ds / "embedding_ablation.json"
    if emb_path.exists():
        with open(emb_path) as f:
            emb = json.load(f)
        agg = emb.get("aggregate", {})
        per = emb.get("per_instance", [])
        if agg:
            agg_df = pd.DataFrame([
                {"condition": "sim(edits-only, both)", "similarity": agg.get("mean_sim_ac", 0)},
                {"condition": "sim(code-only, both)", "similarity": agg.get("mean_sim_bc", 0)},
                {"condition": "sim(edits-only, code-only)", "similarity": agg.get("mean_sim_ab", 0)},
            ])
            alt.Chart(agg_df).mark_bar(color="steelblue").encode(
                alt.X("condition:N", sort="-y", title=""),
                alt.Y("similarity:Q", title="mean cosine similarity", scale=alt.Scale(domain=[0, 1])),
            ).properties(
                title="Embedding ablation: structure vs code as basis",
                width=350,
                height=220,
            ).save(out / "embedding_ablation_aggregate.png")
            print("Saved embedding_ablation_aggregate.png")
        if per:
            per_df = pd.DataFrame(per)
            # Per-instance: sim_ac vs sim_bc scatter (structure vs code dominance)
            scatter = alt.Chart(per_df).mark_circle(size=60, opacity=0.7).encode(
                alt.X("sim_ac:Q", title="sim(edits-only, both)", scale=alt.Scale(domain=[0, 1])),
                alt.Y("sim_bc:Q", title="sim(code-only, both)", scale=alt.Scale(domain=[0, 1])),
                alt.Tooltip(["instance_id", "sim_ac", "sim_bc", "sim_ab"]),
            ).properties(
                title="Per-instance: structure vs code contribution to embedding",
                width=300,
                height=280,
            )
            # Diagonal reference: above = structure dominates, below = code dominates
            diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
            ref = alt.Chart(diag).mark_line(color="gray", strokeDash=[4, 2]).encode(
                alt.X("x:Q"), alt.Y("y:Q"),
            )
            (scatter + ref).save(out / "embedding_ablation_scatter.png")
            print("Saved embedding_ablation_scatter.png")
            # Box plot: sim_ac, sim_bc, sim_ab distributions
            long = per_df.melt(
                id_vars=["instance_id"],
                value_vars=["sim_ac", "sim_bc", "sim_ab"],
                var_name="pair",
                value_name="similarity",
            )
            long["pair"] = long["pair"].map({
                "sim_ac": "edits-only vs both",
                "sim_bc": "code-only vs both",
                "sim_ab": "edits-only vs code-only",
            })
            alt.Chart(long).mark_boxplot(size=25).encode(
                alt.X("pair:N", title=""),
                alt.Y("similarity:Q", title="cosine similarity", scale=alt.Scale(domain=[0, 1])),
            ).properties(
                title="Embedding ablation: similarity distributions",
                width=350,
                height=220,
            ).save(out / "embedding_ablation_distributions.png")
            print("Saved embedding_ablation_distributions.png")

    print(f"\nPlots in {out}")


if __name__ == "__main__":
    main()
