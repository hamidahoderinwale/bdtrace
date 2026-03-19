#!/usr/bin/env python3
"""
Cross-benchmark saturation analysis.

Reads precomputed repr_ablation.json and transfer_metrics.json from each
benchmark's output directory, then produces a set of comparison plots:

  cross_saturation_by_repr.png  - saturation point per representation, per benchmark
  cross_saturation_bar.png      - grouped bar: saturation rank by benchmark x repr
  cross_auc_heatmap.png         - AUC distance vs pass: benchmark x repr
  cross_pass_rate.png           - overall pass rate per benchmark
  cross_region_pass_rates.png   - region-level pass rate distributions per benchmark
  cross_fix_types.png           - fix type distribution comparison (if fix_types.json present)

Usage:
  uv run python scripts/run_cross_benchmark_analysis.py
  uv run python scripts/run_cross_benchmark_analysis.py --benchmarks swe_bench_lite swe_bench_verified
  uv run python scripts/run_cross_benchmark_analysis.py --output-dir notebooks/plots/cross_benchmark
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import altair as alt
except ImportError:
    print("Install with: uv sync --extra notebooks")
    sys.exit(1)


## Benchmark display labels and ordering
BENCHMARK_LABELS = {
    "swe_bench_lite_resolved":             "SWE-bench Lite (300)",
    "swe_bench_verified_resolved_multifile": "SWE-bench Verified (71)",
    "swe_bench_verified_resolved_full":    "SWE-bench Verified (500)",
    "swe_smith_resolved":                  "SWE-smith",
}

REPR_LABELS = {
    "edits":          "raw edits",
    "edits_set_diff": "edit set-diff",
    "modules":        "dependency graph",
    "motifs":         "recurring patterns",
}

## Wong colorblind-safe palette, benchmark order: Lite, Smith, Verified, Full
BENCHMARK_COLORS = ["#0072B2", "#009E73", "#D55E00", "#E69F00"]

THEME_CONFIG = {
    "font": "sans-serif",
    "background": "#fafaf8",
    "axis_color": "#3a3a3a",
    "grid_color": "#e8e8e8",
    "title_size": 13,
    "label_size": 11,
}


def _theme():
    return {
        "config": {
            "background": THEME_CONFIG["background"],
            "font": THEME_CONFIG["font"],
            "axis": {
                "labelColor": THEME_CONFIG["axis_color"],
                "titleColor": THEME_CONFIG["axis_color"],
                "gridColor": THEME_CONFIG["grid_color"],
                "labelFontSize": THEME_CONFIG["label_size"],
                "titleFontSize": THEME_CONFIG["label_size"],
            },
            "title": {
                "color": THEME_CONFIG["axis_color"],
                "fontSize": THEME_CONFIG["title_size"],
                "fontWeight": "normal",
            },
            "legend": {
                "labelFontSize": THEME_CONFIG["label_size"],
                "titleFontSize": THEME_CONFIG["label_size"],
            },
            "view": {"stroke": "transparent"},
        }
    }


def _register_theme():
    try:
        @alt.theme.register("paper", enable=True)
        def _paper_theme() -> alt.theme.ThemeConfig:
            return alt.theme.ThemeConfig(**_theme())
    except AttributeError:
        # Altair < 5.5 fallback
        alt.themes.register("paper", _theme)
        alt.themes.enable("paper")


def _bench_label(k: str) -> str:
    return BENCHMARK_LABELS.get(k, k.replace("_", " "))


def _repr_label(k: str) -> str:
    return REPR_LABELS.get(k, k.replace("_", " "))


def _color_scale(keys: list[str]) -> alt.Scale:
    colors = BENCHMARK_COLORS[: len(keys)]
    return alt.Scale(domain=keys, range=colors)


def load_repr_ablation(ds_dir: Path) -> dict | None:
    """Load repr_ablation.json if present, else fall back to transfer_metrics.json."""
    path = ds_dir / "repr_ablation.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Single-repr fallback: transfer_metrics.json is top-level repr dict
    path = ds_dir / "transfer_metrics.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if "repr" in data:
            return {data["repr"]: data}
    return None


def load_fix_types(ds_dir: Path) -> dict | None:
    path = ds_dir / "fix_types.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_benchmark_data(
    benchmarks: list[str],
    data_root: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns:
        saturation_rows: [{benchmark, repr, saturation_rank, n_tasks, pct}]
        auc_rows:        [{benchmark, repr, auc}]
        pass_rate_rows:  [{benchmark, pass_rate, n_tasks}]
    """
    saturation_rows = []
    auc_rows = []
    pass_rate_rows = []

    for bench in benchmarks:
        ds_dir = data_root / "datasets" / bench
        if not ds_dir.exists():
            print(f"  Skip {bench}: no data at {ds_dir}")
            continue

        ablation = load_repr_ablation(ds_dir)
        if ablation is None:
            print(f"  Skip {bench}: no repr_ablation.json or transfer_metrics.json")
            continue

        label = _bench_label(bench)
        overall_pass_rate = None
        n_tasks = None

        for repr_key, metrics in ablation.items():
            knee = metrics.get("saturation_knee_rank")
            auc = metrics.get("auc_distance_vs_pass")
            rate = metrics.get("overall_pass_rate")
            n = (metrics.get("n_passed", 0) or 0) + (metrics.get("n_failed", 0) or 0)

            if n > 0:
                n_tasks = n
                overall_pass_rate = rate

            if knee is not None and n_tasks:
                saturation_rows.append({
                    "benchmark": label,
                    "benchmark_key": bench,
                    "repr": _repr_label(repr_key),
                    "repr_key": repr_key,
                    "saturation_rank": int(knee),
                    "n_tasks": n_tasks,
                    "saturation_pct": round(100 * knee / n_tasks, 1) if n_tasks else None,
                })
            if auc is not None:
                auc_rows.append({
                    "benchmark": label,
                    "benchmark_key": bench,
                    "repr": _repr_label(repr_key),
                    "repr_key": repr_key,
                    "auc": float(auc),
                })

        if overall_pass_rate is not None and n_tasks is not None:
            pass_rate_rows.append({
                "benchmark": label,
                "benchmark_key": bench,
                "pass_rate": float(overall_pass_rate),
                "n_tasks": int(n_tasks),
            })

    return saturation_rows, auc_rows, pass_rate_rows


def collect_region_pass_rates(benchmarks: list[str], data_root: Path) -> list[dict]:
    rows = []
    for bench in benchmarks:
        ds_dir = data_root / "datasets" / bench
        ablation = load_repr_ablation(ds_dir)
        if not ablation:
            continue
        label = _bench_label(bench)
        # Use first repr with region_pass_rates
        for metrics in ablation.values():
            rpr = metrics.get("region_pass_rates", {})
            if rpr:
                for region, rate in rpr.items():
                    rows.append({
                        "benchmark": label,
                        "benchmark_key": bench,
                        "region": region.split("/")[-1],  # short name
                        "pass_rate": float(rate),
                    })
                break
    return rows


def collect_diversity(benchmarks: list[str], data_root: Path) -> tuple[list[dict], list[dict]]:
    """Load diversity_results.json for each benchmark.

    Returns:
        unique_var_rows:  [{benchmark, repr, unique_variance}]
        rank_corr_rows:   [{benchmark, repr_a, repr_b, rho}]
    """
    uv_rows: list[dict] = []
    rc_rows: list[dict] = []

    for bench in benchmarks:
        path = data_root / "datasets" / bench / "diversity_results.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        label = _bench_label(bench)

        for repr_key, val in (d.get("unique_variances") or {}).items():
            if val is not None and not (isinstance(val, float) and val != val):
                uv_rows.append({
                    "benchmark": label,
                    "repr": _repr_label(repr_key),
                    "repr_key": repr_key,
                    "unique_variance": float(val),
                })

        names = d.get("rank_correlation_names", [])
        matrix = d.get("rank_correlation", [])
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                rho = matrix[i][j] if matrix else None
                if rho is not None:
                    rc_rows.append({
                        "benchmark": label,
                        "repr_a": _repr_label(a),
                        "repr_b": _repr_label(b),
                        "rho": float(rho),
                        "rho_str": f"{float(rho):.2f}",
                    })

    return uv_rows, rc_rows


def plot_diversity_unique_variance(uv_rows: list[dict], benchmarks_order: list[str], out_dir: Path) -> None:
    """Bar charts of unique variance per representation, one panel per benchmark."""
    if not uv_rows:
        return

    df = pd.DataFrame(uv_rows)
    bench_order = [_bench_label(b) for b in benchmarks_order if _bench_label(b) in df["benchmark"].values]
    repr_order = list(REPR_LABELS.values())
    color_scale = alt.Scale(range=["#0072B2", "#009E73", "#E69F00", "#D55E00"])

    panels = []
    for bench_label in bench_order:
        sub = df[df["benchmark"] == bench_label]
        panels.append(
            alt.Chart(sub)
            .mark_bar()
            .encode(
                alt.X("repr:N", sort=repr_order, title=None, axis=alt.Axis(labelAngle=-30)),
                alt.Y("unique_variance:Q", title="unique variance", scale=alt.Scale(domain=[0, 1])),
                alt.Color("repr:N", sort=repr_order, scale=color_scale, legend=None),
                alt.Tooltip(["repr:N", "unique_variance:Q"]),
            )
            .properties(width=160, height=160, title=bench_label)
        )

    alt.hconcat(
        *panels,
        title=alt.TitleParams(
            text="Representation independence by benchmark",
            subtitle="Unique variance: fraction of structural information not captured by other representations.",
            anchor="start",
        ),
    ).save(out_dir / "cross_diversity_unique_variance.png")
    print("Saved cross_diversity_unique_variance.png")


def plot_diversity_rank_correlation(rc_rows: list[dict], benchmarks_order: list[str], out_dir: Path) -> None:
    """Rank correlation heatmaps: one per benchmark, shown side by side."""
    if not rc_rows:
        return

    df = pd.DataFrame(rc_rows)
    bench_order = [_bench_label(b) for b in benchmarks_order if _bench_label(b) in df["benchmark"].values]

    charts = []
    for bench_label in bench_order:
        sub = df[df["benchmark"] == bench_label]
        hm = (
            alt.Chart(sub)
            .mark_rect()
            .encode(
                alt.X("repr_a:N", title=None, axis=alt.Axis(labelAngle=-30)),
                alt.Y("repr_b:N", title=None),
                alt.Color("rho:Q", title="ρ",
                          scale=alt.Scale(scheme="blues", domain=[0, 1], clamp=True)),
                alt.Tooltip(["repr_a:N", "repr_b:N", "rho:Q"]),
            )
        )
        tx = (
            alt.Chart(sub)
            .mark_text(fontSize=10)
            .encode(
                alt.X("repr_a:N"),
                alt.Y("repr_b:N"),
                alt.Text("rho_str:N"),
                color=alt.condition("datum.rho > 0.7", alt.value("white"), alt.value("#333")),
            )
        )
        charts.append(
            (hm + tx).properties(width=180, height=180, title=bench_label)
        )

    alt.hconcat(
        *charts,
        title=alt.TitleParams(
            text="Representation rank correlation by benchmark",
            subtitle="Spearman ρ between pairwise distance vectors. ρ = 1 means representations are redundant.",
            anchor="start",
        ),
    ).save(out_dir / "cross_diversity_rank_corr.png")
    print("Saved cross_diversity_rank_corr.png")


def collect_fix_types(benchmarks: list[str], data_root: Path) -> list[dict]:
    rows = []
    for bench in benchmarks:
        ds_dir = data_root / "datasets" / bench
        ft = load_fix_types(ds_dir)
        if not ft:
            continue
        label = _bench_label(bench)
        total = sum(ft.get("fix_type_distribution", {}).values()) or 1
        for fix_type, count in ft.get("fix_type_distribution", {}).items():
            rows.append({
                "benchmark": label,
                "benchmark_key": bench,
                "fix_type": fix_type,
                "count": int(count),
                "pct": round(100 * count / total, 1),
            })
    return rows


def plot_saturation_bar(sat_df: pd.DataFrame, benchmarks_order: list[str], out_dir: Path) -> None:
    """Grouped bar chart: saturation rank (absolute) by benchmark and representation."""
    if sat_df.empty:
        return

    repr_order = [_repr_label(r) for r in ["edits", "edits_set_diff", "modules", "motifs"]]
    bench_order = [_bench_label(b) for b in benchmarks_order if _bench_label(b) in sat_df["benchmark"].values]

    chart = (
        alt.Chart(sat_df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            alt.X(
                "repr:N",
                sort=repr_order,
                title=None,
                axis=alt.Axis(labelAngle=-30),
            ),
            alt.Y(
                "saturation_rank:Q",
                title="saturation point (task rank)",
                scale=alt.Scale(zero=True),
            ),
            alt.Color(
                "benchmark:N",
                sort=bench_order,
                scale=_color_scale(bench_order),
                title="benchmark",
            ),
            alt.Column(
                "benchmark:N",
                sort=bench_order,
                title=None,
                spacing=12,
            ),
            alt.Tooltip(["benchmark", "repr", "saturation_rank", "n_tasks", "saturation_pct"]),
        )
        .properties(width=140, height=200)
        .resolve_scale(y="shared")
    )

    chart.properties(
        title=alt.TitleParams(
            text="Structural saturation point by benchmark and representation",
            subtitle="Rank at which new tasks stop adding structurally new content",
            anchor="start",
        )
    ).save(out_dir / "cross_saturation_bar.png")
    print("Saved cross_saturation_bar.png")


def plot_saturation_pct(sat_df: pd.DataFrame, benchmarks_order: list[str], out_dir: Path) -> None:
    """Line chart: saturation point as % of benchmark size, by representation."""
    if sat_df.empty or sat_df["saturation_pct"].isna().all():
        return

    repr_order = [_repr_label(r) for r in ["edits", "edits_set_diff", "modules", "motifs"]]
    bench_order = [_bench_label(b) for b in benchmarks_order if _bench_label(b) in sat_df["benchmark"].values]

    chart = (
        alt.Chart(sat_df)
        .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2)
        .encode(
            alt.X("repr:N", sort=repr_order, title="representation", axis=alt.Axis(labelAngle=-30)),
            alt.Y(
                "saturation_pct:Q",
                title="saturation point (% of tasks)",
                scale=alt.Scale(zero=True, domain=[0, max(20, sat_df["saturation_pct"].max() + 2)]),
            ),
            alt.Color("benchmark:N", sort=bench_order, scale=_color_scale(bench_order), title="benchmark"),
            alt.Tooltip(["benchmark", "repr", "saturation_rank", "n_tasks", "saturation_pct"]),
        )
        .properties(width=380, height=240)
    )

    reference = (
        alt.Chart(pd.DataFrame({"y": [5]}))
        .mark_rule(strokeDash=[4, 3], color="#888", opacity=0.7)
        .encode(alt.Y("y:Q"))
    )

    (chart + reference).properties(
        title=alt.TitleParams(
            text="Saturation as share of benchmark size",
            subtitle="Dashed line: 5% threshold. Benchmarks saturate well before that.",
            anchor="start",
        )
    ).save(out_dir / "cross_saturation_pct.png")
    print("Saved cross_saturation_pct.png")


def plot_auc_heatmap(auc_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: AUC (distance vs pass) by benchmark and representation."""
    if auc_df.empty:
        return

    chart = (
        alt.Chart(auc_df)
        .mark_rect()
        .encode(
            alt.X("repr:N", title="representation", axis=alt.Axis(labelAngle=-30)),
            alt.Y("benchmark:N", title=None),
            alt.Color(
                "auc:Q",
                title="AUC",
                scale=alt.Scale(scheme="blues", domain=[0.3, 0.7], clamp=True),
            ),
            alt.Tooltip(["benchmark", "repr", "auc"]),
        )
        .properties(width=280, height=160, title=alt.TitleParams(
            text="AUC: average distance to agent-solved examples vs outcome",
            subtitle="0.5 = no signal. Low AUC means structural proximity does not predict solvability.",
            anchor="start",
        ))
    )

    text = (
        alt.Chart(auc_df)
        .mark_text(fontSize=10, fontWeight="bold")
        .encode(
            alt.X("repr:N"),
            alt.Y("benchmark:N"),
            alt.Text("auc:Q", format=".2f"),
        )
    )

    (chart + text).save(out_dir / "cross_auc_heatmap.png")
    print("Saved cross_auc_heatmap.png")


def plot_pass_rate(pr_df: pd.DataFrame, benchmarks_order: list[str], out_dir: Path) -> None:
    """Horizontal bar: overall pass rate per benchmark."""
    if pr_df.empty:
        return

    bench_order = [_bench_label(b) for b in benchmarks_order if _bench_label(b) in pr_df["benchmark"].values]

    chart = (
        alt.Chart(pr_df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            alt.Y("benchmark:N", sort=bench_order, title=None),
            alt.X(
                "pass_rate:Q",
                title="overall pass rate",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            alt.Color("benchmark:N", sort=bench_order, scale=_color_scale(bench_order), legend=None),
            alt.Tooltip(["benchmark", "pass_rate", "n_tasks"]),
        )
        .properties(width=340, height=100 + 30 * len(pr_df))
    )

    reference = (
        alt.Chart(pd.DataFrame({"x": [pr_df["pass_rate"].mean()]}))
        .mark_rule(strokeDash=[4, 3], color="#888", opacity=0.7)
        .encode(alt.X("x:Q"))
    )

    (chart + reference).properties(
        title=alt.TitleParams(
            text="Overall pass rate by benchmark",
            subtitle="Dashed line: mean across benchmarks",
            anchor="start",
        )
    ).save(out_dir / "cross_pass_rate.png")
    print("Saved cross_pass_rate.png")


def plot_region_pass_rates(region_rows: list[dict], benchmarks_order: list[str], out_dir: Path) -> None:
    """Box-plot-style distribution of per-region pass rates per benchmark."""
    if not region_rows:
        return

    df = pd.DataFrame(region_rows)
    bench_order = [_bench_label(b) for b in benchmarks_order if _bench_label(b) in df["benchmark"].values]

    chart = (
        alt.Chart(df)
        .mark_boxplot(size=22, extent=1.5)
        .encode(
            alt.X("benchmark:N", sort=bench_order, title=None, axis=alt.Axis(labelAngle=-20)),
            alt.Y(
                "pass_rate:Q",
                title="per-repo pass rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            alt.Color("benchmark:N", sort=bench_order, scale=_color_scale(bench_order), legend=None),
            alt.Tooltip(["benchmark", "region", "pass_rate"]),
        )
        .properties(
            width=280,
            height=240,
            title=alt.TitleParams(
                text="Per-repository pass rate distribution",
                subtitle="High spread indicates uneven coverage across repos",
                anchor="start",
            ),
        )
    )

    chart.save(out_dir / "cross_region_pass_rates.png")
    print("Saved cross_region_pass_rates.png")


def plot_fix_types(fix_rows: list[dict], benchmarks_order: list[str], out_dir: Path) -> None:
    """Stacked bar of fix type distribution per benchmark (% of tasks)."""
    if not fix_rows:
        return

    df = pd.DataFrame(fix_rows)
    bench_order = [_bench_label(b) for b in benchmarks_order if _bench_label(b) in df["benchmark"].values]

    # Top fix types by total count across benchmarks
    top_types = (
        df.groupby("fix_type")["count"].sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    df["fix_type_display"] = df["fix_type"].apply(lambda x: x if x in top_types else "other")
    agg = df.groupby(["benchmark", "fix_type_display"])["pct"].sum().reset_index()

    fix_order = top_types + (["other"] if "other" in agg["fix_type_display"].values else [])

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            alt.X("benchmark:N", sort=bench_order, title=None, axis=alt.Axis(labelAngle=-20)),
            alt.Y("pct:Q", title="% of tasks", scale=alt.Scale(domain=[0, 100])),
            alt.Color(
                "fix_type_display:N",
                sort=fix_order,
                scale=alt.Scale(range=["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#BBBBBB"]),
                title="fix type",
            ),
            alt.Tooltip(["benchmark", "fix_type_display", "pct"]),
            alt.Order("fix_type_display:N", sort="ascending"),
        )
        .properties(
            width=300,
            height=260,
            title=alt.TitleParams(
                text="Fix type distribution by benchmark",
                subtitle="Shows whether benchmarks test similar or different kinds of bugs",
                anchor="start",
            ),
        )
    )

    chart.save(out_dir / "cross_fix_types.png")
    print("Saved cross_fix_types.png")


def plot_saturation_summary_table(
    sat_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Compact summary table image: benchmark, n_tasks, pass_rate, saturation_rank range.
    Useful for paper figures.
    """
    if sat_df.empty or pr_df.empty:
        return

    summary = (
        sat_df.groupby(["benchmark", "n_tasks"])
        .agg(sat_min=("saturation_rank", "min"), sat_max=("saturation_rank", "max"))
        .reset_index()
    )
    summary = summary.merge(pr_df[["benchmark", "pass_rate"]], on="benchmark", how="left")
    summary["pass_rate_pct"] = (summary["pass_rate"] * 100).round(0).astype(int).astype(str) + "%"
    summary["saturation_range"] = summary["sat_min"].astype(str) + " - " + summary["sat_max"].astype(str)
    summary["pct_of_tasks"] = (100 * summary["sat_max"] / summary["n_tasks"]).round(1).astype(str) + "%"

    display_cols = ["benchmark", "n_tasks", "pass_rate_pct", "saturation_range", "pct_of_tasks"]
    display_names = {
        "benchmark": "benchmark",
        "n_tasks": "tasks",
        "pass_rate_pct": "pass rate",
        "saturation_range": "saturation (rank)",
        "pct_of_tasks": "saturation (% of tasks)",
    }
    summary_display = summary[display_cols].rename(columns=display_names)

    # Render as an Altair table (text marks)
    melted = summary_display.melt(id_vars=["benchmark"], var_name="metric", value_name="value")
    melted["value"] = melted["value"].astype(str)

    col_order = [v for k, v in display_names.items() if k != "benchmark"]
    bench_order = list(summary_display["benchmark"])

    table = (
        alt.Chart(melted)
        .mark_text(align="center", baseline="middle", fontSize=11)
        .encode(
            alt.X("metric:N", sort=col_order, title=None, axis=alt.Axis(orient="top", labelAngle=-20, labelFontWeight="bold")),
            alt.Y("benchmark:N", sort=bench_order, title=None),
            alt.Text("value:N"),
        )
        .properties(width=460, height=30 + 30 * len(bench_order), title="Cross-benchmark saturation summary")
    )

    table.save(out_dir / "cross_summary_table.png")
    print("Saved cross_summary_table.png")


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark structural saturation analysis")
    parser.add_argument(
        "--benchmarks", "-b", nargs="+",
        default=["swe_bench_lite_resolved", "swe_smith_resolved"],
        help="Dataset names in output/datasets/",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        default=Path("notebooks/plots/cross_benchmark"),
    )
    args = parser.parse_args()

    _register_theme()

    data_root = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmarks: {args.benchmarks}")
    sat_rows, auc_rows, pr_rows = collect_benchmark_data(args.benchmarks, data_root)

    if not sat_rows and not auc_rows:
        print("No data found. Run the pipeline first for each benchmark.")
        sys.exit(1)

    sat_df = pd.DataFrame(sat_rows)
    auc_df = pd.DataFrame(auc_rows)
    pr_df = pd.DataFrame(pr_rows)

    region_rows = collect_region_pass_rates(args.benchmarks, data_root)
    fix_rows = collect_fix_types(args.benchmarks, data_root)
    uv_rows, rc_rows = collect_diversity(args.benchmarks, data_root)

    bench_order = [b for b in args.benchmarks if _bench_label(b) in sat_df["benchmark"].values or _bench_label(b) in pr_df["benchmark"].values]

    print(f"\nGenerating plots in {out_dir}")
    plot_saturation_bar(sat_df, bench_order, out_dir)
    plot_saturation_pct(sat_df, bench_order, out_dir)
    plot_auc_heatmap(auc_df, out_dir)
    plot_pass_rate(pr_df, bench_order, out_dir)
    plot_region_pass_rates(region_rows, bench_order, out_dir)
    plot_fix_types(fix_rows, bench_order, out_dir)
    plot_saturation_summary_table(sat_df, pr_df, out_dir)
    plot_diversity_unique_variance(uv_rows, args.benchmarks, out_dir)
    plot_diversity_rank_correlation(rc_rows, args.benchmarks, out_dir)

    print(f"\nDone. {out_dir}")


if __name__ == "__main__":
    main()
