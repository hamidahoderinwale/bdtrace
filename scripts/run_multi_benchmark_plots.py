#!/usr/bin/env python3
"""
Generate distributional plots for science-of-evals multi-benchmark study.

Reads from output/datasets/{dataset}/ for each benchmark. Run benchmarks separately;
plot after. Add benchmarks one by one or in chunks.

Plots (simple, distributional):
  - distance_distribution.png: Histogram of distance-to-passed-centroid (passed vs failed)
  - saturation_curve.png: Cumulative pass rate vs rank by distance (closest first)
  - instances_per_region.png: Bar chart of instance count per stratum
  - cross_benchmark_saturation.png: Overlay saturation curves when multiple benchmarks

Usage:
  uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified
  uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified swe_bench_lite
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

from analysis.io import load_labels, load_matrices
from analysis.transfer.saturation import distance_to_passed_centroid


def load_pass_fail(path: Path) -> dict[str, bool]:
    """Load instance_id -> pass (True) / fail (False) mapping."""
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: bool(v) for k, v in data.items()}
        if isinstance(data, list):
            return {r["instance_id"]: bool(r.get("resolved", r.get("pass", False))) for r in data}
        raise ValueError("JSON must be dict or list of records")

    if path.suffix == ".jsonl":
        result = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                result[r["instance_id"]] = bool(r.get("resolved", r.get("pass", False)))
        return result

    if path.suffix in (".parquet", ".csv"):
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        col = "resolved" if "resolved" in df.columns else "pass"
        if col not in df.columns:
            raise ValueError(f"Need 'resolved' or 'pass' column. Got: {list(df.columns)}")
        return dict(zip(df["instance_id"].astype(str), df[col].astype(bool)))

    raise ValueError(f"Unsupported format: {path.suffix}")


def load_benchmark_config() -> dict:
    import yaml

    path = Path(__file__).resolve().parent.parent / "configs" / "benchmarks.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("benchmarks", {})


def get_benchmark_data_dir(benchmark_id: str, configs: dict, data_root: Path) -> Path | None:
    """Resolve benchmark to output/datasets/{dataset}/ path."""
    if benchmark_id not in configs:
        return None
    dataset = configs[benchmark_id]["dataset"]
    path = data_root / "datasets" / dataset
    return path if path.exists() else None


def load_transfer_data(
    benchmark_dir: Path,
    data_root: Path,
    synthetic_pass_rate: float = 0.6,
    pass_fail_map: dict[str, bool] | None = None,
):
    """Load distances, labels, pass/fail. Compute per-instance distances if needed."""
    dist_path = benchmark_dir / "distances.parquet"
    lbl_path = benchmark_dir / "labels.parquet"
    transfer_path = benchmark_dir / "transfer_metrics.json"

    if not dist_path.exists() or not lbl_path.exists():
        return None

    matrices = load_matrices(dist_path)
    D = (
        matrices["edits_set_diff"]
        if "edits_set_diff" in matrices
        else (matrices["edits"] if "edits" in matrices else next(iter(matrices.values())))
    )

    df_lbl = pd.read_parquet(lbl_path)
    labels = np.array(df_lbl["stratum"].tolist()) if "stratum" in df_lbl.columns else np.array(df_lbl.iloc[:, -1].tolist())
    instance_ids = df_lbl["instance_id"].astype(str).tolist() if "instance_id" in df_lbl.columns else [str(i) for i in range(len(df_lbl))]

    n = D.shape[0]
    passed_mask = np.zeros(n, dtype=bool)

    use_external_pass_fail = pass_fail_map is not None
    matched = 0
    if use_external_pass_fail:
        for i, iid in enumerate(instance_ids):
            if iid in pass_fail_map:
                passed_mask[i] = pass_fail_map[iid]
                matched += 1
        if matched == 0:
            use_external_pass_fail = False
        else:
            print(f"  Matched {matched}/{n} instances with pass-fail")

    if not use_external_pass_fail and transfer_path.exists():
        with open(transfer_path) as f:
            tm = json.load(f)
        if "per_instance" in tm:
            for i, rec in enumerate(tm["per_instance"]):
                if i < n:
                    passed_mask[i] = rec.get("passed", False)
            dists = np.array([rec["distance_to_passed_centroid"] for rec in tm["per_instance"][:n]])
        else:
            rng = np.random.default_rng(42)
            passed_mask = rng.random(n) < synthetic_pass_rate
            dists = distance_to_passed_centroid(D, passed_mask)
    else:
        if not use_external_pass_fail:
            rng = np.random.default_rng(42)
            passed_mask = rng.random(n) < synthetic_pass_rate
        dists = distance_to_passed_centroid(D, passed_mask)

    return {
        "distances": dists,
        "passed": passed_mask,
        "labels": labels,
        "instance_ids": instance_ids,
        "n": n,
        "matched": matched if use_external_pass_fail else n,
    }


def plot_distance_distribution(data_list: list[tuple[str, dict]], out_dir: Path) -> None:
    """Distance-to-passed-centroid: overlaid histograms + box plot for passed vs failed."""
    rows = []
    for bench_id, d in data_list:
        for i in range(d["n"]):
            rows.append({
                "benchmark": bench_id,
                "distance": float(d["distances"][i]),
                "outcome": "passed" if d["passed"][i] else "failed",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Overlaid histograms (not stacked): passed vs failed
    hist = alt.Chart(df).mark_bar(opacity=0.6).encode(
        alt.X("distance:Q", bin=alt.Bin(maxbins=20), title="distance to passed centroid"),
        alt.Y("count():Q", title="instances"),
        alt.Color("outcome:N", scale=alt.Scale(domain=["failed","passed"], range=["#E69F00", "#0072B2"]), title=""),
    ).properties(width=350, height=220, title="Distance distribution (passed vs failed)")

    if df["benchmark"].nunique() > 1:
        hist = hist.properties(width=220, height=200).facet(column=alt.Column("benchmark:N", title=""))

    hist.save(out_dir / "distance_distribution.png")

    # Box plot: cleaner summary of separation (passed vs failed)
    box = alt.Chart(df).mark_boxplot(size=30).encode(
        alt.X("outcome:N", title=""),
        alt.Y("distance:Q", title="distance to passed centroid"),
        alt.Color("outcome:N", scale=alt.Scale(domain=["failed","passed"], range=["#E69F00", "#0072B2"]), legend=None),
    ).properties(width=280, height=200, title="Distance by outcome (box plot)")

    if df["benchmark"].nunique() > 1:
        box = box.properties(width=220, height=200).facet(column=alt.Column("benchmark:N", title=""))

    box.save(out_dir / "distance_by_outcome.png")
    print("Saved distance_distribution.png, distance_by_outcome.png")


def plot_saturation_curve(data_list: list[tuple[str, dict]], out_dir: Path) -> None:
    """Cumulative pass rate vs rank by distance (closest to passed centroid first)."""
    rows = []
    for bench_id, d in data_list:
        order = np.argsort(d["distances"])
        passed = d["passed"]
        for rank, idx in enumerate(order):
            cum_pass_rate = np.mean(passed[order[: rank + 1]])
            rows.append({
                "benchmark": bench_id,
                "rank": rank,
                "rank_pct": rank / max(1, d["n"] - 1),
                "cumulative_pass_rate": cum_pass_rate,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    chart = alt.Chart(df).mark_line().encode(
        alt.X("rank_pct:Q", title="rank by distance (0=closest)"),
        alt.Y("cumulative_pass_rate:Q", title="cumulative pass rate", scale=alt.Scale(domain=[0, 1])),
        alt.Color("benchmark:N", title=""),
    ).properties(width=350, height=220, title="Saturation curve")

    chart.save(out_dir / "saturation_curve.png")
    print("Saved saturation_curve.png")


def plot_instances_per_region(data_list: list[tuple[str, dict]], out_dir: Path) -> None:
    """Bar chart: instance count per stratum per benchmark."""
    rows = []
    for bench_id, d in data_list:
        for region in np.unique(d["labels"]):
            count = np.sum(d["labels"] == region)
            rows.append({"benchmark": bench_id, "region": str(region), "count": int(count)})
    df = pd.DataFrame(rows)
    if df.empty:
        return

    if df["benchmark"].nunique() > 1:
        chart = alt.Chart(df).mark_bar().encode(
            alt.X("region:N", title="stratum"),
            alt.Y("count:Q", title="instances"),
            alt.Color("benchmark:N", title=""),
        ).properties(width=220, height=200).facet(column=alt.Column("benchmark:N", title=""))
    else:
        chart = alt.Chart(df).mark_bar(color="steelblue").encode(
            alt.X("region:N", title="stratum"),
            alt.Y("count:Q", title="instances"),
        ).properties(width=350, height=220, title="Instances per region")

    chart.save(out_dir / "instances_per_region.png")
    print("Saved instances_per_region.png")


_MODEL_DISPLAY = {
    "20240402_sweagent_gpt4": "GPT-4 (SWE-agent)",
    "20240620_sweagent_claude3.5sonnet": "Claude 3.5 (SWE-agent)",
    "20240728_sweagent_gpt4o": "GPT-4o (SWE-agent)",
    "20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128": "Qwen2.5-72b (SWE-Fixer)",
    "20250306_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor": "Qwen2.5-72b (SWE-Fixer)",
}


def _model_display_name(short: str) -> str:
    return _MODEL_DISPLAY.get(short, short)


def main():
    parser = argparse.ArgumentParser(description="Multi-benchmark distributional plots")
    parser.add_argument("--benchmarks", "-b", nargs="+", required=True, help="Benchmark ids")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("notebooks/plots/multi_benchmark"))
    parser.add_argument("--data-dir", type=Path, default=Path("output"))
    parser.add_argument("--synthetic-pass-rate", type=float, default=0.6)
    parser.add_argument("--pass-fail", "-p", type=Path, help="Pass/fail per instance (JSON/JSONL/parquet)")
    parser.add_argument("--pass-fail-dir", type=Path, help="Dir of pass-fail files (model_name.json) for multi-model")
    args = parser.parse_args()

    configs = load_benchmark_config()
    data_root = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pass_fail_maps: list[tuple[str, dict[str, bool]]] = []
    if args.pass_fail:
        pass_fail_maps = [("default", load_pass_fail(args.pass_fail))]
    elif args.pass_fail_dir and args.pass_fail_dir.exists():
        for p in sorted(args.pass_fail_dir.glob("*.json")):
            model_name = p.stem
            pass_fail_maps.append((model_name, load_pass_fail(p)))

    if pass_fail_maps:
        print(f"Using pass-fail from {'single file' if len(pass_fail_maps) == 1 else f'{len(pass_fail_maps)} models'}")

    data_list = []
    for bid in args.benchmarks:
        bench_dir = get_benchmark_data_dir(bid, configs, data_root)
        if bench_dir is None:
            bench_dir = data_root / "datasets" / bid
        if not bench_dir.exists():
            print(f"Skip {bid}: no data at {bench_dir}")
            continue

        if pass_fail_maps:
            for model_label, pf_map in pass_fail_maps:
                d = load_transfer_data(bench_dir, data_root, args.synthetic_pass_rate, pass_fail_map=pf_map)
                if d is None:
                    continue
                # Skip models where fewer than 50% of benchmark instances matched
                if d.get("matched", 0) < 0.5 * d["n"]:
                    continue
                # model_label may include split prefix (e.g. verified_20240402_rag_gpt4)
                short = model_label.split("_", 1)[-1] if "_" in model_label else model_label
                display = _model_display_name(short)
                # When multiple benchmarks, append benchmark label to distinguish
                bench_label = configs.get(bid, {}).get("description", bid) if len(args.benchmarks) > 1 else ""
                bench_short = bench_label.split(",")[0].strip() if bench_label else ""
                full_label = f"{display} [{bench_short}]" if bench_short else display
                data_list.append((full_label, d))
        else:
            d = load_transfer_data(bench_dir, data_root, args.synthetic_pass_rate)
            if d is not None:
                data_list.append((bid, d))

    if not data_list:
        print("No benchmark data loaded.")
        sys.exit(1)

    print(f"Plotting {len(data_list)} benchmarks: {[b for b, _ in data_list]}")

    plot_distance_distribution(data_list, out_dir)
    plot_saturation_curve(data_list, out_dir)
    plot_instances_per_region(data_list, out_dir)

    print(f"\nPlots in {out_dir}")


if __name__ == "__main__":
    main()
