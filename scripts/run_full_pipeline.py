#!/usr/bin/env python3
"""
Run full pipeline over multi-file SWE-bench instances.

Resolves patches (with full module graph from repo), extracts representations,
builds distance matrices, runs diversity analysis, and generates plots.

Uses SWE-bench_Verified by default (71 multi-file instances). SWE-bench_Lite has 0.

Usage:
  python scripts/run_full_pipeline.py --limit 30
  python scripts/run_full_pipeline.py --limit 71 --output-dir output
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    """Run command; exit on failure."""
    print(f"\n>>> {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: diff_resolution → extraction → matrices → diversity → plots")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--limit", type=int, default=None, help="Limit instances (applies to resolution)")
    parser.add_argument("--skip-resolution", action="store_true", help="Skip diff_resolution (use existing jsonl)")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--eval", action="store_true", help="Run behavioral eval (requires OPENROUTER_API_KEY)")
    parser.add_argument("--transfer", action="store_true", help="Run transfer analysis")
    parser.add_argument("--pass-fail", type=Path, default=None, help="Pass/fail JSON for transfer (from fetch_swebench_results)")
    parser.add_argument(
        "--dataset",
        default="swe_bench_verified_resolved_multifile",
        choices=[
            "swe_bench_verified_resolved_multifile",
            "swe_bench_verified_resolved_full",
            "swe_bench_lite_resolved_multifile",
            "swe_bench_lite_resolved",
            "swe_smith_resolved",
            "swe_smith_stratified",
        ],
        help="Dataset config. swe_bench_verified_resolved_full = 500 instances for eval saturation.",
    )
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repos = out / "repos"
    if "verified_full" in args.dataset or args.dataset == "swe_bench_verified_resolved_full":
        jsonl_path = "resolved_traces_verified_full.jsonl"
    elif "verified" in args.dataset:
        jsonl_path = "resolved_traces_verified_multifile.jsonl"
    elif args.dataset == "swe_bench_lite_resolved":
        jsonl_path = "resolved_traces_lite_full.jsonl"
    elif args.dataset == "swe_smith_resolved":
        jsonl_path = "resolved_traces_swe_smith.jsonl"
    elif args.dataset == "swe_smith_stratified":
        jsonl_path = "resolved_traces_swe_smith_stratified.jsonl"
    elif "multifile" in args.dataset:
        jsonl_path = "resolved_traces_multifile.jsonl"
    else:
        jsonl_path = "resolved_traces.jsonl"
    jsonl = out / jsonl_path
    ds_dir = out / "datasets" / args.dataset
    # Smith uses "train" split; everything else uses "test"
    parquet = ds_dir / ("train.parquet" if "smith" in args.dataset else "test.parquet")

    if not args.skip_resolution:
        if args.dataset in ("swe_smith_resolved", "swe_smith_stratified"):
            hf_dataset = "SWE-bench/SWE-smith"
        elif "verified" in args.dataset:
            hf_dataset = "princeton-nlp/SWE-bench_Verified"
        else:
            hf_dataset = "princeton-nlp/SWE-bench_Lite"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_diff_resolution.py"),
            "--output", str(jsonl),
            "--repos-cache", str(repos),
            "--dataset", hf_dataset,
        ]
        # Full 500 Verified or full Lite: no multi-file filter. Multifile: only 2+ file patches.
        if args.dataset not in ("swe_bench_verified_resolved_full", "swe_bench_lite_resolved"):
            cmd.append("--multi-file-only")
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        run(cmd)

    run([
        sys.executable,
        str(ROOT / "scripts" / "run_extraction_pipeline.py"),
        "--datasets", args.dataset,
        "--output-dir", str(out),
        "--no-hf-export",
    ])

    if not parquet.exists():
        print(f"Parquet not found: {parquet}")
        sys.exit(1)

    run([
        sys.executable,
        str(ROOT / "scripts" / "build_distance_matrices.py"),
        "--input", str(parquet),
        "--output", str(ds_dir),
        "--approach", "jaccard",  # structural is slow for large ASTs
    ])

    run([
        sys.executable,
        str(ROOT / "scripts" / "run_diversity_analysis.py"),
        "--matrices", str(ds_dir / "distances.parquet"),
        "--labels", str(ds_dir / "labels.parquet"),
    ])

    if not args.skip_plots:
        run([
            sys.executable,
            str(ROOT / "scripts" / "run_plots.py"),
            "--output-dir", str(ROOT / "notebooks" / "plots"),
            "--data-dir", str(out),
            "--dataset", args.dataset,
        ])

    if args.eval:
        eval_dir = ds_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable,
            str(ROOT / "eval" / "run_eval.py"),
            "--input", str(jsonl),
            "--dataset", args.dataset,
            "--save-records", str(eval_dir / "records_with_behavioral.json"),
            "--output", str(eval_dir / "divergence_results.json"),
        ])
        run([
            sys.executable,
            str(ROOT / "scripts" / "run_procedure_divergence.py"),
            "--input", str(eval_dir / "records_with_behavioral.json"),
            "--output", str(eval_dir / "procedure_divergence.parquet"),
        ])

    if args.transfer:
        transfer_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_transfer_analysis.py"),
            "--dataset", args.dataset,
            "--output-dir", str(out),
            "--per-instance",
        ]
        if args.pass_fail and args.pass_fail.exists():
            transfer_cmd.extend(["--pass-fail", str(args.pass_fail)])
        else:
            transfer_cmd.extend(["--synthetic-pass-rate", "0.6"])
        run(transfer_cmd)

    print("\nDone. Outputs:")
    print(f"  Traces: {jsonl}")
    print(f"  Parquet: {parquet}")
    print(f"  Distances: {ds_dir / 'distances.parquet'}")
    print(f"  Plots: {ROOT / 'notebooks' / 'plots' / args.dataset}")


if __name__ == "__main__":
    main()
