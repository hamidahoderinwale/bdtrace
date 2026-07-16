#!/usr/bin/env python3
"""
Run full pipeline for one benchmark. Outputs to output/datasets/{dataset}/.

Add benchmarks one by one. Plots can be run after with run_multi_benchmark_plots.py.

Usage:
  uv run python scripts/run_benchmark_pipeline.py --benchmark swe_bench_verified
  uv run --env-file .venv/.env python scripts/run_benchmark_pipeline.py --benchmark swe_bench_verified --eval --transfer
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_benchmark_config() -> dict:
    import yaml

    path = ROOT / "configs" / "benchmarks.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("benchmarks", {})


def main():
    parser = argparse.ArgumentParser(description="Run pipeline for one benchmark")
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark id from configs/benchmarks.yaml")
    parser.add_argument("--skip-resolution", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run behavioral eval (needs API key)")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--pass-fail", type=Path, default=None, help="Pass/fail JSON for transfer (from fetch_swebench_results)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configs = load_benchmark_config()
    if args.benchmark not in configs:
        print(f"Unknown benchmark: {args.benchmark}. Available: {list(configs)}")
        sys.exit(1)

    cfg = configs[args.benchmark]
    dataset = cfg["dataset"]

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_full_pipeline.py"),
        "--dataset", dataset,
    ]
    if args.skip_resolution:
        cmd.append("--skip-resolution")
    if args.skip_plots:
        cmd.append("--skip-plots")
    if args.eval:
        cmd.append("--eval")
    if args.transfer:
        cmd.append("--transfer")
    if args.pass_fail:
        cmd.extend(["--pass-fail", str(args.pass_fail)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    env = None
    if args.eval:
        env_file = ROOT / ".venv" / ".env"
        if env_file.exists():
            print("Use: uv run --env-file .venv/.env python scripts/run_benchmark_pipeline.py ... for eval")

    print(f">>> Running pipeline for {args.benchmark} (dataset={dataset})")
    r = subprocess.run(cmd, cwd=ROOT)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
