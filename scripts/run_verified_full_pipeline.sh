#!/usr/bin/env bash
# Full 500 SWE-bench Verified pipeline for eval saturation study.
#
# Step 1: Diff resolution + extraction + matrices (~1-2 hours for 500)
# Step 2: Fetch pass/fail from experiments repo
# Step 3: Transfer with real pass/fail
# Step 4: Plots
#
# Usage:
#   ./scripts/run_verified_full_pipeline.sh
#   EXPERIMENTS_DIR=/path/to/experiments ./scripts/run_verified_full_pipeline.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Diff resolution + extraction + matrices (500 instances) ==="
uv run python scripts/run_full_pipeline.py \
  --dataset swe_bench_verified_resolved_full \
  --skip-plots

echo ""
echo "=== Step 2: Fetch pass/fail from experiments repo ==="
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-}"
if [ -z "$EXPERIMENTS_DIR" ] || [ ! -d "$EXPERIMENTS_DIR" ]; then
  echo "Clone experiments repo first:"
  echo "  git clone --depth 1 https://github.com/SWE-bench/experiments.git /path/to/experiments"
  echo "  EXPERIMENTS_DIR=/path/to/experiments ./scripts/run_verified_full_pipeline.sh"
  exit 1
fi

uv run python scripts/fetch_swebench_results.py \
  --experiments-dir "$EXPERIMENTS_DIR" \
  --split verified \
  --models 20240402_sweagent_gpt4 20240620_sweagent_claude3.5sonnet \
  --output-dir output/swebench_results

echo ""
echo "=== Step 3: Transfer with real pass/fail ==="
uv run python scripts/run_full_pipeline.py \
  --dataset swe_bench_verified_resolved_full \
  --skip-resolution --skip-plots --transfer \
  --pass-fail output/swebench_results/verified_20240402_sweagent_gpt4.json

echo ""
echo "=== Step 4: Plots ==="
uv run python scripts/run_multi_benchmark_plots.py \
  --benchmarks swe_bench_verified_full \
  --pass-fail-dir output/swebench_results

echo ""
echo "Done. Plots in notebooks/plots/multi_benchmark/"
