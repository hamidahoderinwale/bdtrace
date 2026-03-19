#!/usr/bin/env bash
# Run transfer + plots with real pass/fail. Use with tmux for long runs.
#
# Usage:
#   tmux new -s swebench 'cd /path/to/bidirect-align-dev-traces && ./scripts/run_with_real_passfail.sh'
#   # Or: tmux attach -t swebench

set -e
cd "$(dirname "$0")/.."

EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/tmp/swebench-experiments}"
if [ ! -d "$EXPERIMENTS_DIR" ]; then
  echo "Cloning experiments repo..."
  git clone --depth 1 https://github.com/SWE-bench/experiments.git "$EXPERIMENTS_DIR"
fi

echo "=== Fetching real pass/fail ==="
uv run python scripts/fetch_swebench_results.py --experiments-dir "$EXPERIMENTS_DIR" --split verified \
  --models 20240402_sweagent_gpt4 20240620_sweagent_claude3.5sonnet \
  --output-dir output/swebench_results

echo ""
echo "=== Transfer analysis with real pass/fail ==="
uv run python scripts/run_transfer_analysis.py --dataset swe_bench_verified_resolved_multifile \
  --pass-fail output/swebench_results/verified_20240402_sweagent_gpt4.json --per-instance

echo ""
echo "=== Multi-benchmark plots ==="
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified \
  --pass-fail-dir output/swebench_results

echo ""
echo "Done. Plots in notebooks/plots/multi_benchmark/"
