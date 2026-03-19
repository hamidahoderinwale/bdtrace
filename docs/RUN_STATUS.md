# Run Status

## Prerequisites

```bash
uv sync --extra parquet --extra analysis --extra notebooks
```

Set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) in `.venv/.env` for eval and embedding ablation.

**Traces:** `resolved_traces_verified_multifile.jsonl` must exist. If resolution is slow, copy from `resolved_traces.jsonl` (SWE-bench Lite) as fallback:
```bash
cp output/resolved_traces.jsonl output/resolved_traces_verified_multifile.jsonl
```

## Output layout

All dataset-specific outputs live under `output/datasets/{dataset}/`:

| Path | Contents |
|------|----------|
| `output/datasets/swe_bench_verified_resolved_multifile/` | test.parquet, distances.parquet, labels.parquet, diversity_*.parquet |
| `output/datasets/swe_bench_verified_resolved_multifile/eval/` | records_with_behavioral.json, divergence_results.json, procedure_divergence.parquet |
| `output/datasets/swe_bench_verified_resolved_multifile/` | transfer_metrics.json, embedding_ablation.json |
| `notebooks/plots/swe_bench_verified_resolved_multifile/structural/` | Structural/edits rung: distributions, complexity, op types |
| `notebooks/plots/swe_bench_verified_resolved_multifile/` | Diversity, rank correlation, retrieval agreement, divergence_from_baseline, procedure_divergence_gap |

Older outputs (e.g. from swe_bench_lite) remain in their original locations.

## LLM config (OpenRouter)

Behavioral, mechanistic, and functional representations require an LLM. Set `OPENROUTER_API_KEY` in `.venv/.env` (preferred) or `OPENAI_API_KEY`.

```bash
uv run --env-file .venv/.env python eval/run_eval.py --input output/resolved_traces_verified_multifile.jsonl --dataset swe_bench_verified_resolved_multifile --limit 5
```

See `configs/dspy_config.py`. When both keys are set, OpenRouter is used.

## Commands

### Full pipeline (structural only)

```bash
uv sync --extra parquet --extra analysis --extra notebooks
uv run python scripts/run_full_pipeline.py --skip-resolution --skip-plots
```

### With eval and transfer

```bash
uv run --env-file .venv/.env python scripts/run_full_pipeline.py --skip-resolution --eval --transfer
```

### Individual steps (dataset-centric)

```bash
# Eval: behavioral + divergence
uv run --env-file .venv/.env python eval/run_eval.py --input output/resolved_traces_verified_multifile.jsonl --dataset swe_bench_verified_resolved_multifile --save-records output/datasets/swe_bench_verified_resolved_multifile/eval/records_with_behavioral.json

# Procedure divergence (after eval)
uv run python scripts/run_procedure_divergence.py --input output/datasets/swe_bench_verified_resolved_multifile/eval/records_with_behavioral.json --output output/datasets/swe_bench_verified_resolved_multifile/eval/procedure_divergence.parquet

# Transfer analysis (synthetic pass/fail for testing)
uv run python scripts/run_transfer_analysis.py --dataset swe_bench_verified_resolved_multifile --synthetic-pass-rate 0.6

# Embedding ablation
uv run --env-file .venv/.env python scripts/run_embedding_ablation.py --dataset swe_bench_verified_resolved_multifile --limit 5

# Plots
uv run python scripts/run_plots.py --output-dir notebooks/plots --data-dir output --dataset swe_bench_verified_resolved_multifile
```

## Science-of-evals (multi-benchmark)

Run benchmarks separately; plot after. Add one by one or in chunks.

```bash
# Per benchmark
uv run python scripts/run_benchmark_pipeline.py --benchmark swe_bench_verified --skip-resolution
uv run --env-file .venv/.env python scripts/run_benchmark_pipeline.py --benchmark swe_bench_verified --eval --transfer

# Multi-benchmark plots (distributional)
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified swe_bench_lite
```

**Plots:** `notebooks/plots/multi_benchmark/` — distance_distribution, saturation_curve, instances_per_region.
**Config:** `configs/benchmarks.yaml`

### Real pass/fail from SWE-bench experiments

The [SWE-bench/experiments](https://github.com/SWE-bench/experiments) repo has pre-populated `results/results.json` per model. Fetch and use:

```bash
# Clone experiments repo (one-time)
git clone --depth 1 https://github.com/SWE-bench/experiments.git /path/to/experiments

# List available models
uv run python scripts/fetch_swebench_results.py --experiments-dir /path/to/experiments --split verified

# Fetch one or more models
uv run python scripts/fetch_swebench_results.py --experiments-dir /path/to/experiments --split verified \
  --models 20240402_sweagent_gpt4 20240402_rag_gpt4 20240620_sweagent_claude3.5sonnet \
  --output-dir output/swebench_results

# Plots with real pass/fail (single model)
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified \
  --pass-fail output/swebench_results/verified_20240402_sweagent_gpt4.json

# Multi-model comparison (saturation curves by model)
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified \
  --pass-fail-dir output/swebench_results
```

**Note:** Instance overlap depends on your dataset. `swe_bench_verified_resolved_multifile` has 50 instances (from resolved traces); SWE-bench Verified has 500. Match count = intersection of instance_ids.

### Full 500 Verified (eval saturation study)

For full instance coverage, run the pipeline on all 500 SWE-bench Verified instances:

```bash
# 1. Diff resolution (clones repos, ~1–2 hours for 500)
uv run python scripts/run_full_pipeline.py --dataset swe_bench_verified_resolved_full --skip-plots

# 2. Fetch real pass/fail from experiments repo
git clone --depth 1 https://github.com/SWE-bench/experiments.git /path/to/experiments
uv run python scripts/fetch_swebench_results.py --experiments-dir /path/to/experiments --split verified \
  --models 20240402_sweagent_gpt4 20240620_sweagent_claude3.5sonnet --output-dir output/swebench_results

# 3. Re-run transfer with real pass/fail (skip resolution and extraction)
uv run python scripts/run_full_pipeline.py --dataset swe_bench_verified_resolved_full --skip-resolution --skip-plots --transfer \
  --pass-fail output/swebench_results/verified_20240402_sweagent_gpt4.json

# 4. Plots
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified_full \
  --pass-fail-dir output/swebench_results
```

Or use the benchmark pipeline:

```bash
uv run python scripts/run_benchmark_pipeline.py --benchmark swe_bench_verified_full
# Then fetch pass/fail and re-run transfer + plots as above
```
