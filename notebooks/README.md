# Notebooks

Exploration notebooks from the first-generation pipeline (parquet certificates + DSPy).
The current analysis line lives in `analysis/preferences/` and `scripts/agent_trajectories_paper/`;
these notebooks are kept for provenance and for two results not yet promoted to `findings.md`.

## Contents

| Notebook | Purpose | Status |
|----------|---------|--------|
| `analysis.ipynb` | Load certificates, summary stats, distributions, stratum breakdown, diversity | Stale: reads `output/datasets/swe_bench_lite/test.parquet`, which must be regenerated (steps below) |
| `behavioral_analysis.ipynb` | Agent trajectories × reference-patch structure | Stale, but holds an unpromoted result: search count is the only behavioral measure that correlates with success (rho ≈ 0.16–0.20); mean edit ops 1,871 solved vs 3,222 unsolved |
| `inter_eval_analysis.ipynb` | Does the human patch's structure predict agent solvability? | Stale, and its motifs AUC = 1.0 is annotated in-notebook as suspected vocabulary leakage — quarantined, do not cite (see findings.md grounding audit) |

## Regenerating inputs (legacy pipeline)

```bash
uv sync --extra notebooks
python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --output-dir output
python scripts/build_distance_matrices.py --input output/datasets/swe_bench_lite/test.parquet --reprs tokens edits_set_diff edits_tree modules_graph
python scripts/run_diversity_analysis.py --matrices output/datasets/swe_bench_lite/distances.parquet --labels output/datasets/swe_bench_lite/labels.parquet
python scripts/run_multi_benchmark_plots.py
```

(The `run_plots.py` this file used to reference does not exist; `run_multi_benchmark_plots.py` is the plotting entry point.)

## Analysis file formats

| Artifact | Format | Schema |
|----------|--------|--------|
| Distances | `distances.parquet` | i, j, d_edits, d_modules, d_motifs |
| Labels | `labels.parquet` | index, instance_id, stratum |
| Diversity metrics | `diversity_metrics.parquet` | repr, stratum_ratio, silhouette, unique_variance |
| Rank correlation | `rank_correlation.parquet` | repr_i, repr_j, rho |
