# Notebooks

Analysis and visualization for procedural info theory.

## Setup

```bash
uv sync --extra notebooks
```

## Run

1. Extract data first: `python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --output-dir output`
2. Build matrices: `python scripts/build_distance_matrices.py --input output/datasets/swe_bench_lite/test.parquet --reprs tokens edits_set_diff edits_tree modules_graph`
3. Diversity: `python scripts/run_diversity_analysis.py --matrices output/datasets/swe_bench_lite/distances.parquet --labels output/datasets/swe_bench_lite/labels.parquet`
4. Plots: `python scripts/run_plots.py --output-dir notebooks/plots --data-dir output`
5. Or open `analysis.ipynb` and run all cells.

## Contents

| Notebook | Purpose |
|----------|---------|
| `analysis.ipynb` | Load certificates, summary stats, distributions, stratum breakdown, diversity analysis (when matrices available) |

## Outputs

| Plot | Source | Requires |
|------|--------|----------|
| `distributions.png` | analysis.ipynb | extraction parquet |
| `stratum_counts.png` | analysis.ipynb | extraction parquet |
| `rank_correlation.png` | analysis.ipynb | distances.parquet |
| `stratum_ratios.png` | analysis.ipynb | distances.parquet |
| `diversity_scores.png` | analysis.ipynb | distances.parquet |
| `per_instance_rho.png` | analysis.ipynb | per_instance_rep_correlation.parquet |
| `divergence_from_baseline.png` | analysis.ipynb | eval_results.json (run_eval --output) |
| `procedure_divergence_gap.png` | analysis.ipynb | procedure_divergence.parquet |
| `retrieval_agreement.png` | analysis.ipynb | per_instance_pair_rho.parquet |
| `embedding_ablation_aggregate.png` | run_plots.py | embedding_ablation.json |
| `embedding_ablation_scatter.png` | run_plots.py | embedding_ablation.json |
| `embedding_ablation_distributions.png` | run_plots.py | embedding_ablation.json |
| `op_types_per_instance.png` | run_plots.py | extraction parquet (edits) |
| `action_coverage.png` | run_plots.py | extraction parquet (edits) — action types by trajectory coverage |
| `action_type_saturation.png` | run_plots.py | extraction parquet (edits with operations) |
| `complexity_by_n_stages.png` | run_plots.py | extraction parquet (edits) |
| `complexity_by_stages.png` | run_plots.py | extraction parquet (edits) |
| `diversity_by_stage.png` | run_plots.py | distances.parquet, labels.parquet |

## Analysis file formats (Parquet primary)

| Artifact | Format | Schema |
|----------|--------|--------|
| Distances | `distances.parquet` | i, j, d_edits, d_modules, d_motifs |
| Labels | `labels.parquet` | index, instance_id, stratum |
| Diversity metrics | `diversity_metrics.parquet` | repr, stratum_ratio, silhouette, unique_variance |
| Rank correlation | `rank_correlation.parquet` | repr_i, repr_j, rho |
