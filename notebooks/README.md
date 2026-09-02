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
uv run bdtrace notebooks   # extract -> distances -> diversity -> plots, default swe_bench_lite
uv run bdtrace lab         # JupyterLab in this directory
```

`bdtrace notebooks` chains the four stage scripts with the defaults these notebooks
expect; the stages are also reachable individually (`bdtrace certs extract|distances|diversity`,
`-h` shows each underlying script's options). `bdtrace --help` lists the full command set.

## Analysis file formats

| Artifact | Format | Schema |
|----------|--------|--------|
| Distances | `distances.parquet` | i, j, d_edits, d_modules, d_motifs |
| Labels | `labels.parquet` | index, instance_id, stratum |
| Diversity metrics | `diversity_metrics.parquet` | repr, stratum_ratio, silhouette, unique_variance |
| Rank correlation | `rank_correlation.parquet` | repr_i, repr_j, rho |
