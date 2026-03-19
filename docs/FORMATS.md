# Analysis File Formats

Parquet is the primary format for analysis artifacts: queryable, schema-preserving, portable.

## Distances

**File:** `distances.parquet`

| Column | Type | Description |
|--------|------|-------------|
| i | int | Instance index (row) |
| j | int | Instance index (col), j > i (upper triangle) |
| d_edits | float | Jaccard distance on edit operation types |
| d_edits_set_diff | float | Symmetric set diff on op types: \|A Δ B\|/(\|A\|+\|B\|), number of ops to transform |
| d_modules | float | Jaccard distance on module tokens |
| d_motifs | float | Cosine distance on motif soft membership |
| d_tokens | float | (structural) Levenshtein on token sequences |
| d_edits_tree | float | (structural) Tree edit distance on AST when available |
| d_modules_graph | float | (structural) Graph distance (symmetric edge diff) |

Use `--approach structural` with `--reprs tokens edits_set_diff edits_tree modules_graph`.

## Labels

**File:** `labels.parquet`

| Column | Type | Description |
|--------|------|-------------|
| index | int | Instance index (0..n-1) |
| instance_id | str | Instance identifier |
| stratum | str | Stratum label (e.g. repo) |

## Diversity Results

**File:** `diversity_metrics.parquet`

| Column | Type | Description |
|--------|------|-------------|
| repr | str | Representation name |
| stratum_ratio | float | Within/across stratum mean distance ratio |
| stratum_overlap | float | P(within < across); raw distances, >0.5 = separable |
| silhouette | float | Silhouette score (precomputed) |
| unique_variance | float | Residual variance after regressing on others |

**File:** `rank_correlation.parquet`

| Column | Type | Description |
|--------|------|-------------|
| repr_i | str | Representation name |
| repr_j | str | Representation name |
| rho | float | Spearman rank correlation |

**File:** `per_instance_rep_correlation.parquet`

| Column | Type | Description |
|--------|------|-------------|
| index | int | Instance index |
| instance_id | str | Instance identifier |
| stratum | str | Stratum label |
| mean_rho | float | Mean Spearman ρ across all repr pairs (how consistently instance is expressed) |
| min_rho | float | Min ρ across pairs (most variable repr pair for this instance) |

Low mean_rho = instance expressed differently across representations.

## Transfer Metrics

**File:** `transfer_metrics.json` (from `run_transfer_analysis.py`)

| Field | Type | Description |
|-------|------|--------------|
| auc_distance_vs_pass | float | AUC for distance-to-passed-centroid vs pass (higher = better separation) |
| saturation_knee_rank | int | Rank (by distance, closest first) where cumulative pass rate flattens |
| coverage_summary | str | Human-readable: "saturates at ~N instances (rank by distance)" |
| overall_pass_rate | float | n_passed / n |
| mean_distance_passed | float | Mean distance to centroid among passed instances |
| mean_distance_failed | float | Mean distance to centroid among failed instances |
| knn | dict | kNN transfer accuracy for k ∈ {3, 5, 10} |
| per_instance | list | Optional; instance_id, passed, distance_to_passed_centroid |

---

## Procedural Divergence

**File:** `procedure_divergence.parquet` (from `run_procedure_divergence.py`)

| Column | Type | Description |
|--------|------|-------------|
| instance_id | str | Instance identifier |
| proc_a | str | Procedure name (behavioral, mechanistic, functional) |
| proc_b | str | Procedure name |
| terminal_diverged | bool | Outputs differ at terminal stage |
| terminal_distance | float | Semantic (embedding) distance between annotation outputs |
| structural_agreement | float | 1 − cert distance; high by construction (shared edits) |
| semantic_agreement | float | 1 − embedding distance |
| gap | float | structural_agreement − semantic_agreement; high = annotations add different info |
| structural_diverged | bool | Certificates differ (inherited from structural stage) |
| annotation_introduced | bool | Divergence introduced at annotation stage (structural agreed) |

Full S(P_a, P_b, stage) in JSON output. See `docs/PROCEDURAL_DIVERGENCE.md`.

**File:** `per_instance_pair_rho.parquet`

| Column | Type | Description |
|--------|------|-------------|
| index | int | Instance index |
| instance_id | str | Instance identifier |
| repr_i | str | Representation name |
| repr_j | str | Representation name |
| rho | float | Spearman ρ between distance profiles for this instance |

## Legacy Formats

- `matrices.npz` — numpy arrays for backward compat
- `labels.json` — list of stratum strings
- `diversity_results.json` — full nested output
