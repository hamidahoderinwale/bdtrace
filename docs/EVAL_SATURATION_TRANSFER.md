# Eval Saturation and Structural Transfer

**Purpose:** Use procedural structure to measure eval saturation and coverage. Structure defines *in-distribution*: similar instances are redundant for eval design; models should work on in-distribution instances.

---

## Thesis

**Evals are the object of study.** Procedural structure (edits, modules) operationalizes in-distribution: instances in the same structural region are similar; adding more from that region yields diminishing returns. Instances far from evaluated regions are unsaturated — higher evaluation priority. Models should pass in-distribution instances if they pass neighbors.

---

## Transfer Formulations

### 1. Distance-to-Passed-Centroid

For agent M with passed set P and failed set F:

- Centroid: mean embedding (or mean distance profile) of passed instances in representation space.
- For new instance x: `d(x) = distance(x, centroid(P))`.
- **Prediction:** P(pass | x) decreases with d(x). Calibrate threshold τ: if d(x) < τ → saturated (likely pass); if d(x) ≥ τ → unsaturated.

**Limitation:** Assumes passed region is convex. Multi-modal pass regions need clustering.

### 2. k-Nearest-Neighbor Transfer

- For new instance x: k nearest neighbors among all evaluated instances (or among passed only).
- **Prediction:** P(pass | x) ≈ (number of passed among kNN) / k.
- **Saturation:** If kNN pass rate > threshold (e.g. 0.8) → saturated.

**Advantage:** No convexity assumption. Uses local structure.

### 3. Structural Region Pass Rate

- Cluster instances by structural representation (e.g. KMeans on distance matrix rows, or stratum).
- Per cluster: pass rate = |passed ∩ cluster| / |cluster|.
- **Prediction:** P(pass | x) = pass_rate(cluster(x)).
- **Saturation:** High pass-rate clusters are saturated; low pass-rate or empty clusters are unsaturated.

---

## Data Requirements

| Input | Source | Status |
|-------|--------|--------|
| Distance matrices | `build_distance_matrices.py` | Done |
| Labels (instance_id, stratum) | `labels.parquet` | Done |
| Pass/fail per instance per model | SWE-bench results or custom eval | **Needed** |

**Pass/fail options:**

1. **SWE-bench published results:** Fetch per-instance resolved/unresolved from [swebench.com](https://swebench.com) or sb-cli. Merge by `instance_id`.
2. **Custom eval:** Run harness on model predictions; output `{instance_id: bool}`.
3. **Proxy (no agent):** Use gold patch as "reference" — study structural coverage of benchmark. Which regions have human solutions? Informs transfer potential, not agent transfer.

---

## Study Design

### Phase 1: Single-Model Transfer

**Input:** Distance matrix D (edits or modules), pass/fail labels for model M.

**Experiments:**

1. **Calibration:** Fit P(pass) vs. distance-to-passed-centroid. Report AUC, calibration curve.
2. **kNN transfer accuracy:** For k ∈ {3, 5, 10}, predict pass from kNN vote. Report accuracy, precision, recall.
3. **Saturation curve:** Sort instances by distance to passed-centroid; plot cumulative pass rate. Identify knee (saturation point).

**Output:** `transfer_metrics.json`: AUC, best_k, threshold, saturation_knee.

### Phase 2: Cross-Representation Transfer

**Question:** Does transfer in edits space predict transfer in modules space?

- Train predictor: edits-distance → pass. Evaluate on instances where modules-distance disagrees (low per-instance ρ).
- If edits transfer fails but modules transfer holds → representation matters for transfer.

### Phase 3: Cross-Stratum Transfer

**Question:** Does transfer within repo generalize across repos?

- Per-stratum (repo) pass rate. Correlation of pass rates across strata.
- Leave-one-repo-out: train on strata A, B, C; predict stratum D. Measures cross-repo transfer.

### Phase 4: Transfer Difficulty

**Question:** Are low-ρ instances (different neighbors in edits vs modules) harder to transfer?

- Stratify by per_instance_rho (low / medium / high).
- Compare transfer accuracy per stratum. Hypothesis: low ρ → harder to predict from single representation.

---

## Implementation Outline

```
analysis/
  transfer/
    __init__.py
    saturation.py      # distance_to_centroid, knn_transfer, region_pass_rate
    calibration.py    # fit P(pass|d), AUC, threshold

scripts/
  run_transfer_analysis.py   # --distances, --labels, --pass-fail, --output

configs/
  # pass_fail: path to JSON/parquet {instance_id: bool} or CSV with model column
```

**Script usage:**

```bash
# Dataset-centric (infers paths)
uv run python scripts/run_transfer_analysis.py --dataset swe_bench_verified_resolved_multifile --synthetic-pass-rate 0.6

# With pass/fail from custom eval
uv run python scripts/run_transfer_analysis.py --dataset swe_bench_verified_resolved_multifile --pass-fail output/model_x_results.json

# Explicit paths
python scripts/run_transfer_analysis.py \
  --distances output/datasets/swe_bench_verified_resolved_multifile/distances.parquet \
  --labels output/datasets/swe_bench_verified_resolved_multifile/labels.parquet \
  --pass-fail output/model_x_results.json \
  --output output/datasets/swe_bench_verified_resolved_multifile/transfer_metrics.json
```

---

## Extensions (Transfer Learning)

1. **Few-shot selection:** Pick k instances for few-shot prompt. Choose instances from unsaturated regions to maximize coverage.
2. **Curriculum:** Order eval instances by distance-to-passed-centroid (far first) for efficient capability discovery.
3. **Benchmark expansion:** Identify structural holes (regions with no instances); add tasks there.
4. **Model comparison:** Compare transfer curves across models. Steeper curve = better structural generalization.
