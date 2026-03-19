# Science of Evals: Multi-Dataset Saturation Study

**Goal:** Measure eval saturation across multiple benchmarks to understand when evals stop adding information, how benchmarks relate structurally, and how to design evals that maximize coverage.

---

## Framing

**Primary object of study:** Evals — when they saturate, how to maximize coverage, how benchmarks relate.

**Procedural structure:** Operational definition of *in-distribution*. Structurally similar instances are in-distribution; models should work on them if they work on neighbors. Structure is the lens for measuring eval coverage, not the main hypothesis.

**Implication:** If an eval has many instances in the same structural region, they add diminishing returns. Saturation curves and coverage maps identify where to add instances and when to stop.

---

## Research Questions

1. **Saturation curves:** How fast does each benchmark saturate? (Adding instances from the same structural region yields diminishing returns.)
2. **Cross-benchmark transfer:** Does structural similarity within benchmark A predict transfer to benchmark B?
3. **Benchmark coverage:** Which structural regions are over/under-sampled across the eval ecosystem?
4. **Model comparison:** Do saturation curves differ by model? Steeper = better structural generalization.

---

## Methodology

### Core Pipeline (per benchmark)

1. **Structural extraction:** Patches → edits, modules, motifs (or equivalent for non-patch benchmarks).
2. **Distance matrices:** Pairwise distances in representation space.
3. **Pass/fail:** Per-instance resolved/unresolved from published results or custom harness.
4. **Saturation metrics:**
   - Distance-to-passed-centroid calibration (AUC, threshold).
   - kNN transfer accuracy (k ∈ {3, 5, 10}).
   - Saturation knee: point where cumulative pass rate flattens.
   - Region pass rate: per-stratum or per-cluster pass rate.

### Cross-Benchmark

5. **Unified representation space:** Embed all benchmarks in a shared space (e.g. edits op-type vocabulary, or learned embedding). Compute cross-benchmark distances.
6. **Transfer across benchmarks:** Train saturation predictor on A; evaluate on B. Does structural similarity in A predict pass in B?
7. **Coverage map:** Cluster all instances across benchmarks. Which clusters have instances from multiple benchmarks? Which are benchmark-specific?

---

## Dataset Selection

### Tier 1: Code repair (patch-based, structural alignment)

| Dataset | Instances | Structural signal | Pass/fail source |
|---------|-----------|-------------------|------------------|
| **SWE-bench Verified** | 500 | Patches, multi-file | swebench.com, sb-cli |
| **SWE-bench Lite** | 300 | Patches | swebench.com |
| **SWE-bench Full** | 2,294 | Patches | swebench.com |

**Why:** Same representation pipeline (edits, modules, motifs). Direct pass/fail from harness. Multi-file gives modules signal.

### Tier 2: Code generation (different task, same language)

| Dataset | Instances | Structural signal | Pass/fail source |
|---------|-----------|-------------------|------------------|
| **HumanEval** | 164 | Generated code, AST | pass@k |
| **MBPP** | 974 | Generated code | pass@k |
| **DS-1000** | 1,000 | Data science code | pass@k |

**Why:** Different task type (generate vs repair). Tests whether structural notions transfer. Requires adapting extraction (solution code → edits-like representation, or AST diff from stub to solution).

### Tier 3: Extended repair / harder

| Dataset | Instances | Notes |
|---------|-----------|-------|
| **SWE-bench Pro** | 1,865 | Enterprise, long-horizon |
| **SWE-bench Multilingual** | 300 | 9 languages; cross-lingual structural alignment |
| **NaturalCodeBench** | 402 | Real user queries, 6 domains |

**Why:** Stress-test saturation at scale. Multilingual tests representation generality.

---

## Recommended Order

1. **SWE-bench family first:** Verified (500), Lite (300). Same pipeline, published results. Establishes baseline saturation curves.
2. **Add SWE-bench Full:** Scale to 2,294. Does saturation knee shift? Coverage map.
3. **HumanEval / MBPP:** Adapt extraction for generation (solution AST, or diff from problem to solution). Cross-task transfer.
4. **SWE-bench Pro / Multilingual:** Scale and generality.

---

## Implementation (modular)

**Per benchmark (run separately, add one by one):**
```bash
uv run python scripts/run_benchmark_pipeline.py --benchmark swe_bench_verified
uv run --env-file .venv/.env python scripts/run_benchmark_pipeline.py --benchmark swe_bench_verified --eval --transfer
```

**Plots (run after, distributional):**
```bash
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified
uv run python scripts/run_multi_benchmark_plots.py --benchmarks swe_bench_verified swe_bench_lite
```

**Output layout:**
- Per benchmark: `output/datasets/{dataset}/` (distances, labels, transfer_metrics)
- Plots: `notebooks/plots/multi_benchmark/` (distance_distribution, saturation_curve, instances_per_region)

**Plots (simple, distributional):**
| Plot | What it shows |
|------|----------------|
| `distance_distribution.png` | Histogram of distance-to-passed-centroid (passed vs failed) |
| `saturation_curve.png` | Cumulative pass rate vs rank by distance (closest first) |
| `instances_per_region.png` | Bar chart of instance count per stratum |

**Config:** `configs/benchmarks.yaml` maps benchmark_id → dataset.

---

## Data Requirements

| Input | SWE-bench | HumanEval/MBPP |
|-------|-----------|----------------|
| Traces / instances | Resolved patches (before/after) | Problem + solution code |
| Structural extraction | edits, modules, motifs | AST of solution, or diff(problem, solution) |
| Pass/fail | resolved from harness | pass@k from execution |
| Published results | swebench.com, sb-cli | OpenCompass, EvalPlus |

---

## Success Criteria

- Saturation curves for ≥3 benchmarks.
- Cross-benchmark transfer accuracy (A→B) reported.
- Coverage map: structural clusters with benchmark membership.
- Actionable: "Benchmark X saturates at N instances; add instances from clusters C, D for coverage."
