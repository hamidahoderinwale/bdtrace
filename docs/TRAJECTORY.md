# Project Trajectory

## Current State Assessment

**Working**
- Extraction pipeline: SWE-bench Lite → edits, modules, motifs certificates
- Modules: populated from patch file paths (co-edit graph)
- Motifs: diff-based path wired (op types from certificate)
- Diversity analysis: rank correlation, stratum ratios, silhouette, unique variance
- HF export: parquet + JSON, push to midah/procedural-info-theory

**Multi-file run:** `run_full_pipeline.py` with `swe_bench_verified_resolved_multifile` produces `output/resolved_traces_verified_multifile.jsonl`. Diff resolution succeeds; extraction requires `pydriller` (pip install pydriller).

**Blocked / Empty**
- Edits: `operations: []` — patch hunks fail `ast.parse()`; need full-file resolution
- Motifs: empty because they derive from edits certificates
- Inferred (behavioral, mechanistic, functional): require DSPy LM; not in baseline
- Divergence matrix: stub only

---

## Trajectory

### Phase 1: Fix Edits (Full-File Resolution)

**Goal:** Non-empty edits and motifs from SWE-bench Lite.

| Task | Status | Notes |
|------|--------|-------|
| `data/diff_resolution.py` | Done | Clone, checkout base, apply patch, read full before/after |
| `scripts/run_diff_resolution.py` | Done | Outputs JSONL of resolved traces |
| `data/loaders.py` | Done | `load_from_jsonl` for resolved traces |
| Config `swe_bench_lite_resolved` | Done | Pipeline can use resolved JSONL |
| Verify edits + motifs populated | Pending | Run resolution → extraction on full dataset |

**Output:** Records with `edits[].operations` and `motifs.sequence` populated.

---

### Phase 2: Distance Matrices

**Goal:** Pairwise distance matrices for each representation over the same instance set.

| Task | Status | Notes |
|------|--------|-------|
| `analysis/procedures/divergence_matrix.py` | Done | `build_distance_matrices(records)` |
| Edits distance | Done | Jaccard on op type sets |
| Modules distance | Done | Jaccard on token sets |
| Motifs distance | Done | Cosine on soft_membership |
| `scripts/build_distance_matrices.py` | Done | Load parquet/json, save matrices.npz, labels.json |
| Stratum labels | Done | Derived from `repo` |

---

### Phase 3: Divergence Matrix + Diversity

**Goal:** Understand how representations relate and which separate strata best.

| Task | Status | Notes |
|------|--------|-------|
| Diversity analysis | Done | `analysis/diversity.py`, `scripts/run_diversity_analysis.py` |
| Rank correlation matrix | Done | 5×5 Spearman between flattened distance vectors |
| Within/across stratum ratios | Done | Per representation |
| Silhouette scores | Done | Per representation |
| Unique variance | Done | Per representation |
| Divergence matrix script | Pending | Orchestrate: build matrices → run diversity → report |

**Output:** JSON with `rank_correlation`, `stratum_ratios`, `silhouette_scores`, `unique_variances`.

---

### Phase 4: Inferred Representations (Optional)

**Goal:** Behavioral, mechanistic, functional for full 5-matrix diversity.

| Task | Status | Notes |
|------|--------|-------|
| DSPy config | Done | `configs/dspy_config.py` — OPENAI_API_KEY, DSPY_MODEL env |
| Run eval | Exists | `eval/run_eval.py` — divergence from baseline |
| Add inferred to distance build | Pending | Requires DSPy; add D_behavioral, D_mechanistic when available |

---

### Phase 5: Ablation & Grounding Checks

**Goal:** Procedure-level analysis, grounding verification.

| Task | Status | Notes |
|------|--------|-------|
| `analysis/procedures/ablation.py` | Stub | Ablate representation components |
| `analysis/grounding/check_*.py` | Stubs | Verify inferred fields cite certificate |
| `analysis/procedures/eval_procedure.py` | Stub | Procedure-level evaluation |

---

## Dependency Graph

```
[SWE-bench Lite]
       │
       ▼
[Patch → Trace] ──(optional)──► [Full-File Resolution]
       │
       ▼
[Extraction: edits, modules, motifs]
       │
       ├──────────────────────────────────────┐
       │                                      │
       ▼                                      ▼
[Build Distance Matrices]              [Inferred (DSPy)]
       │                                      │
       ▼                                      │
[Stratum Labels] ◄────────────────────────────┘
       │
       ▼
[Diversity Analysis]
  • Rank correlation matrix
  • Stratum ratios
  • Silhouette scores
  • Unique variance
       │
       ▼
[Report / Visualize]
```

---

## Suggested Order

1. **Phase 1** — Fix edits so downstream has real data.
2. **Phase 2** — Build distance matrices from records.
3. **Phase 3** — Run diversity analysis (already implemented).
4. **Phase 4** — Add inferred when DSPy is configured.
5. **Phase 5** — Ablation and grounding as needed.
