# Experiments: What Is Studied and Why

## Thesis

Represent developer fix workflows at multiple abstraction levels; measure how lossy each level is; keep LLM-derived interpretations grounded in structural evidence.

---

## Structural vs Inferred: What Each Is For

**Structural representations** (edits, modules, motifs, tokens) are computed from code and traces. No LLM.

**Inferred representations** (behavioral, mechanistic, functional) are LLM-generated summaries grounded in structural evidence. They answer: *what does the structural representation support?*

| Inferred | Grounding | Question it answers |
|----------|-----------|----------------------|
| **Behavioral** | Edits certificate | What changed from a caller’s perspective? (input–output, contract) |
| **Mechanistic** | Edits certificate | How does the change work internally? (implementation pattern) |
| **Functional** | Module graph | What is the function’s role in the system? (architect’s view) |

**Research angle:** How much do structural representations contribute to these analyses? If inferred annotations largely restate the certificate → low gap. If they add genuinely different information → high gap. Structural agreement − semantic agreement = the gap.

---

## Intended Studies

### 1. Representation Diversity (Implemented)

**What:** Rank correlation, unique variance, stratum ratio, silhouette across representations.

**Why:** Quantify redundancy (which representations overlap) and distinctness (which add information). Identify which representations separate fixes by repo best.

**Experiments:**
- `scripts/build_distance_matrices.py` — pairwise distances (jaccard, structural, or both)
- `scripts/run_diversity_analysis.py` — diversity metrics

**Outputs:** `rank_correlation.parquet`, `diversity_metrics.parquet`, `per_instance_rep_correlation.parquet`

---

### 2. Per-Instance Representation Variance (Implemented)

**What:** For each instance, mean Spearman ρ between its distance profile across representation pairs.

**Why:** Identify instances expressed differently across representations. Low mean_rho = neighbors in edits space differ from neighbors in modules space.

**Experiments:** Part of `run_diversity_analysis.py`

**Outputs:** `per_instance_rep_correlation.parquet`, `per_instance_pair_rho.parquet`

---

### 3. Divergence from Baseline (Implemented)

**What:** Cosine distance between inferred (behavioral, mechanistic, functional) embeddings and token baseline.

**Why:** Measure how much inferred representations diverge from the raw token view. High divergence = structured view adds semantic lift.

**Experiments:** `eval/run_eval.py --analysis divergence`

**Requires:** DSPy LM

---

### 4. Embedding Ablation: Structure as Basis (Implemented)

**What:** Run behavioral with (a) edits only, (b) code only, (c) both. Compare claim embeddings. If sim(emb_a, emb_c) > sim(emb_b, emb_c) → structure is the basis.

**Why:** Test whether structural input is encoded in the embedding space or if code semantics dominate.

**Experiments:** `scripts/run_embedding_ablation.py --dataset swe_bench_verified_resolved_multifile --limit 10`

**See:** `docs/EMBEDDING_ABLATION.md`

---

### 5. Grounding Verification (Stub)

**What:** Verify that cited certificate fields actually support the claim (e.g. "guard_clause_added" supports "returns early for invalid input").

**Why:** Ensure inferred claims are evidence-based, not hallucinated.

**Status:** `analysis/grounding/check_*.py` stubs exist; not implemented.

---

### 6. Inferred in Diversity Pipeline (Planned)

**What:** Add behavioral, mechanistic, functional to distance matrices; run full diversity analysis including inferred.

**Why:** Complete the 5-matrix (or 8-matrix) diversity view; see if inferred add unique variance.

**Status:** Requires DSPy; add to `build_distance_matrices` when inferred annotations available.

---

### 7. Eval Saturation / Structural Transfer (Implemented)

**What:** Procedural structure defines in-distribution. Distance-to-passed-centroid, kNN transfer, region pass rate. Saturation curves show when evals stop adding information.

**Why:** Evals are the object of study. Structure operationalizes similarity; models should work on in-distribution instances. Saturated regions = lower eval priority; unsaturated = higher. Informs benchmark expansion, few-shot selection.

**Experiments:** `scripts/run_transfer_analysis.py --distances ... --labels ... --pass-fail ...`

**Requires:** Pass/fail per instance (SWE-bench results or custom eval). Use `--synthetic-pass-rate 0.7` for pipeline testing.

**See:** `docs/EVAL_SATURATION_TRANSFER.md`

---

### 8. Procedural Divergence (Implemented)

**What:** Compare procedure outputs by stage and length. Divergence metric depends on procedure length:
- Length 2: structural (edits certificate) — same function, same structural change?
- Length 3: annotation (behavioral/mechanistic/functional) — embedding distance above threshold
- Length 4: reconstruction — fidelity difference (stub)

**Why:** Trace where divergence originated. Procedural summary S(P_a, P_b, stage) separates inherited vs introduced divergence.

**Experiments:** `eval/run_eval.py --save-records records.json` then `scripts/run_procedure_divergence.py --input records.json`

**See:** `docs/PROCEDURAL_DIVERGENCE.md`

---

## Experiment Flow

```
Extraction (edits, modules, motifs, tokens)
    → build_distance_matrices [--approach jaccard|structural|both]
    → run_diversity_analysis
    → diversity metrics, per-instance correlation

With DSPy configured:
    → eval/run_eval.py --from-swe-bench [--limit N] [--save-records records.json]
    → divergence_from_baseline
    → (optional) run_procedure_divergence.py --input records.json
```

---

## Configuration

| Env var | Purpose |
|---------|---------|
| OPENAI_API_KEY or OPENROUTER_API_KEY | DSPy LM (required for inferred) |
| DSPY_MODEL | Model string (default: openai/gpt-4o-mini). For OpenRouter: openrouter/openai/gpt-4o-mini |
| DSPY_TEMPERATURE | Sampling temp (default: 0.0) |
| DSPY_MAX_TOKENS | Max tokens (default: 1024) |
| HF_TOKEN | Push to midah/procedural-info-theory |
