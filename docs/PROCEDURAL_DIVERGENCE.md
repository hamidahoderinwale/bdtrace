# Procedural Divergence

**Purpose:** Compare how different annotation procedures (behavioral, mechanistic, functional) interpret the *same* structural evidence (edits, modules). The gap between structural agreement and semantic agreement measures how much inferred adds beyond what structural already encodes.

---

## Motivation

In a new age of continual learning, closing the loop between benchmark performance and deployment performance requires going beyond rubrics that superficially present features to optimize for. Given two agent instances on the same task, patching a null pointer exception in a Django request handler, one localizes to the correct function immediately and inserts a guard clause with a specific error type, the other searches three unrelated files before producing a top-level try/except that swallows the exception silently. Both pass the test suite because the tests only assume valid input. The problem is surfacing what to measure: the divergence happens at localization, propagates through structural characterization, and only becomes visible at the behavioral level when a test for `request.POST.get("user_id") == None` reveals that the second agent's fix masks the exception rather than handling it. Instead of ground truth that is vaguely specified, comparison becomes the standard: rubrics derived from the distribution of real behavior, with an unlimited upper bound that shifts as agent capabilities improve, allowing evaluation to take place over time.

---

## Model

A **procedure** is a sequence of stages. **Length** is one feature among others (localization, structural representation, annotation type).

**Procedure set:**

```
P_behavioral   = [localization, edits, behavioral]
P_mechanistic  = [localization, edits, mechanistic]
P_functional   = [localization, edits, functional]
```

All three share localization and structural characterization (edits). Only annotation varies. The divergence matrix measures how much the annotation choice matters, with everything else held constant.

---

## Two Comparison Dimensions

**Structural:** Do procedures agree on what changed? Same edits cert → high structural agreement by construction. Baseline.

**Semantic:** Do procedures agree on what the change means? Embedding distance between annotation outputs. Behavioral and mechanistic are grounded in edits but describe different axes; functional is grounded in the module graph. This is where procedures can diverge.

**The gap:** Structural agreement − semantic agreement. High gap = annotations add genuinely different information. Low gap = annotations largely restate the certificate.

---

## Procedural Summary Matrix S(P_a, P_b, stage)

Tracks where divergence originated. Separates:

- **Inherited divergence:** P_a and P_b already differed at an earlier stage; current stage inherits that.
- **Introduced divergence:** P_a and P_b agreed up to this stage but differ at the current output.

**S(P_a, P_b, stage)** = divergence indicator at that stage (0 = agree, 1 = diverge), plus provenance (inherited vs introduced).

Computed by comparing outputs stage-by-stage. If stage 1 outputs differ, stage 2 and 3 are inherited. If stage 1 agrees but stage 2 differs, stage 2 introduces divergence.

---

## Implementation

- `analysis/procedures/procedure_divergence.py`: stage-specific output comparison, divergence matrix by procedure pair and instance, procedural summary S(P_a, P_b, stage).
- Input: records with edits, behavioral, mechanistic, functional (from extraction + eval).
- Output: parquet or JSON with per-instance divergence flags, S matrix, and aggregate stats.
