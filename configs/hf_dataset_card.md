---
language:
  - en
license: mit
tags:
  - code
  - software-engineering
  - swe-bench
  - representations
  - procedural
---

# Procedural-Info-Theory

Baseline: computed representation certificates from SWE-bench Lite. Produced by [bidirect-align-dev-traces](https://github.com/hamidahoderinwale/bidirect-align-dev-traces).

## Scope

**In scope:** Extract inputs from SWE-bench Lite (issue text, base/fix from patch, event sequence), run three computed representations, output well-formed certificates.

**Not in scope:** Inferred representations, procedure-level analysis, divergence matrix, ablation, grounding checks. Those consume these outputs downstream.

## Dataset Structure

| Config | Source | Splits | Description |
|--------|--------|--------|-------------|
| swe_bench_lite | [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) | test | Golden patches → trace → certificates |

## Expected Outputs (Computed Representations)

**edits** — Structural certificate per code_change:
- operation type (guard_clause_added, call_added, etc.)
- AST node locations
- delta (scalar)

**modules** — Trace-based co-edit subgraph:
- co-edit edges (files changed together)
- neighborhood of touched files
- (Import edges require repo snapshot; not in baseline.)

**motifs** — Event subsequences:
- sequence (event types from base to fix)
- motifs (mined subsequences)
- soft_membership (vector over motif vocabulary)

## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| instance_id | string | Instance identifier |
| repo | string | Repository |
| base_commit | string | Base commit |
| edits | list[dict] | Per-file certificates: {operations, delta} |
| modules | list[string] | Co-edit graph tokens |
| motifs | dict | {sequence, motifs, soft_membership} |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("midah/procedural-info-theory", "swe_bench_lite", split="dev")
```

## Creation

```bash
python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --output-dir output
python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --push  # HF_TOKEN required
```

## License

MIT
