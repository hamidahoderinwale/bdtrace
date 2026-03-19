# Embedding Ablation: Structure as Basis of Representation

**Purpose:** Test whether structural input (edits certificate) is part of the basis of the behavioral embedding space.

---

## Setup

Run behavioral with three conditions:

| Condition | before_fn | after_fn | structural_certificate |
|-----------|-----------|----------|------------------------|
| **(a) Edits only** | `"(empty)"` | `"(empty)"` | Full cert |
| **(b) Code only** | Full source | Full source | `{}` |
| **(c) Both** | Full source | Full source | Full cert |

Embed each claim. Compare: `sim(emb_a, emb_c)`, `sim(emb_b, emb_c)`, `sim(emb_a, emb_b)`.

---

## Interpretation

| Pattern | Meaning |
|---------|---------|
| `sim_ac` high | Structure is the basis; full ≈ edits-only |
| `sim_bc` high | Code semantics dominate; full ≈ code-only |
| `sim_ab` low | Edits and code produce different bases |
| `structure_dominates` (sim_ac > sim_bc) | Structural input contributes more to embedding |

---

## Usage

```bash
uv run --env-file .venv/.env python scripts/run_embedding_ablation.py --dataset swe_bench_verified_resolved_multifile --limit 5
```

With explicit paths:

```bash
python scripts/run_embedding_ablation.py --input output/resolved_traces_verified_multifile.jsonl --limit 10 --output output/datasets/swe_bench_verified_resolved_multifile/embedding_ablation.json
```

Requires OPENROUTER_API_KEY or OPENAI_API_KEY (DSPy for behavioral). Embeddings use sentence-transformers (all-MiniLM-L6-v2).

---

## Output

- `per_instance`: sim_ac, sim_bc, sim_ab, claim_a, claim_b, claim_c per instance
- `aggregate`: mean_sim_ac, mean_sim_bc, mean_sim_ab, structure_dominates
