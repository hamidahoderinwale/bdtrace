# Representations for Learning from Developer-Agent Workflows

Multi-level representation extractors for transforming raw developer workflow traces into privacy-preserving abstractions. Uses SWE-bench Lite.

---

## Quick Start

### Baseline Extraction (SWE-bench Lite)

```bash
# Extract: patch → trace → edits/modules/motifs/tokens certificates
python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --output-dir output

# Build distance matrices (token, set-diff, tree, graph)
python scripts/build_distance_matrices.py --input output/datasets/swe_bench_lite/test.parquet --reprs tokens edits_set_diff edits_tree modules_graph

# Diversity analysis (includes per-instance representation correlation)
python scripts/run_diversity_analysis.py --matrices output/datasets/swe_bench_lite/distances.parquet --labels output/datasets/swe_bench_lite/labels.parquet

# Push to midah/procedural-info-theory (uses HF_TOKEN or huggingface_hub login)
python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --push
```

### Legacy Script

```bash
python scripts/run_swe_bench.py --split dev --limit 10 --output traces.jsonl
python scripts/run_swe_bench.py --split test --rung tokens --output tokens.jsonl
```

---

## The 6-Level Representation System

Each level trades privacy for expressiveness:

| **Level** | **Compression** | **Description** | **Use Case** |
|-----------|-----------------|-----------------|--------------|
| **Raw** | 1× | Complete event logs | Ground truth |
| **Tokens** | 10× | Token-type sequences | Research datasets |
| **Edits** | 11× | Edit operations (ADD/MODIFY/REMOVE) | Workflow analysis |
| **Functions** | 39× | Function-level changes & signatures | API tracking |
| **Modules** | 100× | Import + co-edit dependencies across files | Team collaboration |
| **Motifs** | 240× | Abstract workflow patterns | Public sharing |

### Distance Approaches

| Approach | Edits | Modules | Tokens |
|----------|-------|---------|--------|
| **jaccard** (default) | Jaccard on op types | Jaccard on tokens | — |
| **structural** | Tree edit (AST) when available | Graph distance (edge diff) | Levenshtein |

Use `--approach structural` or `--approach both` when building distance matrices.

### Per-Instance Representation Variance

Diversity analysis outputs `per_instance_rep_correlation.parquet`: for each instance, mean Spearman ρ between its distance profile across representation pairs. Low mean_rho = instance expressed differently across representations (e.g. different neighbors in edits vs modules vs motifs).

### Inferred Representations (DSPy)

Behavioral, mechanistic, and functional are LLM-generated summaries grounded in structural evidence. They answer: *what do edits/modules support?* Behavioral = caller view (what changed); mechanistic = internal how; functional = system role (needs module graph). Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` and run `eval/run_eval.py`. See `docs/EXPERIMENTS.md` for the structural-vs-inferred framing.

---

## Structure

```
representations/           # Transformation (Python)
├── computed/             # edits, modules, motifs (structure from code/traces)
├── inferred/             # behavioral, mechanistic, functional (LLM-derived)
├── encoders/             # raw, tokens, functions
└── core/                 # intent, utils

data/
├── swe_bench.py          # SWE-bench Lite loader
└── agent_trajectories.py # (optional) Agent trajectories

configs/
└── datasets.py           # Dataset configs for extraction pipeline

pipeline/
└── utils.py              # Extraction, serialization, HF token

scripts/
├── run_extraction_pipeline.py  # Main pipeline: datasets → parquet + HF export
├── run_swe_bench.py            # Legacy SWE-bench Lite pipeline
├── run_agent_trajectories.py  # Agent trajectories only
├── extract_cursor_data.sh
├── parse_to_traces.py
└── convert_format.py
```

---

## Procedural-Info-Theory Dataset

Outputs are published to [midah/procedural-info-theory](https://huggingface.co/datasets/midah/procedural-info-theory) on Hugging Face.

**Baseline scope:** SWE-bench Lite only. Three computed representations: edits (structural certificate), modules (co-edit subgraph), motifs (event subsequences + soft membership). No inferred representations; enrich later.

---

## Python API

```python
from representations import (
    raw_repr, tokens_repr, semantic_edits_repr,
    functions_repr, file_edit_graph_repr, motifs_repr
)

trace = {"events": [...]}
raw = raw_repr(trace)
tokens = tokens_repr(trace)
edits = semantic_edits_repr(trace)
functions = functions_repr(trace)
modules = file_edit_graph_repr(trace)
motifs = motifs_repr(trace)
```

---

## Graduated Disclosure

- **Public**: Motifs (~240× compression)
- **Team**: Module graphs (~100×)
- **Research**: Tokens (~10×)
- **Internal**: Raw events

---

## License

MIT
