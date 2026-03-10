# Representations for Learning from Developer-Agent Workflows

Multi-level representation extractors for transforming raw developer workflow traces into privacy-preserving abstractions. Supports SWE-bench, agent trajectories (intermediate states), and Cursor exports.

---

## Quick Start

### Baseline Extraction (SWE-bench Lite)

```bash
# Extract: patch → trace → edits/modules/motifs certificates
python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --output-dir output

# Push to midah/procedural-info-theory (uses HF_TOKEN or huggingface_hub login)
python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --push
```

### SWE-bench / SWE-bench Lite (Legacy Script)

```bash
python scripts/run_swe_bench.py --dataset lite --split dev --limit 10 --output traces.jsonl
python scripts/run_swe_bench.py --dataset lite --split test --rung tokens --output tokens.jsonl
```

### Custom Traces

```bash
./scripts/extract_cursor_data.sh ./cursor_exports
python scripts/parse_to_traces.py --input ./cursor_exports --output traces.jsonl

python -c "
from representations import motifs_repr, tokens_repr, semantic_edits_repr
trace = {'events': [...]}
motifs = motifs_repr(trace)
tokens = tokens_repr(trace)
edits = semantic_edits_repr(trace)
"
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

### Inferred Representations (DSPy)

Behavioral, mechanistic, and functional annotations are LLM-derived and require grounding (edits certificate, module graph). Not included in the extraction pipeline by default.

---

## Structure

```
representations/           # Transformation (Python)
├── computed/             # edits, modules, motifs (structure from code/traces)
├── inferred/             # behavioral, mechanistic, functional (LLM-derived)
├── encoders/             # raw, tokens, functions
└── core/                 # intent, utils

data/
├── swe_bench.py          # SWE-bench / SWE-bench Lite
└── agent_trajectories.py # Agent trajectories (intermediate states)

configs/
└── datasets.py           # Dataset configs for extraction pipeline

pipeline/
└── utils.py              # Extraction, serialization, HF token

scripts/
├── run_extraction_pipeline.py  # Main pipeline: datasets → parquet + HF export
├── run_swe_bench.py            # Legacy SWE-bench pipeline
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

## Agent Trajectories

SWE-bench records only base_commit → fix_commit; the golden patch spans that gap with no checkpoints. Agent trajectories provide intermediate states (reasoning, commands, observations).

```python
from data.agent_trajectories import load_agent_trajectories, agent_trajectory_to_trace

for trace in load_agent_trajectories(dataset_id="nebius/SWE-agent-trajectories", split="train", limit=10):
    # trace has events: prompt, model_reasoning, terminal_command, observation
    tokens = tokens_repr(trace)
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
