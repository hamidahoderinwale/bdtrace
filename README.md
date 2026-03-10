# Representations for Learning from Developer-Agent Workflows

Multi-level representation extractors for transforming raw developer workflow traces into privacy-preserving abstractions.

---

## Quick Start

### SWE-bench / SWE-bench Lite

```bash
# Fetch from Hugging Face, convert to traces, run representations
python scripts/run_swe_bench.py --dataset lite --split dev --limit 10 --output traces.jsonl

# Single rung (e.g. tokens only)
python scripts/run_swe_bench.py --dataset lite --split test --rung tokens --output tokens.jsonl
```

### Custom traces

```bash
# 1. Extract raw data from Cursor databases (optional)
./scripts/extract_cursor_data.sh ./cursor_exports

# 2. Parse to traces
python scripts/parse_to_traces.py --input ./cursor_exports --output traces.jsonl

# 3. Transform traces
python -c "
from representations import motifs_repr, tokens_repr, semantic_edits_repr
trace = {'events': [...]}  # Your trace data
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

### Example Outputs

```json
// Raw
{"events": [{"type": "code_change", "file": "utils.ts", "diff": "+15, -8", "before": "...", "after": "..."}]}

// Tokens
["FUNCTION_DECL", "ASYNC", "PARAM:input", "RETURN_TYPE:Promise", "AWAIT", "CALL:process"]

// Edits
["EDIT(modify)→OP:async_wrapper", "EDIT(delete)→OP:remove_function"]

// Functions
["MODIFY processData params:(string)→(string,Config) return:void→Promise<string>"]

// Modules
["utils.ts→api.ts (imports, co-edited 5×)", "api.ts→config.ts (depends_on)"]

// Motifs
["PROMPT→EXPLORE→REFACTOR→ABSTRACT→TEST→COMMIT", "intent:'refactoring', freq:23"]
```

---

## Structure

```
representations/           # Transformation (Python)
├── encoders/
│   ├── raw.py, tokens.py, edits.py, functions.py, motifs.py
│   └── modules/           # Module graph (import + co-edit)
│       ├── import_extractor.py
│       └── graph.py
└── core/                 # Intent extraction

data/
└── swe_bench.py          # SWE-bench / SWE-bench Lite (HF fetch, patch parsing)

scripts/
├── run_swe_bench.py        # SWE-bench pipeline: fetch → trace → representations
├── extract_cursor_data.sh  # Extract from Cursor SQLite DBs
├── parse_to_traces.py      # Parse raw exports to trace format
└── convert_format.py       # Convert traces (JSONL ↔ Parquet)
```

---

## Python API

```python
from representations import (
    raw_repr, tokens_repr, semantic_edits_repr,
    functions_repr, module_graph_repr, file_edit_graph_repr, motifs_repr
)

# Trace-based (no repo on disk)
trace = {"events": [...]}
raw = raw_repr(trace)
tokens = tokens_repr(trace)
edits = semantic_edits_repr(trace)
functions = functions_repr(trace)
modules = file_edit_graph_repr(trace)  # co-edit from events
motifs = motifs_repr(trace)

# Repo-based (full import + co-edit graph)
graph = module_graph_repr(
    repo_path="/path/to/repo",
    commit="abc123",
    touched_files=["src/foo.py", "src/bar.py"],  # optional, restricts subgraph
)
# graph["nodes"], graph["import_edges"], graph["coedit_edges"], graph["graph"]
```

---

## SWE-bench

Load [SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) or [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) from Hugging Face. Patches are parsed into unified diff format; each instance becomes a trace with `problem_statement` as prompt and `patch`/`test_patch` as code_change events.

```python
from data.swe_bench import load_swe_bench_lite, swe_bench_instance_to_trace
from representations import tokens_repr, semantic_edits_repr

for trace in load_swe_bench_lite(split="dev", limit=10):
    tokens = tokens_repr(trace)
    edits = semantic_edits_repr(trace)
    # ...
```

## Data Extraction (Cursor)

Extract data directly from Cursor databases:

```bash
# Extract raw data
./scripts/extract_cursor_data.sh ./cursor_exports

# Parse to traces
python scripts/parse_to_traces.py --input ./cursor_exports --output traces.jsonl

# Convert formats
python scripts/convert_format.py --input traces.jsonl --output traces.parquet
```

---

## Graduated Disclosure
- **Public**: Motifs (~240× compression)
- **Team**: Module graphs (~100×)
- **Research**: Tokens (~10×)
- **Internal**: Raw events

---

## Key Algorithms

**Motif Mining**: PrefixSpan (frequent subsequences) + Sequitur (grammar compression)

---

## License

MIT
