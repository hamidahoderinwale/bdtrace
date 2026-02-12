# Rung Extractors

This package provides consistent rung extraction functions for all abstraction levels.

## Structure

- **core/**: Core utilities and shared logic
    - `canonicalization.py`: Event canonicalization and event sequence building
    - `intent.py`: Intent extraction and canonicalization from prompts
    - `intent_hierarchical.py`: Multi-level intent modeling
    - `utils.py`: Helper functions (AST parsing, tokenization, PII redaction)
- **encoders/**: Individual implementation for each representation level
    - `raw.py`, `tokens.py`, `edits.py`, `functions.py`, `modules.py`, `motifs.py`
- **motif_mining.py**: Statistical sequence pattern mining (using [PrefixSpan](https://github.com/chuanconggao/PrefixSpan-py) and [Sequitur](https://github.com/shobrook/sequitur))

## Defaults

All functions have defaults set to "everything on" for completeness:
- `include_prompts=True` (include prompts in representations)
- `use_statistical_mining=True` (use PrefixSpan/Sequitur)
- `include_metadata=True` (include metadata in raw representation)
- `redact_pii_enabled=True` (redact PII by default)

## Usage

```python
from rung_extractors import (
    tokens_repr,
    motifs_repr,
    raw_repr,
    semantic_edits_repr,
    functions_repr,
    module_graph_repr,
)

# All functions use defaults (everything on)
tokens = tokens_repr(trace)  # include_prompts=True by default
motifs = motifs_repr(trace)  # use_statistical_mining=True, include_prompts=True by default
raw = raw_repr(trace)  # include_metadata=True, redact_pii_enabled=True by default
```

## Rungs

1. **raw**: Raw representation (code_change, prompt, metadata) with PII redaction
2. **tokens**: Token-level sequences with canonicalized identifiers
3. **semantic_edits**: Semantic edit operations
4. **functions**: Function-level changes
5. **module_graph**: Module/file-level relationships
6. **motifs**: High-level workflow patterns
