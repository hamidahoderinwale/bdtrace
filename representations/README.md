# Representations

Folder name carries representation identity; filenames describe what the file does.

## Structure

```
representations/
├── computed/           # edits, modules, motifs (structure from code/traces)
│   ├── edits/          # certificate_distance, operation_divergence, tree_edit_distance
│   ├── modules/        # graph_distance, edge_divergence
│   └── motifs/         # motif_distance, vocabulary_coverage, dtw_similarity
├── inferred/           # behavioral, mechanistic, functional (LLM-derived)
│   ├── behavioral/     # BehavioralModule, behavioral_repr, claim_distance, field_distances
│   ├── mechanistic/    # MechanisticModule, mechanistic_repr, pattern_distance, location_overlap
│   ├── functional/     # FunctionalModule, functional_repr, role_distance, grounding_overlap
│   ├── modules.py      # InferredRepresentationsModule (composed)
│   └── utils/          # embed, provenance, distances
├── core/                # intent, utils (parsers, AST helpers)
└── encoders/            # raw, tokens, functions
```

## Pattern

- `{repr}.py` — encoder
- `distance.py` — distance metric for that representation
- DSPy modules (`*Module`) support save/load, batch; functions are thin wrappers

## Optional parsers

Token extraction defaults to Python stdlib AST only. For JS/Java AST parsing:

```bash
pip install .[parsers]
```

Then pass `parsers=["python", "javascript"]` or `parsers=["python", "java"]` to `tokens_repr()`.
