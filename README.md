# Procedural Clio: Structural Analysis of LLM Agent Fix Strategies

Behavioral observation of fix procedures from LLM agent traces on SWE-bench, without relying on agent self-report.

## Core findings

1. **Structural patterns predict difficulty, semantics don't.** FIM on edit certificates separates difficulty 4.6x better than any semantic grouping (issue text, predicted fix descriptions, or fix descriptions from agent traces).

2. **Agents can't describe their own fix strategies.** Self-reported edit operations match actual patch structure at F1=0.20. Observe behavior, don't ask.

3. **Agents use different structural approaches to the same problem.** On co-solved instances, agents produce identical edit certificates only 24% of the time (median Jaccard 0.56). The LLM backbone drives strategy, not the scaffold.

4. **The hard part is composition, not primitives.** 43.8% of agent failures are composition failures: the agent has individually demonstrated every required edit operation but can't combine them. For the hardest instances, it's 50.4%.

5. **More benchmark instances don't help.** Strategy coverage saturates early. SWE-smith over-samples easy patterns (52.7% return-value changes) while under-representing hard ones.

See [findings.md](findings.md) for the full record with methodology, decision traces, and literature anchors.

## Setup

```bash
uv sync
source .venv/bin/activate
```

Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in `.venv/.env` for LLM-based analyses.

## Representation pipeline

| Level | What it captures | Coverage |
|-------|-----------------|----------|
| Edit certificates | Set of (direction, AST-node-type) pairs from the patch | 289/300 (96%) |
| Scoped certificates | Edit type + file path + function/class scope + patch size | 300/300 |
| Contextual edit ops | Edit type + parent AST node (e.g. `ADD_For@FunctionDef`) | 203/300 (68%) |
| Fix intent labels | 12-category semantic taxonomy per hunk | 289/300 |

## Key scripts

**Analysis:**
- `scripts/compositional_generalization.py` -- classify failures as novel primitive vs novel composition across 84 agents
- `scripts/fim_difficulty_analysis.py` -- connect FIM patterns to 84-agent ease data
- `scripts/semantic_vs_structural.py` -- nearest-neighbor, UMAP, and variance comparison
- `scripts/validate_grounding.py` -- measure self-report accuracy (grounding failure)
- `scripts/compare_representations.py` -- kNN prediction across representation types

**Pipeline:**
- `scripts/build_canonical_forms.py` -- FIM closed itemsets from edit certificates
- `scripts/build_scoped_certificates.py` -- oracle scoped certificates with file/scope/size
- `scripts/build_agent_scoped_certs.py` -- agent scoped certificates + oracle alignment

**Figures:**
- `scripts/pairwise_figures_v2.py` -- agent comparison: strip plots, divergence scatter, vocabulary, instance flow
- `scripts/scoped_figures.py` -- file navigation, scope decomposition, minimality, instance anatomy
- `scripts/cluster_fix_descriptions.py` -- variance comparison across all groupings
- `scripts/build_figures.py` -- paper-level conceptual figures

## Data

- `output/leaderboard/lite_results.msgpack` -- 84 agents, pass/fail per instance
- `output/resolved_traces_lite_full.jsonl` -- 300 oracle traces with file paths and content
- `output/canonical_forms/` -- FIM patterns and instance assignments
- `output/compositional_generalization/` -- failure classification and composition gap data
- `output/pairwise_agent_comparison/` -- agent edit certificates and pairwise Jaccard
- `output/scoped_certificates/` -- enriched certificates with file/scope information

## Structure

```
analysis/          -- core analysis modules (AST edits, scoped ops, procedures)
representations/   -- computed and inferred representations
scripts/           -- all runnable scripts
configs/           -- benchmark configs, DSPy config
data/              -- data loaders
eval/              -- evaluation pipeline
output/            -- all generated data and figures
findings.md        -- full research record
```

## License

MIT
