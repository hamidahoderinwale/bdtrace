# Procedural Clio: Structural Analysis of LLM Agent Fix Strategies

Behavioral observation of coding-agent work from its artifacts — patch structure and action sequences — without relying on agent self-report. Companion analysis repo for **[Agent trajectories as programs (arXiv:2606.16988)](https://arxiv.org/abs/2606.16988)**; the reusable fingerprinting library that grew out of this work is **[procgrep](https://github.com/hamidahoderinwale/procgrep)**.

## Core findings

From patch structure (edit certificates):

1. **Structural patterns predict difficulty, semantics don't.** Frequent-itemset patterns over edit certificates separate difficulty 4.6x better than any semantic grouping (issue text, predicted fixes, or fix descriptions from agent traces).
2. **Agents can't describe their own fix strategies.** Self-reported edit operations match actual patch structure at F1=0.20. Observe behavior, don't ask.
3. **Agents use different structural approaches to the same problem.** On co-solved instances, agents produce identical edit certificates only 24% of the time. The LLM backbone drives strategy, not the scaffold.
4. **The hard part is composition, not primitives.** 43.8% of failures are composition failures: the agent has demonstrated every required edit operation but can't combine them.
5. **More benchmark instances don't help.** Strategy coverage saturates early; SWE-smith over-samples easy patterns and under-represents hard ones.

From action sequences (trajectories as programs — the paper's line):

6. **Procedural fingerprints identify the agent** (leakage-controlled), and a distilled student's nearest behavioral neighbor is its own teacher: lineage (0.25) is far closer than scaffold (0.53) or era (0.52) changes.
7. **Distillation transfers the vocabulary but narrows its use.** Canonical action vocabulary transfers completely (Jaccard 1.00) while the distribution concentrates (entropy 2.28→1.87) — rigidity, not selective loss; the drift is neither outcome-dependent nor localized.
8. **Failure is predictable from a short action prefix** (AUC 0.69 at three actions), grounding budget allocation across task attempts.
9. **Thinking models narrate least of what they do**: reverse coverage (did→says) is 0.20–0.25 for extended-thinking models vs 0.50–0.71 for older ones.

See [findings.md](findings.md) for the full record — including the grounding audit that separates reproduced claims from retracted ones, and results still awaiting write-up.

## Setup

```bash
uv sync
source .venv/bin/activate
```

Set `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in `.env` for LLM-based analyses.

## Representation pipeline

| Level | What it captures | Coverage |
|-------|-----------------|----------|
| Edit certificates | Set of (direction, AST-node-type) pairs from the patch | 289/300 (96%) |
| Scoped certificates | Edit type + file path + function/class scope + patch size | 300/300 |
| Contextual edit ops | Edit type + parent AST node (e.g. `ADD_For@FunctionDef`) | 203/300 (68%) |
| Fix intent labels | 12-category semantic taxonomy per hunk | 289/300 |
| Canonical action sequences | Trajectory as atoms (search/read/edit/test/...), BPE-learned procedures on top | 9 agents, 2,639 trajectories |

## Where things live

```
analysis/            -- core analysis modules (AST edits, scoped ops, procedures;
                        preferences/ = BPE + canonicalization, the current line)
representations/     -- computed (edits/modules/motifs) and inferred (first-gen DSPy) representations
scripts/             -- runnable analyses; agent_trajectories_paper/ = the paper's figure scripts
                        (its README maps script -> figure -> grounding status)
distillation_run/    -- teacher/student rollout analysis (fingerprints in-tree;
                        bulk .traj rollouts moving to HF, see MOVED.md when present)
docs/papers/         -- paper sources + figures
pipeline/, eval/,    -- first-generation (parquet + DSPy) pipeline, kept for provenance;
configs/, data/         notebooks/README.md marks what is stale
figures/             -- story-grouped figures
findings.md          -- full research record (findings, grounding audit, decision traces)
```

Generated data lives under `output/` (gitignored); each findings.md entry names the script that regenerates its numbers.

## License

MIT
