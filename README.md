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

## CLI

`uv sync` installs `bdtrace`. The main path is trace data: pull it out of a
local agent store, inspect it, re-serialize it, push it to the Hugging Face hub.
JSONL is the canonical interchange throughout.

```
trace import --source claude|cursor|swe_agent|openhands   local store -> traces.jsonl
trace spec [--in F]        the record spec; with --in, audit a file (counts, coverage, event types)
trace head --in F [-n --skip --interval --out]   view samples; --out makes it a segmented export
trace export --in F --out G       re-serialize: .jsonl .jsonl.gz .jsonl.zst .parquet .msgpack
trace push --in F --repo-id you/name [--dry-run]   HF dataset push (parquet-backed, private default)
trace sessiongrep|fetch|parse     sessiongrep session files, S3 trajectory fetch, raw Cursor dumps
transform list | <name> | all --in F [--llm]     apply representation extractors to records
config                     which model key is active (own .env, or the org 1Password key)
certs extract|distances|diversity                edit certificates and the measures over them
paper|fig|analysis|tools [<script>]              grouped script families (bare noun lists)
notebooks | lab | run <stem> | scripts [filter]  legacy chain, JupyterLab, any script
```

Shorthand: `bdtrace import|export|push` work bare (the tool name carries the
noun; also `im`/`ex`), and `tf` = transform, `cfg` = config, `nb` = notebooks,
`ls` = scripts. `--help` works at every level; status goes to stderr and data
to stdout, so commands pipe (`--in -` and `--out -` for stdin/stdout).

A real session, Claude Code store to hub:

```console
$ bdtrace import --limit 3 --out t.jsonl
3 traces -> t.jsonl
$ bdtrace trace spec --in t.jsonl
t.jsonl: 3 records, 2,431,755 bytes
events per record: min 225, median 715, max 1673
event types: run 1601, prompt 471, other 239, read 137, edit 123, search 38, test 4
$ bdtrace trace head --in t.jsonl -n 1 --interval 2026-08-01..   # windowed sample view
$ bdtrace ex --in t.jsonl --out t.jsonl.gz                       # 585,582 bytes (4.2x smaller)
$ bdtrace trace push --in t.jsonl --repo-id midah/x --dry-run
dry-run: 3 rows -> midah/x (private=True); features: instance_id, repo, cwd, events, prompts
```

What each transformation produces, on real records (`bdtrace tf <name> --in F`):

```
tokens      ["prompt", "run", "run", "search", "read", "run", ...]      event-type sequence
functions   ["token_in_namespace", "validate", "contains_norm", ...]   touched functions
motifs      ["M_1a2628fbdb", "M_d886ee7904", ...]                      recurring action motifs
raw         {"code_changes": [...], "prompts": [...], "metadata": {...}}
edits       {"operations": [{"type": "return_added", "location": "line 2",
             "node_type": "Return"}, ...], "delta": 6}                 AST edit certificate
behavioral / mechanistic / functional   inferred via DSPy; needs a model key (--llm)
```

Model keys for the inferred transformations: set `OPENROUTER_API_KEY` or
`OPENAI_API_KEY` in `.env`, or, for taste org members, none at all: a 1Password
login (`op signin`) is enough and the CLI reads the org's shared OpenRouter key
from the vault. The reference lives in the code; the key never does.

Every export writes a `<file>.meta.json` sidecar (sha256, record count, the
projection that shaped it, and the record JSON Schema); a hub push also
publishes `croissant.json` (Croissant 1.1 with PROV-O, semver via
`--dataset-version`). Consuming the exports from any dataframe library is one
line — nothing bidirect-specific is required:

```python
pd.read_json("traces.jsonl", lines=True)          # pandas (also .jsonl.gz)
pl.read_ndjson("traces.jsonl")                     # polars
load_dataset("you/name")                           # datasets, after a push
```

Tests run with `uv run python -m pytest bidirect analysis/ingest` and on every
push via GitHub Actions. From a bare `pip install git+...`, the trace and
transform commands work anywhere; script-backed commands need this checkout.

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
