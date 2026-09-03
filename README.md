# bdtrace

Developer traces out of local agent stores: import, inspect, anonymize, export, share.
Also the research home of [Agent trajectories as programs (arXiv:2606.16988)](https://arxiv.org/abs/2606.16988):
findings live in [findings.md](findings.md), and the fingerprinting library that grew out of the work is
[procgrep](https://github.com/hamidahoderinwale/procgrep).

## Share your traces

With [uv](https://docs.astral.sh/uv/) installed, no clone needed:

```console
$ uv tool install bdtrace     # or, before the first release: 'bdtrace @ git+https://github.com/hamidahoderinwale/bdtrace'
$ bdtrace import --source claude --out traces.jsonl    # your local Claude Code sessions (or: cursor)
138 traces -> traces.jsonl
$ bdtrace trace head --in traces.jsonl -n 2            # eyeball before sharing
$ bdtrace ex --in traces.jsonl --out traces.jsonl.gz --anonymize   # identity stripped, ~4x smaller
```

Send the `.jsonl.gz`; its `.meta.json` sidecar carries counts, sha256, summary stats, and the record
schema. Or push a private Hugging Face dataset: install with the `[hub]` extra and
`bdtrace push --in traces.jsonl --repo-id you/my-traces`.

The base install is small on purpose. A command that needs more names its extra:
`semantic` (embedding search), `hub` (HF push), `llm` (inferred representations), `parquet`.

## Commands

```
trace import --source claude|cursor|swe_agent|openhands    local store -> traces.jsonl
trace spec [--in F]                 the record spec; with --in, audit a file
trace head --in F [-n --skip --interval --out]             samples, windowed slices
trace export --in F --out G [--types --compact --anonymize --interval]
                                    .jsonl .jsonl.gz .jsonl.zst .parquet .msgpack
trace query --in F [--grep R] [--where f=v] [--interval I] [--semantic "..."] [--sort time]
                                    additive filters; embedding-ranked with --semantic
trace index --in F [--model M]      embed once, incrementally; query then embeds only the query
trace push --in F --repo-id you/name [--dry-run]           HF dataset, private by default
trace sessiongrep|fetch|parse       sessiongrep export, S3 fetch, raw Cursor dumps
transform list | <name> | all --in F [--llm]               representation extractors
config                              which model key is active
certs extract|distances|diversity   edit certificates and measures over them
paper|fig|analysis|tools [<script>] script families (bare noun lists them)
notebooks | lab | run <stem> | scripts [filter]
```

`bdtrace import|export|push` work bare (also `im`/`ex`); `tf` `cfg` `nb` `ls` abbreviate.
`--help` at every level. Data on stdout, status on stderr, so commands pipe (`--in -`, `--out -`).

## Transformations

`bdtrace tf <name>|all --in F` applies the repo's representation extractors to trace or patch records:
`tokens` (event-type sequence), `functions`, `motifs`, `raw`, `edits` (AST edit certificate), and the
DSPy-inferred `behavioral` / `mechanistic` / `functional` behind `--llm`.

Model keys: `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in `.env`; taste org members need neither — a
1Password login is enough and the CLI reads the org's shared key from the vault (the reference is
committed, the key never is).

Exports load anywhere: `pd.read_json(..., lines=True)`, `pl.read_ndjson(...)`, or `load_dataset(...)`
after a push. A hub push also publishes `croissant.json` (Croissant 1.1, PROV-O provenance).

## With procgrep

[procgrep](https://github.com/hamidahoderinwale/procgrep) is the analysis half: canonical action
atoms, BPE-learned procedures, Jensen-Shannon divergence between agents. bdtrace gets the traces
out and makes them safe to share; procgrep measures them. An export feeds it directly through the
`bdtrace` adapter:

```console
$ bdtrace import --out traces.jsonl
$ procgrep canonicalize --input traces.jsonl --output atoms.jsonl \
    --adapter bdtrace --trace-id-field instance_id
wrote 3 canonical traces to atoms.jsonl
```

Anonymized exports canonicalize the same way, since the action structure survives anonymization.
Every record carries its source as `agent` (`--agent` overrides it at import), which is the field
procgrep groups and compares by, so cross-agent analyses work on a mixed export without remapping.

## Working in the repo

```bash
git clone https://github.com/hamidahoderinwale/bdtrace
cd bdtrace
uv sync --all-extras       # full research stack; plain `uv sync` is just the CLI
```

Tests: `uv run python -m pytest bdtrace analysis/ingest`, run in CI on every push. The tool install
carries the trace and transform commands anywhere; script-backed commands (`paper`, `fig`,
`notebooks`, `run`) need this checkout.

## Layout

```
bdtrace/             -- the CLI: import, export, query, index, metadata
analysis/            -- analysis modules; ingest/ = the local-store trace parsers
representations/     -- computed (edits/modules/motifs) and inferred (DSPy) representations
scripts/             -- runnable analyses; agent_trajectories_paper/ maps script -> figure -> grounding
docs/                -- paper sources and figures, distillation_run/, first-generation pipeline inputs
findings.md          -- full research record (findings, grounding audit, decision traces)
```

Generated data lives under `output/` (gitignored); every findings.md entry names the script that
regenerates its numbers.

