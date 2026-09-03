# bdtrace

Developer traces out of local agent stores: import, read, anonymize, share.
Also the research home of [Agent trajectories as programs (arXiv:2606.16988)](https://arxiv.org/abs/2606.16988):
findings live in [findings.md](findings.md), and the fingerprinting library that grew out of the work
is [procgrep](https://github.com/hamidahoderinwale/procgrep).

## Share your traces

With [uv](https://docs.astral.sh/uv/) installed, no clone needed:

```console
$ uv tool install bdtrace     # before the first release: 'bdtrace @ git+https://github.com/hamidahoderinwale/bdtrace'
$ bdtrace import --source claude --out traces.jsonl    # your local Claude Code sessions (or: cursor)
138 traces -> traces.jsonl
$ bdtrace trace show --in traces.jsonl                 # read one, as prompts and what followed
$ bdtrace ex --in traces.jsonl --out share.jsonl --anonymize --strict
audit: no residual identity
```

`--anonymize` strips identity; `--strict` then re-reads the output and refuses to leave the file on
disk if any survived. Send `share.jsonl` (or `.jsonl.gz`, ~4x smaller); its `.meta.json` sidecar
carries counts, sha256, summary stats and the record schema. To publish instead, install the `[hub]`
extra and `bdtrace push --in share.jsonl --repo-id you/my-traces`, which writes a private
parquet-backed dataset plus `croissant.json` provenance.

The base install is small on purpose. A command needing more names its extra: `semantic` (embedding
search), `hub` (Hugging Face), `llm` (inferred representations), `parquet`.

## Check before you share

Anonymization is a denylist: it removes the classes it knows. `trace audit` is the other half, and
it separates a rule failure from a judgment call:

```console
$ bdtrace trace audit --in share.jsonl
138 records scanned
RESIDUAL — none: every class the anonymizer covers is clean
CANDIDATES — identity-shaped tokens no rule can name; review these:
  Taste-AI              17   close with --redact Taste-AI
  rl.tastelabs.com       3   close with --redact rl.tastelabs.com
```

A **residual** is a defect: something the anonymizer claims to remove survived. A **candidate** is a
token sitting where identity sits (a handle after `--author`, an internal hostname) that no pattern
can recognise as identity. Close those with `--redact TERM` on the export, repeatable.

Covered classes: home directories in POSIX, Windows, URL-encoded and workspace-slug spellings; the
local login, hostname and git identity, learned at runtime so it adapts to whoever runs it; emails;
IP and MAC addresses; git remotes and forge owners, including bare `github.com/acme` and
`repos/acme`; credentials by issuer prefix and by field name. Not covered, by construction: personal
names in prose and other people's handles written bare. That is what the candidate list is for.

## Reading a session

A trace stores a flat event list, which is the wrong shape for reading. `trace show` renders it as
turns, one prompt and the actions it caused, consecutive same-type actions collapsed:

```console
$ bdtrace trace show --in traces.jsonl
claude-0714fa67 /Users/hamidah
225 events  2026-08-10T18:21:59 .. 2026-08-11T02:09:05

▸ go ahead and do research on udom and its utilities for understanding cross-framework…
    $ run   ×2  grep -ril 'udom' …, grep -rl 'udom' …
    ⌕ search    ToolSearch
    ▪ read      .claude/projects/…
    ✎ edit      judge/lint.py

▸ what is the actual experiment.
    $ run   ×3  gh api 'repos/…/git…, for f in examples/eval/…
```

The turn is also the analytic unit, since it is what one instruction bought. `--turns` reports that
shape across a corpus: on 138 local sessions, 5,770 turns with a median of 2 actions each and a long
tail to 141.

## Commands

```
trace import --source claude|cursor|swe_agent|openhands    local store -> traces.jsonl
trace show --in F [--id X] [-n N] [--turns]       read as prompt-anchored turns
trace audit --in F [--redact TERM] [--strict]     what identity survived
trace spec [--in F]                               the record spec; with --in, a file's stats
trace head --in F [-n --skip --interval --out]    raw samples, windowed slices
trace export --in F --out G [--types --compact --anonymize --redact --strict --interval]
                                                  .jsonl .jsonl.gz .jsonl.zst .parquet .msgpack
trace query --in F [--grep R] [--where f=v] [--interval I] [--semantic "…"] [--sort time]
                                                  additive filters; embedding-ranked with --semantic
trace index --in F [--model M]                    embed once, incrementally; query embeds only the query
trace push --in F --repo-id you/name [--dry-run]  HF dataset, private by default
trace sessiongrep|fetch|parse                     sessiongrep export, S3 fetch, raw Cursor dumps
transform list [--examples] | <name> | all --in F [--llm]     representation extractors
config                                            which model key is active
certs extract|distances|diversity                 edit certificates and measures over them
paper|fig|analysis|tools [<script>]               script families (bare noun lists them)
notebooks | lab | run <stem> | scripts [filter]
```

`bdtrace import|export|push` work bare (also `im`/`ex`); `tf` `cfg` `nb` `ls` abbreviate. `--help` at
every level. Data on stdout, status on stderr, so commands pipe (`--in -`, `--out -`).

## Transformations

`bdtrace tf <name>|all --in F` applies the repo's representation extractors to trace or patch
records. `tf list --examples` prints real captured output for each:

```
tokens      trace  ["prompt", "run", "run", "search", "read", "run", ...]  (6208 items)
functions   trace  ["token_in_namespace", "validate", "contains_norm", ...]
motifs      trace  ["M_1a2628fbdb", "M_d886ee7904", ...]  (377 items)
edits       patch  {"operations": [{"type": "return_added", "location": "line 2", …}], "delta": 6}
behavioral / mechanistic / functional   inferred via DSPy, behind --llm
```

## Keys

Everything local needs none: import, show, audit, export, interval slicing and semantic search all
run with no account. Two things reach outward, and both use one ladder: your environment, then
`.env`, then the org's shared secret in 1Password, then the provider's own cached login.

| for | your own | or, in the taste org |
|---|---|---|
| inferred transforms (`--llm`) | `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | `op signin`, nothing else |
| `trace push` to Hugging Face | `HF_TOKEN`, or `huggingface-cli login` | `op signin`, nothing else |

Org members need no key of their own: vault membership is the gate and their own 1Password login is
the auth, so access is granted and revoked centrally. The `op://` references are committed (they are
not secrets); the values never are, and only the source name is ever printed. `bdtrace config`
reports which rung is live. Override a reference with `BDTRACE_OP_HF` or `BDTRACE_OP_OPENROUTER`.

Exports load anywhere: `pd.read_json(..., lines=True)`, `pl.read_ndjson(...)`, or `load_dataset(...)`
after a push.

## With procgrep

[procgrep](https://github.com/hamidahoderinwale/procgrep) is the analysis half: canonical action
atoms, BPE-learned procedures, Jensen-Shannon divergence between agents. bdtrace gets traces out and
makes them safe to share; procgrep measures them. An export feeds it through the `bdtrace` adapter:

```console
$ procgrep canonicalize --input traces.jsonl --output atoms.jsonl \
    --adapter bdtrace --trace-id-field instance_id
```

Anonymized exports canonicalize identically, since action structure survives anonymization. Every
record carries its source as `agent` (`--agent` overrides at import), the field procgrep groups by,
so cross-agent analyses work on a mixed export without remapping.

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
bdtrace/             -- the CLI: import, show, audit, export, query, index, metadata
analysis/            -- analysis modules; ingest/ = the local-store trace parsers
representations/     -- computed (edits/modules/motifs) and inferred (DSPy) representations
scripts/             -- runnable analyses; agent_trajectories_paper/ maps script -> figure -> grounding
docs/                -- paper sources and figures, distillation_run/, first-generation pipeline inputs
findings.md          -- full research record (findings, grounding audit, decision traces)
```

Generated data lives under `output/` (gitignored); every findings.md entry names the script that
regenerates its numbers.
