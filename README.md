# bdtrace

Your coding-agent sessions are sitting on your disk in a format nothing can read. bdtrace gets them
out, makes them safe to share, and hands them to analysis.

```
  local agent store          one JSONL record per session          somewhere else
  ────────────────           ────────────────────────────          ──────────────
  Claude Code                                                      a file you send
  Cursor          ──import──▶   {instance_id, agent, events[]}  ──▶  a Hugging Face dataset
  SWE-agent                                 │                        procgrep analysis
  OpenHands                                 │
                                    show · spec · query
                                    export · anonymize · audit
```

Everything between import and send is local and needs no account. One record shape carries every
source, so the same commands work whatever produced the session.

This is also the research home of [Agent trajectories as programs (arXiv:2606.16988)](https://arxiv.org/abs/2606.16988):
findings live in [findings.md](findings.md), and the fingerprinting library that grew out of the work
is [procgrep](https://github.com/hamidahoderinwale/procgrep).

## Share your traces

With [uv](https://docs.astral.sh/uv/) installed, no clone needed:

```console
$ uv tool install bdtrace     # before the first release: 'bdtrace @ git+https://github.com/hamidahoderinwale/bdtrace'
$ bdtrace import --source claude                       # your local Claude Code sessions (or: cursor)
138 traces -> traces.jsonl
$ bdtrace trace show --in traces.jsonl                 # read one, as prompts and what followed
$ bdtrace ex --in traces.jsonl --anonymize --strict
writing traces.anon.jsonl
export jsonl: 138 rec [00:02, 61 rec/s]
traces.anon.jsonl (34,554,690 bytes) + traces.anon.jsonl.meta.json
audit: no residual identity
```

`--anonymize` strips identity, `--strict` re-reads the result and refuses to leave the file on disk
if any survived, and the output names itself for what was done to it, so a shareable file is never
mistaken for a raw one. Send `traces.anon.jsonl`, or `.jsonl.gz` for a quarter of the size. Each
export carries a `.meta.json` sidecar with counts, sha256, summary stats and the record schema.

To publish instead: `bdtrace push --in traces.anon.jsonl --repo-id you/my-traces` writes a private
parquet-backed dataset plus `croissant.json` provenance.

The base install is small on purpose. A command needing more names its extra: `semantic` (embedding
search), `hub` (Hugging Face), `llm` (inferred representations), `parquet`.

## Check before you share

Anonymization is a denylist: it removes the classes it knows about. `trace audit` is the other half,
and it separates a rule failure from a judgment call:

```console
$ bdtrace trace audit --in traces.anon.jsonl
138 records scanned
RESIDUAL — none: every class the anonymizer covers is clean
CANDIDATES — identity-shaped tokens no rule can name; review these:
  Taste-AI              17   close with --redact Taste-AI
  rl.tastelabs.com       3   close with --redact rl.tastelabs.com
```

A **residual** is a defect: something the anonymizer claims to remove survived. A **candidate** is a
token sitting where identity sits, such as a handle after `--author` or an internal hostname, that no
pattern can recognise as identity. Close those with `--redact TERM`, repeatable. The `.meta.json`
sidecar is scanned too, since it travels with the artifact.

Covered: home directories in POSIX, Windows, URL-encoded and workspace-slug spellings; your login,
hostname and git identity, learned at runtime so it adapts to whoever runs it; emails; IP and MAC
addresses; git remotes and forge owners, including bare `github.com/acme` and `repos/acme`;
credentials by issuer prefix and by field name. Not covered, by construction: personal names in prose
and other people's handles written bare. That is what the candidate list is for.

## Reading a session

A trace stores a flat event list, which is the wrong shape for reading. `trace show` renders it as
turns: one prompt and the actions it caused, consecutive same-type actions collapsed.

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
shape across a corpus: on 138 local sessions, 5,770 turns with a median of 2 actions each and a tail
to 141.

## Finding sessions

Filters are additive, so they compose into one selection:

```console
$ bdtrace trace query --in traces.jsonl --semantic "debugging the anonymizer" \
    --interval 7d --top-k 20 --out hits.jsonl
```

`--grep` is a regex, `--where field=value` matches a field, `--interval` takes `A..B` ISO bounds or a
relative `7d` / `24h` / `2w`, and `--semantic` embeds the survivors and ranks by cosine similarity.
Structural filters run before any embedding, so the model only ever sees what survived them. `--sort
time` orders by first event instead of by score, and `--out` writes the matches.

Repeated searching over a growing corpus is worth an index: `bdtrace trace index --in traces.jsonl`
embeds each record once, skips anything unchanged on later runs, and lets a query embed only the
query string.

## Commands

Grouped by what they are for.

```
get      trace import --source claude|cursor|swe_agent|openhands [--input P] [--agent A]
look     trace show   --in F [--id X] [-n N] [--turns]        as prompt-anchored turns
         trace spec   [--in F]                                the record spec, or a file's stats
         trace head   --in F [-n --skip --interval --out]     raw records, windowed
find     trace query  --in F [--grep R] [--where f=v] [--interval I] [--semantic "…"] [--sort time]
         trace index  --in F [--model M]                      persistent embedding index
shape    trace export --in F [--out G] [--types --compact --anonymize --redact --strict --interval]
                                                              .jsonl .jsonl.gz .jsonl.zst .parquet .msgpack
         trace audit  --in F [--redact TERM] [--strict]       what identity survived
send     trace push   --in F --repo-id you/name [--dry-run]   HF dataset, private by default
         trace sessiongrep                                    session files for the sessiongrep index
derive   transform list [--examples] | <name> | all --in F [--llm]
         certs extract|distances|diversity                    edit certificates and measures over them
other    config · notebooks · lab · run <stem> · scripts [filter] · paper|fig|analysis|tools [<script>]
         trace fetch|parse                                    S3 trajectory fetch, raw Cursor dumps
```

`bdtrace import|export|push` work bare, since the tool name already carries the trace noun; `im` and
`ex` are shorter still, and `tf` `cfg` `nb` `ls` abbreviate the rest. `--help` works at every level.
Data goes to stdout and status to stderr, so commands pipe, and `-` works for either end.

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

Exports load anywhere: `pd.read_json(..., lines=True)`, `pl.read_ndjson(...)`, or `load_dataset(...)`
after a push.

## Keys

Local work needs none. Two things reach outward, and they authenticate differently on purpose.

**The hub is your own identity.** A push creates the dataset under whoever owns the token, so there
is no shared one. With no login, push offers to make one:

```console
$ bdtrace push --in traces.anon.jsonl --repo-id you/my-traces
bdtrace: no Hugging Face login found.
run `hf auth login` now? [Y/n]
```

`HF_TOKEN` in your environment or `.env` works too, and in a script it fails with the command rather
than hanging on a prompt.

**Model access is shared spend**, so it can come from the org. Set `OPENROUTER_API_KEY` or
`OPENAI_API_KEY` yourself, or, as a taste org member, just `op signin`: the CLI reads the org's key
from 1Password, where vault membership is the gate and access is revoked centrally. The `op://`
reference is committed and overridable with `BDTRACE_OP_OPENROUTER`; the value never is, and only its
source is printed. `bdtrace config` reports which rung is live.

## With procgrep

[procgrep](https://github.com/hamidahoderinwale/procgrep) is the analysis half: canonical action
atoms, BPE-learned procedures, Jensen-Shannon divergence between agents. bdtrace gets traces out and
makes them safe; procgrep measures them. An export feeds it through the `bdtrace` adapter:

```console
$ procgrep canonicalize --input traces.jsonl --output atoms.jsonl \
    --adapter bdtrace --trace-id-field instance_id
```

Anonymized exports canonicalize identically, since action structure survives anonymization. Every
record carries its source as `agent`, the field procgrep groups by, so cross-agent analyses work on a
mixed export without remapping.

## Working in the repo

```bash
git clone https://github.com/hamidahoderinwale/bdtrace
cd bdtrace
uv sync --all-extras       # full research stack; plain `uv sync` is just the CLI
```

Tests: `uv run python -m pytest bdtrace analysis/ingest`, run in CI on every push. The tool install
carries the trace and transform commands anywhere; script-backed commands (`paper`, `fig`,
`notebooks`, `run`) need this checkout.

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
