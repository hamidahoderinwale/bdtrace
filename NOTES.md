# Notes

## 2026-09-10 — intent-release branch (paper repo: trace-funnel-paper)

**Intent.** Make the privacy mechanism a selection criterion instead of a fixed anonymizer. A
declared research intent (utility columns) and declared sensitive columns pick the operating
point on a measured utility-versus-leakage frontier over candidate representations. Lands as
`trace import --source swechat|specstory`, type-driven field operators in `transform` (keep,
hash, bucket, coarsen, drop) that generate the candidate lattice, `trace funnel --utility ...
--sensitive ...` that scores every lattice point against chance and prints the frontier, and
`trace release --at <point> --min-users k` that writes the artifact and runs `audit` on it.

**Design decisions.**

1. **Operators are per field type, never per rung.** Benefit: the candidate set is generated
   from the record spec and grows with it; no hand-ordered ladder to defend. Price: the lattice
   is large and needs a greedy search (single-field ablations, then compose).
2. **Leakage and utility are the same instrument, a classifier over the representation, read
   against a chance floor.** Benefit: one code path, comparable numbers. Price: a weak
   classifier understates leakage; report it as a lower bound and say so in the output.
3. **`anonymize` stays as it is.** Benefit: the identity denylist and `audit` remain the last
   line before sharing. Price: two mechanisms to explain; `release` calls `audit`, so a user
   sees one.

## 2026-09-02 — trace metadata and summary stats

1. **One `summarize()` feeds both surfaces** (`trace spec --in` renders it; export sidecars embed
   it verbatim). Benefit: the CLI readout and the metadata never disagree. Price: sidecars for
   non-jsonl exports summarize the pre-projection source (decoding the artifact would double the
   write cost) and say so in `summary_of`.
2. **Croissant only at the publish boundary** (hub push), lightweight `*.meta.json` on every
   export — the org placement rule: Croissant where a consumer machine acts on it; JSON Schema is
   the validation authority everywhere.
3. **Line formats stream, columnar formats materialize.** jsonl/gz/zst exports are constant-memory
   at any size (tqdm progress on stderr); parquet/msgpack need the rows in hand by nature.
4. **Embedding index (PR, trace-index branch): flat numpy + content-addressed increments.**
   Benefit: repeated semantic queries over a longitudinal corpus embed only the query; exact
   cosine, no ANN dep below ~1M vectors. Price: a full matrix rewrite on rebuild, and one
   `.index/` dir to keep beside each corpus; model change = full re-embed by design.

## 2026-09-02 — bdtrace CLI

**Intent.** One installed entry point (`bdtrace`) over the repo's runnable surface, shaped
noun-verb around its objects; the transformation registry is the centerpiece.

**Design decisions.**

1. **Transformations wrap `representations/`, scripts are dispatched by runpy.** Benefit: the
   transform commands survive a hosted (non-editable) install because they import packaged code,
   and script dispatch never re-declares any script's argparse. Price: script commands (`run`,
   `paper`, `fig`, ...) need a repo checkout; only `transform`/`config` work from a bare install.
2. **`transform all` includes LLM-backed transforms only with `--llm`.** Benefit: `all` is safe
   and free by default; spend is opt-in per invocation. Price: one extra flag for the full pass.
3. **Model key ladder: own env/.env, then the org's shared OpenRouter key via the signed-in
   1Password CLI** (`op://infra / preview/Shared - OpenRouter/credential`; the reference is
   committed, the key never is). Benefit: org membership alone grants model access, gated
   and revocable through vault ACLs; outsiders use their own key. Price: shared key means shared
   billing with no per-user attribution; preview vault key, so prod spend is untouched.
4. **Exported name trap:** `representations.semantic_edits_repr` is trace-shaped; the
   before/after extractor is `semantic_edits_repr_source`. The registry uses `_source`; calling
   the bare name with two strings returns `[]` silently.
