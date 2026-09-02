# Notes

## 2026-09-02 — bidirect CLI

**Intent.** One installed entry point (`bidirect`) over the repo's runnable surface, shaped
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
   committed, the key never is). Benefit: taste org membership alone grants model access, gated
   and revocable through vault ACLs; outsiders use their own key. Price: shared key means shared
   billing with no per-user attribution; preview vault key, so prod spend is untouched.
4. **Exported name trap:** `representations.semantic_edits_repr` is trace-shaped; the
   before/after extractor is `semantic_edits_repr_source`. The registry uses `_source`; calling
   the bare name with two strings returns `[]` silently.
