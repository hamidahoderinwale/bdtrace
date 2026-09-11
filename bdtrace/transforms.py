"""Registry of the repo's representation transformations, CLI-applicable.

Each entry wraps one extractor from representations/ so `bdtrace transform`
can enumerate them, apply one, or apply all to a JSONL of records. Two record
shapes exist: "trace" transforms take a whole trace dict (an `events` list),
"patch" transforms take before/after source fields. Imports are lazy so
listing costs nothing and LLM deps load only when an inferred transform runs.

The privacy operators in bdtrace/privacy.py live in the same registry with two
more shapes. "rewrite": the record goes in and the same record comes out with
one field type changed, nothing is added under `reprs`. "corpus": the same,
after one pass over the whole input builds the operator (min_users needs the
per-value developer counts before it can rewrite anything). They carry a `family`
(the privacy family they belong to), take their parameters explicitly
(`--param k=v`, typed from the operator's own signature), and are excluded from
`all` because each one destroys information and several need a salt.
"""

import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Transform:
    fn_name: str          # attribute in `module`
    kind: str             # "trace" (a trace dict) | "patch" (before/after fields) | "rewrite" (record -> record)
                          # | "corpus" (built from a pass over every record, then record -> record)
    llm: bool             # needs a configured LM (DSPy inferred representation)
    desc: str
    example: str          # real captured output, truncated; "" when none has been run
    module: str = "representations"   # where fn_name is resolved
    family: str = ""      # privacy family for rewrite operators; "" for representations


TRANSFORMS = {
    "tokens": Transform("tokens_repr", "trace", False, "token sequence over the trace's events",
                        '["prompt", "run", "run", "search", "read", "run", ...]  (6208 items)'),
    "raw": Transform("raw_repr", "trace", False, "normalized raw trace (events + metadata)",
                     '{"code_changes": [...], "prompts": [66 items], "metadata": {...}}'),
    "functions": Transform("functions_repr", "trace", False, "touched-function sequence",
                           '["token_in_namespace", "validate", "contains_norm", "catalog_at", ...]'),
    "motifs": Transform("motifs_repr", "trace", False, "recurring action motifs (statistical mining)",
                        '["M_1a2628fbdb", "M_d886ee7904", "M_9d98503d3b", ...]  (377 items)'),
    # note: bare `semantic_edits_repr` is the trace-shaped variant; `_source` is the before/after one
    "edits": Transform("semantic_edits_repr_source", "patch", False, "AST edit certificate from before/after source",
                       '{"operations": [{"type": "return_added", "location": "line 2",\n"node_type": "Return"}, ...], "delta": 6}'),
    "behavioral": Transform("behavioral_repr", "patch", True, "input-output behavioral claim (inferred)", ""),
    "mechanistic": Transform("mechanistic_repr", "patch", True, "mechanism-of-change description (inferred)", ""),
    "functional": Transform("functional_repr", "patch", True, "role/impact description (inferred)", ""),
}

PRIVACY_FAMILIES = ("pseudonymization", "generalization", "suppression", "perturbation", "order")


def _op(fn_name: str, family: str, desc: str, kind: str = "rewrite") -> Transform:
    return Transform(fn_name, kind, False, desc, "", module="bdtrace.privacy", family=family)


# one operator per field type; the help string says what leaves the record
TRANSFORMS.update({
    "hash_paths": _op("hash_paths", "pseudonymization",
                      "file_path -> salted sha256 hex; the path text leaves, its identity as a token stays"),
    "hash_commands": _op("hash_commands", "pseudonymization",
                         "command -> salted sha256 hex; the command text leaves, its identity as a token stays"),
    "paths_to_dir": _op("paths_to_dir", "generalization", "file_path -> its directory; the basename leaves"),
    "paths_to_ext": _op("paths_to_ext", "generalization",
                        "file_path -> lowercase extension or <noext>; directory and name leave"),
    "paths_to_basename": _op("paths_to_basename", "generalization", "file_path -> its basename; the directory leaves"),
    "commands_to_head": _op("commands_to_head", "generalization",
                            "command -> first token, lowercased; every argument leaves"),
    "commands_to_class": _op("commands_to_class", "generalization",
                             "command -> test|search|read|run|other (the ingest classifier); the command text leaves"),
    "text_to_length_bucket": _op("text_to_length_bucket", "generalization",
                                 "prompt text -> short|medium|long; the prompt text leaves"),
    "drop_field": _op("drop_field", "suppression",
                      "one named details key leaves every event (and the prompts copies)"),
    "min_users": _op("min_users", "suppression",
                     "file_path and command values seen under < k developers leave (two passes over the file)",
                     kind="corpus"),
    "truncate_events": _op("truncate_events", "suppression",
                           "events after the first n leave; prompts are cut to match"),
    "jitter_timestamps": _op("jitter_timestamps", "perturbation",
                             "timestamps shift by one seeded offset per record; absolute time leaves, intervals stay"),
    "shuffle_events": _op("shuffle_events", "order",
                          "event order leaves (seeded permutation); the multiset of events stays"),
    "verbs_only": _op("verbs_only", "order",
                      "every details block leaves (prompts too); type and timestamp stay"),
})

DEFAULT_MODEL = "openai/gpt-4o-mini"

# Org members' fallback: the shared OpenRouter key in the org's 1Password vault.
# The reference is not a secret; access is gated by vault membership and the
# member's own 1Password login. Anyone outside the org sets their own key.


def resolve_api_key() -> tuple[str, str] | None:
    """(key, source) from the shared ladder in `creds`: env, .env, then the org vault."""
    from bdtrace.creds import resolve

    found = resolve(("OPENROUTER_API_KEY", "OPENAI_API_KEY"), op_key="openrouter")
    if found and found[1].startswith("1Password"):
        os.environ["OPENROUTER_API_KEY"] = found[0]  # in-process only, so provider prefixing sees it
    return found


def configure_llm(model: str | None) -> str:
    """Configure DSPy, OpenRouter-first (same convention as the scripts)."""
    import dspy

    resolved = resolve_api_key()
    if not resolved:
        sys.exit("bdtrace: inferred transforms need OPENROUTER_API_KEY or OPENAI_API_KEY in env/.env,\n"
                 "or a 1Password login if you are in the org (`op signin`); see `bdtrace config`")
    api_key, source = resolved
    print(f"model key: {source}", file=sys.stderr)
    name = model or os.environ.get("BDTRACE_MODEL", DEFAULT_MODEL)
    if os.environ.get("OPENROUTER_API_KEY") and not name.startswith("openrouter/"):
        name = f"openrouter/{name}" if "/" in name else f"openrouter/openai/{name}"
    dspy.configure(lm=dspy.LM(model=name, api_key=api_key, temperature=0.0, max_tokens=1024, cache=True))
    return name


def _fn(t: Transform):
    return getattr(importlib.import_module(t.module), t.fn_name)


def operator_params(t: Transform) -> list[inspect.Parameter]:
    """A privacy operator's explicit parameters: everything after the record, read from
    its signature so the CLI and the listing cannot drift from the function."""
    return list(inspect.signature(_fn(t)).parameters.values())[1:]


def signature_text(t: Transform) -> str:
    return ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default}"
                     for p in operator_params(t))


def resolve_params(name: str, given: dict[str, str]) -> dict:
    """Typed parameters for one privacy operator from `--param k=v` strings: coerced by the
    signature's annotation, unknown names refused, missing ones named. A salt not given is
    read from BDTRACE_SALT (env, then .env) so it need not sit in shell history; only the
    source is printed, never the value, and no operator writes it into a record."""
    t = TRANSFORMS[name]
    params = {p.name: p for p in operator_params(t)}
    unknown = sorted(set(given) - set(params))
    if unknown:
        sys.exit(f"bdtrace: `{name}` has no parameter {', '.join(unknown)}; it takes: {signature_text(t) or 'none'}")
    out: dict = {}
    for k, v in given.items():
        ann = params[k].annotation
        try:
            out[k] = ann(v) if ann in (int, float) else v
        except ValueError:
            sys.exit(f"bdtrace: `{name}` parameter {k} must be {ann.__name__}, got {v!r}")
    if "salt" in params and "salt" not in out:
        from bdtrace.creds import resolve

        found = resolve(("BDTRACE_SALT",))
        if not found:
            sys.exit(f"bdtrace: `{name}` needs a salt: --param salt=... or BDTRACE_SALT in env/.env\n"
                     "(the salt is a key, not part of the record; the env var keeps it out of shell history)")
        out["salt"] = found[0]
        print(f"salt: {found[1]}", file=sys.stderr)
    missing = [p for p in params if p not in out and params[p].default is inspect.Parameter.empty]
    if missing:
        sys.exit(f"bdtrace: `{name}` needs --param {' '.join(f'{m}=...' for m in missing)}")
    return out


def _records(in_path: Path, limit: int | None):
    """The records `apply` will see, line for line with the same limit, for a corpus operator's first pass."""
    with open(in_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def apply(names: list[str], in_path: Path, out_path: Path,
          before_field: str, after_field: str, limit: int | None, params: dict | None = None) -> None:
    """Representation transforms add `reprs[name]` to each record; rewrite operators
    replace the record, in the order given, with `params` passed to each of them; a
    corpus operator is first built from a pass over the input, then applied like a rewrite."""
    picked = {n: TRANSFORMS[n] for n in names}
    params = params or {}
    fns = {}
    for n, t in picked.items():
        fn = _fn(t)
        if t.kind == "corpus":
            try:
                fn = fn(_records(in_path, limit), **params)
            except ValueError as e:
                sys.exit(f"bdtrace: {n}: {e}")
        fns[n] = fn
    n_in = n_err = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if limit is not None and n_in >= limit:
                break
            record = json.loads(line)
            n_in += 1
            for name, t in picked.items():
                if t.family:
                    # not caught: a privacy rewrite that fails must not emit the record unprotected
                    try:
                        record = fns[name](record, **params) if t.kind == "rewrite" else fns[name](record)
                    except ValueError as e:  # a bad parameter, e.g. an unknown scope or a negative n
                        sys.exit(f"bdtrace: {name}: {e}")
                    continue
                reprs = record.setdefault("reprs", {})
                try:
                    if t.kind == "trace":
                        reprs[name] = fns[name](record)
                    else:
                        reprs[name] = fns[name](record.get(before_field, ""), record.get(after_field, ""))
                except Exception as e:  # a bad record shouldn't kill the pass; the error is recorded on the row
                    reprs[name] = {"error": f"{type(e).__name__}: {e}"}
                    n_err += 1
            fout.write(json.dumps(record, default=str) + "\n")
    print(f"{n_in} records -> {out_path} ({', '.join(picked)}; {n_err} per-record errors)", file=sys.stderr)


def list_table(examples: bool = False) -> str:
    width = max(map(len, TRANSFORMS))

    def rows(llm: bool) -> list[str]:
        out = []
        for n, t in TRANSFORMS.items():
            if t.llm is not llm or t.family:
                continue
            out.append(f"  {n:<{width}}  {t.kind:<5}  {t.desc}")
            if examples:
                # honest about coverage: an inferred transform has no sample because
                # none has been run here, and inventing one would misrepresent output
                sample = t.example or "(no sample recorded: needs a model key to run)"
                out += [f"  {'':<{width}}         {line}" for line in sample.splitlines()]
        return out

    lines = ["computed (no API key needed):", *rows(llm=False),
             "inferred (DSPy; needs OPENROUTER_API_KEY or OPENAI_API_KEY):", *rows(llm=True)]
    ops = {n: t for n, t in TRANSFORMS.items() if t.family}
    if ops:
        lines.append("privacy operators (record -> record, one field type each; chain calls to compose; not in `all`;")
        lines.append("  parameters via --param k=v, the salt also via BDTRACE_SALT):")
        heads = {n: f"{n}({signature_text(t)})" for n, t in ops.items()}
        w = max(map(len, heads.values()))
        for family in PRIVACY_FAMILIES:
            members = [n for n, t in ops.items() if t.family == family]
            if members:
                lines.append(f"  {family}:")
                lines += [f"    {heads[n]:<{w}}  {ops[n].kind:<7}  {ops[n].desc}" for n in members]
    lines.append("record shapes: trace = a trace dict with an `events` list; patch = before/after source fields;")
    lines.append("  rewrite = a trace dict in, the same record out with one field type changed;")
    lines.append("  corpus = the same, after one pass over the whole input builds the operator")
    lines.append("measured basis (inter_eval diversity, Lite + SWE-Smith): edits and module graph carry the")
    lines.append("  independent structural signal; raw-edits vs edit set-diff are rho=1.0 redundant.")
    lines.append("  The inferred representations have no redundancy verdict yet.")
    return "\n".join(lines)


def config_report() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    def status(var: str) -> str:
        return "set" if os.environ.get(var) else "missing"
    from bdtrace.creds import OP_REFS, describe
    org = describe("openrouter")
    from bdtrace.export import resolve_hf_token
    hf = resolve_hf_token()
    hf_state = f"{hf[1]}" if hf else "not logged in (bdtrace push offers `hf auth login`)"
    return "\n".join([
        f"OPENROUTER_API_KEY  {status('OPENROUTER_API_KEY')}   (preferred provider; set in .env)",
        f"OPENAI_API_KEY      {status('OPENAI_API_KEY')}   (fallback)",
        f"org model key {org}",
        f"                    ({OP_REFS['openrouter']}; used when no key of your own is set)",
        f"hugging face        {hf_state}   (your own identity; never shared)",
        f"BDTRACE_MODEL       {os.environ.get("BDTRACE_MODEL", f'unset (default {DEFAULT_MODEL})')}",
        "inferred transforms run at temperature 0.0 with the DSPy cache on",
    ])
