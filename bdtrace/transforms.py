"""Registry of the repo's representation transformations, CLI-applicable.

Each entry wraps one extractor from representations/ so `bdtrace transform`
can enumerate them, apply one, or apply all to a JSONL of records. Two record
shapes exist: "trace" transforms take a whole trace dict (an `events` list),
"patch" transforms take before/after source fields. Imports are lazy so
listing costs nothing and LLM deps load only when an inferred transform runs.
"""

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Transform:
    fn_name: str          # attribute in the representations package
    kind: str             # "trace" (record is a trace dict) | "patch" (before/after fields)
    llm: bool             # needs a configured LM (DSPy inferred representation)
    desc: str
    example: str          # real captured output, truncated; "" when none has been run


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

DEFAULT_MODEL = "openai/gpt-4o-mini"

# Org members' fallback: the shared OpenRouter key in the Taste 1Password vault.
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
                 "or a 1Password login if you are in the taste org (`op signin`); see `bdtrace config`")
    api_key, source = resolved
    print(f"model key: {source}", file=sys.stderr)
    name = model or os.environ.get("BDTRACE_MODEL", DEFAULT_MODEL)
    if os.environ.get("OPENROUTER_API_KEY") and not name.startswith("openrouter/"):
        name = f"openrouter/{name}" if "/" in name else f"openrouter/openai/{name}"
    dspy.configure(lm=dspy.LM(model=name, api_key=api_key, temperature=0.0, max_tokens=1024, cache=True))
    return name


def _fn(t: Transform):
    return getattr(importlib.import_module("representations"), t.fn_name)


def apply(names: list[str], in_path: Path, out_path: Path,
          before_field: str, after_field: str, limit: int | None) -> None:
    picked = {n: TRANSFORMS[n] for n in names}
    fns = {n: _fn(t) for n, t in picked.items()}
    n_in = n_err = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if limit is not None and n_in >= limit:
                break
            record = json.loads(line)
            n_in += 1
            reprs = record.setdefault("reprs", {})
            for name, t in picked.items():
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
            if t.llm is not llm:
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
    lines.append("record shapes: trace = a trace dict with an `events` list; patch = before/after source fields")
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
    org, org_hf = describe("openrouter"), describe("huggingface")
    return "\n".join([
        f"OPENROUTER_API_KEY  {status('OPENROUTER_API_KEY')}   (preferred provider; set in .env)",
        f"OPENAI_API_KEY      {status('OPENAI_API_KEY')}   (fallback)",
        f"taste org model key {org}",
        f"                    ({OP_REFS['openrouter']})",
        f"taste org HF token  {org_hf}",
        f"                    ({OP_REFS['huggingface']})",
        f"HF_TOKEN            {status('HF_TOKEN')}   (Hugging Face fetches, e.g. rollout trajectories)",
        f"BDTRACE_MODEL       {os.environ.get("BDTRACE_MODEL", f'unset (default {DEFAULT_MODEL})')}",
        "inferred transforms run at temperature 0.0 with the DSPy cache on",
    ])
