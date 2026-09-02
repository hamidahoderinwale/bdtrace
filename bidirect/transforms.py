"""Registry of the repo's representation transformations, CLI-applicable.

Each entry wraps one extractor from representations/ so `bidirect transform`
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


TRANSFORMS = {
    "tokens": Transform("tokens_repr", "trace", False, "token sequence over the trace's events"),
    "raw": Transform("raw_repr", "trace", False, "normalized raw trace (events + metadata)"),
    "functions": Transform("functions_repr", "trace", False, "touched-function sequence"),
    "motifs": Transform("motifs_repr", "trace", False, "recurring action motifs (statistical mining)"),
    # note: bare `semantic_edits_repr` is the trace-shaped variant; `_source` is the before/after one
    "edits": Transform("semantic_edits_repr_source", "patch", False, "AST edit certificate from before/after source"),
    "behavioral": Transform("behavioral_repr", "patch", True, "input-output behavioral claim (inferred)"),
    "mechanistic": Transform("mechanistic_repr", "patch", True, "mechanism-of-change description (inferred)"),
    "functional": Transform("functional_repr", "patch", True, "role/impact description (inferred)"),
}

DEFAULT_MODEL = "openai/gpt-4o-mini"

# Org members' fallback: the shared OpenRouter key in the Taste 1Password vault.
# The reference is not a secret; access is gated by vault membership and the
# member's own 1Password login. Anyone outside the org sets their own key.
OP_REF = os.environ.get("BIDIRECT_OP_REF", "op://infra / preview/Shared - OpenRouter/credential")


def _org_key() -> str | None:
    """Read the org-shared key via the signed-in 1Password CLI, if available."""
    import subprocess

    try:
        out = subprocess.run(["op", "read", OP_REF], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def resolve_api_key() -> tuple[str, str] | None:
    """(key, source): own env/.env first, then the org vault."""
    from dotenv import load_dotenv

    load_dotenv()
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(var):
            return os.environ[var], var
    key = _org_key()
    if key:
        os.environ["OPENROUTER_API_KEY"] = key  # in-process only, so provider prefixing sees it
        return key, "1Password (taste org shared key)"
    return None


def configure_llm(model: str | None) -> str:
    """Configure DSPy, OpenRouter-first (same convention as the scripts)."""
    import dspy

    resolved = resolve_api_key()
    if not resolved:
        sys.exit("bidirect: inferred transforms need OPENROUTER_API_KEY or OPENAI_API_KEY in env/.env,\n"
                 "or a 1Password login if you are in the taste org (`op signin`); see `bidirect config`")
    api_key, source = resolved
    print(f"model key: {source}", file=sys.stderr)
    name = model or os.environ.get("BIDIRECT_MODEL", DEFAULT_MODEL)
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
    print(f"{n_in} records -> {out_path} ({', '.join(picked)}; {n_err} per-record errors)")


def list_table() -> str:
    width = max(map(len, TRANSFORMS))
    lines = ["computed (no API key needed):"]
    lines += [f"  {n:<{width}}  {t.kind:<5}  {t.desc}" for n, t in TRANSFORMS.items() if not t.llm]
    lines.append("inferred (DSPy; needs OPENROUTER_API_KEY or OPENAI_API_KEY):")
    lines += [f"  {n:<{width}}  {t.kind:<5}  {t.desc}" for n, t in TRANSFORMS.items() if t.llm]
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
    org = "reachable (op signed in)" if _org_key() else "not reachable (no op CLI / not signed in / not in org)"
    return "\n".join([
        f"OPENROUTER_API_KEY  {status('OPENROUTER_API_KEY')}   (preferred provider; set in .env)",
        f"OPENAI_API_KEY      {status('OPENAI_API_KEY')}   (fallback)",
        f"taste org key       {org}",
        f"                    ({OP_REF}; used automatically when no key of your own is set)",
        f"HF_TOKEN            {status('HF_TOKEN')}   (Hugging Face fetches, e.g. rollout trajectories)",
        f"BIDIRECT_MODEL      {os.environ.get('BIDIRECT_MODEL', f'unset (default {DEFAULT_MODEL})')}",
        "inferred transforms run at temperature 0.0 with the DSPy cache on",
    ])
