#!/usr/bin/env python3
"""Investigate SWE-Smith edit-ops that do not appear in Lite.

1. Build full Lite vocab from resolved_traces_lite_full.jsonl.
2. Build full SWE-Smith vocab from resolved_traces_swe_smith.jsonl
   (first 5000 traces — same cap as the transfer experiment).
3. Compute swe_smith_vocab - lite_vocab.
4. For each novel op:
   - Count traces in which it occurs.
   - Sample up to 3 instance ids.
   - For one instance: emit a snippet from the first code_change event that
     fires this op (first 10 lines of the unified diff).
   - Classify automatically by heuristics:
       synthetic_artifact: op surfaces on obviously-synthetic repo names
           (SWE-bench__* prefix + non-Python or stdlib-only contents).
       normalization_miss: op is already present in Lite after casefold
           normalization, or is a case-variant of an existing op
           (e.g. ADD_if vs ADD_If).
       rare_ast_node: op maps to an uncommon but real Python AST node
           (e.g. AsyncFunctionDef, MatchValue, TypeAlias, NamedExpr,
            FormattedValue, JoinedStr, Slice, Starred, Yield, YieldFrom).
       genuine_novel: none of the above apply and the op has >= 3
           independent SWE-Smith occurrences.
       unclear: op has <3 occurrences and none of the heuristics fire.
5. Writes output/pdiff_smoke_test/novel_ops_investigation.json and prints
   a small summary table.

Usage:
    python -m scripts.figures.investigate_novel_ops
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from analysis.pdiff import view_from_trace

ROOT = Path(__file__).resolve().parents[2]
LITE = ROOT / "output" / "resolved_traces_lite_full.jsonl"
SWE_SMITH = ROOT / "output" / "resolved_traces_swe_smith.jsonl"
OUT = ROOT / "output" / "pdiff_smoke_test" / "novel_ops_investigation.json"

SWE_SMITH_CAP = 5000

# AST nodes that are real Python but rarely used.
RARE_AST_NODES = {
    "AsyncFunctionDef", "AsyncFor", "AsyncWith", "Await",
    "NamedExpr",       # walrus operator (PEP 572)
    "Match", "MatchAs", "MatchClass", "MatchMapping", "MatchOr",
    "MatchSequence", "MatchSingleton", "MatchStar", "MatchValue",
    "TypeAlias", "TypeVar", "ParamSpec", "TypeVarTuple",
    "FormattedValue", "JoinedStr",  # f-string internals
    "Starred", "YieldFrom",
    "IfExp",          # conditional expression x if y else z
    "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Slice", "ExtSlice", "Index",  # subscript pieces
    "Lambda",
    "Nonlocal", "Global",
    "AnnAssign", "AugAssign",
}


def _iter_traces(path: Path, cap: int | None = None):
    count = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield trace
            count += 1
            if cap is not None and count >= cap:
                return


def _repo_is_synthetic(repo: str | None) -> bool:
    if not repo:
        return False
    # SWE-bench-format synthetic repos used by SWE-Smith look like
    # "swesmith/<repo>" or carry a "SWE-bench__" prefix.
    low = repo.lower()
    return "swesmith" in low or repo.startswith("SWE-bench__")


def _first_diff_snippet(trace: dict, op: str, lines: int = 10) -> str:
    """Return the first ~N lines of a unified-diff-like snippet for the code
    change event where `op` fires. Best-effort: scans before/after content and
    truncates.
    """
    import difflib
    events = trace.get("events") or []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") != "code_change":
            continue
        d = ev.get("details") or {}
        before = d.get("before_content") or ""
        after = d.get("after_content") or ""
        fp = d.get("file_path", "")
        if before == after:
            continue
        raw_lines = list(difflib.unified_diff(
            before.splitlines(keepends=False),
            after.splitlines(keepends=False),
            fromfile=fp, tofile=fp, lineterm="", n=2,
        ))
        snippet = "\n".join(raw_lines[:lines])
        if snippet:
            return snippet
    return "(no code_change event with usable before/after content)"


def _ast_node_of(op: str) -> str | None:
    m = re.match(r"^(ADD|DEL)_(.+)$", op)
    if not m:
        return None
    return m.group(2)


def _classify(
    op: str,
    occurrences: int,
    sample_repos: list[str],
    lite_vocab_lower: set[str],
    lite_vocab: set[str],
) -> tuple[str, str]:
    """Return (classification, reasoning).

    Heuristic order (first match wins):
      1. Case-variant of an existing Lite op -> normalization_miss.
         (Lite + SWE-Smith should normalize through the same map; when they
         don\'t, it\'s a coverage gap in _NORMALIZE_OPS, not a new pattern.)
      2. Rare but real AST node (NamedExpr, Lambda, AsyncFunctionDef,
         AnnAssign, YieldFrom, etc.) -> rare_ast_node.
      3. Direction-flipped sibling (op is DEL_X or ADD_X and the opposite
         direction ADD_X / DEL_X is already in Lite) -> synthetic_artifact.
         SWE-Smith generates bugs by reversing fixes, so deletions of
         ClassDef / While / Break / async that aren\'t in Lite often surface
         because Lite only holds the forward fix direction.
      4. >=3 occurrences and none of the above -> genuine_novel.
      5. Otherwise -> unclear.
    """
    node = _ast_node_of(op)

    # (1) Case-variant already in Lite under casefold.
    if op.casefold() in lite_vocab_lower and op not in lite_vocab:
        return ("normalization_miss",
                f"Case-variant: \'{op}\' matches a Lite op under casefold")

    # (2) Rare AST node heuristic.
    if node and node in RARE_AST_NODES:
        return ("rare_ast_node",
                f"Maps to real Python AST node \'{node}\' (uncommon)")

    # (3) Direction-flipped sibling already in Lite.
    if op.startswith("ADD_") and node:
        sibling = f"DEL_{node}"
        if sibling in lite_vocab:
            return ("synthetic_artifact",
                    f"Direction-flipped: Lite has \'{sibling}\' but not \'{op}\'; "
                    "SWE-Smith reverses fixes to synthesize bugs")
    if op.startswith("DEL_") and node:
        sibling = f"ADD_{node}"
        if sibling in lite_vocab:
            return ("synthetic_artifact",
                    f"Direction-flipped: Lite has \'{sibling}\' but not \'{op}\'; "
                    "SWE-Smith reverses fixes to synthesize bugs")

    # (4) Otherwise if occurs often, call it genuine.
    if occurrences >= 3:
        return ("genuine_novel",
                f"{occurrences} occurrences, AST node \'{node}\' not rare, "
                "no direction-flip sibling in Lite")

    return ("unclear",
            f"Only {occurrences} occurrences; no heuristic fires")


def main() -> int:
    print("Scanning Lite for full edit-op vocabulary...")
    lite_vocab: set[str] = set()
    for trace in _iter_traces(LITE):
        view = view_from_trace(trace)
        lite_vocab.update(view.edits)
    print(f"  Lite vocab: {len(lite_vocab)} ops")

    print(f"Scanning SWE-Smith (cap {SWE_SMITH_CAP}) for edit-op occurrences...")
    op_traces: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for trace in _iter_traces(SWE_SMITH, cap=SWE_SMITH_CAP):
        view = view_from_trace(trace)
        iid = trace.get("instance_id") or ""
        repo = trace.get("repo")
        for op in view.edits:
            op_traces[op].append((iid, repo))
    swe_vocab = set(op_traces)
    print(f"  SWE-Smith vocab: {len(swe_vocab)} ops")

    novel_ops = sorted(swe_vocab - lite_vocab)
    print(f"  Novel (in SWE-Smith, not in Lite): {len(novel_ops)}")

    # For each novel op, grab up to 3 sample instance ids and their repos,
    # then pull a diff snippet from the first one.
    lite_vocab_lower = {o.casefold() for o in lite_vocab}
    iid_to_trace: dict[str, dict] = {}
    # Re-scan SWE-Smith once more to resolve sample iids to traces (keep memory bounded).
    sample_iids_needed: set[str] = set()
    per_op_sample: dict[str, list[tuple[str, str | None]]] = {}
    for op in novel_ops:
        occ = op_traces[op]
        sample = occ[:3]
        per_op_sample[op] = sample
        sample_iids_needed.update(i for i, _ in sample)

    print(f"  Fetching {len(sample_iids_needed)} sample traces for diff snippets...")
    for trace in _iter_traces(SWE_SMITH, cap=SWE_SMITH_CAP):
        iid = trace.get("instance_id") or ""
        if iid in sample_iids_needed and iid not in iid_to_trace:
            iid_to_trace[iid] = trace
        if len(iid_to_trace) == len(sample_iids_needed):
            break

    ops_out = []
    summary = {"genuine_novel": 0, "synthetic_artifact": 0,
               "normalization_miss": 0, "rare_ast_node": 0, "unclear": 0}
    for op in novel_ops:
        occurrences = len(op_traces[op])
        sample = per_op_sample[op]
        sample_iids = [i for i, _ in sample]
        sample_repos = [r for _, r in sample if r]

        snippet = "(no sample trace found)"
        for iid, _ in sample:
            tr = iid_to_trace.get(iid)
            if tr is not None:
                snippet = _first_diff_snippet(tr, op)
                break

        cls, reason = _classify(op, occurrences, sample_repos,
                                lite_vocab_lower, lite_vocab)
        summary[cls] = summary.get(cls, 0) + 1

        ops_out.append({
            "op": op,
            "occurrences": occurrences,
            "sample_instance_ids": sample_iids,
            "sample_repos": sample_repos,
            "sample_diff_snippet": snippet,
            "classification": cls,
            "reasoning": reason,
        })

    # Sort output by occurrence count desc for readability.
    ops_out.sort(key=lambda r: -r["occurrences"])

    results = {
        "n_lite_ops": len(lite_vocab),
        "n_swe_smith_ops": len(swe_vocab),
        "n_novel_ops": len(novel_ops),
        "ops": ops_out,
        "summary": summary,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT}")

    print("\n=== Novel SWE-Smith edit-op classification ===")
    print(f"{'op':<28s} {'count':>6s}  {'class':<20s} repos")
    for row in ops_out:
        repos = ",".join(sorted({r for r in row["sample_repos"] if r}))[:40]
        print(f"{row['op']:<28s} {row['occurrences']:>6d}  "
              f"{row['classification']:<20s} {repos}")
    print()
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k:<22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
