#!/usr/bin/env python3
"""
Filter benchmark instances by structural edit pattern.

Takes a traces JSONL file and a target FIM pattern (set of edit types).
Finds instances whose edit certificate is a superset of the target pattern.

This is the building block for constructing structurally targeted evaluations:
given a known-hard pattern, find all instances that require it.

Usage:
  # Find instances matching the hardest FIM pattern (ease ~0.02):
  uv run python scripts/filter_by_structural_form.py \
    --traces output/resolved_traces_lite_full.jsonl \
    --pattern ADD_If ADD_Compare ADD_Constant ADD_Attribute ADD_Call ADD_Name

  # Run across all benchmarks for a given pattern:
  uv run python scripts/filter_by_structural_form.py \
    --traces output/resolved_traces_lite_full.jsonl \
              output/resolved_traces_verified_full.jsonl \
              output/resolved_traces_swe_smith.jsonl \
    --pattern ADD_If ADD_Compare ADD_Constant ADD_Attribute ADD_Call ADD_Name

  # Use a named preset pattern:
  uv run python scripts/filter_by_structural_form.py \
    --traces output/resolved_traces_lite_full.jsonl \
    --preset hard_compare

  # Save matching instance IDs to a file:
  uv run python scripts/filter_by_structural_form.py \
    --traces output/resolved_traces_swe_smith.jsonl \
    --preset hard_compare \
    --output output/hard_instance_training/matches_smith_hard_compare.json

  # Run all preset patterns across all available benchmarks:
  uv run python scripts/filter_by_structural_form.py --all-presets
"""

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence

ROOT = Path(__file__).resolve().parent.parent

# Normalization map -- must match build_canonical_forms.py exactly
_NORMALIZE_OPS = {
    "ADD_if": "ADD_If", "DEL_if": "DEL_If",
    "ADD_for": "ADD_For", "DEL_for": "DEL_For",
    "ADD_return": "ADD_Return", "DEL_return": "DEL_Return",
    "ADD_raise": "ADD_Raise", "DEL_raise": "DEL_Raise",
    "ADD_try": "ADD_Try", "DEL_try": "DEL_Try",
    "ADD_while": "ADD_While", "DEL_while": "DEL_While",
    "ADD_with": "ADD_With", "DEL_with": "DEL_With",
    "ADD_def": "ADD_FunctionDef", "DEL_def": "DEL_FunctionDef",
    "ADD_class": "ADD_ClassDef", "DEL_class": "DEL_ClassDef",
    "ADD_elif": "ADD_If", "DEL_elif": "DEL_If",
    "ADD_else": "ADD_If", "DEL_else": "DEL_If",
    "ADD_except": "ADD_ExceptHandler", "DEL_except": "DEL_ExceptHandler",
    "ADD_assert": "ADD_Assert",
}

# Preset patterns from the FIM difficulty analysis (fim_difficulty_summary.json).
# Named by what makes them structurally distinct.
PRESET_PATTERNS = {
    "hard_compare": {
        "description": "Conditional + comparison + attribute access (ease ~0.02, n=7 in Lite)",
        "pattern": frozenset({
            "ADD_If", "ADD_Compare", "ADD_Constant",
            "ADD_Attribute", "ADD_Call", "ADD_Name",
        }),
    },
    "hard_expr": {
        "description": "Conditional + expression + attribute access (ease ~0.02, n=5 in Lite)",
        "pattern": frozenset({
            "ADD_If", "ADD_Expr", "ADD_Constant",
            "ADD_Attribute", "ADD_Call", "ADD_Name",
        }),
    },
    "easy_return": {
        "description": "Return value change (ease=0.67, n=5 in Lite)",
        "pattern": frozenset({"ADD_Return", "DEL_Return"}),
    },
    "easy_assign": {
        "description": "Variable assignment swap (ease=0.81, n=5 in Lite)",
        "pattern": frozenset({
            "ADD_Assign", "ADD_Constant", "ADD_Name",
            "DEL_Assign", "DEL_Constant", "DEL_Name",
        }),
    },
    "medium_branch_mod": {
        "description": "Conditional branch modification (ease=0.40, n=32 in Lite)",
        "pattern": frozenset({"ADD_If", "DEL_If"}),
    },
    "medium_branch_add": {
        "description": "Conditional branch + return addition (ease=0.26, n=12 in Lite)",
        "pattern": frozenset({"ADD_If", "ADD_Return"}),
    },
    "hard_full_replace": {
        "description": "Full node replacement (ease=0.27, n=30 in Lite)",
        "pattern": frozenset({
            "ADD_Assign", "ADD_Attribute", "ADD_Call", "ADD_Constant", "ADD_Name",
            "DEL_Assign", "DEL_Attribute", "DEL_Call", "DEL_Constant", "DEL_Name",
        }),
    },
}

# Default trace files to scan when --all-presets is used
DEFAULT_TRACES = [
    ROOT / "output" / "resolved_traces_lite_full.jsonl",
    ROOT / "output" / "resolved_traces_verified_full.jsonl",
    ROOT / "output" / "resolved_traces_swe_smith.jsonl",
]


def load_certs(traces_path: Path) -> dict[str, frozenset[str]]:
    """Load edit certificates from a traces JSONL file.

    Each instance gets a frozenset of normalized AST edit operations
    extracted from its oracle patch.
    """
    certs: dict[str, frozenset[str]] = {}
    with open(traces_path) as f:
        for line in f:
            trace = json.loads(line)
            ops: list[str] = []
            for ev in trace["events"]:
                if ev["type"] != "code_change":
                    continue
                d = ev["details"]
                if not d["file_path"].endswith(".py"):
                    continue
                before = d["before_content"].splitlines(keepends=True)
                after = d["after_content"].splitlines(keepends=True)
                raw = "".join(difflib.unified_diff(
                    before, after, fromfile=d["file_path"], tofile=d["file_path"],
                ))
                if not raw:
                    continue
                diff = f"diff --git a/{d['file_path']} b/{d['file_path']}\n" + raw
                ops.extend(patch_to_ast_sequence(diff))
            if ops:
                norm = frozenset(_NORMALIZE_OPS.get(op, op) for op in ops)
                certs[trace["instance_id"]] = norm
    return certs


def filter_by_pattern(
    certs: dict[str, frozenset[str]],
    target: frozenset[str],
) -> list[str]:
    """Return instance IDs whose certificate is a superset of the target pattern."""
    return sorted(iid for iid, cert in certs.items() if target.issubset(cert))


def benchmark_label(path: Path) -> str:
    """Human-readable label from the trace filename."""
    name = path.stem
    if "lite" in name:
        return "SWE-bench Lite"
    if "verified" in name:
        return "SWE-bench Verified"
    if "smith" in name:
        return "SWE-smith"
    return name


def run_filter(
    traces_paths: list[Path],
    target: frozenset[str],
    pattern_name: str | None = None,
) -> dict:
    """Run the filter across one or more trace files. Returns a summary dict."""
    label = pattern_name or " + ".join(sorted(target))
    results: dict[str, dict] = {}

    for path in traces_paths:
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue

        bench = benchmark_label(path)
        certs = load_certs(path)
        matches = filter_by_pattern(certs, target)

        results[bench] = {
            "trace_file": str(path.relative_to(ROOT)),
            "total_instances": len(certs),
            "matching_instances": len(matches),
            "match_rate": len(matches) / len(certs) if certs else 0,
            "instance_ids": matches,
        }

        pct = 100 * results[bench]["match_rate"]
        print(f"  {bench:25s}  {len(matches):5d} / {len(certs):5d}  ({pct:5.1f}%)")

    return {
        "pattern_name": label,
        "pattern": sorted(target),
        "benchmarks": results,
    }


def run_all_presets() -> list[dict]:
    """Run all preset patterns across all available trace files."""
    available = [p for p in DEFAULT_TRACES if p.exists()]
    if not available:
        print("No trace files found. Expected files in output/.")
        return []

    all_results = []
    for name, info in PRESET_PATTERNS.items():
        print(f"\n--- {name}: {info['description']} ---")
        print(f"    pattern: {sorted(info['pattern'])}")
        result = run_filter(available, info["pattern"], pattern_name=name)
        all_results.append(result)

    return all_results


def print_summary_table(all_results: list[dict]) -> None:
    """Print a compact summary table across all patterns and benchmarks."""
    if not all_results:
        return

    # Collect all benchmark names
    bench_names = []
    for r in all_results:
        for b in r["benchmarks"]:
            if b not in bench_names:
                bench_names.append(b)

    # Header
    col_w = 18
    print(f"\n{'Pattern':<22s}", end="")
    for b in bench_names:
        print(f"  {b:>{col_w}s}", end="")
    print()
    print("-" * (22 + (col_w + 2) * len(bench_names)))

    # Rows
    for r in all_results:
        name = r["pattern_name"]
        print(f"{name:<22s}", end="")
        for b in bench_names:
            if b in r["benchmarks"]:
                info = r["benchmarks"][b]
                cell = f"{info['matching_instances']}/{info['total_instances']}"
                print(f"  {cell:>{col_w}s}", end="")
            else:
                print(f"  {'--':>{col_w}s}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Filter benchmark instances by structural edit pattern.",
    )
    parser.add_argument(
        "--traces", nargs="+", type=Path,
        help="Path(s) to traces JSONL file(s)",
    )
    parser.add_argument(
        "--pattern", nargs="+",
        help="Target FIM pattern as space-separated edit types (e.g. ADD_If ADD_Compare)",
    )
    parser.add_argument(
        "--preset", choices=list(PRESET_PATTERNS.keys()),
        help="Use a named preset pattern instead of --pattern",
    )
    parser.add_argument(
        "--all-presets", action="store_true",
        help="Run all preset patterns across all available trace files",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Save matching instance IDs to this JSON file",
    )
    args = parser.parse_args()

    if args.all_presets:
        all_results = run_all_presets()
        print_summary_table(all_results)

        # Save full results
        out_path = ROOT / "output" / "hard_instance_training" / "pattern_coverage.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved full results to {out_path.relative_to(ROOT)}")
        return

    # Need either --pattern or --preset, plus at least one --traces
    if not args.traces:
        parser.error("--traces is required (unless using --all-presets)")
    if not args.pattern and not args.preset:
        parser.error("Provide --pattern or --preset")

    if args.preset:
        target = PRESET_PATTERNS[args.preset]["pattern"]
        pattern_name = args.preset
    else:
        target = frozenset(args.pattern)
        pattern_name = None

    print(f"Pattern: {sorted(target)}")
    result = run_filter(args.traces, target, pattern_name=pattern_name)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
