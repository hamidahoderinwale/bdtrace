#!/usr/bin/env python3
"""
Run evaluation: grounding lift and divergence from baseline.

Loads traces, runs inferred representations with and without grounding,
computes grounding lift and divergence-from-baseline analyses.

Requires DSPy LM for inference: dspy.configure(lm=dspy.LM(...))

Usage:
    python eval/run_eval.py --from-swe-bench --limit 3
    python eval/run_eval.py --input traces.jsonl --analysis grounding-lift
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from representations import (
    behavioral_repr,
    file_edit_graph_repr,
    functional_repr,
    mechanistic_repr,
    semantic_edits_repr,
    tokens_repr,
)

from eval.analysis import divergence_from_baseline
from eval.metrics import grounding_check_score, grounding_lift


def _first_code_change(trace: dict) -> tuple[str, str, str | None]:
    """Extract before_content, after_content, file_path from first code_change event."""
    for event in trace.get("events", []):
        if not isinstance(event, dict):
            continue
        t = (event.get("type") or "").lower()
        if t not in ("code_change", "file_change", "entry_created"):
            continue
        details = event.get("details") or {}
        if not isinstance(details, dict):
            continue
        before = details.get("before_content") or ""
        after = details.get("after_content") or ""
        if before or after:
            return before, after, details.get("file_path")
    return "", "", None


def _run_inferred_for_trace(
    trace: dict,
    with_grounding: bool,
) -> dict:
    """Run behavioral, mechanistic, functional for one trace. with_grounding=False passes None."""
    before, after, file_path = _first_code_change(trace)
    if not before and not after:
        return {"behavioral": None, "mechanistic": None, "functional": None}

    cert = None
    module_ctx = None
    if with_grounding:
        cert = semantic_edits_repr(before, after, file_path)
        module_tokens = file_edit_graph_repr(trace)
        module_ctx = module_tokens if module_tokens else None

    behavioral = behavioral_repr(before, after, structural_certificate=cert)
    mechanistic = mechanistic_repr(before, after, structural_certificate=cert)
    functional = functional_repr(before, after, module_context=module_ctx)

    return {"behavioral": behavioral, "mechanistic": mechanistic, "functional": functional}


def main():
    parser = argparse.ArgumentParser(description="Run grounding lift and divergence analyses")
    parser.add_argument("--from-swe-bench", action="store_true", help="Load from SWE-bench Lite")
    parser.add_argument("--input", type=Path, help="JSONL of traces (from run_swe_bench)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--analysis",
        choices=["grounding-lift", "divergence", "all"],
        default="all",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    traces = []
    if args.from_swe_bench:
        from data.swe_bench import load_swe_bench_lite

        for t in load_swe_bench_lite(split="dev", limit=args.limit):
            traces.append(t)
    elif args.input and args.input.exists():
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "meta" in rec:
                        continue
                    if "events" in rec:
                        traces.append(rec)
                    elif "instance_id" in rec and rec.get("tokens") is not None:
                        traces.append(rec)
                except json.JSONDecodeError:
                    continue
    else:
        print("Provide --from-swe-bench or --input <jsonl>")
        sys.exit(1)

    if not traces:
        print("No traces loaded.")
        sys.exit(1)

    has_events = any(t.get("events") for t in traces)

    records_with = []
    records_without = []
    records_full = []

    for trace in traces:
        if has_events:
            with_ann = _run_inferred_for_trace(trace, with_grounding=True)
            without_ann = _run_inferred_for_trace(trace, with_grounding=False)
            tokens = tokens_repr(trace)
        else:
            with_ann = {"behavioral": None, "mechanistic": None, "functional": None}
            without_ann = with_ann
            tokens = trace.get("tokens", [])

        rec = {
            "instance_id": trace.get("instance_id"),
            "tokens": tokens,
            "behavioral": with_ann["behavioral"] if has_events else trace.get("behavioral"),
            "mechanistic": with_ann["mechanistic"] if has_events else trace.get("mechanistic"),
            "functional": with_ann["functional"] if has_events else trace.get("functional"),
        }
        records_full.append(rec)
        records_with.append(with_ann)
        records_without.append(without_ann)

    results = {}

    if args.analysis in ("grounding-lift", "all") and has_events:
        beh_with = [r["behavioral"] for r in records_with if r.get("behavioral")]
        beh_without = [r["behavioral"] for r in records_without if r.get("behavioral")]
        mech_with = [r["mechanistic"] for r in records_with if r["mechanistic"]]
        mech_without = [r["mechanistic"] for r in records_without if r["mechanistic"]]
        func_with = [r["functional"] for r in records_with if r["functional"]]
        func_without = [r["functional"] for r in records_without if r["functional"]]

        results["grounding_lift"] = {
            "behavioral": grounding_lift(beh_with, beh_without, "behavioral"),
            "mechanistic": grounding_lift(mech_with, mech_without, "mechanistic"),
            "functional": grounding_lift(func_with, func_without, "functional"),
        }
        results["grounding_scores_with"] = {
            "behavioral": grounding_check_score(beh_with, "behavioral"),
            "mechanistic": grounding_check_score(mech_with, "mechanistic"),
            "functional": grounding_check_score(func_with, "functional"),
        }
        results["grounding_scores_without"] = {
            "behavioral": grounding_check_score(beh_without, "behavioral"),
            "mechanistic": grounding_check_score(mech_without, "mechanistic"),
            "functional": grounding_check_score(func_without, "functional"),
        }
    elif args.analysis in ("grounding-lift", "all") and not has_events:
        results["grounding_lift"] = "skipped: no events (need traces with code_change events)"

    if args.analysis in ("divergence", "all"):
        div = divergence_from_baseline(
            records_full,
            baseline_key="tokens",
            structured_keys=["behavioral", "mechanistic", "functional"],
        )
        results["divergence_from_baseline"] = {
            "per_procedure": div["per_procedure"],
            "n_instances": len(div.get("per_instance", [])),
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote results to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
