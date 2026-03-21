#!/usr/bin/env python3
"""
Run evaluation: inferred representations and divergence from baseline.

Loads traces, runs inferred representations (behavioral, mechanistic, functional)
with structural grounding, computes divergence-from-baseline analysis.

Requires OPENROUTER_API_KEY or OPENAI_API_KEY (configs.dspy_config).
Optional: DSPY_MODEL, DSPY_TEMPERATURE, DSPY_MAX_TOKENS.

Usage:
    python eval/run_eval.py --from-swe-bench --limit 3
    python eval/run_eval.py --input traces.jsonl --analysis divergence
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from project root or .venv before DSPy config
_env_paths = [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]
for p in _env_paths:
    if p.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(p)
        except ImportError:
            pass
        break

from configs.dspy_config import configure_dspy

from representations import (
    behavioral_repr,
    file_edit_graph_repr,
    functional_repr,
    mechanistic_repr,
    semantic_edits_repr,
    tokens_repr,
)

from eval.analysis import divergence_from_baseline


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


_CONTENT_CHAR_LIMIT = 20_000


def _truncate(text: str, limit: int = _CONTENT_CHAR_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _run_inferred_for_trace(trace: dict) -> dict:
    """Run behavioral, mechanistic, functional for one trace with structural grounding."""
    before, after, file_path = _first_code_change(trace)
    if not before and not after:
        return {"behavioral": None, "mechanistic": None, "functional": None, "edits": None}

    before = _truncate(before)
    after = _truncate(after)

    cert = semantic_edits_repr(before, after, file_path)
    module_tokens = file_edit_graph_repr(trace)
    module_ctx = module_tokens if module_tokens else None

    behavioral = behavioral_repr(before, after, structural_certificate=cert)
    mechanistic = mechanistic_repr(before, after, structural_certificate=cert)
    functional = functional_repr(before, after, module_context=module_ctx)

    edits = [cert] if isinstance(cert, dict) else (cert if isinstance(cert, list) else [])
    return {"behavioral": behavioral, "mechanistic": mechanistic, "functional": functional, "edits": edits}


def main():
    parser = argparse.ArgumentParser(description="Run inferred representations and divergence analysis")
    parser.add_argument("--from-swe-bench", action="store_true", help="Load from SWE-bench Lite")
    parser.add_argument("--input", type=Path, help="JSONL of traces (from run_swe_bench)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--analysis",
        choices=["divergence", "all"],
        default="all",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-records", type=Path, default=None, help="Save records (with behavioral/mechanistic/functional) for procedure divergence")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for output path (e.g. swe_bench_verified_resolved_multifile)")
    args = parser.parse_args()

    # Default output paths when dataset given
    if args.dataset:
        eval_dir = Path("output") / "datasets" / args.dataset / "eval"
        if args.save_records is None:
            args.save_records = eval_dir / "records_with_behavioral.json"
        if args.output is None:
            args.output = eval_dir / "divergence_results.json"

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
    if has_events and not configure_dspy():
        print("DSPy LM not configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    records_full = []

    for trace in traces:
        if has_events:
            ann = _run_inferred_for_trace(trace)
            tokens = tokens_repr(trace)
        else:
            ann = {"behavioral": None, "mechanistic": None, "functional": None, "edits": None}
            tokens = trace.get("tokens", [])

        rec = {
            "instance_id": trace.get("instance_id"),
            "tokens": tokens,
            "behavioral": ann["behavioral"] if has_events else trace.get("behavioral"),
            "mechanistic": ann["mechanistic"] if has_events else trace.get("mechanistic"),
            "functional": ann["functional"] if has_events else trace.get("functional"),
            "edits": ann.get("edits") if has_events else trace.get("edits"),
        }
        records_full.append(rec)

    results = {}

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

    if args.save_records:
        args.save_records.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_records, "w") as f:
            json.dump(records_full, f, indent=2, default=str)
        print(f"Wrote {len(records_full)} records to {args.save_records}")


if __name__ == "__main__":
    main()
