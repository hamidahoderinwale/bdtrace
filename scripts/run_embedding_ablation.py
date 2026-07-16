#!/usr/bin/env python3
"""
Run embedding ablation: is structure part of the basis of representation?

Runs behavioral with (a) edits only, (b) code only, (c) both.
Compares claim embeddings: sim(emb_a, emb_c), sim(emb_b, emb_c), sim(emb_a, emb_b).

Requires OPENAI_API_KEY (DSPy). Uses sentence-transformers for embedding.

Usage:
  python scripts/run_embedding_ablation.py --dataset swe_bench_verified_resolved_multifile --limit 5
  python scripts/run_embedding_ablation.py --input output/resolved_traces_verified_multifile.jsonl --limit 10 --output output/datasets/swe_bench_verified_resolved_multifile/embedding_ablation.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for p in [Path(__file__).resolve().parent.parent / ".venv" / ".env", Path(__file__).resolve().parent.parent / ".env"]:
    if p.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(p)
        except ImportError:
            pass
        break

from configs.dspy_config import configure_dspy
from representations import behavioral_repr, semantic_edits_repr_source
from analysis.embedding_ablation.structure_basis import run_embedding_ablation


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Embedding ablation: structure as basis")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. swe_bench_verified_resolved_multifile). Infers input/output paths.")
    parser.add_argument("--input", type=Path, help="JSONL of traces (resolved)")
    parser.add_argument("--from-swe-bench", action="store_true", help="Load from SWE-bench Lite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Base output dir for dataset-centric paths")
    args = parser.parse_args()

    if args.dataset:
        base = args.output_dir / "datasets" / args.dataset
        if args.input is None:
            jsonl_name = "resolved_traces_verified_multifile.jsonl" if "verified" in args.dataset else "resolved_traces_multifile.jsonl"
            args.input = args.output_dir / jsonl_name
        if args.output is None:
            args.output = base / "embedding_ablation.json"

    traces = []
    if args.from_swe_bench:
        from data.swe_bench import load_swe_bench_lite
        for t in load_swe_bench_lite(split="test", limit=args.limit):
            traces.append(t)
    elif args.input and args.input.exists():
        with open(args.input) as f:
            for i, line in enumerate(f):
                if args.limit and i >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        print("Provide --input <jsonl> or --from-swe-bench", file=sys.stderr)
        sys.exit(1)

    if not traces:
        print("No traces loaded.", file=sys.stderr)
        sys.exit(1)

    if not configure_dspy():
        print("DSPy not configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    records = []
    for trace in traces:
        before, after, file_path = _first_code_change(trace)
        if not before and not after:
            continue
        cert = semantic_edits_repr_source(before, after, file_path)
        records.append({
            "instance_id": trace.get("instance_id") or trace.get("repo") or "unknown",
            "before_fn": before,
            "after_fn": after,
            "structural_certificate": cert,
        })

    if not records:
        print("No records with code changes.", file=sys.stderr)
        sys.exit(1)

    print(f"Running ablation on {len(records)} traces...", file=sys.stderr)
    results = run_embedding_ablation(records, behavioral_repr)

    out = json.dumps(results, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(out)

    agg = results.get("aggregate", {})
    if agg:
        print(f"\nAggregate: sim_ac={agg.get('mean_sim_ac', 0):.3f} sim_bc={agg.get('mean_sim_bc', 0):.3f} sim_ab={agg.get('mean_sim_ab', 0):.3f} structure_dominates={agg.get('structure_dominates', False)}", file=sys.stderr)


if __name__ == "__main__":
    main()
