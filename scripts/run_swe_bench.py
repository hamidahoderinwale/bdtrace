#!/usr/bin/env python3
"""
Run representation pipeline on SWE-bench / SWE-bench Lite.

Fetches from Hugging Face, converts to trace format, applies representations,
and optionally exports to JSON/JSONL.

Usage:
    python scripts/run_swe_bench.py --dataset lite --split dev --limit 5 --output traces.jsonl
    python scripts/run_swe_bench.py --dataset full --split test --rung tokens --output tokens.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.swe_bench import load_swe_bench, load_swe_bench_lite
from representations import (
    file_edit_graph_repr,
    functions_repr,
    motifs_repr,
    raw_repr,
    semantic_edits_repr,
    tokens_repr,
)

# Trace-based: use file_edit_graph_repr (co-edit from events).
# Repo-based: use module_graph_repr(repo_path, commit, touched_files) for full import+coedit.
RUNG_FUNCS = {
    "raw": raw_repr,
    "tokens": tokens_repr,
    "edits": semantic_edits_repr,
    "functions": functions_repr,
    "modules": file_edit_graph_repr,
    "motifs": motifs_repr,
}


def main():
    parser = argparse.ArgumentParser(description="Run representations on SWE-bench")
    parser.add_argument("--dataset", choices=["lite", "full"], default="lite")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rung", choices=list(RUNG_FUNCS), default=None, help="Single rung or all if omitted")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    args = parser.parse_args()

    loader = load_swe_bench_lite if args.dataset == "lite" else load_swe_bench
    rungs = [args.rung] if args.rung else list(RUNG_FUNCS)

    records = []
    for trace in loader(split=args.split, limit=args.limit):
        rec = {
            "instance_id": trace.get("instance_id"),
            "repo": trace.get("repo"),
            "base_commit": trace.get("base_commit"),
        }
        for rung in rungs:
            func = RUNG_FUNCS[rung]
            rec[rung] = func(trace)
        records.append(rec)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            if args.format == "jsonl":
                for r in records:
                    f.write(json.dumps(r, default=str) + "\n")
            else:
                json.dump(records, f, indent=2, default=str)
        print(f"Wrote {len(records)} records to {args.output}")
    else:
        for i, r in enumerate(records[:3]):
            print(f"--- Instance {i} ({r.get('instance_id')}) ---")
            for rung in rungs:
                val = r.get(rung)
                if isinstance(val, list) and len(val) > 10:
                    print(f"  {rung}: {val[:5]}... ({len(val)} items)")
                else:
                    print(f"  {rung}: {val}")


if __name__ == "__main__":
    main()
