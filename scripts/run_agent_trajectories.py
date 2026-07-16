#!/usr/bin/env python3
"""
Run representation pipeline on agent trajectories.

Loads from Hugging Face (e.g. nebius/SWE-agent-trajectories), converts to trace
format with intermediate states, applies representations, and optionally exports.

Usage:
    python scripts/run_agent_trajectories.py --limit 5 --output traces.jsonl
    python scripts/run_agent_trajectories.py --dataset nebius/SWE-agent-trajectories --rung tokens
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.agent_trajectories import load_agent_trajectories
from representations import (
    file_edit_graph_repr,
    functions_repr,
    motifs_repr,
    raw_repr,
    semantic_edits_repr,
    tokens_repr,
)

RUNG_FUNCS = {
    "raw": raw_repr,
    "tokens": tokens_repr,
    "edits": semantic_edits_repr,
    "functions": functions_repr,
    "modules": file_edit_graph_repr,
    "motifs": motifs_repr,
}


def main():
    parser = argparse.ArgumentParser(description="Run representations on agent trajectories")
    parser.add_argument(
        "--dataset",
        default="nebius/SWE-agent-trajectories",
        help="Hugging Face dataset ID",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--rung",
        choices=list(RUNG_FUNCS),
        default=None,
        help="Single rung or all if omitted",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    args = parser.parse_args()

    rungs = [args.rung] if args.rung else list(RUNG_FUNCS)

    records = []
    for trace in load_agent_trajectories(
        dataset_id=args.dataset,
        split=args.split,
        limit=args.limit,
    ):
        rec = {
            "instance_id": trace.get("instance_id"),
            "model_name": trace.get("model_name"),
            "target": trace.get("target"),
            "exit_status": trace.get("exit_status"),
            "event_count": len(trace.get("events", [])),
        }
        for rung in rungs:
            func = RUNG_FUNCS[rung]
            try:
                rec[rung] = func(trace)
            except (TypeError, KeyError, ValueError) as e:
                rec[rung] = None
                rec[f"{rung}_error"] = str(e)
        records.append(rec)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        meta = {"timestamp_utc": datetime.now(UTC).isoformat(), "count": len(records)}
        with open(args.output, "w") as f:
            if args.format == "jsonl":
                f.write(json.dumps({"meta": meta}, default=str) + "\n")
                for r in records:
                    f.write(json.dumps(r, default=str) + "\n")
            else:
                json.dump({"meta": meta, "records": records}, f, indent=2, default=str)
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
