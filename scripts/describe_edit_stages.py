#!/usr/bin/env python3
"""
Generate staged edit narratives for SWE-bench patches.

For each instance, parses the patch into @@ hunks, extracts per-chunk
AST sequences, and generates one grounded sentence per chunk via DSPy.

Output per instance:
    {
      "instance_id": "astropy__astropy-14182",
      "fix_type": "api_change",
      "n_chunks": 3,
      "chunks": [
        {
          "header": "@@ -27,7 +27,6 @@",
          "file": "astropy/io/ascii/rst.py",
          "sequence": ["DEL_Assign"],
          "description": "Removed hardcoded start_line class attribute."
        },
        ...
      ],
      "staged_narrative": "Removed hardcoded start_line class attribute. Replaced example block with executable doctest. Added header_rows parameter support to the write method."
    }

Usage:
    uv run python scripts/describe_edit_stages.py --limit 5
    uv run python scripts/describe_edit_stages.py --output output/staged_descriptions.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_paths = [
    Path(__file__).resolve().parent.parent / ".venv" / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]
for _p in _env_paths:
    if _p.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_p)
        except ImportError:
            pass
        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--fix-types", type=Path,
                        default=Path("output/datasets/swe_bench_lite_resolved/fix_types.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/staged_descriptions.json"))
    args = parser.parse_args()

    from configs.dspy_config import configure_dspy
    configure_dspy(model=args.model)

    from datasets import load_dataset
    from analysis.procedures.ast_edit_sequences import patch_to_chunks
    from representations.inferred.fix_type import ChunkDescriber

    print("Loading patches from HuggingFace...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    rows = list(ds)
    if args.limit:
        rows = rows[:args.limit]

    ft_map: dict[str, str] = {}
    if args.fix_types.exists():
        with open(args.fix_types) as f:
            ft_data = json.load(f)
        ft_map = {r["instance_id"]: r["fix_type"] for r in ft_data["results"]}

    describer = ChunkDescriber()
    results = []

    for row in tqdm.tqdm(rows, desc="Describing edit stages"):
        iid = row["instance_id"]
        patch = row.get("patch", "") or ""
        if not patch.strip():
            continue

        chunks = patch_to_chunks(patch)
        non_empty = [c for c in chunks if not c.is_empty]
        if not non_empty:
            continue

        chunk_records = []
        descriptions = []
        for chunk in non_empty:
            try:
                desc = describer(chunk)
            except Exception as e:
                desc = f"(error: {e})"
            descriptions.append(desc)
            chunk_records.append({
                "header": chunk.header,
                "file": chunk.file_path,
                "sequence": chunk.sequence,
                "removed_lines": chunk.removed_lines[:4],
                "added_lines": chunk.added_lines[:4],
                "description": desc,
            })

        results.append({
            "instance_id": iid,
            "fix_type": ft_map.get(iid, "unknown"),
            "n_chunks": len(non_empty),
            "chunks": chunk_records,
            "staged_narrative": " ".join(descriptions),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_instances": len(results),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(results)} staged narratives → {args.output}")


if __name__ == "__main__":
    main()
