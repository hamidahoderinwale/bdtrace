"""Fetch raw SWE-agent .traj files from the public S3 submissions bucket.

Why: the pre-extracted parquets in output/trajectories/ contain only the
8-token hand-coded DSL + file lists. For Paper 2's canonicalize-then-BPE
methodology we need the raw action strings ("pip install -e .[dev]",
"edit 1:1\\n...", etc.) that live in the original .traj JSON files.

What this script does:
  1. For each of the 3 committed SWE-agents, fetch the raw .traj for every
     SWE-bench Lite instance.
  2. Cache locally at output/trajectories/.cache/{model_id}/{instance_id}.json.
  3. Produce a manifest.json documenting the fetch run (timestamps, URLs,
     file counts, sizes, sha256 hashes, any failures).
  4. Write a human-readable NOTE.md in the cache directory explaining what
     was fetched, why, and how it relates to existing parquets.
  5. Idempotent: skips files already cached.

Source: https://swe-bench-submissions.s3.amazonaws.com (public).

Usage:
    python -m analysis.preferences.fetch_raw_trajectories
    python -m analysis.preferences.fetch_raw_trajectories --limit 10  # quick test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.swebench_trajectories import fetch_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "output" / "trajectories" / ".cache"

# The three target agents committed for Paper 2 action-layer analysis.
AGENTS = [
    "20240402_sweagent_gpt4",
    "20240620_sweagent_claude3.5sonnet",
    "20240728_sweagent_gpt4o",
]
SPLIT = "lite"


def load_lite_instance_ids() -> list[str]:
    """Load SWE-bench Lite instance_ids from the HF dataset."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return [str(x) for x in ds["instance_id"]]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one_with_provenance(
    instance_id: str,
    model_id: str,
) -> dict[str, Any]:
    """Fetch a single trajectory; return provenance record."""
    cache_path = CACHE_DIR / model_id / f"{instance_id}.json"

    if cache_path.exists():
        return {
            "instance_id": instance_id,
            "model_id": model_id,
            "status": "cached",
            "path": str(cache_path.relative_to(PROJECT_ROOT)),
            "size_bytes": cache_path.stat().st_size,
            "sha256": sha256_of_file(cache_path),
        }

    t0 = time.time()
    data = fetch_trajectory(instance_id, model_id, SPLIT, CACHE_DIR)
    elapsed = time.time() - t0

    if data is None:
        return {
            "instance_id": instance_id,
            "model_id": model_id,
            "status": "failed",
            "elapsed_sec": round(elapsed, 2),
        }

    return {
        "instance_id": instance_id,
        "model_id": model_id,
        "status": "fetched",
        "path": str(cache_path.relative_to(PROJECT_ROOT)),
        "size_bytes": cache_path.stat().st_size,
        "sha256": sha256_of_file(cache_path),
        "elapsed_sec": round(elapsed, 2),
    }


def write_note(cache_dir: Path, run_metadata: dict) -> None:
    """Write a human-readable NOTE.md in the cache directory."""
    note_path = cache_dir / "NOTE.md"
    content = f"""# Raw trajectory cache

## What is this?

Cached raw `.traj` JSON files fetched from the public SWE-bench submissions
S3 bucket. These are the full agent trajectories (steps, observations,
responses, state, thoughts) — the unabridged source that the pre-extracted
parquets in `output/trajectories/*.parquet` were derived from.

## Why do we have both this and the parquets?

The parquets collapse each trajectory's action sequence into an 8-token
hand-coded DSL (`CREATE, EDIT, NAV, OPEN, OTHER, RUN, SEARCH, SUBMIT`). For
Paper 2's canonicalize-then-BPE methodology we need the raw action strings
(e.g. `"pip install -e .[dev]"`, `"edit 1:1\\n...\\nend_of_edit"`) to apply
richer canonicalization. The parquets remain useful for feature counts and
are unchanged by this fetch.

## What's in this directory?

```
{cache_dir.name}/
  20240402_sweagent_gpt4/
    astropy__astropy-12907.json
    astropy__astropy-14365.json
    ...
  20240620_sweagent_claude3.5sonnet/
    ...
  20240728_sweagent_gpt4o/
    ...
  manifest.json            # per-file provenance (timestamps, sizes, sha256)
  NOTE.md                  # this file
```

## Fetch run metadata

- Run date: {run_metadata['run_started_at']}
- Source: `https://swe-bench-submissions.s3.amazonaws.com`
- Split: `{run_metadata['split']}`
- Agents fetched: {', '.join(run_metadata['agents'])}
- Instances requested: {run_metadata['n_instances']}
- Files fetched this run: {run_metadata['n_fetched']}
- Files already cached: {run_metadata['n_cached']}
- Failures: {run_metadata['n_failed']}
- Total size: {run_metadata['total_bytes'] / 1e6:.1f} MB

## How to use these files

```python
import json
from pathlib import Path
p = Path('output/trajectories/.cache/20240402_sweagent_gpt4/astropy__astropy-12907.json')
with open(p) as f:
    raw = json.load(f)
# raw['trajectory'] is a list of steps, each with:
#   'action' (raw command string)
#   'observation' (command output)
#   'response' (agent's natural-language response)
#   'state' (open_file, working_dir)
#   'thought' (agent's chain-of-thought)
```

For the Paper 2 pipeline, only `action` is used (as input to canonicalization
and BPE). Other fields are available for downstream analyses if needed.

## Canonicalization pipeline (downstream)

1. Parse each `action` string into `(verb, args)`.
2. Type-tag file-path args (`SRC_PY`, `TEST_PY`, `REPRO_PY`, `CONFIG_PY`).
3. Strip non-semantic literals (line numbers, commit hashes).
4. Output canonical atoms like `EDIT(SRC_PY, LINE:n)`, `RUN(pytest, TEST_PY)`.

See `analysis/preferences/canonicalize.py` for the implementation (to be
added).

## Related

- `data/swebench_trajectories.py` — the fetcher used here.
- `output/trajectories/*.parquet` — pre-extracted feature tables (unchanged).
- `analysis/preferences/variance_decomp.py`, `pair_features.py`, `task_diversity.py`
  — today's pilot analyses, using the parquet-level 8-token DSL.
"""
    note_path.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of instances (for quick test)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--agents", nargs="+", default=AGENTS,
                        help="Model IDs (default: all 3 committed agents)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading SWE-bench Lite instance IDs...")
    instance_ids = load_lite_instance_ids()
    if args.limit:
        instance_ids = instance_ids[:args.limit]
    print(f"  {len(instance_ids)} instances × {len(args.agents)} agents = "
          f"{len(instance_ids) * len(args.agents)} total files")

    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    records: list[dict[str, Any]] = []

    # Fetch in parallel per (instance, agent) pair
    tasks = [(iid, model) for model in args.agents for iid in instance_ids]

    print(f"\nFetching with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one_with_provenance, iid, model): (iid, model)
            for iid, model in tasks
        }
        done = 0
        t_start = time.time()
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(tasks)} done, {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA")

    n_fetched = sum(1 for r in records if r["status"] == "fetched")
    n_cached = sum(1 for r in records if r["status"] == "cached")
    n_failed = sum(1 for r in records if r["status"] == "failed")
    total_bytes = sum(r.get("size_bytes", 0) for r in records)

    run_metadata = {
        "run_started_at": run_started_at,
        "run_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "https://swe-bench-submissions.s3.amazonaws.com",
        "split": SPLIT,
        "agents": args.agents,
        "n_instances": len(instance_ids),
        "n_total_attempts": len(tasks),
        "n_fetched": n_fetched,
        "n_cached": n_cached,
        "n_failed": n_failed,
        "total_bytes": total_bytes,
    }

    manifest = {
        "run": run_metadata,
        "records": records,
    }

    manifest_path = CACHE_DIR / "manifest.json"
    # Append to manifest if it exists (keep history of fetch runs)
    if manifest_path.exists():
        with open(manifest_path) as f:
            prior = json.load(f)
        if isinstance(prior, dict) and "runs" in prior:
            prior["runs"].append(manifest)
            manifest = prior
        else:
            manifest = {"runs": [prior if "run" in prior else {}, manifest]}
    else:
        manifest = {"runs": [manifest]}

    manifest_path.write_text(json.dumps(manifest, indent=2))
    write_note(CACHE_DIR, run_metadata)

    print(f"\nRun complete:")
    print(f"  fetched: {n_fetched}")
    print(f"  already cached: {n_cached}")
    print(f"  failed: {n_failed}")
    print(f"  total size: {total_bytes / 1e6:.1f} MB")
    print(f"  manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"  note: {(CACHE_DIR / 'NOTE.md').relative_to(PROJECT_ROOT)}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
