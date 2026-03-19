#!/usr/bin/env python3
"""
Baseline extraction: datasets → three computed representations → certificates.

Usage:
  python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --output-dir output
  python scripts/run_extraction_pipeline.py --datasets humaneval mbpp livecodebench bigcodebench
  python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --push
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.datasets import DATASET_CONFIGS, get_loader
from pipeline.utils import extract_dataset, get_hf_token, serialize_for_storage


def _run_one(job: dict) -> list[str]:
    """Extract one dataset, write outputs, return log lines. Module-level for pickling."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from configs.datasets import get_loader
    from pipeline.utils import extract_dataset, serialize_for_storage

    dataset_name = job["dataset_name"]
    limit = job["limit"]
    parquet_dir = Path(job["parquet_dir"])
    hf_export_dir = Path(job["hf_export_dir"])
    no_parquet = job["no_parquet"]
    no_hf_export = job["no_hf_export"]
    timestamp = job["timestamp"]

    logs = []
    loader_fn, config = get_loader(dataset_name)
    splits = config["splits"]
    loader_kwargs = config.get("loader_kwargs", {})

    for split in splits:
        loader = loader_fn(split=split, limit=limit, **loader_kwargs)
        records = list(extract_dataset(loader, limit=limit))

        if not records:
            logs.append(f"  {dataset_name}/{split}: 0 records (skipping)")
            continue

        meta = {
            "dataset": dataset_name,
            "split": split,
            "count": len(records),
            "timestamp_utc": timestamp,
        }

        if not no_parquet:
            try:
                import pandas as pd
                df = pd.DataFrame([serialize_for_storage(r) for r in records])
                split_dir = parquet_dir / dataset_name
                split_dir.mkdir(parents=True, exist_ok=True)
                out_path = split_dir / f"{split}.parquet"
                df.to_parquet(out_path, index=False)
                logs.append(f"  {dataset_name}/{split}: {len(records)} -> {out_path}")
            except ImportError:
                logs.append("  pandas required. Install: pip install pandas pyarrow")
        else:
            logs.append(f"  {dataset_name}/{split}: {len(records)} records")

        if not no_hf_export:
            split_dir = hf_export_dir / dataset_name
            split_dir.mkdir(parents=True, exist_ok=True)
            json_path = split_dir / f"{split}.json"
            with open(json_path, "w") as f:
                json.dump({"meta": meta, "records": records}, f, indent=2, default=str)
            logs.append(f"    HF export: {json_path}")

    return logs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run extraction pipeline for procedural-info-theory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["swe_bench_lite"],
        choices=list(DATASET_CONFIGS),
        help="Datasets to process.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--push-hub", type=str, default=None)
    parser.add_argument("--push", action="store_true", help="Push to midah/procedural-info-theory")
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--no-hf-export", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).isoformat()
    output_dir = args.output_dir.resolve()
    parquet_dir = output_dir / "datasets"
    hf_export_dir = output_dir / "hf_export"

    if not args.no_parquet:
        parquet_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_hf_export:
        hf_export_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "dataset_name": name,
            "limit": args.limit,
            "parquet_dir": str(parquet_dir),
            "hf_export_dir": str(hf_export_dir),
            "no_parquet": args.no_parquet,
            "no_hf_export": args.no_hf_export,
            "timestamp": timestamp,
        }
        for name in args.datasets
    ]

    max_workers = min(len(jobs), 4)
    if max_workers <= 1:
        for job in jobs:
            for line in _run_one(job):
                print(line)
    else:
        print(f"Running {len(jobs)} datasets in parallel (workers={max_workers})")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, job): job["dataset_name"] for job in jobs}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    for line in future.result():
                        print(line)
                except Exception as e:
                    print(f"  {name}: FAILED — {e}")

    push_hub = args.push_hub or ("midah/procedural-info-theory" if args.push else None)
    if push_hub:
        try:
            from datasets import Dataset, DatasetDict
            configs = {}
            for dataset_name in args.datasets:
                ds_dir = hf_export_dir / dataset_name
                if not ds_dir.exists():
                    continue
                splits_d = {}
                for jf in ds_dir.glob("*.json"):
                    split = jf.stem
                    with open(jf) as f:
                        data = json.load(f)
                    recs = data.get("records", [])
                    if recs:
                        splits_d[split] = Dataset.from_list(
                            [serialize_for_storage(r) for r in recs]
                        )
                if splits_d:
                    configs[dataset_name] = DatasetDict(splits_d)
            for config_name, ds_dict in configs.items():
                ds_dict.push_to_hub(
                    push_hub,
                    config_name=config_name,
                    token=get_hf_token(),
                    commit_message=f"Add {config_name} extraction {timestamp[:10]}",
                )
                print(f"  Pushed {config_name} to {push_hub}")
        except Exception as e:
            print(f"  Push failed: {e}")


if __name__ == "__main__":
    main()
