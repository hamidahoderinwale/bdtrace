#!/usr/bin/env python3
"""
Baseline extraction: SWE-bench Lite → three computed representations → certificates.

Usage:
  python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --output-dir output
  python scripts/run_extraction_pipeline.py --datasets swe_bench_lite --push
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.datasets import DATASET_CONFIGS, get_loader
from pipeline.utils import extract_dataset, get_hf_token, serialize_for_storage


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run extraction pipeline for procedural-info-theory")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["swe_bench_lite"],
        choices=list(DATASET_CONFIGS),
        help="Datasets to process. Baseline: swe_bench_lite only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output root. Parquet under datasets/, JSON under hf_export/",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per split")
    parser.add_argument(
        "--push-hub",
        type=str,
        default=None,
        help="Push to HF Hub. Default: midah/procedural-info-theory when set via --push",
    )
    parser.add_argument("--push", action="store_true", help="Push outputs to midah/procedural-info-theory")
    parser.add_argument("--no-parquet", action="store_true", help="Skip local parquet write")
    parser.add_argument("--no-hf-export", action="store_true", help="Skip HF JSON export")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).isoformat()
    output_dir = args.output_dir.resolve()
    parquet_dir = output_dir / "datasets"
    hf_export_dir = output_dir / "hf_export"

    if not args.no_parquet:
        parquet_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_hf_export:
        hf_export_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        loader_fn, config = get_loader(dataset_name)
        splits = config["splits"]
        loader_kwargs = config.get("loader_kwargs", {})

        for split in splits:
            loader = loader_fn(split=split, limit=args.limit, **loader_kwargs)
            records = list(extract_dataset(loader, limit=args.limit))

            if not records:
                print(f"  {dataset_name}/{split}: 0 records (skipping)")
                continue

            meta = {
                "dataset": dataset_name,
                "split": split,
                "count": len(records),
                "timestamp_utc": timestamp,
            }

            if not args.no_parquet:
                try:
                    import pandas as pd

                    df = pd.DataFrame([serialize_for_storage(r) for r in records])
                    split_dir = parquet_dir / dataset_name
                    split_dir.mkdir(parents=True, exist_ok=True)
                    out_path = split_dir / f"{split}.parquet"
                    df.to_parquet(out_path, index=False)
                    print(f"  {dataset_name}/{split}: {len(records)} -> {out_path}")
                except ImportError:
                    print("  pandas required for parquet. Install: pip install pandas pyarrow")
            else:
                print(f"  {dataset_name}/{split}: {len(records)} records")

            if not args.no_hf_export:
                split_dir = hf_export_dir / dataset_name
                split_dir.mkdir(parents=True, exist_ok=True)
                json_path = split_dir / f"{split}.json"
                with open(json_path, "w") as f:
                    json.dump({"meta": meta, "records": records}, f, indent=2, default=str)
                print(f"    HF export: {json_path}")

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
                print(f"  Pushed config {config_name} to {push_hub}")

            card_path = Path(__file__).resolve().parent.parent / "configs" / "hf_dataset_card.md"
            if card_path.exists():
                try:
                    from huggingface_hub import HfApi

                    HfApi(token=get_hf_token()).upload_file(
                        path_or_fileobj=str(card_path),
                        path_in_repo="README.md",
                        repo_id=push_hub,
                        repo_type="dataset",
                        commit_message="Update dataset card",
                    )
                    print(f"  Updated README.md on {push_hub}")
                except Exception as e:
                    print(f"  README upload skipped: {e}")
        except ImportError as e:
            print(f"  Push failed: {e}. Install: pip install datasets")
        except Exception as e:
            print(f"  Push failed: {e}")


if __name__ == "__main__":
    main()
