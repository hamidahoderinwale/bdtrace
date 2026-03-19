"""
Dataset configs for extraction pipeline.

Each config: source_id, loader, splits, and optional limits.
"""

from collections.abc import Callable, Iterator
from typing import Any

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "swe_bench_lite": {
        "source_id": "princeton-nlp/SWE-bench_Lite",
        "loader": "data.swe_bench:load_swe_bench_lite",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {},
    },
    "swe_bench_lite_resolved": {
        "source_id": "output/resolved_traces_lite_full.jsonl",
        "loader": "data.loaders:load_from_jsonl",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {"path": "output/resolved_traces_lite_full.jsonl"},
    },
    "swe_bench_lite_resolved_multifile": {
        "source_id": "output/resolved_traces_multifile.jsonl",
        "loader": "data.loaders:load_from_jsonl",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {"path": "output/resolved_traces_multifile.jsonl"},
    },
    "swe_bench_verified_resolved_multifile": {
        "source_id": "output/resolved_traces_verified_multifile.jsonl",
        "loader": "data.loaders:load_from_jsonl",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {"path": "output/resolved_traces_verified_multifile.jsonl"},
    },
    "swe_bench_verified_resolved_full": {
        "source_id": "output/resolved_traces_verified_full.jsonl",
        "loader": "data.loaders:load_from_jsonl",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {"path": "output/resolved_traces_verified_full.jsonl"},
    },
    "swe_smith_resolved": {
        "source_id": "output/resolved_traces_swe_smith.jsonl",
        "loader": "data.loaders:load_from_jsonl",
        "splits": ["train"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {"path": "output/resolved_traces_swe_smith.jsonl"},
    },
    "swe_smith_stratified": {
        "source_id": "output/resolved_traces_swe_smith_stratified.jsonl",
        "loader": "data.loaders:load_from_jsonl",
        "splits": ["train"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {"path": "output/resolved_traces_swe_smith_stratified.jsonl"},
    },
    "humaneval": {
        "source_id": "openai_humaneval",
        "loader": "data.loaders_ext:load_humaneval",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {},
    },
    "mbpp": {
        "source_id": "google-research-datasets/mbpp",
        "loader": "data.loaders_ext:load_mbpp",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {},
    },
    "livecodebench": {
        "source_id": "livecodebench/code_generation_lite",
        "loader": "data.loaders_ext:load_livecodebench",
        "splits": ["test"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {},
    },
    "bigcodebench": {
        "source_id": "bigcode/bigcodebench",
        "loader": "data.loaders_ext:load_bigcodebench",
        "splits": ["v0.1.2"],
        "instance_id_field": "instance_id",
        "loader_kwargs": {},
    },
}


def get_loader(dataset_name: str) -> tuple[Callable[..., Iterator[dict]], dict[str, Any]]:
    """Resolve loader callable and config for a dataset."""
    import importlib

    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS)}")

    mod_path, func_name = config["loader"].rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    loader = getattr(mod, func_name)
    return loader, config
