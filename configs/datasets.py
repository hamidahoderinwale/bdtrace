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
        "splits": ["dev", "test"],
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
