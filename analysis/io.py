"""
I/O for analysis artifacts: matrices, labels, diversity results.

Supports Parquet (primary) and legacy formats (npz, json).
"""

import json
from pathlib import Path

import numpy as np

from .diversity import REPR_NAMES


def load_matrices(path: Path) -> dict[str, np.ndarray]:
    """Load distance matrices from parquet, npz, or json."""
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        cols = [c for c in df.columns if c.startswith("d_")]
        n = int(df[["i", "j"]].max().max()) + 1
        matrices = {}
        for c in cols:
            name = c[2:]
            D = np.zeros((n, n))
            for _, row in df.iterrows():
                i, j = int(row["i"]), int(row["j"])
                D[i, j] = D[j, i] = row[c]
            matrices[name] = D
        return matrices

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        result = {}
        for name in REPR_NAMES:
            if name in keys:
                result[name] = data[name]
            elif f"D_{name}" in keys:
                result[name] = data[f"D_{name}"]
        return result

    if path.suffix == ".json":
        with open(path) as f:
            d = json.load(f)
        result = {}
        for name in REPR_NAMES:
            if name in d:
                result[name] = np.array(d[name])
            elif f"D_{name}" in d:
                result[name] = np.array(d[f"D_{name}"])
        return result

    raise ValueError(f"Unsupported format: {path.suffix}. Use .parquet, .npz, or .json")


def load_labels(path: Path) -> np.ndarray:
    """Load stratum labels from parquet, npy, json, or jsonl."""
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        if "stratum" in df.columns:
            return np.array(df["stratum"].tolist())
        if "label" in df.columns:
            return np.array(df["label"].tolist())
        return np.array(df.iloc[:, -1].tolist())

    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return np.array(data)
        if "labels" in data:
            return np.array(data["labels"])
        raise ValueError("JSON must be list or have 'labels' key")

    if path.suffix == ".jsonl":
        labels = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                labels.append(rec.get("stratum") or rec.get("label") or rec.get("instance_id", ""))
        return np.array(labels)

    raise ValueError(f"Unsupported format: {path.suffix}")
