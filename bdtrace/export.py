"""Serialization/compression layer and Hugging Face push for trace records.

A trace record is a plain dict shaped like one row of
output/resolved_traces_lite_full.jsonl (scalar fields such as instance_id /
repo / base_commit plus nested fields such as events / prompts / modules).

Formats: jsonl, jsonl.gz, jsonl.zst (needs the optional `zstandard` package),
parquet (needs the "parquet" extra: pandas + pyarrow), msgpack (repo
convention: one packed list of records, read back with
``msgpack.unpack(raw=False)``).

All writes are atomic: a temp file in the destination directory, then
``os.replace``.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

FORMATS = ("jsonl", "jsonl.gz", "jsonl.zst", "parquet", "msgpack")

# Longest suffix first so "x.jsonl.gz" resolves before "x.jsonl".
_SUFFIX_TO_FMT = (
    (".jsonl.zst", "jsonl.zst"),
    (".jsonl.gz", "jsonl.gz"),
    (".jsonl", "jsonl"),
    (".parquet", "parquet"),
    (".msgpack", "msgpack"),
)


def infer_format(path: Path) -> str:
    """Format name for a path, from its (possibly compound) suffix."""
    name = path.name.lower()
    for suffix, fmt in _SUFFIX_TO_FMT:
        if name.endswith(suffix):
            return fmt
    raise ValueError(
        f"cannot infer trace format from {path.name!r}; "
        f"use one of {', '.join(s for s, _ in _SUFFIX_TO_FMT)} or pass fmt explicitly"
    )


def _import_zstandard():
    try:
        import zstandard
    except ImportError as e:  # dependency is deliberately not added to the repo
        raise ImportError(
            "jsonl.zst needs the `zstandard` package, which is not installed; "
            "install it with `uv pip install zstandard` (or choose jsonl.gz)"
        ) from e
    return zstandard


def _import_parquet():
    try:
        import pandas
        import pyarrow  # noqa: F401  (pandas needs it as the parquet engine)
    except ImportError as e:
        raise ImportError(
            'parquet export needs the "parquet" extra (pandas + pyarrow); '
            "install it with `uv sync --extra parquet`"
        ) from e
    return pandas


def _materialize(records: Iterable[dict] | Path) -> list[dict]:
    """Records as a list; a Path is loaded via load_traces (any known format)."""
    if isinstance(records, (str, Path)):
        return load_traces(Path(records))
    return list(records)


def _stream(records: Iterable[dict] | Path) -> Iterable[dict]:
    """Lazy record iterator: a .jsonl Path streams line by line (constant memory);
    anything else falls back to _materialize. Keeps GB-scale jsonl exports flat."""
    if isinstance(records, (str, Path)) and str(records).endswith(".jsonl"):
        def gen():
            with open(records, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        return gen()
    if isinstance(records, (str, Path)):
        return _materialize(records)
    return records


def _progress(rows: Iterable[dict], what: str) -> Iterable[dict]:
    """tqdm on stderr (the hf CLI look); silent when stderr is not a terminal."""
    try:
        from tqdm import tqdm
        return tqdm(rows, desc=what, unit=" rec", disable=None)
    except ImportError:
        return rows


_NESTED_FIELDS_KEY = "bdtrace_nested_fields"


def _flatten(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Rows with nested (dict/list) values JSON-encoded, plus those field names.

    A field counts as nested if it is a dict or list in ANY record; the whole
    column is then JSON-encoded so the column has one type.
    """
    nested = sorted(
        {k for r in records for k, v in r.items() if isinstance(v, (dict, list))}
    )
    if not nested:
        return [dict(r) for r in records], nested
    flat = []
    for r in records:
        row = dict(r)
        for k in nested:
            if k in row:
                row[k] = json.dumps(row[k], ensure_ascii=False)
        flat.append(row)
    return flat, nested


def _unflatten(rows: list[dict], nested: list[str]) -> list[dict]:
    out = []
    for row in rows:
        r = dict(row)
        for k in nested:
            if k in r and isinstance(r[k], str):
                r[k] = json.loads(r[k])
        out.append(r)
    return out


def export_traces(
    records: Iterable[dict] | Path, out: Path, fmt: str | None = None
) -> Path:
    """Write trace records to `out` in `fmt` (inferred from suffix when None).

    `records` is an iterable of record dicts or a Path to an existing dump in
    any supported format. Formats: jsonl; jsonl.gz (stdlib gzip); jsonl.zst
    (only if `zstandard` is importable); parquet (records are flattened —
    nested dict/list fields are stored as JSON strings in their columns, and
    the column names are kept in the parquet metadata so load_traces can
    decode them); msgpack (one packed list of records, matching the repo's
    ``msgpack.unpack(raw=False)`` convention).

    The write is atomic (temp file in the destination directory, then
    os.replace). Returns `out`.
    """
    out = Path(out)
    fmt = fmt or infer_format(out)
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    # line formats stream (constant memory, any size); parquet/msgpack materialize by nature
    streaming = fmt in ("jsonl", "jsonl.gz", "jsonl.zst")
    rows = _progress(_stream(records), f"export {fmt}") if streaming else _materialize(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        if fmt == "jsonl":
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        elif fmt == "jsonl.gz":
            with os.fdopen(fd, "wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        elif fmt == "jsonl.zst":
            zstandard = _import_zstandard()
            with os.fdopen(fd, "wb") as raw:
                with zstandard.ZstdCompressor().stream_writer(raw) as zf:
                    for r in rows:
                        line = json.dumps(r, ensure_ascii=False, default=str) + "\n"
                        zf.write(line.encode("utf-8"))
        elif fmt == "msgpack":
            import msgpack

            with os.fdopen(fd, "wb") as f:
                msgpack.pack(rows, f)
        elif fmt == "parquet":
            pd = _import_parquet()
            import pyarrow as pa
            import pyarrow.parquet as pq

            flat, nested = _flatten(rows)
            table = pa.Table.from_pandas(pd.DataFrame(flat), preserve_index=False)
            meta = dict(table.schema.metadata or {})
            meta[_NESTED_FIELDS_KEY.encode()] = json.dumps(nested).encode()
            table = table.replace_schema_metadata(meta)
            os.close(fd)
            pq.write_table(table, tmp)
        os.replace(tmp, out)
    finally:
        if tmp.exists() and tmp != out:
            tmp.unlink()
    return out


def load_traces(path: Path, fmt: str | None = None) -> list[dict]:
    """Read trace records back from any format export_traces writes."""
    path = Path(path)
    fmt = fmt or infer_format(path)
    if fmt == "jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if fmt == "jsonl.gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if fmt == "jsonl.zst":
        zstandard = _import_zstandard()
        import io

        with open(path, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as zf:
                text = io.TextIOWrapper(zf, encoding="utf-8")
                return [json.loads(line) for line in text if line.strip()]
    if fmt == "msgpack":
        import msgpack

        with open(path, "rb") as f:
            return msgpack.unpack(f, raw=False)
    if fmt == "parquet":
        _import_parquet()
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta = table.schema.metadata or {}
        nested = json.loads(meta.get(_NESTED_FIELDS_KEY.encode(), b"[]"))
        return _unflatten(table.to_pylist(), nested)
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


def resolve_hf_token() -> tuple[str, str] | None:
    """(token, source): own env/.env first, then huggingface_hub's cached login.

    Mirrors transforms.resolve_api_key. The token value is never printed or
    logged — only its source name is safe to surface.
    """
    from dotenv import load_dotenv

    load_dotenv()
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "HF_TOKEN"
    from huggingface_hub import get_token

    token = get_token()
    if token:
        return token, "huggingface_hub cached login"
    return None


def push_traces(
    records_or_path: Iterable[dict] | Path,
    repo_id: str,
    private: bool = True,
    dry_run: bool = False,
) -> str:
    """Push trace records to a Hugging Face dataset repo (parquet on the hub).

    Builds a datasets.Dataset from the records with nested dict/list fields
    JSON-encoded into string columns (same flattening as the parquet export,
    so heterogeneous event/prompt structures never fight Arrow type
    unification; consumers json.loads those columns). `dry_run=True` builds
    the Dataset and returns a rows/features report WITHOUT any network call
    or token resolution. Otherwise pushes with ``push_to_hub`` (private by
    default) and returns the dataset URL. Token values are never printed.
    """
    from datasets import Dataset

    rows = _materialize(records_or_path)
    flat, nested = _flatten(rows)
    ds = Dataset.from_list(flat)
    if dry_run:
        features = ", ".join(f"{k}: {v.dtype}" for k, v in ds.features.items())
        note = f" (JSON-encoded columns: {', '.join(nested)})" if nested else ""
        return f"dry-run: {ds.num_rows} rows -> {repo_id} (private={private}); features: {features}{note}"
    resolved = resolve_hf_token()
    if resolved is None:
        raise RuntimeError(
            "no Hugging Face token: set HF_TOKEN in env or .env, "
            "or log in once with `huggingface-cli login`"
        )
    token, source = resolved
    print(f"hf token: {source}")
    ds.push_to_hub(repo_id, private=private, token=token)
    return f"https://huggingface.co/datasets/{repo_id}"
