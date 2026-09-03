"""Persistent embedding index over a trace JSONL.

`build_index()` maintains `<corpus>.index/` beside the corpus: one float32
matrix (`vectors.npy`, L2-normalized rows) plus `index_meta.json` whose
`ids[i]` and `text_sha256[i]` label row i. Increments are content-addressed —
a record is re-embedded only when its `record_text` hash is new — and after
every build the matrix mirrors the corpus exactly (removed or changed records
lose their old rows). A model change invalidates the whole index; vectors from
different models never share a matrix.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from bdtrace.spec import _iter_records

VECTORS_FILE = "vectors.npy"
META_FILE = "index_meta.json"


def index_dir(in_path: Path | str) -> Path:
    return Path(str(in_path) + ".index")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _encode(texts: list[str], model_name: str) -> np.ndarray:
    """Embed `texts` with a sentence-transformers model. Tests monkeypatch
    this; everything above it is model-free."""
    from sentence_transformers import SentenceTransformer

    from bdtrace.query import EMBED_BATCH

    model = SentenceTransformer(model_name)
    chunks = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunks.append(model.encode(texts[i : i + EMBED_BATCH], normalize_embeddings=True, show_progress_bar=False))
        if len(texts) > EMBED_BATCH:
            print(f"bdtrace index: embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", file=sys.stderr)
    return np.vstack(chunks)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def load_index(in_path: Path | str) -> tuple[np.ndarray, dict] | None:
    """Return (vectors, meta) for the index beside `in_path`, or None when
    the index is absent or internally inconsistent (row/label mismatch)."""
    d = index_dir(in_path)
    vec_path, meta_path = d / VECTORS_FILE, d / META_FILE
    if not (vec_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        vectors = np.load(vec_path)
    except (json.JSONDecodeError, ValueError, OSError):
        return None
    ids, hashes = meta.get("ids"), meta.get("text_sha256")
    if not isinstance(ids, list) or not isinstance(hashes, list):
        return None
    if vectors.ndim != 2 or len(ids) != vectors.shape[0] or len(hashes) != vectors.shape[0]:
        return None
    return vectors, meta


def build_index(in_path: Path | str, model: str | None = None) -> dict:
    """Bring `<in_path>.index/` in sync with the corpus, embedding only
    records whose (id, text hash) is not already indexed. Returns the new
    meta dict. Status goes to stderr; the write is atomic (tmp then replace).
    """
    from bdtrace.query import DEFAULT_MODEL, record_text

    model = model or DEFAULT_MODEL
    in_path = Path(in_path)

    corpus: list[tuple[str, str, str]] = []  # (id, sha, text) in file order
    seen_ids: set[str] = set()
    for n, record in enumerate(_iter_records(in_path)):
        rid = record.get("instance_id")
        if rid is None:
            raise ValueError(f"record {n} has no instance_id; the index needs one per record")
        if rid in seen_ids:
            raise ValueError(f"duplicate instance_id `{rid}`; index rows must be uniquely labeled")
        seen_ids.add(rid)
        text = record_text(record)
        corpus.append((rid, text_sha256(text), text))

    cached: dict[tuple[str, str], int] = {}
    old_vectors: np.ndarray | None = None
    loaded = load_index(in_path)
    if loaded is not None:
        old_vectors, old_meta = loaded
        if old_meta.get("model") != model:
            print(
                f"bdtrace index: model changed ({old_meta.get('model')} -> {model}), full rebuild",
                file=sys.stderr,
            )
        else:
            cached = {(rid, sha): i for i, (rid, sha) in enumerate(zip(old_meta["ids"], old_meta["text_sha256"], strict=True))}

    new = [(rid, sha, text) for rid, sha, text in corpus if (rid, sha) not in cached]
    new_vectors = _normalize(_encode([text for _, _, text in new], model)) if new else None
    new_row = {(rid, sha): i for i, (rid, sha, _) in enumerate(new)}

    rows = []
    for rid, sha, _ in corpus:
        if (rid, sha) in cached:
            rows.append(old_vectors[cached[(rid, sha)]])
        else:
            rows.append(new_vectors[new_row[(rid, sha)]])
    dims = int(new_vectors.shape[1]) if new_vectors is not None else (int(old_vectors.shape[1]) if old_vectors is not None else 0)
    matrix = _normalize(np.vstack(rows)) if rows else np.zeros((0, dims), dtype=np.float32)

    meta = {
        "model": model,
        "dims": int(matrix.shape[1]),
        "ids": [rid for rid, _, _ in corpus],
        "text_sha256": [sha for _, sha, _ in corpus],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(corpus),
    }

    d = index_dir(in_path)
    d.mkdir(parents=True, exist_ok=True)
    tmp_vec = d / (VECTORS_FILE + ".tmp.npy")
    np.save(tmp_vec, matrix)
    os.replace(tmp_vec, d / VECTORS_FILE)
    tmp_meta = d / (META_FILE + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp_meta, d / META_FILE)

    print(f"bdtrace index: {len(new)} new, {len(corpus) - len(new)} cached ({len(corpus)} total) -> {d}", file=sys.stderr)
    return meta


def lookup(in_path: Path | str, model: str, records: list[dict]) -> np.ndarray | None:
    """Row vectors for `records` from the stored index, in the order given —
    or None when the index is absent, was built with another model, or does
    not cover every record at its current text hash (stale rows never rank).
    """
    from bdtrace.query import record_text

    loaded = load_index(in_path)
    if loaded is None:
        return None
    vectors, meta = loaded
    if meta.get("model") != model:
        return None
    row = {(rid, sha): i for i, (rid, sha) in enumerate(zip(meta["ids"], meta["text_sha256"], strict=True))}
    selected = []
    for record in records:
        key = (record.get("instance_id"), text_sha256(record_text(record)))
        i = row.get(key)
        if i is None:
            return None
        selected.append(i)
    return vectors[selected]
