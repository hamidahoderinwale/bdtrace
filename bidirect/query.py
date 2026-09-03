"""Additive filtering and semantic search over trace JSONL.

`query()` streams records through structural filters — a case-insensitive
regex over each record's text content, `field=value` equality on top-level
fields, and a time interval — and can then rank the survivors against a
natural-language query with a sentence-transformers model. Filters AND
together, and the embedding model only ever sees structural survivors.
`record_text()` is the one text view both the regex and the embedder read.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path

from bidirect.spec import _in_window, _iter_records, interval_bounds

DEFAULT_MODEL = "all-MiniLM-L6-v2"
TEXT_EVENT_CAP = 30  # events contributing detail strings to record_text
TEXT_CHAR_CAP = 2000  # cap on a record's text summary
EMBED_BATCH = 256


def record_text(record: dict, max_events: int = TEXT_EVENT_CAP, char_cap: int = TEXT_CHAR_CAP) -> str:
    """One plain-text view of a record: prompt text plus the first
    `max_events` events' detail strings, capped at `char_cap` chars. Pure —
    grep and the embedder both read this, so they see the same text."""
    parts: list[str] = []
    for p in record.get("prompts") or []:
        parts.append(p.get("text", "") if isinstance(p, dict) else str(p))
    for e in (record.get("events") or [])[:max_events]:
        details = e.get("details")
        if isinstance(details, dict):
            parts.extend(v for v in details.values() if isinstance(v, str))
        elif isinstance(details, str):
            parts.append(details)
    return "\n".join(s for s in parts if s)[:char_cap]


def _match_where(record: dict, clauses) -> bool:
    for clause in clauses:
        field, sep, value = clause.partition("=")
        if not sep:
            raise ValueError(f"bad where clause `{clause}` (use field=value)")
        rv = record.get(field)
        if not (rv == value or str(rv) == value or (rv is None and value == "null")):
            return False
    return True


def _embed(texts: list[str], model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    chunks = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunks.append(model.encode(texts[i : i + EMBED_BATCH], normalize_embeddings=True, show_progress_bar=False))
        if len(texts) > EMBED_BATCH:
            print(f"bdtrace query: embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", file=sys.stderr)
    import numpy as np

    return model, np.vstack(chunks)


def query(
    in_path: Path | str,
    grep: str | None = None,
    where: list[str] = (),
    interval: str | None = None,
    semantic: str | None = None,
    top_k: int = 20,
    min_score: float | None = None,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
) -> Iterator[tuple[dict, float | None]]:
    """Yield (record, score) for records passing every given filter.

    Structural filters (`where` equality, `interval` window, `grep` regex over
    `record_text`) run first and AND together. With `semantic` set, survivors
    are ranked by cosine similarity between the query and each record's
    `record_text` embedding — the top `top_k` are yielded best-first (or all
    scoring >= `min_score` when that is set instead); without it, records
    stream through in file order with score None. `limit` caps the yield count
    in both modes.

    When `<in_path>.index/` exists (see `bidirect.index.build_index`), matches
    `model`, and covers every survivor at its current text hash, ranking reads
    the stored vectors and only the query string is embedded; otherwise the
    on-the-fly path above runs unchanged.
    """
    pattern = re.compile(grep, re.IGNORECASE) if grep else None
    since, until = interval_bounds(interval)

    def survivors() -> Iterator[dict]:
        for r in _iter_records(Path(in_path) if in_path != "-" else in_path):
            if where and not _match_where(r, where):
                continue
            if not _in_window(r, since, until):
                continue
            if pattern and not pattern.search(record_text(r)):
                continue
            yield r

    if semantic is None:
        for n, r in enumerate(survivors()):
            if limit is not None and n >= limit:
                return
            yield r, None
        return

    records = list(survivors())
    if not records:
        return
    rec_embs = None
    if in_path != "-":
        from bidirect.index import _encode, lookup

        rec_embs = lookup(in_path, model, records)
    if rec_embs is not None:
        print(f"bdtrace query: index hit, embedding only the query with {model}", file=sys.stderr)
        q_emb = _encode([semantic], model)[0]
    else:
        print(f"bdtrace query: embedding {len(records)} records with {model}", file=sys.stderr)
        emb_model, rec_embs = _embed([record_text(r) for r in records], model)
        q_emb = emb_model.encode([semantic], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = rec_embs @ q_emb
    ranked = sorted(zip(records, scores, strict=True), key=lambda rs: -rs[1])
    if min_score is not None:
        ranked = [(r, s) for r, s in ranked if s >= min_score]
    else:
        ranked = ranked[:top_k]
    if limit is not None:
        ranked = ranked[:limit]
    for r, s in ranked:
        yield r, float(s)
