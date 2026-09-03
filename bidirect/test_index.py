"""Tests for bidirect.index — content-addressed incremental builds, model
invalidation, corpus mirroring, and index/no-index query equivalence. The
encoder is monkeypatched with a deterministic text-hash embedding, so no model
loads; every assertion is bookkeeping over ids and hashes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bidirect import index as idx
from bidirect import query as q
from bidirect.query import query, record_text

DIMS = 8


def fake_encode(texts: list[str], model_name: str = "fake") -> np.ndarray:
    """Deterministic unit vector per text, seeded by its content hash."""
    rows = []
    for t in texts:
        rng = np.random.default_rng(int(idx.text_sha256(t)[:8], 16))
        v = rng.standard_normal(DIMS).astype(np.float32)
        rows.append(v / np.linalg.norm(v))
    return np.vstack(rows)


class FakeModel:
    def encode(self, texts, **kwargs):
        return fake_encode(list(texts))


@pytest.fixture()
def encoder(monkeypatch):
    """Route both the index build and query's on-the-fly path through
    fake_encode, recording every text the 'model' sees."""
    embedded: list[str] = []

    def spy_encode(texts, model_name):
        embedded.extend(texts)
        return fake_encode(texts)

    monkeypatch.setattr(idx, "_encode", spy_encode)
    monkeypatch.setattr(q, "_embed", lambda texts, model_name: (FakeModel(), spy_encode(texts, model_name)))
    return embedded


def _rec(instance_id, prompt):
    return {
        "instance_id": instance_id,
        "repo": "r/r",
        "events": [{"type": "prompt", "details": {"text": prompt}}],
        "prompts": [{"text": prompt}],
    }


RECORDS = [
    _rec("astro-1", "fix the separability matrix for nested compound models"),
    _rec("bread-1", "the sourdough starter recipe needs more hydration"),
    _rec("css-1", "make the button blue with rounded corners"),
]


def _write(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture()
def traces(tmp_path: Path) -> Path:
    return _write(tmp_path / "traces.jsonl", RECORDS)


def test_build_writes_matrix_labeled_by_id_and_hash(traces, encoder):
    meta = idx.build_index(traces, model="fake")
    assert meta["n"] == 3 and meta["dims"] == DIMS and meta["model"] == "fake"
    assert meta["ids"] == ["astro-1", "bread-1", "css-1"]
    assert meta["text_sha256"] == [idx.text_sha256(record_text(r)) for r in RECORDS]
    vectors, disk_meta = idx.load_index(traces)
    assert vectors.shape == (3, DIMS) and vectors.dtype == np.float32
    assert disk_meta["ids"] == meta["ids"]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_incremental_build_embeds_only_new_records(traces, encoder):
    idx.build_index(traces, model="fake")
    assert len(encoder) == 3
    _write(traces, RECORDS + [_rec("new-1", "a brand new trace about databases")])
    meta = idx.build_index(traces, model="fake")
    # exactly one more text reached the encoder: the new record's
    assert len(encoder) == 4
    assert encoder[3] == record_text(_rec("new-1", "a brand new trace about databases"))
    assert meta["n"] == 4 and meta["ids"][-1] == "new-1"


def test_changed_record_is_reembedded_and_removed_record_dropped(traces, encoder):
    idx.build_index(traces, model="fake")
    changed = [_rec("astro-1", "now about wcs coordinate frames instead"), RECORDS[2]]
    meta = idx.build_index(_write(traces, changed), model="fake")
    # astro-1's new text was embedded; bread-1 is gone; css-1 came from cache
    assert encoder[3:] == [record_text(changed[0])]
    assert meta["ids"] == ["astro-1", "css-1"]
    vectors, _ = idx.load_index(traces)
    np.testing.assert_allclose(vectors, fake_encode([record_text(r) for r in changed]), rtol=1e-5)


def test_model_mismatch_triggers_full_rebuild(traces, encoder, capsys):
    idx.build_index(traces, model="fake")
    meta = idx.build_index(traces, model="fake-v2")
    assert len(encoder) == 6  # all three embedded again, none reused
    assert meta["model"] == "fake-v2"
    assert "model changed (fake -> fake-v2), full rebuild" in capsys.readouterr().err


def test_query_with_index_matches_query_without(traces, encoder):
    semantic = "fix separability matrix computation"
    without = [(r["instance_id"], s) for r, s in query(traces, semantic=semantic, top_k=3, model="fake")]
    idx.build_index(traces, model="fake")
    baseline = len(encoder)
    with_index = [(r["instance_id"], s) for r, s in query(traces, semantic=semantic, top_k=3, model="fake")]
    assert [i for i, _ in with_index] == [i for i, _ in without]
    np.testing.assert_allclose([s for _, s in with_index], [s for _, s in without], rtol=1e-5)
    assert len(encoder) == baseline + 1  # indexed query embedded only the query string


def test_query_falls_back_when_index_is_stale_or_foreign(traces, encoder):
    idx.build_index(traces, model="fake")
    # different model than the index: fall back, all survivors re-embedded
    n0 = len(encoder)
    list(query(traces, semantic="anything", model="fake-v2"))
    assert len(encoder) == n0 + 3
    # corpus grew past the index: fall back rather than rank stale coverage
    _write(traces, RECORDS + [_rec("new-1", "fresh uncovered record")])
    n1 = len(encoder)
    list(query(traces, semantic="anything", model="fake"))
    assert len(encoder) == n1 + 4


def test_structural_filters_mask_the_index(traces, encoder):
    idx.build_index(traces, model="fake")
    results = list(query(traces, grep="sourdough", semantic="baking bread", model="fake"))
    assert [r["instance_id"] for r, _ in results] == ["bread-1"]


def test_duplicate_instance_id_is_rejected(tmp_path, encoder):
    p = _write(tmp_path / "dup.jsonl", [RECORDS[0], RECORDS[0]])
    with pytest.raises(ValueError, match="duplicate instance_id"):
        idx.build_index(p, model="fake")
