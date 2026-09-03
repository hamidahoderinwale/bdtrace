"""Tests for bidirect.query — structural filter additivity on synthetic
fixtures (no model), plus one semantic ranking test that loads MiniLM and is
skipped where sentence-transformers is unavailable."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bdtrace.query import query, record_text


def _rec(instance_id, repo, prompt, detail, ts):
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "a" * 40,
        "events": [
            {"type": "prompt", "details": {"text": prompt}, "timestamp": ts},
            {"type": "code_change", "details": {"file_path": detail}, "timestamp": ts},
        ],
        "prompts": [{"text": prompt}],
    }


RECORDS = [
    _rec(
        "astro-1",
        "astropy/astropy",
        "fix the separability matrix for nested compound models",
        "astropy/modeling/separable.py",
        "2022-03-03T15:14:54Z",
    ),
    _rec(
        "bread-1",
        "kitchen/bakery",
        "the sourdough starter recipe needs more hydration for baking bread",
        "recipes/sourdough.md",
        "2023-06-01T09:00:00Z",
    ),
    _rec(
        "css-1",
        "web/site",
        "make the button blue with rounded corners in the stylesheet",
        "static/style.css",
        "2024-01-15T12:00:00Z",
    ),
]


@pytest.fixture()
def traces(tmp_path: Path) -> Path:
    p = tmp_path / "traces.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    return p


def _ids(results):
    return [r["instance_id"] for r, _ in results]


def test_no_filters_streams_all(traces):
    results = list(query(traces))
    assert _ids(results) == ["astro-1", "bread-1", "css-1"]
    assert all(score is None for _, score in results)


def test_grep_is_case_insensitive_regex(traces):
    assert _ids(query(traces, grep="SEPARABILITY")) == ["astro-1"]
    assert _ids(query(traces, grep=r"sour(dough|cream)")) == ["bread-1"]


def test_grep_sees_event_details_not_just_prompts(traces):
    # style.css appears only in an event's details, never in a prompt
    assert _ids(query(traces, grep=r"style\.css")) == ["css-1"]


def test_where_equality_on_top_level_fields(traces):
    assert _ids(query(traces, where=["repo=kitchen/bakery"])) == ["bread-1"]
    assert _ids(query(traces, where=["instance_id=css-1"])) == ["css-1"]
    assert _ids(query(traces, where=["repo=nope/nope"])) == []


def test_where_null_matches_missing_field(traces):
    assert _ids(query(traces, where=["cwd=null"])) == ["astro-1", "bread-1", "css-1"]


def test_bad_where_clause_raises(traces):
    with pytest.raises(ValueError, match="bad where clause"):
        list(query(traces, where=["no-equals-sign"]))


def test_interval_window(traces):
    assert _ids(query(traces, interval="2023-01-01..2024-01-01")) == ["bread-1"]
    assert _ids(query(traces, interval="2023-01-01..")) == ["bread-1", "css-1"]
    assert _ids(query(traces, interval="..2023-01-01")) == ["astro-1"]


def test_filters_are_additive(traces):
    # grep alone matches astro-1; the where clause excludes it -> nothing
    assert _ids(query(traces, grep="separability", where=["repo=web/site"])) == []
    # interval alone matches bread-1 and css-1; grep narrows to css-1
    assert _ids(query(traces, grep="button", interval="2023-01-01..")) == ["css-1"]


def test_limit_caps_streaming_output(traces):
    assert _ids(query(traces, limit=2)) == ["astro-1", "bread-1"]


def test_record_text_is_pure_and_capped():
    r = RECORDS[0]
    assert record_text(r) == record_text(r)
    assert "separability matrix" in record_text(r)
    assert "separable.py" in record_text(r)  # event detail strings included
    assert len(record_text(r, char_cap=50)) == 50


HAVE_ST = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(not HAVE_ST, reason="sentence-transformers not installed")
def test_semantic_ranks_on_topic_record_first(traces):
    results = list(query(traces, semantic="fix separability matrix computation", top_k=3))
    assert _ids(results)[0] == "astro-1"
    scores = [s for _, s in results]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)
