"""Tests for the residual-identity audit.

The audit exists because the anonymizer is a denylist: it removes what its rules
name. These tests fix the contract that a class the anonymizer claims is reported
when it survives, and that a token in an identity position is surfaced as a
candidate even though no rule can name it.
"""

import json

import pytest

from bdtrace import spec
from bdtrace.audit import audit, report


def _write(tmp_path, *records):
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return p


@pytest.mark.parametrize("text,cls", [
    ("cd /Users/jdoe/work", "home path"),
    ("~/.claude/projects/-Users-jdoe/x.jsonl", "workspace slug"),
    ("mail jdoe@example.com", "email"),
    ("gh api repos/acme/widget", "repos owner"),
    ("clone https://github.com/acme/widget", "forge owner"),
    ("export TOKEN=ghp_" + "b" * 36, "credential"),
])
def test_each_residual_class_is_reported(tmp_path, text, cls):
    result = audit(_write(tmp_path, {"events": [{"details": {"command": text}}]}))
    assert cls in result["residual"], f"{cls} not reported for {text!r}"


def test_anonymized_output_has_no_residual(tmp_path):
    raw = {"instance_id": "x", "events": [{"details": {
        "command": "cd /Users/jdoe && gh api repos/acme/widget && git push https://github.com/acme/widget",
        "text": "ping jdoe@example.com",
    }}]}
    clean = spec.anonymize(raw)
    result = audit(_write(tmp_path, clean))
    assert result["residual"] == {}, result["residual"]


def test_identity_in_prose_is_a_candidate_not_a_residual(tmp_path):
    """No rule can see a bare handle, so it must surface for a human instead."""
    rec = {"events": [{"details": {"command": "git log --author zhuohaouw"}}]}
    result = audit(_write(tmp_path, rec))
    assert "zhuohaouw" in result["candidates"]
    assert result["residual"] == {}


def test_ordinary_words_are_not_candidates(tmp_path):
    rec = {"events": [{"details": {"command": "git log --author the", "text": "user: none"}}]}
    result = audit(_write(tmp_path, rec))
    assert result["candidates"] == {}


def test_extra_terms_are_counted_as_residual(tmp_path):
    rec = {"events": [{"details": {"text": "the other mac is hamidaho"}}]}
    result = audit(_write(tmp_path, rec), ("hamidaho",))
    assert result["residual"] == {"term 'hamidaho'": 1}


def test_samples_are_masked(tmp_path):
    result = audit(_write(tmp_path, {"events": [{"details": {"command": "cd /Users/jdoemarks"}}]}))
    sample = result["residual_samples"]["home path"][0]
    assert "jdoemarks" not in sample and sample.startswith("/Users/")


def test_report_states_clean_when_nothing_residual(tmp_path):
    text = report(audit(_write(tmp_path, {"events": []})))
    assert "RESIDUAL — none" in text
