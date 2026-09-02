"""Round-trip tests for bidirect.export — every format, plus a dry-run push.

No network calls: push_traces is exercised with dry_run=True only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bidirect.export import export_traces, infer_format, load_traces, push_traces


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# Three synthetic records shaped like rows of output/resolved_traces_lite_full.jsonl:
# scalar fields plus nested lists of dicts, with heterogeneous event details.
RECORDS = [
    {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "base_commit": "a" * 40,
        "events": [
            {
                "type": "prompt",
                "details": {"text": "fix the bug — síl vous plaît"},
                "timestamp": "2022-03-03T15:14:54Z",
            },
            {
                "type": "code_change",
                "details": {"file_path": "pkg/mod.py", "before_content": "x = 1\n", "after_content": "x = 2\n"},
                "timestamp": "2022-03-03T15:20:00Z",
            },
        ],
        "prompts": ["fix the bug"],
        "modules": ["pkg.mod"],
        "modules_edges": [["pkg.mod", "pkg.util"]],
        "modules_from_repo": True,
    },
    {
        "instance_id": "demo__repo-2",
        "repo": "demo/repo",
        "base_commit": "b" * 40,
        "events": [],
        "prompts": [],
        "modules": [],
        "modules_edges": [],
        "modules_from_repo": False,
    },
    {
        "instance_id": "demo__repo-3",
        "repo": "demo/other",
        "base_commit": "c" * 40,
        "events": [
            {
                "type": "prompt",
                "details": {"text": "add a test", "content": "add a test"},
                "timestamp": "2023-01-01T00:00:00Z",
            }
        ],
        "prompts": ["add a test"],
        "modules": ["other.core", "other.io"],
        "modules_edges": [],
        "modules_from_repo": True,
    },
]

FORMAT_CASES = [
    pytest.param("jsonl", id="jsonl"),
    pytest.param("jsonl.gz", id="jsonl.gz"),
    pytest.param(
        "jsonl.zst",
        id="jsonl.zst",
        marks=pytest.mark.skipif(not _have("zstandard"), reason="zstandard not installed"),
    ),
    pytest.param(
        "parquet",
        id="parquet",
        marks=pytest.mark.skipif(
            not (_have("pandas") and _have("pyarrow")),
            reason='the "parquet" extra (pandas + pyarrow) is not installed',
        ),
    ),
    pytest.param("msgpack", id="msgpack"),
]


@pytest.mark.parametrize("fmt", FORMAT_CASES)
def test_round_trip(tmp_path: Path, fmt: str):
    out = export_traces(RECORDS, tmp_path / f"traces.{fmt}")
    assert out.exists() and out.stat().st_size > 0
    assert load_traces(out) == RECORDS


@pytest.mark.parametrize("fmt", FORMAT_CASES)
def test_explicit_fmt_overrides_suffix(tmp_path: Path, fmt: str):
    out = export_traces(RECORDS, tmp_path / "traces.bin", fmt=fmt)
    assert load_traces(out, fmt=fmt) == RECORDS


def test_infer_format_compound_suffixes():
    assert infer_format(Path("t.jsonl")) == "jsonl"
    assert infer_format(Path("t.jsonl.gz")) == "jsonl.gz"
    assert infer_format(Path("t.jsonl.zst")) == "jsonl.zst"
    assert infer_format(Path("t.parquet")) == "parquet"
    assert infer_format(Path("t.msgpack")) == "msgpack"


def test_unknown_suffix_raises():
    with pytest.raises(ValueError, match="cannot infer"):
        export_traces(RECORDS, Path("traces.csv"))


def test_unknown_fmt_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown format"):
        export_traces(RECORDS, tmp_path / "t.jsonl", fmt="csv")


def test_export_from_path_input(tmp_path: Path):
    src = export_traces(RECORDS, tmp_path / "src.jsonl")
    out = export_traces(src, tmp_path / "copy.msgpack")
    assert load_traces(out) == RECORDS


def test_atomic_replace_of_existing_file(tmp_path: Path):
    out = tmp_path / "traces.jsonl"
    export_traces(RECORDS[:1], out)
    export_traces(RECORDS, out)  # update in place, atomically
    assert load_traces(out) == RECORDS
    assert list(tmp_path.glob("*.tmp")) == []  # no temp litter


def test_gzip_is_actually_compressed(tmp_path: Path):
    plain = export_traces(RECORDS, tmp_path / "t.jsonl")
    gz = export_traces(RECORDS, tmp_path / "t.jsonl.gz")
    assert gz.stat().st_size < plain.stat().st_size


def test_push_traces_dry_run():
    report = push_traces(RECORDS, "someone/some-traces", dry_run=True)
    assert "dry-run" in report
    assert "3 rows" in report
    assert "someone/some-traces" in report
    assert "private=True" in report
    assert "events" in report  # nested column named in the features report


def test_push_traces_dry_run_from_path(tmp_path: Path):
    src = export_traces(RECORDS, tmp_path / "src.jsonl")
    report = push_traces(src, "someone/some-traces", private=False, dry_run=True)
    assert "3 rows" in report and "private=False" in report
