"""CLI seam tests: dispatch, aliases, readouts. No network, no LLM, no scripts run."""

import json

import pytest

from bdtrace import cli


def run_cli(argv, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["bdtrace", *argv])
    code = 0
    exit_msg = ""
    try:
        cli.main()
    except SystemExit as e:
        # sys.exit(str) keeps the message in e.code; the interpreter would print it
        # to stderr at exit, which pytest never reaches — surface it as stderr here
        if isinstance(e.code, int):
            code = e.code
        else:
            code, exit_msg = 1, str(e.code or "")
    out = capsys.readouterr()
    return code, out.out, out.err + exit_msg


@pytest.fixture
def traces_file(tmp_path):
    recs = [
        {"instance_id": "a", "repo": "r", "events": [
            {"type": "prompt", "details": {"text": "fix it"}, "timestamp": "2026-01-05T00:00:00Z"}]},
        {"instance_id": "b", "repo": None, "events": [
            {"type": "edit", "details": {}, "timestamp": "2026-02-05T00:00:00Z"}]},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_help_and_version(monkeypatch, capsys):
    code, out, _ = run_cli(["--help"], monkeypatch, capsys)
    assert code == 0 and "trace import" in out
    code, out, _ = run_cli(["--version"], monkeypatch, capsys)
    assert code == 0 and out.strip()[0].isdigit()


def test_unknown_command_fails(monkeypatch, capsys):
    code, _, err = run_cli(["bogus"], monkeypatch, capsys)
    assert code != 0 and "unknown command" in err


def test_spec_text_and_audit(traces_file, monkeypatch, capsys):
    code, out, _ = run_cli(["trace", "spec"], monkeypatch, capsys)
    assert code == 0 and "instance_id" in out
    code, out, _ = run_cli(["trace", "spec", "--in", str(traces_file)], monkeypatch, capsys)
    assert code == 0 and "2 records" in out and "prompt 1" in out


def test_head_interval_and_slice(traces_file, monkeypatch, capsys, tmp_path):
    code, out, _ = run_cli(["trace", "head", "--in", str(traces_file),
                            "--interval", "2026-01-01..2026-02-01"], monkeypatch, capsys)
    assert code == 0 and '"a"' in out and '"b"' not in out
    sliced = tmp_path / "s.jsonl"
    code, _, err = run_cli(["trace", "head", "--in", str(traces_file), "-n", "1",
                            "--skip", "1", "--out", str(sliced)], monkeypatch, capsys)
    assert code == 0 and json.loads(sliced.read_text())["instance_id"] == "b" and "1 records" in err


def test_transform_list_via_alias(monkeypatch, capsys):
    code, out, _ = run_cli(["tf", "list"], monkeypatch, capsys)
    assert code == 0 and "edits" in out and "inferred" in out


def test_trace_usage_lists_module_verbs(monkeypatch, capsys):
    code, _, err = run_cli(["trace"], monkeypatch, capsys)
    assert code != 0 and "import|export|push|spec|head" in err
