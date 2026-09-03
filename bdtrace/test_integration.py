"""The bdtrace -> procgrep composition, held in place from this side.

The README tells people to pipe an export straight into procgrep's canonicalizer:

    bdtrace import --out traces.jsonl
    procgrep canonicalize --input traces.jsonl --output atoms.jsonl \\
        --adapter bdtrace --trace-id-field instance_id

That line spans two repos, so nothing in either one's CI notices when it stops
working. Two directions can break, and each gets a test.

`test_event_taxonomy_matches_adapter_coverage` guards the half this repo owns.
It reads `bdtrace.spec.EVENT_TYPES` and asserts it is exactly the set of event
types the procgrep `bdtrace` adapter knows how to map. Adding a new event type
here fails that test immediately, which is the point: the failure is the
reminder to teach procgrep about it before the type reaches an export. It needs
no procgrep and runs everywhere.

`test_canonicalize_through_procgrep` guards the other half by actually running
the documented command over a synthetic export covering every event type, and
asserting the exact atom sequence that comes back. procgrep is deliberately not
a dependency of this repo -- it is the analysis half, installed in its own venv --
so this test resolves a procgrep CLI at runtime and skips with a clear reason
when there isn't one.

On locating fields in procgrep's output: procgrep owns that schema and this repo
cannot import it, so `_atom_sequence` and `_field` search a short list of
plausible key names rather than hardcoding one. The *values* are still asserted
exactly. A rename on procgrep's side surfaces as a loud "could not find ..."
failure listing the keys that were actually present, which is itself a report
that the composition drifted.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bdtrace.spec import EVENT_TYPES, parse_types

# procgrep's `bdtrace` adapter maps one bdtrace event type to one canonical atom,
# except `run`, which is disambiguated by the command it carries. `code_change`
# is the legacy resolved-trace spelling of `edit` (see spec.parse_types), kept
# because old exports still carry it.
ADAPTER_ATOM_MAP: dict[str, str | None] = {
    "prompt": "prompt_ai",
    "edit": "edit",
    "code_change": "edit",  # legacy alias, not emitted by current imports
    "read": "read_file",
    "search": "search_repo",
    "test": "run_test",
    "run": None,  # command-dependent: version_control or run_code
    "other": "other",
}

# Event types the adapter covers that bdtrace no longer emits. Kept separate so
# the taxonomy assertion below is an equality, not a subset check.
LEGACY_EVENT_TYPES = {"code_change"}

# A `run` whose command starts with one of these is version control, not code.
VERSION_CONTROL_COMMANDS = ("git", "gh", "hg", "svn")


def test_event_taxonomy_matches_adapter_coverage():
    """The set of event types this repo can emit == the set procgrep can map.

    Failing here means one of two things, and the diff says which: a new event
    type was added to `bdtrace.spec.EVENT_TYPES` and procgrep's `bdtrace`
    adapter has not been taught to map it (exports will canonicalize that type
    to nothing useful), or a type was dropped here and the map above is stale.
    """
    emitted = set(EVENT_TYPES)
    covered = set(ADAPTER_ATOM_MAP) - LEGACY_EVENT_TYPES

    assert emitted == covered, (
        "bdtrace's event taxonomy has drifted from procgrep's bdtrace adapter.\n"
        f"  emitted here but unmapped in procgrep: {sorted(emitted - covered)}\n"
        f"  mapped in procgrep but no longer emitted: {sorted(covered - emitted)}\n"
        "Teach procgrep's adapter about the new type(s), then update "
        "ADAPTER_ATOM_MAP in this file."
    )


def test_legacy_event_types_still_accepted_by_spec():
    """The legacy aliases the adapter carries are the ones spec.py still accepts.

    `code_change` is only worth mapping in procgrep for as long as this repo's
    own filters tolerate it; if spec.parse_types stops accepting it, the adapter
    entry is dead weight and the map above should drop it too.
    """
    assert parse_types(",".join(sorted(LEGACY_EVENT_TYPES))) == LEGACY_EVENT_TYPES


def test_taxonomy_map_is_self_consistent():
    """Every mapped type resolves to an atom, except the command-dependent one."""
    unresolved = {t for t, atom in ADAPTER_ATOM_MAP.items() if atom is None}
    assert unresolved == {"run"}, f"only `run` is command-dependent; found {sorted(unresolved)}"


# Where a procgrep checkout's own venv tends to live. The analysis scripts in
# scripts/agent_trajectories_paper/ point at the same layout on the other
# machine, hence both home directories.
_CANDIDATE_BINS = (
    Path.home() / "learning-from-dev" / "procgrep" / ".venv" / "bin" / "procgrep",
    Path("/Users/hamidah/learning-from-dev/procgrep/.venv/bin/procgrep"),
    Path("/Users/hamidaho/learning-from-dev/procgrep/.venv/bin/procgrep"),
)


def _runnable(cmd: list[str]) -> bool:
    """True when `cmd --help` actually executes. Swallows OSError because a
    sandboxed or permission-restricted run raises rather than returning nonzero,
    and that is a skip, not a failure."""
    try:
        proc = subprocess.run(cmd + ["--help"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _resolve_procgrep() -> tuple[list[str] | None, str]:
    """Find a runnable procgrep CLI, or explain why there isn't one.

    Order: an explicit PROCGREP_BIN override, this venv's PATH, procgrep
    importable from this interpreter, then the known checkout venvs.
    """
    tried: list[str] = []

    override = os.environ.get("PROCGREP_BIN")
    if override:
        tried.append(f"$PROCGREP_BIN={override}")
        if _runnable([override]):
            return [override], ""

    on_path = shutil.which("procgrep")
    if on_path:
        tried.append(f"PATH: {on_path}")
        if _runnable([on_path]):
            return [on_path], ""

    try:
        importable = importlib.util.find_spec("procgrep") is not None
    except (ImportError, ValueError):
        importable = False
    if importable:
        tried.append(f"{sys.executable} -m procgrep")
        if _runnable([sys.executable, "-m", "procgrep"]):
            return [sys.executable, "-m", "procgrep"], ""

    for candidate in dict.fromkeys(str(p) for p in _CANDIDATE_BINS):
        tried.append(candidate)
        if _runnable([candidate]):
            return [candidate], ""

    return None, (
        "procgrep is not available, so the bdtrace -> procgrep composition was not exercised. "
        "procgrep is intentionally NOT a dependency of this repo; it lives in its own checkout "
        "and venv. Point PROCGREP_BIN at a procgrep executable to run this test. "
        f"Looked for a runnable CLI at: {', '.join(tried)}."
    )


def _require_procgrep() -> list[str]:
    cmd, reason = _resolve_procgrep()
    if cmd is None:
        pytest.skip(reason)
    return cmd


# One synthetic export exercising every branch of the adapter, in order. The
# expected atom sequence below is positionally aligned with these events.
TRACE_UNDER_TEST = {
    "instance_id": "claude-code-integration-1",
    "agent": "claude-code",
    "repo": "hamidahoderinwale/bdtrace",
    "base_commit": "0" * 40,
    "events": [
        {"type": "prompt", "details": {"text": "make the export feed procgrep"},
         "timestamp": "2026-01-01T00:00:00Z"},
        {"type": "read", "details": {"tool": "Read", "file_path": "bdtrace/export.py"},
         "timestamp": "2026-01-01T00:01:00Z"},
        {"type": "search", "details": {"tool": "Grep", "pattern": "canonicalize"},
         "timestamp": "2026-01-01T00:02:00Z"},
        {"type": "edit", "details": {"tool": "Edit", "file_path": "bdtrace/export.py"},
         "timestamp": "2026-01-01T00:03:00Z"},
        {"type": "code_change", "details": {"file_path": "bdtrace/spec.py",
                                            "before_content": "x = 1\n", "after_content": "x = 2\n"},
         "timestamp": "2026-01-01T00:04:00Z"},
        {"type": "test", "details": {"tool": "Bash", "command": "pytest bdtrace -q"},
         "timestamp": "2026-01-01T00:05:00Z"},
        # one `run` per version-control front-end, then a plain one
        *[
            {"type": "run", "details": {"tool": "Bash", "command": f"{vc} status"},
             "timestamp": f"2026-01-01T00:0{6 + i}:00Z"}
            for i, vc in enumerate(VERSION_CONTROL_COMMANDS)
        ],
        {"type": "run", "details": {"tool": "Bash", "command": "python -c 'print(1)'"},
         "timestamp": "2026-01-01T00:10:00Z"},
        {"type": "other", "details": {"description": "thinking"},
         "timestamp": "2026-01-01T00:11:00Z"},
    ],
    "prompts": ["make the export feed procgrep"],
}

EXPECTED_ATOMS = [
    "prompt_ai",
    "read_file",
    "search_repo",
    "edit",       # edit
    "edit",       # code_change, the legacy spelling
    "run_test",
    *["version_control"] * len(VERSION_CONTROL_COMMANDS),
    "run_code",
    "other",
]

# A second record, so multi-record canonicalization and per-record agent
# attribution are both covered.
SECOND_TRACE = {
    "instance_id": "cursor-integration-2",
    "agent": "cursor",
    "repo": "hamidahoderinwale/bdtrace",
    "base_commit": "1" * 40,
    "events": [
        {"type": "prompt", "details": {"text": "and again"}, "timestamp": "2026-01-02T00:00:00Z"},
    ],
    "prompts": ["and again"],
}

# procgrep owns the canonical-trace schema; these are the plausible spellings.
_ATOM_KEYS = ("atoms",)  # procgrep writes the atom list here (traces_to_records)
_ATOM_ITEM_KEYS = ("atom", "type", "name", "action", "label")
_TRACE_ID_KEYS = ("trace_id",)  # set from --trace-id-field
_NESTED_KEYS = ("metadata", "meta", "trace", "attrs")


def _field(record: dict, keys: tuple[str, ...]):
    """First present value among `keys`, looking one level into metadata dicts."""
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    for nest in _NESTED_KEYS:
        inner = record.get(nest)
        if isinstance(inner, dict):
            for key in keys:
                if inner.get(key) is not None:
                    return inner[key]
    return None


def _atom_sequence(record: dict) -> list[str]:
    """Pull the ordered atom labels out of one canonical trace record."""
    raw = _field(record, _ATOM_KEYS)
    if not isinstance(raw, list):
        pytest.fail(
            "could not find an atom sequence on procgrep's canonical trace. "
            f"Looked for {list(_ATOM_KEYS)}; record keys were {sorted(record)}. "
            "If procgrep renamed the field, update _ATOM_KEYS in this file."
        )
    atoms = []
    for item in raw:
        if isinstance(item, str):
            atoms.append(item)
        elif isinstance(item, dict):
            label = _field(item, _ATOM_ITEM_KEYS)
            if label is None:
                pytest.fail(f"atom entry has no recognizable label; keys were {sorted(item)}")
            atoms.append(str(label))
        else:
            pytest.fail(f"unexpected atom entry of type {type(item).__name__}: {item!r}")
    return atoms


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_canonicalize_through_procgrep(tmp_path: Path):
    """Run the README's command for real and check every atom it produces.

    This is the test that fails when procgrep's `bdtrace` adapter changes shape,
    loses a mapping, or stops recognizing the export format.
    """
    procgrep = _require_procgrep()

    traces = tmp_path / "traces.jsonl"
    with open(traces, "w") as f:
        for record in (TRACE_UNDER_TEST, SECOND_TRACE):
            f.write(json.dumps(record) + "\n")

    atoms_path = tmp_path / "atoms.jsonl"
    cmd = procgrep + [
        "canonicalize",
        "--input", str(traces),
        "--output", str(atoms_path),
        "--adapter", "bdtrace",
        "--trace-id-field", "instance_id",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode == 0, (
        f"`{' '.join(cmd)}` exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert atoms_path.exists(), f"procgrep reported success but wrote no output\nstdout: {proc.stdout}"

    canonical = _read_jsonl(atoms_path)
    assert len(canonical) == 2, f"expected one canonical trace per input record, got {len(canonical)}"

    by_id = {}
    for record in canonical:
        trace_id = _field(record, _TRACE_ID_KEYS)
        assert trace_id is not None, (
            "--trace-id-field instance_id did not put an id on the canonical trace; "
            f"record keys were {sorted(record)}"
        )
        by_id[str(trace_id)] = record

    assert set(by_id) == {TRACE_UNDER_TEST["instance_id"], SECOND_TRACE["instance_id"]}, (
        f"trace ids did not survive canonicalization: got {sorted(by_id)}"
    )

    main = by_id[TRACE_UNDER_TEST["instance_id"]]
    assert _atom_sequence(main) == EXPECTED_ATOMS

    # the agent label is what every downstream procgrep comparison groups by,
    # so it has to survive the adapter
    for record_in in (TRACE_UNDER_TEST, SECOND_TRACE):
        out = by_id[record_in["instance_id"]]
        assert _field(out, ("agent",)) == record_in["agent"], (
            f"agent did not survive onto the canonical trace for {record_in['instance_id']}; "
            f"record keys were {sorted(out)}"
        )

    assert _atom_sequence(by_id[SECOND_TRACE["instance_id"]]) == ["prompt_ai"]


def test_every_emitted_event_type_is_exercised_by_the_pipe_fixture():
    """The synthetic export above actually covers the whole taxonomy.

    Without this, a new event type could be added to EVENT_TYPES, mapped in
    procgrep, added to ADAPTER_ATOM_MAP -- and still never travel through the
    real pipe. Runs everywhere; it only inspects this file's fixture.
    """
    exercised = {e["type"] for e in TRACE_UNDER_TEST["events"]}
    assert exercised == set(ADAPTER_ATOM_MAP), (
        "TRACE_UNDER_TEST does not exercise every mapped event type; "
        f"missing {sorted(set(ADAPTER_ATOM_MAP) - exercised)}"
    )
    assert len(EXPECTED_ATOMS) == len(TRACE_UNDER_TEST["events"]), (
        "EXPECTED_ATOMS must stay positionally aligned with TRACE_UNDER_TEST['events']"
    )
