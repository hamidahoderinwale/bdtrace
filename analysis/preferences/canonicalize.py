"""Canonicalize raw SWE-agent action strings into comparable atomic tokens.

The raw action is a shell-like command string like `"pip install -e .[dev]\\n"`
or `"edit 1:1\\n...from marshmallow.fields import TimeDelta\\nend_of_edit\\n"`.
Surface form varies enormously across trajectories even when the underlying
practice is the same.

Canonicalization:
  1. Parse action into (verb, args).
  2. Type-tag file-path args (SRC_PY / TEST_PY / REPRO_PY / CONFIG_PY / DOC / OTHER).
  3. Strip non-semantic literals (specific line numbers, commit hashes, edit
     content).
  4. Preserve semantically important distinctions (running pytest vs grep;
     running tests vs running repro scripts).

Output: one canonical atom per action, like `OPEN_SRC_PY`, `EDIT_TEST_PY`,
`RUN_PYTEST_TEST_PY`, `SHELL_LS`, `SUBMIT`.

This is OverCode-style "remove non-semantic variation" canonicalization.
Downstream: BPE on sequences of canonical atoms discovers emergent motifs.

Design principles:
- The atom vocabulary is NOT pre-specified in the paper sense — it's the set
  of (verb, typed-arg) pairs that actually occur. Vocabulary size emerges
  from the data, not hand-designed.
- Type-tagging is semantic normalization, not emergent clustering. The
  emergent step is BPE on top of these atoms.
- Unknown / unparseable actions fall back to `UNKNOWN_<verb>` rather than
  `OTHER` so we can inspect and iterate.
"""

from __future__ import annotations

import re
from typing import Optional

# File-path typing (adapted from analysis/procedures/corpus_motifs.py)
_TEST_PATTERN = re.compile(r"(?:^|/)tests?/|test_|_test\.py$|/test_", re.IGNORECASE)
_REPRO_PATTERN = re.compile(r"reproduc|repro\.py|debug\.py|^/tmp/", re.IGNORECASE)
_CONFIG_PATTERN = re.compile(
    r"setup\.py|pyproject\.toml|setup\.cfg|settings\.py|conftest\.py|tox\.ini",
    re.IGNORECASE,
)
_DOC_PATTERN = re.compile(r"\.md$|\.rst$|README|CHANGELOG", re.IGNORECASE)
_PY_PATTERN = re.compile(r"\.py$")


def file_type(path: str) -> str:
    """Classify a file path into one of: SRC_PY, TEST_PY, REPRO_PY, CONFIG_PY, DOC, OTHER."""
    if not path:
        return "OTHER"
    if _REPRO_PATTERN.search(path):
        return "REPRO_PY"
    if _CONFIG_PATTERN.search(path):
        return "CONFIG_PY"
    if _TEST_PATTERN.search(path):
        return "TEST_PY"
    if _DOC_PATTERN.search(path):
        return "DOC"
    if _PY_PATTERN.search(path):
        return "SRC_PY"
    return "OTHER"


def _extract_path(text: str) -> Optional[str]:
    """Extract the first file-path-looking token."""
    m = re.search(r'([\w/._-]+\.(?:py|md|rst|cfg|toml|txt|yml|yaml))', text)
    return m.group(1) if m else None


def _looks_like_py_run(cmd: str) -> bool:
    """Does this look like a python / pytest / unittest invocation?"""
    return bool(re.match(r"(python3?|pytest|pytest-\d|py\.test|unittest)\b", cmd))


def _looks_like_shell(cmd: str) -> bool:
    """Does this look like a plain shell command?"""
    shell_verbs = {"ls", "cd", "pwd", "mkdir", "rm", "mv", "cp", "touch", "cat",
                   "echo", "grep", "find", "which", "head", "tail", "wc", "chmod",
                   "export", "env", "source"}
    first = cmd.split(None, 1)[0] if cmd.strip() else ""
    return first in shell_verbs


def _strip_env_prefix(line: str) -> str:
    """Strip `KEY=VALUE` env-var assignments that prefix shell commands.

    Example: "DJANGO_SETTINGS_MODULE=tests.settings python manage.py test" ->
             "python manage.py test"
    """
    parts = line.split()
    i = 0
    while i < len(parts) and re.match(r"^[A-Z_][A-Z0-9_]*=.*", parts[i]):
        i += 1
    return " ".join(parts[i:]) if i < len(parts) else line


# Known test-runner patterns that aren't a plain python/pytest invocation
_TEST_RUNNER_PATTERNS = re.compile(
    r"^(?:\./bin/test|\./tests/runtests\.py|\./run_multiple_times\.sh|"
    r"django-admin|manage\.py|runtests\.py)",
    re.IGNORECASE,
)


def canonicalize(raw_action: str) -> str:
    """Canonicalize a raw SWE-agent action string to a single atomic token.

    Returns strings like:
      OPEN_SRC_PY, EDIT_TEST_PY, CREATE_REPRO_PY, NAV, FIND_NAME,
      RUN_PYTEST_TEST_PY, RUN_PYTHON_REPRO_PY, SHELL_LS, SUBMIT, UNKNOWN_<verb>
    """
    if not raw_action or not raw_action.strip():
        return "EMPTY"

    # Normalize whitespace; take first "logical" line for multi-line actions
    s = raw_action.strip()
    first_line = s.split("\n", 1)[0].strip()
    # Strip leading env-var assignments (e.g. DJANGO_SETTINGS_MODULE=... or PYTHONPATH=...)
    first_line = _strip_env_prefix(first_line)
    # Strip leading comment markers
    if first_line.startswith("#"):
        return "COMMENT"

    # Handle SUBMIT early
    if "submit" in first_line.lower():
        return "SUBMIT"

    # Handle exit_cost / errors
    if first_line.startswith("exit_cost") or first_line.startswith("exit_error"):
        return "EXIT_ERROR"

    # Parse verb
    parts = first_line.split(None, 1)
    verb = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    # SWE-agent tool commands
    if verb == "open":
        path = _extract_path(rest)
        return f"OPEN_{file_type(path) if path else 'UNKNOWN'}"

    if verb == "create":
        path = _extract_path(rest)
        return f"CREATE_{file_type(path) if path else 'UNKNOWN'}"

    if verb == "edit":
        # edit uses multi-line; check full action for file context
        # SWE-agent's edit doesn't take file path (edits the open file), so
        # we don't know the file type from the edit command alone.
        # Type-tag via open_file in state would be nicer, but for now we flatten.
        return "EDIT"

    if verb in ("goto", "scroll_up", "scroll_down"):
        return "NAV"

    if verb in ("find_file", "find",):
        return "FIND_FILE"

    if verb in ("search_file", "search_dir", "search"):
        return "SEARCH"

    # Run commands (python / pytest / unittest)
    if _looks_like_py_run(first_line):
        path = _extract_path(rest)
        runtime = "PYTEST" if first_line.startswith(("pytest", "py.test")) else "PYTHON"
        ftype = file_type(path) if path else "ALL"
        return f"RUN_{runtime}_{ftype}"

    # Test-runner scripts (Django, custom ./bin/test, etc.)
    if _TEST_RUNNER_PATTERNS.match(first_line):
        return "RUN_TEST_SCRIPT"

    # pip / package installation
    if verb == "pip":
        return "RUN_PIP"

    # Linters / doc-builders / build tools
    if verb in ("pylint", "flake8", "mypy", "ruff", "black"):
        return "RUN_LINT"
    if verb in ("sphinx-build", "sphinx-apidoc"):
        return "RUN_DOCBUILD"
    if verb in ("make",):
        return "RUN_MAKE"
    if verb in ("sed", "awk"):
        return f"SHELL_{verb.upper()}"

    # Plain shell
    if _looks_like_shell(first_line):
        return f"SHELL_{verb.upper()}"

    # Anything else gets flagged with its verb for inspection
    return f"UNKNOWN_{verb.upper()}" if verb else "UNKNOWN_EMPTY"


def _extract_open_file_from_state(state: str | dict) -> Optional[str]:
    """Parse open_file out of a step's state (which may be a JSON string or dict)."""
    if not state:
        return None
    if isinstance(state, str):
        # state is sometimes a JSON string like '{"open_file": "path", ...}'
        try:
            import json
            state = json.loads(state)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(state, dict):
        of = state.get("open_file")
        if of and of != "n/a":
            return str(of)
    return None


def canonicalize_trajectory(raw_trajectory: list[dict]) -> list[str]:
    """Canonicalize a full trajectory (list of step dicts).

    Uses `state.open_file` to enrich bare EDIT / NAV atoms with the file type
    of the currently-open file.
    """
    out = []
    for step in raw_trajectory:
        action = step.get("action", "")
        canon = canonicalize(action)
        # Enrich EDIT and NAV with file type via state.open_file
        if canon in ("EDIT", "NAV"):
            of = _extract_open_file_from_state(step.get("state"))
            if of:
                canon = f"{canon}_{file_type(of)}"
        out.append(canon)
    return out


def vocab_stats(token_sequences: list[list[str]]) -> dict:
    """Summary stats over a corpus of canonicalized sequences."""
    from collections import Counter
    all_tokens = [t for seq in token_sequences for t in seq]
    counter = Counter(all_tokens)
    return {
        "n_sequences": len(token_sequences),
        "total_tokens": len(all_tokens),
        "unique_tokens": len(counter),
        "mean_seq_length": (len(all_tokens) / len(token_sequences)) if token_sequences else 0,
        "most_common_20": counter.most_common(20),
        "unknown_tokens": [(t, c) for t, c in counter.most_common() if t.startswith("UNKNOWN_")],
    }


if __name__ == "__main__":
    # Smoke test with the sample trajectory actions we inspected earlier
    samples = [
        "ls -F\n",
        "open setup.py\n",
        "pip install -e .[dev]\n",
        "create reproduce.py\n",
        "edit 1:1\nfrom marshmallow.fields import TimeDelta\nend_of_edit\n",
        "python reproduce.py\n",
        "find_file \"fields.py\" src\n",
        "pytest tests/test_fields.py\n",
        "goto 1474\n",
        "scroll_down\n",
        "submit\n",
    ]
    print("Canonicalization smoke test:")
    for action in samples:
        canon = canonicalize(action)
        action_snippet = action.strip()[:40]
        print(f"  {action_snippet!r:45s} -> {canon}")
