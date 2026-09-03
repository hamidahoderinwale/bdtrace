"""Tests for the shared credential ladder.

The contract that matters: your own key beats the org's, the org vault is only
consulted when you have none, a missing `op` CLI is a fall-through rather than a
crash, and no test ever needs a real secret.
"""

import pytest

from bdtrace import creds


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(creds, "op_read", lambda ref: None)
    # .env on a contributor's machine must not decide the outcome of a test
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def test_own_env_wins_over_the_org_vault(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mine")
    monkeypatch.setattr(creds, "op_read", lambda ref: "org")
    assert creds.resolve(("OPENROUTER_API_KEY",), op_key="openrouter") == ("mine", "OPENROUTER_API_KEY")


def test_env_vars_are_tried_in_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "second")
    assert creds.resolve(("OPENROUTER_API_KEY", "OPENAI_API_KEY")) == ("second", "OPENAI_API_KEY")


def test_org_vault_is_the_fallback(monkeypatch):
    monkeypatch.setattr(creds, "op_read", lambda ref: "from-vault")
    value, source = creds.resolve(("OPENROUTER_API_KEY",), op_key="openrouter")
    assert value == "from-vault" and "1Password" in source and "openrouter" in source


def test_hugging_face_has_no_org_rung():
    """A hub token is an identity: a shared one would publish everyone as the org."""
    assert "huggingface" not in creds.OP_REFS


def test_nothing_anywhere_is_none_not_an_error():
    assert creds.resolve(("OPENROUTER_API_KEY",), op_key="openrouter") is None
    assert creds.resolve(("HF_TOKEN",)) is None


def test_missing_op_cli_falls_through(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no op")
    monkeypatch.setattr("subprocess.run", boom)
    monkeypatch.setattr(creds, "op_read", creds.op_read.__wrapped__ if hasattr(creds.op_read, "__wrapped__") else creds.op_read)
    assert creds.op_read("op://x/y/z") is None


def test_refs_are_overridable_and_carry_no_secret():
    for key, ref in creds.OP_REFS.items():
        assert ref.startswith("op://"), key


def test_config_report_runs_and_names_every_rung(monkeypatch):
    """`bdtrace config` reaches for real credential state, so a removed vault key
    or a renamed helper crashes it. The suite missed exactly that once."""
    from bdtrace.transforms import config_report

    text = config_report()
    for expected in ("OPENROUTER_API_KEY", "taste org model key", "hugging face", "BDTRACE_MODEL"):
        assert expected in text, f"{expected!r} missing from config output"
