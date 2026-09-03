"""One credential ladder, used by every command that needs a key.

Order is always: your own environment, then `.env`, then the org's shared secret
in 1Password, then whatever the provider's own CLI already cached. The 1Password
step is what lets a taste org member run model-backed and hub-backed commands
with no key of their own: vault membership is the gate and their own `op` login
is the auth, so access is granted and revoked centrally. The op:// reference is
not a secret and is committed; the value never is, and only the source name is
ever printed.
"""

import os
import subprocess

# Org-shared secrets. Overridable so a different org, or a personal vault, works
# without a code change.
#
# Only shared SPEND belongs here. A Hugging Face token is an identity, not a
# budget: a push creates a dataset under whoever owns the token, so a shared one
# would publish everybody's traces as the org. The hub path uses the person's own
# login instead (see export.ensure_hf_login).
OP_REFS = {
    "openrouter": os.environ.get("BDTRACE_OP_OPENROUTER",
                                 "op://infra / preview/Shared - OpenRouter/credential"),
}


def op_read(ref: str) -> str | None:
    """Read one secret through the signed-in 1Password CLI. Absent CLI, absent
    session, or no access all mean the same thing here: fall through quietly."""
    try:
        out = subprocess.run(["op", "read", ref], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def resolve(env_vars: tuple[str, ...], op_key: str | None = None) -> tuple[str, str] | None:
    """(value, source) from the first rung that has it, or None. Loads `.env` first
    so a project-local key beats the org's, which is what a contributor expects."""
    from dotenv import load_dotenv

    load_dotenv()
    for var in env_vars:
        if os.environ.get(var):
            return os.environ[var], var
    if op_key:
        value = op_read(OP_REFS[op_key])
        if value:
            return value, f"1Password ({op_key}, taste org shared)"
    return None


def describe(op_key: str) -> str:
    """Whether the org secret is reachable right now, for `bdtrace config`."""
    return "reachable (op signed in)" if op_read(OP_REFS[op_key]) else \
        "not reachable (no op CLI / not signed in / not in org)"
