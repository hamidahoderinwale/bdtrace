"""
DSPy LM configuration for inferred representations.

Uses env vars:
  OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY
  DSPY_MODEL (e.g. openai/gpt-4o-mini)
  DSPY_TEMPERATURE, DSPY_MAX_TOKENS

When both keys are set, OpenRouter is used. For OpenRouter, model becomes openrouter/openai/gpt-4o-mini.
"""

import os
from typing import Any

_DSPY_CONFIGURED = False


def configure_dspy(
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    **kwargs: Any,
) -> bool:
    """
    Configure DSPy LM for inferred representations.

    Args:
        model: LM model string. Default from DSPY_MODEL env.
        api_key: API key. Default from OPENROUTER_API_KEY or OPENAI_API_KEY env.
        temperature: Sampling temperature. Default from DSPY_TEMPERATURE env or 0.0.
        max_tokens: Max tokens. Default from DSPY_MAX_TOKENS env or 1024.
        cache: Whether to cache responses.
        **kwargs: Passed to dspy.LM.

    Returns:
        True if configured, False if skipped (e.g. no API key).
    """
    global _DSPY_CONFIGURED

    import dspy

    # Prefer OpenRouter when set (user may have working key). Else OpenAI.
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    api_key = api_key or openrouter_key or openai_key
    if not api_key:
        return False

    model = model or os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini")
    use_openrouter = bool(openrouter_key)
    if use_openrouter:
        api_key = openrouter_key
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}" if "/" in model else f"openrouter/openai/{model}"
        os.environ["OPENROUTER_API_KEY"] = api_key
    else:
        if model.startswith("openrouter/"):
            model = "openai/gpt-4o-mini"

    temp_str = os.environ.get("DSPY_TEMPERATURE")
    temperature = temperature if temperature is not None else (float(temp_str) if temp_str else 0.0)
    max_str = os.environ.get("DSPY_MAX_TOKENS")
    max_tokens = max_tokens if max_tokens is not None else (int(max_str) if max_str else 1024)

    lm_kwargs = dict(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        **kwargs,
    )
    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    _DSPY_CONFIGURED = True
    return True


def is_configured() -> bool:
    """Return True if DSPy LM has been configured."""
    return _DSPY_CONFIGURED
