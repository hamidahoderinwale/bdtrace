import json

from ..core.utils import _extract_code_tokens


def tokens_repr(
    trace: dict,
    include_prompts: bool = True,
    parsers: list[str] | None = None,
) -> list[str]:
    """
    Tokens: computed, intra-function, baseline.
    Input: raw source. Unit: character/token.
    Analysis: edit distance (difflib, python-Levenshtein).

    Args:
        trace: Event trace dict
        include_prompts: Whether to include prompt-derived tokens
        parsers: Languages to use AST parsers for. Default ["python"] (stdlib only).
            Add "javascript" or "java" to use esprima/javalang when installed.
    """
    if not trace or not isinstance(trace, dict):
        return []

    tokens = []
    events = trace.get("events", [])
    if not events:
        return []

    for event in events:
        if not isinstance(event, dict):
            continue

        try:
            event_type = (event.get("type") or "").lower()
            details = event.get("details", {})

            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except (json.JSONDecodeError, TypeError):
                    details = {}

            if isinstance(details, dict):
                code_content = details.get("after_content") or details.get("before_content") or details.get("code", "")
                file_path = details.get("file_path") or details.get("file")

                if code_content and isinstance(code_content, str):
                    try:
                        code_tokens = _extract_code_tokens(
                            code_content, file_path, parsers=parsers
                        )
                        tokens.extend(code_tokens[:200])
                        continue
                    except (SyntaxError, ValueError, TypeError, OSError):
                        pass

            kind = event.get("type") or event.get("annotation") or event.get("intent")
            if kind:
                tokens.append(str(kind))
        except (KeyError, TypeError, ValueError):
            continue

    return tokens


def tokens_repr_str(
    trace: dict, limit: int = 200, parsers: list[str] | None = None
) -> str:
    """Extract tokens as a string representation."""
    tokens = tokens_repr(trace, include_prompts=True, parsers=parsers)
    if not tokens:
        return "EMPTY_TRACE"
    token_str = " ".join(tokens[:limit])
    if len(tokens) > limit:
        token_str += f" ... [truncated from {len(tokens)} tokens]"
    return token_str
