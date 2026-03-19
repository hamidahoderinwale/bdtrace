"""
Evaluation metrics for inferred representations.

is_grounded, grounding_check_score: used for grounding verification (e.g. check_*.py).
"""

import re
from typing import Any

_UNGROUNDED_PATTERNS = re.compile(
    r"^(none|n/a|na|\.|-|—|—)$",
    re.IGNORECASE,
)


def is_grounded(annotation: dict[str, Any], repr_type: str = "behavioral") -> bool:
    """
    Check if annotation has a non-trivial grounding reference.

    Ungrounded: empty, "none", "n/a", etc.
    """
    grounded = annotation.get("grounded")
    if grounded is None:
        return False
    s = str(grounded).strip()
    if not s:
        return False
    if _UNGROUNDED_PATTERNS.match(s):
        return False
    return True


def grounding_check_score(
    annotations: list[dict[str, Any]],
    repr_type: str = "behavioral",
) -> float:
    """
    Fraction of annotations that are grounded (non-empty, non-trivial grounding ref).

    Returns 0.0 if no annotations.
    """
    if not annotations:
        return 0.0
    grounded_count = sum(1 for a in annotations if is_grounded(a, repr_type))
    return grounded_count / len(annotations)

