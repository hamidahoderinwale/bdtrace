"""
Provenance for inferred annotations.

Records model, grounding hash, timestamp, and optional run_id so annotations
can be traced back to their inputs and reproduced.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def hash_grounding(grounding: dict[str, Any] | list[Any] | str | None) -> str:
    """Stable hash of grounding (certificate or module graph). Links annotation to source."""
    if grounding is None:
        return ""
    try:
        canonical = json.dumps(grounding, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        return ""


def get_dspy_model_name() -> str:
    """Current LM model from DSPy settings, or empty if not configured."""
    import dspy

    lm = getattr(dspy.settings, "lm", None)
    if lm is None and hasattr(dspy.settings, "get"):
        lm = dspy.settings.get("lm")
    if lm is not None and hasattr(lm, "model"):
        return str(lm.model) or ""
    return ""


def provenance(
    grounding: dict[str, Any] | list[Any] | str | None,
    *,
    run_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Build provenance dict for an annotation.

    Args:
        grounding: Edits certificate or module context — hashed for traceability.
        run_id: Optional experiment/run identifier for batching.
        model: Override model name; defaults to dspy.settings.lm.model.

    Returns:
        Dict with grounding_hash, model, timestamp_utc, run_id (if provided).
    """
    return {
        "grounding_hash": hash_grounding(grounding),
        "model": model or get_dspy_model_name(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **({"run_id": run_id} if run_id else {}),
    }
