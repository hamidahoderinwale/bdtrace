"""
Mechanistic: inferred, intra-function, edits certificate.

Input: function source + edits certificate.
Grounding: edits certificate. Unit: transformation pattern.
Distance: distance.cosine_distance (sentence-transformers).
Implementer's perspective: how the change was made.
"""

from typing import Any

import dspy

from ..utils import embed_text, format_structural_certificate, provenance


class MechanisticSignature(dspy.Signature):
    """Describe only internal mechanism. Do not describe behavioral consequences."""

    before_fn = dspy.InputField(desc="function source before edit")
    after_fn = dspy.InputField(desc="function source after edit")
    structural_certificate = dspy.InputField(desc="typed edits certificate")
    mechanism = dspy.OutputField(desc="ordered sequence of internal steps")
    pattern = dspy.OutputField(desc="transformation pattern name")
    locations = dspy.OutputField(desc="AST node types and positions")
    grounded = dspy.OutputField(desc="which certificate fields anchor each step")


class MechanisticModule(dspy.Module):
    """DSPy module for mechanistic representation. Supports save/load and batch."""

    def __init__(self, predictor: dspy.Predict | None = None):
        super().__init__()
        self.predictor = predictor or dspy.Predict(MechanisticSignature)

    def forward(
        self,
        before_fn: str,
        after_fn: str,
        structural_certificate: dict[str, Any] | list[dict] | None = None,
        *,
        embed_model: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        cert = structural_certificate or {}
        cert_str = format_structural_certificate(cert)
        out = self.predictor(
            before_fn=before_fn or "(empty)",
            after_fn=after_fn or "(empty)",
            structural_certificate=cert_str,
        )
        mechanism = getattr(out, "mechanism", "") or ""
        return {
            "steps": mechanism.split("\n") if mechanism else [],
            "mechanism": mechanism,
            "pattern": getattr(out, "pattern", "") or "",
            "locations": getattr(out, "locations", "") or "",
            "grounded_in": cert,
            "embedding": embed_text(mechanism, embed_model),
            "provenance": provenance(cert, run_id=run_id),
        }


def mechanistic_repr(
    before_fn: str,
    after_fn: str,
    structural_certificate: dict[str, Any] | list[dict] | None = None,
    embed_model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Describe internal mechanism. Returns mechanism, pattern, locations, embedding."""
    return MechanisticModule()(
        before_fn=before_fn,
        after_fn=after_fn,
        structural_certificate=structural_certificate,
        embed_model=embed_model,
        run_id=run_id,
    )
