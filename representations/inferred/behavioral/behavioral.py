"""
Behavioral: inferred, intra-function, edits certificate.

Input: function source + edits certificate.
Grounding: edits certificate. Unit: behavioral claim.
Distance: distance.cosine_distance (sentence-transformers).
Caller's perspective: what changed about the contract.
"""

from typing import Any

import dspy

from ..utils import embed_text, format_structural_certificate, provenance


class BehavioralSignature(dspy.Signature):
    """Describe only input-output behavioral consequences. Ground every claim in the certificate."""

    before_fn = dspy.InputField(desc="function source before edit")
    after_fn = dspy.InputField(desc="function source after edit")
    structural_certificate = dspy.InputField(desc="typed edits certificate")
    claim = dspy.OutputField(desc="one-sentence behavioral change claim")
    before_behavior = dspy.OutputField(desc="what callers could observe before")
    after_behavior = dspy.OutputField(desc="what callers can observe after")
    testable = dspy.OutputField(desc="concrete input/output pair that confirms the claim")
    grounded = dspy.OutputField(desc="which certificate field entails this claim")


class BehavioralModule(dspy.Module):
    """DSPy module for behavioral representation. Supports save/load and batch."""

    def __init__(self, predictor: dspy.Predict | None = None):
        super().__init__()
        self.predictor = predictor or dspy.Predict(BehavioralSignature)

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
        claim = getattr(out, "claim", "") or ""
        grounded = getattr(out, "grounded", "") or ""
        return {
            "claim": claim,
            "before": getattr(out, "before_behavior", "") or "",
            "after": getattr(out, "after_behavior", "") or "",
            "testable": getattr(out, "testable", "") or "",
            "grounded": grounded,
            "grounded_in": cert,
            "embedding": embed_text(claim, embed_model),
            "provenance": provenance(cert, run_id=run_id),
        }


def behavioral_repr(
    before_fn: str,
    after_fn: str,
    structural_certificate: dict[str, Any] | list[dict] | None = None,
    embed_model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Describe input-output behavioral consequences. Returns claim, before/after, testable, embedding."""
    module = BehavioralModule()
    return module(
        before_fn=before_fn,
        after_fn=after_fn,
        structural_certificate=structural_certificate,
        embed_model=embed_model,
        run_id=run_id,
    )
