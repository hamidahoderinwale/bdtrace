"""
Functional: inferred, inter-file, module graph.

Input: function source + module graph.
Grounding: module graph. Unit: system role.
Distance: distance.cosine_distance (sentence-transformers).
Architect's perspective: why this function exists.
"""

from typing import Any

import dspy

from ..utils import embed_text, format_module_context, provenance


class FunctionalSignature(dspy.Signature):
    """Describe function's role in the system. Ground every claim in module context."""

    before_fn = dspy.InputField(desc="function source before edit")
    after_fn = dspy.InputField(desc="function source after edit")
    module_context = dspy.InputField(desc="import edges and co-edit weights")
    role = dspy.OutputField(desc="function's purpose relative to callers and callees")
    role_change = dspy.OutputField(desc="how the function's role changed")
    impact = dspy.OutputField(desc="which callers or downstream components are affected")
    grounded = dspy.OutputField(desc="which module graph edges support the impact claim")


class FunctionalModule(dspy.Module):
    """DSPy module for functional representation. Supports save/load and batch."""

    def __init__(self, predictor: dspy.Predict | None = None):
        super().__init__()
        self.predictor = predictor or dspy.Predict(FunctionalSignature)

    def forward(
        self,
        before_fn: str,
        after_fn: str,
        module_context: dict[str, Any] | list[str] | None = None,
        *,
        embed_model: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        ctx = module_context or {}
        ctx_str = format_module_context(ctx)
        out = self.predictor(
            before_fn=before_fn or "(empty)",
            after_fn=after_fn or "(empty)",
            module_context=ctx_str,
        )
        role = getattr(out, "role", "") or ""
        return {
            "role": role,
            "role_change": getattr(out, "role_change", "") or "",
            "system_impact": getattr(out, "impact", "") or "",
            "grounded_in": ctx,
            "embedding": embed_text(role, embed_model),
            "provenance": provenance(ctx, run_id=run_id),
        }


def functional_repr(
    before_fn: str,
    after_fn: str,
    module_context: dict[str, Any] | list[str] | None = None,
    embed_model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Describe function's role in the system. Returns role, role_change, system_impact, embedding."""
    return FunctionalModule()(
        before_fn=before_fn,
        after_fn=after_fn,
        module_context=module_context,
        embed_model=embed_model,
        run_id=run_id,
    )
