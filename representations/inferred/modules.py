"""
Composed DSPy modules for inferred representations.

InferredRepresentationsModule runs behavioral, mechanistic, and functional
annotators and returns all three. Use for batch processing or when you need
the full multi-perspective annotation.
"""

from typing import Any

import dspy

from .behavioral import BehavioralModule
from .functional import FunctionalModule
from .mechanistic import MechanisticModule


class InferredRepresentationsModule(dspy.Module):
    """
    Composed module: behavioral + mechanistic + functional.

    Call with before_fn, after_fn, structural_certificate, module_context.
    Returns dict with behavioral, mechanistic, functional keys.
    """

    def __init__(
        self,
        behavioral: BehavioralModule | None = None,
        mechanistic: MechanisticModule | None = None,
        functional: FunctionalModule | None = None,
    ):
        super().__init__()
        self.behavioral = behavioral or BehavioralModule()
        self.mechanistic = mechanistic or MechanisticModule()
        self.functional = functional or FunctionalModule()

    def forward(
        self,
        before_fn: str,
        after_fn: str,
        structural_certificate: dict[str, Any] | list[dict] | None = None,
        module_context: dict[str, Any] | list[str] | None = None,
        *,
        embed_model: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        behavioral = self.behavioral(
            before_fn=before_fn,
            after_fn=after_fn,
            structural_certificate=structural_certificate,
            embed_model=embed_model,
            run_id=run_id,
        )
        mechanistic = self.mechanistic(
            before_fn=before_fn,
            after_fn=after_fn,
            structural_certificate=structural_certificate,
            embed_model=embed_model,
            run_id=run_id,
        )
        functional = self.functional(
            before_fn=before_fn,
            after_fn=after_fn,
            module_context=module_context,
            embed_model=embed_model,
            run_id=run_id,
        )
        return {
            "behavioral": behavioral,
            "mechanistic": mechanistic,
            "functional": functional,
        }
