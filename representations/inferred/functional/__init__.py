"""Functional: inferred, architect's perspective."""

from .distance import (
    DEFAULT_WEIGHTS,
    aggregate_distance,
    grounding_overlap,
    impact_distance,
    role_distance,
)
from .functional import FunctionalModule, FunctionalSignature, functional_repr

__all__ = [
    "DEFAULT_WEIGHTS",
    "functional_repr",
    "role_distance",
    "impact_distance",
    "grounding_overlap",
    "aggregate_distance",
    "FunctionalModule",
    "FunctionalSignature",
]
