"""Inferred representations: behavioral, mechanistic, functional."""

from .behavioral import BehavioralModule, BehavioralSignature, behavioral_repr, claim_distance, field_distances
from .behavioral import aggregate_distance as behavioral_aggregate_distance
from .functional import (
    FunctionalModule,
    FunctionalSignature,
    functional_repr,
    grounding_overlap,
    impact_distance,
    role_distance,
)
from .functional import aggregate_distance as functional_aggregate_distance
from .mechanistic import (
    MechanisticModule,
    MechanisticSignature,
    location_overlap,
    mechanism_distance,
    mechanistic_repr,
    pattern_distance,
)
from .mechanistic import aggregate_distance as mechanistic_aggregate_distance
from .modules import InferredRepresentationsModule

__all__ = [
    "behavioral_repr",
    "mechanistic_repr",
    "functional_repr",
    "claim_distance",
    "field_distances",
    "behavioral_aggregate_distance",
    "pattern_distance",
    "mechanism_distance",
    "location_overlap",
    "mechanistic_aggregate_distance",
    "role_distance",
    "impact_distance",
    "grounding_overlap",
    "functional_aggregate_distance",
    "BehavioralModule",
    "BehavioralSignature",
    "MechanisticModule",
    "MechanisticSignature",
    "FunctionalModule",
    "FunctionalSignature",
    "InferredRepresentationsModule",
]
