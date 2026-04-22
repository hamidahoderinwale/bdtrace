"""Procedural diff — comparison primitive over agent trajectories.

Public API:
    diff(a, b) -> Diff
    signature(trajectories) -> Signature
    ood_score(trajectory, reference, level="edits") -> float
    build_reference_vocabulary(trajectories, min_count=1) -> dict

See `core` module docstring for the formal definition.
"""

from .core import (
    Diff,
    Signature,
    TrajectoryView,
    build_reference_vocabulary,
    diff,
    ood_items,
    ood_score,
    signature,
    view_from_patch,
    view_from_trace,
)
from .distances import edit_distance, module_distance, scope_distance, token_distance

__all__ = [
    "Diff",
    "Signature",
    "TrajectoryView",
    "build_reference_vocabulary",
    "diff",
    "edit_distance",
    "module_distance",
    "ood_items",
    "ood_score",
    "scope_distance",
    "signature",
    "token_distance",
    "view_from_patch",
    "view_from_trace",
]
