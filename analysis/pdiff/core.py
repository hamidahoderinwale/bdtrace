"""
Procedural diff — a composable comparison primitive over agent trajectories.

Formal definition
-----------------
A trajectory T is a record produced by an agent solving a task, exposing at
minimum a patch (sequence of file edits) and optionally a scope + module
footprint. The procedural-diff operation

    pdiff : T × T → Diff

produces a structured output with four bounded [0, 1] sub-distances, one per
representation level:

    Diff.tokens   ∈ [0, 1]   normalized Levenshtein on AST token sequences
    Diff.edits    ∈ [0, 1]   Dice on AST edit-operation sets
    Diff.scopes   ∈ [0, 1]   Jaccard on touched function/class scopes
    Diff.modules  ∈ [0, 1]   Jaccard on touched file stems

Each sub-distance is None when the corresponding level is unavailable for
either input (e.g. no parseable scopes). Zero means identical at that level;
one means disjoint. The Dice form on edits is bounded but not a true metric;
the other three are metrics.

Aggregations over a population of trajectories yield a `Signature`: per-level
mean distance to reference, per-level edit-op frequency vector, and the set
of edit ops seen. Signatures compose with reference vocabularies to produce
OOD scores via `ood_score`.

Wrapping, not replacing
-----------------------
`pdiff` composes existing extractors:

  * `analysis.procedures.ast_edit_sequences.patch_to_ast_sequence` — tokens
  * `analysis.procedures.scoped_edit_ops.trace_to_scoped_cert` — edit / scope / module
  * `analysis.procedures.scoped_edit_ops._NORMALIZE_OPS` — canonical edit-op names

Nothing is reimplemented; the existing representation pipeline remains the
source of truth.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from analysis.procedures.ast_edit_sequences import patch_to_ast_sequence
from analysis.procedures.scoped_edit_ops import (
    _NORMALIZE_OPS,
    trace_to_scoped_cert,
)

from .distances import edit_distance, module_distance, scope_distance, token_distance


@dataclass(frozen=True)
class TrajectoryView:
    """Normalized view of a trajectory: the four levels, ready for diffing."""

    tokens: list[str]
    edits: frozenset[str]
    scopes: frozenset[str]
    modules: frozenset[str]
    instance_id: str | None = None

    @property
    def has_tokens(self) -> bool:
        return len(self.tokens) > 0

    @property
    def has_edits(self) -> bool:
        return len(self.edits) > 0

    @property
    def has_scopes(self) -> bool:
        return len(self.scopes) > 0

    @property
    def has_modules(self) -> bool:
        return len(self.modules) > 0


@dataclass(frozen=True)
class Diff:
    """Structured procedural diff between two trajectories.

    Each field is a bounded [0, 1] distance, or None when that level is
    unavailable for either input. `components` retains the raw sets/sequences
    used so callers can inspect what drove the distance.
    """

    tokens: float | None
    edits: float | None
    scopes: float | None
    modules: float | None
    components: dict[str, Any] = field(default_factory=dict)

    @property
    def available_levels(self) -> list[str]:
        """Levels where both inputs had content."""
        return [
            name
            for name in ("tokens", "edits", "scopes", "modules")
            if getattr(self, name) is not None
        ]

    def mean(self, weights: dict[str, float] | None = None) -> float | None:
        """Mean of available levels, optionally weighted. None if no levels available."""
        levels = self.available_levels
        if not levels:
            return None
        if weights is None:
            return sum(getattr(self, lv) for lv in levels) / len(levels)
        total_w = sum(weights.get(lv, 0.0) for lv in levels)
        if total_w == 0:
            return None
        return sum(getattr(self, lv) * weights.get(lv, 0.0) for lv in levels) / total_w


@dataclass(frozen=True)
class Signature:
    """Aggregated diff statistics across a population of trajectories.

    Fields:
      n:              number of trajectories in the aggregate
      edit_vocab:     set of all edit-ops observed
      edit_freq:      Counter mapping edit-op to frequency across population
      module_vocab:   set of all file stems observed
      scope_vocab:    set of all scopes observed
      mean_lengths:   mean token count per trajectory
    """

    n: int
    edit_vocab: frozenset[str]
    edit_freq: Counter
    module_vocab: frozenset[str]
    scope_vocab: frozenset[str]
    mean_tokens: float


def _normalize_ops(raw: Iterable[str]) -> frozenset[str]:
    return frozenset(_NORMALIZE_OPS.get(op, op) for op in raw)


def _fallback_tokens_from_cert(cert: dict) -> list[str]:
    """If no patch-level tokens are available, synthesize a stable token list
    from the edit certificate so token_distance still has something to work on.
    """
    return list(cert.get("edit_cert", []) or [])


def _safe_scoped_cert(trace: dict) -> dict | None:
    try:
        return trace_to_scoped_cert(trace)
    except (KeyError, ValueError, TypeError):
        return None


def view_from_trace(trace: dict) -> TrajectoryView:
    """Build a `TrajectoryView` from a resolved-trace record.

    Expected trace fields: `instance_id`, `events` (list of dicts with
    `type == "code_change"` entries carrying `details.before_content` and
    `details.after_content`). See `trace_to_scoped_cert` for details.
    """
    cert = _safe_scoped_cert(trace)
    if cert is None:
        return TrajectoryView(
            tokens=[],
            edits=frozenset(),
            scopes=frozenset(),
            modules=frozenset(),
            instance_id=trace.get("instance_id") if isinstance(trace, dict) else None,
        )

    tokens = _tokens_from_trace(trace)
    if not tokens:
        tokens = _fallback_tokens_from_cert(cert)

    return TrajectoryView(
        tokens=tokens,
        edits=_normalize_ops(cert.get("edit_cert", []) or []),
        scopes=frozenset(cert.get("scopes_touched", []) or []),
        modules=frozenset([cert["file_module"]] if cert.get("file_module") else []),
        instance_id=cert.get("instance_id"),
    )


def view_from_patch(
    patch: str,
    *,
    file_paths: Iterable[str] | None = None,
    scopes: Iterable[str] | None = None,
    instance_id: str | None = None,
) -> TrajectoryView:
    """Build a view directly from a patch string.

    Useful for agent patches where we don't have a full trace but do have
    the raw diff. Callers can pass file_paths / scopes if they extracted
    them separately.
    """
    if not patch:
        return TrajectoryView(
            tokens=[],
            edits=frozenset(),
            scopes=frozenset(scopes or []),
            modules=frozenset(file_paths or []),
            instance_id=instance_id,
        )

    raw = patch_to_ast_sequence(patch)
    return TrajectoryView(
        tokens=list(raw),
        edits=_normalize_ops(raw),
        scopes=frozenset(scopes or []),
        modules=frozenset(file_paths or []),
        instance_id=instance_id,
    )


def _tokens_from_trace(trace: dict) -> list[str]:
    """Reconstruct a patch string from the trace and extract AST tokens."""
    import difflib

    if not isinstance(trace, dict):
        return []
    events = trace.get("events") or []
    diff_parts = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") != "code_change":
            continue
        d = ev.get("details") or {}
        fp = d.get("file_path", "")
        if not fp.endswith(".py"):
            continue
        before = d.get("before_content") or ""
        after = d.get("after_content") or ""
        if before == after:
            continue
        raw = "".join(difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fp,
            tofile=fp,
        ))
        if raw:
            diff_parts.append(f"diff --git a/{fp} b/{fp}\n" + raw)
    if not diff_parts:
        return []
    return patch_to_ast_sequence("\n".join(diff_parts))


def diff(a: TrajectoryView | dict, b: TrajectoryView | dict) -> Diff:
    """Compute procedural diff between two trajectories.

    Accepts `TrajectoryView` directly, or a raw trace dict (auto-converted).
    Levels where either side has no content return None for that field.
    """
    va = a if isinstance(a, TrajectoryView) else view_from_trace(a)
    vb = b if isinstance(b, TrajectoryView) else view_from_trace(b)

    return Diff(
        tokens=token_distance(va.tokens, vb.tokens) if (va.has_tokens or vb.has_tokens) else None,
        edits=edit_distance(va.edits, vb.edits) if (va.has_edits or vb.has_edits) else None,
        scopes=scope_distance(va.scopes, vb.scopes) if (va.has_scopes or vb.has_scopes) else None,
        modules=module_distance(va.modules, vb.modules) if (va.has_modules or vb.has_modules) else None,
        components={
            "a_tokens_len": len(va.tokens),
            "b_tokens_len": len(vb.tokens),
            "a_edits": set(va.edits),
            "b_edits": set(vb.edits),
            "a_scopes": set(va.scopes),
            "b_scopes": set(vb.scopes),
            "a_modules": set(va.modules),
            "b_modules": set(vb.modules),
        },
    )


def signature(trajectories: Iterable[TrajectoryView | dict]) -> Signature:
    """Aggregate diffs across a population.

    Returns a `Signature` capturing the edit-op vocabulary, module footprint,
    scope footprint, edit-op frequencies, and mean token length.
    """
    edit_freq: Counter = Counter()
    edit_vocab: set[str] = set()
    module_vocab: set[str] = set()
    scope_vocab: set[str] = set()
    total_tokens = 0
    n = 0

    for t in trajectories:
        v = t if isinstance(t, TrajectoryView) else view_from_trace(t)
        n += 1
        edit_freq.update(v.edits)
        edit_vocab.update(v.edits)
        module_vocab.update(v.modules)
        scope_vocab.update(v.scopes)
        total_tokens += len(v.tokens)

    mean_tokens = total_tokens / n if n else 0.0
    return Signature(
        n=n,
        edit_vocab=frozenset(edit_vocab),
        edit_freq=edit_freq,
        module_vocab=frozenset(module_vocab),
        scope_vocab=frozenset(scope_vocab),
        mean_tokens=mean_tokens,
    )


def build_reference_vocabulary(
    trajectories: Iterable[TrajectoryView | dict],
    *,
    min_count: int = 1,
) -> dict[str, frozenset[str]]:
    """Build a reference vocabulary from a corpus of trajectories.

    Returns a dict with keys `edits`, `modules`, `scopes` mapping to the set
    of items observed at least `min_count` times across the corpus. Used as
    input to `ood_score`.
    """
    edit_freq: Counter = Counter()
    module_freq: Counter = Counter()
    scope_freq: Counter = Counter()

    for t in trajectories:
        v = t if isinstance(t, TrajectoryView) else view_from_trace(t)
        edit_freq.update(v.edits)
        module_freq.update(v.modules)
        scope_freq.update(v.scopes)

    def _keep(counter: Counter) -> frozenset[str]:
        return frozenset(item for item, c in counter.items() if c >= min_count)

    return {
        "edits": _keep(edit_freq),
        "modules": _keep(module_freq),
        "scopes": _keep(scope_freq),
    }


def ood_score(
    trajectory: TrajectoryView | dict,
    reference: dict[str, frozenset[str]],
    *,
    level: str = "edits",
) -> float:
    """Procedural OOD score for a trajectory against a reference vocabulary.

    Defined as the fraction of the trajectory's items at `level` that do NOT
    appear in the reference:

        ood = |items(t) \\ reference[level]| / |items(t)|

    Zero means every item appears in reference (fully in-distribution); one
    means every item is novel relative to the reference. Returns 0.0 for an
    empty trajectory (nothing to be out-of-distribution with).

    We chose vocabulary-coverage OOD over nearest-neighbor distance because
    it's O(|items|) rather than O(|reference|), composes cleanly with the
    saturation / compositionality claims, and is interpretable per-item
    (which ops specifically are novel?). Nearest-neighbor OOD is a natural
    extension for later work.
    """
    if level not in reference:
        raise KeyError(f"reference has no '{level}' vocabulary; keys: {sorted(reference)}")

    v = trajectory if isinstance(trajectory, TrajectoryView) else view_from_trace(trajectory)
    items = getattr(v, level, None)
    if items is None:
        raise AttributeError(f"TrajectoryView has no attribute '{level}'")

    if not items:
        return 0.0

    ref = reference[level]
    novel = [x for x in items if x not in ref]
    return len(novel) / len(items)


def ood_items(
    trajectory: TrajectoryView | dict,
    reference: dict[str, frozenset[str]],
    *,
    level: str = "edits",
) -> list[str]:
    """Return the specific items novel relative to reference. For interpretability."""
    v = trajectory if isinstance(trajectory, TrajectoryView) else view_from_trace(trajectory)
    items = getattr(v, level, frozenset())
    ref = reference.get(level, frozenset())
    return sorted(x for x in items if x not in ref)
