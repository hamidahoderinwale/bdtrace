"""
Bounded distances for procedural diff: tokens, edit-types, modules.

All distances return a float in [0, 1]. Zero means identical; one means disjoint.
Tokens use normalized Levenshtein (sensitive to order and length); edit-types use
a Dice-style normalization over the symmetric difference (bounded, not a true
metric — triangle inequality fails); modules use Jaccard.
"""

from collections.abc import Iterable


def _as_list(seq: Iterable[str] | None) -> list[str]:
    if seq is None:
        return []
    return list(seq)


def _as_set(items: Iterable[str] | None) -> set[str]:
    if items is None:
        return set()
    return set(items)


def token_distance(a: Iterable[str] | None, b: Iterable[str] | None) -> float:
    """Normalized Levenshtein distance over token sequences.

    d = lev(a, b) / max(|a|, |b|, 1)

    Sensitive to ordering and length. Bounded in [0, 1]. Two empty sequences
    are treated as identical (distance 0).
    """
    sa = _as_list(a)
    sb = _as_list(b)
    if not sa and not sb:
        return 0.0

    n, m = len(sa), len(sb)
    if n == 0:
        return 1.0
    if m == 0:
        return 1.0

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if sa[i - 1] == sb[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    denom = max(n, m, 1)
    return prev[m] / denom


def edit_distance(a: Iterable[str] | None, b: Iterable[str] | None) -> float:
    """Dice-style distance over edit-type sets.

    d = |A △ B| / (|A| + |B|)

    Bounded in [0, 1]. Not a true metric — triangle inequality fails — but
    penalises disagreement more heavily than Jaccard when both sets are large.
    Empty-vs-empty returns 0.0.
    """
    sa = _as_set(a)
    sb = _as_set(b)
    if not sa and not sb:
        return 0.0
    total = len(sa) + len(sb)
    if total == 0:
        return 0.0
    sym = len(sa ^ sb)
    return sym / total


def module_distance(a: Iterable[str] | None, b: Iterable[str] | None) -> float:
    """Jaccard distance over module / file-stem sets.

    d = 1 - |A ∩ B| / |A ∪ B|

    Bounded in [0, 1]. Proper metric. Empty-vs-empty returns 0.0.
    """
    sa = _as_set(a)
    sb = _as_set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - len(sa & sb) / len(union)


def scope_distance(a: Iterable[str] | None, b: Iterable[str] | None) -> float:
    """Jaccard distance over touched scopes (FunctionDef:name, ClassDef:name).

    Same form as `module_distance` but applied to scope identifiers.
    """
    return module_distance(a, b)
