"""
Corpus-level procedural pattern mining for agent trajectories.

Addresses the core flaw in motif_mining.py: algorithms were run per-instance,
producing instance-specific hashes with no cross-instance signal.

This module:
1. Extracts file-typed action sequences (OPEN_SRC, EDIT_TEST, RUN_REPRO, etc.)
   so the alphabet is ~12 symbols and patterns recur meaningfully across instances.
2. Runs PrefixSpan across the full corpus (database of sequences), finding
   subsequences that appear in >= min_support instances.
3. Runs Sequitur across the full corpus by treating the concatenated sequence
   as one string with instance-boundary markers.
4. Represents each instance as a binary vector over the discovered pattern vocabulary
   — this is directly usable as a distance matrix input.

Usage:
    seqs = [typed_action_sequence(raw_traj) for raw_traj in trajectories]
    vocab = mine_corpus_patterns(seqs, min_support=0.1)
    vectors = encode_sequences(seqs, vocab)
    D = pairwise_jaccard(vectors)
"""

import re
from collections import Counter
from typing import Any

## File typing

_TEST_PATTERNS = re.compile(r'test_|_test\.|/tests?/|test\.py', re.IGNORECASE)
_REPRO_PATTERNS = re.compile(r'reproduc|repro\.|debug\.|/tmp/', re.IGNORECASE)
_CONFIG_PATTERNS = re.compile(r'settings|config|setup\.py|pyproject', re.IGNORECASE)


def _file_type(path: str) -> str:
    """Classify a file path into src / test / repro / config / other."""
    if not path:
        return "other"
    if _REPRO_PATTERNS.search(path):
        return "repro"
    if _TEST_PATTERNS.search(path):
        return "test"
    if _CONFIG_PATTERNS.search(path):
        return "config"
    if path.endswith(".py"):
        return "src"
    return "other"


def _classify_action(action: str) -> tuple[str, str]:
    """
    Return (action_type, file_type) for a raw action string.

    action_type: EDIT | OPEN | FIND | RUN | CREATE | NAV | SUBMIT | SHELL
    file_type:   src | test | repro | config | other | none
    """
    a = action.strip()
    lower = a.lower()
    m = re.search(r'([\w/._-]+\.py)', a)
    ftype = _file_type(m.group(1)) if m else "none"

    if lower.startswith("edit"):
        return "EDIT", ftype
    if lower.startswith("open"):
        return "OPEN", ftype
    if lower.startswith("find_file") or lower.startswith("find ") or lower.startswith("grep") or lower.startswith("search"):
        return "FIND", ftype
    if lower.startswith("create"):
        return "CREATE", ftype
    if lower.startswith("goto") or lower.startswith("scroll"):
        return "NAV", "none"
    if "submit" in lower:
        return "SUBMIT", "none"
    if any(lower.startswith(p) for p in ["python", "pytest", "bash", "./bin", "django", "pip", "pylint"]):
        return "RUN", ftype
    if any(lower.startswith(p) for p in ["cd", "ls", "pwd", "mkdir", "rm ", "touch", "echo", "cat", "exit_cost"]):
        return "SHELL", "none"
    return "OTHER", ftype


def typed_action_sequence(raw_trajectory: list[dict]) -> list[str]:
    """
    Convert raw trajectory steps to file-typed action token sequence.

    Tokens: OPEN_SRC, OPEN_TEST, OPEN_REPRO, EDIT_SRC, EDIT_TEST, RUN_repro,
            FIND, CREATE_REPRO, NAV, SUBMIT, SHELL, RUN, OTHER
    No run-length encoding — preserves repetition structure.
    """
    tokens = []
    for step in raw_trajectory:
        action = step.get("action", "").strip()
        if not action:
            continue
        atype, ftype = _classify_action(action)
        if ftype and ftype != "none":
            tokens.append(f"{atype}_{ftype.upper()}")
        else:
            tokens.append(atype)
    return tokens


def compress_sequence(seq: list[str]) -> list[str]:
    """Run-length encode: collapse consecutive identical tokens."""
    if not seq:
        return []
    result = [seq[0]]
    for tok in seq[1:]:
        if tok != result[-1]:
            result.append(tok)
    return result


## Corpus-level PrefixSpan

def _prefixspan_db(
    database: list[list[str]],
    min_support: int,
) -> list[tuple[list[str], int]]:
    """
    PrefixSpan over a database of sequences.

    Patterns grow until no extension meets min_support — no length cap.
    Returns list of (pattern, support_count) where support_count is the
    number of sequences in the database that contain the pattern as a subsequence.
    """
    results: list[tuple[list[str], int]] = []

    def _project(prefix: list[str], projected: list[list[str]]) -> None:
        # Count which symbols can extend the prefix in >= min_support sequences
        counts: Counter = Counter()
        for seq in projected:
            seen: set[str] = set()
            for tok in seq:
                if tok not in seen:
                    counts[tok] += 1
                    seen.add(tok)

        for sym, cnt in counts.items():
            if cnt < min_support:
                continue
            new_prefix = prefix + [sym]
            results.append((new_prefix, cnt))

            # Project: for each sequence, take the suffix after first occurrence of sym
            new_projected = []
            for seq in projected:
                try:
                    idx = seq.index(sym)
                    suffix = seq[idx + 1:]
                    if suffix:
                        new_projected.append(suffix)
                except ValueError:
                    pass

            if new_projected:
                _project(new_prefix, new_projected)

    _project([], database)
    # Deduplicate (PrefixSpan can return duplicates via different paths)
    seen = set()
    unique = []
    for pat, sup in results:
        key = tuple(pat)
        if key not in seen:
            seen.add(key)
            unique.append((pat, sup))
    return unique


def mine_corpus_patterns(
    sequences: list[list[str]],
    min_support: float = 0.1,
    compress: bool = True,
) -> list[tuple[list[str], int]]:
    """
    Mine frequent subsequence patterns across a corpus of action sequences.

    Patterns grow until no extension meets min_support — length emerges
    from the data rather than being capped.

    Args:
        sequences: List of token sequences (one per instance).
        min_support: Fraction of sequences that must contain the pattern (0–1).
        compress: If True, run-length encode sequences before mining.

    Returns:
        List of (pattern_tokens, support_count), sorted by support descending.
    """
    db = [compress_sequence(s) if compress else s for s in sequences]
    min_count = max(2, int(len(db) * min_support))
    patterns = _prefixspan_db(db, min_count)
    # Only return patterns of length >= 2 (single tokens aren't patterns)
    patterns = [(p, s) for p, s in patterns if len(p) >= 2]
    return sorted(patterns, key=lambda x: -x[1])


## Corpus-level Sequitur (grammar induction across concatenated corpus)

def _find_repeated_digrams(
    seq: list[str],
    boundary: str = "__BOUNDARY__",
) -> dict[tuple[str, str], int]:
    """Count digram frequencies, excluding those crossing boundaries."""
    counts: Counter = Counter()
    for i in range(len(seq) - 1):
        if seq[i] == boundary or seq[i + 1] == boundary:
            continue
        counts[(seq[i], seq[i + 1])] += 1
    return dict(counts)


def mine_corpus_sequitur(
    sequences: list[list[str]],
    min_support: float = 0.05,
    compress: bool = True,
    max_rules: int = 50,
) -> list[tuple[tuple[str, ...], int]]:
    """
    Simplified Sequitur across the full corpus.

    Concatenates all sequences with boundary markers, finds repeated digrams
    and trigrams that appear in >= min_support fraction of instances,
    and returns them as shared grammar rules.

    Real Sequitur would iteratively compress the sequence; this approximation
    finds the same recurring units without full grammar construction.

    Returns:
        List of (pattern_tuple, support_count) sorted by support descending.
    """
    db = [compress_sequence(s) if compress else s for s in sequences]
    min_count = max(2, int(len(db) * min_support))
    BOUNDARY = "__BOUNDARY__"

    # Per-instance bigram/trigram presence (not count — presence per instance)
    bigram_instances: dict[tuple[str, str], set[int]] = {}
    trigram_instances: dict[tuple[str, str, str], set[int]] = {}

    for inst_idx, seq in enumerate(db):
        for i in range(len(seq) - 1):
            bg = (seq[i], seq[i + 1])
            bigram_instances.setdefault(bg, set()).add(inst_idx)
        for i in range(len(seq) - 2):
            tg = (seq[i], seq[i + 1], seq[i + 2])
            trigram_instances.setdefault(tg, set()).add(inst_idx)

    rules: list[tuple[tuple[str, ...], int]] = []
    for bg, instances in bigram_instances.items():
        if len(instances) >= min_count:
            rules.append((bg, len(instances)))
    for tg, instances in trigram_instances.items():
        if len(instances) >= min_count:
            rules.append((tg, len(instances)))

    return sorted(rules, key=lambda x: -x[1])[:max_rules]


## Encoding: instance → binary vector over pattern vocabulary

def encode_sequences(
    sequences: list[list[str]],
    patterns: list[tuple[list[str] | tuple[str, ...], int]],
    compress: bool = True,
) -> list[list[int]]:
    """
    Encode each sequence as a binary vector over the pattern vocabulary.

    Entry[i][j] = 1 if sequence i contains pattern j as a subsequence.
    This is directly usable for Jaccard distance computation.
    """
    db = [compress_sequence(s) if compress else s for s in sequences]
    vocab = [p for p, _ in patterns]

    def _contains(seq: list[str], pattern: list[str] | tuple[str, ...]) -> bool:
        pi = 0
        for tok in seq:
            if tok == pattern[pi]:
                pi += 1
                if pi == len(pattern):
                    return True
        return False

    vectors = []
    for seq in db:
        vec = [1 if _contains(seq, pat) else 0 for pat in vocab]
        vectors.append(vec)
    return vectors


def tfidf_vectors(
    sequences: list[list[str]],
    patterns: list[tuple[list[str] | tuple[str, ...], int]],
    compress: bool = True,
) -> list[list[float]]:
    """
    Encode each sequence as a TF-IDF weighted pattern vector.

    TF: binary presence (1 if the pattern appears as a subsequence, else 0).
    IDF: log(N / df) where df is the pattern's support count from PrefixSpan.

    This down-weights patterns that fire in most instances (e.g. ADD_Name → ADD_Name)
    and up-weights rare, discriminative patterns — without requiring a hand-picked
    node filter. Pair with cosine_distance for the distance matrix.
    """
    import math
    N = len(sequences)
    binary = encode_sequences(sequences, patterns, compress=compress)
    idf = [math.log(N / max(sup, 1)) for _, sup in patterns]
    return [
        [b * w for b, w in zip(vec, idf)]
        for vec in binary
    ]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    """Cosine distance (1 - cosine similarity) between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    return 1.0 - dot / (n1 * n2)


def jaccard_distance(v1: list[int], v2: list[int]) -> float:
    """Jaccard distance between two binary pattern vectors."""
    inter = sum(a & b for a, b in zip(v1, v2))
    union = sum(a | b for a, b in zip(v1, v2))
    return 0.0 if union == 0 else 1.0 - inter / union


def pairwise_jaccard(vectors: list[list[int]]) -> list[list[float]]:
    """Compute n×n pairwise Jaccard distance matrix."""
    n = len(vectors)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(vectors[i], vectors[j])
            D[i][j] = D[j][i] = d
    return D
