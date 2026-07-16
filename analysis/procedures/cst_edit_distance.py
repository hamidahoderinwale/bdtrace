"""
Tree-sitter CST-based edit distance.

For each instance, parse before_content and after_content using tree-sitter.
Compute the net change in node-type distribution:
  - ADD_<node_type>: node types that increased in count from before → after
  - DEL_<node_type>: node types that decreased in count from before → after

This gives a language-model-agnostic vocabulary of ~60 canonical node types
vs the 184 hand-crafted op types in the edits certificates.

Usage:
  from analysis.procedures.cst_edit_distance import build_cst_distance_matrix
  D, vocab = build_cst_distance_matrix(records)  # records have before_content/after_content
"""

from collections import Counter
from typing import Optional

import numpy as np

# Punctuation and delimiter nodes to skip — pure syntax, no semantic content
_SKIP_TYPES = frozenset({
    "(", ")", "[", "]", "{", "}", ",", ".", ":", ";", "->",
    '"', "'", '"""', "'''", "\\n", "\\",
    "indent", "dedent", "newline",
    "comment", "string_start", "string_end", "string_content",
    "escape_sequence",
})


def _parse_node_counts(source: Optional[str]) -> Counter:
    """Parse Python source with tree-sitter and return a Counter of node types."""
    if not source:
        return Counter()
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        return Counter()

    PY = Language(tspython.language())
    parser = Parser(PY)

    src_bytes = source.encode("utf-8", errors="replace")
    tree = parser.parse(src_bytes)

    counts: Counter = Counter()
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type not in _SKIP_TYPES:
            counts[node.type] += 1
        stack.extend(node.children)
    return counts


def patch_to_cst_tokens(before: Optional[str], after: Optional[str]) -> list[str]:
    """
    Return a list of ADD_<type> / DEL_<type> tokens representing the net
    change in CST node-type distribution from before → after.

    Tokens are repeated by their net count so TF-IDF captures magnitude.
    """
    before_counts = _parse_node_counts(before)
    after_counts = _parse_node_counts(after)

    tokens: list[str] = []
    all_types = set(before_counts) | set(after_counts)
    for t in all_types:
        delta = after_counts[t] - before_counts[t]
        if delta > 0:
            tokens.extend([f"ADD_{t}"] * delta)
        elif delta < 0:
            tokens.extend([f"DEL_{t}"] * (-delta))
    return tokens


def build_cst_distance_matrix(
    records: list[dict],
    before_key: str = "before_content",
    after_key: str = "after_content",
) -> tuple[np.ndarray, list[str]]:
    """
    Build an (n, n) TF-IDF cosine distance matrix from CST node-type diffs.

    Records must have before_content and after_content fields.
    Returns (D, vocab) where D is the distance matrix and vocab is the token list.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances

    docs = []
    for rec in records:
        tokens = patch_to_cst_tokens(rec.get(before_key), rec.get(after_key))
        docs.append(" ".join(tokens) if tokens else "")

    n_nonempty = sum(1 for d in docs if d)
    print(f"  CST docs: {n_nonempty}/{len(docs)} non-empty")

    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]+", min_df=2)
    X = vec.fit_transform(docs)
    print(f"  CST vocab size: {len(vec.vocabulary_)} tokens")

    D = cosine_distances(X)
    np.fill_diagonal(D, 0.0)

    return D, list(vec.vocabulary_.keys())
