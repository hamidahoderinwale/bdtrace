"""
Tree edit distance between edits certificates.

Input: two edits certificates (typed operation, node locations, implicated variables).
Output: scalar distance + which operations diverged.
"""

import ast
from typing import Any


def _ast_to_zss(node: ast.AST | None) -> Any:
    """Convert ast.AST to zss.Node."""
    try:
        from zss import Node
    except ImportError:
        raise ImportError("certificate_distance requires zss: pip install zss") from None

    if node is None:
        return Node("")
    n = Node(type(node).__name__)
    for child in ast.iter_child_nodes(node):
        n.addkid(_ast_to_zss(child))
    return n


def _parse_ast_from_dump(dump: str | None) -> ast.AST | None:
    """Reconstruct AST from ast.dump() string. Returns None if invalid."""
    if not dump or not isinstance(dump, str):
        return None
    try:
        ns = {n: getattr(ast, n) for n in dir(ast) if not n.startswith("_")}
        return eval(dump, {"__builtins__": {}}, ns)
    except (SyntaxError, NameError, TypeError, ValueError):
        return None


def certificate_distance(
    cert_a: dict[str, Any],
    cert_b: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """
    Tree edit distance between two edits certificates.
    Extracts AST from each certificate (ast_after, ast_before, or ast_after_dump).
    Returns (scalar_distance, divergence_info).
    """
    tree_a = cert_a.get("ast_after") if isinstance(cert_a, dict) else None
    tree_b = cert_b.get("ast_after") if isinstance(cert_b, dict) else None
    if tree_a is None and isinstance(cert_a, dict):
        tree_a = _parse_ast_from_dump(cert_a.get("ast_after_dump"))
    if tree_b is None and isinstance(cert_b, dict):
        tree_b = _parse_ast_from_dump(cert_b.get("ast_after_dump"))

    if tree_a is None and tree_b is None:
        return 0.0, {"operation_divergence": operation_divergence(cert_a, cert_b)}

    try:
        from zss import simple_distance

        na = _ast_to_zss(tree_a)
        nb = _ast_to_zss(tree_b)
        scalar = float(simple_distance(na, nb))
    except ImportError:
        scalar = float("inf")

    div = operation_divergence(cert_a, cert_b)
    return scalar, {"operation_divergence": div}


def operation_divergence(
    cert_a: dict[str, Any],
    cert_b: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare operation type sets between two certificates.
    Returns which operations appear in one but not the other.
    Useful for attribution: did procedures diverge because they saw different operations?
    """
    ops_a = cert_a.get("operations", []) if isinstance(cert_a, dict) else []
    ops_b = cert_b.get("operations", []) if isinstance(cert_b, dict) else []

    types_a = {op.get("type") for op in ops_a if isinstance(op, dict) and op.get("type")}
    types_b = {op.get("type") for op in ops_b if isinstance(op, dict) and op.get("type")}

    only_in_a = types_a - types_b
    only_in_b = types_b - types_a

    return {
        "only_in_a": list(only_in_a),
        "only_in_b": list(only_in_b),
        "in_both": list(types_a & types_b),
    }


def tree_edit_distance(tree_a: ast.AST | Any, tree_b: ast.AST | Any) -> float:
    """Tree edit distance between two AST trees (low-level)."""
    try:
        from zss import Node, simple_distance

        def is_zss(obj: Any) -> bool:
            return isinstance(obj, Node)

        na = _ast_to_zss(tree_a) if not is_zss(tree_a) else tree_a
        nb = _ast_to_zss(tree_b) if not is_zss(tree_b) else tree_b
        return float(simple_distance(na, nb))
    except ImportError:
        raise ImportError("tree_edit_distance requires zss: pip install zss") from None
