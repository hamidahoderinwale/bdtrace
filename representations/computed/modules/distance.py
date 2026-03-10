"""
Graph edit distance between two module graphs.

Input: two networkx subgraphs (import edges + co-edit weights).
Output: scalar distance + which edges diverged.
Full GED is NP-hard; symmetric diff is tractable and sufficient.
"""

from typing import Any

import networkx as nx


def graph_distance(
    graph_a: nx.Graph | nx.DiGraph,
    graph_b: nx.Graph | nx.DiGraph,
) -> tuple[float, dict[str, Any]]:
    """
    Symmetric edge difference as proxy for graph edit distance.
    Returns (normalized_distance, divergence_info).
    """
    div = edge_divergence(graph_a, graph_b)
    edges_a = set(graph_a.edges())
    edges_b = set(graph_b.edges())
    union_diff = len(div["only_in_a"]) + len(div["only_in_b"])
    union_all = len(edges_a | edges_b)
    if union_all == 0:
        scalar = 0.0
    else:
        scalar = union_diff / union_all
    return scalar, div


def edge_divergence(
    graph_a: nx.Graph | nx.DiGraph,
    graph_b: nx.Graph | nx.DiGraph,
) -> dict[str, Any]:
    """
    Return the actual divergent edges, not just the scalar.
    Which import relationships appear in one subgraph but not the other.
    """
    edges_a = set(graph_a.edges())
    edges_b = set(graph_b.edges())
    only_in_a = edges_a - edges_b
    only_in_b = edges_b - edges_a
    return {
        "only_in_a": list(only_in_a),
        "only_in_b": list(only_in_b),
        "in_both": list(edges_a & edges_b),
    }


def graph_edit_distance(
    G1: nx.Graph | nx.DiGraph,
    G2: nx.Graph | nx.DiGraph,
    node_match: Any = None,
    edge_match: Any = None,
    timeout: float | None = None,
) -> float:
    """Full graph edit distance (NP-hard). Use graph_distance for tractable alternative."""
    try:
        return nx.graph_edit_distance(
            G1,
            G2,
            node_match=node_match,
            edge_match=edge_match,
            timeout=timeout,
        )
    except (nx.NetworkXError, nx.NetworkXNoPath):
        return float("inf")
