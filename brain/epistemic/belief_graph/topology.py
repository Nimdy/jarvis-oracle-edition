"""Graph topology queries for the Belief Confidence Graph (Layer 7).

All queries are read-only and operate on the EdgeStore snapshot.
No mutations to edges or beliefs.
"""

from __future__ import annotations

from collections import deque
from typing import Any


def get_support_strength(belief_id: str, edge_store: Any) -> float:
    """Sum of incoming 'supports' edge strengths for a belief."""
    incoming = edge_store.get_incoming(belief_id)
    return sum(e.strength for e in incoming if e.edge_type == "supports")


def get_contradiction_pressure(belief_id: str, edge_store: Any) -> float:
    """Sum of incoming + outgoing 'contradicts' edge strengths."""
    incoming = edge_store.get_incoming(belief_id)
    outgoing = edge_store.get_outgoing(belief_id)
    return sum(
        e.strength for e in incoming if e.edge_type == "contradicts"
    ) + sum(
        e.strength for e in outgoing if e.edge_type == "contradicts"
    )


def get_dependents(belief_id: str, edge_store: Any) -> list[str]:
    """Return belief IDs that depend_on this belief (incoming depends_on edges)."""
    incoming = edge_store.get_incoming(belief_id)
    return [e.source_belief_id for e in incoming if e.edge_type == "depends_on"]


def get_prerequisites(belief_id: str, edge_store: Any) -> list[str]:
    """Return belief IDs that this belief depends_on (outgoing depends_on edges)."""
    outgoing = edge_store.get_outgoing(belief_id)
    return [e.target_belief_id for e in outgoing if e.edge_type == "depends_on"]


def get_roots(edge_store: Any) -> list[str]:
    """Return belief IDs with no incoming support or depends_on edges (root beliefs)."""
    all_targets: set[str] = set()
    all_sources: set[str] = set()

    for eid, edge in edge_store._edges.items():
        if edge.edge_type in ("supports", "depends_on"):
            all_targets.add(edge.target_belief_id)
            all_sources.add(edge.source_belief_id)

    all_beliefs = all_targets | all_sources
    return [b for b in all_beliefs if b not in all_targets]


def get_leaves(edge_store: Any) -> list[str]:
    """Return belief IDs with no outgoing support or depends_on edges (leaf beliefs)."""
    all_targets: set[str] = set()
    all_sources: set[str] = set()

    for edge in edge_store._edges.values():
        if edge.edge_type in ("supports", "depends_on"):
            all_targets.add(edge.target_belief_id)
            all_sources.add(edge.source_belief_id)

    all_beliefs = all_targets | all_sources
    return [b for b in all_beliefs if b not in all_sources]


def find_path(source_id: str, target_id: str, edge_store: Any, max_depth: int = 10) -> list[str] | None:
    """BFS shortest path from source to target through any edge type.  Returns belief ID list or None."""
    if source_id == target_id:
        return [source_id]

    visited: set[str] = {source_id}
    queue: deque[list[str]] = deque([[source_id]])

    while queue:
        path = queue.popleft()
        if len(path) > max_depth:
            continue
        current = path[-1]

        for edge in edge_store.get_outgoing(current):
            next_id = edge.target_belief_id
            if next_id == target_id:
                return path + [next_id]
            if next_id not in visited:
                visited.add(next_id)
                queue.append(path + [next_id])

    return None


def get_connected_components(edge_store: Any) -> list[set[str]]:
    """Return connected components treating all edges as undirected."""
    all_beliefs: set[str] = set()
    adjacency: dict[str, set[str]] = {}

    for edge in edge_store._edges.values():
        s, t = edge.source_belief_id, edge.target_belief_id
        all_beliefs.add(s)
        all_beliefs.add(t)
        adjacency.setdefault(s, set()).add(t)
        adjacency.setdefault(t, set()).add(s)

    visited: set[str] = set()
    components: list[set[str]] = []

    for belief in all_beliefs:
        if belief in visited:
            continue
        component: set[str] = set()
        stack = [belief]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)

    return components


def get_belief_centrality(edge_store: Any) -> dict[str, float]:
    """Degree centrality: (in_degree + out_degree) / max_possible.

    Returns a dict of belief_id -> centrality [0, 1].
    """
    degree: dict[str, int] = {}

    for edge in edge_store._edges.values():
        degree[edge.source_belief_id] = degree.get(edge.source_belief_id, 0) + 1
        degree[edge.target_belief_id] = degree.get(edge.target_belief_id, 0) + 1

    if not degree:
        return {}

    max_degree = max(degree.values())
    if max_degree == 0:
        return {k: 0.0 for k in degree}

    return {k: round(v / max_degree, 4) for k, v in degree.items()}


def get_top_beliefs_by_centrality(edge_store: Any, n: int = 10) -> list[tuple[str, float]]:
    """Return top N beliefs by degree centrality."""
    centrality = get_belief_centrality(edge_store)
    sorted_beliefs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_beliefs[:n]
