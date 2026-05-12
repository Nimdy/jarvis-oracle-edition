"""Graph integrity metrics for the Belief Confidence Graph (Layer 7).

Monitors graph health: orphan rate, fragmentation, cycles, dangling
dependencies, and support from quarantined beliefs.  Feeds into Layer 6
epistemic domain calibrator.
"""

from __future__ import annotations

from collections import deque
from typing import Any


def compute_integrity(edge_store: Any, belief_store: Any) -> dict[str, Any]:
    """Compute all integrity metrics in a single pass.

    Returns a dict suitable for dashboard display and Layer 6 integration.
    """
    active_belief_ids = {b.belief_id for b in belief_store.get_active_beliefs()}
    all_beliefs = {b.belief_id: b for b in belief_store.get_active_beliefs()}

    graph_belief_ids: set[str] = set()
    for edge in edge_store._edges.values():
        graph_belief_ids.add(edge.source_belief_id)
        graph_belief_ids.add(edge.target_belief_id)

    orphan_count = len(active_belief_ids - graph_belief_ids)
    orphan_rate = orphan_count / max(len(active_belief_ids), 1)

    components = _connected_components(edge_store)
    fragmentation = len(components) / max(len(graph_belief_ids), 1) if graph_belief_ids else 0.0

    cycle_count = _count_cycles(edge_store)

    support_contradict_ratio = _support_contradict_ratio(edge_store)

    dangling_dep_count, dangling_dep_rate = _dangling_dependency_rate(edge_store, active_belief_ids)

    quarantined_ids = {
        b.belief_id for b in belief_store._beliefs.values()
        if b.resolution_state == "quarantined"
    }
    support_from_quarantined, quarantined_rate = _support_from_quarantined_rate(
        edge_store, quarantined_ids,
    )

    total_edges = len(edge_store._edges)
    score = _compute_health_score(
        orphan_rate, fragmentation, cycle_count, dangling_dep_rate,
        quarantined_rate, total_edges,
    )

    return {
        "orphan_count": orphan_count,
        "orphan_rate": round(orphan_rate, 4),
        "active_beliefs": len(active_belief_ids),
        "graph_beliefs": len(graph_belief_ids),
        "component_count": len(components),
        "largest_component": max((len(c) for c in components), default=0),
        "fragmentation": round(fragmentation, 4),
        "cycle_count": cycle_count,
        "support_contradict_ratio": round(support_contradict_ratio, 4),
        "dangling_dependency_count": dangling_dep_count,
        "dangling_dependency_rate": round(dangling_dep_rate, 4),
        "support_from_quarantined_count": support_from_quarantined,
        "support_from_quarantined_rate": round(quarantined_rate, 4),
        "total_edges": total_edges,
        "health_score": round(score, 4),
    }


def _connected_components(edge_store: Any) -> list[set[str]]:
    """Undirected connected components of the graph."""
    adjacency: dict[str, set[str]] = {}
    for edge in edge_store._edges.values():
        s, t = edge.source_belief_id, edge.target_belief_id
        adjacency.setdefault(s, set()).add(t)
        adjacency.setdefault(t, set()).add(s)

    visited: set[str] = set()
    components: list[set[str]] = []
    for node in adjacency:
        if node in visited:
            continue
        component: set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            component.add(n)
            for neighbor in adjacency.get(n, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)
    return components


def _count_cycles(edge_store: Any, max_depth: int = 8) -> int:
    """Detect directed cycles up to max_depth using DFS.

    Returns count of unique cycles found (capped for performance).
    """
    adjacency: dict[str, list[str]] = {}
    for edge in edge_store._edges.values():
        if edge.edge_type in ("supports", "depends_on", "derived_from"):
            adjacency.setdefault(edge.source_belief_id, []).append(edge.target_belief_id)

    all_nodes = set(adjacency.keys())
    for targets in adjacency.values():
        all_nodes.update(targets)

    cycles_found = 0
    cycle_cap = 20

    for start in all_nodes:
        if cycles_found >= cycle_cap:
            break
        visited: set[str] = set()
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack and cycles_found < cycle_cap:
            node, path = stack.pop()
            for neighbor in adjacency.get(node, []):
                if neighbor == start and len(path) > 1:
                    cycles_found += 1
                    break
                if neighbor not in visited and len(path) < max_depth:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))

    return cycles_found


def _support_contradict_ratio(edge_store: Any) -> float:
    """Ratio of support edges to contradiction edges.  Higher is healthier."""
    support_count = len(edge_store._by_type.get("supports", set()))
    contradict_count = len(edge_store._by_type.get("contradicts", set()))
    if contradict_count == 0:
        return float(support_count) if support_count else 1.0
    return support_count / contradict_count


def _dangling_dependency_rate(
    edge_store: Any, active_belief_ids: set[str],
) -> tuple[int, float]:
    """Count depends_on edges whose target (prerequisite) is no longer active."""
    depends_on_edges = edge_store.get_by_type("depends_on")
    if not depends_on_edges:
        return 0, 0.0
    dangling = sum(
        1 for e in depends_on_edges
        if e.target_belief_id not in active_belief_ids
    )
    return dangling, dangling / len(depends_on_edges)


def _support_from_quarantined_rate(
    edge_store: Any, quarantined_ids: set[str],
) -> tuple[int, float]:
    """Count support edges whose source belief is quarantined."""
    support_edges = edge_store.get_by_type("supports")
    if not support_edges:
        return 0, 0.0
    bad = sum(
        1 for e in support_edges
        if e.source_belief_id in quarantined_ids
    )
    return bad, bad / len(support_edges)


def _compute_health_score(
    orphan_rate: float,
    fragmentation: float,
    cycle_count: int,
    dangling_dep_rate: float,
    quarantined_rate: float,
    total_edges: int,
) -> float:
    """Weighted composite health score [0, 1].  Higher is healthier."""
    if total_edges == 0:
        return 1.0

    score = 1.0
    score -= orphan_rate * 0.20
    score -= min(fragmentation, 1.0) * 0.15
    score -= min(cycle_count / 10.0, 1.0) * 0.15
    score -= dangling_dep_rate * 0.25
    score -= quarantined_rate * 0.25

    return max(0.0, min(1.0, score))
