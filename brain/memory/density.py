"""Intelligent memory density — 4-axis scoring for memory health and richness."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from consciousness.events import Memory


@dataclass(frozen=True)
class MemoryDensity:
    """Composite density score with per-axis breakdowns."""
    associative_richness: float  # 0-1: connectivity quality
    temporal_coherence: float    # 0-1: sequential relationship quality
    semantic_clustering: float   # 0-1: topical grouping quality
    distribution_score: float    # 0-1: recency balance quality
    overall: float               # weighted composite
    memory_count: int
    count_factor: float          # min(1, count/20) scaling


def calculate_density(memories: list[Memory]) -> MemoryDensity:
    """Calculate 4-axis density for a set of memories."""
    if not memories:
        return MemoryDensity(0, 0, 0, 0, 0, 0, 0)

    count = len(memories)
    count_factor = min(1.0, count / 50.0)

    assoc = _associative_richness(memories)
    temporal = _temporal_coherence(memories)
    semantic = _semantic_clustering(memories)
    distrib = _distribution_score(memories)

    overall = (assoc * 0.35 + semantic * 0.25 + temporal * 0.2 + distrib * 0.2) * count_factor

    return MemoryDensity(
        associative_richness=round(assoc, 4),
        temporal_coherence=round(temporal, 4),
        semantic_clustering=round(semantic, 4),
        distribution_score=round(distrib, 4),
        overall=round(overall, 4),
        memory_count=count,
        count_factor=round(count_factor, 4),
    )


def _associative_richness(memories: list[Memory]) -> float:
    """Score sparse association health instead of clique density.

    Jarvis is expected to form a useful sparse graph, not a near-complete graph.
    The old metric divided by n*(n-1)/2 which punished healthy systems as memory
    count grew. Here we score:
    - average links per memory against a realistic sparse target
    - coverage: how many memories participate in at least one link
    - weight support on linked memories
    """
    n = len(memories)
    if n < 2:
        return 0.0

    weight_lookup = {m.id: m.weight for m in memories}
    total_links = sum(len(m.associations) for m in memories)
    avg_links = total_links / n if n > 0 else 0.0
    target_sparse_links = 8.0
    link_score = min(1.0, avg_links / target_sparse_links)

    connected = sum(1 for m in memories if m.associations)
    coverage_score = connected / n if n > 0 else 0.0

    weighted_sum = 0.0
    total_seen = 0.0
    for m in memories:
        for assoc_id in m.associations:
            w = weight_lookup.get(assoc_id)
            if w is None:
                continue
            weighted_sum += (m.weight + w) / 2.0
            total_seen += 1.0

    weighted_ratio = min(1.0, weighted_sum / total_seen) if total_seen > 0 else 0.0

    return link_score * 0.45 + coverage_score * 0.35 + weighted_ratio * 0.2


def _temporal_coherence(memories: list[Memory]) -> float:
    """Score whether memories have temporally nearby supporting neighbors.

    The old metric only checked consecutive pairs across the entire history,
    which makes mature long-running systems look incoherent. Instead, treat a
    memory as temporally coherent when it has at least one nearby neighbor that
    shares tags or an explicit association.
    """
    if len(memories) < 2:
        return 0.0

    sorted_mems = sorted(memories, key=lambda m: m.timestamp)
    coherent = 0
    total = len(sorted_mems)
    time_windows_s = (600, 3600, 21600)

    for i, current in enumerate(sorted_mems):
        current_tags = set(current.tags)
        matched = False
        for j in range(max(0, i - 4), min(total, i + 5)):
            if i == j:
                continue
            other = sorted_mems[j]
            diff = abs(other.timestamp - current.timestamp)
            if diff > time_windows_s[-1]:
                continue
            shared_tags = current_tags & set(other.tags)
            mutual_assoc = current.id in other.associations or other.id in current.associations
            if not shared_tags and not mutual_assoc:
                continue
            if diff <= time_windows_s[0]:
                matched = True
                break
            if diff <= time_windows_s[1] and (len(shared_tags) >= 1 or mutual_assoc):
                matched = True
                break
            if diff <= time_windows_s[2] and mutual_assoc:
                matched = True
                break
        if matched:
            coherent += 1

    return coherent / total if total > 0 else 0.0


def _semantic_clustering(memories: list[Memory]) -> float:
    """Score whether memories form recurring topical groups.

    This is coverage-oriented instead of edge-density-oriented so it reflects
    semantic organization even when the association graph is intentionally sparse.
    """
    if not memories:
        return 0.0

    clusters: dict[str, list[Memory]] = {}
    for m in memories:
        meaningful_tags = [
            t for t in m.tags
            if not t.startswith("speaker:")
            and not t.startswith("schema:")
            and not t.startswith("fact_kind:")
            and not t.startswith("preference_kind:")
            and not t.startswith("interest_kind:")
        ]
        tag = meaningful_tags[0] if meaningful_tags else (m.tags[0] if m.tags else "untagged")
        clusters.setdefault(tag, []).append(m)

    if len(clusters) <= 1:
        return 0.7

    repeated_members = 0
    weighted_cluster_sizes = 0.0
    repeated_cluster_count = 0

    for group in clusters.values():
        group_size = len(group)
        if group_size < 2:
            continue
        repeated_members += group_size
        mean_weight = sum(m.weight for m in group) / group_size
        weighted_cluster_sizes += min(1.0, group_size / 6.0) * mean_weight
        repeated_cluster_count += 1

    repeated_coverage = repeated_members / len(memories)
    cluster_size_score = (
        weighted_cluster_sizes / repeated_cluster_count if repeated_cluster_count > 0 else 0.0
    )
    return repeated_coverage * 0.7 + cluster_size_score * 0.3


def _distribution_score(memories: list[Memory]) -> float:
    """Score healthy time-horizon balance for a mature persistent system."""
    now = time.time()
    recent = 0      # < 15 min
    medium = 0      # 15 min - 24h
    long_term = 0   # > 24h

    for m in memories:
        age = now - m.timestamp
        if age < 900:
            recent += 1
        elif age < 86400:
            medium += 1
        else:
            long_term += 1

    total = len(memories)
    if total == 0:
        return 0.0

    actual = (
        recent / total,
        medium / total,
        long_term / total,
    )
    target = (0.15, 0.25, 0.60)
    diff = sum(abs(a - t) for a, t in zip(actual, target))
    return max(0.0, 1.0 - diff / 2.0)
