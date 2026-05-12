"""Semantic memory clustering — groups memories by embedding similarity.

Clusters are persistent cognitive structures that organize memories into
meaningful groups. They mirror the hippocampal memory consolidation in
biological systems (SyntheticSoul §4.2): new memories are absorbed into
existing clusters during dream cycles, clusters evolve over time through
growth/split/merge, and the structural map persists across restarts.

Key invariant: dream cycles EVOLVE the cluster structure, never destroy it.
Full re-clustering only happens at initial boot with zero existing clusters.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from consciousness.events import Memory

logger = logging.getLogger(__name__)


@dataclass
class MemoryCluster:
    id: str
    cluster_type: str  # emotional, temporal, causal, conceptual, experiential
    memory_ids: list[str] = field(default_factory=list)
    coherence: float = 0.0
    topic: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tag_signature: set[str] = field(default_factory=set)  # union of member tags for similarity


@dataclass
class ClusterInsight:
    patterns: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    trends: list[str] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


_TYPE_KEYWORDS = {
    "emotional": {"emotion", "feeling", "mood", "happy", "sad", "angry", "joy", "empathy"},
    "temporal": {"morning", "evening", "yesterday", "today", "routine", "schedule", "time"},
    "causal": {"because", "therefore", "caused", "effect", "result", "consequence"},
    "conceptual": {"idea", "concept", "theory", "abstract", "philosophy", "thought"},
    "experiential": {"experience", "conversation", "interaction", "observation", "memory"},
}

SPLIT_THRESHOLD = 25
MERGE_SIMILARITY_THRESHOLD = 0.5
MIN_CLUSTER_SIZE = 2


def _classify_cluster_type(memories: list[Memory]) -> str:
    """Classify cluster type by keyword and tag heuristics."""
    tag_counts: dict[str, int] = {}
    text_lower = ""
    for m in memories:
        for tag in m.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        payload = getattr(m, "payload", "")
        if isinstance(payload, str):
            text_lower += payload.lower() + " "

    best_type = "experiential"
    best_score = 0
    for ctype, keywords in _TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 2
            if kw in tag_counts:
                score += 3
        if score > best_score:
            best_score = score
            best_type = ctype
    return best_type


def _compute_tag_similarity(a: Memory, b: Memory) -> float:
    """Jaccard similarity on tags."""
    tags_a = set(a.tags)
    tags_b = set(b.tags)
    if not tags_a and not tags_b:
        return 0.0
    intersection = tags_a & tags_b
    union = tags_a | tags_b
    return len(intersection) / len(union) if union else 0.0


class MemoryClusterEngine:
    """Persistent cognitive structure that organizes memories into clusters.

    Clusters evolve over time through dream-cycle consolidation:
    - New memories are absorbed into matching existing clusters
    - Unclustered memories form new clusters via agglomerative grouping
    - Overgrown clusters split, similar small clusters merge
    - Dead references (evicted memories) are pruned

    The cluster map persists across restarts via save/restore in persistence.py.
    """

    def __init__(self, similarity_threshold: float = 0.3, max_cluster_size: int = 20) -> None:
        self._clusters: list[MemoryCluster] = []
        self._memory_to_cluster: dict[str, str] = {}  # memory_id -> cluster_id
        self._similarity_threshold = similarity_threshold
        self._max_cluster_size = max_cluster_size

    # ------------------------------------------------------------------
    # Dream-cycle entry point: incremental consolidation
    # ------------------------------------------------------------------

    def consolidate_memories(self, memories: list[Memory],
                             all_memory_ids: set[str] | None = None) -> list[MemoryCluster]:
        """Evolve the cluster structure with new memories (dream-cycle entry).

        This is the primary method called during dream cycles. It:
        1. Prunes dead memory references from existing clusters
        2. Absorbs new (unclustered) memories into existing clusters
        3. Groups remaining unclustered memories into new clusters
        4. Splits overgrown clusters
        5. Merges similar small clusters

        Unlike cluster_memories(), this method PRESERVES the existing cluster
        structure and evolves it incrementally — matching the SyntheticSoul
        paper's hippocampal consolidation model.

        Args:
            memories: The batch of memories to consolidate (typically recent).
            all_memory_ids: If provided, prune cluster refs to only these IDs.

        Returns:
            The current list of all clusters (existing + new).
        """
        if len(memories) < 2 and not self._clusters:
            return []

        # Phase 1: prune dead references from existing clusters
        if all_memory_ids is not None:
            self._prune_dead_refs(all_memory_ids)

        # Phase 2: identify which input memories are already clustered
        new_memories = [m for m in memories if m.id not in self._memory_to_cluster]

        # Phase 3: absorb new memories into existing clusters
        still_unclustered: list[Memory] = []
        absorbed = 0
        for mem in new_memories:
            cluster_id = self.add_memory(mem)
            if cluster_id:
                absorbed += 1
            else:
                still_unclustered.append(mem)

        # Phase 4: group remaining unclustered memories into new clusters
        new_clusters_formed = 0
        if len(still_unclustered) >= 2:
            new_clusters = self._agglomerative_cluster(still_unclustered)
            for cluster in new_clusters:
                self._clusters.append(cluster)
                for mid in cluster.memory_ids:
                    self._memory_to_cluster[mid] = cluster.id
                new_clusters_formed += 1

        # Phase 5: split overgrown clusters
        splits = self._split_overgrown(memories)

        # Phase 6: merge similar small clusters
        merges = self._merge_similar()

        if absorbed or new_clusters_formed or splits or merges:
            logger.info(
                "Consolidation: absorbed %d, new clusters %d, splits %d, merges %d (total: %d clusters, %d assigned)",
                absorbed, new_clusters_formed, splits, merges,
                len(self._clusters), len(self._memory_to_cluster),
            )

        return self._clusters

    # ------------------------------------------------------------------
    # Full rebuild (initial boot only, when no clusters exist)
    # ------------------------------------------------------------------

    def cluster_memories(self, memories: list[Memory]) -> list[MemoryCluster]:
        """Full re-clustering using agglomerative approach on tag similarity.

        Only used when the cluster structure is empty (first boot, or after
        reset). During normal operation, use consolidate_memories() instead.
        """
        if len(memories) < 2:
            return []

        if self._clusters:
            return self.consolidate_memories(memories)

        new_clusters = self._agglomerative_cluster(memories)
        self._clusters = new_clusters
        self._memory_to_cluster.clear()
        for cluster in self._clusters:
            for mid in cluster.memory_ids:
                self._memory_to_cluster[mid] = cluster.id

        logger.info("Initial clustering: %d memories into %d clusters",
                     len(memories), len(self._clusters))
        return self._clusters

    # ------------------------------------------------------------------
    # Single-memory incremental add
    # ------------------------------------------------------------------

    def add_memory(self, memory: Memory) -> str | None:
        """Incremental: add a single memory to the best existing cluster."""
        if not self._clusters:
            return None

        best_cluster = None
        best_sim = 0.0

        for cluster in self._clusters:
            if len(cluster.memory_ids) >= self._max_cluster_size:
                continue
            sim = self._memory_cluster_similarity(memory, cluster)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster and best_sim >= self._similarity_threshold:
            best_cluster.memory_ids.append(memory.id)
            best_cluster.tag_signature.update(memory.tags)
            best_cluster.updated_at = time.time()
            self._memory_to_cluster[memory.id] = best_cluster.id
            return best_cluster.id

        return None

    # ------------------------------------------------------------------
    # Serialization / persistence
    # ------------------------------------------------------------------

    def get_clusters(self) -> list[dict[str, Any]]:
        return [
            {"id": c.id, "type": c.cluster_type, "size": len(c.memory_ids),
             "coherence": round(c.coherence, 3), "topic": c.topic}
            for c in self._clusters
        ]

    def get_clusters_full(self) -> list[dict[str, Any]]:
        """Return full cluster data including memory_ids for persistence."""
        return [
            {
                "id": c.id,
                "cluster_type": c.cluster_type,
                "memory_ids": c.memory_ids,
                "coherence": c.coherence,
                "topic": c.topic,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "tag_signature": sorted(c.tag_signature) if c.tag_signature else [],
            }
            for c in self._clusters
        ]

    def restore_clusters(self, data: list[dict[str, Any]], valid_memory_ids: set[str] | None = None) -> int:
        """Restore clusters from persisted data, pruning stale memory refs."""
        self._clusters.clear()
        self._memory_to_cluster.clear()
        restored = 0
        for item in data:
            mids = item.get("memory_ids", [])
            if valid_memory_ids is not None:
                mids = [mid for mid in mids if mid in valid_memory_ids]
            if len(mids) < MIN_CLUSTER_SIZE:
                continue
            cluster = MemoryCluster(
                id=item.get("id", str(uuid.uuid4())[:12]),
                cluster_type=item.get("cluster_type", "experiential"),
                memory_ids=mids,
                coherence=item.get("coherence", 0.0),
                topic=item.get("topic", "general"),
                created_at=item.get("created_at", time.time()),
                updated_at=item.get("updated_at", time.time()),
                tag_signature=set(item.get("tag_signature", [])),
            )
            self._clusters.append(cluster)
            for mid in mids:
                self._memory_to_cluster[mid] = cluster.id
            restored += 1
        if restored:
            logger.info("Restored %d memory clusters (%d memory assignments)",
                        restored, len(self._memory_to_cluster))
        return restored

    # ------------------------------------------------------------------
    # Insights (read-only analysis of current cluster state)
    # ------------------------------------------------------------------

    def get_insights(self, memories: list[Memory]) -> ClusterInsight:
        insight = ClusterInsight()

        type_counts: dict[str, int] = {}
        for c in self._clusters:
            type_counts[c.cluster_type] = type_counts.get(c.cluster_type, 0) + 1

        for ctype, count in type_counts.items():
            if count >= 2:
                insight.patterns.append(f"Multiple {ctype} clusters ({count}) suggest recurring {ctype} themes")

        isolated = sum(1 for m in memories if m.id not in self._memory_to_cluster)
        if isolated > len(memories) * 0.4:
            insight.anomalies.append(f"{isolated} memories ({isolated*100//max(len(memories),1)}%) are unclustered")

        large_clusters = [c for c in self._clusters if len(c.memory_ids) > 10]
        if large_clusters:
            insight.trends.append(f"{len(large_clusters)} large cluster(s) dominating memory organization")

        for c1 in self._clusters:
            for c2 in self._clusters:
                if c1.id >= c2.id:
                    continue
                ids1, ids2 = set(c1.memory_ids), set(c2.memory_ids)
                shared = sum(
                    1 for mid in ids1
                    for m in memories if m.id == mid
                    for assoc in m.associations if assoc in ids2
                )
                if shared > 0:
                    insight.connections.append(f"Clusters '{c1.topic}' and '{c2.topic}' share {shared} cross-links")

        if not self._clusters and len(memories) > 5:
            insight.gaps.append("No clusters formed despite having memories — topics may be too diverse")

        return insight

    def find_cluster_for(self, memory_id: str) -> str | None:
        return self._memory_to_cluster.get(memory_id)

    # ------------------------------------------------------------------
    # Internal: agglomerative clustering on a batch of memories
    # ------------------------------------------------------------------

    def _agglomerative_cluster(self, memories: list[Memory]) -> list[MemoryCluster]:
        """Build new clusters from a batch of unclustered memories."""
        if len(memories) < 2:
            return []

        active: list[list[Memory]] = [[m] for m in memories]

        while len(active) > 1:
            best_sim = -1.0
            best_i, best_j = 0, 1

            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    sim = self._cluster_similarity(active[i], active[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_i, best_j = i, j

            if best_sim < self._similarity_threshold:
                break
            if len(active[best_i]) + len(active[best_j]) > self._max_cluster_size:
                break

            merged = active[best_i] + active[best_j]
            active[best_i] = merged
            active.pop(best_j)

        result: list[MemoryCluster] = []
        for group in active:
            if len(group) < MIN_CLUSTER_SIZE:
                continue
            tag_sig: set[str] = set()
            for m in group:
                tag_sig.update(m.tags)
            cluster = MemoryCluster(
                id=str(uuid.uuid4())[:12],
                cluster_type=_classify_cluster_type(group),
                memory_ids=[m.id for m in group],
                coherence=self._compute_coherence(group),
                topic=self._extract_topic(group),
                tag_signature=tag_sig,
            )
            result.append(cluster)
        return result

    # ------------------------------------------------------------------
    # Internal: cluster evolution operations
    # ------------------------------------------------------------------

    def _prune_dead_refs(self, valid_ids: set[str]) -> int:
        """Remove references to memories that no longer exist."""
        pruned = 0
        dead_clusters: list[str] = []
        for cluster in self._clusters:
            before = len(cluster.memory_ids)
            cluster.memory_ids = [mid for mid in cluster.memory_ids if mid in valid_ids]
            removed = before - len(cluster.memory_ids)
            if removed:
                pruned += removed
                cluster.updated_at = time.time()
            if len(cluster.memory_ids) < MIN_CLUSTER_SIZE:
                dead_clusters.append(cluster.id)

        for cid in dead_clusters:
            self._clusters = [c for c in self._clusters if c.id != cid]

        stale_mids = [mid for mid, cid in self._memory_to_cluster.items()
                      if mid not in valid_ids or cid in dead_clusters]
        for mid in stale_mids:
            del self._memory_to_cluster[mid]

        if pruned:
            logger.debug("Pruned %d dead refs, removed %d empty clusters",
                         pruned, len(dead_clusters))
        return pruned

    def _split_overgrown(self, memories: list[Memory]) -> int:
        """Split clusters that exceed the split threshold."""
        splits = 0
        to_add: list[MemoryCluster] = []
        to_remove: list[str] = []

        mem_lookup = {m.id: m for m in memories}

        for cluster in self._clusters:
            if len(cluster.memory_ids) < SPLIT_THRESHOLD:
                continue

            cluster_mems = [mem_lookup[mid] for mid in cluster.memory_ids if mid in mem_lookup]
            if len(cluster_mems) < 4:
                continue

            mid = len(cluster_mems) // 2
            group_a = cluster_mems[:mid]
            group_b = cluster_mems[mid:]

            if len(group_a) >= MIN_CLUSTER_SIZE and len(group_b) >= MIN_CLUSTER_SIZE:
                to_remove.append(cluster.id)
                for group in (group_a, group_b):
                    new_cluster = MemoryCluster(
                        id=str(uuid.uuid4())[:12],
                        cluster_type=_classify_cluster_type(group),
                        memory_ids=[m.id for m in group],
                        coherence=self._compute_coherence(group),
                        topic=self._extract_topic(group),
                    )
                    to_add.append(new_cluster)
                splits += 1

        for cid in to_remove:
            for mid in list(self._memory_to_cluster):
                if self._memory_to_cluster.get(mid) == cid:
                    del self._memory_to_cluster[mid]
            self._clusters = [c for c in self._clusters if c.id != cid]

        for cluster in to_add:
            self._clusters.append(cluster)
            for mid in cluster.memory_ids:
                self._memory_to_cluster[mid] = cluster.id

        return splits

    def _merge_similar(self) -> int:
        """Merge small clusters with high topic/type similarity."""
        merges = 0
        merged_ids: set[str] = set()

        small = [c for c in self._clusters
                 if len(c.memory_ids) <= 4 and c.id not in merged_ids]

        for i, c1 in enumerate(small):
            if c1.id in merged_ids:
                continue
            for c2 in small[i + 1:]:
                if c2.id in merged_ids:
                    continue
                if c1.cluster_type != c2.cluster_type:
                    continue
                combined_size = len(c1.memory_ids) + len(c2.memory_ids)
                if combined_size > self._max_cluster_size:
                    continue
                topic_overlap = set(c1.topic.lower().split()) & set(c2.topic.lower().split())
                if not topic_overlap and c1.topic != c2.topic:
                    continue

                c1.memory_ids.extend(c2.memory_ids)
                c1.updated_at = time.time()
                for mid in c2.memory_ids:
                    self._memory_to_cluster[mid] = c1.id
                merged_ids.add(c2.id)
                merges += 1

        if merged_ids:
            self._clusters = [c for c in self._clusters if c.id not in merged_ids]

        return merges

    # ------------------------------------------------------------------
    # Internal: similarity / scoring helpers
    # ------------------------------------------------------------------

    def _cluster_similarity(self, group_a: list[Memory], group_b: list[Memory]) -> float:
        total = 0.0
        count = 0
        for a in group_a:
            for b in group_b:
                total += _compute_tag_similarity(a, b)
                count += 1
        return total / count if count > 0 else 0.0

    def _memory_cluster_similarity(self, memory: Memory, cluster: MemoryCluster) -> float:
        mem_tags = set(memory.tags)
        if not mem_tags:
            return 0.0
        cluster_tags = cluster.tag_signature or {cluster.topic}
        type_kws = _TYPE_KEYWORDS.get(cluster.cluster_type, set())
        all_cluster_tags = cluster_tags | type_kws

        overlap = len(mem_tags & all_cluster_tags)
        union = len(mem_tags | all_cluster_tags)
        return overlap / union if union else 0.0

    def _compute_coherence(self, group: list[Memory]) -> float:
        if len(group) < 2:
            return 1.0
        total = 0.0
        count = 0
        for i, a in enumerate(group):
            for b in group[i+1:]:
                total += _compute_tag_similarity(a, b)
                count += 1
        return total / count if count > 0 else 0.0

    def _extract_topic(self, group: list[Memory]) -> str:
        tag_freq: dict[str, int] = {}
        for m in group:
            for tag in m.tags:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
        if not tag_freq:
            return "general"
        return max(tag_freq, key=tag_freq.get)


memory_cluster_engine = MemoryClusterEngine()
