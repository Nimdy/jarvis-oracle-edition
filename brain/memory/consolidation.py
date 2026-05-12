"""Memory consolidation engine — merges high-coherence clusters during dream cycles.

Implements Memory Bubble Merging from the Synthetic Soul architecture:
clusters of related memories are compressed into summary memories, with
source memories tagged and downweighted so they naturally evict over time.

Consolidation decisions are scored by two signals:
  - cluster coherence (tag similarity within the group)
  - retrieval heat (how often cluster members are actually used)

Conflict detection skips clusters with known epistemic tensions.
Consolidated outputs start at a trust discount (0.85x source weight)
and must survive waking retrieval to earn full trust.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from nanoid import generate as nanoid

if TYPE_CHECKING:
    from consciousness.events import Memory
    from memory.clustering import MemoryCluster

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    count: int = 0
    summaries: list[Any] = field(default_factory=list)
    skipped_conflict: int = 0
    skipped_too_small: int = 0
    skipped_low_score: int = 0


class MemoryConsolidationEngine:
    MAX_PER_CYCLE = 3
    MIN_CLUSTER_SIZE = 3
    MIN_COHERENCE = 0.65
    MIN_ACCESS_HEAT = 3
    TRUST_DISCOUNT = 0.85

    def run_consolidation(
        self,
        clusters: list[Any],
        memories: list[Any],
        contradiction_engine: Any = None,
    ) -> ConsolidationResult:
        """Consolidate top-scoring clusters into summary memories.

        Returns ConsolidationResult with new summary Memory objects and stats.
        Caller is responsible for persisting summaries and tagging sources.
        """
        result = ConsolidationResult()
        if not clusters or not memories:
            return result

        memories_map: dict[str, Any] = {m.id: m for m in memories}

        scored: list[tuple[float, Any]] = []
        for cluster in clusters:
            score = self._score_cluster(cluster, memories_map)
            if score < 0:
                if hasattr(cluster, 'coherence') and cluster.coherence < self.MIN_COHERENCE:
                    result.skipped_low_score += 1
                else:
                    result.skipped_too_small += 1
                continue
            scored.append((score, cluster))

        scored.sort(key=lambda x: x[0], reverse=True)

        for score, cluster in scored[:self.MAX_PER_CYCLE * 2]:
            if result.count >= self.MAX_PER_CYCLE:
                break

            cluster_mems = [
                memories_map[mid]
                for mid in cluster.memory_ids
                if mid in memories_map
            ]
            if len(cluster_mems) < self.MIN_CLUSTER_SIZE:
                result.skipped_too_small += 1
                continue

            conflicts = self._detect_conflicts(cluster_mems, contradiction_engine)
            if conflicts:
                result.skipped_conflict += 1
                logger.debug(
                    "Consolidation skipped cluster %s: %d conflicts detected",
                    getattr(cluster, 'id', '?'), len(conflicts),
                )
                continue

            summary = self._build_summary(cluster, cluster_mems)
            if summary is not None:
                result.summaries.append(summary)
                result.count += 1

        if result.count > 0:
            logger.info(
                "Consolidation produced %d summaries (skipped: %d conflict, %d small, %d low-score)",
                result.count, result.skipped_conflict,
                result.skipped_too_small, result.skipped_low_score,
            )
        return result

    _ALREADY_CONSOLIDATED_TAGS = frozenset({
        "consolidated", "dream_consolidation",
        "dream_artifact", "dream_consolidation_proposal",
    })

    def _score_cluster(self, cluster: Any, memories_map: dict[str, Any]) -> float:
        """Score a cluster for consolidation priority.

        Returns -1.0 if the cluster doesn't meet minimum requirements.
        """
        mids = getattr(cluster, 'memory_ids', [])
        cluster_mems = [memories_map[mid] for mid in mids if mid in memories_map]
        if len(cluster_mems) < self.MIN_CLUSTER_SIZE:
            return -1.0

        already_consolidated = sum(
            1 for m in cluster_mems
            if set(getattr(m, 'tags', ())) & self._ALREADY_CONSOLIDATED_TAGS
        )
        if already_consolidated >= len(cluster_mems) * 0.5:
            return -1.0

        coherence = getattr(cluster, 'coherence', 0.0)
        if coherence < self.MIN_COHERENCE:
            return -1.0

        total_access = sum(getattr(m, 'access_count', 0) for m in cluster_mems)
        heat = min(1.0, total_access / 20.0)

        return coherence * 0.6 + heat * 0.4

    def _detect_conflicts(
        self, cluster_memories: list[Any], contradiction_engine: Any,
    ) -> list[tuple[str, str]]:
        """Lightweight conflict check — skip clusters with active tensions."""
        if not contradiction_engine:
            return []

        conflicts: list[tuple[str, str]] = []
        try:
            tensions = contradiction_engine._belief_store.get_active_tensions()
            if not tensions:
                return []

            tension_subjects: set[str] = set()
            for t in tensions:
                if hasattr(t, 'subject'):
                    tension_subjects.add(t.subject.lower())
                if hasattr(t, 'proposition_a'):
                    tension_subjects.add(str(t.proposition_a).lower()[:200])

            for m in cluster_memories:
                payload = getattr(m, 'payload', None)
                if isinstance(payload, dict):
                    text = str(payload.get('text', '')).lower()[:500]
                else:
                    text = str(payload).lower()[:500]
                if not text:
                    continue
                for ts in tension_subjects:
                    if ts and len(ts) > 4 and ts in text:
                        conflicts.append((m.id, ts))
                        break
        except Exception:
            logger.debug("Conflict detection failed", exc_info=True)

        return conflicts

    @staticmethod
    def _strip_meta_headers(text: str) -> str:
        """Remove consolidation/dream meta-headers to extract actual content."""
        import re
        text = re.sub(r'\[Consolidated from \d+ memories\]\s*', '', text)
        text = re.sub(r'\[Dream artifact: [^\]]*\]\s*', '', text)
        text = re.sub(r'Consolidation: \w+\s*\(\d+ memories, coherence=[\d.]+\)\s*', '', text)
        text = re.sub(r'\[\+\d+ more\]\s*', '', text)
        text = re.sub(r'\s*\|\s*', ' ', text).strip()
        return text

    def _build_summary(self, cluster: Any, cluster_mems: list[Any]) -> Any:
        """Create a summary Memory from a cluster's members."""
        from consciousness.events import Memory
        from memory.core import MEMORY_TYPE_CONFIGS

        texts: list[str] = []
        max_weight = 0.0
        all_tags: set[str] = set()
        source_ids: list[str] = []

        for m in cluster_mems:
            source_ids.append(m.id)
            max_weight = max(max_weight, m.weight)

            payload = getattr(m, 'payload', None)
            if isinstance(payload, dict):
                text = payload.get('text', '') or payload.get('summary', '')
            else:
                text = str(payload) if payload else ''
            if text:
                stripped = self._strip_meta_headers(text)
                if len(stripped) >= 20:
                    texts.append(stripped[:800])

            for t in (m.tags or ()):
                if t not in ('dream_insight', 'dream_hypothesis', 'dream_artifact',
                             'sleep_candidate', 'consolidated'):
                    all_tags.add(t)

        if not texts:
            return None

        summary_text = " | ".join(texts[:8])
        if len(texts) > 8:
            summary_text += f" [+{len(texts) - 8} more]"

        common_tags = set()
        if all_tags:
            tag_counts: dict[str, int] = {}
            for m in cluster_mems:
                for t in (m.tags or ()):
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            threshold = max(2, len(cluster_mems) // 2)
            common_tags = {t for t, c in tag_counts.items() if c >= threshold}

        final_tags = tuple(
            sorted({"consolidated", "dream_consolidation"} | common_tags)
        )

        config = MEMORY_TYPE_CONFIGS.get("contextual_insight")
        decay_rate = config.decay_rate if config else 0.02
        priority = config.priority if config else 600

        summary_weight = max_weight * self.TRUST_DISCOUNT

        summary_id = f"consol_{nanoid(size=12)}"
        return Memory(
            id=summary_id,
            timestamp=time.time(),
            weight=min(0.55, summary_weight),
            tags=final_tags,
            payload={
                "text": f"[Consolidated from {len(source_ids)} memories] {summary_text}",
                "summary": summary_text,
                "source_ids": source_ids,
                "source_count": len(source_ids),
                "coherence": getattr(cluster, 'coherence', 0.0),
                "cluster_topic": getattr(cluster, 'topic', ''),
            },
            type="contextual_insight",
            associations=tuple(source_ids),
            decay_rate=decay_rate,
            priority=priority,
            provenance="consolidation",
        )


memory_consolidation_engine = MemoryConsolidationEngine()
