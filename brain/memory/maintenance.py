"""Memory maintenance — integrity checks, garbage collection, and optimization."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from consciousness.events import Memory, event_bus

logger = logging.getLogger(__name__)

MAX_MEMORIES = int(os.environ.get("JARVIS_MAX_MEMORIES", "2000"))
GC_THRESHOLD = MAX_MEMORIES
WEAK_THRESHOLD = 0.05
AUTO_REPAIR_THRESHOLD = 3


@dataclass(frozen=True)
class IntegrityReport:
    is_healthy: bool
    score: float
    issues: tuple[str, ...]
    orphaned_associations: int
    duplicate_ids: int
    invalid_weights: int
    timestamp: float


@dataclass(frozen=True)
class MaintenanceResult:
    status: str  # healthy, degraded, corrupted
    removed_count: int
    repaired_count: int


class MemoryMaintenance:
    """Systematic memory health checks and repair operations."""

    _instance: MemoryMaintenance | None = None

    def __init__(self) -> None:
        self._last_gc_time: float = 0.0
        self._last_integrity_check: float = 0.0
        self._total_gc_runs: int = 0
        self._total_repaired: int = 0

    @classmethod
    def get_instance(cls) -> MemoryMaintenance:
        if cls._instance is None:
            cls._instance = MemoryMaintenance()
        return cls._instance

    def perform_integrity_check(self, memories: list[Memory]) -> IntegrityReport:
        """Full integrity scan: duplicates, orphans, invalid weights, broken refs."""
        issues: list[str] = []

        # Duplicate IDs
        ids = [m.id for m in memories]
        unique_ids = set(ids)
        duplicate_ids = len(ids) - len(unique_ids)
        if duplicate_ids > 0:
            issues.append(f"{duplicate_ids} duplicate memory IDs")

        # Orphaned associations
        orphaned = self._count_orphaned_associations(memories)
        if orphaned > 0:
            issues.append(f"{orphaned} orphaned associations")

        # Invalid weights
        invalid_weights = sum(
            1 for m in memories
            if m.weight < 0 or m.weight > 1
        )
        if invalid_weights > 0:
            issues.append(f"{invalid_weights} invalid weights")

        total_issues = duplicate_ids + orphaned + invalid_weights
        max_issues = max(1, len(memories) * 0.1)
        score = max(0.0, 1.0 - total_issues / max_issues)

        self._last_integrity_check = time.time()

        return IntegrityReport(
            is_healthy=total_issues == 0,
            score=round(score, 4),
            issues=tuple(issues),
            orphaned_associations=orphaned,
            duplicate_ids=duplicate_ids,
            invalid_weights=invalid_weights,
            timestamp=time.time(),
        )

    def garbage_collect(self, memories: list[Memory], max_count: int = MAX_MEMORIES) -> tuple[list[Memory], MaintenanceResult]:
        """Remove lowest-retention memories when over capacity."""
        if len(memories) <= max_count:
            return memories, MaintenanceResult("healthy", 0, 0)

        now = time.time()
        scored = [(m, self._retention_score(m, now)) for m in memories]
        scored.sort(key=lambda x: (x[0].is_core, x[1]), reverse=True)

        kept = [m for m, _ in scored[:max_count]]
        kept_ids = {m.id for m in kept}
        evicted_ids = [m.id for m in memories if m.id not in kept_ids]

        try:
            from memory.search import get_vector_store
            vs = get_vector_store()
            if vs:
                for mid in evicted_ids:
                    vs.remove(mid)
        except Exception:
            pass

        self._last_gc_time = now
        self._total_gc_runs += 1
        logger.info("Memory GC: removed %d memories (kept %d)", len(evicted_ids), len(kept))

        return kept, MaintenanceResult("healthy", len(evicted_ids), 0)

    def clean_orphaned_associations(self, memories: list[Memory]) -> tuple[list[Memory], int]:
        """Remove associations pointing to non-existent memories."""
        mem_ids = {m.id for m in memories}
        repaired = 0
        cleaned: list[Memory] = []

        for m in memories:
            valid_assocs = tuple(a for a in m.associations if a in mem_ids)
            if len(valid_assocs) != len(m.associations):
                repaired += 1
                cleaned.append(Memory(
                    **{**asdict(m),
                       "associations": valid_assocs,
                       "association_count": len(valid_assocs)}
                ))
            else:
                cleaned.append(m)

        if repaired > 0:
            self._total_repaired += repaired
            logger.info("Cleaned %d orphaned associations", repaired)

        return cleaned, repaired

    def optimize_storage(self, memories: list[Memory]) -> tuple[list[Memory], MaintenanceResult]:
        """Remove weak, isolated, old memories that add no value."""
        now = time.time()
        one_day = 86400.0
        kept: list[Memory] = []
        removed = 0

        for m in memories:
            if m.is_core:
                kept.append(m)
                continue

            is_weak = m.weight < WEAK_THRESHOLD
            is_isolated = len(m.associations) == 0
            is_old = (now - m.timestamp) > one_day

            if is_weak and is_isolated and is_old:
                removed += 1
            else:
                kept.append(m)

        if removed > 0:
            logger.info("Storage optimization: removed %d weak memories", removed)

        return kept, MaintenanceResult("healthy", removed, 0)

    def run_full_maintenance(self, memories: list[Memory]) -> dict[str, Any]:
        """Run complete maintenance cycle: integrity check, clean, GC, optimize."""
        results: dict[str, Any] = {}

        integrity = self.perform_integrity_check(memories)
        results["integrity"] = {
            "healthy": integrity.is_healthy,
            "score": integrity.score,
            "issues": list(integrity.issues),
        }

        current = list(memories)

        if integrity.orphaned_associations > 0:
            current, orphan_repaired = self.clean_orphaned_associations(current)
            results["orphan_cleanup"] = orphan_repaired

        if len(current) > GC_THRESHOLD:
            current, gc_result = self.garbage_collect(current)
            results["gc"] = {"removed": gc_result.removed_count}

        current, opt_result = self.optimize_storage(current)
        results["optimization"] = {"removed": opt_result.removed_count}

        results["final_count"] = len(current)
        results["memories"] = current
        return results

    def get_state(self) -> dict[str, Any]:
        return {
            "last_gc_time": self._last_gc_time,
            "last_integrity_check": self._last_integrity_check,
            "total_gc_runs": self._total_gc_runs,
            "total_repaired": self._total_repaired,
        }

    def _retention_score(self, memory: Memory, now: float) -> float:
        age_s = now - memory.timestamp
        recency_bonus = max(0.0, 1.0 - age_s / 86400.0) if age_s < 86400 else 0.0
        assoc_bonus = min(0.3, len(memory.associations) * 0.1)
        core_bonus = 1.0 if memory.is_core else 0.0
        priority_bonus = memory.priority / 1000.0 * 0.3
        access_bonus = min(0.3, memory.access_count * 0.03)
        access_recency = 0.0
        if memory.last_accessed > 0:
            since_accessed = now - memory.last_accessed
            access_recency = max(0.0, 1.0 - since_accessed / 172800.0) * 0.2
        return (
            memory.weight
            + recency_bonus * 0.2
            + assoc_bonus
            + core_bonus
            + priority_bonus
            + access_bonus
            + access_recency
        )

    def _count_orphaned_associations(self, memories: list[Memory]) -> int:
        mem_ids = {m.id for m in memories}
        return sum(
            1 for m in memories
            for a in m.associations if a not in mem_ids
        )


memory_maintenance = MemoryMaintenance.get_instance()
