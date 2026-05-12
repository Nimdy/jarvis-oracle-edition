"""In-memory storage for Memory objects — CRUD, decay, trim, stats."""

from __future__ import annotations

import math
import os
import threading
import time as _time
from dataclasses import asdict
from typing import Any

from consciousness.events import Memory, event_bus, MEMORY_ASSOCIATED, MEMORY_TRIMMED
from memory.core import memory_core

# Weight caps to prevent saturation (all non-core memories must earn salience)
_NEW_MEMORY_WEIGHT_CAP = 0.55       # max initial weight for non-core memories
_REINFORCED_WEIGHT_CAP = 0.75       # max weight after reinforcement multiplier

# Tags that slow decay (emotional, identity-tied memories fade slower)
_SLOW_DECAY_TAGS = frozenset({
    "emotion", "user_preference", "identity", "core", "relationship",
    "self_reflection",
})

# Dream-origin tags: memories with these tags skip the reinforcement multiplier
# and are categorically non-belief-bearing (see contradiction_engine._DREAM_INELIGIBLE_TAGS)
_DREAM_ORIGIN_TAGS = frozenset({
    "dream_insight",
    "dream_hypothesis",
    "sleep_candidate",
    "dream_artifact",
    "dream_consolidation",
})

_MAX_ASSOCIATIONS_PER_MEMORY = 30


class MemoryStorage:
    _instance: MemoryStorage | None = None

    def __init__(self, max_capacity: int = int(os.environ.get("JARVIS_MAX_MEMORIES", "2000"))) -> None:
        self._memories: list[Memory] = []
        self._max_capacity = max_capacity
        self._last_decay_time: float = _time.time()
        self._reinforcement_multiplier: float = 1.0
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> MemoryStorage:
        if cls._instance is None:
            cls._instance = MemoryStorage()
        return cls._instance

    # -- CRUD ---------------------------------------------------------------

    def set_reinforcement_multiplier(self, multiplier: float) -> None:
        """Policy NN control: scale newly added memory weights. >1 = stronger retention."""
        self._reinforcement_multiplier = max(0.5, min(2.0, multiplier))

    def add(
        self,
        memory: Memory,
        creation_context: dict[str, Any] | None = None,
    ) -> bool:
        if not memory_core.validate_memory(memory):
            return False

        if memory.provenance == "unknown":
            inferred = self._infer_provenance(memory)
            if inferred != "unknown":
                memory = Memory(**{**asdict(memory), "provenance": inferred})

        weight = memory.weight
        if not memory.is_core:
            weight = min(weight, _NEW_MEMORY_WEIGHT_CAP)

        _tags = set(memory.tags) if memory.tags else set()
        if self._reinforcement_multiplier != 1.0 and not (_tags & _DREAM_ORIGIN_TAGS):
            weight = min(_REINFORCED_WEIGHT_CAP, weight * self._reinforcement_multiplier)

        if abs(weight - memory.weight) > 1e-6:
            memory = Memory(**{**asdict(memory), "weight": weight})

        is_new = False
        trim_result = None
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.id == memory.id:
                    self._memories[i] = memory
                    return True
            self._memories.append(memory)
            is_new = True
            if len(self._memories) > self._max_capacity:
                trim_result = self._auto_trim_unlocked()

        if trim_result:
            self._post_trim_cleanup(trim_result)
        if is_new:
            self._log_creation(memory, creation_context)
        return True

    def _log_creation(
        self, memory: Memory, ctx: dict[str, Any] | None,
    ) -> None:
        """Log every new memory birth to the lifecycle log at the storage boundary."""
        try:
            from memory.lifecycle_log import memory_lifecycle_log
            c = ctx or {}
            memory_lifecycle_log.log_created(
                memory_id=memory.id,
                memory_type=memory.type,
                weight=memory.weight,
                decay_rate=memory.decay_rate,
                tags=memory.tags,
                payload=memory.payload,
                source=c.get("source", memory.type),
                user_present=c.get("user_present", False),
                speaker_known=c.get("speaker_known", False),
                conversation_active=c.get("conversation_active", False),
                mode=c.get("mode", "passive"),
                memory_count=len(self._memories),
                provenance=memory.provenance,
                salience_advised=c.get("salience_advised", False),
            )
        except Exception:
            pass

    def get(self, memory_id: str) -> Memory | None:
        with self._lock:
            for m in self._memories:
                if m.id == memory_id:
                    return m
        return None

    def get_all(self) -> list[Memory]:
        with self._lock:
            return list(self._memories)

    def get_recent(self, count: int) -> list[Memory]:
        with self._lock:
            return list(self._memories[-count:])

    def get_by_type(self, mem_type: str) -> list[Memory]:
        with self._lock:
            return [m for m in self._memories if m.type == mem_type]

    def get_by_tag(self, tag: str) -> list[Memory]:
        with self._lock:
            return [m for m in self._memories if tag in m.tags]

    def remove(self, memory_id: str) -> bool:
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.id == memory_id:
                    self._memories.pop(i)
                    return True
        return False

    def count(self) -> int:
        with self._lock:
            return len(self._memories)

    def clear(self) -> None:
        with self._lock:
            self._memories.clear()

    # -- Association graph --------------------------------------------------

    def associate(self, memory_id_a: str, memory_id_b: str) -> bool:
        """Create a bidirectional association between two memories."""
        if memory_id_a == memory_id_b:
            return False

        with self._lock:
            idx_a = idx_b = -1
            for i, m in enumerate(self._memories):
                if m.id == memory_id_a:
                    idx_a = i
                elif m.id == memory_id_b:
                    idx_b = i
            if idx_a == -1 or idx_b == -1:
                return False

            a, b = self._memories[idx_a], self._memories[idx_b]

            if (
                len(a.associations) >= _MAX_ASSOCIATIONS_PER_MEMORY
                or len(b.associations) >= _MAX_ASSOCIATIONS_PER_MEMORY
            ):
                return False

            new_assoc_a = tuple(set(a.associations + (memory_id_b,)))
            new_assoc_b = tuple(set(b.associations + (memory_id_a,)))

            self._memories[idx_a] = Memory(
                **{**asdict(a),
                   "associations": new_assoc_a,
                   "association_count": len(new_assoc_a)}
            )
            self._memories[idx_b] = Memory(
                **{**asdict(b),
                   "associations": new_assoc_b,
                   "association_count": len(new_assoc_b)}
            )

        event_bus.emit(MEMORY_ASSOCIATED,
                       memory_id_a=memory_id_a, memory_id_b=memory_id_b)
        return True

    def reinforce(self, memory_id: str, boost: float = 0.1) -> bool:
        """Boost a memory's weight, respecting the reinforcement cap for non-core memories."""
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.id == memory_id:
                    cap = 1.0 if m.is_core else _REINFORCED_WEIGHT_CAP
                    new_weight = min(cap, m.weight + boost)
                    self._memories[i] = Memory(
                        id=m.id, timestamp=m.timestamp,
                        weight=new_weight,
                        tags=m.tags, payload=m.payload, type=m.type,
                        associations=m.associations, decay_rate=m.decay_rate,
                        is_core=m.is_core, last_validated=m.last_validated,
                        association_count=m.association_count, priority=m.priority,
                        provenance=m.provenance,
                        identity_owner=m.identity_owner, identity_owner_type=m.identity_owner_type,
                        identity_subject=m.identity_subject, identity_subject_type=m.identity_subject_type,
                        identity_scope_key=m.identity_scope_key, identity_confidence=m.identity_confidence,
                        identity_needs_resolution=m.identity_needs_resolution,
                        access_count=m.access_count, last_accessed=m.last_accessed,
                    )
                    try:
                        from memory.lifecycle_log import memory_lifecycle_log
                        memory_lifecycle_log.log_reinforced(memory_id, boost, new_weight)
                    except Exception:
                        pass
                    return True
        return False

    def adjust_weight(self, memory_id: str, delta: float) -> bool:
        """Nudge memory weight up or down without changing decay characteristics."""
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.id == memory_id:
                    floor = 0.1 if m.is_core else 0.05
                    cap = 1.0 if m.is_core else _REINFORCED_WEIGHT_CAP
                    new_weight = max(floor, min(cap, m.weight + delta))
                    if abs(new_weight - m.weight) < 1e-6:
                        return True
                    self._memories[i] = Memory(
                        id=m.id, timestamp=m.timestamp,
                        weight=new_weight,
                        tags=m.tags, payload=m.payload, type=m.type,
                        associations=m.associations, decay_rate=m.decay_rate,
                        is_core=m.is_core, last_validated=m.last_validated,
                        association_count=m.association_count, priority=m.priority,
                        provenance=m.provenance,
                        identity_owner=m.identity_owner, identity_owner_type=m.identity_owner_type,
                        identity_subject=m.identity_subject, identity_subject_type=m.identity_subject_type,
                        identity_scope_key=m.identity_scope_key, identity_confidence=m.identity_confidence,
                        identity_needs_resolution=m.identity_needs_resolution,
                        access_count=m.access_count, last_accessed=m.last_accessed,
                    )
                    try:
                        from memory.lifecycle_log import memory_lifecycle_log
                        if delta > 0:
                            memory_lifecycle_log.log_reinforced(memory_id, delta, new_weight)
                    except Exception:
                        pass
                    return True
        return False

    def record_access(self, memory_id: str) -> bool:
        """Increment access_count and update last_accessed for a retrieved memory."""
        now = _time.time()
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.id == memory_id:
                    self._memories[i] = Memory(
                        **{**asdict(m), "access_count": m.access_count + 1, "last_accessed": now}
                    )
                    return True
        return False

    def tag_consolidated(self, memory_ids: list[str], summary_id: str) -> int:
        """Mark source memories as consolidated and reduce their weight.

        Source memories aren't deleted — they're tagged and downweighted so they
        naturally lose retention score battles against their summary over time.
        """
        target_set = set(memory_ids)
        tagged = 0
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.id in target_set:
                    new_tags = tuple(sorted(
                        set(m.tags) | {"consolidated", f"consolidated_into:{summary_id}"}
                    ))
                    new_weight = m.weight * 0.6
                    self._memories[i] = Memory(
                        **{**asdict(m), "tags": new_tags, "weight": new_weight}
                    )
                    tagged += 1
        return tagged

    def downweight(
        self, memory_id: str, weight_factor: float = 0.6, decay_rate_factor: float = 3.0,
    ) -> bool:
        """Reduce a memory's weight and accelerate its decay (for superseded knowledge).

        Creates a replacement frozen Memory with updated fields.
        """
        with self._lock:
            for i, m in enumerate(self._memories):
                if m.id == memory_id:
                    self._memories[i] = Memory(
                        id=m.id, timestamp=m.timestamp,
                        weight=max(0.05, m.weight * weight_factor),
                        tags=m.tags, payload=m.payload, type=m.type,
                        associations=m.associations,
                        decay_rate=min(0.1, m.decay_rate * decay_rate_factor),
                        is_core=m.is_core, last_validated=m.last_validated,
                        association_count=m.association_count, priority=m.priority,
                        provenance=m.provenance,
                        identity_owner=m.identity_owner, identity_owner_type=m.identity_owner_type,
                        identity_subject=m.identity_subject, identity_subject_type=m.identity_subject_type,
                        identity_scope_key=m.identity_scope_key, identity_confidence=m.identity_confidence,
                        identity_needs_resolution=m.identity_needs_resolution,
                        access_count=m.access_count, last_accessed=m.last_accessed,
                    )
                    return True
        return False

    def get_related(self, memory_id: str, depth: int = 2) -> list[Memory]:
        """Depth-first traversal of the association graph, capped at 50 results."""
        with self._lock:
            start = None
            for m in self._memories:
                if m.id == memory_id:
                    start = m
                    break
            if start is None:
                return []
            lookup: dict[str, Memory] = {m.id: m for m in self._memories}
        visited: set[str] = {memory_id}
        result: list[Memory] = []

        stack: list[tuple[str, int]] = [(memory_id, 0)]
        while stack and len(result) < 50:
            current_id, current_depth = stack.pop()
            current = lookup.get(current_id)
            if current is None:
                continue
            for assoc_id in current.associations:
                if assoc_id in visited:
                    continue
                visited.add(assoc_id)
                neighbour = lookup.get(assoc_id)
                if neighbour is None:
                    continue
                result.append(neighbour)
                if len(result) >= 50:
                    break
                if current_depth + 1 < depth:
                    stack.append((assoc_id, current_depth + 1))
        return result

    def get_association_stats(self) -> dict:
        """Summary statistics for the association graph."""
        with self._lock:
            total_assoc = 0
            max_count = 0
            most_connected_id = ""
            isolated = 0

            for m in self._memories:
                n = len(m.associations)
                total_assoc += n
                if n == 0:
                    isolated += 1
                if n > max_count:
                    max_count = n
                    most_connected_id = m.id

            count = len(self._memories)
        return {
            "total_connections": total_assoc // 2,
            "avg_per_memory": total_assoc / count if count else 0.0,
            "isolated_count": isolated,
            "most_connected_id": most_connected_id,
        }

    # -- Decay & maintenance ------------------------------------------------

    def decay_all(self) -> int:
        """Timestamp-aware exponential decay: weight *= e^(-decay_rate * days_elapsed).

        Memories with emotional/identity tags decay at half rate.
        Core memories never decay.
        """
        now = _time.time()

        with self._lock:
            elapsed_days = (now - self._last_decay_time) / 86400.0
            self._last_decay_time = now

            if elapsed_days < 0.0001:
                return 0

            decayed = 0
            _decay_log_ids: list[tuple[str, float, float]] = []
            updated: list[Memory] = []
            for m in self._memories:
                if m.is_core:
                    updated.append(m)
                    continue

                rate = m.decay_rate
                if any(t in _SLOW_DECAY_TAGS for t in m.tags):
                    rate *= 0.5

                factor = math.exp(-rate * elapsed_days)
                new_weight = max(0.0, m.weight * factor)

                if abs(new_weight - m.weight) > 1e-6:
                    decayed += 1
                    if m.weight >= 0.1 and new_weight < 0.1:
                        _decay_log_ids.append((m.id, m.weight, new_weight))
                    updated.append(Memory(
                        id=m.id, timestamp=m.timestamp, weight=new_weight,
                        tags=m.tags, payload=m.payload, type=m.type,
                        associations=m.associations, decay_rate=m.decay_rate,
                        is_core=m.is_core, last_validated=m.last_validated,
                        association_count=m.association_count, priority=m.priority,
                        provenance=m.provenance,
                        identity_owner=m.identity_owner, identity_owner_type=m.identity_owner_type,
                        identity_subject=m.identity_subject, identity_subject_type=m.identity_subject_type,
                        identity_scope_key=m.identity_scope_key, identity_confidence=m.identity_confidence,
                        identity_needs_resolution=m.identity_needs_resolution,
                        access_count=m.access_count, last_accessed=m.last_accessed,
                    ))
                else:
                    updated.append(m)
            self._memories = updated

        if _decay_log_ids:
            try:
                from memory.lifecycle_log import memory_lifecycle_log
                for mid, old_w, new_w in _decay_log_ids:
                    memory_lifecycle_log.log_decayed(mid, old_w, new_w)
            except Exception:
                pass

        return decayed

    def auto_trim(self) -> int:
        with self._lock:
            result = self._auto_trim_unlocked()
        if result:
            self._post_trim_cleanup(result)
            return result[0]
        return 0

    def _auto_trim_unlocked(self) -> tuple[int, set[str], dict[str, Memory], dict[str, float], int] | None:
        """Actual trim logic — caller must hold self._lock.

        Returns (trimmed_count, evicted_ids, before_map, retention_scores,
        remaining_count) for post-lock cleanup, or None if nothing trimmed.
        """
        now = _time.time()
        before_map = {m.id: m for m in self._memories}
        self._memories = [m for m in self._memories if m.is_core or m.weight > 0.05]

        def _retention_score(m: Memory) -> float:
            age_s = now - m.timestamp
            recency_bonus = max(0.0, 1.0 - age_s / 86400.0) if age_s < 86400 else 0.0
            association_bonus = min(0.3, len(m.associations) * 0.1)
            core_bonus = 1.0 if m.is_core else 0.0
            priority_bonus = m.priority / 1000.0 * 0.3
            access_bonus = min(0.3, m.access_count * 0.03)
            access_recency = 0.0
            if m.last_accessed > 0:
                since_accessed = now - m.last_accessed
                access_recency = max(0.0, 1.0 - since_accessed / 172800.0) * 0.2
            return (
                m.weight
                + recency_bonus * 0.2
                + association_bonus
                + core_bonus
                + priority_bonus
                + access_bonus
                + access_recency
            )

        retention_scores = {m.id: _retention_score(m) for m in self._memories}

        self._memories.sort(key=_retention_score, reverse=True)
        self._memories = self._memories[: self._max_capacity]
        after_ids = {m.id for m in self._memories}
        evicted_ids = set(before_map.keys()) - after_ids
        trimmed = len(evicted_ids)

        if trimmed > 0:
            self._clean_orphaned_associations()
            return (trimmed, evicted_ids, before_map, retention_scores, len(self._memories))

        return None

    def _post_trim_cleanup(
        self, result: tuple[int, set[str], dict[str, Memory], dict[str, float], int],
    ) -> None:
        """Run cleanup OUTSIDE the lock after a trim pass."""
        trimmed, evicted_ids, before_map, retention_scores, remaining = result
        self._clean_vector_store(evicted_ids)
        self._clean_index(evicted_ids, before_map)
        event_bus.emit(MEMORY_TRIMMED, count=trimmed, remaining=remaining)

        try:
            from memory.lifecycle_log import memory_lifecycle_log
            for mid in evicted_ids:
                evicted_mem = before_map.get(mid)
                memory_lifecycle_log.log_evicted(
                    mid,
                    final_weight=evicted_mem.weight if evicted_mem else 0.0,
                    retention_score=retention_scores.get(mid, 0.0),
                )
        except Exception:
            pass

    @staticmethod
    def _clean_vector_store(evicted_ids: set[str]) -> None:
        """Remove evicted memory embeddings from the vector store."""
        try:
            from memory.search import get_vector_store
            vs = get_vector_store()
            if vs:
                for mid in evicted_ids:
                    vs.remove(mid)
        except Exception:
            pass

    @staticmethod
    def _clean_index(evicted_ids: set[str], before_map: dict[str, Memory]) -> None:
        """Remove evicted memories from the tag/type index."""
        try:
            from memory.index import memory_index
            for mid in evicted_ids:
                mem = before_map.get(mid)
                if mem:
                    memory_index.remove_memory(mem)
        except Exception:
            pass

    def _clean_orphaned_associations(self) -> None:
        """Strip dangling association references after a trim pass."""
        live_ids = {m.id for m in self._memories}
        updated = []
        for m in self._memories:
            dead = [a for a in m.associations if a not in live_ids]
            if dead:
                pruned = tuple(a for a in m.associations if a in live_ids)
                updated.append(Memory(
                    id=m.id, timestamp=m.timestamp, weight=m.weight,
                    tags=m.tags, payload=m.payload, type=m.type,
                    associations=pruned, decay_rate=m.decay_rate,
                    is_core=m.is_core, last_validated=m.last_validated,
                    association_count=len(pruned), priority=m.priority,
                    provenance=m.provenance,
                    identity_owner=m.identity_owner, identity_owner_type=m.identity_owner_type,
                    identity_subject=m.identity_subject, identity_subject_type=m.identity_subject_type,
                    identity_scope_key=m.identity_scope_key, identity_confidence=m.identity_confidence,
                    identity_needs_resolution=m.identity_needs_resolution,
                    access_count=m.access_count, last_accessed=m.last_accessed,
                ))
            else:
                updated.append(m)
        self._memories = updated

    # -- Stats --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            by_type: dict[str, int] = {}
            by_provenance: dict[str, int] = {}
            total_weight = 0.0
            core_count = 0
            weights: list[float] = []
            strong_count = 0
            weak_count = 0
            for m in self._memories:
                by_type[m.type] = by_type.get(m.type, 0) + 1
                prov = getattr(m, "provenance", "unknown")
                by_provenance[prov] = by_provenance.get(prov, 0) + 1
                total_weight += m.weight
                weights.append(m.weight)
                if m.is_core:
                    core_count += 1
                if m.weight > 0.6:
                    strong_count += 1
                elif m.weight < 0.2:
                    weak_count += 1
            total = len(self._memories)

            weight_bins = [0] * 10
            for w in weights:
                idx = min(9, int(w * 10))
                weight_bins[idx] += 1

            sorted_w = sorted(weights)
            median = sorted_w[len(sorted_w) // 2] if sorted_w else 0.0

        return {
            "total": total,
            "core_count": core_count,
            "avg_weight": total_weight / total if total else 0,
            "median_weight": median,
            "strong_count": strong_count,
            "weak_count": weak_count,
            "weight_bins": weight_bins,
            "by_type": by_type,
            "by_provenance": by_provenance,
        }

    def get_recent_with_provenance(self, count: int = 20) -> list[dict[str, Any]]:
        """Return recent memories with provenance for dashboard observability."""
        with self._lock:
            recent = self._memories[-count:]
        return [
            {
                "id": m.id[:12],
                "type": m.type,
                "provenance": getattr(m, "provenance", "unknown"),
                "weight": round(m.weight, 3),
                "age_s": round(_time.time() - m.timestamp),
                "payload_preview": (m.payload[:60] if isinstance(m.payload, str)
                                    else str(m.payload)[:60]),
            }
            for m in reversed(recent)
        ]

    def get_tag_frequency(self) -> dict[str, int]:
        with self._lock:
            freq: dict[str, int] = {}
            for m in self._memories:
                for tag in m.tags:
                    freq[tag] = freq.get(tag, 0) + 1
        return freq

    # -- Serialisation ------------------------------------------------------

    def to_json(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(m) for m in self._memories]

    def load_from_json(self, data: list[dict[str, Any]]) -> int:
        with self._lock:
            existing_ids = {m.id for m in self._memories}
            loaded = 0
            for item in data:
                try:
                    if "tags" in item:
                        item["tags"] = tuple(item["tags"])
                    if "associations" in item:
                        item["associations"] = tuple(item["associations"])
                    mem = Memory(**item)
                    if mem.id in existing_ids:
                        continue
                    if memory_core.validate_memory(mem):
                        self._memories.append(mem)
                        existing_ids.add(mem.id)
                        loaded += 1
                except (TypeError, KeyError):
                    continue

            migrated = self._run_load_migrations()
        if migrated:
            import logging
            logging.getLogger(__name__).info(
                "Load migration: removed %d duplicate/junk memories", migrated,
            )
        return loaded

    @staticmethod
    def _infer_provenance(m: Memory) -> str:
        """Best-effort provenance recovery for legacy memories saved as 'unknown'.

        Uses memory type and tags to infer the most likely provenance. This is
        lossy but recovers most of the epistemic signal for downstream learners
        (ranker provenance tie-breaker, salience provenance feature, context
        builder research-backed section).
        """
        tags = set(m.tags) if m.tags else set()

        if "evidence:peer_reviewed" in tags or "evidence:codebase_verified" in tags:
            return "external_source"
        if m.type == "observation":
            return "observed"
        if m.type == "user_preference":
            return "user_claim"
        if m.type == "core":
            return "seed"
        if m.type in ("factual_knowledge", "contextual_insight"):
            if any(t.startswith("source:") for t in tags):
                return "external_source"
            return "model_inference"
        if m.type == "self_improvement" or "self_improve" in tags:
            return "experiment_result"
        if m.type == "error_recovery":
            return "model_inference"
        if "dream_insight" in tags or "reflection" in tags:
            return "model_inference"
        if "cluster_summary" in tags or "association_discovery" in tags:
            return "derived_pattern"
        if m.type == "conversation":
            return "conversation"
        return "unknown"

    def _run_load_migrations(self) -> int:
        """One-time migrations that run on every load to clean legacy data.

        1. Dedup observation memories (keep one per unique payload)
        2. Cap dream_insight count to MAX (evict oldest over cap)
        3. Reset saturated weights on non-core memories
        4. Backfill provenance for legacy 'unknown' memories
        """
        before = len(self._memories)

        seen_payloads: dict[str, str] = {}
        dedup_remove: set[str] = set()
        for m in self._memories:
            if m.type == "observation":
                payload_key = str(m.payload)
                if payload_key in seen_payloads:
                    dedup_remove.add(m.id)
                else:
                    seen_payloads[payload_key] = m.id

        if dedup_remove:
            self._memories = [m for m in self._memories if m.id not in dedup_remove]

        dream_insights = [
            (i, m) for i, m in enumerate(self._memories)
            if "dream_insight" in m.tags
        ]
        from consciousness.consciousness_system import MAX_DREAM_INSIGHT_MEMORIES
        if len(dream_insights) > MAX_DREAM_INSIGHT_MEMORIES:
            dream_insights.sort(key=lambda x: x[1].timestamp)
            to_remove = {
                idx for idx, _ in dream_insights[:-MAX_DREAM_INSIGHT_MEMORIES]
            }
            self._memories = [
                m for i, m in enumerate(self._memories) if i not in to_remove
            ]

        _TYPE_TARGET_WEIGHTS = {
            "observation": 0.30,
            "conversation": 0.50,
            "factual_knowledge": 0.45,
            "contextual_insight": 0.35,
            "self_improvement": 0.40,
            "error_recovery": 0.25,
            "task_completed": 0.40,
        }
        updated: list[Memory] = []
        for m in self._memories:
            if m.is_core:
                updated.append(m)
                continue
            target = _TYPE_TARGET_WEIGHTS.get(m.type)
            if target is not None and m.weight > _REINFORCED_WEIGHT_CAP:
                updated.append(Memory(
                    id=m.id, timestamp=m.timestamp, weight=target,
                    tags=m.tags, payload=m.payload, type=m.type,
                    associations=m.associations, decay_rate=m.decay_rate,
                    is_core=m.is_core, last_validated=m.last_validated,
                    association_count=m.association_count, priority=m.priority,
                    provenance=m.provenance,
                    identity_owner=m.identity_owner, identity_owner_type=m.identity_owner_type,
                    identity_subject=m.identity_subject, identity_subject_type=m.identity_subject_type,
                    identity_scope_key=m.identity_scope_key, identity_confidence=m.identity_confidence,
                    identity_needs_resolution=m.identity_needs_resolution,
                    access_count=m.access_count, last_accessed=m.last_accessed,
                ))
            else:
                updated.append(m)
        self._memories = updated

        backfilled = 0
        final: list[Memory] = []
        for m in self._memories:
            if m.provenance == "unknown":
                inferred = self._infer_provenance(m)
                if inferred != "unknown":
                    final.append(Memory(
                        id=m.id, timestamp=m.timestamp, weight=m.weight,
                        tags=m.tags, payload=m.payload, type=m.type,
                        associations=m.associations, decay_rate=m.decay_rate,
                        is_core=m.is_core, last_validated=m.last_validated,
                        association_count=m.association_count, priority=m.priority,
                        provenance=inferred,
                        identity_owner=m.identity_owner, identity_owner_type=m.identity_owner_type,
                        identity_subject=m.identity_subject, identity_subject_type=m.identity_subject_type,
                        identity_scope_key=m.identity_scope_key, identity_confidence=m.identity_confidence,
                        identity_needs_resolution=m.identity_needs_resolution,
                        access_count=m.access_count, last_accessed=m.last_accessed,
                    ))
                    backfilled += 1
                else:
                    final.append(m)
            else:
                final.append(m)
        self._memories = final

        if backfilled:
            import logging
            logging.getLogger(__name__).info(
                "Provenance backfill: inferred provenance for %d legacy memories", backfilled,
            )

        # ── Phase 5: identity backfill for legacy memories ──
        identity_backfilled = 0
        identity_final: list[Memory] = []
        _IDENTITY_MAP: dict[str, tuple[str, str, str, str, float]] = {
            "seed": ("jarvis", "self", "jarvis", "self", 1.0),
            "external_source": ("external", "library", "external", "library", 0.95),
            "model_inference": ("jarvis", "self", "jarvis", "self", 0.9),
            "experiment_result": ("jarvis", "self", "jarvis", "self", 0.9),
            "observed": ("scene", "environment", "scene", "environment", 0.85),
        }
        for m in self._memories:
            if m.identity_owner:
                identity_final.append(m)
                continue

            owner_id, owner_type, subj_id, subj_type, conf = "", "", "", "", 0.3
            prov = m.provenance

            if prov in _IDENTITY_MAP:
                owner_id, owner_type, subj_id, subj_type, conf = _IDENTITY_MAP[prov]
            elif prov in ("user_claim", "conversation"):
                speaker_tag = ""
                for t in m.tags:
                    if t.startswith("speaker:"):
                        speaker_tag = t[8:].strip()
                        break
                if speaker_tag and speaker_tag != "unknown":
                    owner_id = speaker_tag
                    owner_type = "primary_user"
                    subj_id = speaker_tag
                    subj_type = "primary_user"
                    conf = 0.8 if prov == "user_claim" else 0.7
                else:
                    owner_type = "unknown"
                    subj_type = "unknown"
                    conf = 0.3

            if owner_type:
                scope_key = f"{owner_type}:{owner_id}" if owner_type and owner_id else ""
                identity_final.append(Memory(
                    id=m.id, timestamp=m.timestamp, weight=m.weight,
                    tags=m.tags, payload=m.payload, type=m.type,
                    associations=m.associations, decay_rate=m.decay_rate,
                    is_core=m.is_core, last_validated=m.last_validated,
                    association_count=m.association_count, priority=m.priority,
                    provenance=m.provenance,
                    identity_owner=owner_id, identity_owner_type=owner_type,
                    identity_subject=subj_id, identity_subject_type=subj_type,
                    identity_scope_key=scope_key, identity_confidence=conf,
                    identity_needs_resolution=False,
                ))
                identity_backfilled += 1
            else:
                identity_final.append(m)

        self._memories = identity_final

        if identity_backfilled:
            import logging
            logging.getLogger(__name__).info(
                "Identity backfill: inferred identity for %d legacy memories", identity_backfilled,
            )

        return before - len(self._memories)


memory_storage = MemoryStorage.get_instance()
