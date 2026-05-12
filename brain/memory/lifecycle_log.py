"""Memory lifecycle logger — tracks creation, reinforcement, retrieval, injection, decay, eviction.

Produces training data for the SalienceModel NN by joining creation events with
lifecycle outcomes: memories that got reinforced/retrieved are positive examples;
memories that decayed to eviction unused are negative examples.

Storage: append-only JSONL at ~/.jarvis/memory_lifecycle_log.jsonl
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from memory.maintenance import MAX_MEMORIES as _MAX_MEMORIES
except ImportError:
    _MAX_MEMORIES = 2000

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
LOG_PATH = JARVIS_DIR / "memory_lifecycle_log.jsonl"
MAX_LOG_SIZE_MB = 10
_CREATION_BUFFER_CAP = 500
_MIN_LIFECYCLE_AGE_S = 3600.0  # 1 hour minimum age before a memory qualifies for training


@dataclass(frozen=True)
class LifecycleEvent:
    """Single lifecycle event for a memory."""
    memory_id: str
    event_type: str
    timestamp: float
    data: dict


@dataclass
class CreationRecord:
    """Cached creation data for joining with later lifecycle events."""
    memory_id: str
    memory_type: str
    initial_weight: float
    initial_decay_rate: float
    tag_count: int
    payload_length: int
    source: str
    user_present: bool
    speaker_known: bool
    conversation_active: bool
    mode: str
    memory_count_at_creation: int
    created_at: float
    provenance: str = "unknown"
    salience_advised: bool = False
    reinforced: bool = False
    retrieved: bool = False
    injected: bool = False
    reinforce_count: int = 0
    retrieval_count: int = 0
    peak_weight: float = 0.0
    evicted: bool = False
    evicted_at: float = 0.0


@dataclass
class SalienceTrainingPair:
    """Joined creation + lifecycle ready for salience model training."""
    features: list[float]
    store_label: float
    weight_label: float
    decay_label: float


class MemoryLifecycleLog:
    """Thread-safe append-only log for memory lifecycle events."""

    _instance: MemoryLifecycleLog | None = None

    def __init__(self, path: str | Path = "") -> None:
        self._path = Path(path) if path else LOG_PATH
        self._lock = threading.Lock()
        self._initialized = False
        self._total_events = 0
        self._creations: OrderedDict[str, CreationRecord] = OrderedDict()
        self._boot_ts: float = time.time()
        self._rehydrated: bool = False
        self._rehydrated_count: int = 0

    @classmethod
    def get_instance(cls) -> MemoryLifecycleLog:
        if cls._instance is None:
            cls._instance = MemoryLifecycleLog()
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def rehydrate(self, max_creations: int = 200) -> int:
        """Replay recent creation events from JSONL into in-memory buffers.

        Rebuilds CreationRecords with lifecycle annotations (reinforced, retrieved,
        injected, evicted) so salience metrics are meaningful immediately after restart.
        """
        if self._rehydrated or not self._path.exists():
            return 0

        try:
            lines = self._path.read_text().strip().split("\n")
            tail = lines[-min(len(lines), max_creations * 10):]
        except Exception:
            return 0

        creations: dict[str, CreationRecord] = {}
        for line in tail:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            etype = rec.get("type", "")
            mid = rec.get("mid", "")
            if not mid:
                continue

            is_creation = "weight" in rec and "decay_rate" in rec and "source" in rec
            if is_creation:
                mem_type = rec.get("mem_type") or etype
                creations[mid] = CreationRecord(
                    memory_id=mid,
                    memory_type=mem_type,
                    initial_weight=rec.get("weight", 0.5),
                    initial_decay_rate=rec.get("decay_rate", 0.001),
                    tag_count=rec.get("tags", 0),
                    payload_length=rec.get("payload_len", 0),
                    source=rec.get("source", "unknown"),
                    user_present=rec.get("user_present", False),
                    speaker_known=False,
                    conversation_active=False,
                    mode=rec.get("mode", "passive"),
                    memory_count_at_creation=0,
                    created_at=rec.get("t", 0.0),
                    peak_weight=rec.get("weight", 0.5),
                    provenance=rec.get("provenance", "unknown"),
                    salience_advised=rec.get("salience_advised", False),
                )
            elif etype == "reinforced" and mid in creations:
                creations[mid].reinforced = True
                creations[mid].reinforce_count += 1
                nw = rec.get("new_weight", 0.0)
                creations[mid].peak_weight = max(creations[mid].peak_weight, nw)
            elif etype == "retrieved" and mid in creations:
                creations[mid].retrieved = True
                creations[mid].retrieval_count += 1
            elif etype == "injected" and mid in creations:
                creations[mid].injected = True
            elif etype == "evicted" and mid in creations:
                creations[mid].evicted = True
                creations[mid].evicted_at = rec.get("t", 0.0)

        recent = list(creations.values())[-max_creations:]
        count = 0
        with self._lock:
            for cr in recent:
                self._creations[cr.memory_id] = cr
                if len(self._creations) > _CREATION_BUFFER_CAP:
                    self._creations.popitem(last=False)
                count += 1
            self._total_events += count

        self._rehydrated = True
        self._rehydrated_count = count
        if count > 0:
            logger.info("Rehydrated %d lifecycle creation records from JSONL warm-start", count)
        return count

    def log_created(
        self,
        memory_id: str,
        memory_type: str,
        weight: float,
        decay_rate: float,
        tags: tuple[str, ...],
        payload: Any,
        source: str = "unknown",
        user_present: bool = False,
        speaker_known: bool = False,
        conversation_active: bool = False,
        mode: str = "passive",
        memory_count: int = 0,
        provenance: str = "unknown",
        salience_advised: bool = False,
    ) -> None:
        """Log memory creation with context features for salience training."""
        payload_text = payload if isinstance(payload, str) else str(payload)
        record = CreationRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            initial_weight=weight,
            initial_decay_rate=decay_rate,
            tag_count=len(tags),
            payload_length=min(len(payload_text), 500),
            source=source,
            user_present=user_present,
            speaker_known=speaker_known,
            conversation_active=conversation_active,
            mode=mode,
            memory_count_at_creation=memory_count,
            created_at=time.time(),
            peak_weight=weight,
            provenance=provenance,
            salience_advised=salience_advised,
        )

        with self._lock:
            self._creations[memory_id] = record
            self._creations.move_to_end(memory_id)
            if len(self._creations) > _CREATION_BUFFER_CAP:
                self._creations.popitem(last=False)

        self._append_event("created", memory_id, {
            "mem_type": memory_type,
            "weight": round(weight, 4),
            "decay_rate": round(decay_rate, 5),
            "tags": len(tags),
            "payload_len": record.payload_length,
            "source": source,
            "user_present": user_present,
            "mode": mode,
            "provenance": provenance,
            "salience_advised": salience_advised,
        })

    def log_reinforced(self, memory_id: str, boost: float, new_weight: float) -> None:
        _was_salience_advised = False
        with self._lock:
            rec = self._creations.get(memory_id)
            if rec:
                rec.reinforced = True
                rec.reinforce_count += 1
                rec.peak_weight = max(rec.peak_weight, new_weight)
                _was_salience_advised = rec.salience_advised

        if _was_salience_advised:
            try:
                from memory.salience import get_salience_model
                salience = get_salience_model()
                if salience:
                    salience.record_validated_prediction()
            except Exception:
                pass

        self._append_event("reinforced", memory_id, {
            "boost": round(boost, 4),
            "new_weight": round(new_weight, 4),
        })

    def log_retrieved(self, memory_id: str) -> None:
        with self._lock:
            rec = self._creations.get(memory_id)
            if rec:
                rec.retrieved = True
                rec.retrieval_count += 1

        self._append_event("retrieved", memory_id, {})

    def log_injected(self, memory_id: str) -> None:
        with self._lock:
            rec = self._creations.get(memory_id)
            if rec:
                rec.injected = True

        self._append_event("injected", memory_id, {})

    def log_decayed(self, memory_id: str, old_weight: float, new_weight: float) -> None:
        self._append_event("decayed", memory_id, {
            "old_weight": round(old_weight, 4),
            "new_weight": round(new_weight, 4),
        })

    def log_evicted(self, memory_id: str, final_weight: float, retention_score: float = 0.0) -> None:
        _was_salience_advised = False
        with self._lock:
            rec = self._creations.get(memory_id)
            if rec:
                rec.evicted = True
                rec.evicted_at = time.time()
                _was_salience_advised = rec.salience_advised

        if _was_salience_advised:
            try:
                from memory.salience import get_salience_model
                salience = get_salience_model()
                if salience:
                    salience.record_validated_prediction()
            except Exception:
                pass

        self._append_event("evicted", memory_id, {
            "final_weight": round(final_weight, 4),
            "retention_score": round(retention_score, 4),
        })

    def get_salience_training_pairs(self, limit: int = 500) -> list[SalienceTrainingPair]:
        """Join creation records with lifecycle outcomes for salience training.

        Only includes memories old enough to have meaningful lifecycle data.
        """
        now = time.time()
        pairs: list[SalienceTrainingPair] = []

        with self._lock:
            records = list(self._creations.values())

        for rec in records:
            age = now - rec.created_at
            if age < _MIN_LIFECYCLE_AGE_S:
                continue

            features = build_creation_features(
                source=rec.source,
                initial_weight=rec.initial_weight,
                user_present=rec.user_present,
                mode=rec.mode,
                memory_count=rec.memory_count_at_creation,
                speaker_known=rec.speaker_known,
                conversation_active=rec.conversation_active,
                memory_type=rec.memory_type,
                payload_length=rec.payload_length,
                provenance=getattr(rec, "provenance", "unknown"),
            )

            was_useful = rec.reinforced or rec.retrieved or rec.injected

            if was_useful and rec.retrieval_count >= 2:
                store_label = 1.0
                weight_label = min(1.0, rec.peak_weight)
                decay_label = max(0.0, min(1.0, rec.initial_decay_rate * 50.0))
            elif was_useful:
                store_label = 0.8
                weight_label = min(1.0, rec.peak_weight * 0.9)
                decay_label = max(0.0, min(1.0, rec.initial_decay_rate * 50.0))
            elif rec.evicted:
                store_label = 0.2
                weight_label = 0.1
                decay_label = 0.8
            elif age > 86400.0 and not was_useful:
                store_label = 0.3
                weight_label = 0.2
                decay_label = 0.6
            else:
                continue

            pairs.append(SalienceTrainingPair(
                features=features,
                store_label=store_label,
                weight_label=weight_label,
                decay_label=decay_label,
            ))

        return pairs[-limit:]

    def get_effectiveness_metrics(self) -> dict[str, Any]:
        """Salience effectiveness: wasted rate, useful rate, prediction error.

        All computation outside the lock.
        """
        now = time.time()
        with self._lock:
            records = list(self._creations.values())

        mature = [r for r in records if (now - r.created_at) >= _MIN_LIFECYCLE_AGE_S]
        if not mature:
            return {
                "wasted_rate": 0.0, "useful_rate": 0.0,
                "weight_error": 0.0, "decay_error": 0.0,
                "mature_count": 0,
            }

        useful = [r for r in mature if r.reinforced or r.retrieved or r.injected]
        wasted = [r for r in mature if r.evicted and not (r.reinforced or r.retrieved)]

        wasted_rate = len(wasted) / len(mature) if mature else 0.0
        useful_rate = len(useful) / len(mature) if mature else 0.0

        weight_errors: list[float] = []
        decay_errors: list[float] = []
        for r in mature:
            if r.peak_weight > 0:
                weight_errors.append(abs(r.initial_weight - r.peak_weight))
            if r.evicted and r.evicted_at > r.created_at:
                actual_lifetime_h = (r.evicted_at - r.created_at) / 3600.0
                predicted_lifetime_h = (1.0 / max(r.initial_decay_rate, 1e-6)) * 24.0 if r.initial_decay_rate > 0 else 168.0
                decay_errors.append(abs(actual_lifetime_h - predicted_lifetime_h) / max(predicted_lifetime_h, 1.0))

        avg_weight_err = sum(weight_errors) / len(weight_errors) if weight_errors else 0.0
        avg_decay_err = sum(decay_errors) / len(decay_errors) if decay_errors else 0.0

        _DECAY_THRESHOLDS = [(0.2, "good"), (0.5, "fair")]
        decay_status = "poor"
        for threshold, label in _DECAY_THRESHOLDS:
            if avg_decay_err <= threshold:
                decay_status = label
                break

        return {
            "wasted_rate": round(wasted_rate, 4),
            "useful_rate": round(useful_rate, 4),
            "weight_error": round(avg_weight_err, 4),
            "decay_error": round(avg_decay_err, 4),
            "decay_status": decay_status,
            "mature_count": len(mature),
            "useful_count": len(useful),
            "wasted_count": len(wasted),
        }

    def get_stats(self) -> dict[str, Any]:
        """Snapshot for dashboard. Lock-safe (no nested calls)."""
        with self._lock:
            total_created = len(self._creations)
            useful = sum(1 for r in self._creations.values() if r.reinforced or r.retrieved)
            evicted = sum(1 for r in self._creations.values() if r.evicted)

        return {
            "total_events": self._total_events,
            "tracked_creations": total_created,
            "useful_memories": useful,
            "evicted_memories": evicted,
            "log_exists": self._path.exists(),
            "log_size_kb": round(self._path.stat().st_size / 1024, 1) if self._path.exists() else 0,
            "boot_ts": round(self._boot_ts, 3),
            "rehydrated": self._rehydrated,
            "rehydrated_count": self._rehydrated_count,
        }

    def get_new_event_count(self) -> int:
        return self._total_events

    def _append_event(self, event_type: str, memory_id: str, data: dict) -> None:
        self._total_events += 1
        self._append({
            "type": event_type,
            "mid": memory_id,
            "t": round(time.time(), 3),
            **data,
        })

    def _append(self, entry: dict[str, Any]) -> None:
        if not self._initialized:
            self.init()
        with self._lock:
            try:
                if self._path.exists() and self._path.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
                    self._rotate()
                with open(self._path, "a") as f:
                    f.write(json.dumps(entry, separators=(",", ":"), default=str) + "\n")
            except Exception as exc:
                logger.debug("Lifecycle log write failed: %s", exc)

    def _rotate(self) -> None:
        try:
            lines = self._path.read_text().strip().split("\n")
            keep = lines[len(lines) // 2:]
            self._path.write_text("\n".join(keep) + "\n")
            logger.info("Rotated lifecycle log: %d -> %d entries", len(lines), len(keep))
        except Exception:
            pass


_SOURCE_MAP = {
    "conversation": 0, "research": 1, "dream": 2, "reflection": 3,
    "proactive": 4, "study": 5, "preference": 6, "transaction": 7,
    "unknown": 8,
}
_TYPE_MAP = {
    "core": 0, "conversation": 1, "observation": 2, "task_completed": 3,
    "user_preference": 4, "factual_knowledge": 5, "error_recovery": 6,
    "contextual_insight": 7, "self_improvement": 8,
}
_MODE_MAP = {
    "gestation": 0, "passive": 1, "conversational": 2, "reflective": 3,
    "focused": 4, "sleep": 5, "dreaming": 6, "deep_learning": 7,
}
_PROVENANCE_MAP = {
    "observed": 0, "user_claim": 1, "conversation": 2,
    "model_inference": 3, "external_source": 4, "experiment_result": 5,
    "derived_pattern": 6, "seed": 7, "unknown": 8,
}


def build_creation_features(
    *,
    source: str,
    initial_weight: float,
    user_present: bool,
    mode: str,
    memory_count: int,
    speaker_known: bool,
    conversation_active: bool,
    memory_type: str,
    payload_length: int,
    provenance: str,
) -> list[float]:
    """Build the 11-dim feature vector for salience model inference/training.

    Shared between get_salience_training_pairs() and engine.remember() advisory.
    """
    return [
        _SOURCE_MAP.get(source, 8) / 8.0,
        initial_weight,
        1.0 if user_present else 0.0,
        _MODE_MAP.get(mode, 1) / 7.0,
        min(memory_count, _MAX_MEMORIES) / float(_MAX_MEMORIES),
        0.0,  # similar_exists placeholder
        1.0 if speaker_known else 0.0,
        1.0 if conversation_active else 0.0,
        _TYPE_MAP.get(memory_type, 1) / 8.0,
        payload_length / 500.0,
        _PROVENANCE_MAP.get(provenance, 8) / 8.0,
    ]


memory_lifecycle_log = MemoryLifecycleLog.get_instance()
