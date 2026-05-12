"""Memory creation and validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from nanoid import generate as nanoid

from consciousness.events import Memory, MemoryType


@dataclass(frozen=True)
class MemoryTypeConfig:
    decay_rate: float
    can_prune: bool
    min_weight: float
    priority: int


MEMORY_TYPE_CONFIGS: dict[MemoryType, MemoryTypeConfig] = {
    "core":               MemoryTypeConfig(decay_rate=0.001, can_prune=False, min_weight=0.1, priority=1000),
    "conversation":       MemoryTypeConfig(decay_rate=0.03,  can_prune=True,  min_weight=0,   priority=500),
    "observation":        MemoryTypeConfig(decay_rate=0.06,  can_prune=True,  min_weight=0,   priority=150),
    "task_completed":     MemoryTypeConfig(decay_rate=0.04,  can_prune=True,  min_weight=0,   priority=400),
    "user_preference":    MemoryTypeConfig(decay_rate=0.005, can_prune=False, min_weight=0.1, priority=900),
    "factual_knowledge":  MemoryTypeConfig(decay_rate=0.01,  can_prune=True,  min_weight=0,   priority=700),
    "error_recovery":     MemoryTypeConfig(decay_rate=0.05,  can_prune=True,  min_weight=0,   priority=300),
    "contextual_insight": MemoryTypeConfig(decay_rate=0.02,  can_prune=True,  min_weight=0,   priority=600),
    "self_improvement":   MemoryTypeConfig(decay_rate=0.015, can_prune=True,  min_weight=0,   priority=650),
}

VALID_MEMORY_TYPES = set(MEMORY_TYPE_CONFIGS.keys())


@dataclass
class CreateMemoryData:
    type: MemoryType
    payload: Any
    weight: float
    tags: Sequence[str]
    decay_rate: float | None = None
    provenance: str = "unknown"
    identity_owner: str = ""
    identity_owner_type: str = ""
    identity_subject: str = ""
    identity_subject_type: str = ""
    identity_scope_key: str = ""
    identity_confidence: float = 0.0
    identity_needs_resolution: bool = False


class MemoryCore:
    _instance: MemoryCore | None = None

    @classmethod
    def get_instance(cls) -> MemoryCore:
        if cls._instance is None:
            cls._instance = MemoryCore()
        return cls._instance

    def create_memory(self, data: CreateMemoryData) -> Memory | None:
        if not self._validate_create_data(data):
            return None

        cfg = MEMORY_TYPE_CONFIGS[data.type]
        now = time.time()

        weight = max(0.0, min(1.0, data.weight))
        if data.identity_needs_resolution and data.identity_confidence < 0.45:
            weight = min(weight, 0.5)

        return Memory(
            id=f"mem_{nanoid(size=21)}",
            timestamp=now,
            weight=weight,
            tags=tuple(data.tags),
            payload=data.payload,
            type=data.type,
            associations=(),
            decay_rate=data.decay_rate if data.decay_rate is not None else cfg.decay_rate,
            is_core=data.type in ("core", "user_preference"),
            last_validated=now,
            association_count=0,
            priority=cfg.priority,
            provenance=data.provenance or "unknown",
            identity_owner=data.identity_owner,
            identity_owner_type=data.identity_owner_type,
            identity_subject=data.identity_subject,
            identity_subject_type=data.identity_subject_type,
            identity_scope_key=data.identity_scope_key,
            identity_confidence=data.identity_confidence,
            identity_needs_resolution=data.identity_needs_resolution,
        )

    def create_core_memory(self, payload: Any, tags: Sequence[str], provenance: str = "seed") -> Memory | None:
        return self.create_memory(CreateMemoryData(
            type="core",
            payload=payload,
            tags=[*tags, "core"],
            weight=0.9,
            decay_rate=MEMORY_TYPE_CONFIGS["core"].decay_rate,
            provenance=provenance,
        ))

    def validate_memory(self, obj: Any) -> bool:
        if not isinstance(obj, Memory):
            return False
        return (
            isinstance(obj.id, str)
            and isinstance(obj.timestamp, (int, float))
            and 0 <= obj.weight <= 1
            and isinstance(obj.tags, (list, tuple))
            and obj.type in VALID_MEMORY_TYPES
        )

    def get_type_config(self, mem_type: MemoryType) -> MemoryTypeConfig:
        return MEMORY_TYPE_CONFIGS[mem_type]

    @staticmethod
    def _validate_create_data(data: CreateMemoryData) -> bool:
        return (
            data is not None
            and data.type in VALID_MEMORY_TYPES
            and data.payload is not None
            and isinstance(data.weight, (int, float))
            and isinstance(data.tags, (list, tuple))
        )


memory_core = MemoryCore.get_instance()


def canonical_remember(data: CreateMemoryData) -> Memory | None:
    """Canonical memory write path — routes through engine.remember() for
    quarantine soft-gate, salience advisory, and creation context.

    Falls back to direct write during early boot or if engine.remember() fails.
    """
    try:
        from consciousness.consciousness_system import _active_consciousness
        if _active_consciousness and hasattr(_active_consciousness, '_engine_ref'):
            engine = _active_consciousness._engine_ref
            from consciousness.engine import ConsciousnessEngine
            if isinstance(engine, ConsciousnessEngine) and getattr(engine, '_restore_complete', False):
                return engine.remember(data)
    except Exception:
        pass

    return _direct_memory_write(data)


def _direct_memory_write(data: CreateMemoryData) -> Memory | None:
    """Fallback write path when engine is unavailable (early boot, tests)."""
    from memory.storage import memory_storage
    from memory.index import memory_index
    from consciousness.events import event_bus, MEMORY_WRITE

    try:
        from memory.gate import memory_gate as _mg
        if _mg.synthetic_session_active():
            # Truth-boundary guard: synthetic perception exercise must not
            # create lived-history memory through the fallback write path.
            return None
    except Exception:
        pass

    if not data.identity_owner:
        _stamp_identity_fallback(data)

    memory = memory_core.create_memory(data)
    if not memory:
        return None
    memory_storage.add(memory)
    memory_index.add_memory(memory)
    try:
        from memory.search import index_memory
        index_memory(memory)
    except Exception:
        pass
    event_bus.emit(MEMORY_WRITE, memory=memory,
                   memory_id=getattr(memory, "id", ""),
                   salience=getattr(memory, "weight", 0.5),
                   tags=list(getattr(memory, "tags", ())))
    return memory


def _stamp_identity_fallback(data: CreateMemoryData) -> None:
    """Best-effort identity stamping for the fallback write path."""
    try:
        from identity.resolver import identity_resolver
        ctx = identity_resolver.resolve_for_memory(
            provenance=data.provenance or "unknown",
            actor="system" if data.type in ("core", "self_improvement", "error_recovery") else "",
            speaker="",
        )
        scope = identity_resolver.build_scope(ctx, data.payload, data.type)
        data.identity_owner = scope.owner_id
        data.identity_owner_type = scope.owner_type
        data.identity_subject = scope.subject_id
        data.identity_subject_type = scope.subject_type
        data.identity_scope_key = scope.scope_key
        data.identity_confidence = scope.confidence
        data.identity_needs_resolution = scope.needs_resolution
    except Exception:
        pass
