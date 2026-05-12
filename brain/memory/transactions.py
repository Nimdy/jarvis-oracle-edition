"""Atomic memory transactions with integrity checks and rollback."""

from __future__ import annotations

import hashlib
import logging
import time as _time
from dataclasses import asdict, dataclass, field
from typing import Literal

from consciousness.events import (
    Memory,
    event_bus,
    MEMORY_WRITE,
    MEMORY_TRANSACTION_COMPLETE,
    MEMORY_TRANSACTION_ROLLBACK,
)
from memory.storage import memory_storage
from memory.index import memory_index

logger = logging.getLogger(__name__)

OperationType = Literal["add", "update", "remove", "associate", "reinforce"]


@dataclass
class MemoryOperation:
    type: OperationType
    memory_id: str
    memory: Memory | None = None
    updates: dict | None = None
    target_id: str | None = None
    strength: float | None = None


@dataclass
class MemorySnapshot:
    memories: list[dict]
    checksum: str

    @staticmethod
    def capture(memories: list[Memory]) -> MemorySnapshot:
        dicts = [asdict(m) for m in memories]
        parts = []
        for m in memories:
            parts.append(f"{m.id}:{m.weight}:{len(m.associations)}")
        raw = "|".join(parts)
        checksum = hashlib.md5(raw.encode()).hexdigest()
        return MemorySnapshot(memories=dicts, checksum=checksum)


@dataclass
class TransactionResult:
    success: bool
    operations_completed: int
    integrity_violations: list[str] = field(default_factory=list)
    snapshot_before: MemorySnapshot | None = None


class MemoryTransaction:
    """Execute a batch of memory operations atomically with rollback on failure."""

    def execute(self, operations: list[MemoryOperation]) -> TransactionResult:
        snapshot = MemorySnapshot.capture(memory_storage.get_all())
        completed = 0

        try:
            for op in operations:
                self._execute_op(op)
                completed += 1

                if op.type in ("add", "remove"):
                    violations = self.validate_integrity(memory_storage.get_all())
                    if violations:
                        self.rollback(snapshot)
                        event_bus.emit(
                            MEMORY_TRANSACTION_ROLLBACK,
                            operations_attempted=len(operations),
                            completed=completed,
                            violations=violations,
                            timestamp=_time.time(),
                        )
                        return TransactionResult(
                            success=False,
                            operations_completed=completed,
                            integrity_violations=violations,
                            snapshot_before=snapshot,
                        )
        except Exception as exc:
            logger.error("Transaction failed at op %d: %s", completed, exc)
            self.rollback(snapshot)
            event_bus.emit(
                MEMORY_TRANSACTION_ROLLBACK,
                operations_attempted=len(operations),
                completed=completed,
                violations=[str(exc)],
                timestamp=_time.time(),
            )
            return TransactionResult(
                success=False,
                operations_completed=completed,
                integrity_violations=[str(exc)],
                snapshot_before=snapshot,
            )

        event_bus.emit(
            MEMORY_TRANSACTION_COMPLETE,
            operations_completed=completed,
            timestamp=_time.time(),
        )
        return TransactionResult(
            success=True,
            operations_completed=completed,
            snapshot_before=snapshot,
        )

    def validate_integrity(self, memories: list[Memory]) -> list[str]:
        violations: list[str] = []
        seen_ids: set[str] = set()
        all_ids = {m.id for m in memories}

        for m in memories:
            if m.id in seen_ids:
                violations.append(f"duplicate id: {m.id}")
            seen_ids.add(m.id)

            if not m.id or m.timestamp <= 0:
                violations.append(
                    f"missing required field on {m.id or '(empty id)'}: "
                    f"id={'present' if m.id else 'empty'}, "
                    f"timestamp={m.timestamp}"
                )

            if not 0.0 <= m.weight <= 1.0:
                violations.append(
                    f"weight out of bounds for {m.id}: {m.weight}"
                )

            for assoc_id in m.associations:
                if assoc_id not in all_ids:
                    violations.append(
                        f"orphaned association on {m.id}: "
                        f"references non-existent {assoc_id}"
                    )

        return violations

    def rollback(self, snapshot: MemorySnapshot) -> None:
        memory_storage.clear()
        memory_storage.load_from_json(snapshot.memories)
        all_memories = memory_storage.get_all()
        memory_index.rebuild(all_memories)
        try:
            from memory.search import get_vector_store
            vs = get_vector_store()
            if vs is not None and vs.available:
                vs.rebuild_from_memories(all_memories)
        except Exception:
            logger.warning("Vector store rebuild failed during rollback", exc_info=True)
        logger.info(
            "Transaction rolled back to snapshot (checksum=%s, memories=%d)",
            snapshot.checksum,
            len(snapshot.memories),
        )

    def _execute_op(self, op: MemoryOperation) -> None:
        # Truth-boundary guard: synthetic perception exercise sessions must
        # not create lived-history memory through transactional writes.
        try:
            from memory.gate import memory_gate as _mg
            _synthetic = _mg.synthetic_session_active()
        except Exception:
            _synthetic = False

        if op.type == "add":
            if op.memory is None:
                raise ValueError(f"add operation requires a Memory object (id={op.memory_id})")
            if _synthetic:
                return
            memory_storage.add(op.memory)
            memory_index.add_memory(op.memory)
            try:
                from memory.search import index_memory
                index_memory(op.memory)
            except Exception:
                pass
            event_bus.emit(MEMORY_WRITE, memory=op.memory, memory_id=op.memory.id,
                           salience=op.memory.weight, tags=list(op.memory.tags))

        elif op.type == "update":
            existing = memory_storage.get(op.memory_id)
            if existing is None:
                raise ValueError(f"update target not found: {op.memory_id}")
            if _synthetic:
                return
            updates = op.updates or {}
            data = asdict(existing)
            data.update(updates)
            data["tags"] = tuple(data["tags"])
            data["associations"] = tuple(data["associations"])
            updated = Memory(**data)
            memory_storage.remove(op.memory_id)
            memory_storage.add(updated)
            memory_index.add_memory(updated)
            try:
                from memory.search import index_memory
                index_memory(updated)
            except Exception:
                pass
            event_bus.emit(MEMORY_WRITE, memory=updated, memory_id=updated.id,
                           salience=updated.weight, tags=list(updated.tags))

        elif op.type == "remove":
            if not memory_storage.remove(op.memory_id):
                raise ValueError(f"remove target not found: {op.memory_id}")

        elif op.type == "associate":
            if op.target_id is None:
                raise ValueError(f"associate operation requires target_id (source={op.memory_id})")
            if not memory_storage.associate(op.memory_id, op.target_id):
                raise ValueError(
                    f"associate failed: {op.memory_id} <-> {op.target_id}"
                )

        elif op.type == "reinforce":
            boost = op.strength if op.strength is not None else 0.1
            if not memory_storage.reinforce(op.memory_id, boost):
                raise ValueError(f"reinforce target not found: {op.memory_id}")

        else:
            raise ValueError(f"unknown operation type: {op.type}")


memory_transaction = MemoryTransaction()
