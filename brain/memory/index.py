"""Tag and type indexing for memory retrieval."""

from __future__ import annotations

import threading

from consciousness.events import Memory


class MemoryIndex:
    _instance: MemoryIndex | None = None

    def __init__(self) -> None:
        self._tag_index: dict[str, set[str]] = {}
        self._type_index: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> MemoryIndex:
        if cls._instance is None:
            cls._instance = MemoryIndex()
        return cls._instance

    def add_memory(self, memory: Memory) -> None:
        with self._lock:
            for tag in memory.tags:
                self._tag_index.setdefault(tag, set()).add(memory.id)
            self._type_index.setdefault(memory.type, set()).add(memory.id)

    def remove_memory(self, memory: Memory) -> None:
        with self._lock:
            for tag in memory.tags:
                s = self._tag_index.get(tag)
                if s:
                    s.discard(memory.id)
            s = self._type_index.get(memory.type)
            if s:
                s.discard(memory.id)

    def get_by_tag(self, tag: str) -> list[str]:
        with self._lock:
            return list(self._tag_index.get(tag, set()))

    def get_by_type(self, mem_type: str) -> list[str]:
        with self._lock:
            return list(self._type_index.get(mem_type, set()))

    def get_tag_frequency(self) -> dict[str, int]:
        with self._lock:
            return {tag: len(ids) for tag, ids in self._tag_index.items()}

    def rebuild(self, memories: list[Memory]) -> None:
        with self._lock:
            self._tag_index.clear()
            self._type_index.clear()
            for m in memories:
                for tag in m.tags:
                    self._tag_index.setdefault(tag, set()).add(m.id)
                self._type_index.setdefault(m.type, set()).add(m.id)

    def clear(self) -> None:
        with self._lock:
            self._tag_index.clear()
            self._type_index.clear()


memory_index = MemoryIndex.get_instance()
