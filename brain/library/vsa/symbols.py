"""Deterministic symbol dictionary for HRR.

Symbols are organized into namespaces (``"role"``, ``"relation"``,
``"entity"``, ``"bucket"``, etc.) so that a single HRR substrate can host
multiple disjoint vocabularies without collision. Within a namespace, vectors
are keyed by ``(namespace, name, cfg.seed, cfg.dim)`` and memoized.

This module is pure and deterministic. Retrieval of an already-seeded symbol
returns the same object reference every time, so callers may use ``is`` for
cheap equality checks if they so choose.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import numpy as np

from library.vsa.hrr import HRRConfig, make_symbol


class SymbolDictionary:
    """Cached, namespaced deterministic HRR symbol store.

    Thread-safe. Each ``(namespace, name)`` pair maps to exactly one vector
    for the lifetime of the dictionary.
    """

    def __init__(self, cfg: HRRConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._vectors: Dict[Tuple[str, str], np.ndarray] = {}

    @property
    def config(self) -> HRRConfig:
        return self._cfg

    def get(self, namespace: str, name: str) -> np.ndarray:
        """Return the vector for ``(namespace, name)``, creating it if absent."""
        key = (namespace, name)
        cached = self._vectors.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._vectors.get(key)
            if cached is not None:
                return cached
            seed_name = f"{namespace}:{name}"
            vec = make_symbol(seed_name, self._cfg)
            self._vectors[key] = vec
            return vec

    def role(self, name: str) -> np.ndarray:
        return self.get("role", name)

    def relation(self, name: str) -> np.ndarray:
        return self.get("relation", name)

    def entity(self, name: str) -> np.ndarray:
        return self.get("entity", name)

    def bucket(self, name: str) -> np.ndarray:
        return self.get("bucket", name)

    def known(self, namespace: str) -> Tuple[str, ...]:
        """Return names seeded under ``namespace`` (sorted, for determinism)."""
        with self._lock:
            return tuple(sorted(n for ns, n in self._vectors if ns == namespace))

    def __len__(self) -> int:
        return len(self._vectors)
