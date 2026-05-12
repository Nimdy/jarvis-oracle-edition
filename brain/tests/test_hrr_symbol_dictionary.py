"""Tests for SymbolDictionary (brain/library/vsa/symbols.py)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from library.vsa.hrr import HRRConfig, similarity
from library.vsa.symbols import SymbolDictionary


def _dict(dim: int = 1024, seed: int = 0) -> SymbolDictionary:
    return SymbolDictionary(HRRConfig(dim=dim, seed=seed))


def test_same_key_returns_same_vector():
    d = _dict()
    v1 = d.get("role", "subject")
    v2 = d.get("role", "subject")
    np.testing.assert_array_equal(v1, v2)
    assert v1 is v2, "cached hit should return the same object"


def test_distinct_keys_near_orthogonal():
    d = _dict()
    names = ("subject", "object", "is_in", "has_state", "emotion", "topic")
    vecs = [d.role(n) for n in names]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = similarity(vecs[i], vecs[j])
            assert abs(sim) < 0.15, f"{names[i]} vs {names[j]} too similar: {sim}"


def test_namespaces_isolated():
    """Same name under different namespaces should produce different vectors."""
    d = _dict()
    role_user = d.role("user")
    entity_user = d.entity("user")
    assert not np.allclose(role_user, entity_user)
    assert abs(similarity(role_user, entity_user)) < 0.2


def test_seed_isolation():
    """Different dictionary seeds produce disjoint vocabularies."""
    d0 = _dict(seed=0)
    d1 = _dict(seed=1)
    v0 = d0.entity("topic:health")
    v1 = d1.entity("topic:health")
    assert not np.allclose(v0, v1)


def test_len_and_known():
    d = _dict()
    assert len(d) == 0
    d.role("a")
    d.role("b")
    d.entity("c")
    assert len(d) == 3
    assert d.known("role") == ("a", "b")
    assert d.known("entity") == ("c",)


def test_thread_safety_no_duplicates_under_contention():
    import threading

    d = _dict()
    errors: list[str] = []

    def worker():
        for i in range(200):
            d.role(f"name_{i % 10}")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Only 10 distinct names under role namespace regardless of contention.
    assert len(d) == 10, f"expected 10 role symbols, got {len(d)}; errors={errors}"
