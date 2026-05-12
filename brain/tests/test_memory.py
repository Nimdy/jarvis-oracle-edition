"""Tests for memory search and persistence."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.core import MemoryCore, CreateMemoryData
from memory.storage import MemoryStorage
from memory.index import MemoryIndex
from memory import search as mem_search
from memory.persistence import MemoryPersistence


def test_memory_search():
    ms = MemoryStorage(max_capacity=100)
    mc = MemoryCore()
    mi = MemoryIndex()

    # Patch the singletons for testing
    orig_storage = mem_search.memory_storage
    mem_search.memory_storage = ms

    m1 = mc.create_memory(CreateMemoryData(
        type="conversation", payload="We talked about Python programming", weight=0.8, tags=["technical", "python"],
    ))
    m2 = mc.create_memory(CreateMemoryData(
        type="observation", payload="User was drinking coffee", weight=0.3, tags=["observation"],
    ))
    m3 = mc.create_memory(CreateMemoryData(
        type="conversation", payload="Discussion about Python vs JavaScript", weight=0.6, tags=["technical"],
    ))

    ms.add(m1)
    ms.add(m2)
    ms.add(m3)
    mi.rebuild(ms.get_all())

    results = mem_search.search("python")
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].weight >= results[1].weight, "Results should be sorted by weight"

    results2 = mem_search.search("coffee")
    assert len(results2) == 1

    results3 = mem_search.search("nonexistent_xyz")
    assert len(results3) == 0

    mem_search.memory_storage = orig_storage
    print("  PASS: memory search")


def test_memory_persistence():
    ms = MemoryStorage(max_capacity=100)
    mc = MemoryCore()

    m1 = mc.create_memory(CreateMemoryData(
        type="core", payload="Test core memory", weight=0.9, tags=["test", "core"],
    ))
    m2 = mc.create_memory(CreateMemoryData(
        type="conversation", payload="Test conversation", weight=0.7, tags=["test"],
    ))
    ms.add(m1)
    ms.add(m2)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        # Save
        p = MemoryPersistence(path=tmp_path)
        p._interval_s = 999  # don't auto-save
        # Patch persistence to use our storage
        import memory.persistence as mp
        orig = mp.memory_storage
        mp.memory_storage = ms
        assert p.save(), "Save should succeed"

        # Verify file
        with open(tmp_path) as f:
            data = json.load(f)
        assert len(data) == 2, f"Expected 2 memories in file, got {len(data)}"

        # Load into fresh storage
        ms2 = MemoryStorage(max_capacity=100)
        mp.memory_storage = ms2
        loaded = p.load()
        assert loaded == 2, f"Expected to load 2 memories, got {loaded}"
        assert ms2.count() == 2

        mp.memory_storage = orig
    finally:
        os.unlink(tmp_path)

    print("  PASS: memory persistence save/load")


def test_memory_index():
    mi = MemoryIndex()
    mc = MemoryCore()

    m1 = mc.create_memory(CreateMemoryData(
        type="conversation", payload="test1", weight=0.5, tags=["alpha", "beta"],
    ))
    m2 = mc.create_memory(CreateMemoryData(
        type="observation", payload="test2", weight=0.3, tags=["beta", "gamma"],
    ))

    mi.add_memory(m1)
    mi.add_memory(m2)

    assert m1.id in mi.get_by_tag("alpha")
    assert len(mi.get_by_tag("beta")) == 2
    assert len(mi.get_by_type("conversation")) == 1

    freq = mi.get_tag_frequency()
    assert freq["beta"] == 2
    assert freq["alpha"] == 1

    print("  PASS: memory index")


if __name__ == "__main__":
    print("\n=== Memory Tests ===\n")
    test_memory_search()
    test_memory_persistence()
    test_memory_index()
    print("\n  All memory tests passed!\n")
