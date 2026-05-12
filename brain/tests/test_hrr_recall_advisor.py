"""Tests for the HRR recall advisor (brain/memory/hrr_recall_advisor.py).

Key invariants enforced here:

* The advisor is a pure **in-memory** observer: no files created under HOME,
  no files created under ``~/.jarvis/cache/hrr_memory_vectors.jsonl`` (the
  durable cache was explicitly deferred for this sprint).
* LRU cache caps at 2000 entries.
* Observation ring buffer caps at 500 entries.
* Disabled runtime is a cheap no-op.
* No forbidden writer imports.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from library.vsa.runtime_config import HRRRuntimeConfig
from memory.hrr_recall_advisor import HRRRecallAdvisor


# ---------------------------------------------------------------------------
# Test fixtures: minimal cue / candidate stand-ins.
# ---------------------------------------------------------------------------


@dataclass
class _FakeCue:
    text: str = "remember when we talked about the weather yesterday"
    cue_class: str = "ambient_text"
    cue_strength: float = 0.8


@dataclass
class _FakeMemory:
    content: str = ""


@dataclass
class _FakeCandidate:
    memory_id: str = "mem_0001"
    memory: Any = field(default_factory=_FakeMemory)
    resonance: float = 0.75
    semantic_score: float = 0.8
    tag_score: float = 0.5
    provenance_weight: float = 1.0
    mode_fit: float = 1.0


def _cue() -> _FakeCue:
    return _FakeCue()


def _mk_chain(n: int) -> list:
    chain = []
    for i in range(n):
        mem = _FakeMemory(content=f"memory number {i} about weather and sky")
        chain.append(_FakeCandidate(memory_id=f"mem_{i:04d}", memory=mem))
    return chain


# ---------------------------------------------------------------------------
# Disabled-runtime → pure no-op
# ---------------------------------------------------------------------------


def test_advisor_is_noop_when_disabled():
    cfg = HRRRuntimeConfig(enabled=False, dim=128)
    a = HRRRecallAdvisor(cfg)
    chain = _mk_chain(3)
    out = a.observe(_cue(), chain[0], chain, governance_action="consider")
    assert out is None
    s = a.status()
    assert s["enabled"] is False
    assert s["samples_total"] == 0
    assert s["samples_retained"] == 0
    assert s["cache_size"] == 0
    assert s["ring_capacity"] == 500
    assert s["lru_capacity"] == 2000


# ---------------------------------------------------------------------------
# Enabled runtime: records metrics, caps LRU + ring
# ---------------------------------------------------------------------------


def test_advisor_records_metrics():
    cfg = HRRRuntimeConfig(enabled=True, dim=256)
    a = HRRRecallAdvisor(cfg)
    chain = _mk_chain(5)
    out = a.observe(_cue(), chain[0], chain, governance_action="consider")
    assert out is not None
    assert out["seed_id"] == "mem_0000"
    assert out["chain_size"] == 5
    assert out["cache_size"] >= 5
    assert out["side_effects"] == 0
    s = a.status()
    assert s["samples_total"] == 1
    assert s["samples_retained"] == 1
    assert s["cache_size"] >= 5


def test_advisor_lru_cache_caps_at_2000():
    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    a = HRRRecallAdvisor(cfg)
    # Shove 2500 distinct memories through a sequence of observations.
    for batch in range(25):
        chain = []
        for i in range(100):
            mid = f"mem_b{batch:03d}_{i:03d}"
            chain.append(_FakeCandidate(memory_id=mid, memory=_FakeMemory(content=f"m {mid}")))
        a.observe(_cue(), chain[0], chain, governance_action="consider")
    assert len(a._cache) <= HRRRecallAdvisor.LRU_CAPACITY
    assert a.status()["cache_size"] <= HRRRecallAdvisor.LRU_CAPACITY


def test_advisor_ring_buffer_caps_at_500():
    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    a = HRRRecallAdvisor(cfg)
    chain = _mk_chain(3)
    total = HRRRecallAdvisor.RING_CAPACITY + 100
    for i in range(total):
        # Vary the seed id so observations are distinct.
        chain[0] = _FakeCandidate(memory_id=f"mem_seed_{i:05d}",
                                   memory=_FakeMemory(content=f"seed {i}"))
        a.observe(_cue(), chain[0], chain, governance_action="consider")
    s = a.status()
    assert s["samples_total"] == total
    assert s["samples_retained"] == HRRRecallAdvisor.RING_CAPACITY


def test_advisor_none_inputs_are_safe():
    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    a = HRRRecallAdvisor(cfg)
    assert a.observe(None, _FakeCandidate(), [_FakeCandidate()]) is None
    assert a.observe(_cue(), None, [_FakeCandidate()]) is None
    assert a.observe(_cue(), _FakeCandidate(), None) is None
    assert a.status()["samples_total"] == 0


def test_advisor_tolerates_exception_in_encoding():
    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    a = HRRRecallAdvisor(cfg)

    class _BoomCue:
        @property
        def text(self):
            raise RuntimeError("boom")

        cue_class = "ambient_text"

    # Must not raise.
    out = a.observe(_BoomCue(), _FakeCandidate(), [_FakeCandidate()])
    assert out is None


# ---------------------------------------------------------------------------
# Filesystem truth boundary: NO writes anywhere during a burst of observations
# ---------------------------------------------------------------------------


def test_advisor_makes_zero_filesystem_writes_under_home(tmp_path, monkeypatch):
    """Under a throwaway HOME, running 200 observations must create 0 files.

    This is the structural enforcement of "in-memory only" for this sprint.
    """
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    cfg = HRRRuntimeConfig(enabled=True, dim=128)
    a = HRRRecallAdvisor(cfg)
    chain = _mk_chain(5)
    for i in range(200):
        chain[0] = _FakeCandidate(memory_id=f"mem_{i:05d}",
                                   memory=_FakeMemory(content=f"m {i}"))
        a.observe(_cue(), chain[0], chain, governance_action="consider")

    survivors = [p for p in fake_home.rglob("*") if p.is_file()]
    assert survivors == [], f"advisor wrote unexpected files under HOME: {survivors}"

    # And the explicitly-forbidden durable cache path MUST NOT exist.
    forbidden = fake_home / ".jarvis" / "cache" / "hrr_memory_vectors.jsonl"
    assert not forbidden.exists()


# ---------------------------------------------------------------------------
# FractalRecallEngine integration: advisor activates behind env flag only
# ---------------------------------------------------------------------------


def test_fractal_engine_wires_advisor_when_flag_on(monkeypatch):
    from memory.fractal_recall import FractalRecallEngine

    class _DummyStorage:
        def get_all(self): return []

    class _DummyVectorStore:
        available = False

    monkeypatch.setenv("ENABLE_HRR_SHADOW", "1")
    engine = FractalRecallEngine(memory_storage=_DummyStorage(), vector_store=_DummyVectorStore())
    assert engine._hrr_advisor is not None


def test_fractal_engine_does_not_wire_advisor_when_flag_off(monkeypatch, tmp_path):
    from memory.fractal_recall import FractalRecallEngine

    class _DummyStorage:
        def get_all(self): return []

    class _DummyVectorStore:
        available = False

    # Clear both layers (env + runtime_flags.json) so the test asserts the
    # actual safe-default behavior rather than the operator's persisted state.
    monkeypatch.delenv("ENABLE_HRR_SHADOW", raising=False)
    monkeypatch.delenv("JARVIS_RUNTIME_FLAGS", raising=False)
    monkeypatch.setenv("JARVIS_RUNTIME_FLAGS", str(tmp_path / "_absent_runtime_flags.json"))
    engine = FractalRecallEngine(memory_storage=_DummyStorage(), vector_store=_DummyVectorStore())
    assert engine._hrr_advisor is None


# ---------------------------------------------------------------------------
# Import-graph guard: no forbidden writer imports
# ---------------------------------------------------------------------------


def test_advisor_source_has_no_forbidden_imports():
    import memory.hrr_recall_advisor as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from policy.state_encoder",
        "from policy.policy_nn",
        "from epistemic.belief_graph.bridge",
        "from memory.persistence",
        "from memory.storage",
        "from memory.canonical",
        "from autonomy",
        "from identity.kernel",
        # And no filesystem primitives that could sneak durability in.
        "import json",    # we have no json need in this module
        "open(",
    )
    for token in forbidden:
        assert token not in src, f"advisor must not contain {token!r}"
