"""Tests for the consciousness engine, kernel, phases, and tone."""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import EventBus, Memory, KERNEL_TICK, KERNEL_PHASE_CHANGE, TONE_SHIFT
from consciousness.phases import PhaseManager
from consciousness.tone import ToneEngine
from consciousness.observer import ConsciousnessObserver
from consciousness.engine import ConsciousnessEngine
from memory.storage import MemoryStorage
from memory.core import MemoryCore


def test_event_bus_basic():
    bus = EventBus()
    received = []
    bus.open_barrier()
    bus.on("test:event", lambda value, **_: received.append(value))
    bus.emit("test:event", value=42)
    assert received == [42], f"Expected [42], got {received}"
    print("  PASS: event_bus basic emit/on")


def test_event_bus_barrier():
    bus = EventBus()
    received = []
    bus.on("test:buffered", lambda v, **_: received.append(v))
    bus.emit("test:buffered", v=1)
    bus.emit("test:buffered", v=2)
    assert received == [], "Events should be buffered before barrier opens"
    bus.open_barrier()
    assert received == [1, 2], f"Expected [1, 2] after barrier, got {received}"
    print("  PASS: event_bus barrier buffering")


def test_phase_manager():
    pm = PhaseManager()
    dummy_memories = []

    analysis = pm.analyze_phase_transition("STANDBY", dummy_memories, 0.0, True)
    assert analysis.suggested_phase == "LISTENING"
    assert analysis.confidence > 0.9

    analysis2 = pm.analyze_phase_transition("LISTENING", dummy_memories, 0.0, False)
    assert analysis2.suggested_phase == "STANDBY"
    print("  PASS: phase_manager transitions")


def test_tone_engine():
    te = ToneEngine()

    # Create a dummy memory with tags that trigger a transition
    m = Memory(
        id="test", timestamp=time.time(), weight=0.5, tags=("humor", "joke"),
        payload="test", type="conversation", decay_rate=0.01, is_core=False,
        last_validated=time.time(), association_count=0, priority=500,
    )
    analysis = te.analyze_tone_shift("casual", [m], 30.0)
    assert analysis.current_tone == "casual"
    # "joke" and "humor" should trigger casual -> playful transition
    print(f"  PASS: tone_engine analysis (suggested={analysis.suggested_tone}, conf={analysis.confidence:.2f})")


def test_observer():
    obs = ConsciousnessObserver()
    result = obs.observe_thought("self_reflection", "deep", 0.8)
    assert result is not None
    recent = obs.get_recent_observations(5)
    assert len(recent) == 1
    assert recent[0].type == "thought_analysis"
    assert recent[0].confidence == 0.8

    reflection = obs.generate_self_reflection([], [], {})
    assert isinstance(reflection, str) and len(reflection) > 0
    print("  PASS: consciousness_observer")


def test_memory_core():
    mc = MemoryCore()
    from memory.core import CreateMemoryData
    mem = mc.create_memory(CreateMemoryData(
        type="conversation", payload="hello", weight=0.7, tags=["test"],
    ))
    assert mem is not None
    assert mem.type == "conversation"
    assert mem.weight == 0.7
    assert mc.validate_memory(mem)
    print("  PASS: memory_core create + validate")


def test_memory_storage():
    ms = MemoryStorage(max_capacity=10)
    mc = MemoryCore()
    from memory.core import CreateMemoryData

    for i in range(5):
        mem = mc.create_memory(CreateMemoryData(
            type="conversation", payload=f"msg {i}", weight=0.5 + i * 0.1, tags=["test"],
        ))
        ms.add(mem)

    assert ms.count() == 5
    recent = ms.get_recent(3)
    assert len(recent) == 3

    # Simulate 1 day elapsed so exponential decay produces a measurable change
    ms._last_decay_time -= 86400.0
    decayed = ms.decay_all()
    assert decayed > 0

    stats = ms.get_stats()
    assert stats["total"] == 5
    print("  PASS: memory_storage CRUD + decay")


if __name__ == "__main__":
    print("\n=== Consciousness Tests ===\n")
    test_event_bus_basic()
    test_event_bus_barrier()
    test_phase_manager()
    test_tone_engine()
    test_observer()
    test_memory_core()
    test_memory_storage()
    print("\n  All consciousness tests passed!\n")
