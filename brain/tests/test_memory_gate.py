"""Tests for CueGate (memory/gate.py) — single authority for memory access policy."""

import threading
import time

import pytest

from memory.gate import MemoryGate


@pytest.fixture
def gate():
    return MemoryGate()


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_closed(self, gate: MemoryGate):
        assert not gate.is_open()

    def test_default_can_read(self, gate: MemoryGate):
        assert gate.can_read()

    def test_default_observation_writes_allowed(self, gate: MemoryGate):
        assert gate.can_observation_write()

    def test_default_consolidation_inactive(self, gate: MemoryGate):
        assert not gate.can_consolidation_write()

    def test_default_stats(self, gate: MemoryGate):
        stats = gate.get_stats()
        assert stats["is_open"] is False
        assert stats["depth"] == 0
        assert stats["observation_writes_allowed"] is True
        assert stats["consolidation_active"] is False
        assert stats["current_mode"] == "passive"


# ---------------------------------------------------------------------------
# Read sessions (RAII guard)
# ---------------------------------------------------------------------------

class TestReadSessions:
    def test_session_opens_and_closes(self, gate: MemoryGate):
        with gate.session("test_read", actor="test"):
            assert gate.is_open()
            assert gate.get_stats()["depth"] == 1
        assert not gate.is_open()

    def test_nested_sessions(self, gate: MemoryGate):
        with gate.session("outer"):
            assert gate.get_stats()["depth"] == 1
            with gate.session("inner"):
                assert gate.get_stats()["depth"] == 2
            assert gate.get_stats()["depth"] == 1
        assert gate.get_stats()["depth"] == 0

    def test_session_closes_on_exception(self, gate: MemoryGate):
        with pytest.raises(ValueError):
            with gate.session("failing"):
                assert gate.is_open()
                raise ValueError("boom")
        assert not gate.is_open()

    def test_total_opens_tracked(self, gate: MemoryGate):
        for _ in range(5):
            with gate.session("iter"):
                pass
        assert gate.get_stats()["total_opens"] == 5

    def test_manual_open_close(self, gate: MemoryGate):
        gate.open("manual", actor="test")
        assert gate.is_open()
        gate.close("manual", actor="test")
        assert not gate.is_open()

    def test_close_at_zero_stays_zero(self, gate: MemoryGate):
        gate.close("spurious")
        assert gate.get_stats()["depth"] == 0


# ---------------------------------------------------------------------------
# Mode transitions — observation write policy
# ---------------------------------------------------------------------------

class TestModePolicy:
    @pytest.mark.parametrize("mode", ["conversational", "focused", "passive", "gestation"])
    def test_waking_modes_allow_observation_writes(self, gate: MemoryGate, mode: str):
        gate.set_mode(mode)
        assert gate.can_observation_write()

    @pytest.mark.parametrize("mode", ["dreaming", "sleep", "reflective", "deep_learning"])
    def test_blocking_modes_block_observation_writes(self, gate: MemoryGate, mode: str):
        gate.set_mode(mode)
        assert not gate.can_observation_write()

    def test_mode_transition_records_transition(self, gate: MemoryGate):
        gate.set_mode("dreaming")
        stats = gate.get_stats()
        transitions = stats["recent_transitions"]
        assert len(transitions) >= 1
        assert transitions[-1]["action"] == "obs_writes_blocked"

    def test_mode_recovery(self, gate: MemoryGate):
        gate.set_mode("dreaming")
        assert not gate.can_observation_write()
        gate.set_mode("conversational")
        assert gate.can_observation_write()
        transitions = gate.get_stats()["recent_transitions"]
        assert transitions[-1]["action"] == "obs_writes_enabled"

    def test_same_policy_mode_no_extra_transition(self, gate: MemoryGate):
        gate.set_mode("conversational")
        gate.set_mode("focused")
        transitions = gate.get_stats()["recent_transitions"]
        obs_transitions = [t for t in transitions if t["action"].startswith("obs_writes_")]
        assert len(obs_transitions) == 0

    def test_stats_reflect_mode(self, gate: MemoryGate):
        gate.set_mode("dreaming")
        stats = gate.get_stats()
        assert stats["current_mode"] == "dreaming"
        assert stats["observation_writes_allowed"] is False


# ---------------------------------------------------------------------------
# Consolidation lifecycle
# ---------------------------------------------------------------------------

class TestConsolidation:
    def test_begin_end(self, gate: MemoryGate):
        assert not gate.can_consolidation_write()
        gate.begin_consolidation("test")
        assert gate.can_consolidation_write()
        gate.end_consolidation("test")
        assert not gate.can_consolidation_write()

    def test_consolidation_window_raii(self, gate: MemoryGate):
        with gate.consolidation_window("test_window"):
            assert gate.can_consolidation_write()
        assert not gate.can_consolidation_write()

    def test_consolidation_window_on_exception(self, gate: MemoryGate):
        with pytest.raises(RuntimeError):
            with gate.consolidation_window("failing"):
                assert gate.can_consolidation_write()
                raise RuntimeError("dream crash")
        assert not gate.can_consolidation_write()

    def test_consolidation_records_transitions(self, gate: MemoryGate):
        with gate.consolidation_window("cycle_1"):
            pass
        transitions = gate.get_stats()["recent_transitions"]
        actions = [t["action"] for t in transitions]
        assert "consolidation_begin" in actions
        assert "consolidation_end" in actions

    def test_consolidation_independent_of_mode(self, gate: MemoryGate):
        gate.set_mode("dreaming")
        assert not gate.can_observation_write()
        with gate.consolidation_window("dream"):
            assert gate.can_consolidation_write()
            assert not gate.can_observation_write()


# ---------------------------------------------------------------------------
# Synthetic exercise session — truth boundary guard
# ---------------------------------------------------------------------------

class TestSyntheticSession:
    def test_default_synthetic_inactive(self, gate: MemoryGate):
        assert gate.synthetic_session_active() is False
        stats = gate.get_stats()
        assert stats["synthetic_active"] is False
        assert stats["synthetic_sources"] == []

    def test_begin_blocks_observation_writes(self, gate: MemoryGate):
        gate.set_mode("conversational")
        assert gate.can_observation_write()
        gate.begin_synthetic_session("synthetic-exercise")
        assert gate.synthetic_session_active()
        assert not gate.can_observation_write()

    def test_begin_blocks_consolidation_writes(self, gate: MemoryGate):
        gate.begin_consolidation("dream_cycle")
        assert gate.can_consolidation_write()
        gate.begin_synthetic_session("synthetic-exercise")
        assert not gate.can_consolidation_write()
        gate.end_synthetic_session("synthetic-exercise")
        assert gate.can_consolidation_write()

    def test_end_releases_gate(self, gate: MemoryGate):
        gate.set_mode("conversational")
        gate.begin_synthetic_session("synthetic-exercise")
        assert not gate.can_observation_write()
        gate.end_synthetic_session("synthetic-exercise")
        assert gate.can_observation_write()
        assert not gate.synthetic_session_active()

    def test_multi_source_requires_all_to_end(self, gate: MemoryGate):
        gate.set_mode("conversational")
        gate.begin_synthetic_session("source_a")
        gate.begin_synthetic_session("source_b")
        assert gate.synthetic_session_active()
        gate.end_synthetic_session("source_a")
        assert gate.synthetic_session_active()
        assert not gate.can_observation_write()
        gate.end_synthetic_session("source_b")
        assert not gate.synthetic_session_active()
        assert gate.can_observation_write()

    def test_synthetic_overrides_waking_mode(self, gate: MemoryGate):
        gate.set_mode("conversational")
        assert gate.can_observation_write()
        gate.begin_synthetic_session("synthetic-exercise")
        assert not gate.can_observation_write()

    def test_begin_and_end_record_transitions(self, gate: MemoryGate):
        gate.begin_synthetic_session("src1")
        gate.end_synthetic_session("src1")
        actions = [t["action"] for t in gate.get_stats()["recent_transitions"]]
        assert "synthetic_begin" in actions
        assert "synthetic_end" in actions

    def test_stats_expose_sources(self, gate: MemoryGate):
        gate.begin_synthetic_session("src_x")
        gate.begin_synthetic_session("src_y")
        stats = gate.get_stats()
        assert stats["synthetic_active"] is True
        assert stats["synthetic_sources"] == ["src_x", "src_y"]

    def test_end_unknown_source_is_noop(self, gate: MemoryGate):
        gate.end_synthetic_session("never_started")
        assert not gate.synthetic_session_active()

    def test_re_entrant_begin_same_source(self, gate: MemoryGate):
        gate.begin_synthetic_session("src")
        gate.begin_synthetic_session("src")
        gate.end_synthetic_session("src")
        assert not gate.synthetic_session_active()


# ---------------------------------------------------------------------------
# Transition history
# ---------------------------------------------------------------------------

class TestTransitionHistory:
    def test_history_bounded(self, gate: MemoryGate):
        for i in range(300):
            gate.open(f"flood_{i}")
            gate.close(f"flood_{i}")
        stats = gate.get_stats()
        all_transitions = gate._transitions
        assert len(all_transitions) <= 200

    def test_transition_fields(self, gate: MemoryGate):
        gate.open("test_reason", actor="test_actor")
        t = gate.get_stats()["recent_transitions"][-1]
        assert t["action"] == "open"
        assert t["reason"] == "test_reason"
        assert t["actor"] == "test_actor"
        assert "ts" in t
        assert "obs_writes" in t
        assert "consolidation" in t


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_sessions(self, gate: MemoryGate):
        errors = []

        def worker(n):
            try:
                for _ in range(50):
                    with gate.session(f"worker_{n}"):
                        time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert gate.get_stats()["depth"] == 0

    def test_concurrent_mode_changes(self, gate: MemoryGate):
        modes = ["conversational", "dreaming", "sleep", "focused", "reflective"]
        errors = []

        def flipper(n):
            try:
                for i in range(20):
                    gate.set_mode(modes[i % len(modes)])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=flipper, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Integration: observer-like query pattern
# ---------------------------------------------------------------------------

class TestObserverPattern:
    """Simulates how the observer queries CueGate instead of stance profiles."""

    def test_waking_observation_write_allowed(self, gate: MemoryGate):
        gate.set_mode("conversational")
        assert gate.can_observation_write()

    def test_dreaming_blocks_observation_but_allows_consolidation(self, gate: MemoryGate):
        gate.set_mode("dreaming")
        assert not gate.can_observation_write()
        with gate.consolidation_window("dream"):
            assert gate.can_consolidation_write()
            assert not gate.can_observation_write()

    def test_artifact_validation_blocked_during_consolidation(self, gate: MemoryGate):
        """Artifacts should not be promoted during an active consolidation."""
        assert not gate.can_consolidation_write()
        gate.begin_consolidation("dream")
        assert gate.can_consolidation_write()
        gate.end_consolidation("dream")
        assert not gate.can_consolidation_write()
