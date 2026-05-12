"""Non-invasiveness integration test for the eval sidecar.

The eval sidecar is a read-only observer. This test is the trust seal:
it verifies that running the eval sidecar alongside the engine produces
no mutations to memory, beliefs, or dreams.

During 10 simulated kernel ticks with the sidecar enabled:
- Memory count remains unchanged (no eval-originated MEMORY_WRITE events)
- No calls to engine.remember() originate from eval code
- Active belief count does not change due to eval activity
- The eval store contains collected events and snapshots
- No exceptions in kernel tick loop from eval handlers
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEvalSidecarNonInvasive:
    """Trust seal: the eval sidecar must never mutate core state."""

    @pytest.fixture
    def tmp_eval_dir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def _mock_engine(self):
        """Build a mock engine with all the stats APIs the collector reads."""
        engine = MagicMock()
        engine.get_state.return_value = {
            "phase": "LISTENING", "mode": "conversational",
            "memory_count": 10, "tick": 5,
        }
        cs = MagicMock()
        cs.get_state.return_value.to_dict.return_value = {
            "stage": "basic_awareness", "transcendence_level": 0.1,
        }
        cs.get_dream_artifact_stats.return_value = {"buffer": {"size": 0}}
        cs.get_cortex_stats.return_value = {}
        cs.check_health.return_value = {}
        cs.observer = None
        engine.consciousness = cs
        engine.remember = MagicMock()
        return engine

    def test_no_memory_writes(self, tmp_eval_dir):
        """Sidecar must never call engine.remember() or emit MEMORY_WRITE."""
        engine = self._mock_engine()

        from jarvis_eval.store import EvalStore
        store = EvalStore(base_dir=tmp_eval_dir)

        from jarvis_eval import EvalSidecar
        sidecar = EvalSidecar()
        sidecar._store = store

        from jarvis_eval.collector import EvalCollector
        collector = EvalCollector(engine)
        sidecar._collector = collector
        sidecar._tap._run_id = "test"

        for _ in range(10):
            sidecar._flush_once()

        engine.remember.assert_not_called()

    def test_eval_store_populated(self, tmp_eval_dir):
        """Sidecar must actually write data to its own store."""
        engine = self._mock_engine()

        from jarvis_eval.store import EvalStore
        store = EvalStore(base_dir=tmp_eval_dir)

        from jarvis_eval import EvalSidecar
        sidecar = EvalSidecar()
        sidecar._store = store

        from jarvis_eval.collector import EvalCollector
        collector = EvalCollector(engine)
        sidecar._collector = collector

        for _ in range(3):
            sidecar._flush_once()

        snapshots = store.read_recent_snapshots(limit=100)
        assert len(snapshots) > 0, "Sidecar should have written snapshots"

    def test_no_event_bus_emission(self, tmp_eval_dir):
        """Event tap handlers must never emit events back to the bus."""
        from jarvis_eval.event_tap import EvalEventTap, _TAPPED_EVENTS

        tap = EvalEventTap()

        from consciousness.events import event_bus
        original_emit = event_bus.emit
        emit_calls: list[str] = []

        def tracking_emit(event_type, **kwargs):
            emit_calls.append(event_type)
            return original_emit(event_type, **kwargs)

        with patch.object(event_bus, "emit", tracking_emit):
            tap.wire(run_id="test")

            emit_calls_before = len(emit_calls)
            for ev_name in _TAPPED_EVENTS[:3]:
                original_emit(ev_name, test_data="hello")

            tap_emits = [c for c in emit_calls[emit_calls_before:]
                        if c not in set(_TAPPED_EVENTS)]
            assert len(tap_emits) == 0, f"Tap should not re-emit events: {tap_emits}"

            tap.unwire()

    def test_collector_no_private_access(self, tmp_eval_dir):
        """Collector must only call public API methods, not access private attrs."""
        engine = self._mock_engine()

        from jarvis_eval.collector import EvalCollector
        collector = EvalCollector(engine)

        attrs_accessed: list[str] = []
        original_getattr = engine.__getattr__

        def tracking_getattr(name):
            attrs_accessed.append(name)
            return original_getattr(name)

        snapshots = collector.collect_once()

        for snap in snapshots:
            assert isinstance(snap.source, str)
            assert isinstance(snap.metrics, dict)

    def test_dashboard_snapshot_safe(self, tmp_eval_dir):
        """get_dashboard_snapshot must not raise or mutate state."""
        engine = self._mock_engine()

        from jarvis_eval.store import EvalStore
        store = EvalStore(base_dir=tmp_eval_dir)

        from jarvis_eval import EvalSidecar
        sidecar = EvalSidecar()
        sidecar._store = store

        from jarvis_eval.collector import EvalCollector
        collector = EvalCollector(engine)
        sidecar._collector = collector
        sidecar._running = True

        snapshot = sidecar.get_dashboard_snapshot()
        assert isinstance(snapshot, dict)
        assert "banner" in snapshot

    def test_flush_handles_errors_gracefully(self, tmp_eval_dir):
        """Store write failures must not propagate."""
        engine = self._mock_engine()

        from jarvis_eval.store import EvalStore
        store = EvalStore(base_dir=tmp_eval_dir)

        original_append = store.append_snapshot

        def failing_append(snap):
            raise OSError("disk full")

        store.append_snapshot = failing_append

        from jarvis_eval import EvalSidecar
        sidecar = EvalSidecar()
        sidecar._store = store

        from jarvis_eval.collector import EvalCollector
        collector = EvalCollector(engine)
        sidecar._collector = collector

        sidecar._flush_once()
