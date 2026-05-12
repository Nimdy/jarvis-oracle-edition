"""Unit tests for the eval sidecar: contracts, store, collector, baselines."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Contracts ──────────────────────────────────────────────────────────────

from jarvis_eval.contracts import EvalEvent, EvalRun, EvalScore, EvalSnapshot


class TestContracts:
    def test_eval_event_frozen(self):
        ev = EvalEvent(event_type="test")
        with pytest.raises(AttributeError):
            ev.event_type = "changed"

    def test_eval_event_roundtrip(self):
        ev = EvalEvent(event_type="test", payload={"k": 1}, mode="shadow")
        d = ev.to_dict()
        assert d["event_type"] == "test"
        assert d["payload"] == {"k": 1}
        restored = EvalEvent.from_dict(d)
        assert restored.event_type == ev.event_type
        assert restored.payload == ev.payload

    def test_eval_snapshot_roundtrip(self):
        snap = EvalSnapshot(source="memory", metrics={"total": 42})
        d = snap.to_dict()
        restored = EvalSnapshot.from_dict(d)
        assert restored.source == "memory"
        assert restored.metrics == {"total": 42}

    def test_eval_score_roundtrip(self):
        score = EvalScore(category="epistemic", score=85.0, sample_size=100)
        d = score.to_dict()
        restored = EvalScore.from_dict(d)
        assert restored.category == "epistemic"
        assert restored.score == 85.0

    def test_eval_run_defaults(self):
        run = EvalRun()
        assert run.mode == "shadow"
        assert run.run_id  # non-empty
        assert run.started_at > 0

    def test_from_dict_ignores_extra_keys(self):
        d = {"event_type": "test", "extra_key": "ignored"}
        ev = EvalEvent.from_dict(d)
        assert ev.event_type == "test"

    def test_auto_generated_ids_unique(self):
        ids = {EvalEvent(event_type="t").event_id for _ in range(100)}
        assert len(ids) == 100


# ── Store ──────────────────────────────────────────────────────────────────

from jarvis_eval.store import EvalStore


class TestStore:
    @pytest.fixture
    def tmp_dir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_append_event(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        ev = EvalEvent(event_type="test", payload={"x": 1})
        store.append_event(ev)
        assert store._total_events_written == 1
        records = store.read_recent_events(limit=10)
        assert len(records) == 1
        assert records[0]["event_type"] == "test"

    def test_append_snapshot(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        snap = EvalSnapshot(source="memory", metrics={"total": 5})
        store.append_snapshot(snap)
        assert store._total_snapshots_written == 1
        records = store.read_recent_snapshots(limit=10)
        assert len(records) == 1
        assert records[0]["source"] == "memory"

    def test_read_tail_limit(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        for i in range(20):
            store.append_event(EvalEvent(event_type=f"e{i}"))
        records = store.read_recent_events(limit=5)
        assert len(records) == 5
        assert records[0]["event_type"] == "e15"

    def test_empty_read(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        assert store.read_recent_events() == []
        assert store.read_recent_snapshots() == []
        assert store.read_recent_scores() == []

    def test_flush_meta(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        store.append_event(EvalEvent(event_type="t"))
        store.flush_meta()
        meta_path = tmp_dir / "eval_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["total_events_written"] == 1
        assert meta["schema_version"] == 1

    def test_get_meta(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        meta = store.get_meta()
        assert meta["total_events_written"] == 0
        assert "scoring_version" in meta

    def test_get_file_sizes(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        sizes = store.get_file_sizes()
        assert sizes["events"] == 0
        store.append_event(EvalEvent(event_type="t"))
        sizes = store.get_file_sizes()
        assert sizes["events"] > 0

    def test_rotation(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        with patch("jarvis_eval.config.MAX_JSONL_SIZE_MB", 0):
            store.append_event(EvalEvent(event_type="t1"))
            store.append_event(EvalEvent(event_type="t2"))
        rotated = tmp_dir / "eval_events.jsonl.1"
        assert rotated.exists() or (tmp_dir / "eval_events.jsonl").exists()

    def test_meta_persistence_across_instances(self, tmp_dir):
        store1 = EvalStore(base_dir=tmp_dir)
        store1.append_event(EvalEvent(event_type="t"))
        store1.flush_meta()

        store2 = EvalStore(base_dir=tmp_dir)
        assert store2._total_events_written == 1

    def test_dropped_event_count(self, tmp_dir):
        store = EvalStore(base_dir=tmp_dir)
        original_append = store._append_jsonl

        def fail_append(path, record):
            return False

        store._append_jsonl = fail_append
        store.append_event(EvalEvent(event_type="t"))
        assert store._dropped_event_count == 1


class TestSnapshotSelection:
    def test_latest_by_source_uses_timestamp_not_list_order(self):
        from jarvis_eval.dashboard_adapter import _latest_by_source

        latest = _latest_by_source([
            {"source": "autonomy", "timestamp": 200.0, "metrics": {"autonomy_level": 2}},
            {"source": "truth_calibration", "timestamp": 150.0, "metrics": {"truth_score": 0.5}},
            {"source": "autonomy", "timestamp": 100.0, "metrics": {"autonomy_level": 1}},
        ])

        assert latest["autonomy"]["autonomy_level"] == 2
        assert latest["truth_calibration"]["truth_score"] == 0.5


class TestScorecardTruthFallback:
    def test_drift_alert_count_uses_canonical_key(self):
        from jarvis_eval.scorecards import _count_active_drift_alerts

        truth = {
            "active_drift_alerts": [{"domain": "prediction"}],
            "drift_alerts": [{"domain": "prediction"}, {"domain": "autonomy"}],
        }
        assert _count_active_drift_alerts(truth) == 1

    def test_drift_alert_count_falls_back_to_alias_key(self):
        from jarvis_eval.scorecards import _count_active_drift_alerts

        truth = {
            "drift_alerts": [{"domain": "prediction"}, {"domain": "autonomy"}],
        }
        assert _count_active_drift_alerts(truth) == 2


# ── Baselines ──────────────────────────────────────────────────────────────

from jarvis_eval.baselines import classify


class TestBaselines:
    def test_inverted_green(self):
        assert classify("contradiction_debt", 0.02) == "green"

    def test_inverted_yellow(self):
        assert classify("contradiction_debt", 0.10) == "yellow"

    def test_inverted_red(self):
        assert classify("contradiction_debt", 0.50) == "red"

    def test_normal_green(self):
        assert classify("soul_integrity_index", 0.80) == "green"

    def test_normal_yellow(self):
        assert classify("soul_integrity_index", 0.55) == "yellow"

    def test_normal_red(self):
        assert classify("soul_integrity_index", 0.30) == "red"

    def test_none_value(self):
        assert classify("contradiction_debt", None) == "grey"

    def test_unknown_metric(self):
        assert classify("nonexistent_metric", 0.5) == "grey"


# ── Collector ──────────────────────────────────────────────────────────────

from jarvis_eval.collector import EvalCollector


class TestCollector:
    def _mock_engine(self):
        engine = MagicMock()
        engine.get_state.return_value = {
            "phase": "LISTENING",
            "mode": "conversational",
            "memory_count": 42,
            "tick": 100,
        }
        cs = MagicMock()
        cs.get_state.return_value.to_dict.return_value = {
            "stage": "self_reflective",
            "transcendence_level": 0.5,
        }
        cs.get_dream_artifact_stats.return_value = {"buffer": {"size": 3}}
        cs.get_cortex_stats.return_value = {"ranker": {"enabled": True}}
        cs.check_health.return_value = {"memory": {"healthy": True}}
        cs.observer = None
        engine.consciousness = cs
        return engine

    def test_collect_once_returns_snapshots(self):
        engine = self._mock_engine()
        collector = EvalCollector(engine)
        collector._run_id = "test-run"
        snapshots = collector.collect_once()
        assert len(snapshots) > 0
        sources = {s.source for s in snapshots}
        assert "engine_state" in sources
        assert "consciousness" in sources

    def test_collect_once_handles_failures(self):
        engine = MagicMock()
        engine.get_state.side_effect = RuntimeError("boom")
        engine.consciousness = None
        collector = EvalCollector(engine)
        snapshots = collector.collect_once()
        assert collector._collect_errors >= 1

    def test_snapshot_shape(self):
        engine = self._mock_engine()
        collector = EvalCollector(engine)
        snapshots = collector.collect_once()
        for snap in snapshots:
            assert hasattr(snap, "source")
            assert hasattr(snap, "metrics")
            assert hasattr(snap, "timestamp")
            assert isinstance(snap.metrics, dict)

    def test_get_stats(self):
        engine = self._mock_engine()
        collector = EvalCollector(engine)
        collector.collect_once()
        stats = collector.get_stats()
        assert stats["snapshots_collected"] > 0
        assert stats["last_collect_ts"] > 0


# ── Event Tap ──────────────────────────────────────────────────────────────

from jarvis_eval.event_tap import EvalEventTap, _safe_payload


class TestEventTap:
    def test_safe_payload_primitives(self):
        p = _safe_payload({"a": 1, "b": "hello", "c": True, "d": None})
        assert p == {"a": 1, "b": "hello", "c": True, "d": None}

    def test_safe_payload_complex(self):
        p = _safe_payload({"obj": {"nested": "val"}, "lst": [1, 2, 3]})
        assert p["obj"] == {"nested": "val"}
        assert p["lst"] == [1, 2, 3]

    def test_safe_payload_truncation(self):
        """Non-primitive types get str() truncated; strings pass through."""
        obj = type("Big", (), {"__str__": lambda s: "x" * 500})()
        p = _safe_payload({"x": obj})
        assert len(p["x"]) <= 200

    def test_safe_payload_max_keys(self):
        big = {f"k{i}": i for i in range(50)}
        p = _safe_payload(big, max_keys=5)
        assert len(p) == 5

    def test_safe_payload_prioritizes_correlation_keys(self):
        big = {f"k{i}": i for i in range(50)}
        big["trace_id"] = "trc_test"
        big["request_id"] = "req_test"
        p = _safe_payload(big, max_keys=3)
        assert "trace_id" in p
        assert "request_id" in p

    def test_drain_clears_buffer(self):
        tap = EvalEventTap()
        from jarvis_eval.contracts import EvalEvent
        with tap._lock:
            tap._buffer.append(EvalEvent(event_type="t"))
            tap._events_buffered = 1
        events = tap.drain()
        assert len(events) == 1
        assert len(tap._buffer) == 0

    def test_get_stats(self):
        tap = EvalEventTap()
        stats = tap.get_stats()
        assert stats["wired"] is False
        assert stats["buffer_size"] == 0

    def test_unwire_without_wire(self):
        tap = EvalEventTap()
        tap.unwire()  # should not raise


# ── Dashboard Adapter ─────────────────────────────────────────────────────

from jarvis_eval.dashboard_adapter import build_dashboard_snapshot


class TestDashboardAdapter:
    def test_empty_inputs(self):
        result = build_dashboard_snapshot(
            store_meta={"created_at": 1000},
            store_file_sizes={},
            recent_snapshots=[],
            recent_events=[],
            recent_scores=[],
            collector_stats={},
            tap_stats={},
        )
        assert "banner" in result
        assert result["banner"]["mode"] == "shadow"
        assert result["banner"]["composite_enabled"] is False
        assert "integrity" in result
        assert "scoreboard" in result
        assert "validation_pack" in result

    def test_integrity_cards(self):
        snaps = [
            {"source": "contradiction", "timestamp": 1000,
             "metrics": {"contradiction_debt": 0.03}},
            {"source": "soul_integrity", "timestamp": 1000,
             "metrics": {"current_index": 0.85}},
        ]
        result = build_dashboard_snapshot(
            store_meta={"created_at": 1000},
            store_file_sizes={},
            recent_snapshots=snaps,
            recent_events=[],
            recent_scores=[],
            collector_stats={},
            tap_stats={},
        )
        cards = result["integrity"]
        debt_card = next(c for c in cards if c["label"] == "Contradiction Debt")
        assert debt_card["value"] == 0.03
        assert debt_card["color"] == "green"

    def test_scoreboard_placeholder(self):
        result = build_dashboard_snapshot(
            store_meta={"created_at": 1000},
            store_file_sizes={},
            recent_snapshots=[],
            recent_events=[],
            recent_scores=[],
            collector_stats={},
            tap_stats={},
        )
        sb = result["scoreboard"]
        assert len(sb["bars"]) == 7
        assert all(b["score"] is None for b in sb["bars"])
        assert sb["composite"] is None
        assert sb["badge"] == "experimental"
