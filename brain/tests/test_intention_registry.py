"""Tests for cognition.intention_registry.

Stage 0 infrastructure: verify the registry CRUD semantics, honest-failure
outcomes (resolved/failed/stale/abandoned), thread safety, atomic persistence,
and boot restore.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _fresh_registry(tmp_home: Path):
    """Return a fresh IntentionRegistry instance bound to tmp_home.

    Reloads the module with JARVIS paths pointed at tmp_home, so persistence
    does not touch the real ~/.jarvis.
    """
    os.environ["HOME"] = str(tmp_home)
    import importlib
    import sys as _sys
    # The cognition package __init__ shadows the submodule attribute with the
    # singleton. Resolve the real module via sys.modules after import.
    import cognition.intention_registry  # noqa: F401  (trigger submodule import)
    ir_mod = _sys.modules["cognition.intention_registry"]
    ir_mod = importlib.reload(ir_mod)
    ir_mod.JARVIS_DIR = tmp_home / ".jarvis"
    ir_mod.REGISTRY_PATH = ir_mod.JARVIS_DIR / "intention_registry.json"
    ir_mod.OUTCOMES_PATH = ir_mod.JARVIS_DIR / "intention_outcomes.jsonl"
    ir_mod.JARVIS_DIR.mkdir(parents=True, exist_ok=True)
    ir_mod.IntentionRegistry._instance = None
    reg = ir_mod.IntentionRegistry.get_instance()
    return ir_mod, reg


def test_register_rejects_empty_backing_job_id():
    with tempfile.TemporaryDirectory() as tmp:
        _ir, reg = _fresh_registry(Path(tmp))
        rid = reg.register(
            utterance="let me look into that",
            commitment_phrase="let me look into that",
            commitment_type="deferred_action",
            backing_job_id="",
            backing_job_kind="library_ingest",
        )
        assert rid is None, "empty backing_job_id must be rejected"
        assert reg.get_open_count() == 0


def test_register_and_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        _ir, reg = _fresh_registry(Path(tmp))
        rid = reg.register(
            utterance="I'll get back to you",
            commitment_phrase="I'll get back to you",
            commitment_type="follow_up",
            backing_job_id="job_abc123",
            backing_job_kind="autonomy_research",
        )
        assert rid is not None
        assert reg.get_open_count() == 1
        rec = reg.get_by_backing_job("job_abc123")
        assert rec is not None
        assert rec.outcome == "open"
        assert rec.commitment_type == "follow_up"


def test_resolve_closes_open_record():
    with tempfile.TemporaryDirectory() as tmp:
        _ir, reg = _fresh_registry(Path(tmp))
        reg.register(
            utterance="let me process",
            commitment_phrase="let me process",
            commitment_type="deferred_action",
            backing_job_id="job_1",
            backing_job_kind="library_ingest",
        )
        ok = reg.resolve(backing_job_id="job_1", outcome="resolved", reason="ingest_ok")
        assert ok is True
        assert reg.get_open_count() == 0
        rec = reg.get_by_backing_job("job_1")
        assert rec is not None
        assert rec.outcome == "resolved"
        assert rec.resolution_reason == "ingest_ok"
        assert rec.resolved_at > 0


def test_resolve_rejects_unsupported_outcome():
    with tempfile.TemporaryDirectory() as tmp:
        _ir, reg = _fresh_registry(Path(tmp))
        reg.register(
            utterance="x", commitment_phrase="x", commitment_type="generic",
            backing_job_id="job_x", backing_job_kind="k",
        )
        ok = reg.resolve(backing_job_id="job_x", outcome="made_up")
        assert ok is False
        assert reg.get_open_count() == 1


def test_abandon_path():
    with tempfile.TemporaryDirectory() as tmp:
        _ir, reg = _fresh_registry(Path(tmp))
        rid = reg.register(
            utterance="x", commitment_phrase="x", commitment_type="generic",
            backing_job_id="job_ab", backing_job_kind="k",
        )
        assert reg.abandon(intention_id=rid, reason="user_cancelled") is True
        assert reg.get_open_count() == 0
        status = reg.get_status()
        assert status["total_abandoned"] >= 1


def test_stale_sweep_marks_aged_records():
    ir_mod = None
    with tempfile.TemporaryDirectory() as tmp:
        ir_mod, reg = _fresh_registry(Path(tmp))
        rid = reg.register(
            utterance="x", commitment_phrase="x", commitment_type="generic",
            backing_job_id="job_old", backing_job_kind="k",
        )
        # Manually age the record
        rec = reg.get_by_backing_job("job_old")
        rec.created_at = time.time() - 3600.0
        n = reg.stale_sweep(max_age_s=60.0)
        assert n == 1
        assert reg.get_open_count() == 0
        status = reg.get_status()
        assert status["total_stale"] >= 1


def test_get_status_shape():
    with tempfile.TemporaryDirectory() as tmp:
        _ir, reg = _fresh_registry(Path(tmp))
        s = reg.get_status()
        for key in ("open_count", "resolved_buffer_count",
                    "most_recent_open_intention_age_s",
                    "oldest_open_intention_age_s",
                    "total_registered", "total_resolved",
                    "total_failed", "total_stale",
                    "total_abandoned", "outcome_histogram_7d",
                    "errors", "loaded"):
            assert key in s, f"status missing key: {key}"
        for k in ("resolved", "failed", "stale", "abandoned"):
            assert k in s["outcome_histogram_7d"]


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ir_mod, reg = _fresh_registry(Path(tmp))
        reg.register(
            utterance="I'll get back to you",
            commitment_phrase="I'll get back to you",
            commitment_type="follow_up",
            backing_job_id="job_keep",
            backing_job_kind="autonomy_research",
        )
        reg.register(
            utterance="let me process",
            commitment_phrase="let me process",
            commitment_type="deferred_action",
            backing_job_id="job_done",
            backing_job_kind="library_ingest",
        )
        reg.resolve(backing_job_id="job_done", outcome="resolved", reason="ingest_ok")
        assert reg.save() is True
        assert ir_mod.REGISTRY_PATH.exists()
        assert ir_mod.OUTCOMES_PATH.exists()

        # Fresh singleton, same path
        ir_mod.IntentionRegistry._instance = None
        fresh = ir_mod.IntentionRegistry.get_instance()
        loaded_open = fresh.load()
        assert loaded_open == 1
        assert fresh.get_open_count() == 1
        assert fresh.get_by_backing_job("job_keep") is not None
        resolved = fresh.get_by_backing_job("job_done")
        assert resolved is not None
        assert resolved.outcome == "resolved"


def test_outcome_histogram_counts_last_7d():
    with tempfile.TemporaryDirectory() as tmp:
        _ir, reg = _fresh_registry(Path(tmp))
        for i, outcome in enumerate(["resolved", "failed", "resolved"]):
            reg.register(
                utterance=f"x{i}", commitment_phrase=f"x{i}", commitment_type="generic",
                backing_job_id=f"job_{i}", backing_job_kind="k",
            )
            reg.resolve(backing_job_id=f"job_{i}", outcome=outcome, reason="t")
        hist = reg.get_status()["outcome_histogram_7d"]
        assert hist["resolved"] == 2
        assert hist["failed"] == 1


if __name__ == "__main__":
    import traceback
    fns = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for f in fns:
        try:
            f()
            passed += 1
            print(f"  PASS: {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {f.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"  ERROR: {f.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
