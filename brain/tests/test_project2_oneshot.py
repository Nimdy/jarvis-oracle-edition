"""Pins for feat/project-2-oneshot remaining [build-now] couples."""
from __future__ import annotations

import os
from pathlib import Path

from cognition.self_view.grounding import p2_active_default
from hemisphere.distillation import DistillationCollector
from hemisphere.weight_room_gate import enforces, may_promote
from goals.signal_producers import detect_metric_deficits, get_producer_stats


def test_p2_default_on_without_env(monkeypatch):
    monkeypatch.delenv("OSV_P2_ACTIVE", raising=False)
    assert p2_active_default() is True


def test_goals_observe_requires_api_key():
    src = Path(__file__).resolve().parent.parent.joinpath("dashboard/app.py").read_text()
    idx = src.find('@app.post("/api/goals/observe"')
    assert idx > 0
    assert "Depends(_require_api_key)" in src[idx:idx + 180]


def test_metric_health_deficits_are_unactionable():
    before = get_producer_stats()["metric_unactionable_skipped"]
    health = {"components": {"processing_health": 0.1, "memory_health": 0.1,
                             "personality_health": 0.1, "event_health": 0.1}}
    # Warmup: call enough times with high uptime.
    for _ in range(4):
        detect_metric_deficits(health, None, None, uptime_s=1000.0)
    after = get_producer_stats()["metric_unactionable_skipped"]
    assert after >= before
    signals = detect_metric_deficits(health, None, None, uptime_s=1000.0)
    assert all(s.source != "health_monitor" for s in signals)


def test_live_shadow_accuracy_none_below_min_n(monkeypatch):
    from collections import deque
    import threading
    monkeypatch.setattr(DistillationCollector, "_write_jsonl", classmethod(lambda cls, path, signal: None))
    c = DistillationCollector.__new__(DistillationCollector)
    c._buffers = {}
    c._quarantine = {}
    c._lock = threading.Lock()
    c._counts = {}
    c._synthetic_counts = {}
    c._quarantine_counts = {}
    c._last_seen = {}
    c._last_lived_seen = {}
    c._recent_dedup_keys = deque(maxlen=200)
    assert c.live_shadow_accuracy("claim_verdict", min_n=10) is None
    for i in range(12):
        c.record("claim_verdict", "label", {"x": i}, metadata={"correct": i % 2 == 0}, origin="live")
    acc = c.live_shadow_accuracy("claim_verdict", min_n=10)
    assert acc is not None
    assert 0.0 <= acc <= 1.0


def test_weight_room_enforces_default_off(monkeypatch):
    monkeypatch.delenv("WEIGHT_ROOM_ENFORCES", raising=False)
    assert enforces() is False
    d = may_promote("claim_classifier", lived=0, synthetic=500, live_shadow_accuracy=None)
    assert d["allowed"] is True
    assert d["would_allow"] is False
    monkeypatch.setenv("WEIGHT_ROOM_ENFORCES", "true")
    d2 = may_promote("claim_classifier", lived=0, synthetic=500, live_shadow_accuracy=None)
    assert d2["enforces"] is True
    assert d2["allowed"] is False
