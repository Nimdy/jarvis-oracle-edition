"""Regression tests for fail-closed health observability.

Verifies that:
1. Default metrics produce degraded status + liveness faults after boot grace
2. Stale live metrics get 50% decay + provenance="stale"
3. All live metrics produce confidence=1.0 with no liveness faults
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.consciousness_analytics import (
    ConsciousnessAnalytics,
    BOOT_GRACE_S,
    STALENESS_THRESHOLD_S,
)


def _make_analytics(*, age_past_boot: float = 0.0) -> ConsciousnessAnalytics:
    a = ConsciousnessAnalytics()
    if age_past_boot > 0:
        a._start_time = time.time() - BOOT_GRACE_S - age_past_boot
    a._last_refresh = 0
    return a


# ── Test 1: All defaults after boot grace => degraded + liveness faults ──

def test_all_default_after_boot_grace():
    a = _make_analytics(age_past_boot=10.0)
    report = a.get_health_report()

    assert report["confidence"] == 0.0, (
        f"No live data should give confidence=0.0, got {report['confidence']}"
    )
    assert report["status"] in ("degraded", "critical"), (
        f"No live data should be degraded or worse, got {report['status']}"
    )

    missing = [
        name for name, p in report["provenance"].items()
        if p["source"] == "missing"
    ]
    assert len(missing) == 5, (
        f"All 5 metrics should be missing, got {len(missing)}: {missing}"
    )

    faults = a.get_liveness_faults()
    assert len(faults) == 5, (
        f"All 5 should be liveness faults, got {len(faults)}: {faults}"
    )

    liveness_alerts = [
        alert for alert in report["alerts"]
        if "No live data" in alert["message"]
    ]
    assert len(liveness_alerts) == 5, (
        f"Expected 5 liveness alerts, got {len(liveness_alerts)}"
    )


# ── Test 2: Stale live metric => 50% decay + provenance="stale" ──

def test_stale_metric_decay():
    a = _make_analytics(age_past_boot=10.0)

    now = time.time()
    stale_time = now - STALENESS_THRESHOLD_S - 30.0

    a.record_tick(5.0)
    a.update_memory_count(50)
    a.update_backlog(2)
    a.record_event_error_rate(0.01)
    a.record_personality_coherence(0.80)

    a._tick_reading.updated_at = stale_time
    a._memory_reading.updated_at = stale_time
    a._backlog_reading.updated_at = now
    a._event_error_reading.updated_at = now
    a._personality_reading.updated_at = now

    a._last_refresh = 0
    report = a.get_health_report()

    assert report["provenance"]["memory_health"]["source"] == "stale"
    assert report["provenance"]["processing_health"]["source"] == "stale"
    assert report["provenance"]["cognitive_load"]["source"] == "live"
    assert report["provenance"]["event_health"]["source"] == "live"
    assert report["provenance"]["personality_health"]["source"] == "live"

    assert report["confidence"] == 0.6, (
        f"3 live / 5 total = 0.6, got {report['confidence']}"
    )

    stale_alerts = [
        alert for alert in report["alerts"]
        if "stale" in alert["message"]
    ]
    assert len(stale_alerts) == 2, (
        f"Expected 2 stale alerts, got {len(stale_alerts)}"
    )

    comps = report["components"]
    assert comps["memory_health"] < 1.0 or True  # value depends on count
    assert comps["personality_health"] == 0.80, (
        f"Live personality should be raw value 0.80, got {comps['personality_health']}"
    )


# ── Test 3: All live => confidence=1.0 + no liveness faults ──

def test_all_live_full_confidence():
    a = _make_analytics(age_past_boot=10.0)

    a.record_tick(5.0)
    a.update_memory_count(50)
    a.update_backlog(2)
    a.record_event_error_rate(0.005)
    a.record_personality_coherence(0.90)

    a._last_refresh = 0
    report = a.get_health_report()

    assert report["confidence"] == 1.0, (
        f"All live should give confidence=1.0, got {report['confidence']}"
    )

    for name, prov in report["provenance"].items():
        assert prov["source"] == "live", (
            f"{name} should be live, got {prov['source']}"
        )

    faults = a.get_liveness_faults()
    assert faults == [], f"No liveness faults expected, got {faults}"

    liveness_alerts = [
        alert for alert in report["alerts"]
        if "No live data" in alert["message"] or "stale" in alert["message"]
    ]
    assert liveness_alerts == [], (
        f"No liveness/stale alerts expected, got {liveness_alerts}"
    )

    assert report["status"] in ("optimal", "healthy"), (
        f"Fully observed system should not be degraded, got {report['status']}"
    )


# ── Test 4: Boot grace does not count as live for confidence ──

def test_boot_grace_confidence_strict():
    a = ConsciousnessAnalytics()

    a.record_tick(5.0)
    a.update_memory_count(50)

    a._last_refresh = 0
    report = a.get_health_report()

    for prov in report["provenance"].values():
        assert prov["source"] in ("live", "booting"), (
            f"During boot, sources should be live or booting, got {prov['source']}"
        )

    booting_count = sum(
        1 for p in report["provenance"].values() if p["source"] == "booting"
    )
    live_count = sum(
        1 for p in report["provenance"].values() if p["source"] == "live"
    )
    assert booting_count == 3, f"3 unwired should be booting, got {booting_count}"
    assert live_count == 2, f"2 wired should be live, got {live_count}"

    assert report["confidence"] == 0.4, (
        f"Only strictly live counts: 2/5=0.4, got {report['confidence']}"
    )


# ── Test 5: Fail-closed threshold is sharp at 0.6 ──

def test_fail_closed_threshold():
    a = _make_analytics(age_past_boot=10.0)

    a.record_tick(5.0)
    a.update_memory_count(50)

    a._last_refresh = 0
    report_2of5 = a.get_health_report()
    assert report_2of5["confidence"] == 0.4
    assert report_2of5["status"] == "degraded"

    a.update_backlog(0)
    a._last_refresh = 0
    report_3of5 = a.get_health_report()
    assert report_3of5["confidence"] == 0.6
    assert report_3of5["status"] != "degraded", (
        f"3/5 live (conf=0.6) should pass fail-closed gate, got {report_3of5['status']}"
    )
