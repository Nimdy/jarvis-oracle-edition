"""brain<->Pi link-health telemetry + derived device status (nervous-system view).

These cover the operational-dashboard tracer: event rate / last-seen / reconnects
computed from brain-stamped receipt times, and the per-device operational status
derived ONLY from telemetry that actually flows (no fabricated green lights).
"""
from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

import pytest

try:
    from perception.server import PerceptionServer
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("perception server import unavailable", allow_module_level=True)

try:
    from dashboard.pi5_devices import derive_pi5_devices as _derive_pi5_devices
except Exception:  # pragma: no cover
    _derive_pi5_devices = None


def _stub(recv, counts, connections, cc=1, dc=0):
    now = time.time()
    return SimpleNamespace(
        _event_recv_log=deque(recv),
        _event_counts=dict(counts),
        _connections=dict(connections),
        _connect_count=cc,
        _disconnect_count=dc,
        _last_sensor_connect=now - 5,
        _last_sensor_disconnect=now - 60,
    )


class TestLinkHealth:
    def test_empty(self):
        h = PerceptionServer.get_link_health(_stub([], {}, {}, cc=0))
        assert h["connected_count"] == 0
        assert h["total_events"] == 0
        assert h["last_event_age_s"] is None
        assert h["types"] == {}

    def test_rates_and_last_seen(self):
        now = time.time()
        recv = [(now - 1, "face_crop"), (now - 2, "face_crop"),
                (now - 0.5, "scene_summary"),
                (now - 120, "scene_caption")]  # caption is OUTSIDE the 60s window
        counts = {"face_crop": 2, "scene_summary": 1, "scene_caption": 1}
        h = PerceptionServer.get_link_health(_stub(recv, counts, {"pi5-senses": object()}))
        assert h["connected_count"] == 1
        assert h["connected"] == ["pi5-senses"]
        assert h["total_events"] == 4
        t = h["types"]
        assert t["face_crop"]["total"] == 2
        assert t["face_crop"]["rate_hz"] == pytest.approx(2 / 60, abs=1e-3)
        assert t["face_crop"]["last_seen_age_s"] >= 1.0
        # counted in total + last_seen, but rate 0 (outside window)
        assert t["scene_caption"]["rate_hz"] == 0.0
        assert t["scene_caption"]["last_seen_age_s"] >= 119

    def test_window_param(self):
        now = time.time()
        recv = [(now - 5, "x"), (now - 90, "x")]  # one inside, one outside 120s
        h = PerceptionServer.get_link_health(_stub(recv, {"x": 2}, {}), window_s=120.0)
        assert h["window_s"] == 120.0
        assert h["types"]["x"]["rate_hz"] == pytest.approx(2 / 120, abs=1e-3)

    def test_reconnect_counters_surfaced(self):
        h = PerceptionServer.get_link_health(_stub([], {}, {}, cc=4, dc=3))
        assert h["connect_count"] == 4
        assert h["disconnect_count"] == 3


@pytest.mark.skipif(_derive_pi5_devices is None, reason="dashboard.app import unavailable")
class TestDeriveDevices:
    def test_operational_from_real_signals(self):
        cache = {
            "sensors": ["pi5-senses"],
            "sensor_health": {"pi5-senses": {"camera_fps": 14.5, "uptime_s": 3600,
                                             "audio_playing": False}},
            "link": {"types": {
                "scene_summary": {"last_seen_age_s": 2.0, "rate_hz": 0.5, "total": 100},
                "scene_caption": {"last_seen_age_s": 5.0, "rate_hz": 0.05, "total": 10},
            }},
            "speakers": {"available": True, "current": "office"},
            "lidar": {},
        }
        devs = {d["name"]: d for d in _derive_pi5_devices(cache)}
        assert devs["Pi node"]["status"] == "up"
        assert devs["Camera (imx519)"]["status"] == "operational"
        assert devs["Hailo-10H NPU"]["status"] == "operational"
        assert devs["Hailo-10H NPU"]["present"] is True
        assert devs["Speaker"]["status"] == "available"
        assert devs["Microphone"]["status"] == "telemetry_pending"  # honest: no mic telemetry yet
        assert devs["RPLIDAR"]["status"] == "absent"

    def test_stale_and_down(self):
        cache = {
            "sensors": [],
            "sensor_health": {},
            "link": {"types": {"scene_summary": {"last_seen_age_s": 600.0, "rate_hz": 0.0, "total": 5}}},
            "speakers": {},
            "lidar": {},
        }
        devs = {d["name"]: d for d in _derive_pi5_devices(cache)}
        assert devs["Pi node"]["status"] == "down"
        assert devs["Hailo-10H NPU"]["status"] == "stale"  # inference 600s ago
        assert devs["Speaker"]["status"] == "down"
        assert devs["RPLIDAR"]["present"] is False

    def test_lidar_present_when_telemetry_flows(self):
        cache = {"sensors": ["pi5-senses"], "sensor_health": {}, "link": {},
                 "speakers": {}, "lidar": {"rplidar-a1m8": {"sectors": {"front": 1.2}}}}
        devs = {d["name"]: d for d in _derive_pi5_devices(cache)}
        assert devs["RPLIDAR"]["present"] is True
        assert devs["RPLIDAR"]["status"] == "operational"

    def test_no_fabricated_camera_when_no_signal(self):
        cache = {"sensors": ["pi5-senses"], "sensor_health": {"pi5-senses": {}},
                 "link": {"types": {}}, "speakers": {}, "lidar": {}}
        devs = {d["name"]: d for d in _derive_pi5_devices(cache)}
        # no fps, no frame events -> not asserted operational
        assert devs["Camera (imx519)"]["present"] is False
        assert devs["Camera (imx519)"]["status"] == "unknown"
