"""Pi5 body — RPLIDAR telemetry handler (brain side).

The brain stores 2D LIDAR sector summaries as TELEMETRY-ONLY: surfaced for the
world model + dashboard, never written into beliefs/memory.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

try:
    from consciousness.events import PerceptionEvent
    from perception.server import PerceptionServer
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("perception server import unavailable", allow_module_level=True)


def _ev(data):
    return PerceptionEvent(source="system", type="lidar_scan", timestamp=1.0, data=data)


def test_lidar_scan_stored_telemetry_only():
    stub = SimpleNamespace(_event_buffer=deque(maxlen=10), _lidar_telemetry={})
    PerceptionServer._process_event(stub, _ev({
        "sensor": "rplidar_a1m8", "scan_hz": 5.5, "points": 700, "range_max_m": 12,
        "sectors": {"front": 1.42, "left": 0.88, "right": 2.31},
        "open_sectors": ["right"], "scan_quality": "healthy",
    }), "rplidar-a1m8")
    t = stub._lidar_telemetry["rplidar-a1m8"]
    # the telemetry-only contract is enforced at the store
    assert t["authority"] == "spatial_telemetry_only"
    assert t["writes_beliefs"] is False
    assert t["sectors"]["front"] == 1.42 and t["scan_quality"] == "healthy"
    assert t["points"] == 700 and t["open_sectors"] == ["right"]


def test_scan_2d_alias_also_handled():
    stub = SimpleNamespace(_event_buffer=deque(maxlen=10), _lidar_telemetry={})
    ev = PerceptionEvent(source="system", type="scan_2d", timestamp=1.0,
                         data={"sensor": "rplidar_a1m8", "sectors": {"front": 2.0}})
    PerceptionServer._process_event(stub, ev, "rplidar-a1m8")
    assert stub._lidar_telemetry["rplidar-a1m8"]["sectors"]["front"] == 2.0


def test_get_lidar_telemetry_returns_copy():
    stub = SimpleNamespace(_lidar_telemetry={"x": {"a": 1}})
    out = PerceptionServer.get_lidar_telemetry(stub)
    assert out == {"x": {"a": 1}} and out is not stub._lidar_telemetry
