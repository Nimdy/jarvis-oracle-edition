"""PR1 (Stage 2b): the brain-side room-model surface is additive + INERT.

It must be empty until the Pi streams raw points_polar (Stage 2a), and every read is
stamped telemetry-only / writes_beliefs=False. Tested via the unbound method on a
stub so we don't stand up the whole websocket server.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    from perception.server import PerceptionServer
    from cognition.lidar_room import LidarRoomModel
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("perception.server import unavailable", allow_module_level=True)

_get = PerceptionServer.get_lidar_room_model


def test_empty_with_no_rooms():
    stub = SimpleNamespace(_lidar_rooms={})
    assert _get(stub) == {}


def test_room_without_points_is_honest_empty_and_telemetry_only():
    stub = SimpleNamespace(_lidar_rooms={"pi-lidar": LidarRoomModel()})
    out = _get(stub)["pi-lidar"]
    assert out["reason"] == "insufficient_coverage"   # no ingest → empty-by-design
    assert out["authority"] == "spatial_telemetry_only"
    assert out["writes_beliefs"] is False


def test_specific_sensor_selector():
    stub = SimpleNamespace(_lidar_rooms={"pi-lidar": LidarRoomModel()})
    assert "pi-lidar" in _get(stub, "pi-lidar")
    assert _get(stub, "nope") == {}
