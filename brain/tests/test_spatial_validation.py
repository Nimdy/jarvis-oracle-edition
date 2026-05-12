"""Spatial validation — delta thresholds, contradiction triggers, anchor revalidation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cognition.spatial_schema import (
    AUTHORITY_LEVELS,
    CLASS_MOVE_THRESHOLDS,
    STABLE_WINDOWS_REQUIRED,
    SpatialAnchor,
    SpatialTrack,
)
from cognition.spatial_validation import RejectionLedger, SpatialValidator


def _track(
    eid: str = "obj_cup",
    label: str = "cup",
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
    status: str = "stable",
    confidence: float = 0.85,
    windows: int = 5,
) -> SpatialTrack:
    return SpatialTrack(
        entity_id=eid,
        label=label,
        track_status=status,
        position_room_m=pos,
        confidence=confidence,
        samples=10,
        stable_windows=windows,
        authority=AUTHORITY_LEVELS["stable_track"],
    )


def _anchor(
    aid: str = "anchor_desk",
    label: str = "desk",
    pos: tuple[float, float, float] = (0.0, -0.5, 1.0),
) -> SpatialAnchor:
    return SpatialAnchor(
        anchor_id=aid,
        anchor_type="desk_plane",
        label=label,
        position_room_m=pos,
        confidence=0.9,
        calibration_version=1,
        authority=AUTHORITY_LEVELS["stable_anchor"],
    )


# -- Rejection ledger --


def test_rejection_ledger_records_and_counts():
    ledger = RejectionLedger()
    ledger.record("a", "track_to_delta", "not_stable")
    ledger.record("b", "track_to_delta", "not_stable")
    ledger.record("c", "track_to_delta", "low_confidence")
    assert ledger.total_rejections == 3
    assert ledger.get_counts()["not_stable"] == 2
    assert ledger.get_counts()["low_confidence"] == 1
    assert len(ledger.get_recent(10)) == 3


# -- Validator rejects unstable --


def test_rejects_provisional_track():
    v = SpatialValidator()
    t = _track(status="provisional")
    result = v.validate_track_to_delta(t, {}, 1)
    assert result is None
    assert v.rejection_ledger.total_rejections == 1


def test_rejects_insufficient_windows():
    v = SpatialValidator()
    t = _track(windows=1)
    result = v.validate_track_to_delta(t, {}, 1)
    assert result is None


# -- Anchor authority enforcement --


def test_rejects_when_anchor_conflict():
    v = SpatialValidator()
    anchor = SpatialAnchor(
        anchor_id="anchor_mon",
        anchor_type="monitor_center",
        label="monitor",
        position_room_m=(0.0, 0.0, 1.0),
        confidence=0.9,
        calibration_version=1,
        authority=AUTHORITY_LEVELS["stable_anchor"],
    )
    t = _track(
        eid="obj_mon", label="monitor",
        pos=(0.0, 0.0, 2.0),
        confidence=0.85, windows=5,
    )
    # First call establishes position
    v._previous_positions["obj_mon"] = (0.0, 0.0, 0.5)
    result = v.validate_track_to_delta(t, {"anchor_mon": anchor}, 1)
    assert result is None
    assert "anchor_authority_conflict" in v.rejection_ledger.get_counts()


# -- Delta detection --


def test_no_delta_on_first_observation():
    v = SpatialValidator()
    t = _track(pos=(0.0, 0.0, 1.0))
    result = v.validate_track_to_delta(t, {}, 1)
    assert result is None


def test_no_delta_below_jitter_threshold():
    v = SpatialValidator()
    t1 = _track(pos=(0.0, 0.0, 1.0))
    v.validate_track_to_delta(t1, {}, 1)

    t2 = _track(pos=(0.01, 0.0, 1.0))
    result = v.validate_track_to_delta(t2, {}, 1)
    assert result is None


def test_delta_detected_above_threshold():
    v = SpatialValidator()
    cup_threshold = CLASS_MOVE_THRESHOLDS.get("cup", 0.05)

    t1 = _track(pos=(0.0, 0.0, 1.0))
    v.validate_track_to_delta(t1, {}, 1)

    v._consecutive_windows["obj_cup"] = 5
    t2 = _track(pos=(cup_threshold + 0.5, 0.0, 1.0))
    result = v.validate_track_to_delta(t2, {}, 1)

    assert result is not None
    assert result.delta_type == "moved"
    assert result.distance_m > cup_threshold
    assert result.validated is True
    assert len(result.reason_codes) >= 1


# -- Missing entity --


def test_check_missing_entity():
    v = SpatialValidator()
    delta = v.check_missing_entity(
        "obj_cup", "cup", (0.3, -0.1, 1.2), 1,
    )
    assert delta.delta_type == "missing"
    assert delta.from_position_m == (0.3, -0.1, 1.2)
    assert "entity_vanished_from_stable_position" in delta.reason_codes


# -- State snapshot --


def test_get_state():
    v = SpatialValidator()
    t = _track(status="provisional")
    v.validate_track_to_delta(t, {}, 1)
    state = v.get_state()
    assert "total_validated" in state
    assert "total_promoted" in state
    assert "total_rejections" in state
    assert "rejection_counts" in state
    assert "recent_rejections" in state
    assert "recent_deltas" in state


def test_reset_for_relocalization_clears_temporal_baselines():
    v = SpatialValidator()
    v._previous_positions["obj_cup"] = (0.0, 0.0, 1.0)
    v._consecutive_windows["obj_cup"] = 4
    v.reset_for_relocalization(profile_id="scene_a", reason="matched_profile")
    assert v._previous_positions == {}
    assert v._consecutive_windows == {}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
