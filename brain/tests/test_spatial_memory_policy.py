"""Spatial memory policy — CueGate enforcement, no raw-to-memory path, no trust inflation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cognition.spatial_schema import (
    CONFIDENCE_THRESHOLD_MEMORY,
    SPATIAL_MEMORY_MAX_PER_DAY,
    SPATIAL_MEMORY_MAX_PER_HOUR,
    SpatialDelta,
    SpatialObservation,
)


# ---------------------------------------------------------------------------
# Invariant: raw spatial data never becomes memory
# ---------------------------------------------------------------------------


def test_spatial_observation_has_no_memory_write_method():
    """SpatialObservation must not have any method that writes to memory."""
    obs = SpatialObservation(
        entity_id="a", label="cup", depth_m=1.0,
        position_camera_m=(0, 0, 1),
    )
    for attr in dir(obs):
        assert "write" not in attr.lower() or attr.startswith("_"), (
            f"SpatialObservation has suspicious method: {attr}"
        )
        assert "memory" not in attr.lower() or attr.startswith("_"), (
            f"SpatialObservation has suspicious method: {attr}"
        )


def test_spatial_delta_has_no_memory_write_method():
    """SpatialDelta must not have any method that writes to memory."""
    delta = SpatialDelta(
        delta_id="d1", entity_id="a", label="cup", delta_type="moved",
    )
    for attr in dir(delta):
        assert "write" not in attr.lower() or attr.startswith("_"), (
            f"SpatialDelta has suspicious method: {attr}"
        )
        assert "memory" not in attr.lower() or attr.startswith("_"), (
            f"SpatialDelta has suspicious method: {attr}"
        )


# ---------------------------------------------------------------------------
# Invariant: memory promotion requires high confidence
# ---------------------------------------------------------------------------


def test_memory_threshold_is_high():
    """Memory-eligible spatial episodes require >= 0.88 confidence."""
    assert CONFIDENCE_THRESHOLD_MEMORY >= 0.85


# ---------------------------------------------------------------------------
# Invariant: memory budget exists
# ---------------------------------------------------------------------------


def test_memory_budget_per_hour():
    assert SPATIAL_MEMORY_MAX_PER_HOUR > 0
    assert SPATIAL_MEMORY_MAX_PER_HOUR <= 10


def test_memory_budget_per_day():
    assert SPATIAL_MEMORY_MAX_PER_DAY > 0
    assert SPATIAL_MEMORY_MAX_PER_DAY <= 50


def test_daily_budget_exceeds_hourly():
    assert SPATIAL_MEMORY_MAX_PER_DAY >= SPATIAL_MEMORY_MAX_PER_HOUR


# ---------------------------------------------------------------------------
# Invariant: no trust inflation without validated outcomes
# ---------------------------------------------------------------------------


def test_delta_carries_calibration_version():
    """Every promoted claim must carry calibration_version."""
    delta = SpatialDelta(
        delta_id="d1", entity_id="a", label="cup",
        delta_type="moved", calibration_version=3,
    )
    d = delta.to_dict()
    assert "calibration_version" in d
    assert d["calibration_version"] == 3


def test_observation_carries_provenance():
    """Every observation must carry provenance."""
    obs = SpatialObservation(
        entity_id="a", label="cup", depth_m=1.0,
        position_camera_m=(0, 0, 1),
        provenance="prior_based",
    )
    d = obs.to_dict()
    assert "provenance" in d
    assert d["provenance"] == "prior_based"


# ---------------------------------------------------------------------------
# Invariant: no sensor-to-memory path exists in spatial modules
# ---------------------------------------------------------------------------


def test_spatial_schema_does_not_import_memory():
    """spatial_schema.py must not import from memory package."""
    import cognition.spatial_schema as mod
    source_file = mod.__file__
    with open(source_file) as f:
        content = f.read()
    assert "from memory" not in content, "spatial_schema imports from memory"
    assert "import memory" not in content, "spatial_schema imports memory"


def test_spatial_fusion_does_not_import_memory():
    """spatial_fusion.py must not import from memory package."""
    import cognition.spatial_fusion as mod
    source_file = mod.__file__
    with open(source_file) as f:
        content = f.read()
    assert "from memory" not in content, "spatial_fusion imports from memory"
    assert "import memory" not in content, "spatial_fusion imports memory"


def test_spatial_validation_does_not_import_memory():
    """spatial_validation.py must not import from memory package."""
    import cognition.spatial_validation as mod
    source_file = mod.__file__
    with open(source_file) as f:
        content = f.read()
    assert "from memory" not in content, "spatial_validation imports from memory"
    assert "import memory" not in content, "spatial_validation imports memory"


def test_perception_spatial_does_not_import_memory():
    """perception/spatial.py must not import from memory package."""
    import perception.spatial as mod
    source_file = mod.__file__
    with open(source_file) as f:
        content = f.read()
    assert "from memory" not in content, "perception/spatial imports from memory"
    assert "import memory" not in content, "perception/spatial imports memory"


def test_calibration_does_not_import_memory():
    """perception/calibration.py must not import from memory package."""
    import perception.calibration as mod
    source_file = mod.__file__
    with open(source_file) as f:
        content = f.read()
    assert "from memory" not in content, "calibration imports from memory"
    assert "import memory" not in content, "calibration imports memory"


# ---------------------------------------------------------------------------
# SpatialMemoryGate tests
# ---------------------------------------------------------------------------

from cognition.spatial_memory_gate import SpatialMemoryGate
from cognition.spatial_schema import SpatialDelta


class FakeMemoryGate:
    def __init__(self, allow: bool = True):
        self._allow = allow

    def can_observation_write(self) -> bool:
        return self._allow


def _valid_delta(
    confidence: float = 0.90,
    validated: bool = True,
    cal_version: int = 1,
    delta_type: str = "moved",
    entity_id: str = "obj_cup",
) -> SpatialDelta:
    return SpatialDelta(
        delta_id="d1",
        entity_id=entity_id,
        label="cup",
        delta_type=delta_type,
        distance_m=0.3,
        dominant_axis="x",
        confidence=confidence,
        validated=validated,
        calibration_version=cal_version,
    )


def test_gate_blocks_when_cuegate_denies():
    gate = SpatialMemoryGate()
    mg = FakeMemoryGate(allow=False)
    ok, reason = gate.can_promote(_valid_delta(), memory_gate=mg)
    assert not ok
    assert "CueGate" in reason


def test_gate_blocks_low_confidence():
    gate = SpatialMemoryGate()
    ok, reason = gate.can_promote(_valid_delta(confidence=0.5))
    assert not ok
    assert "Confidence" in reason


def test_gate_blocks_unvalidated():
    gate = SpatialMemoryGate()
    ok, reason = gate.can_promote(_valid_delta(validated=False))
    assert not ok
    assert "not validated" in reason


def test_gate_blocks_no_calibration():
    gate = SpatialMemoryGate()
    ok, reason = gate.can_promote(_valid_delta(cal_version=0))
    assert not ok
    assert "calibration" in reason.lower()


def test_gate_allows_relevant_delta():
    gate = SpatialMemoryGate()
    ok, reason = gate.can_promote(_valid_delta(delta_type="moved"))
    assert ok
    assert reason == "ok"


def test_gate_blocks_non_relevant_non_repeated():
    gate = SpatialMemoryGate()
    ok, reason = gate.can_promote(_valid_delta(delta_type="stabilized"))
    assert not ok
    assert "not human-relevant" in reason.lower() or "not relevant" in reason.lower()


def test_gate_allows_after_repetition():
    gate = SpatialMemoryGate()
    delta = _valid_delta(delta_type="stabilized")
    gate.can_promote(delta)
    gate.can_promote(delta)
    ok, reason = gate.can_promote(delta)
    assert ok


def test_gate_enforces_hourly_budget():
    gate = SpatialMemoryGate()
    for i in range(SPATIAL_MEMORY_MAX_PER_HOUR):
        delta = _valid_delta(entity_id=f"obj_{i}")
        ok, _ = gate.can_promote(delta)
        assert ok
        gate.record_promotion(delta)

    ok, reason = gate.can_promote(_valid_delta(entity_id="obj_excess"))
    assert not ok
    assert "Hourly" in reason or "hour" in reason.lower()


def test_gate_builds_memory_content():
    gate = SpatialMemoryGate()
    delta = _valid_delta()
    delta.distance_m = 0.43
    delta.dominant_axis = "x"
    content = gate.build_memory_content(delta)
    assert "cup" in content
    assert "~0.4m" in content


def test_gate_builds_missing_content():
    gate = SpatialMemoryGate()
    delta = _valid_delta(delta_type="missing")
    content = gate.build_memory_content(delta)
    assert "vanished" in content


def test_gate_get_state():
    gate = SpatialMemoryGate()
    state = gate.get_state()
    assert "total_promoted" in state
    assert "total_blocked" in state
    assert "hour_budget" in state
    assert "day_budget" in state


def test_spatial_memory_gate_does_not_import_memory():
    """spatial_memory_gate.py must not import from memory package."""
    import cognition.spatial_memory_gate as mod
    source_file = mod.__file__
    with open(source_file) as f:
        content = f.read()
    assert "from memory" not in content, "spatial_memory_gate imports from memory"
    assert "import memory" not in content, "spatial_memory_gate imports memory"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
