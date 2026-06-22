"""Tests for the spatial place consolidator (read-only, zero-authority, vector-free).

Covers the validated must-fix list (docs/SPATIAL_PLACE_CONSOLIDATION.md):
same-geometry -> same place_id, calibration-invariance, different-geometry -> different
place, unavailable geometry -> fail-closed, and records vector-free + authority-false +
PRE-MATURE.
"""

import json

from memory.spatial_place_consolidator import (
    AUTHORITY_FLAGS,
    LANE,
    STATUS,
    SpatialPlaceConsolidator,
    _place_id,
    _same_place,
    _stable_anchors,
    _VECTOR_KEYS,
)


def _world(*entities):
    return {"entities": [
        {"label": lbl, "position_room_m": pos, "confidence": 0.9} for lbl, pos in entities
    ]}


def _anchors(*entities):
    return _stable_anchors(_world(*entities))


def test_same_geometry_same_place_id():
    a = _anchors(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3]))
    b = _anchors(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3]))
    assert _place_id(a) is not None
    assert _place_id(a) == _place_id(b)


def test_calibration_invariant_same_room():
    """A room-frame re-origin (recalibration) must NOT change place identity."""
    a = _anchors(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3]), ("keyboard", [0.1, 1.2, 0.5]))
    shifted = _anchors(
        ("tv", [10.4, 10.8, 11.85]), ("chair", [10.0, 11.5, 10.3]), ("keyboard", [10.1, 11.2, 10.5])
    )
    assert _same_place(a, shifted)
    assert _place_id(a) == _place_id(shifted)


def test_different_geometry_different_place():
    a = _anchors(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3]), ("keyboard", [0.1, 1.2, 0.5]))
    moved = _anchors(("tv", [0.4, 0.8, 1.85]), ("chair", [3.0, 3.5, 3.3]), ("keyboard", [0.1, 1.2, 0.5]))
    assert not _same_place(a, moved)
    assert _place_id(a) != _place_id(moved)


def test_unavailable_geometry_fails_closed():
    """<2 stable anchors -> cannot confirm same room -> own place (never a false merge)."""
    one = _anchors(("tv", [0.4, 0.8, 1.85]))
    none = _anchors(("mouse", [0.1, 0.1, 0.1]))  # transient label -> excluded
    assert _place_id(one) is None
    assert len(none) == 0
    assert not _same_place(one, _anchors(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3])))


def test_records_vector_free_and_zero_authority(tmp_path):
    album = tmp_path / "episodic"
    album.mkdir()
    rec = {
        "session_id": "s1", "calibration_version": 7, "captured_ts": 100.0,
        "world": _world(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3])),
    }
    (album / "s1.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = SpatialPlaceConsolidator(album).consolidate()

    assert out["sessions"] == 1 and out["places"] == 1
    assert out["status"] == STATUS == "PRE-MATURE"
    assert out["lane"] == LANE
    assert out["authority"]["writes_memory"] is False
    assert out["authority"]["influences_policy"] is False

    blob = json.dumps(out)
    for vk in _VECTOR_KEYS:
        assert f'"{vk}"' not in blob, f"vector key {vk} leaked into output"

    for place in out["place_records"]:
        assert place["record_kind"] == "place_consolidated"
        assert place["loaded_from_store"] is True  # aggregate, never a live observation
        assert all(v is False for k, v in place["authority"].items() if k != "no_raw_vectors_in_api")


def test_consolidate_groups_same_room_across_calibrations(tmp_path):
    """Two sessions of the same room under different calibrations -> ONE place."""
    album = tmp_path / "episodic"
    album.mkdir()
    s1 = {
        "session_id": "s1", "calibration_version": 1, "captured_ts": 100.0,
        "world": _world(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3]), ("keyboard", [0.1, 1.2, 0.5])),
    }
    s2 = {  # same room, frame re-origin +10 (recalibration)
        "session_id": "s2", "calibration_version": 2, "captured_ts": 200.0,
        "world": _world(("tv", [10.4, 10.8, 11.85]), ("chair", [10.0, 11.5, 10.3]), ("keyboard", [10.1, 11.2, 10.5])),
    }
    (album / "s1.jsonl").write_text(json.dumps(s1) + "\n", encoding="utf-8")
    (album / "s2.jsonl").write_text(json.dumps(s2) + "\n", encoding="utf-8")
    out = SpatialPlaceConsolidator(album).consolidate()

    assert out["sessions"] == 2
    assert out["places"] == 1, "same room across calibrations must consolidate to one place"
    place = out["place_records"][0]
    assert place["member_count"] == 2
    assert place["calibration_versions"] == [1, 2]
    assert out["compression_ratio"] == 2.0


def test_empty_album_is_safe(tmp_path):
    out = SpatialPlaceConsolidator(tmp_path / "nonexistent").consolidate()
    assert out["sessions"] == 0 and out["places"] == 0
    assert out["compression_ratio"] == 0.0


def test_key_strength_honesty(tmp_path):
    """Place keys are labeled by geometric confidence: >=3 anchors = strong, 2 = weak (one pair)."""
    album = tmp_path / "episodic"
    album.mkdir()
    strong = {  # 3 anchors -> >=2 inter-anchor pairs -> strong
        "session_id": "strong", "calibration_version": 1, "captured_ts": 100.0,
        "world": _world(("tv", [0.4, 0.8, 1.85]), ("chair", [0.0, 1.5, 0.3]), ("keyboard", [0.1, 1.2, 0.5])),
    }
    weak = {  # different room, 2 anchors -> a single distance -> weak_single_pair
        "session_id": "weak", "calibration_version": 1, "captured_ts": 200.0,
        "world": _world(("tv", [5.0, 5.0, 5.0]), ("desk", [6.0, 6.0, 6.0])),
    }
    (album / "strong.jsonl").write_text(json.dumps(strong) + "\n", encoding="utf-8")
    (album / "weak.jsonl").write_text(json.dumps(weak) + "\n", encoding="utf-8")
    out = SpatialPlaceConsolidator(album).consolidate()

    assert out["places"] == 2  # different rooms, not merged
    assert sorted(p["key_strength"] for p in out["place_records"]) == ["strong", "weak_single_pair"]
    assert out["places_keyed_strong"] == 1
    assert out["places_keyed_weak_single_pair"] == 1
    assert out["no_raw_vectors_in_api"] is True
