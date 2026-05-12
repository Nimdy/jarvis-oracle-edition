"""Layer 3B Scene Tracker — Unit Tests.

Tests entity matching, state machine transitions, permanence decay,
region-aware behavior, display surface masking, and eviction.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from perception.scene_types import SceneDetection, DISPLAY_SURFACE_LABELS
from perception.scene_tracker import (
    SceneTracker,
    PROMOTION_STABLE_CYCLES,
    REMOVAL_MISSING_CYCLES,
    MAX_ENTITIES,
    REMOVED_RETENTION_CYCLES,
)
from perception.scene_regions import infer_region, estimate_region_visibility


# ── Entity matching ──────────────────────────────────────────────────────────

def test_new_detection_creates_candidate():
    """First time seeing an object creates a candidate entity."""
    tracker = SceneTracker()
    dets = [SceneDetection("cup", 0.9, (400, 500, 450, 600))]
    snap = tracker.update(dets, 1920, 1080)
    assert len(snap.entities) == 1
    assert snap.entities[0].state == "candidate"
    assert snap.entities[0].label == "cup"


def test_repeated_detection_promotes_to_visible():
    """After PROMOTION_STABLE_CYCLES, entity promotes to visible."""
    tracker = SceneTracker()
    det = SceneDetection("keyboard", 0.85, (300, 700, 600, 750))
    for i in range(PROMOTION_STABLE_CYCLES):
        snap = tracker.update([det], 1920, 1080)

    visible = [e for e in snap.entities if e.state == "visible"]
    assert len(visible) >= 1
    assert visible[0].label == "keyboard"


def test_promotion_delta_emitted():
    """Entity promotion emits an entity_promoted delta."""
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (400, 500, 450, 600))
    all_promo_deltas = []
    for _ in range(PROMOTION_STABLE_CYCLES + 1):
        snap = tracker.update([det], 1920, 1080)
        all_promo_deltas.extend(d for d in snap.deltas if d.event == "entity_promoted")
    assert len(all_promo_deltas) >= 1
    assert all_promo_deltas[0].label == "cup"


def test_same_class_same_region_matches():
    """Two detections of same class in overlapping regions match the same entity."""
    tracker = SceneTracker()
    snap1 = tracker.update(
        [SceneDetection("cup", 0.9, (400, 500, 450, 600))], 1920, 1080,
    )
    snap2 = tracker.update(
        [SceneDetection("cup", 0.85, (405, 505, 455, 605))], 1920, 1080,
    )
    assert len(snap2.entities) == 1


def test_different_class_creates_separate_entities():
    """Detections of different classes create separate entities."""
    tracker = SceneTracker()
    dets = [
        SceneDetection("cup", 0.9, (400, 500, 450, 600)),
        SceneDetection("keyboard", 0.8, (300, 700, 600, 750)),
    ]
    snap = tracker.update(dets, 1920, 1080)
    labels = {e.label for e in snap.entities}
    assert "cup" in labels
    assert "keyboard" in labels
    assert len(snap.entities) == 2


# ── State transitions ────────────────────────────────────────────────────────

def test_unmatched_entity_decays_to_missing_in_visible_region():
    """Entity not matched in a visible region transitions to missing."""
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (1500, 700, 1550, 800))
    for _ in range(PROMOTION_STABLE_CYCLES):
        tracker.update([det], 1920, 1080)

    snap = tracker.update([], 1920, 1080)
    cup = [e for e in snap.entities if e.label == "cup"][0]
    assert cup.state in ("missing", "occluded")


def test_entity_removed_after_enough_missing_cycles():
    """Entity transitions to removed after REMOVAL_MISSING_CYCLES in visible region."""
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (1500, 700, 1550, 800))
    for _ in range(PROMOTION_STABLE_CYCLES):
        tracker.update([det], 1920, 1080)

    for _ in range(REMOVAL_MISSING_CYCLES + 1):
        snap = tracker.update([], 1920, 1080)

    cup = [e for e in snap.entities if e.label == "cup"]
    if cup:
        assert cup[0].state == "removed"


def test_removed_entity_evicted_after_bounded_retention():
    """Removed objects stay briefly for history, then leave live snapshots."""
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (1500, 700, 1550, 800))
    for _ in range(PROMOTION_STABLE_CYCLES):
        tracker.update([det], 1920, 1080)

    for _ in range(REMOVAL_MISSING_CYCLES):
        snap = tracker.update([], 1920, 1080)

    cups = [e for e in snap.entities if e.label == "cup"]
    assert cups and cups[0].state == "removed"

    for _ in range(REMOVED_RETENTION_CYCLES + 1):
        snap = tracker.update([], 1920, 1080)

    assert [e for e in snap.entities if e.label == "cup"] == []


def test_removed_history_does_not_count_as_active_entity():
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (1500, 700, 1550, 800))
    for _ in range(PROMOTION_STABLE_CYCLES):
        tracker.update([det], 1920, 1080)

    for _ in range(REMOVAL_MISSING_CYCLES):
        snap = tracker.update([], 1920, 1080)

    d = snap.to_dict()
    assert d["entity_count"] == 1
    assert d["active_entity_count"] == 0
    assert d["removed_entity_count"] == 1


def test_removal_delta_emitted():
    """Entity removal emits an entity_removed or entity_missing delta across decay cycles."""
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (1500, 700, 1550, 800))
    for _ in range(PROMOTION_STABLE_CYCLES):
        tracker.update([det], 1920, 1080)

    all_delta_events: set[str] = set()
    for _ in range(REMOVAL_MISSING_CYCLES + 2):
        snap = tracker.update([], 1920, 1080)
        for d in snap.deltas:
            all_delta_events.add(d.event)

    assert "entity_removed" in all_delta_events or "entity_missing" in all_delta_events


# ── Permanence confidence ────────────────────────────────────────────────────

def test_permanence_grows_on_repeated_observation():
    """Permanence confidence increases when entity is repeatedly seen."""
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (400, 500, 450, 600))
    snap1 = tracker.update([det], 1920, 1080)
    p1 = snap1.entities[0].permanence_confidence

    snap2 = tracker.update([det], 1920, 1080)
    p2 = snap2.entities[0].permanence_confidence
    assert p2 > p1


def test_permanence_decays_when_unseen():
    """Permanence confidence decreases when entity is not seen."""
    tracker = SceneTracker()
    det = SceneDetection("cup", 0.9, (1500, 700, 1550, 800))
    for _ in range(PROMOTION_STABLE_CYCLES):
        tracker.update([det], 1920, 1080)

    snap_before = tracker.update([det], 1920, 1080)
    p_before = snap_before.entities[0].permanence_confidence

    snap_after = tracker.update([], 1920, 1080)
    cups = [e for e in snap_after.entities if e.label == "cup"]
    assert len(cups) == 1
    assert cups[0].permanence_confidence < p_before


# ── Region-aware decay ───────────────────────────────────────────────────────

def test_occluded_region_slows_decay():
    """Entity in heavily occluded region decays slower than in visible-empty region."""
    tracker1 = SceneTracker()
    tracker2 = SceneTracker()
    det = SceneDetection("cup", 0.9, (400, 700, 450, 750))

    for _ in range(PROMOTION_STABLE_CYCLES):
        tracker1.update([det], 1920, 1080)
        tracker2.update([det], 1920, 1080)

    person_bbox = [(200, 300, 800, 1080)]
    snap1 = tracker1.update([], 1920, 1080, person_bboxes=person_bbox)
    snap2 = tracker2.update([], 1920, 1080, person_bboxes=[])

    cups1 = [e for e in snap1.entities if e.label == "cup"]
    cups2 = [e for e in snap2.entities if e.label == "cup"]
    assert len(cups1) == 1 and len(cups2) == 1
    assert cups1[0].permanence_confidence >= cups2[0].permanence_confidence


# ── Display surface masking ──────────────────────────────────────────────────

def test_monitor_is_display_surface():
    """Monitor/TV/laptop detections are flagged as display surfaces."""
    for label in DISPLAY_SURFACE_LABELS:
        tracker = SceneTracker()
        dets = [SceneDetection(label, 0.9, (200, 100, 900, 600))]
        snap = tracker.update(dets, 1920, 1080)
        surf = [e for e in snap.entities if e.label == label]
        assert len(surf) >= 1, f"Expected entity for {label}"
        assert surf[0].is_display_surface, f"{label} should be display surface"


def test_display_surfaces_tracked():
    """Display surfaces are maintained in the display_surfaces registry."""
    tracker = SceneTracker()
    dets = [SceneDetection("monitor", 0.9, (200, 100, 900, 600))]
    snap = tracker.update(dets, 1920, 1080)
    assert len(snap.display_surfaces) >= 1
    assert snap.display_surfaces[0].kind == "monitor"


def test_display_surface_cross_label_matching():
    """A 'tv' detection should match an existing 'monitor' display surface."""
    tracker = SceneTracker()
    monitor = SceneDetection("monitor", 0.9, (200, 100, 900, 600))
    tracker.update([monitor], 1920, 1080)
    assert len(tracker._display_surfaces) == 1

    tv = SceneDetection("tv", 0.85, (210, 110, 895, 595))
    tracker.update([tv], 1920, 1080)
    assert len(tracker._display_surfaces) == 1, (
        "Cross-label 'tv' should match existing 'monitor' surface"
    )


def test_display_surface_eviction():
    """Stale display surfaces are evicted after enough unseen cycles."""
    from perception.scene_tracker import DISPLAY_SURFACE_STALE_CYCLES
    tracker = SceneTracker()
    monitor = SceneDetection("monitor", 0.9, (200, 100, 900, 600))
    tracker.update([monitor], 1920, 1080)
    assert len(tracker._display_surfaces) == 1

    for _ in range(DISPLAY_SURFACE_STALE_CYCLES + 1):
        tracker.update([], 1920, 1080)
    assert len(tracker._display_surfaces) == 0, (
        "Display surface should be evicted after enough unseen cycles"
    )


def test_objects_inside_display_bbox_are_masked():
    """Objects detected inside a known display surface are not physical entities."""
    tracker = SceneTracker()
    monitor = SceneDetection("monitor", 0.95, (200, 100, 900, 600))
    tracker.update([monitor], 1920, 1080)

    cup_inside = SceneDetection("cup", 0.8, (400, 300, 450, 400))
    cup_outside = SceneDetection("cup", 0.8, (1200, 700, 1250, 800))
    snap = tracker.update([monitor, cup_inside, cup_outside], 1920, 1080)

    physical_cups = [e for e in snap.entities
                     if e.label == "cup" and not e.is_display_surface]
    assert len(physical_cups) == 1


# ── Eviction ─────────────────────────────────────────────────────────────────

def test_max_entities_eviction():
    """Tracker caps entity count at MAX_ENTITIES."""
    tracker = SceneTracker()
    for i in range(MAX_ENTITIES + 20):
        x1 = i * 30
        dets = [SceneDetection(f"obj_{i}", 0.5, (x1, 700, x1 + 20, 750))]
        tracker.update(dets, 1920, 1080)
        tracker.update([], 1920, 1080)
        tracker.update([], 1920, 1080)
        tracker.update([], 1920, 1080)
        tracker.update([], 1920, 1080)
        tracker.update([], 1920, 1080)

    assert len(tracker._entities) <= MAX_ENTITIES + 5


# ── Region inference ─────────────────────────────────────────────────────────

def test_infer_region_desk_left():
    assert infer_region((50, 700, 100, 800), 1920, 1080) == "desk_left"


def test_infer_region_monitor_zone():
    assert infer_region((500, 400, 600, 500), 1920, 1080) == "monitor_zone"


def test_infer_region_background():
    assert infer_region((500, 50, 600, 150), 1920, 1080) == "background"


def test_infer_region_none_bbox():
    assert infer_region(None, 1920, 1080) == "unknown"


# ── Region visibility ────────────────────────────────────────────────────────

def test_no_person_full_visibility():
    vis = estimate_region_visibility([], 1920, 1080)
    assert all(v == 1.0 for v in vis.values())


def test_person_reduces_visibility():
    person = [(0, 0, 960, 1080)]
    vis = estimate_region_visibility(person, 1920, 1080)
    assert vis["desk_left"] < 1.0
    assert vis["desk_right"] >= vis["desk_left"]


# ── Snapshot structure ───────────────────────────────────────────────────────

def test_snapshot_to_dict_has_required_keys():
    tracker = SceneTracker()
    snap = tracker.update(
        [SceneDetection("cup", 0.9, (400, 700, 450, 800))], 1920, 1080,
    )
    d = snap.to_dict()
    assert "entity_count" in d
    assert "active_entity_count" in d
    assert "removed_entity_count" in d
    assert "visible_count" in d
    assert "stable_count" in d
    assert "entities" in d
    assert "deltas" in d
    assert "display_surfaces" in d
    assert "region_visibility" in d


def test_get_state_matches_snapshot():
    tracker = SceneTracker()
    det = SceneDetection("keyboard", 0.85, (300, 700, 600, 750))
    tracker.update([det], 1920, 1080)
    state = tracker.get_state()
    assert "entity_count" in state
    assert "entities" in state
    assert "recent_deltas" in state


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
