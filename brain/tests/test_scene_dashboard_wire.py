"""Pins the 2026-08-24 /api/scene zeros: tracker never seeded, persons dropped.

Lived: Hailo sees a person ~0.9 at 15fps; GPU caption names monitors/keyboard/desk;
/api/scene stays update_count=0. Causes (not HRR):
  1. empty Pi scene_summary (persons stripped) returned before person_bboxes stored
  2. VLM enrichment required an existing snapshot (chicken-egg with 1)
  3. whole-token parse missed plurals ("monitors" != "monitor")
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from perception.scene_tracker import SceneTracker
from perception_orchestrator import PerceptionOrchestrator
from dashboard.snapshot import _decorate_scene_payload


_NAVY_CAPTION = (
    'A person wearing a gray "U.S. NAVY" t-shirt is seated at a desk, '
    "facing three computer monitors. The desk holds a keyboard, mouse, "
    "and other peripherals."
)


def _stub_orch() -> PerceptionOrchestrator:
    orch = PerceptionOrchestrator.__new__(PerceptionOrchestrator)
    orch._scene_tracker = SceneTracker()
    orch._last_scene_snapshot = None
    orch._last_person_bboxes = []
    orch._last_scene_description = ""
    orch._last_scene_source = ""
    orch._last_scene_ts = 0.0
    orch._object_memory = {}
    orch._scene_analysis_in_progress = False
    orch._last_scene_analysis_time = 0.0
    orch._loop = None
    orch._display_classifier = SimpleNamespace(
        classify_from_description=lambda *a, **k: [],
    )
    return orch


def test_empty_scene_summary_still_records_person_bboxes() -> None:
    orch = _stub_orch()
    orch._on_scene_summary(
        detections=[],
        frame_size=[640, 480],
        scene_change_score=0.0,
        person_bboxes=[[80, 40, 400, 470]],
    )
    assert orch.get_person_bboxes() == [(80, 40, 400, 470)]
    snap = orch._last_scene_snapshot
    assert snap is not None
    assert snap.entities == []
    assert snap.region_visibility
    assert any(v < 1.0 for v in snap.region_visibility.values())


def test_vlm_feed_seeds_tracker_without_prior_snapshot() -> None:
    orch = _stub_orch()
    orch._feed_vlm_to_tracker(_NAVY_CAPTION)
    snap = orch.get_scene_snapshot()
    assert snap is not None
    labels = {e.label for e in snap.entities}
    assert {"desk", "monitor", "keyboard", "mouse"} <= labels
    assert all(e.source == "vlm" for e in snap.entities)
    assert orch._scene_tracker._update_count == 1


def test_scene_payload_exposes_caption_and_person_count() -> None:
    perc = SimpleNamespace(
        get_scene_caption_state=lambda: {
            "text": "a keyboard on a desk",
            "source": "desktop_gpu",
            "age_s": 4.2,
            "edge_active": False,
        },
        get_person_bboxes=lambda: [(1, 2, 3, 4)],
    )
    out = _decorate_scene_payload({"update_count": 0, "entities": []}, perc)
    assert out["caption"]["text"].startswith("a keyboard")
    assert out["person_bbox_count"] == 1
    assert out["update_count"] == 0


def test_pi_disconnect_drops_person_present_latch() -> None:
    src = Path(__file__).resolve().parents[2] / "pi" / "main.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("def _on_brain_disconnect")
    end = text.index("\n    def ", start + 1)
    body = text[start:end]
    assert "self._was_person_present = False" in body


def test_empty_caption_uses_short_interval_even_when_present() -> None:
    orch = _stub_orch()
    orch._scene_interval_away = 300.0
    orch._scene_interval_present = 1800.0
    orch._last_scene_description = ""
    assert orch._scene_analysis_interval(user_here=True) == 300.0
    orch._last_scene_description = "three monitors on a desk"
    assert orch._scene_analysis_interval(user_here=True) == 1800.0
    assert orch._scene_analysis_interval(user_here=False) == 300.0


def test_scene_context_keeps_caption_and_candidate_objects() -> None:
    orch = _stub_orch()
    orch.ambient_proc = None
    orch.presence = None
    orch._last_scene_description = _NAVY_CAPTION
    orch._feed_vlm_to_tracker(_NAVY_CAPTION)
    ctx = orch.get_scene_context()
    assert "Visual:" in ctx
    assert "monitor" in ctx or "Displays:" in ctx
    assert "desk" in ctx or "keyboard" in ctx


def test_ingest_scene_description_is_brain_side_room_inventory() -> None:
    orch = _stub_orch()
    orch._object_memory = {}
    orch.ingest_scene_description(_NAVY_CAPTION, source="desktop_gpu")
    assert orch._last_scene_source == "desktop_gpu"
    assert "monitors" in orch._last_scene_description.lower()
    labels = {e.label for e in orch.get_scene_snapshot().entities}
    assert {"desk", "monitor", "keyboard", "mouse"} <= labels
    assert "desk" in orch._object_memory
    assert "chair" not in orch._object_memory  # caption does not name a chair


def test_empty_hailo_summary_does_not_decay_vlm_objects() -> None:
    """Pi stays person-only. Empty object lists must not wipe brain VLM inventory."""
    orch = _stub_orch()
    orch._feed_vlm_to_tracker(_NAVY_CAPTION)
    labels_before = {e.label for e in orch.get_scene_snapshot().entities}
    orch._on_scene_summary(
        detections=[],
        frame_size=[640, 480],
        scene_change_score=0.0,
        person_bboxes=[[80, 40, 400, 470]],
    )
    snap = orch.get_scene_snapshot()
    labels_after = {e.label for e in snap.entities}
    assert labels_before <= labels_after
    assert snap.region_visibility
    assert any(v < 1.0 for v in snap.region_visibility.values())


def test_empty_summary_schedules_brain_first_look_not_pi_yolo() -> None:
    orch = _stub_orch()
    orch._scene_analysis_in_progress = False
    orch._last_scene_analysis_time = 0.0
    orch._loop = None  # no event loop in unit test — must not raise
    orch._on_scene_summary(detections=[], person_bboxes=[[10, 10, 100, 200]])
    src = Path(__file__).resolve().parent.parent / "perception_orchestrator.py"
    text = src.read_text(encoding="utf-8")
    assert "Room inventory is a BRAIN VLM read" in text
    assert "_maybe_request_first_look" in text
    assert "refresh_person_occlusion" in text
    assert "asyncio.wait_for" in text
    assert "_scene_analyze_timeout_s" in text
    assert '"timeout"' in text


def test_pi_snapshot_copies_frame_and_supports_fresh_grab() -> None:
    """Lived 2026-08-25: /snapshot served identical JPEGs while Hailo still ran."""
    root = Path(__file__).resolve().parents[2]
    snap = (root / "pi" / "main.py").read_text(encoding="utf-8")
    start = snap.index("async def snapshot_handler")
    end = snap.index("\n    async def ", start + 1)
    body = snap[start:end]
    assert "grab" in body
    assert "capture_frame" in body
    assert "X-Frame-Age-Ms" in body
    assert "no-store" in body
    det = (root / "pi" / "senses" / "vision" / "detector.py").read_text(encoding="utf-8")
    assert "frame.copy()" in det
    assert "def _keep_frame" in det
