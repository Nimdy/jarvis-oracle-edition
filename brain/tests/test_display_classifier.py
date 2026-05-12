"""Layer 3B Display Classifier — Unit Tests.

Tests content type classification from VLM descriptions,
activity label mapping, and display surface interaction.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from perception.display_classifier import DisplayClassifier
from perception.scene_types import DisplaySurface


def _make_surface(sid="monitor_1", kind="monitor"):
    return DisplaySurface(
        surface_id=sid, kind=kind, bbox=(200, 100, 900, 600),
        confidence=0.9, first_seen_ts=0, last_seen_ts=0,
    )


# ── Content type classification ──────────────────────────────────────────────

def test_game_detection():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "The screen shows a first-person shooter with a crosshair and minimap visible.",
        [_make_surface()],
    )
    assert len(results) == 1
    assert results[0].content_type == "game"
    assert results[0].activity_label == "gaming"
    assert results[0].confidence > 0.3


def test_code_editor_detection():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "A code editor with syntax highlighting is open, showing Python functions and imports.",
        [_make_surface()],
    )
    assert len(results) == 1
    assert results[0].content_type == "code_editor"
    assert results[0].activity_label == "coding"


def test_terminal_detection():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "A terminal window with a bash shell prompt showing command line output.",
        [_make_surface()],
    )
    assert len(results) == 1
    assert results[0].content_type == "terminal"
    assert results[0].activity_label == "coding"


def test_browser_detection():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "A Chrome browser window showing a website with multiple tabs open.",
        [_make_surface()],
    )
    assert len(results) == 1
    assert results[0].content_type == "browser"
    assert results[0].activity_label == "browsing"


def test_video_detection():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "A YouTube video is playing with playback controls and subtitles visible.",
        [_make_surface()],
    )
    assert len(results) == 1
    assert results[0].content_type == "video"
    assert results[0].activity_label == "watching"


def test_unknown_content():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "A blank white screen with nothing visible.",
        [_make_surface()],
    )
    assert len(results) == 1
    assert results[0].content_type == "unknown"


def test_empty_description():
    dc = DisplayClassifier()
    results = dc.classify_from_description("", [_make_surface()])
    assert len(results) == 0


# ── Multi-surface ────────────────────────────────────────────────────────────

def test_multiple_surfaces_get_same_classification():
    dc = DisplayClassifier()
    surfaces = [
        _make_surface("mon_1", "monitor"),
        _make_surface("mon_2", "monitor"),
    ]
    results = dc.classify_from_description(
        "The monitor shows a game with HUD elements.",
        surfaces,
    )
    assert len(results) == 2
    assert all(r.content_type == "game" for r in results)


# ── No surface fallback ─────────────────────────────────────────────────────

def test_no_surface_still_classifies_if_strong_signal():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "A game with crosshair and minimap and HUD bars is visible.",
        [],
    )
    assert len(results) == 1
    assert results[0].content_type == "game"
    assert results[0].surface_id == "inferred"
    assert results[0].confidence < 0.9


def test_no_surface_no_signal_returns_empty():
    dc = DisplayClassifier()
    results = dc.classify_from_description("A blank area.", [])
    assert len(results) == 0


# ── Confidence scoring ───────────────────────────────────────────────────────

def test_more_pattern_hits_higher_confidence():
    dc = DisplayClassifier()
    weak = dc.classify_from_description(
        "Something that looks like a game.", [_make_surface()],
    )
    strong = dc.classify_from_description(
        "An FPS game with crosshair, minimap, HUD bars, and ammo counter visible.",
        [_make_surface()],
    )
    assert len(weak) == 1 and len(strong) == 1
    assert strong[0].confidence >= weak[0].confidence


# ── Semantic summary ─────────────────────────────────────────────────────────

def test_semantic_summary_populated():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "The user's monitor shows a code editor with syntax highlighting.",
        [_make_surface()],
    )
    assert len(results) == 1
    assert results[0].semantic_summary != ""
    assert "code" in results[0].semantic_summary.lower()


# ── Last classifications cache ───────────────────────────────────────────────

def test_last_classifications_stored():
    dc = DisplayClassifier()
    dc.classify_from_description(
        "A game with crosshair visible.", [_make_surface()],
    )
    cached = dc.get_last_classifications()
    assert "monitor_1" in cached
    assert cached["monitor_1"].content_type == "game"


def test_to_dict_format():
    dc = DisplayClassifier()
    results = dc.classify_from_description(
        "Code editor with syntax highlighting.", [_make_surface()],
    )
    d = results[0].to_dict()
    assert "surface_id" in d
    assert "content_type" in d
    assert "confidence" in d
    assert "activity_label" in d


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
