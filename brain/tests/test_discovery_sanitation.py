"""Tests for capability discovery phrase sanitation — rejects emotional/complaint text."""

import pytest
from skills.discovery import (
    is_actionable_capability_phrase,
    CapabilityFamilyNormalizer,
    _REJECTED_FAMILY,
)


class TestIsActionableCapabilityPhrase:
    """The filter must reject emotional/complaint/vague text."""

    def test_reject_emotional_complaint(self):
        assert not is_actionable_capability_phrase("better serve you than to operate in the shadows")

    def test_reject_stance_language(self):
        assert not is_actionable_capability_phrase("you should be more transparent")

    def test_reject_dont_do_that(self):
        assert not is_actionable_capability_phrase("don't do that again")

    def test_reject_i_dont_like(self):
        assert not is_actionable_capability_phrase("i don't like what you did")

    def test_reject_better_serve(self):
        assert not is_actionable_capability_phrase("better serve me")

    def test_reject_single_word(self):
        assert not is_actionable_capability_phrase("useless")

    def test_reject_affective_cluster(self):
        assert not is_actionable_capability_phrase("terrible useless waste")

    def test_reject_stop_doing(self):
        assert not is_actionable_capability_phrase("stop doing that")

    def test_accept_sing_a_song(self):
        assert is_actionable_capability_phrase("sing a song")

    def test_accept_draw_image(self):
        assert is_actionable_capability_phrase("draw an image")

    def test_accept_control_camera(self):
        assert is_actionable_capability_phrase("control the camera")

    def test_accept_detect_faces(self):
        assert is_actionable_capability_phrase("detect faces in the image")

    def test_accept_translate_text(self):
        assert is_actionable_capability_phrase("translate this document")

    def test_accept_process_data(self):
        assert is_actionable_capability_phrase("process csv data")

    def test_accept_zoom_camera(self):
        assert is_actionable_capability_phrase("zoom the camera in")

    def test_reject_who_told_you(self):
        assert not is_actionable_capability_phrase("who told you to do that")


class TestNormalizerRejectsJunk:
    """The normalizer should return _REJECTED_FAMILY for non-actionable text."""

    def test_emotional_complaint_returns_rejected(self):
        norm = CapabilityFamilyNormalizer()
        fam = norm.normalize(None, "better serve you than to operate in the shadows")
        assert fam.family_id == "_rejected"

    def test_stance_returns_rejected(self):
        norm = CapabilityFamilyNormalizer()
        fam = norm.normalize(None, "you should be more honest with me")
        assert fam.family_id == "_rejected"

    def test_builtin_not_rejected(self):
        norm = CapabilityFamilyNormalizer()
        fam = norm.normalize("camera_control", "zoom the camera")
        assert fam.family_id == "camera_control"

    def test_valid_skill_creates_dynamic(self):
        norm = CapabilityFamilyNormalizer()
        fam = norm.normalize(None, "sing a beautiful song")
        assert fam.family_id != "_rejected"

    def test_rejected_family_sentinel(self):
        assert _REJECTED_FAMILY.family_id == "_rejected"
        assert _REJECTED_FAMILY.domain == "rejected"
