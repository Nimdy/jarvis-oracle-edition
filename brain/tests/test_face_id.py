"""Face ID known-flag: 0.55 is a crop cosine floor, not an EMA veto.

Lived 2026-08-28 office (corner camera, ~1 crop/min, desk ≠ lens):
  Face ID: face_392 (raw=0.689, smoothed=0.501, known=False)
Gallery was David (15 crops, blend sim=0.84). Enroll lane LIVE/WIRED.
known never fired because SCORE_EMA mixed desk junk 0.04–0.43 into the
smoother, then the 0.55 floor was applied to smoothed instead of the crop.

Do not lower SIMILARITY_THRESHOLD.
"""

from __future__ import annotations

import collections
import math
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from perception.face_id import FaceIdentifier


def _unit_vec(dim: int = 128, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _with_cosine(base: np.ndarray, cosine: float, seed: int = 7) -> np.ndarray:
    other = _unit_vec(dim=len(base), seed=seed)
    other = other - float(np.dot(other, base)) * base
    other = other / (np.linalg.norm(other) + 1e-8)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    v = cosine * base + math.sqrt(max(0.0, 1.0 - cosine * cosine)) * other
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _make_face_id(profiles=None, ema=None) -> FaceIdentifier:
    fid = FaceIdentifier.__new__(FaceIdentifier)
    fid._lock = threading.Lock()
    fid._session = MagicMock()
    fid.available = True
    fid._profiles = {}
    fid._next_unknown_id = 1
    fid._score_ema = dict(ema or {})
    fid._recent_crops_b64 = collections.deque(maxlen=5)
    fid._input_name = "input"
    fid._input_shape = (1, 3, 112, 112)
    fid._persist_dir = Path("/tmp")
    fid._profiles_path = Path("/tmp/face_profiles.json")
    if profiles:
        for name, emb in profiles.items():
            fid._profiles[name] = {
                "embedding": np.array(emb, dtype=np.float32),
                "registered": 0,
                "last_seen": 0,
                "interaction_count": 0,
                "enrollment_crops": 3,
            }
    return fid


def _identify_with(fid: FaceIdentifier, embedding: np.ndarray) -> dict:
    crop = np.zeros((112, 112, 3), dtype=np.uint8)
    with patch.object(fid, "_extract_embedding", return_value=embedding):
        with patch.object(fid, "_save_profiles"):
            return fid.identify(crop)


class TestFaceKnownUsesCropFloor:
    def test_threshold_is_not_lowered(self):
        assert FaceIdentifier.SIMILARITY_THRESHOLD == 0.55

    def test_lived_raw_069_is_known_despite_poisoned_ema(self):
        """10:19:55 raw=0.689 smoothed=0.501 known=False — must become known."""
        profile = _unit_vec(seed=1)
        fid = _make_face_id({"David": profile}, ema={"David": 0.40})
        result = _identify_with(fid, _with_cosine(profile, 0.689, seed=11))

        assert result["closest_match"] == "David"
        assert result["raw_score"] == pytest.approx(0.689, abs=0.02)
        assert result["is_known"] is True
        assert result["name"] == "David"

    def test_raw_just_over_floor_is_known(self):
        profile = _unit_vec(seed=2)
        fid = _make_face_id({"David": profile}, ema={"David": 0.30})
        result = _identify_with(fid, _with_cosine(profile, 0.56, seed=12))
        assert result["is_known"] is True
        assert result["name"] == "David"

    def test_raw_under_floor_with_low_ema_is_unknown(self):
        profile = _unit_vec(seed=3)
        fid = _make_face_id({"David": profile}, ema={"David": 0.30})
        result = _identify_with(fid, _with_cosine(profile, 0.50, seed=13))
        assert result["is_known"] is False
        assert result["closest_match"] == "David"
        assert result["name"].startswith("face_")


class TestDeskJunkDoesNotPoisonOrLock:
    def test_desk_crop_does_not_write_ema(self):
        profile = _unit_vec(seed=4)
        fid = _make_face_id({"David": profile}, ema={"David": 0.60})
        result = _identify_with(fid, _with_cosine(profile, 0.04, seed=14))
        assert fid._score_ema["David"] == 0.60
        assert result["is_known"] is False

    def test_high_ema_does_not_declare_known_on_desk_junk(self):
        """Persist/solo-keep carry identity. A 0.04 crop is not a face lock."""
        profile = _unit_vec(seed=5)
        fid = _make_face_id({"David": profile}, ema={"David": 0.70})
        result = _identify_with(fid, _with_cosine(profile, 0.04, seed=15))
        assert result["is_known"] is False

    def test_plausible_dip_can_stay_known_via_ema(self):
        profile = _unit_vec(seed=6)
        fid = _make_face_id({"David": profile}, ema={"David": 0.70})
        result = _identify_with(fid, _with_cosine(profile, 0.48, seed=16))
        assert result["is_known"] is True
        assert result["name"] == "David"

    def test_lived_morning_sequence_locks_on_look_not_desk(self):
        """Replay 10:14–10:27: desk junk, then a 0.69 look, then sit-down 0.04."""
        profile = _unit_vec(seed=8)
        fid = _make_face_id({"David": profile}, ema={"David": 0.30})
        desk = _identify_with(fid, _with_cosine(profile, 0.043, seed=20))
        assert desk["is_known"] is False
        ema_after_desk = fid._score_ema["David"]
        look = _identify_with(fid, _with_cosine(profile, 0.689, seed=21))
        assert look["is_known"] is True
        assert look["name"] == "David"
        ema_after_look = fid._score_ema["David"]
        sit = _identify_with(fid, _with_cosine(profile, 0.042, seed=22))
        assert sit["is_known"] is False
        assert fid._score_ema["David"] == ema_after_look
        assert ema_after_look >= ema_after_desk
