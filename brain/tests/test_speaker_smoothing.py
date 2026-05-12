"""Tests for speaker_id.py voice score smoothing (Wave 1: Identity Hardening).

Validates that SpeakerIdentifier uses EMA-smoothed scores for known/unknown
decisions, matching the face_id pattern. Tests cover: pre-decision smoothing,
dampening of outliers, EMA clearing on enrollment/removal/merge, raw_score
in return dict, and the sub-threshold adaptive path using raw scores.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def _make_speaker_id(profiles=None):
    """Create a SpeakerIdentifier with mocked model and optional pre-set profiles."""
    with patch("perception.speaker_id.SPEECHBRAIN_AVAILABLE", True):
        with patch("perception.speaker_id.SpeakerIdentifier._load_ecapa_model"):
            from perception.speaker_id import SpeakerIdentifier
            sid = SpeakerIdentifier.__new__(SpeakerIdentifier)
            sid._lock = __import__("threading").Lock()
            sid._model = MagicMock()
            sid.available = True
            sid._profiles = {}
            sid._next_unknown_id = 1
            sid._last_embedding = None
            sid._score_ema = {}
            sid._profiles_path = MagicMock()
            if profiles:
                for name, emb in profiles.items():
                    sid._profiles[name] = {
                        "embedding": np.array(emb, dtype=np.float32),
                        "registered": 0,
                        "last_seen": 0,
                        "interaction_count": 0,
                        "enrollment_clips": 1,
                    }
            return sid


def _unit_vec(dim=192, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestVoiceSmoothingPreDecision:
    """EMA smoothing happens before the known/unknown threshold check."""

    def test_smoothed_score_used_for_decision(self):
        """A raw score below threshold can still produce is_known=True via EMA."""
        profile_emb = _unit_vec(seed=1)
        sid = _make_speaker_id({"alice": profile_emb})

        sid._model.encode_batch = MagicMock(
            return_value=MagicMock(squeeze=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=profile_emb.copy())
                ))
            )))
        )

        # First call: raw=1.0 (perfect match), smoothed=1.0, seeds EMA
        r1 = sid.identify(np.zeros(16000, dtype=np.float32))
        assert r1["is_known"] is True
        assert r1["confidence"] > 0.9

        # Now create a slightly different embedding that scores 0.48 raw (below 0.50 threshold)
        noisy = profile_emb * 0.48 + _unit_vec(seed=99) * 0.52
        noisy = noisy / np.linalg.norm(noisy)
        sid._model.encode_batch = MagicMock(
            return_value=MagicMock(squeeze=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=noisy)
                ))
            )))
        )

        r2 = sid.identify(np.zeros(16000, dtype=np.float32))
        # smoothed = 0.35 * raw + 0.65 * prev_ema
        # Even if raw dips, the smoothed score should stay above threshold
        # because the EMA from the first call was very high
        assert r2["confidence"] > sid.SIMILARITY_THRESHOLD, \
            f"Smoothed score {r2['confidence']:.3f} should exceed threshold"

    def test_raw_score_in_return_dict(self):
        """Return dict includes raw_score alongside smoothed confidence."""
        profile_emb = _unit_vec(seed=1)
        sid = _make_speaker_id({"alice": profile_emb})

        sid._model.encode_batch = MagicMock(
            return_value=MagicMock(squeeze=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=profile_emb.copy())
                ))
            )))
        )

        result = sid.identify(np.zeros(16000, dtype=np.float32))
        assert "raw_score" in result
        assert isinstance(result["raw_score"], float)
        assert result["raw_score"] >= 0

    def test_unknown_returns_include_raw_score(self):
        """Early-exit unknown results also include raw_score=0."""
        sid = _make_speaker_id()
        sid.available = False
        result = sid.identify(np.zeros(16000, dtype=np.float32))
        assert result["raw_score"] == 0.0

    def test_short_audio_returns_raw_score(self):
        """Audio shorter than 0.5s returns raw_score=0."""
        sid = _make_speaker_id({"alice": _unit_vec(seed=1)})
        result = sid.identify(np.zeros(4000, dtype=np.float32))
        assert result["raw_score"] == 0.0
        assert result["is_known"] is False


class TestVoiceSmoothingDampening:
    """EMA dampens sudden score drops, preventing volatile identity flips."""

    def test_single_low_score_does_not_flip_to_unknown(self):
        """After several high-confidence matches, one low score shouldn't flip."""
        profile_emb = _unit_vec(seed=1)
        sid = _make_speaker_id({"bob": profile_emb})

        def _mock_embed(emb_val):
            sid._model.encode_batch = MagicMock(
                return_value=MagicMock(squeeze=MagicMock(return_value=MagicMock(
                    cpu=MagicMock(return_value=MagicMock(
                        numpy=MagicMock(return_value=emb_val)
                    ))
                )))
            )

        # Build up high EMA with 3 perfect matches
        for _ in range(3):
            _mock_embed(profile_emb.copy())
            r = sid.identify(np.zeros(16000, dtype=np.float32))
            assert r["is_known"] is True

        # Now inject a very poor match (near-random embedding)
        poor_emb = _unit_vec(seed=999)
        _mock_embed(poor_emb)
        r_poor = sid.identify(np.zeros(16000, dtype=np.float32))

        # The EMA should still be well above threshold: 0.35*low + 0.65*~1.0
        # Even with raw near 0, smoothed ~ 0.65 > 0.50
        assert r_poor["is_known"] is True, \
            f"Single outlier should not flip identity: smoothed={r_poor['confidence']:.3f}"

    def test_sustained_low_scores_eventually_flip(self):
        """Multiple consecutive low scores should eventually push EMA below threshold."""
        profile_emb = _unit_vec(seed=1)
        sid = _make_speaker_id({"carol": profile_emb})

        def _mock_embed(emb_val):
            sid._model.encode_batch = MagicMock(
                return_value=MagicMock(squeeze=MagicMock(return_value=MagicMock(
                    cpu=MagicMock(return_value=MagicMock(
                        numpy=MagicMock(return_value=emb_val)
                    ))
                )))
            )

        # Seed EMA with one good match
        _mock_embed(profile_emb.copy())
        sid.identify(np.zeros(16000, dtype=np.float32))

        # Now repeatedly inject near-random embeddings
        poor = _unit_vec(seed=999)
        for i in range(20):
            _mock_embed(poor)
            r = sid.identify(np.zeros(16000, dtype=np.float32))

        # After many bad scores, EMA should be below threshold
        assert r["is_known"] is False, \
            f"Sustained low scores should eventually flip: smoothed={r['confidence']:.3f}"


class TestEMAClearingOnLifecycle:
    """EMA is cleared on enrollment, removal, and merge to prevent stale state."""

    def test_enrollment_clears_ema(self):
        """Enrolling a speaker resets their score EMA."""
        sid = _make_speaker_id({"dave": _unit_vec(seed=1)})
        sid._score_ema["dave"] = 0.95

        with patch.object(sid, "_extract_embedding", return_value=_unit_vec(seed=2)):
            with patch.object(sid, "_save_profiles"):
                sid.enroll_speaker("dave", [np.zeros(16000, dtype=np.float32)])

        assert "dave" not in sid._score_ema

    def test_removal_clears_ema(self):
        """Removing a speaker profile clears their score EMA."""
        sid = _make_speaker_id({"eve": _unit_vec(seed=1)})
        sid._score_ema["eve"] = 0.88

        with patch.object(sid, "_save_profiles"):
            sid.remove_speaker("eve")

        assert "eve" not in sid._score_ema

    def test_merge_clears_both_ema(self):
        """Merging clears EMA for both source and target (centroid changed)."""
        sid = _make_speaker_id({
            "frank": _unit_vec(seed=1),
            "franklin": _unit_vec(seed=2),
        })
        sid._score_ema["frank"] = 0.80
        sid._score_ema["franklin"] = 0.90

        with patch.object(sid, "_save_profiles"):
            sid.merge_into("frank", "franklin")

        assert "frank" not in sid._score_ema
        assert "franklin" not in sid._score_ema


class TestSubThresholdAdaptiveUsesRaw:
    """Sub-threshold adaptive centroid pull uses raw score, not smoothed."""

    def test_adapt_triggers_on_raw_above_adapt_min(self):
        """When raw score >= ADAPT_MIN_SCORE but smoothed < SIMILARITY_THRESHOLD,
        the adaptive centroid pull should still fire."""
        profile_emb = _unit_vec(seed=1)
        sid = _make_speaker_id({"grace": profile_emb})

        # Pre-seed EMA low so smoothed stays below threshold
        sid._score_ema["grace"] = 0.30

        # Create embedding with raw cosine ~0.40 (above ADAPT_MIN_SCORE=0.35)
        mixed = profile_emb * 0.40 + _unit_vec(seed=50) * 0.60
        mixed = mixed / np.linalg.norm(mixed)

        sid._model.encode_batch = MagicMock(
            return_value=MagicMock(squeeze=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=mixed)
                ))
            )))
        )

        with patch.object(sid, "_save_profiles"):
            result = sid.identify(np.zeros(16000, dtype=np.float32))

        # Should be unknown (smoothed below threshold) but adaptive should
        # have assigned speaker_N, not "grace"
        assert result["is_known"] is False
        assert result["name"].startswith("speaker_")


class TestScoreEMAAlpha:
    """SCORE_EMA_ALPHA matches face_id's value."""

    def test_alpha_matches_face_id(self):
        from perception.speaker_id import SpeakerIdentifier
        from perception.face_id import FaceIdentifier
        assert SpeakerIdentifier.SCORE_EMA_ALPHA == FaceIdentifier.SCORE_EMA_ALPHA
