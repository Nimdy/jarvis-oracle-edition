"""Regression tests for STT hallucination defenses.

These test the pure-python filtering layer in brain/perception/stt.py. The
whisper model itself is not exercised (no GPU, no audio); the defenses being
tested are:

    1. _looks_like_hallucination — RMS-gated known-hallucination blocklist.
    2. _collect_segment_text — per-segment no_speech_prob gate.

The family-level contract: well-documented whisper boilerplate ("Thanks for
watching!", "Please subscribe.", etc.) is filtered when the input audio is
near-silent; legitimate user utterances are never filtered regardless of audio
level; loud audio that happens to match a blocklist phrase is allowed through
(defense in depth only fires on suspicious conditions).
"""

from __future__ import annotations

import pytest

from perception.stt import (
    _KNOWN_HALLUCINATIONS,
    _PER_SEGMENT_NO_SPEECH_FLOOR,
    _SUSPICIOUS_RMS_THRESHOLD,
    _collect_segment_text,
    _compute_rms,
    _looks_like_hallucination,
)


# Small fake segment that mirrors the faster-whisper Segment dataclass shape
# (text, no_speech_prob) — only the attributes the filter reads are needed.
class _FakeSegment:
    def __init__(self, text: str, no_speech_prob: float = 0.0) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob


class TestKnownHallucinationBlocklist:
    """`_looks_like_hallucination` should only fire on RMS-quiet + known phrase."""

    @pytest.mark.parametrize("phrase", [
        "Thanks for watching.",
        "Thanks for watching!",
        "thanks for watching",
        "Thank you for watching.",
        "Please subscribe.",
        "Please subscribe!",
        "Please like and subscribe.",
        "I'll see you in the next video.",
        "See you in the next video.",
        "Subtitles by the Amara.org community",
    ])
    def test_quiet_audio_blocks_known_boilerplate(self, phrase: str) -> None:
        assert _looks_like_hallucination(phrase, rms=0.005), (
            f"Should block known hallucination {phrase!r} on quiet audio"
        )

    @pytest.mark.parametrize("phrase", [
        "Thanks for watching.",
        "Please subscribe.",
    ])
    def test_loud_audio_allows_blocklist_phrase(self, phrase: str) -> None:
        # If RMS is above the suspicious-silence floor, we trust whisper's
        # other hallucination defenses (no_speech_prob, compression ratio)
        # and let the output through rather than over-filtering.
        assert not _looks_like_hallucination(phrase, rms=0.1), (
            f"Should not block {phrase!r} when audio is loud enough"
        )

    @pytest.mark.parametrize("phrase", [
        "Hey Jarvis, what's the time",
        "turn on the kitchen lights",
        "what are you most curious about",
        "my name is David",
        "learn speaker diarization",
        # Ambiguous phrases deliberately NOT in the blocklist:
        "Bye.",
        "Goodbye.",
        "you",
        "Amen.",
    ])
    def test_legitimate_utterance_never_filtered(self, phrase: str) -> None:
        # Even at extremely quiet RMS, legitimate or ambiguous user utterances
        # must not be caught by the blocklist. This protects the user's ability
        # to whisper short commands without having them silently discarded.
        assert not _looks_like_hallucination(phrase, rms=0.0001), (
            f"Legitimate phrase {phrase!r} must not be filtered"
        )

    def test_empty_text_not_filtered(self) -> None:
        assert not _looks_like_hallucination("", rms=0.0)
        assert not _looks_like_hallucination("   ", rms=0.0)

    def test_threshold_boundary(self) -> None:
        # Exactly at the threshold should NOT fire (strict <).
        assert not _looks_like_hallucination(
            "Thanks for watching.", rms=_SUSPICIOUS_RMS_THRESHOLD,
        )
        # Just below the threshold should fire.
        assert _looks_like_hallucination(
            "Thanks for watching.", rms=_SUSPICIOUS_RMS_THRESHOLD - 0.001,
        )

    def test_blocklist_deliberately_excludes_ambiguous_phrases(self) -> None:
        """Contract: keep blocklist narrow to avoid false rejections.

        Phrases a human could plausibly say to Jarvis must not be blocklisted.
        If this list grows to include "bye", "you", "goodbye", etc., we've
        drifted into over-filtering legitimate input.
        """
        forbidden_entries = {"bye.", "bye", "goodbye.", "goodbye", "you",
                             "amen.", "amen", "okay.", "yes.", "no."}
        leaked = forbidden_entries & _KNOWN_HALLUCINATIONS
        assert not leaked, (
            f"Blocklist leaked into legitimate-utterance territory: {leaked}"
        )


class TestPerSegmentNoSpeechGate:
    """`_collect_segment_text` filters whisper segments by their own silence estimate."""

    def test_high_no_speech_prob_segment_dropped(self) -> None:
        segs = [
            _FakeSegment("hello world", no_speech_prob=0.1),
            _FakeSegment("thanks for watching", no_speech_prob=0.9),
        ]
        text, total, kept, dropped = _collect_segment_text(
            segs, _PER_SEGMENT_NO_SPEECH_FLOOR,
        )
        assert text == "hello world"
        assert total == 2
        assert kept == 1
        assert len(dropped) == 1
        assert "0.9" in dropped[0]

    def test_low_no_speech_prob_segment_kept(self) -> None:
        segs = [_FakeSegment("real speech", no_speech_prob=0.05)]
        text, total, kept, dropped = _collect_segment_text(segs, 0.6)
        assert text == "real speech"
        assert kept == 1
        assert dropped == []

    def test_multiple_segments_joined(self) -> None:
        segs = [
            _FakeSegment("hello", no_speech_prob=0.1),
            _FakeSegment("world", no_speech_prob=0.2),
        ]
        text, total, kept, _ = _collect_segment_text(segs, 0.6)
        assert text == "hello world"
        assert total == 2
        assert kept == 2

    def test_threshold_is_strict_greater_than(self) -> None:
        # A segment exactly at the floor should be kept (strict >).
        segs = [_FakeSegment("edge case", no_speech_prob=_PER_SEGMENT_NO_SPEECH_FLOOR)]
        text, _, kept, dropped = _collect_segment_text(
            segs, _PER_SEGMENT_NO_SPEECH_FLOOR,
        )
        assert text == "edge case"
        assert kept == 1
        assert dropped == []

    def test_empty_input(self) -> None:
        text, total, kept, dropped = _collect_segment_text([], 0.6)
        assert text == ""
        assert total == kept == 0
        assert dropped == []

    def test_missing_attribute_defaults_to_zero(self) -> None:
        """If a segment lacks no_speech_prob, we treat it as 0 (keep it)."""
        class _BareSeg:
            text = "bare segment"
        segs = [_BareSeg()]
        text, _, kept, _ = _collect_segment_text(segs, 0.6)
        assert text == "bare segment"
        assert kept == 1

    def test_whitespace_only_segments_dropped(self) -> None:
        segs = [
            _FakeSegment("   ", no_speech_prob=0.0),
            _FakeSegment("actual", no_speech_prob=0.0),
        ]
        text, total, kept, _ = _collect_segment_text(segs, 0.6)
        assert text == "actual"
        assert total == 2
        assert kept == 1


class TestRmsComputation:
    """Sanity check the RMS helper since it gates the blocklist."""

    def test_rms_of_silence_is_zero(self) -> None:
        import numpy as np
        assert _compute_rms(np.zeros(16000, dtype=np.float32)) == 0.0

    def test_rms_of_empty_is_zero(self) -> None:
        import numpy as np
        assert _compute_rms(np.array([], dtype=np.float32)) == 0.0

    def test_rms_of_loud_signal(self) -> None:
        import numpy as np
        sig = np.full(16000, 0.5, dtype=np.float32)
        assert abs(_compute_rms(sig) - 0.5) < 1e-6

    def test_rms_handles_none_gracefully(self) -> None:
        assert _compute_rms(None) == 0.0  # type: ignore[arg-type]
