"""Tests for Layer 3A: Identity Continuity Window.

Covers: IdentityFusion persistence window — snapshot eligibility,
confidence decay, presence gating, wake-word clearing, entry gates,
conflict-state protection, and end-to-end lifecycle.
"""

from __future__ import annotations

import time
import pytest

from perception.identity_fusion import (
    IdentityFusion,
    IdentitySnapshot,
    ResolvedIdentity,
    STALE_VOICE_S,
    STALE_FACE_S,
    PERSIST_MAX_S,
    PERSIST_CONFIDENCE_HALF_LIFE_S,
    PERSIST_CONFIDENCE_FLOOR,
    FACE_CONFIRMED_VOICE_BOOST_THRESHOLD,
    VOICE_ACTIVE_MISMATCH_CEILING,
    _PERSIST_ELIGIBLE_METHODS,
    IDENTITY_RESOLVED,
)
from identity.types import CONFIDENCE_THRESHOLDS
from consciousness.events import (
    event_bus,
    PERCEPTION_USER_PRESENT_STABLE,
    PERCEPTION_WAKE_WORD,
)


@pytest.fixture()
def fusion():
    """Create an IdentityFusion instance without global side-effects."""
    f = IdentityFusion()
    f._user_present = True
    return f


def _inject_voice(fusion: IdentityFusion, name: str, confidence: float,
                   is_known: bool = True, age: float = 0.0):
    """Simulate a voice signal at a given age (seconds ago)."""
    from perception.identity_fusion import IdentitySignal
    fusion._voice = IdentitySignal(
        name=name if is_known else "unknown",
        confidence=confidence,
        is_known=is_known,
        timestamp=time.time() - age,
        source="voice",
    )


def _inject_face(fusion: IdentityFusion, name: str, confidence: float,
                  is_known: bool = True, age: float = 0.0):
    """Simulate a face signal at a given age (seconds ago)."""
    from perception.identity_fusion import IdentitySignal
    fusion._face = IdentitySignal(
        name=name if is_known else "unknown",
        confidence=confidence,
        is_known=is_known,
        timestamp=time.time() - age,
        source="face",
    )


# ── Test 1: Fresh voice match resolves normally, snapshot populated ──

def test_fresh_voice_resolves_and_snapshots(fusion):
    _inject_voice(fusion, "David", 0.65)
    fusion._resolve()

    assert fusion._last_resolved.name == "David"
    assert fusion._last_resolved.method == "voice_only"
    assert fusion._last_resolved.is_known is True
    assert fusion._last_known is not None
    assert fusion._last_known.name == "David"
    assert fusion._last_known.confidence == 0.65
    assert fusion._last_known.method == "voice_only"


# ── Test 2: Fresh face match resolves normally, snapshot populated ──

def test_fresh_face_resolves_and_snapshots(fusion):
    _inject_face(fusion, "Alice", 0.72)
    fusion._resolve()

    assert fusion._last_resolved.name == "Alice"
    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_known is not None
    assert fusion._last_known.name == "Alice"
    assert fusion._last_known.confidence == 0.72


# ── Test 3: Verified both resolves normally, snapshot populated ──

def test_verified_both_resolves_and_snapshots(fusion):
    _inject_voice(fusion, "David", 0.70)
    _inject_face(fusion, "David", 0.60)
    fusion._resolve()

    assert fusion._last_resolved.name == "David"
    assert fusion._last_resolved.method == "verified_both"
    assert fusion._last_known is not None
    assert fusion._last_known.name == "David"
    assert fusion._last_known.method == "verified_both"
    assert fusion._last_known.confidence == pytest.approx(0.85, abs=0.01)


# ── Test 4: Stale signals + user present → persisted identity ──

def test_stale_signals_with_user_present_persists(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()

    assert fusion._last_resolved.name == "David"
    assert fusion._last_resolved.method == "persisted"
    assert fusion._last_resolved.is_known is True
    assert fusion._last_resolved.confidence <= 0.70
    assert fusion._last_resolved.confidence >= PERSIST_CONFIDENCE_FLOOR


# ── Test 5: Stale signals + user absent → unknown ──

def test_stale_signals_with_user_absent_resolves_unknown(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None

    fusion._user_present = False
    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()

    assert fusion._last_resolved.name == "unknown"
    assert fusion._last_resolved.method == "no_signal"
    assert fusion._last_resolved.is_known is False


# ── Test 6: Persisted identity expires at hard max window ──

def test_persistence_expires_at_max_window(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None

    fusion._last_known = IdentitySnapshot(
        name="David",
        confidence=0.70,
        method="voice_only",
        captured_at=time.time() - PERSIST_MAX_S - 1,
    )
    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()

    assert fusion._last_resolved.name == "unknown"
    assert fusion._last_resolved.method == "no_signal"


# ── Test 7: Confidence decays from original, not from prior persisted ──

def test_confidence_decays_from_original_snapshot(fusion):
    original_conf = 0.80
    _inject_voice(fusion, "David", original_conf)
    fusion._resolve()

    # After 1 half-life: 0.80 * 0.5 = 0.40, but floor is 0.55 → clamped to floor
    elapsed_1hl = PERSIST_CONFIDENCE_HALF_LIFE_S
    fusion._last_known = IdentitySnapshot(
        name="David",
        confidence=original_conf,
        method="voice_only",
        captured_at=time.time() - elapsed_1hl,
    )
    _inject_voice(fusion, "David", original_conf, age=STALE_VOICE_S + 1)
    fusion._resolve()

    expected_1hl = max(PERSIST_CONFIDENCE_FLOOR, original_conf * 0.5)
    assert fusion._last_resolved.method == "persisted"
    assert fusion._last_resolved.confidence == pytest.approx(expected_1hl, abs=0.02)

    # After 10s (short elapsed), decay should be visible above the floor
    short_elapsed = 10.0
    fusion._last_known = IdentitySnapshot(
        name="David",
        confidence=original_conf,
        method="voice_only",
        captured_at=time.time() - short_elapsed,
    )
    fusion._resolve()

    expected_short = original_conf * (0.5 ** (short_elapsed / PERSIST_CONFIDENCE_HALF_LIFE_S))
    assert expected_short > PERSIST_CONFIDENCE_FLOOR
    assert fusion._last_resolved.confidence == pytest.approx(expected_short, abs=0.02)

    # Verify decay uses original (0.80), not a prior persisted value
    # A second resolve at the same elapsed should give the same result
    fusion._resolve()
    assert fusion._last_resolved.confidence == pytest.approx(expected_short, abs=0.02)


# ── Test 8: Confidence never goes below floor ──

def test_confidence_floor_enforced(fusion):
    _inject_voice(fusion, "David", 0.60)
    fusion._resolve()

    fusion._last_known = IdentitySnapshot(
        name="David",
        confidence=0.60,
        method="voice_only",
        captured_at=time.time() - (PERSIST_MAX_S - 1),
    )
    _inject_voice(fusion, "David", 0.60, age=STALE_VOICE_S + 1)
    fusion._resolve()

    assert fusion._last_resolved.method == "persisted"
    assert fusion._last_resolved.confidence >= PERSIST_CONFIDENCE_FLOOR
    assert fusion._last_resolved.confidence == pytest.approx(
        PERSIST_CONFIDENCE_FLOOR, abs=0.01
    )


# ── Test 9: Wake word clears persisted branch immediately ──

def test_wake_word_clears_persistence(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None
    assert fusion._last_known.name == "David"

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    fusion._on_wake_word()

    assert fusion._last_known is None
    assert fusion._last_resolved.method == "no_signal"
    assert fusion._last_resolved.name == "unknown"


# ── Test 10: Fresh biometric overrides persisted identity ──

def test_fresh_signal_overrides_persisted(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    _inject_voice(fusion, "Alice", 0.80)
    fusion._resolve()
    assert fusion._last_resolved.name == "Alice"
    assert fusion._last_resolved.method == "voice_only"
    assert fusion._last_known.name == "Alice"


# ── Test 11: Unknown/low-confidence identity is never persisted ──

def test_unknown_identity_not_persisted(fusion):
    _inject_voice(fusion, "unknown", 0.30, is_known=False)
    fusion._resolve()

    assert fusion._last_known is None
    assert fusion._last_resolved.method == "no_signal"


def test_low_confidence_identity_not_persisted(fusion):
    soft = CONFIDENCE_THRESHOLDS["soft"]
    _inject_voice(fusion, "David", soft - 0.05)
    fusion._resolve()

    assert fusion._last_resolved.name == "David"
    assert fusion._last_resolved.method == "voice_only"
    assert fusion._last_known is None

    _inject_voice(fusion, "David", soft - 0.05, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "no_signal"


# ── Test 12: Conflict state is never snapshotted as last known ──

def test_conflict_hold_not_snapshotted(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known.name == "David"

    prev_snapshot = fusion._last_known

    _inject_voice(fusion, "David", 0.70)
    _inject_face(fusion, "Bob", 0.60)
    fusion._resolve()
    assert fusion._last_resolved.method == "conflict_hold"
    assert fusion._last_known is prev_snapshot


# ── Additional edge-case tests ──

def test_presence_departure_clears_persistence(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    fusion._on_presence(present=False)

    assert fusion._last_known is None
    assert fusion._last_resolved.method == "no_signal"


def test_get_status_persistence_fields(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()

    status = fusion.get_status()
    assert status["persisted"] is False
    assert status["last_known_identity"] == "David"
    assert status["last_known_confidence"] == 0.70
    assert status["user_present"] is True
    assert status["persist_elapsed_s"] == 0
    assert status["persist_remaining_s"] == 0
    assert status["snapshot_age_s"] >= 0

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()

    status = fusion.get_status()
    assert status["persisted"] is True
    assert status["persist_elapsed_s"] >= 0
    assert status["persist_remaining_s"] <= PERSIST_MAX_S
    assert status["snapshot_age_s"] >= 0


def test_get_state_is_alias(fusion):
    assert fusion.get_state() == fusion.get_status()


def test_wake_word_preserves_fresh_signals(fusion):
    """Wake word should clear persistence but not raw voice/face signals.

    Note: _on_wake_word() calls _resolve() which re-snapshots if voice is
    still fresh. The key invariant is that voice data survives the wake word
    and re-resolves normally (not as persisted).
    """
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_resolved.method == "voice_only"

    fusion._on_wake_word()
    assert fusion._voice.name == "David"
    assert fusion._voice.is_known is True
    assert fusion._last_resolved.method == "voice_only"


def test_persist_eligible_methods_complete():
    expected = {"verified_both", "voice_only", "face_only",
                "conflict_resolved_voice", "conflict_resolved_face"}
    assert _PERSIST_ELIGIBLE_METHODS == expected


# ── Smart wake-word persistence: face-confirmed identity survives ──

def test_wake_word_preserves_when_face_live_and_matching(fusion):
    """Wake word should NOT clear persistence when face is live, known, and matches."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None
    assert fusion._last_known.name == "David"

    # Voice goes stale → enters persisted
    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    # Face appears fresh, confirming David is still here
    _inject_face(fusion, "David", 0.60)

    fusion._on_wake_word()
    assert fusion._last_known is not None, "Face-confirmed identity should survive wake word"
    assert fusion._last_known.name == "David"


def test_wake_word_clears_when_face_unknown(fusion):
    """Wake word should clear persistence when face is unknown/absent."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    # No face signal — only default unknown face
    fusion._on_wake_word()
    assert fusion._last_known is None


def test_wake_word_clears_when_face_mismatches(fusion):
    """Wake word should clear David's persistence when Alice's face is live."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None
    assert fusion._last_known.name == "David"

    # Voice stale → persisted
    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    # Alice's face appears (different person than persisted David)
    _inject_face(fusion, "Alice", 0.65)

    fusion._on_wake_word()
    # David's persistence is cleared; _resolve() picks up Alice via face_only
    assert fusion._last_resolved.name == "Alice"
    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_known.name == "Alice", "Alice should now be the active identity"


def test_wake_word_clears_when_face_stale(fusion):
    """Wake word should clear persistence when face signal is stale."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None

    # Voice stale → persisted
    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    # Face is David but stale
    _inject_face(fusion, "David", 0.60, age=STALE_FACE_S + 1)

    fusion._on_wake_word()
    assert fusion._last_known is None, "Stale face should not preserve persistence"


def test_wake_word_clears_when_face_low_confidence(fusion):
    """Wake word should clear persistence when face confidence is below soft threshold."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_known is not None

    # Voice stale → persisted
    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    # Face matches but confidence too low
    _inject_face(fusion, "David", 0.20)

    fusion._on_wake_word()
    assert fusion._last_known is None, "Low-confidence face should not preserve persistence"


def test_floor_above_quarantine_threshold():
    assert PERSIST_CONFIDENCE_FLOOR == 0.65
    assert PERSIST_CONFIDENCE_FLOOR > CONFIDENCE_THRESHOLDS["quarantine"]


def test_conflict_resolved_gets_snapshotted(fusion):
    """Conflict resolved via higher-confidence winner should be snapshot-eligible."""
    _inject_voice(fusion, "David", 0.70)
    _inject_face(fusion, "Bob", 0.50)
    fusion._conflict_start = time.time() - 10  # past hold window
    fusion._resolve()

    assert fusion._last_resolved.method == "conflict_resolved_voice"
    assert fusion._last_known is not None
    assert fusion._last_known.name == "David"
    assert fusion._last_known.method == "conflict_resolved_voice"


# ── Flip count tests (Layer 8 identity instability feed) ──

def test_flip_count_starts_at_zero(fusion):
    assert fusion._flip_count == 0


def test_flip_count_increments_on_known_name_change(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._flip_count == 0

    _inject_voice(fusion, "Alice", 0.80)
    fusion._resolve()
    assert fusion._flip_count == 1


def test_flip_count_does_not_increment_unknown_to_known(fusion):
    """unknown → David is not a flip (it's a first resolution)."""
    assert fusion._last_resolved.name == "unknown"
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._flip_count == 0


def test_flip_count_does_not_increment_known_to_unknown(fusion):
    """David → unknown is not a flip (it's a signal loss)."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._user_present = False
    fusion._last_known = None
    fusion._resolve()
    assert fusion._last_resolved.name == "unknown"
    assert fusion._flip_count == 0


def test_flip_count_cumulative(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    _inject_voice(fusion, "Alice", 0.80)
    fusion._resolve()
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    _inject_voice(fusion, "Bob", 0.60)
    fusion._resolve()
    assert fusion._flip_count == 3


def test_flip_count_in_get_status(fusion):
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    _inject_voice(fusion, "Alice", 0.80)
    fusion._resolve()

    status = fusion.get_status()
    assert "flip_count" in status
    assert status["flip_count"] == 1


# ── Passive expiry tests (Layer 3A.1) ──

def test_passive_expiry_voice_only_collapses(fusion):
    """voice_only identity should collapse when voice goes stale on status read."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()
    assert fusion._last_resolved.method == "voice_only"

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)

    status = fusion.get_status()
    assert status["method"] in ("persisted", "no_signal")


def test_passive_expiry_current_property(fusion):
    """The current property should trigger expiry check."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)

    resolved = fusion.current
    assert resolved.method in ("persisted", "no_signal")


def test_passive_expiry_verified_both_collapses(fusion):
    """verified_both should collapse when either signal goes stale."""
    _inject_voice(fusion, "David", 0.70)
    _inject_face(fusion, "David", 0.60)
    fusion._resolve()
    assert fusion._last_resolved.method == "verified_both"

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)

    status = fusion.get_status()
    assert status["method"] != "verified_both"


def test_passive_expiry_no_signal_does_not_loop(fusion):
    """no_signal state should not trigger redundant re-resolves."""
    fusion._resolve()
    assert fusion._last_resolved.method == "no_signal"

    fusion._check_expiry()
    assert fusion._last_resolved.method == "no_signal"


def test_passive_expiry_persisted_does_not_retrigger(fusion):
    """persisted method should not trigger expiry check (it manages its own lifecycle)."""
    _inject_voice(fusion, "David", 0.70)
    fusion._resolve()

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()
    assert fusion._last_resolved.method == "persisted"

    fusion._check_expiry()
    assert fusion._last_resolved.method == "persisted"


# ── Per-modality staleness tests ──

def test_face_stays_fresh_beyond_voice_stale_window(fusion):
    """Face signal should survive past STALE_VOICE_S since it uses STALE_FACE_S."""
    _inject_face(fusion, "David", 0.80, age=STALE_VOICE_S + 5)
    fusion._resolve()

    assert fusion._last_resolved.name == "David"
    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_resolved.is_known is True


def test_face_goes_stale_at_face_threshold(fusion):
    """Face signal should be considered stale after STALE_FACE_S."""
    _inject_face(fusion, "David", 0.80)
    fusion._resolve()
    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_known is not None

    _inject_face(fusion, "David", 0.80, age=STALE_FACE_S + 1)
    fusion._resolve()

    assert fusion._last_resolved.method == "persisted"


def test_voice_stale_face_fresh_resolves_face_only(fusion):
    """When voice goes stale but face is still fresh, result should be face_only."""
    _inject_voice(fusion, "David", 0.70)
    _inject_face(fusion, "David", 0.75)
    fusion._resolve()
    assert fusion._last_resolved.method == "verified_both"

    _inject_voice(fusion, "David", 0.70, age=STALE_VOICE_S + 1)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_resolved.name == "David"


def test_stale_thresholds_are_distinct():
    """STALE_VOICE_S and STALE_FACE_S should have different values."""
    assert STALE_VOICE_S < STALE_FACE_S
    assert STALE_VOICE_S == 30.0
    assert STALE_FACE_S == 90.0


# ── Cross-modal voice boost tests ──

def test_cross_modal_boost_promotes_to_verified_both(fusion):
    """Face confirmed + voice near threshold + same name → verified_both."""
    _inject_voice(fusion, "David", 0.45, is_known=False)
    _inject_face(fusion, "David", 0.80)
    fusion._resolve()

    assert fusion._last_resolved.method == "verified_both"
    assert fusion._last_resolved.name == "David"
    assert fusion._last_resolved.is_known is True


def test_cross_modal_boost_requires_high_face_confidence(fusion):
    """Face below 0.70 should not trigger the cross-modal boost."""
    _inject_voice(fusion, "David", 0.45, is_known=False)
    _inject_face(fusion, "David", 0.60)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_only"


def test_cross_modal_boost_does_not_fire_when_voice_too_low(fusion):
    """Voice well below boost threshold but at mismatch ceiling → face_only."""
    _inject_voice(fusion, "unknown", 0.20, is_known=False)
    _inject_face(fusion, "David", 0.85)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_resolved.name == "David"


def test_cross_modal_boost_requires_minimum_voice_score(fusion):
    """Voice score below FACE_CONFIRMED_VOICE_BOOST_THRESHOLD should not boost."""
    _inject_voice(fusion, "David", FACE_CONFIRMED_VOICE_BOOST_THRESHOLD - 0.05, is_known=False)
    _inject_face(fusion, "David", 0.85)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_only"


def test_cross_modal_boost_at_exact_threshold(fusion):
    """Voice score exactly at FACE_CONFIRMED_VOICE_BOOST_THRESHOLD should boost."""
    _inject_voice(fusion, "David", FACE_CONFIRMED_VOICE_BOOST_THRESHOLD, is_known=False)
    _inject_face(fusion, "David", 0.80)
    fusion._resolve()

    assert fusion._last_resolved.method == "verified_both"


def test_cross_modal_boost_does_not_fire_when_voice_already_known(fusion):
    """If voice is already known, the normal verified_both path handles it."""
    _inject_voice(fusion, "David", 0.70, is_known=True)
    _inject_face(fusion, "David", 0.80)
    fusion._resolve()

    assert fusion._last_resolved.method == "verified_both"


def test_cross_modal_boost_adopts_face_name(fusion):
    """Boosted voice signal should adopt the face's name for verified_both."""
    from perception.identity_fusion import IdentitySignal
    fusion._voice = IdentitySignal(
        name="unknown", confidence=0.45, is_known=False,
        timestamp=time.time(), source="voice",
    )
    _inject_face(fusion, "David", 0.85)
    fusion._resolve()

    assert fusion._last_resolved.method == "verified_both"
    assert fusion._last_resolved.name == "David"
    assert fusion._last_resolved.voice_name == "David"


# ── Updated persistence parameter tests ──

def test_half_life_is_90s():
    assert PERSIST_CONFIDENCE_HALF_LIFE_S == 90.0


def test_persist_floor_is_065():
    assert PERSIST_CONFIDENCE_FLOOR == 0.65


# ── Face-present voice-unknown tests ──

def test_active_unknown_voice_with_face_resolves_unknown(fusion):
    """When someone speaks (score near zero) and camera sees a known face,
    the speaker should NOT be attributed to the face person."""
    _inject_voice(fusion, "unknown", 0.004, is_known=False)
    _inject_face(fusion, "David", 0.93)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_present_voice_unknown"
    assert fusion._last_resolved.name == "unknown"
    assert fusion._last_resolved.is_known is False
    assert fusion._last_resolved.face_name == "David"


def test_active_voice_above_mismatch_ceiling_stays_face_only(fusion):
    """Voice confidence at or above VOICE_ACTIVE_MISMATCH_CEILING should
    still resolve to face_only (benefit of the doubt for borderline scores)."""
    _inject_voice(fusion, "unknown", VOICE_ACTIVE_MISMATCH_CEILING, is_known=False)
    _inject_face(fusion, "David", 0.85)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_resolved.name == "David"


def test_stale_voice_with_face_resolves_face_only(fusion):
    """When voice signal is stale (no recent speech), face_only is correct."""
    _inject_voice(fusion, "unknown", 0.004, is_known=False, age=STALE_VOICE_S + 1)
    _inject_face(fusion, "David", 0.90)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_resolved.name == "David"


def test_face_present_voice_unknown_not_snapshotted(fusion):
    """face_present_voice_unknown should NOT create a persistence snapshot
    since the speaker's identity is uncertain."""
    _inject_voice(fusion, "unknown", 0.01, is_known=False)
    _inject_face(fusion, "David", 0.90)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_present_voice_unknown"
    assert fusion._last_known is None


def test_face_present_voice_unknown_threshold(fusion):
    """Voice at 0.19 (below 0.20 ceiling) should trigger face_present_voice_unknown."""
    _inject_voice(fusion, "unknown", 0.19, is_known=False)
    _inject_face(fusion, "David", 0.85)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_present_voice_unknown"
    assert fusion._last_resolved.name == "unknown"


def test_borderline_voice_stays_face_only(fusion):
    """A borderline voice score (0.35, above 0.20 ceiling) should stay face_only
    since short/noisy utterances produce unreliable embeddings."""
    _inject_voice(fusion, "unknown", 0.35, is_known=False)
    _inject_face(fusion, "David", 0.90)
    fusion._resolve()

    assert fusion._last_resolved.method == "face_only"
    assert fusion._last_resolved.name == "David"


# ---------------------------------------------------------------------------
# Synthetic perception exercise — truth boundary guard
# ---------------------------------------------------------------------------

class TestSyntheticGate:
    """Verify IDENTITY_RESOLVED events are suppressed during synthetic sessions."""

    def _collect_events(self, fusion: IdentityFusion) -> list:
        """Subscribe a collector to IDENTITY_RESOLVED and return the list."""
        from consciousness.events import _BarrierState
        if event_bus._barrier != _BarrierState.OPEN:
            event_bus.open_barrier()

        received: list = []

        def _collector(**kwargs):
            received.append(kwargs)

        event_bus.on(IDENTITY_RESOLVED, _collector)
        return received

    def _sample_payload(self) -> dict:
        return {
            "name": "Alice",
            "confidence": 0.85,
            "is_known": True,
            "method": "voice_only",
            "voice_name": "Alice",
            "face_name": "unknown",
            "conflict": False,
        }

    def test_emission_when_inactive(self, fusion: IdentityFusion):
        received = self._collect_events(fusion)
        fusion.set_synthetic_active(False)
        fusion._emit_resolved(self._sample_payload())
        assert len(received) == 1
        assert received[0].get("name") == "Alice"

    def test_emission_suppressed_when_active(self, fusion: IdentityFusion):
        received = self._collect_events(fusion)
        fusion.set_synthetic_active(True)
        fusion._emit_resolved(self._sample_payload())
        fusion._emit_resolved(self._sample_payload())
        assert received == []
        assert fusion._synthetic_suppressed_count == 2

    def test_emission_resumes_after_deactivation(self, fusion: IdentityFusion):
        received = self._collect_events(fusion)
        fusion.set_synthetic_active(True)
        fusion._emit_resolved(self._sample_payload())
        assert received == []
        fusion.set_synthetic_active(False)
        fusion._emit_resolved(self._sample_payload())
        assert len(received) == 1

    def test_state_still_updates_while_synthetic(self, fusion: IdentityFusion):
        """Internal recognition state must keep updating so distillation still
        flows, even though event emission is suppressed."""
        fusion.set_synthetic_active(True)
        fusion._on_voice(name="Alice", confidence=0.85, is_known=True)
        # Voice signal is recorded internally even when event emission is gated
        assert fusion._voice.name == "Alice"
        assert fusion._voice.confidence == pytest.approx(0.85)

    def test_helper_with_none_payload_is_noop(self, fusion: IdentityFusion):
        received = self._collect_events(fusion)
        fusion.set_synthetic_active(False)
        fusion._emit_resolved(None)
        fusion._emit_resolved({})
        assert received == []
