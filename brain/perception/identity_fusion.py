"""Multi-modal identity fusion — combines voice and face signals.

Fusion rules:
  voice=Alice + face=Alice  → VERIFIED (high confidence)
  voice=Alice + face=unknown → ACCEPT voice (camera may not see)
  face=Alice  + voice=absent → ACCEPT face (hasn't spoken yet)
  face=Alice  + voice=active_unknown (score < threshold)
                             → UNKNOWN speaker (different person visible on camera)
  voice=Alice + face=Bob    → CONFLICT (require more evidence)

Emits a single canonical identity that downstream systems (soul,
relationships, permissions) can trust.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from consciousness.events import (
    event_bus,
    PERCEPTION_SPEAKER_IDENTIFIED,
    PERCEPTION_FACE_IDENTIFIED,
    PERCEPTION_USER_PRESENT_STABLE,
    PERCEPTION_WAKE_WORD,
    PERCEPTION_PLAYBACK_COMPLETE,
)
from identity.types import CONFIDENCE_THRESHOLDS

logger = logging.getLogger(__name__)

CONFLICT_HOLD_S = 5.0
VERIFIED_BOOST = 0.15
STALE_VOICE_S = 30.0
STALE_FACE_S = 90.0
VOICE_SHORT_GAP_HOLD_S = 3.0
VOICE_SHORT_GAP_MIN_CONFIDENCE = 0.30
VOICE_SHORT_GAP_CONF_DECAY = 0.90
VOICE_DROP_GRACE_S = 8.0
VOICE_DROP_GRACE_MIN_ACTIVE_CONFIDENCE = 0.05
CONVERSATION_BOUNDARY_GRACE_S = 12.0
CONVERSATION_BOUNDARY_MAX_SNAPSHOT_AGE_S = 45.0

PERSIST_MAX_S = 180.0
PERSIST_CONFIDENCE_HALF_LIFE_S = 90.0
PERSIST_CONFIDENCE_FLOOR = 0.65

FACE_CONFIRMED_VOICE_BOOST_THRESHOLD = 0.40
# Below this, an active voice signal is clearly a different person from the
# face-identified user (cosine similarity near zero = no correlation).
# Conservative threshold: only fires for genuinely unmatched speakers (< 0.20),
# not for noisy short utterances that score in the 0.20-0.50 ambiguous range.
VOICE_ACTIVE_MISMATCH_CEILING = 0.20
_PERSIST_ELIGIBLE_METHODS = frozenset({
    "verified_both", "voice_only", "face_only",
    "conflict_resolved_voice", "conflict_resolved_face",
})

# Recognition state machine constants
ACCUMULATION_FLOOR = 0.25
TENTATIVE_THRESHOLD = 0.45
TENTATIVE_MIN_SIGNALS = 3
EVIDENCE_HALF_LIFE_S = 30.0
COLD_START_BOOST_WINDOW_S = 60.0
COLD_START_THRESHOLD_REDUCTION = 0.05

TENTATIVE_MAX_AGE_S = 60.0
TENTATIVE_BRIDGE_REDUCTION = 0.03
MULTI_PERSON_VOICE_THRESHOLD = 0.55

RecognitionState = str  # "unknown_present" | "tentative_match" | "confirmed_match" | "absent"

IDENTITY_RESOLVED = "perception:identity_resolved"

_active_instance: "IdentityFusion | None" = None


@dataclass
class IdentitySignal:
    name: str = "unknown"
    confidence: float = 0.0
    is_known: bool = False
    timestamp: float = 0.0
    source: str = ""


@dataclass
class ResolvedIdentity:
    name: str = "unknown"
    confidence: float = 0.0
    is_known: bool = False
    method: str = "none"
    voice_name: str = "unknown"
    face_name: str = "unknown"
    conflict: bool = False


@dataclass
class IdentitySnapshot:
    """Decoupled snapshot of a known biometric identity for persistence."""
    name: str
    confidence: float
    method: str
    captured_at: float


@dataclass
class _CandidateEvidence:
    """Accumulated sub-threshold evidence for a single candidate identity."""
    name: str
    signals: int = 0
    weighted_confidence: float = 0.0
    last_signal_at: float = 0.0

    def add(self, confidence: float, now: float) -> None:
        self._decay(now)
        self.signals += 1
        self.weighted_confidence += confidence
        self.last_signal_at = now

    def effective_confidence(self, now: float) -> float:
        self._decay(now)
        return self.weighted_confidence

    def _decay(self, now: float) -> None:
        if self.last_signal_at and now > self.last_signal_at:
            elapsed = now - self.last_signal_at
            factor = 0.5 ** (elapsed / EVIDENCE_HALF_LIFE_S)
            self.weighted_confidence *= factor


class IdentityFusion:
    """Fuses voice and face identity signals into a single canonical identity."""

    def __init__(self) -> None:
        self._voice: IdentitySignal = IdentitySignal(source="voice")
        self._face: IdentitySignal = IdentitySignal(source="face")
        self._last_resolved: ResolvedIdentity = ResolvedIdentity()
        self._conflict_start: float = 0.0
        self._enabled = True
        self._flip_count: int = 0
        # Layer 3A: Identity persistence state
        self._user_present: bool = False
        self._last_known: IdentitySnapshot | None = None
        self._persist_cleared_at: float = 0.0
        # Recognition state machine
        self._recognition_state: RecognitionState = "absent"
        self._voice_candidates: dict[str, _CandidateEvidence] = {}
        self._face_candidates: dict[str, _CandidateEvidence] = {}
        self._tentative_name: str = ""
        self._tentative_confidence: float = 0.0
        self._presence_start: float = 0.0
        self._has_profiles: bool = False
        self._pending_emit: dict[str, Any] | None = None
        # Multi-speaker awareness
        self._visible_person_count: int = 0
        self._voice_trust_state: str = "unknown"
        self._trust_reason: str = ""
        self._resolution_basis: str = ""
        self._threshold_assist_active: bool = False
        self._threshold_assist_name: str = ""
        self._voice_gap_smoothed: bool = False
        self._voice_gap_smoothed_at: float = 0.0
        self._conversation_boundary_until: float = 0.0
        self._conversation_boundary_reason: str = ""
        self._voice_drop_recent_name: str = ""
        self._voice_drop_recent_ts: float = 0.0
        self._voice_drop_recent_method: str = ""
        # Unknown identity continuity
        self._unknown_voice_events: list[dict[str, Any]] = []
        self._UNKNOWN_VOICE_MAX = 20
        self._synthetic_active: bool = False
        self._synthetic_suppressed_count: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        global _active_instance
        _active_instance = self
        event_bus.on(PERCEPTION_SPEAKER_IDENTIFIED, self._on_voice)
        event_bus.on(PERCEPTION_FACE_IDENTIFIED, self._on_face)
        event_bus.on(PERCEPTION_USER_PRESENT_STABLE, self._on_presence)
        event_bus.on(PERCEPTION_WAKE_WORD, self._on_wake_word)
        event_bus.on(PERCEPTION_PLAYBACK_COMPLETE, self._on_playback_complete)
        logger.info("Identity fusion started (persistence window: %ds)", int(PERSIST_MAX_S))

    def set_enabled(self, val: bool) -> None:
        self._enabled = val

    def set_synthetic_active(self, val: bool) -> None:
        """Mark whether a synthetic perception exercise session is active.

        Truth-boundary guard: while active, ``IDENTITY_RESOLVED`` events are
        suppressed so that synthetic TTS audio cannot drive IdentityBoundary
        scope, evidence accumulation, or any downstream identity-side-effect.
        The internal fusion state still updates (recognition logic, candidate
        accumulation, distillation) — only the bus emission is gated.
        """
        with self._lock:
            self._synthetic_active = bool(val)

    def _emit_resolved(self, emit_payload: dict[str, Any] | None) -> None:
        """Single gate for all ``IDENTITY_RESOLVED`` emissions.

        Suppresses emission when a synthetic perception exercise session is
        active. Safe to call with ``None`` (no-op). Must be called OUTSIDE the
        fusion lock because ``event_bus.emit`` can dispatch synchronously to
        listeners that may try to reacquire state.
        """
        if not emit_payload:
            return
        if self._synthetic_active:
            self._synthetic_suppressed_count += 1
            return
        event_bus.emit(IDENTITY_RESOLVED, **emit_payload)

    def _on_voice(self, name="unknown", confidence=0.0, is_known=False, closest_match="", **_):
        if not self._enabled:
            return
        with self._lock:
            self._on_voice_locked(name, confidence, is_known, closest_match)
            emit_payload = self._pending_emit
            self._pending_emit = None
        self._emit_resolved(emit_payload)

    def _on_voice_locked(self, name: str, confidence: float, is_known: bool,
                         closest_match: str = "") -> None:
        now = time.time()
        prev_voice = self._voice

        # Cold-start boost: lower threshold when user is newly present
        if (not is_known and name != "unknown"
                and confidence >= ACCUMULATION_FLOOR
                and self._is_cold_start_window(now)):
            boosted_threshold = CONFIDENCE_THRESHOLDS.get("soft", 0.55) - COLD_START_THRESHOLD_REDUCTION
            if confidence >= boosted_threshold:
                is_known = True

        # Accumulate sub-threshold evidence using the real profile name
        accum_name = closest_match if closest_match else name
        if not is_known and accum_name != "unknown" and confidence >= ACCUMULATION_FLOOR:
            self._accumulate_evidence(self._voice_candidates, accum_name, confidence, now)
            self._record_unknown_voice(confidence, closest_match)

        smoothed_short_gap = False
        if (not is_known and accum_name != "unknown"
                and self._should_smooth_voice_gap(
                    prev_voice=prev_voice,
                    candidate_name=accum_name,
                    confidence=confidence,
                    now=now,
                )):
            is_known = True
            name = prev_voice.name
            confidence = max(confidence, prev_voice.confidence * VOICE_SHORT_GAP_CONF_DECAY)
            smoothed_short_gap = True

        self._voice = IdentitySignal(
            name=name if is_known else "unknown",
            confidence=confidence,
            is_known=is_known,
            timestamp=now,
            source="voice",
        )
        self._voice_gap_smoothed = smoothed_short_gap
        if smoothed_short_gap:
            self._voice_gap_smoothed_at = now
        self._resolve()

    def _should_smooth_voice_gap(
        self,
        *,
        prev_voice: IdentitySignal,
        candidate_name: str,
        confidence: float,
        now: float,
    ) -> bool:
        """Return True when a brief sub-threshold dip should inherit prior voice identity."""
        if candidate_name == "unknown":
            return False
        if not prev_voice.is_known or prev_voice.name == "unknown":
            return False
        if now - prev_voice.timestamp > VOICE_SHORT_GAP_HOLD_S:
            return False
        if candidate_name != prev_voice.name:
            return False
        if confidence < VOICE_SHORT_GAP_MIN_CONFIDENCE:
            return False
        if self._visible_person_count > 1 and confidence < MULTI_PERSON_VOICE_THRESHOLD:
            return False

        # Never smooth across an active conflicting face identity.
        face_fresh = (
            self._face.is_known
            and self._face.name != "unknown"
            and self._face.timestamp
            and (now - self._face.timestamp) < STALE_FACE_S
        )
        if face_fresh and self._face.name != prev_voice.name:
            return False
        return True

    def _on_face(self, name="unknown", confidence=0.0, is_known=False, closest_match="", **_):
        if not self._enabled:
            return
        with self._lock:
            self._on_face_locked(name, confidence, is_known, closest_match)
            emit_payload = self._pending_emit
            self._pending_emit = None
        self._emit_resolved(emit_payload)

    def _on_face_locked(self, name: str, confidence: float, is_known: bool,
                        closest_match: str = "") -> None:
        now = time.time()

        if (not is_known and name != "unknown"
                and confidence >= ACCUMULATION_FLOOR
                and self._is_cold_start_window(now)):
            boosted_threshold = CONFIDENCE_THRESHOLDS.get("soft", 0.55) - COLD_START_THRESHOLD_REDUCTION
            if confidence >= boosted_threshold:
                is_known = True

        accum_name = closest_match if closest_match else name
        if not is_known and accum_name != "unknown" and confidence >= ACCUMULATION_FLOOR:
            self._accumulate_evidence(self._face_candidates, accum_name, confidence, now)

        self._face = IdentitySignal(
            name=name if is_known else "unknown",
            confidence=confidence,
            is_known=is_known,
            timestamp=now,
            source="face",
        )
        self._resolve()

    def _on_presence(self, present=False, **_):
        with self._lock:
            self._on_presence_locked(present)
            emit_payload = self._pending_emit
            self._pending_emit = None
        self._emit_resolved(emit_payload)

    def _on_presence_locked(self, present: bool) -> None:
        self._user_present = present
        if present:
            if self._recognition_state == "absent":
                self._recognition_state = "unknown_present"
                self._presence_start = time.time()
                self._voice_candidates.clear()
                self._face_candidates.clear()
                self._tentative_name = ""
                self._tentative_confidence = 0.0
        else:
            if self._last_known:
                logger.info("User departed — clearing persisted identity '%s'",
                            self._last_known.name)
            self._last_known = None
            self._persist_cleared_at = time.time()
            self._recognition_state = "absent"
            self._tentative_name = ""
            self._tentative_confidence = 0.0
            self._voice_gap_smoothed = False
            self._voice_gap_smoothed_at = 0.0
            self._conversation_boundary_until = 0.0
            self._conversation_boundary_reason = ""
            self._voice_drop_recent_name = ""
            self._voice_drop_recent_ts = 0.0
            self._voice_drop_recent_method = ""
            self._resolve()

    def _on_playback_complete(self, **_):
        with self._lock:
            now = time.time()
            if (self._user_present
                    and (
                        (self._last_resolved.is_known and self._last_resolved.name != "unknown")
                        or (self._last_known is not None and self._last_known.name != "unknown")
                    )):
                self._activate_boundary_grace("post_playback", now=now)

    def _activate_boundary_grace(self, reason: str, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        self._conversation_boundary_until = ts + CONVERSATION_BOUNDARY_GRACE_S
        self._conversation_boundary_reason = reason

    def _boundary_grace_active(self, now: float) -> bool:
        return now < self._conversation_boundary_until

    def _can_preserve_on_boundary_wake(self, now: float) -> bool:
        if not self._boundary_grace_active(now):
            return False
        if not self._user_present:
            return False
        if self._visible_person_count > 1:
            return False
        if self._last_known is None or self._last_known.name == "unknown":
            return False
        if (now - self._last_known.captured_at) > CONVERSATION_BOUNDARY_MAX_SNAPSHOT_AGE_S:
            return False
        return True

    def _can_graceful_degrade_voice_drop(self, face_name: str, voice_confidence: float, now: float) -> bool:
        """Allow brief continuity when voice confidence drops unexpectedly."""
        if self._visible_person_count > 1:
            return False
        if self._voice_drop_recent_name != face_name:
            return False
        if not self._voice_drop_recent_ts:
            return False
        if (now - self._voice_drop_recent_ts) > VOICE_DROP_GRACE_S:
            return False
        if voice_confidence < VOICE_DROP_GRACE_MIN_ACTIVE_CONFIDENCE:
            return False
        return True

    def _on_wake_word(self, **_):
        with self._lock:
            self._on_wake_word_locked()
            emit_payload = self._pending_emit
            self._pending_emit = None
        self._emit_resolved(emit_payload)

    def _on_wake_word_locked(self) -> None:
        now = time.time()
        face_live = (
            self._face.name != "unknown"
            and self._face.is_known
            and self._face.confidence >= CONFIDENCE_THRESHOLDS["soft"]
            and self._face.timestamp
            and (now - self._face.timestamp) < STALE_FACE_S
        )
        if face_live and self._last_known and self._last_known.name == self._face.name:
            logger.info("Wake word — keeping persisted identity '%s' (face confirmed)",
                        self._last_known.name)
        elif self._can_preserve_on_boundary_wake(now):
            logger.info(
                "Wake word — keeping persisted identity '%s' (conversation-boundary grace)",
                self._last_known.name,
            )
        else:
            if self._last_known:
                logger.info("Wake word — clearing persisted identity '%s'%s",
                            self._last_known.name,
                            " (face mismatch)" if face_live and self._last_known and self._last_known.name != self._face.name
                            else " (no face confirmation)")
                self._last_known = None
                self._persist_cleared_at = now
        self._resolve()

    def _accumulate_evidence(
        self, candidates: dict[str, _CandidateEvidence], name: str, confidence: float, now: float,
    ) -> None:
        """Accumulate sub-threshold evidence for a candidate identity."""
        if name not in candidates:
            candidates[name] = _CandidateEvidence(name=name)
        candidates[name].add(confidence, now)

        eff = candidates[name].effective_confidence(now)
        if (candidates[name].signals >= TENTATIVE_MIN_SIGNALS
                and eff >= TENTATIVE_THRESHOLD
                and self._recognition_state == "unknown_present"):
            self._recognition_state = "tentative_match"
            self._tentative_name = name
            self._tentative_confidence = eff
            logger.info(
                "Recognition tentative match: %s (eff=%.2f, signals=%d)",
                name, eff, candidates[name].signals,
            )

        # Cross-modal corroboration: if both modalities point to the same candidate
        other = self._face_candidates if candidates is self._voice_candidates else self._voice_candidates
        if name in other:
            combined = eff + other[name].effective_confidence(now)
            if combined >= TENTATIVE_THRESHOLD and self._recognition_state == "unknown_present":
                self._recognition_state = "tentative_match"
                self._tentative_name = name
                self._tentative_confidence = combined
                logger.info(
                    "Recognition tentative match (cross-modal): %s (combined=%.2f)", name, combined,
                )

    def _is_cold_start_window(self, now: float) -> bool:
        """True if we're in the cold-start boost window after user presence detected."""
        if not self._has_profiles:
            return False
        if not self._presence_start:
            return False
        return (now - self._presence_start) < COLD_START_BOOST_WINDOW_S

    def _update_recognition_state(self, resolved: ResolvedIdentity) -> None:
        """Update the 3-state recognition machine based on the latest resolution."""
        if not self._user_present:
            self._recognition_state = "absent"
            return
        if resolved.is_known and resolved.name != "unknown":
            self._recognition_state = "confirmed_match"
            self._tentative_name = ""
            self._tentative_confidence = 0.0
        elif self._recognition_state == "confirmed_match" and not resolved.is_known:
            self._recognition_state = "unknown_present"
        # tentative_match stays until confirmed or user departs

    def set_has_profiles(self, val: bool) -> None:
        """Inform fusion that biometric profiles exist for cold-start boost."""
        self._has_profiles = val

    def set_visible_persons(self, count: int) -> None:
        """Update the count of visually detected persons (from VisionProcessor)."""
        with self._lock:
            self._visible_person_count = max(0, count)

    def _record_unknown_voice(self, confidence: float, closest_match: str) -> None:
        """Record a provisional unknown voice event for later grounding."""
        now = time.time()
        event = {
            "timestamp": now,
            "confidence": confidence,
            "closest_match": closest_match,
            "visible_persons": self._visible_person_count,
            "face_name": self._face.name if self._face.is_known else "",
            "reason": "voice_below_threshold",
        }
        self._unknown_voice_events.append(event)
        if len(self._unknown_voice_events) > self._UNKNOWN_VOICE_MAX:
            self._unknown_voice_events = self._unknown_voice_events[-self._UNKNOWN_VOICE_MAX:]

    def get_unknown_voice_events(self, max_age_s: float = 300.0) -> list[dict[str, Any]]:
        """Return recent unknown voice events for curiosity/clarification flows."""
        now = time.time()
        with self._lock:
            return [e for e in self._unknown_voice_events if now - e["timestamp"] < max_age_s]

    def _snapshot_if_eligible(self, resolved: ResolvedIdentity, now: float) -> None:
        """Store a persistence-eligible known identity as the carry-forward snapshot."""
        if (resolved.method in _PERSIST_ELIGIBLE_METHODS
                and resolved.confidence >= CONFIDENCE_THRESHOLDS["soft"]):
            self._last_known = IdentitySnapshot(
                name=resolved.name,
                confidence=resolved.confidence,
                method=resolved.method,
                captured_at=now,
            )

    def _resolve(self) -> None:
        now = time.time()
        voice = self._voice if (now - self._voice.timestamp) < STALE_VOICE_S else IdentitySignal(source="voice")
        face = self._face if (now - self._face.timestamp) < STALE_FACE_S else IdentitySignal(source="face")

        v_known = voice.is_known
        f_known = face.is_known
        multi_person = self._visible_person_count > 1
        self._threshold_assist_active = False
        self._threshold_assist_name = ""

        # Cross-modal boost: face authoritative, voice is quality gate
        if (not v_known and f_known
                and face.confidence >= 0.70
                and voice.confidence >= FACE_CONFIRMED_VOICE_BOOST_THRESHOLD):
            v_known = True
            voice = IdentitySignal(
                name=face.name,
                confidence=voice.confidence,
                is_known=True,
                timestamp=voice.timestamp,
                source="voice",
            )

        if v_known and f_known and voice.name == face.name:
            combined_conf = min(1.0, max(voice.confidence, face.confidence) + VERIFIED_BOOST)
            resolved = ResolvedIdentity(
                name=voice.name,
                confidence=combined_conf,
                is_known=True,
                method="verified_both",
                voice_name=voice.name,
                face_name=face.name,
            )
            self._conflict_start = 0.0
            self._snapshot_if_eligible(resolved, now)
            self._resolution_basis = "voice_face_agree"
            self._voice_trust_state = "trusted"
            self._trust_reason = "verified_both" + (", multi_person" if multi_person else "")

        elif v_known and f_known and voice.name != face.name:
            if self._conflict_start == 0:
                self._conflict_start = now
            if now - self._conflict_start < CONFLICT_HOLD_S:
                resolved = ResolvedIdentity(
                    name=self._last_resolved.name if self._last_resolved.is_known else "unknown",
                    confidence=0.0,
                    is_known=self._last_resolved.is_known,
                    method="conflict_hold",
                    voice_name=voice.name,
                    face_name=face.name,
                    conflict=True,
                )
                self._resolution_basis = "conflicted"
            else:
                winner = voice if voice.confidence >= face.confidence else face
                resolved = ResolvedIdentity(
                    name=winner.name,
                    confidence=winner.confidence * 0.7,
                    is_known=True,
                    method=f"conflict_resolved_{winner.source}",
                    voice_name=voice.name,
                    face_name=face.name,
                    conflict=True,
                )
                self._snapshot_if_eligible(resolved, now)
                self._resolution_basis = f"conflict_resolved_{winner.source}"
            self._voice_trust_state = "conflicted"
            self._trust_reason = f"voice={voice.name} vs face={face.name}"

        elif v_known and not f_known:
            # Multi-speaker suppression: when multiple persons visible,
            # require higher voice confidence for voice-only promotion
            if multi_person and voice.confidence < MULTI_PERSON_VOICE_THRESHOLD:
                resolved = ResolvedIdentity(
                    name="unknown",
                    confidence=0.0,
                    is_known=False,
                    method="suppressed_multi_person",
                    voice_name=voice.name,
                    face_name=face.name,
                )
                self._resolution_basis = "suppressed_multi_person"
                self._voice_trust_state = "degraded"
                self._trust_reason = f"multi_person ({self._visible_person_count}), voice_only insufficient"
            else:
                resolved = ResolvedIdentity(
                    name=voice.name,
                    confidence=voice.confidence,
                    is_known=True,
                    method="voice_only",
                    voice_name=voice.name,
                    face_name=face.name,
                )
                self._conflict_start = 0.0
                self._snapshot_if_eligible(resolved, now)
                self._resolution_basis = "voice_only"
                self._voice_trust_state = "trusted" if not multi_person else "tentative"
                self._trust_reason = "voice_only" + (f", multi_person ({self._visible_person_count})" if multi_person else "")

        elif f_known and not v_known:
            voice_is_active = voice.timestamp > 0
            voice_clearly_different = (
                voice_is_active and voice.confidence < VOICE_ACTIVE_MISMATCH_CEILING
            )
            if voice_clearly_different:
                if self._can_graceful_degrade_voice_drop(face.name, voice.confidence, now):
                    resolved = ResolvedIdentity(
                        name=face.name,
                        confidence=max(0.5, face.confidence * 0.8),
                        is_known=True,
                        method="face_voice_drop_grace",
                        voice_name=voice.name,
                        face_name=face.name,
                    )
                    self._resolution_basis = "face_voice_drop_grace"
                    self._voice_trust_state = "degraded"
                    self._trust_reason = "brief voice-drop continuity from recent identity"
                else:
                    resolved = ResolvedIdentity(
                        name="unknown",
                        confidence=0.0,
                        is_known=False,
                        method="face_present_voice_unknown",
                        voice_name=voice.name,
                        face_name=face.name,
                    )
                    self._resolution_basis = "face_present_voice_unknown"
                    self._voice_trust_state = "degraded"
                    self._trust_reason = "different speaker than face-identified user"
            else:
                resolved = ResolvedIdentity(
                    name=face.name,
                    confidence=face.confidence,
                    is_known=True,
                    method="face_only",
                    voice_name=voice.name,
                    face_name=face.name,
                )
                self._snapshot_if_eligible(resolved, now)
                self._resolution_basis = "face_only"
                self._voice_trust_state = "tentative"
                self._trust_reason = "face_only, no voice confirmation"
            self._conflict_start = 0.0

        else:
            self._conflict_start = 0.0
            if (self._last_known
                    and self._last_known.name != "unknown"
                    and self._user_present
                    and (now - self._last_known.captured_at) < PERSIST_MAX_S):
                elapsed = now - self._last_known.captured_at
                decayed_conf = self._last_known.confidence * (
                    0.5 ** (elapsed / PERSIST_CONFIDENCE_HALF_LIFE_S)
                )
                persist_conf = max(PERSIST_CONFIDENCE_FLOOR, decayed_conf)
                resolved = ResolvedIdentity(
                    name=self._last_known.name,
                    confidence=persist_conf,
                    is_known=True,
                    method="persisted",
                    voice_name=voice.name,
                    face_name=face.name,
                )
                self._resolution_basis = "persisted"
                self._voice_trust_state = "tentative"
                self._trust_reason = "persisted from prior identity"

            elif (self._recognition_state == "tentative_match"
                    and self._tentative_name
                    and not multi_person
                    and self._tentative_confidence > 0
                    and self._presence_start
                    and (now - self._presence_start) < TENTATIVE_MAX_AGE_S + COLD_START_BOOST_WINDOW_S):
                # Tentative bridge: use accumulated cross-signal evidence
                tentative_conf = max(0.35, self._tentative_confidence * 0.8)
                resolved = ResolvedIdentity(
                    name=self._tentative_name,
                    confidence=tentative_conf,
                    is_known=True,
                    method="tentative_bridge",
                    voice_name=voice.name,
                    face_name=face.name,
                )
                self._threshold_assist_active = True
                self._threshold_assist_name = self._tentative_name
                self._resolution_basis = "tentative_bridge"
                self._voice_trust_state = "tentative"
                self._trust_reason = f"tentative evidence for {self._tentative_name}"
                logger.info("Tentative bridge: %s (conf=%.2f, evidence=%.2f)",
                            self._tentative_name, tentative_conf, self._tentative_confidence)
            else:
                resolved = ResolvedIdentity(method="no_signal")
                self._resolution_basis = "no_signal"
                self._voice_trust_state = "unknown"
                self._trust_reason = "no signals"

        if resolved.is_known and resolved.name != "unknown":
            self._voice_drop_recent_name = resolved.name
            self._voice_drop_recent_ts = now
            self._voice_drop_recent_method = resolved.method

        prev_name = self._last_resolved.name
        if (resolved.name != prev_name
                and prev_name != "unknown"
                and resolved.name != "unknown"):
            self._flip_count += 1

        if (resolved.name != self._last_resolved.name
                or resolved.method != self._last_resolved.method
                or resolved.conflict != self._last_resolved.conflict):
            self._last_resolved = resolved
            self._update_recognition_state(resolved)
            self._pending_emit = dict(
                name=resolved.name,
                confidence=resolved.confidence,
                is_known=resolved.is_known,
                method=resolved.method,
                conflict=resolved.conflict,
                voice_name=resolved.voice_name,
                face_name=resolved.face_name,
            )
            if resolved.is_known:
                logger.info("Identity resolved: %s via %s (conf=%.2f%s)",
                            resolved.name, resolved.method, resolved.confidence,
                            " CONFLICT" if resolved.conflict else "")
        elif resolved.method == "persisted":
            self._last_resolved = resolved
            self._update_recognition_state(resolved)
        else:
            self._update_recognition_state(resolved)

    def _check_expiry(self) -> None:
        """Re-resolve if current identity is based on signals that have gone stale."""
        method = self._last_resolved.method
        if method in ("no_signal", "persisted"):
            return
        now = time.time()
        voice_stale = (now - self._voice.timestamp) >= STALE_VOICE_S if self._voice.timestamp else True
        face_stale = (now - self._face.timestamp) >= STALE_FACE_S if self._face.timestamp else True
        needs_recheck = False
        if method in ("voice_only", "conflict_resolved_voice") and voice_stale:
            needs_recheck = True
        elif method in ("face_only", "conflict_resolved_face") and face_stale:
            needs_recheck = True
        elif method == "verified_both" and (voice_stale or face_stale):
            needs_recheck = True
        elif method == "conflict_hold" and (voice_stale or face_stale):
            needs_recheck = True
        if needs_recheck:
            self._resolve()

    @property
    def current(self) -> ResolvedIdentity:
        with self._lock:
            self._check_expiry()
            result = self._last_resolved
            emit_payload = self._pending_emit
            self._pending_emit = None
        self._emit_resolved(emit_payload)
        return result

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = self._get_status_locked()
            emit_payload = self._pending_emit
            self._pending_emit = None
        self._emit_resolved(emit_payload)
        return status

    def _get_status_locked(self) -> dict[str, Any]:
        self._check_expiry()
        now = time.time()
        lk = self._last_known
        return {
            "enabled": self._enabled,
            "identity": self._last_resolved.name,
            "confidence": round(self._last_resolved.confidence, 3),
            "method": self._last_resolved.method,
            "is_known": self._last_resolved.is_known,
            "conflict": self._last_resolved.conflict,
            "voice_signal": {
                "name": self._voice.name,
                "confidence": round(self._voice.confidence, 3),
                "age_s": round(now - self._voice.timestamp, 1) if self._voice.timestamp else 0,
            },
            "face_signal": {
                "name": self._face.name,
                "confidence": round(self._face.confidence, 3),
                "age_s": round(now - self._face.timestamp, 1) if self._face.timestamp else 0,
            },
            "persisted": self._last_resolved.method == "persisted",
            "snapshot_age_s": round(now - lk.captured_at, 1) if lk else 0,
            "persist_elapsed_s": round(now - lk.captured_at, 1) if lk and self._last_resolved.method == "persisted" else 0,
            "persist_remaining_s": round(max(0, PERSIST_MAX_S - (now - lk.captured_at)), 1) if lk and self._last_resolved.method == "persisted" else 0,
            "last_known_identity": lk.name if lk else None,
            "last_known_confidence": round(lk.confidence, 3) if lk else 0,
            "user_present": self._user_present,
            "flip_count": self._flip_count,
            "recognition_state": self._recognition_state,
            "tentative_name": self._tentative_name if self._recognition_state == "tentative_match" else None,
            "tentative_confidence": round(self._tentative_confidence, 3) if self._recognition_state == "tentative_match" else 0,
            "cold_start_active": self._is_cold_start_window(now),
            # Wave 3: multi-speaker, trust state, resolution basis
            "voice_trust_state": self._voice_trust_state,
            "trust_reason": self._trust_reason,
            "resolution_basis": self._resolution_basis,
            "visible_person_count": self._visible_person_count,
            "multi_person_suppression_active": (
                self._visible_person_count > 1
                and self._voice_trust_state in ("degraded", "conflicted")
            ),
            "threshold_assist_active": self._threshold_assist_active,
            "threshold_assist_name": self._threshold_assist_name if self._threshold_assist_active else None,
            "unknown_voice_count": len(self._unknown_voice_events),
            "voice_gap_smoothed": self._voice_gap_smoothed,
            "voice_gap_smoothed_age_s": (
                round(now - self._voice_gap_smoothed_at, 1) if self._voice_gap_smoothed_at else 0
            ),
            "conversation_boundary_grace_active": self._boundary_grace_active(now),
            "conversation_boundary_grace_remaining_s": round(
                max(0.0, self._conversation_boundary_until - now), 1,
            ),
            "conversation_boundary_grace_reason": self._conversation_boundary_reason,
            "voice_drop_recent_identity": self._voice_drop_recent_name or None,
            "voice_drop_recent_age_s": (
                round(now - self._voice_drop_recent_ts, 1) if self._voice_drop_recent_ts else 0
            ),
            "voice_drop_recent_method": self._voice_drop_recent_method or None,
        }

    def get_state(self) -> dict[str, Any]:
        return self.get_status()
