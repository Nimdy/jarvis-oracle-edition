"""Speaker-profile feature encoding for the SPEAKER_PROFILE Tier-2 specialist.

P3.8 — third Tier-2 Matrix Protocol specialist. Same template contract as
P3.6 (``positive_memory``) and P3.7 (``negative_memory``):

  * It writes nothing — no memories, beliefs, identity, autonomy, policy
    authority, HRR/P5 state, Soul Integrity, or events. It is pure feature
    engineering.
  * It enters CANDIDATE_BIRTH only. Promotion is gated by the standard
    Matrix Protocol lifecycle in
    ``HemisphereOrchestrator._check_specialist_promotions``; this module
    does not bypass any of it.
  * It produces a real-time inferable scalar in ``[0, 1]`` from the
    current identity-fusion state, addressee context, and speaker
    registry. It does NOT fall back to accuracy-as-proxy.

**Strict no-embedding-leak contract.** Speaker representations
(``speaker_repr`` 192-dim ECAPA-TDNN vectors) MUST NOT cross the encoder
input boundary. The encoder consumes only scalars, booleans, normalised
counts, and short string labels (e.g. ``voice_trust_state``). This is
enforced two ways:

  * The encoder reads keys from a flat context dict only. It never
    accepts ``embedding`` / ``embeddings`` / ``vector`` / ``vectors`` /
    ``ecapa`` keys, and a regression test
    (``test_encoder_does_not_consume_raw_embeddings``) feeds those keys
    in and asserts the output is unchanged.
  * The orchestrator-level builder
    :meth:`HemisphereOrchestrator._build_speaker_profile_context`
    sources only ``IdentityFusion.get_status()`` (which already excludes
    embeddings by design), the ``IdentityResolver`` known-name set, and
    the soul-service relationship count.

Dimension layout (16-dim total, all values clamped to ``[0, 1]``):

  Block A (dims  0-7):  Identity confidence (current speaker)
  Block B (dims  8-11): Addressee / multi-speaker context
  Block C (dims 12-15): Registry / interaction history

The signal value returned by :func:`compute_signal_value` is a weighted
aggregate of those blocks. Block A leads (current-speaker confidence is
the canonical signal source) and Block B is weighted higher than C
because addressee disambiguation is operationally critical for memory
scoping and addressee grounding.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


FEATURE_DIM = 16

# The :class:`HemisphereFocus` value this encoder serves. Kept as a
# string literal (not an enum import) so the schema-emission audit's
# writer-literal scan recognises this module as the live writer for
# ``speaker_profile`` and removes the future-only whitelist entry.
FOCUS_NAME = "speaker_profile"


# Identity-fusion staleness windows used to normalise voice/face age.
# Mirrors the constants in ``brain/perception/identity_fusion.py`` so a
# voice signal at exactly STALE_VOICE_S age contributes 0.0 freshness;
# fresher signals scale up linearly toward 1.0. We deliberately copy
# instead of importing because the encoder must remain a pure function
# of its context dict and stay unit-testable without the perception
# stack.
STALE_VOICE_S = 30.0
STALE_FACE_S = 90.0


# Voice-trust-state mapping. Strings are the canonical labels emitted by
# IdentityFusion.get_status()[``voice_trust_state``]. Anything outside
# this set degrades to 0.0 (unknown / not yet measured).
_VOICE_TRUST_SCALE: dict[str, float] = {
    "stable": 1.0,
    "verified": 1.0,
    "degraded": 0.5,
    "tentative": 0.5,
    "conflicted": 0.0,
    "unknown": 0.0,
}


# Normalisation caps for Block C. These are intentionally small so a
# brain that's been talked to by a handful of people produces a
# saturating signal — speaker_profile should not require an enormous
# enrolment registry to fire. Tuning is conservative; raising the cap
# reduces signal sensitivity at low volume.
KNOWN_SPEAKERS_CAP = 5
INTERACTION_COUNT_CAP = 50
RELATIONSHIPS_CAP = 5
FLIP_COUNT_CAP = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    """Clamp a value to ``[0, 1]`` (defensive against caller errors)."""
    if v != v:  # NaN guard
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    """Read an attribute or mapping key without raising."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        return str(v).strip()
    except Exception:
        return ""


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class SpeakerProfileEncoder:
    """Encodes current speaker identity / addressee state into a 16-dim
    ``[0, 1]`` feature vector.

    All public methods are pure (no side effects). They take plain
    Python dicts / sequences, never live singletons, so the encoder can
    be unit-tested without standing up the brain stack.

    The orchestrator-level helper
    :meth:`HemisphereOrchestrator._build_speaker_profile_context` is
    responsible for gathering live state from
    :class:`IdentityFusion.get_status` (already embedding-free) and
    passing it in as a flat dict.

    The encoder NEVER consumes raw speaker embeddings. See the
    module-level ``no-embedding-leak contract`` note.
    """

    FEATURE_DIM = FEATURE_DIM

    # ------------------------------------------------------------------
    # Block A: Identity confidence (8 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_identity_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block A: 8 dims of current-speaker confidence features.

        Expected keys (all optional, missing fields contribute 0.0):

          * ``identity_confidence`` — fused confidence in ``[0, 1]``.
          * ``is_known`` — boolean.
          * ``voice_confidence``, ``face_confidence`` — modality-level
            confidence in ``[0, 1]``.
          * ``voice_name``, ``face_name`` — string labels for agreement.
          * ``conflict`` — boolean.
          * ``voice_age_s``, ``face_age_s`` — seconds since modality
            last reported.
        """
        identity_conf = _clamp(_as_float(_safe_attr(ctx, "identity_confidence", 0.0)))
        is_known = 1.0 if bool(_safe_attr(ctx, "is_known", False)) else 0.0
        voice_conf = _clamp(_as_float(_safe_attr(ctx, "voice_confidence", 0.0)))
        face_conf = _clamp(_as_float(_safe_attr(ctx, "face_confidence", 0.0)))

        # 4: voice/face agreement. Only fires when both modalities have
        # non-empty names AND those names match. An empty-string or
        # "unknown" label on either side blocks the agreement signal —
        # we will not credit "both unknown" as agreement.
        voice_name = _as_str(_safe_attr(ctx, "voice_name", ""))
        face_name = _as_str(_safe_attr(ctx, "face_name", ""))
        if (
            voice_name
            and face_name
            and voice_name.lower() != "unknown"
            and face_name.lower() != "unknown"
            and voice_name.lower() == face_name.lower()
        ):
            agreement = 1.0
        else:
            agreement = 0.0

        # 5: no-conflict score. Gated on having any modality signal so
        # an empty context yields 0.0 instead of an optimistic 1.0
        # ("we don't have a conflict because nobody talked").
        has_signal = voice_conf > 0.0 or face_conf > 0.0
        conflict = bool(_safe_attr(ctx, "conflict", False))
        no_conflict = 1.0 if (has_signal and not conflict) else 0.0

        # 6: voice freshness — gated on having a voice signal at all
        # (any non-zero confidence OR any reported age). Fresher signals
        # score higher; STALE_VOICE_S maps to 0.0.
        voice_age_s = _as_float(_safe_attr(ctx, "voice_age_s", 0.0))
        if voice_conf > 0.0 or voice_age_s > 0.0:
            voice_fresh = _clamp(1.0 - (voice_age_s / STALE_VOICE_S))
        else:
            voice_fresh = 0.0

        # 7: face freshness — same gate, longer staleness window.
        face_age_s = _as_float(_safe_attr(ctx, "face_age_s", 0.0))
        if face_conf > 0.0 or face_age_s > 0.0:
            face_fresh = _clamp(1.0 - (face_age_s / STALE_FACE_S))
        else:
            face_fresh = 0.0

        return [
            identity_conf,
            is_known,
            voice_conf,
            face_conf,
            agreement,
            no_conflict,
            voice_fresh,
            face_fresh,
        ]

    # ------------------------------------------------------------------
    # Block B: Addressee / multi-speaker context (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_addressee_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block B: 4 dims of addressee-grounding features.

        Expected keys (all optional):

          * ``flip_count`` — int, identity-flips since boot.
          * ``visible_person_count`` — int.
          * ``multi_person_suppression_active`` — boolean.
          * ``cold_start_active`` — boolean.
          * ``voice_trust_state`` — string label.
        """
        voice_conf = _clamp(_as_float(_safe_attr(ctx, "voice_confidence", 0.0)))
        face_conf = _clamp(_as_float(_safe_attr(ctx, "face_confidence", 0.0)))
        has_signal = voice_conf > 0.0 or face_conf > 0.0

        # 0: stability score from flip_count. Many flips → low stability.
        # Gated on has_signal so an empty context (flip_count 0, no
        # modality activity) yields 0.0 stability rather than a
        # fictional perfect-stability prior.
        flip_count = max(0, int(_as_float(_safe_attr(ctx, "flip_count", 0))))
        if has_signal:
            stability = _clamp(1.0 - (flip_count / float(FLIP_COUNT_CAP)))
        else:
            stability = 0.0

        # 1: no-multi-person-suppression score. Gated on having any
        # visible person count > 0 (no people seen → no suppression
        # statement to make).
        visible_count = max(0, int(_as_float(_safe_attr(ctx, "visible_person_count", 0))))
        suppression = bool(_safe_attr(ctx, "multi_person_suppression_active", False))
        if visible_count > 0:
            no_suppression = 0.0 if suppression else 1.0
        else:
            no_suppression = 0.0

        # 2: not-cold-start score. Gated on has_signal so a fresh boot
        # without modality activity is not credited as "warm".
        cold_start = bool(_safe_attr(ctx, "cold_start_active", False))
        if has_signal:
            not_cold = 0.0 if cold_start else 1.0
        else:
            not_cold = 0.0

        # 3: voice trust scalar. Maps the canonical labels to a [0, 1]
        # scale; unknown / missing labels contribute 0.0.
        voice_trust_state = _as_str(_safe_attr(ctx, "voice_trust_state", ""))
        voice_trust = _VOICE_TRUST_SCALE.get(voice_trust_state.lower(), 0.0)

        return [
            stability,
            no_suppression,
            not_cold,
            voice_trust,
        ]

    # ------------------------------------------------------------------
    # Block C: Registry / interaction history (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_registry_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block C: 4 dims of registry / interaction-history features.

        Expected keys (all optional):

          * ``known_speakers_count`` — int (enrolled voice profiles
            and/or known relationship names).
          * ``current_speaker_interaction_count`` — int.
          * ``relationships_count`` — int (soul-service relationship
            registry size).
          * ``rapport_stability`` — pre-normalised in ``[0, 1]``;
            currently advisory, defaults to 0.0 until a calibrated
            rapport-stability accessor exists.
        """
        known_speakers = max(0, int(_as_float(_safe_attr(ctx, "known_speakers_count", 0))))
        known_frac = _clamp(known_speakers / float(KNOWN_SPEAKERS_CAP))

        interactions = max(0, int(_as_float(_safe_attr(ctx, "current_speaker_interaction_count", 0))))
        interaction_frac = _clamp(interactions / float(INTERACTION_COUNT_CAP))

        relationships = max(0, int(_as_float(_safe_attr(ctx, "relationships_count", 0))))
        relationships_frac = _clamp(relationships / float(RELATIONSHIPS_CAP))

        rapport_stability = _clamp(_as_float(_safe_attr(ctx, "rapport_stability", 0.0)))

        return [
            known_frac,
            interaction_frac,
            relationships_frac,
            rapport_stability,
        ]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @staticmethod
    def encode(ctx: Mapping[str, Any]) -> list[float]:
        """Produce the 16-dim ``[0, 1]`` feature vector from context."""
        block_a = SpeakerProfileEncoder.encode_identity_block(ctx)
        block_b = SpeakerProfileEncoder.encode_addressee_block(ctx)
        block_c = SpeakerProfileEncoder.encode_registry_block(ctx)
        vec = block_a + block_b + block_c
        # Hard invariant: dimension and bounds. The orchestrator and the
        # broadcast-slot competition rely on this contract.
        assert len(vec) == FEATURE_DIM, (
            f"SpeakerProfileEncoder.encode produced {len(vec)} dims, "
            f"expected {FEATURE_DIM}"
        )
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"feature[{i}] = {v} out of [0,1]"
        return vec

    @staticmethod
    def compute_signal_value(ctx: Mapping[str, Any]) -> float:
        """Real-time inferable scalar in ``[0, 1]``.

        Weighted aggregate over the three feature blocks:

          * Block A (identity confidence)         at weight 0.50
          * Block B (addressee / multi-speaker)   at weight 0.30
          * Block C (registry / interaction)      at weight 0.20

        Block A leads because current-speaker confidence is the
        canonical signal source for ``speaker_profile``. Block B is
        weighted *higher* than the registry block because addressee
        disambiguation is operationally critical for memory scoping
        and addressee grounding — a brain that knows "someone is
        speaking" but cannot stably ground who-to whom should not
        broadcast a high speaker_profile signal.

        The blocks are averaged independently then combined; this
        keeps a single very-high feature in one block from saturating
        the signal.

        This function is the explicit replacement for the
        accuracy-as-proxy fallback in
        ``HemisphereOrchestrator._compute_signal_value``. For the
        ``speaker_profile`` focus the orchestrator dispatches to this
        function before attempting network inference; the result is
        always defined and always in ``[0, 1]``, even at
        CANDIDATE_BIRTH when the underlying NN is untrained.
        """
        block_a = SpeakerProfileEncoder.encode_identity_block(ctx)
        block_b = SpeakerProfileEncoder.encode_addressee_block(ctx)
        block_c = SpeakerProfileEncoder.encode_registry_block(ctx)

        signal = (
            0.50 * _mean(block_a)
            + 0.30 * _mean(block_b)
            + 0.20 * _mean(block_c)
        )
        return _clamp(signal)
