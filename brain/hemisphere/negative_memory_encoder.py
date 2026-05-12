"""Negative-memory feature encoding for the NEGATIVE_MEMORY Tier-2 specialist.

This is the second Tier-2 Matrix Protocol specialist (P3.7) and the natural
pair to ``positive_memory`` (P3.6). It follows the same template:

  * It writes nothing — no memories, beliefs, identity, autonomy, policy
    authority, HRR/P5 state, Soul Integrity, or events. It is pure feature
    engineering.
  * It enters CANDIDATE_BIRTH only. Promotion to BROADCAST_ELIGIBLE /
    PROMOTED is gated by the standard Matrix Protocol lifecycle in
    ``HemisphereOrchestrator._check_specialist_promotions``; this module
    does not bypass any of it.
  * It produces a real-time inferable scalar in ``[0, 1]`` from current
    perception/memory/system-friction state. It does NOT fall back to
    accuracy-as-proxy. That is the explicit Tier-2 contract: a specialist
    that returns ``performance.accuracy`` when its own inference path is
    unavailable would silently look like a working signal during
    CANDIDATE_BIRTH (where accuracy is 0.0) and would misrank in the
    broadcast-slot competition.

The encoder is *negative-valence-leaning*, and crucially also taps the
system-friction surface (quarantine pressure, contradiction debt, Tier-1
distillation failures) instead of solely tag-matching on memory text. A
``negative_memory`` signal that only fired on negative-tagged memories
would be too easy to starve; combining memory negativity with epistemic
friction telemetry makes the signal robust at low memory volume.

Dimension layout (16-dim total, all values clamped to ``[0, 1]``):

  Block A (dims  0-7):  Memory negativity
  Block B (dims  8-11): Friction / regression episodes
  Block C (dims 12-15): Quarantine / coherence-debt

The signal value returned by :func:`compute_signal_value` is a weighted
aggregate of those blocks (Block A weighted highest because tagged
memory negativity is the canonical signal source for ``negative_memory``;
Block C carries the system-friction backstop that keeps the signal
honest under low memory volume).

Watch item carried from P3.6: ``emotion_depth`` is currently a
reconstruction-fidelity scalar, not a calibrated negative-class
probability. Wiring it here would silently inflate the negative_memory
signal with accuracy data — exactly the failure mode P3.6/P3.7 forbid.
We therefore do NOT consume any emotion_depth output until a calibrated
valence head exists; ``emotion_negative_bias`` is left at 0.0.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


FEATURE_DIM = 16

# The :class:`HemisphereFocus` value this encoder serves. Kept as a
# string literal (not an enum import) so the schema-emission audit's
# writer-literal scan recognises this module as the live writer for
# ``negative_memory`` and removes the future-only whitelist entry.
FOCUS_NAME = "negative_memory"


# Canonical negative-leaning tag substrings. Tags are short discrete
# labels in this codebase (not free text), so substring containment is a
# safe and cheap valence signal. The set is intentionally conservative —
# generic words like "bad" are excluded because they appear in
# neutral/idiom contexts ("bad joke" used affectionately). Add to this
# set with care; over-broad matches cause false-positive negativity
# inflation, which would over-weight ``negative_memory`` in the
# broadcast-slot competition.
NEGATIVE_TAG_SUBSTRINGS: frozenset[str] = frozenset({
    "error",
    "errors",
    "wrong",
    "mistake",
    "mistaken",
    "incorrect",
    "correction",
    "corrected",
    "user_correction",
    "friction",
    "regression",
    "regressed",
    "failure",
    "failed",
    "bug",
    "bugs",
    "issue",
    "issues",
    "problem",
    "problems",
    "rejected",
    "discarded",
    "quarantine",
    "quarantined",
    "denied",
    "conflict",
    "conflicts",
    "contradiction",
    "contradicts",
    "contradicting",
    "frustration",
    "frustrated",
    "angry",
    "anger",
    "upset",
    "sad",
    "sadness",
    "disappointed",
    "disappointment",
    "annoyed",
    "annoying",
    "unsuccessful",
    "complaint",
    "complained",
})


# Tag substrings that indicate an explicit *correction* event — a strict
# subset of NEGATIVE_TAG_SUBSTRINGS used for Block A's correction
# fraction feature. Corrections carry stronger weight than raw negativity
# because they represent confirmed friction events that JARVIS is
# expected to learn from rather than ambient negative-affect tagging.
CORRECTION_TAG_SUBSTRINGS: frozenset[str] = frozenset({
    "correction",
    "corrected",
    "user_correction",
    "amended",
    "retraction",
    "retracted",
    "overridden",
    "revoked",
})


# Recent-memory window for negativity aggregation. Kept small so a fresh
# friction event is felt quickly instead of being washed out by a long
# history of neutral observation.
RECENT_MEMORY_WINDOW = 32

# Recent-episode window (friction / regression block).
RECENT_EPISODE_WINDOW = 16


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


def _tag_matches_any(tags: Iterable[Any], substrings: frozenset[str]) -> bool:
    """Return True if any tag's lowercase form contains any substring."""
    if not tags:
        return False
    for tag in tags:
        if tag is None:
            continue
        try:
            text = str(tag).lower()
        except Exception:
            continue
        if not text:
            continue
        for substr in substrings:
            if substr in text:
                return True
    return False


def _has_negative_tag(tags: Iterable[Any]) -> bool:
    return _tag_matches_any(tags, NEGATIVE_TAG_SUBSTRINGS)


def _has_correction_tag(tags: Iterable[Any]) -> bool:
    return _tag_matches_any(tags, CORRECTION_TAG_SUBSTRINGS)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class NegativeMemoryEncoder:
    """Encodes current system state into a 16-dim ``[0, 1]`` feature vector.

    All public methods are pure (no side effects). They take plain Python
    dicts / sequences, never live singletons, so the encoder can be unit-
    tested without standing up the brain stack.

    The orchestrator-level helper
    :meth:`HemisphereOrchestrator._build_negative_memory_context` is
    responsible for gathering the live state and passing it in.
    """

    FEATURE_DIM = FEATURE_DIM

    # ------------------------------------------------------------------
    # Block A: Memory negativity (8 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_memory_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block A: 8 dims of memory-negativity features.

        Expected keys (all optional, missing fields contribute 0.0):

          * ``recent_memories`` — sequence of memory objects/dicts with
            attributes/keys ``weight``, ``tags``, ``decay_rate``.
          * ``memory_density`` — pre-normalised in ``[0, 1]``.
          * ``max_memories`` — capacity used for density fallback.
        """
        recent = list(_safe_attr(ctx, "recent_memories", []) or [])
        recent = recent[-RECENT_MEMORY_WINDOW:]

        # 0: latest memory salience (weight). Salience is direction-
        # neutral — a vivid memory is louder regardless of valence — but
        # weight gates how strongly the *other* features are read.
        latest = recent[-1] if recent else None
        latest_weight = _clamp(float(_safe_attr(latest, "weight", 0.0) or 0.0))

        # 1: latest memory has negative tag (binary).
        latest_tags = _safe_attr(latest, "tags", ()) or ()
        latest_neg = 1.0 if _has_negative_tag(latest_tags) else 0.0

        # 2: recent-window fraction with at least one negative tag.
        if recent:
            neg_count = sum(
                1 for m in recent
                if _has_negative_tag(_safe_attr(m, "tags", ()) or ())
            )
            recent_neg_frac = _clamp(neg_count / float(len(recent)))
        else:
            recent_neg_frac = 0.0

        # 3: recent-window mean weight (intensity proxy). Same dim as
        # P3.6 by design — Tier-2 specialists share salience telemetry
        # so the broadcast-slot competition sees comparable amplitudes.
        recent_weights = [
            _clamp(float(_safe_attr(m, "weight", 0.0) or 0.0))
            for m in recent
        ]
        recent_mean_weight = _clamp(_mean(recent_weights))

        # 4: recent-window correction-tag fraction. This is the strongest
        # *intentional* negative signal: an explicit correction or
        # retraction. Distinguishing it from generic negativity lets the
        # downstream policy treat "user disagreed with me" louder than
        # "user mentioned an annoying event".
        if recent:
            corr_count = sum(
                1 for m in recent
                if _has_correction_tag(_safe_attr(m, "tags", ()) or ())
            )
            recent_correction_frac = _clamp(corr_count / float(len(recent)))
        else:
            recent_correction_frac = 0.0

        # 5: memory density (proxy for activity volume — caller may pass
        # a pre-normalised value or we compute from list size).
        density = _safe_attr(ctx, "memory_density", None)
        if density is None:
            max_mem = float(_safe_attr(ctx, "max_memories", 1000) or 1000)
            density = len(recent) / max_mem if max_mem > 0 else 0.0
        memory_density = _clamp(float(density))

        # 6: latest-memory decay rate (capped). High decay without
        # reinforcement is a soft negative signal: the memory is being
        # forgotten despite having been recorded. Gated on having an
        # actual memory; with no observed history we report 0.0 instead
        # of a fictional decay pressure.
        if latest is not None:
            latest_decay = float(_safe_attr(latest, "decay_rate", 0.0) or 0.0)
            latest_decay_pressure = _clamp(min(latest_decay * 10.0, 1.0))
        else:
            latest_decay_pressure = 0.0

        # 7: reinforced-negative fraction — both negative-tagged AND
        # high-weight (>= 0.6). This is the *strongest* negativity
        # signal because it intersects salience with negative labelling:
        # a vivid memory carrying explicit negative tags is the canonical
        # "negative memory" the specialist is meant to surface.
        if recent:
            rn_count = 0
            for m in recent:
                weight = float(_safe_attr(m, "weight", 0.0) or 0.0)
                tags = _safe_attr(m, "tags", ()) or ()
                if weight >= 0.6 and _has_negative_tag(tags):
                    rn_count += 1
            reinforced_neg_frac = _clamp(rn_count / float(len(recent)))
        else:
            reinforced_neg_frac = 0.0

        return [
            latest_weight,
            latest_neg,
            recent_neg_frac,
            recent_mean_weight,
            recent_correction_frac,
            memory_density,
            latest_decay_pressure,
            reinforced_neg_frac,
        ]

    # ------------------------------------------------------------------
    # Block B: Friction / regression episodes (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_episode_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block B: 4 dims of friction / regression features.

        Expected keys (all optional):

          * ``recent_episodes`` — sequence of episode objects/dicts with
            optional ``negative_affect`` (bool / float in ``[0,1]``),
            ``positive_affect`` (used to derive negative_affect when
            negative_affect is absent), ``revisit_count`` (int), and
            ``regressed`` (bool — episode discarded / rolled back).
          * ``tier1_failure_rate`` — fraction of recent Tier-1 distillation
            attempts that failed, in ``[0, 1]``.
        """
        episodes = list(_safe_attr(ctx, "recent_episodes", []) or [])
        episodes = episodes[-RECENT_EPISODE_WINDOW:]

        # 0: episode volume (capped, normalised against window). Same dim
        # as P3.6 — high engagement creates more opportunities for
        # friction events but isn't itself negative; the negativity
        # signal lives in dims 1-2 and Block A.
        episode_volume = _clamp(len(episodes) / float(RECENT_EPISODE_WINDOW))

        # 1: negative-affect fraction. Prefers an explicit
        # ``negative_affect`` field; when absent and ``positive_affect``
        # is present we infer ``1 - positive_affect`` rather than
        # silently zeroing the dim. Episodes with neither field
        # contribute nothing (the encoder does not invent affect).
        if episodes:
            na_values: list[float] = []
            for ep in episodes:
                na = _safe_attr(ep, "negative_affect", None)
                if na is None:
                    pa = _safe_attr(ep, "positive_affect", None)
                    if pa is None:
                        continue
                    if isinstance(pa, bool):
                        na_values.append(0.0 if pa else 1.0)
                    else:
                        try:
                            na_values.append(_clamp(1.0 - float(pa)))
                        except (TypeError, ValueError):
                            continue
                    continue
                if isinstance(na, bool):
                    na_values.append(1.0 if na else 0.0)
                else:
                    try:
                        na_values.append(_clamp(float(na)))
                    except (TypeError, ValueError):
                        continue
            episode_neg_frac = _clamp(_mean(na_values)) if na_values else 0.0
        else:
            episode_neg_frac = 0.0

        # 2: regressed-episode fraction (revisit_count > 1 AND
        # ``regressed`` is True, OR explicit ``discarded`` flag). Captures
        # "we kept revisiting this and it kept going wrong" — exactly
        # the negative-memory archetype.
        if episodes:
            reg_count = 0
            for ep in episodes:
                regressed = bool(_safe_attr(ep, "regressed", False))
                discarded = bool(_safe_attr(ep, "discarded", False))
                revisit = int(_safe_attr(ep, "revisit_count", 0) or 0)
                if discarded or (regressed and revisit > 1):
                    reg_count += 1
            episode_regressed_frac = _clamp(reg_count / float(len(episodes)))
        else:
            episode_regressed_frac = 0.0

        # 3: Tier-1 distillation failure rate. The orchestrator already
        # tracks per-focus failure counts; we expose the normalised rate
        # here as a system-friction signal that does not depend on
        # tagged memory volume. Defaults to 0.0 when the field is absent.
        tier1_failure_rate = _clamp(
            float(_safe_attr(ctx, "tier1_failure_rate", 0.0) or 0.0)
        )

        return [
            episode_volume,
            episode_neg_frac,
            episode_regressed_frac,
            tier1_failure_rate,
        ]

    # ------------------------------------------------------------------
    # Block C: Quarantine / coherence-debt (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_friction_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block C: 4 dims of system-friction features.

        Expected keys (all optional):

          * ``quarantine_pressure`` — composite quarantine pressure in
            ``[0, 1]`` (already normalised by ``QuarantinePressure``).
          * ``contradiction_debt`` — epistemic contradiction debt in
            ``[0, 1]``; consumed *raw* (no inversion) because higher
            debt is a stronger negative_memory signal.
          * ``low_confidence_retrieval_rate`` — fraction of recent
            retrieval attempts that fell below the confidence floor.
            Advisory until a recall-telemetry hook exists.
          * ``regression_pressure`` — recent autonomy/hemisphere
            rollback density in ``[0, 1]``. Advisory until a rollback-
            telemetry hook exists.
        """
        quarantine_pressure = _clamp(
            float(_safe_attr(ctx, "quarantine_pressure", 0.0) or 0.0)
        )

        # ``contradiction_debt`` is gated on the field being present so
        # that an empty context yields 0.0 (matches the empty-context
        # contract). A value of 0.0 with the field present means
        # "measured, currently zero", which still maps to 0.0 — both
        # paths are honest.
        debt_raw = _safe_attr(ctx, "contradiction_debt", None)
        if debt_raw is None:
            contradiction_debt = 0.0
        else:
            try:
                contradiction_debt = _clamp(float(debt_raw))
            except (TypeError, ValueError):
                contradiction_debt = 0.0

        low_conf_retrieval = _clamp(
            float(_safe_attr(ctx, "low_confidence_retrieval_rate", 0.0) or 0.0)
        )
        regression_pressure = _clamp(
            float(_safe_attr(ctx, "regression_pressure", 0.0) or 0.0)
        )

        return [
            quarantine_pressure,
            contradiction_debt,
            low_conf_retrieval,
            regression_pressure,
        ]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @staticmethod
    def encode(ctx: Mapping[str, Any]) -> list[float]:
        """Produce the 16-dim ``[0, 1]`` feature vector from context."""
        block_a = NegativeMemoryEncoder.encode_memory_block(ctx)
        block_b = NegativeMemoryEncoder.encode_episode_block(ctx)
        block_c = NegativeMemoryEncoder.encode_friction_block(ctx)
        vec = block_a + block_b + block_c
        # Hard invariant: dimension and bounds. The orchestrator and the
        # broadcast-slot competition rely on this contract.
        assert len(vec) == FEATURE_DIM, (
            f"NegativeMemoryEncoder.encode produced {len(vec)} dims, "
            f"expected {FEATURE_DIM}"
        )
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"feature[{i}] = {v} out of [0,1]"
        return vec

    @staticmethod
    def compute_signal_value(ctx: Mapping[str, Any]) -> float:
        """Real-time inferable scalar in ``[0, 1]``.

        Weighted aggregate over the three feature blocks:

          * Block A (memory negativity)            at weight 0.50
          * Block B (friction / regression)        at weight 0.20
          * Block C (quarantine / coherence-debt)  at weight 0.30

        Block A still leads because tagged memory negativity is the
        canonical signal source. Block C is weighted *higher* than the
        equivalent block in P3.6 because quarantine pressure and
        contradiction debt are direct epistemic-friction telemetry that
        does not require memory tagging volume to fire — without it, a
        sparsely-tagged memory store would starve the signal even
        during real epistemic distress.

        The blocks are averaged independently then combined; this keeps
        a single very-high feature in one block from saturating the
        signal.

        This function is the explicit replacement for the
        accuracy-as-proxy fallback in
        ``HemisphereOrchestrator._compute_signal_value``. For the
        ``negative_memory`` focus the orchestrator dispatches to this
        function before attempting network inference; the result is
        always defined and always in ``[0, 1]``, even at CANDIDATE_BIRTH
        when the underlying NN is untrained.
        """
        block_a = NegativeMemoryEncoder.encode_memory_block(ctx)
        block_b = NegativeMemoryEncoder.encode_episode_block(ctx)
        block_c = NegativeMemoryEncoder.encode_friction_block(ctx)

        signal = (
            0.50 * _mean(block_a)
            + 0.20 * _mean(block_b)
            + 0.30 * _mean(block_c)
        )
        return _clamp(signal)
