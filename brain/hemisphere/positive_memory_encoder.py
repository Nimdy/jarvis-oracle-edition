"""Positive-memory feature encoding for the POSITIVE_MEMORY Tier-2 specialist.

This is the first Tier-2 Matrix Protocol specialist (P3.6) and the template
for the rest of the Tier-2 ladder (P3.7-P3.10). It is **derived-only** and
**shadow-only**:

  * It writes nothing — no memories, beliefs, identity, autonomy, policy
    authority, HRR/P5 state, or events. It is pure feature engineering.
  * It enters CANDIDATE_BIRTH only. Promotion to BROADCAST_ELIGIBLE / PROMOTED
    is gated by the standard Matrix Protocol lifecycle in
    ``HemisphereOrchestrator._check_specialist_promotions``; this module
    does not bypass any of it.
  * It produces a real-time inferable scalar in ``[0, 1]`` from current
    perception/memory/mood state. It does NOT fall back to
    accuracy-as-proxy. That is the explicit P3.6 contract: a Tier-2
    specialist that returns ``performance.accuracy`` when its own
    inference path is unavailable would silently look like a working
    signal during CANDIDATE_BIRTH (where accuracy is 0.0) and would
    misrank in the broadcast-slot competition.

The encoder is *valence-leaning*, not "is the memory positive at all
costs": it picks up positive-tag presence, reinforced episodic recall,
and positive mood bias as additive blocks. Each block is defensive — any
missing data field contributes 0.0 instead of raising.

Dimension layout (16-dim total, all values clamped to ``[0, 1]``):

  Block A (dims  0-7):  Memory valence
  Block B (dims  8-11): Episodic reinforcement
  Block C (dims 12-15): Mood / identity positive bias

The signal value returned by :func:`compute_signal_value` is a weighted
aggregate of those blocks (Block A weighted highest, since memory
valence is the canonical signal source for ``positive_memory``).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


FEATURE_DIM = 16

# The :class:`HemisphereFocus` value this encoder serves. Kept as a
# string literal (not an enum import) so the schema-emission audit's
# writer-literal scan recognises this module as the live writer for
# ``positive_memory`` and removes the future-only whitelist entry.
FOCUS_NAME = "positive_memory"


# Canonical positive-leaning tag substrings. Tags are short discrete
# labels in this codebase (not free text), so substring containment is a
# safe and cheap valence signal. The set is intentionally conservative —
# generic words like "great" are excluded because they appear in
# negative contexts ("great loss"). Add to this set with care; over-
# broad matches cause false-positive valence inflation.
POSITIVE_TAG_SUBSTRINGS: frozenset[str] = frozenset({
    "happy",
    "happiness",
    "joy",
    "joyful",
    "delight",
    "delighted",
    "appreciation",
    "appreciated",
    "gratitude",
    "grateful",
    "thanks",
    "thankful",
    "achievement",
    "accomplishment",
    "completed",
    "milestone",
    "won",
    "win",
    "success",
    "successful",
    "improvement",
    "progress",
    "praise",
    "encouragement",
    "encouraged",
    "satisfied",
    "satisfaction",
    "celebration",
    "celebrated",
    "love",
    "loving",
    "kind",
    "kindness",
    "warm",
    "warmth",
    "calm",
    "peaceful",
    "pleasant",
    "fun",
    "playful",
})


# Positive-leaning trait labels recognised by the personality system.
# These are read from the live identity snapshot when available; missing
# traits contribute 0.0.
POSITIVE_TRAIT_LABELS: frozenset[str] = frozenset({
    "Empathetic",
    "Proactive",
    "Optimistic",
    "Curious",
    "Playful",
    "Patient",
    "Supportive",
})


# Recent-memory window for valence aggregation. Kept small so a fresh
# positive interaction is felt quickly instead of being washed out by a
# long history.
RECENT_MEMORY_WINDOW = 32

# Recent-episode window (episodic reinforcement block).
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


def _has_positive_tag(tags: Iterable[Any]) -> bool:
    """Return True if any tag contains a positive substring.

    Tags can be strings or arbitrary objects with a ``str`` representation;
    we lower-case once and substring-match. False on empty/None.
    """
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
        for substr in POSITIVE_TAG_SUBSTRINGS:
            if substr in text:
                return True
    return False


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class PositiveMemoryEncoder:
    """Encodes current system state into a 16-dim ``[0, 1]`` feature vector.

    All public methods are pure (no side effects). They take plain Python
    dicts / sequences, never live singletons, so the encoder can be unit-
    tested without standing up the brain stack.

    The orchestrator-level helper
    :meth:`HemisphereOrchestrator._build_positive_memory_context` is
    responsible for gathering the live state and passing it in.
    """

    FEATURE_DIM = FEATURE_DIM

    # ------------------------------------------------------------------
    # Block A: Memory valence (8 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_memory_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block A: 8 dims of memory-valence features.

        Expected keys (all optional, missing fields contribute 0.0):

          * ``recent_memories`` — sequence of memory objects/dicts with
            attributes/keys ``weight``, ``tags``, ``type``, ``decay_rate``.
          * ``memory_density`` — pre-normalised in ``[0, 1]``.
          * ``max_memories`` — capacity used for density fallback.
        """
        recent = list(_safe_attr(ctx, "recent_memories", []) or [])
        recent = recent[-RECENT_MEMORY_WINDOW:]

        # 0: latest memory salience (weight)
        latest = recent[-1] if recent else None
        latest_weight = _clamp(float(_safe_attr(latest, "weight", 0.0) or 0.0))

        # 1: latest memory has positive tag (binary)
        latest_tags = _safe_attr(latest, "tags", ()) or ()
        latest_pos = 1.0 if _has_positive_tag(latest_tags) else 0.0

        # 2: recent-window fraction with at least one positive tag
        if recent:
            pos_count = sum(
                1 for m in recent
                if _has_positive_tag(_safe_attr(m, "tags", ()) or ())
            )
            recent_pos_frac = _clamp(pos_count / float(len(recent)))
        else:
            recent_pos_frac = 0.0

        # 3: recent-window mean weight (intensity proxy)
        recent_weights = [
            _clamp(float(_safe_attr(m, "weight", 0.0) or 0.0))
            for m in recent
        ]
        recent_mean_weight = _clamp(_mean(recent_weights))

        # 4: recent-window fraction marked as ``core`` type (long-term
        # reinforced memory); core memories carry strong identity weight
        # and meaningfully shift the positive-memory baseline.
        if recent:
            core_count = sum(
                1 for m in recent
                if _safe_attr(m, "type", "observation") == "core"
            )
            recent_core_frac = _clamp(core_count / float(len(recent)))
        else:
            recent_core_frac = 0.0

        # 5: memory density (proxy for activity volume — caller may pass
        # a pre-normalised value or we compute from list size)
        density = _safe_attr(ctx, "memory_density", None)
        if density is None:
            max_mem = float(_safe_attr(ctx, "max_memories", 1000) or 1000)
            density = len(recent) / max_mem if max_mem > 0 else 0.0
        memory_density = _clamp(float(density))

        # 6: latest memory persistence (1 - decay_rate*10, clamped). Gated
        # on having an actual memory: with no observed history we report
        # 0.0 instead of an optimistic prior. This keeps a CANDIDATE_BIRTH
        # specialist with no input data from broadcasting a fictional
        # positive baseline.
        if latest is not None:
            latest_decay = float(_safe_attr(latest, "decay_rate", 0.01) or 0.01)
            latest_persistence = _clamp(1.0 - min(latest_decay * 10.0, 1.0))
        else:
            latest_persistence = 0.0

        # 7: reinforced-positive fraction — both positive-tagged AND
        # high-weight (>= 0.6). This is the *strongest* valence signal
        # because it intersects salience with positive labelling.
        if recent:
            rp_count = 0
            for m in recent:
                weight = float(_safe_attr(m, "weight", 0.0) or 0.0)
                tags = _safe_attr(m, "tags", ()) or ()
                if weight >= 0.6 and _has_positive_tag(tags):
                    rp_count += 1
            reinforced_pos_frac = _clamp(rp_count / float(len(recent)))
        else:
            reinforced_pos_frac = 0.0

        return [
            latest_weight,
            latest_pos,
            recent_pos_frac,
            recent_mean_weight,
            recent_core_frac,
            memory_density,
            latest_persistence,
            reinforced_pos_frac,
        ]

    # ------------------------------------------------------------------
    # Block B: Episodic reinforcement (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_episode_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block B: 4 dims of episodic-reinforcement features.

        Expected keys (all optional):

          * ``recent_episodes`` — sequence of episode objects/dicts with
            optional ``positive_affect`` (bool / float in ``[0,1]``) and
            ``revisit_count`` (int).
          * ``recent_turn_count`` — recent dialog-turn count (int).
          * ``max_turns`` — normalisation cap (default 50).
        """
        episodes = list(_safe_attr(ctx, "recent_episodes", []) or [])
        episodes = episodes[-RECENT_EPISODE_WINDOW:]

        # 0: episode volume (capped, normalised against window)
        episode_volume = _clamp(len(episodes) / float(RECENT_EPISODE_WINDOW))

        # 1: positive-affect fraction
        if episodes:
            pa_values: list[float] = []
            for ep in episodes:
                pa = _safe_attr(ep, "positive_affect", None)
                if pa is None:
                    continue
                if isinstance(pa, bool):
                    pa_values.append(1.0 if pa else 0.0)
                else:
                    try:
                        pa_values.append(_clamp(float(pa)))
                    except (TypeError, ValueError):
                        continue
            episode_pos_frac = _clamp(_mean(pa_values)) if pa_values else 0.0
        else:
            episode_pos_frac = 0.0

        # 2: reinforced-episode fraction (revisit_count > 1)
        if episodes:
            rev_count = sum(
                1 for ep in episodes
                if int(_safe_attr(ep, "revisit_count", 0) or 0) > 1
            )
            episode_reinforced_frac = _clamp(rev_count / float(len(episodes)))
        else:
            episode_reinforced_frac = 0.0

        # 3: dialog continuity — recent turn count normalised
        turn_count = float(_safe_attr(ctx, "recent_turn_count", 0) or 0)
        max_turns = float(_safe_attr(ctx, "max_turns", 50) or 50)
        dialog_continuity = _clamp(
            turn_count / max_turns if max_turns > 0 else 0.0
        )

        return [
            episode_volume,
            episode_pos_frac,
            episode_reinforced_frac,
            dialog_continuity,
        ]

    # ------------------------------------------------------------------
    # Block C: Mood / identity positive bias (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_mood_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block C: 4 dims of mood / identity positive bias.

        Expected keys (all optional):

          * ``traits`` — sequence of trait label strings.
          * ``mood_positivity`` — pre-normalised mood positivity in ``[0, 1]``.
          * ``emotion_positive_bias`` — emotion-classifier positive-class
            probability in ``[0, 1]``.
          * ``contradiction_debt`` — epistemic contradiction debt in ``[0, 1]``;
            we use ``1 - debt`` as a coherence-positive feature.
        """
        traits = _safe_attr(ctx, "traits", ()) or ()
        if traits:
            try:
                pos_traits = sum(1 for t in traits if t in POSITIVE_TRAIT_LABELS)
                pos_trait_frac = _clamp(pos_traits / float(len(traits)))
            except TypeError:
                pos_trait_frac = 0.0
        else:
            pos_trait_frac = 0.0

        mood_positivity = _clamp(
            float(_safe_attr(ctx, "mood_positivity", 0.0) or 0.0)
        )
        emotion_positive_bias = _clamp(
            float(_safe_attr(ctx, "emotion_positive_bias", 0.0) or 0.0)
        )

        # ``coherence_positive`` is only meaningful when contradiction
        # debt has been measured. Without that evidence we report 0.0 so
        # an empty context produces a 0.0 signal instead of a 1.0
        # coherence prior — matching the "no-data → no-signal" contract
        # demanded for Tier-2 broadcast competition.
        debt_raw = _safe_attr(ctx, "contradiction_debt", None)
        if debt_raw is None:
            coherence_positive = 0.0
        else:
            try:
                contradiction_debt = _clamp(float(debt_raw))
            except (TypeError, ValueError):
                contradiction_debt = 0.0
            coherence_positive = _clamp(1.0 - contradiction_debt)

        return [
            pos_trait_frac,
            mood_positivity,
            emotion_positive_bias,
            coherence_positive,
        ]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @staticmethod
    def encode(ctx: Mapping[str, Any]) -> list[float]:
        """Produce the 16-dim ``[0, 1]`` feature vector from context."""
        block_a = PositiveMemoryEncoder.encode_memory_block(ctx)
        block_b = PositiveMemoryEncoder.encode_episode_block(ctx)
        block_c = PositiveMemoryEncoder.encode_mood_block(ctx)
        vec = block_a + block_b + block_c
        # Hard invariant: dimension and bounds. The orchestrator and the
        # broadcast-slot competition rely on this contract.
        assert len(vec) == FEATURE_DIM, (
            f"PositiveMemoryEncoder.encode produced {len(vec)} dims, "
            f"expected {FEATURE_DIM}"
        )
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"feature[{i}] = {v} out of [0,1]"
        return vec

    @staticmethod
    def compute_signal_value(ctx: Mapping[str, Any]) -> float:
        """Real-time inferable scalar in ``[0, 1]``.

        Weighted aggregate over the three feature blocks:

          * Block A (memory valence) at weight 0.50
          * Block B (episodic reinforcement) at weight 0.30
          * Block C (mood / identity positive bias) at weight 0.20

        The blocks are averaged independently then combined; this keeps a
        single very-high feature in one block from saturating the signal.

        This function is the explicit replacement for the
        accuracy-as-proxy fallback in
        ``HemisphereOrchestrator._compute_signal_value``. For the
        ``positive_memory`` focus the orchestrator dispatches to this
        function before attempting network inference; the result is
        always defined and always in ``[0, 1]``, even at CANDIDATE_BIRTH
        when the underlying NN is untrained.
        """
        block_a = PositiveMemoryEncoder.encode_memory_block(ctx)
        block_b = PositiveMemoryEncoder.encode_episode_block(ctx)
        block_c = PositiveMemoryEncoder.encode_mood_block(ctx)

        signal = (
            0.50 * _mean(block_a)
            + 0.30 * _mean(block_b)
            + 0.20 * _mean(block_c)
        )
        return _clamp(signal)
