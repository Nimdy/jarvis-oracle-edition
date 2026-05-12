"""Temporal-pattern feature encoding for the TEMPORAL_PATTERN Tier-2
specialist.

P3.9 — fourth Tier-2 Matrix Protocol specialist. Same template contract
as P3.6 (``positive_memory``), P3.7 (``negative_memory``), and P3.8
(``speaker_profile``):

  * It writes nothing — no memories, beliefs, identity, autonomy,
    policy authority, HRR/P5 state, Soul Integrity, or events. It is
    pure feature engineering.
  * It enters CANDIDATE_BIRTH only. Promotion is gated by the standard
    Matrix Protocol lifecycle in
    ``HemisphereOrchestrator._check_specialist_promotions``; this
    module does not bypass any of it.
  * It produces a real-time inferable scalar in ``[0, 1]`` from
    cadence / recency / mode-stability statistics. It does NOT fall
    back to accuracy-as-proxy.

**Strict privacy / truth-boundary contract.** The user explicitly
flagged this lane as the highest-risk Tier-2 specialist for boundary
drift. ``temporal_pattern`` is allowed to produce a *derived* bounded
signal:

  * "current temporal context is familiar / unfamiliar"
  * "interaction rhythm is stable / irregular"
  * "presence cadence resembles recent windows"

It is NOT allowed to claim or memorise specific user-schedule facts
("David is here at 9 AM", "user comes home at 6 PM"). This is enforced
two ways:

  * The encoder consumes only counts, time deltas, mode dwell scalars,
    and history-length integers. Its inputs and outputs are flat
    ``[0, 1]`` scalars; nothing about *whose* timestamp produced the
    activity ever crosses the boundary.
  * The encoder source is statically scanned by a regression test
    (``test_encoder_module_does_not_make_schedule_claims``) for any
    reference to ``hour_of_day`` / ``day_of_week`` / ``weekday`` /
    ``schedule`` / ``calendar``-shaped fields. None are present.
    Adding any such field in a future PR forces a deliberate
    sign-off on the privacy contract.

Dimension layout (16-dim total, all values clamped to ``[0, 1]``):

  Block A (dims  0-7):  Activity recency / density
  Block B (dims  8-11): Mode stability / transitions
  Block C (dims 12-15): Cadence smoothness

The signal value returned by :func:`compute_signal_value` is a weighted
aggregate over those blocks. Block A leads (recency / density is the
canonical signal source) and Block B is weighted higher than C because
mode-dwell stability is more directly observable than gap-cv smoothness
at low activity volume.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


FEATURE_DIM = 16

# The :class:`HemisphereFocus` value this encoder serves. Kept as a
# string literal (not an enum import) so the schema-emission audit's
# writer-literal scan recognises this module as the live writer for
# ``temporal_pattern`` and removes the future-only whitelist entry.
FOCUS_NAME = "temporal_pattern"


# Recency / density horizons. These are intentionally fixed in the
# encoder (not configurable from context) so the signal scale stays
# stable across boots — temporal-pattern interpretation breaks badly
# if the horizons drift between runs.
VERY_SHORT_HORIZON_S = 60.0       # last minute
SHORT_HORIZON_S = 600.0           # last 10 minutes
MEDIUM_HORIZON_S = 3600.0         # last hour
IDLE_HORIZON_S = 1800.0           # 30 minutes idle = full idle credit
MODE_DWELL_HORIZON_S = 300.0      # 5 minutes of mode dwell = saturated

# Density caps — small numbers so a brain that's actually being
# interacted with can saturate features in the 10-minute and 1-hour
# windows. Tuning is conservative; raising the cap reduces signal
# sensitivity at low volume.
DENSITY_10MIN_CAP = 20
DENSITY_1HOUR_CAP = 60
DENSITY_24HOUR_CAP = 200

# Mode-transition load. Above this threshold the mode is thrashing
# (e.g. boot oscillation, bug); below it the system is stable.
TRANSITIONS_PER_HOUR_CAP = 20
HISTORY_LEN_CAP = 50

# Coefficient-of-variation cap. CV >= this is treated as "fully
# irregular cadence"; CV < this scales linearly toward 0.
CADENCE_CV_CAP = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    if v != v:  # NaN guard
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
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


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(_as_float(v, default))
    except Exception:
        return default


def _as_seq(v: Any) -> Sequence[float]:
    if v is None:
        return ()
    if isinstance(v, (str, bytes)):
        return ()
    try:
        out = []
        for item in v:
            f = _as_float(item)
            out.append(f)
        return tuple(out)
    except TypeError:
        return ()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _gap_cv(timestamps: Sequence[float]) -> float | None:
    """Coefficient of variation of inter-arrival gaps in a timestamp
    sequence. Returns ``None`` if fewer than two timestamps are
    available (no gap to characterise).

    The CV is std/mean — it is dimensionless, so it does not need
    another horizon constant to normalise. We clamp at
    :data:`CADENCE_CV_CAP` so a single huge outlier does not collapse
    the cadence-smoothness feature to 0.0.
    """
    if not timestamps or len(timestamps) < 2:
        return None
    sorted_ts = sorted(float(t) for t in timestamps if _as_float(t) > 0.0)
    if len(sorted_ts) < 2:
        return None
    gaps = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]
    gaps = [g for g in gaps if g > 0.0]
    if not gaps:
        return None
    m = sum(gaps) / float(len(gaps))
    if m <= 0.0:
        return None
    var = sum((g - m) ** 2 for g in gaps) / float(len(gaps))
    std = var ** 0.5
    return std / m


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TemporalPatternEncoder:
    """Encodes cadence / recency / rhythm state into a 16-dim
    ``[0, 1]`` feature vector.

    All public methods are pure (no side effects). They take plain
    Python dicts / sequences, never live singletons, so the encoder
    can be unit-tested without standing up the brain stack.

    The orchestrator-level helper
    :meth:`HemisphereOrchestrator._build_temporal_pattern_context` is
    responsible for gathering live state from ``memory_storage``
    timestamps and ``mode_manager.get_state()`` and passing it in as
    a flat dict.

    The encoder NEVER claims or stores user-schedule facts. See the
    module-level ``privacy / truth-boundary contract`` note.
    """

    FEATURE_DIM = FEATURE_DIM

    # ------------------------------------------------------------------
    # Block A: Activity recency / density (8 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_recency_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block A: 8 dims of recency / density features.

        Expected keys (all optional, missing fields contribute 0.0):

          * ``seconds_since_last_activity`` — float (e.g. since most
            recent memory write, wake event, STT decode, etc.).
          * ``count_last_10min`` — int.
          * ``count_last_1hour`` — int.
          * ``count_last_24hour`` — int.
          * ``idle_seconds`` — float (session idle duration).
        """
        # Sentinel: a missing or huge ``seconds_since_last_activity``
        # collapses the recency features to 0.0, matching the
        # empty-context invariant.
        last_age = _as_float(_safe_attr(ctx, "seconds_since_last_activity", float("inf")))
        if last_age < 0.0 or last_age != last_age:
            last_age = float("inf")

        very_short = _clamp(1.0 - (last_age / VERY_SHORT_HORIZON_S))
        short = _clamp(1.0 - (last_age / SHORT_HORIZON_S))
        medium = _clamp(1.0 - (last_age / MEDIUM_HORIZON_S))

        c10 = max(0, _as_int(_safe_attr(ctx, "count_last_10min", 0)))
        c60 = max(0, _as_int(_safe_attr(ctx, "count_last_1hour", 0)))
        c1440 = max(0, _as_int(_safe_attr(ctx, "count_last_24hour", 0)))
        density_10 = _clamp(c10 / float(DENSITY_10MIN_CAP))
        density_60 = _clamp(c60 / float(DENSITY_1HOUR_CAP))
        density_1440 = _clamp(c1440 / float(DENSITY_24HOUR_CAP))

        # Idle inverse: a missing idle_seconds defaults to a large
        # value (effectively infinite idle), collapsing the feature
        # to 0.0 for empty context.
        idle_s = _as_float(_safe_attr(ctx, "idle_seconds", float("inf")))
        if idle_s < 0.0 or idle_s != idle_s:
            idle_s = float("inf")
        idle_inverse = _clamp(1.0 - (idle_s / IDLE_HORIZON_S))

        # Any-recent-activity flag — purely a presence indicator. Gates
        # downstream blocks against firing on truly empty context.
        any_recent = 1.0 if c60 > 0 or last_age <= MEDIUM_HORIZON_S else 0.0

        return [
            very_short,
            short,
            medium,
            density_10,
            density_60,
            density_1440,
            idle_inverse,
            any_recent,
        ]

    # ------------------------------------------------------------------
    # Block B: Mode stability / transitions (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_mode_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block B: 4 dims of mode-stability features.

        Expected keys (all optional):

          * ``mode`` — string label (e.g. ``"passive"``,
            ``"reflective"``).
          * ``mode_duration_s`` — float seconds the brain has been in
            the current mode.
          * ``mode_history_len`` — int, length of the mode-transition
            history buffer (capped server-side at ~50).
          * ``mode_transitions_last_hour`` — int.
        """
        mode_label = _safe_attr(ctx, "mode", "")
        try:
            mode_label = str(mode_label or "").strip()
        except Exception:
            mode_label = ""
        has_mode_signal = bool(mode_label)

        # Dwell saturates after MODE_DWELL_HORIZON_S; a fresh mode
        # change reads near 0, a long stable dwell reads near 1.
        dwell = _as_float(_safe_attr(ctx, "mode_duration_s", 0.0))
        if has_mode_signal and dwell > 0.0:
            dwell_score = _clamp(dwell / MODE_DWELL_HORIZON_S)
        else:
            dwell_score = 0.0

        transitions = max(0, _as_int(_safe_attr(ctx, "mode_transitions_last_hour", 0)))
        if has_mode_signal:
            low_transition_score = _clamp(
                1.0 - (transitions / float(TRANSITIONS_PER_HOUR_CAP)),
            )
            # Without an explicit transitions count, fall through to
            # the no-mode-signal path so empty context yields 0.0.
            if transitions == 0 and dwell <= 0.0:
                low_transition_score = 0.0
        else:
            low_transition_score = 0.0

        # Mode-known score: 1.0 only if we have a non-empty mode label.
        # No optimistic prior on empty context.
        mode_known_score = 1.0 if has_mode_signal else 0.0

        # History-low score: 0 history items + has signal = stable
        # boot; 50 items = chaotic transitions. Gated on mode signal.
        history_len = max(0, _as_int(_safe_attr(ctx, "mode_history_len", 0)))
        if has_mode_signal:
            history_low = _clamp(1.0 - (history_len / float(HISTORY_LEN_CAP)))
        else:
            history_low = 0.0

        return [
            dwell_score,
            low_transition_score,
            mode_known_score,
            history_low,
        ]

    # ------------------------------------------------------------------
    # Block C: Cadence smoothness (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_cadence_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block C: 4 dims of cadence-smoothness features.

        Expected keys (all optional):

          * ``recent_activity_timestamps`` — sequence of monotonically
            increasing UNIX timestamps in the last hour. The encoder
            computes inter-arrival CV; <2 timestamps yields 0.0.
          * ``medium_activity_timestamps`` — sequence covering the
            last 24 hours.
          * ``activity_count_last_30min`` / ``activity_count_prior_30min``
            — used to compute presence-continuity ratio.
          * ``rhythm_familiarity`` — pre-normalised in ``[0, 1]``;
            currently advisory, defaults to 0.0 until a calibrated
            "this hour resembles recent windows" accessor exists.
        """
        # Short-window cadence CV (last hour-ish).
        recent_ts = _as_seq(_safe_attr(ctx, "recent_activity_timestamps", ()))
        cv_short = _gap_cv(recent_ts)
        if cv_short is None:
            gap_cv_inverse_short = 0.0
        else:
            gap_cv_inverse_short = _clamp(1.0 - (cv_short / CADENCE_CV_CAP))

        # Medium-window cadence CV (last 24h-ish).
        medium_ts = _as_seq(_safe_attr(ctx, "medium_activity_timestamps", ()))
        cv_medium = _gap_cv(medium_ts)
        if cv_medium is None:
            gap_cv_inverse_medium = 0.0
        else:
            gap_cv_inverse_medium = _clamp(1.0 - (cv_medium / CADENCE_CV_CAP))

        # Presence continuity: ratio of activity in last 30m to prior
        # 30m. >=1.0 means activity is sustained or rising; near 0
        # means the rhythm is collapsing. Gated on having any recent
        # activity at all.
        recent_30 = _as_int(_safe_attr(ctx, "activity_count_last_30min", 0))
        prior_30 = _as_int(_safe_attr(ctx, "activity_count_prior_30min", 0))
        if recent_30 > 0 or prior_30 > 0:
            denom = max(1.0, float(prior_30))
            presence_ratio = float(recent_30) / denom
            # Map ratio into [0, 1]: 0 → 0.0 (collapsed), 1 → 1.0
            # (sustained), >1 → 1.0 (rising — saturated).
            presence_continuity = _clamp(presence_ratio)
        else:
            presence_continuity = 0.0

        rhythm_familiarity = _clamp(
            _as_float(_safe_attr(ctx, "rhythm_familiarity", 0.0)),
        )

        return [
            gap_cv_inverse_short,
            gap_cv_inverse_medium,
            presence_continuity,
            rhythm_familiarity,
        ]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @staticmethod
    def encode(ctx: Mapping[str, Any]) -> list[float]:
        """Produce the 16-dim ``[0, 1]`` feature vector from context."""
        block_a = TemporalPatternEncoder.encode_recency_block(ctx)
        block_b = TemporalPatternEncoder.encode_mode_block(ctx)
        block_c = TemporalPatternEncoder.encode_cadence_block(ctx)
        vec = block_a + block_b + block_c
        assert len(vec) == FEATURE_DIM, (
            f"TemporalPatternEncoder.encode produced {len(vec)} dims, "
            f"expected {FEATURE_DIM}"
        )
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"feature[{i}] = {v} out of [0,1]"
        return vec

    @staticmethod
    def compute_signal_value(ctx: Mapping[str, Any]) -> float:
        """Real-time inferable scalar in ``[0, 1]``.

        Weighted aggregate over the three feature blocks:

          * Block A (recency / density)        at weight 0.50
          * Block B (mode stability)           at weight 0.30
          * Block C (cadence smoothness)       at weight 0.20

        Block A leads because activity recency / density is the
        canonical "is this temporal context familiar?" signal — a
        brain that has recent, dense activity has more grounded
        temporal context than a brain that hasn't seen any input in
        an hour. Block B is weighted higher than C because mode-dwell
        stability is more directly observable than gap-CV smoothness
        at low activity volume.

        This function is the explicit replacement for the
        accuracy-as-proxy fallback in
        ``HemisphereOrchestrator._compute_signal_value``. For the
        ``temporal_pattern`` focus the orchestrator dispatches to this
        function before attempting network inference; the result is
        always defined and always in ``[0, 1]``, even at
        CANDIDATE_BIRTH when the underlying NN is untrained.
        """
        block_a = TemporalPatternEncoder.encode_recency_block(ctx)
        block_b = TemporalPatternEncoder.encode_mode_block(ctx)
        block_c = TemporalPatternEncoder.encode_cadence_block(ctx)

        signal = (
            0.50 * _mean(block_a)
            + 0.30 * _mean(block_b)
            + 0.20 * _mean(block_c)
        )
        return _clamp(signal)
