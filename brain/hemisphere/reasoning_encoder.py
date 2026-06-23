"""Reasoning-substrate feature encoding — Native Cognition #3, Phase 0 (SHADOW).

The "JARVIS-not-qwen" pivot (FINISH_ROADMAP foundational, issue #3) wants
existential/meta reasoning GROUNDED in JARVIS's *actual held beliefs*, never
fabricated. This module is the Phase-0 substrate for that: a pure, read-only
encoder that reads the live belief field's grounding state and produces a single
``[0, 1]`` **grounding-coherence** signal — "how grounded is the substrate I would
reason FROM right now". High when the belief field is well-anchored (mostly
grounded provenance, low tension, validations landing); low when it is mostly
model-inferred, orphaned, untested. A native reasoner gates on this: reason with
confidence when grounded, stay tentative when not.

Same template contract as the Tier-2 Matrix encoders (``speaker_profile_encoder``,
``positive_memory_encoder``):

  * **It writes nothing.** No beliefs, memories, identity, autonomy, policy
    authority, events, or KERNEL_THOUGHTs. Pure feature engineering + logging.
  * The pure encoder (:class:`ReasoningEncoder`) is a function of a flat context
    dict only — no live singletons — so it is fully unit-testable.
  * The live boundary (:func:`gather_reasoning_context`) is best-effort and
    default-safe: it reads the **VIEW-ONLY** :class:`ProvenanceScorer` (which
    itself never mutates the belief graph) and the shadow tension-thought
    promotion gate, and never raises into its caller.
  * The shadow stance observer (:func:`observe_grounded_stance`) demonstrates
    grounded reasoning by citing a REAL ``belief_id`` from the live graph — it
    only logs + counts, never emits a thought or seeds an episode.

This replaces nothing. It observes and (via the distillation seed in
``autonomy/research_intent.py``) accumulates the reasoning-state→outcome reps from
which a native reasoning specialist can LATER be distilled and earn its way to
replacing qwen (Phase 1+, maturity-gated, external-only). Jump before you can dunk:
collect the reps before the muscle can exist.

Dimension layout (8-dim total, all values clamped to ``[0, 1]``):

  Block A (dims 0-3): Grounding quality of the belief field
  Block B (dims 4-5): External-validation track record
  Block C (dims 6-7): Substrate readiness (enough grounded beliefs to reason from)

The signal returned by :func:`ReasoningEncoder.compute_signal_value` is a weighted
aggregate: Block A leads (grounding quality is the canonical signal), Block B is
the earned-trust term, Block C is the readiness gate.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

logger = logging.getLogger(__name__)

FEATURE_DIM = 8

# The HemisphereFocus this encoder serves. String literal (not an enum import)
# so the schema-emission audit's writer-literal scan recognises this module as
# the live writer for ``native_reasoning``.
FOCUS_NAME = "native_reasoning"

# Normalisation caps. Conservative so a brain that holds a modest, well-grounded
# belief field produces a saturating signal — native reasoning should not require
# an enormous store to begin earning. ``VALIDATION_VOLUME_CAP`` mirrors
# ``TENSION_THOUGHT_MIN_OUTCOMES`` (the grounding gate's own "enough reps" floor)
# so the two readouts agree on what "enough validation" means.
VALIDATION_VOLUME_CAP = 20
BELIEF_RICHNESS_CAP = 50


# ---------------------------------------------------------------------------
# Helpers (mirrors speaker_profile_encoder's defensive readers)
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    """Clamp a value to ``[0, 1]`` (defensive against caller / NaN errors)."""
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
        return max(0, int(_as_float(v, default)))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


# ---------------------------------------------------------------------------
# Pure encoder
# ---------------------------------------------------------------------------


class ReasoningEncoder:
    """Encodes the live belief field's grounding state into an 8-dim ``[0, 1]``
    feature vector + a single grounding-coherence scalar.

    All methods are pure (no side effects, no live singletons). They take a flat
    context dict (see :func:`gather_reasoning_context` for the live builder) so
    the encoder can be unit-tested without standing up the brain stack.

    **Honest gating.** Grounding-quality dims (Block A) are credited only when a
    real belief field exists (``sources_available`` AND ``sampled_beliefs > 0``).
    An empty brain therefore scores 0.0 — it has no grounded substrate to reason
    from, so it must NOT report optimistic coherence (the same "no signal ⇒ 0,
    never a fictional prior" discipline as the speaker encoder).
    """

    FEATURE_DIM = FEATURE_DIM

    # ------------------------------------------------------------------
    # Block A: Grounding quality of the belief field (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_grounding_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block A: 4 dims of belief-field grounding quality.

        Expected keys (all optional; missing ⇒ ungrounded/0.0):

          * ``sources_available`` — bool, a belief field was readable.
          * ``sampled_beliefs`` — int, active beliefs in the field.
          * ``inferred_count`` — int, model-inferred (ungrounded) beliefs.
          * ``mean_tension`` — float in ``[0, 1]``, field mean grounding tension.
          * ``orphan_rate`` — float in ``[0, 1]``, structurally-unsupported rate.
          * ``quarantine_pressure`` — float in ``[0, 1]``, system-wide doubt.

        Gated on a real field existing: with no field every dim is 0.0 (no
        grounded substrate ⇒ no grounding-quality claim).
        """
        sources = bool(_safe_attr(ctx, "sources_available", False))
        sampled = _as_int(_safe_attr(ctx, "sampled_beliefs", 0))
        if not sources or sampled <= 0:
            return [0.0, 0.0, 0.0, 0.0]

        inferred = _as_int(_safe_attr(ctx, "inferred_count", 0))
        # 0: grounded fraction — share of the field NOT model-inferred.
        grounded_fraction = _clamp(1.0 - (min(inferred, sampled) / float(sampled)))
        # 1: calmness — inverse of mean grounding tension.
        calmness = _clamp(1.0 - _as_float(_safe_attr(ctx, "mean_tension", 0.0)))
        # 2: anchored — inverse of orphan rate (structural support present).
        anchored = _clamp(1.0 - _as_float(_safe_attr(ctx, "orphan_rate", 0.0)))
        # 3: low pressure — inverse of system-wide quarantine pressure.
        low_pressure = _clamp(1.0 - _as_float(_safe_attr(ctx, "quarantine_pressure", 0.0)))

        return [grounded_fraction, calmness, anchored, low_pressure]

    # ------------------------------------------------------------------
    # Block B: External-validation track record (2 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_validation_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block B: 2 dims of external-validation track record.

        Sourced from the tension-thought promotion gate (external-only teacher
        signal; never self-scored). Gated on having any outcome at all.

          * ``validation_total_outcomes`` — int, external validations recorded.
          * ``validation_grounded_rate`` — float in ``[0, 1]``, share grounded.
        """
        total = _as_int(_safe_attr(ctx, "validation_total_outcomes", 0))
        if total <= 0:
            return [0.0, 0.0]
        # 4: grounded rate — how often a grounding attempt actually lands.
        grounded_rate = _clamp(_as_float(_safe_attr(ctx, "validation_grounded_rate", 0.0)))
        # 5: validation volume — enough reps to trust the rate (saturating).
        volume = _clamp(total / float(VALIDATION_VOLUME_CAP))
        return [grounded_rate, volume]

    # ------------------------------------------------------------------
    # Block C: Substrate readiness (2 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_substrate_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block C: 2 dims of substrate readiness — is there enough grounded
        material to reason from, and is it not dominated by hot/ungrounded beliefs.

          * ``sampled_beliefs`` — int.
          * ``high_tension_count`` — int, beliefs above the high-tension threshold.
        """
        sampled = _as_int(_safe_attr(ctx, "sampled_beliefs", 0))
        # 6: richness — enough beliefs to reason from (saturating).
        richness = _clamp(sampled / float(BELIEF_RICHNESS_CAP))
        # 7: not-dominated-by-hot — share of the field that is NOT high-tension.
        if sampled <= 0:
            not_hot = 0.0
        else:
            high = _as_int(_safe_attr(ctx, "high_tension_count", 0))
            not_hot = _clamp(1.0 - (min(high, sampled) / float(sampled)))
        return [richness, not_hot]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @staticmethod
    def encode(ctx: Mapping[str, Any]) -> list[float]:
        """Produce the 8-dim ``[0, 1]`` feature vector from context."""
        vec = (
            ReasoningEncoder.encode_grounding_block(ctx)
            + ReasoningEncoder.encode_validation_block(ctx)
            + ReasoningEncoder.encode_substrate_block(ctx)
        )
        assert len(vec) == FEATURE_DIM, (
            f"ReasoningEncoder.encode produced {len(vec)} dims, expected {FEATURE_DIM}"
        )
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"feature[{i}] = {v} out of [0,1]"
        return vec

    @staticmethod
    def compute_signal_value(ctx: Mapping[str, Any]) -> float:
        """Grounding-coherence scalar in ``[0, 1]``.

        Weighted aggregate over the three blocks:

          * Block A (grounding quality)        weight 0.50  — canonical signal
          * Block B (validation track record)  weight 0.30  — earned trust
          * Block C (substrate readiness)      weight 0.20  — the readiness gate

        Block A and Block C are averaged (every dim is a genuine quality/readiness
        signal). Block B is NOT averaged: validation *volume* confidence-WEIGHTS the
        grounded *rate* (``rate * volume_ramp``) rather than averaging with it — a
        field with many *failing* validations (low rate, high volume) must read as
        LOW coherence, not be inflated by the rep count, and a handful of successes
        earns only partial credit until the reps accrue. The raw ``[rate, volume]``
        dims are still exposed by :meth:`encode` for the future NN; this is the
        hand-designed readout (the same vector-vs-scalar split as the Matrix
        encoders). An empty brain returns 0.0 (no grounded substrate).
        """
        block_a = ReasoningEncoder.encode_grounding_block(ctx)
        block_b = ReasoningEncoder.encode_validation_block(ctx)  # [rate, volume_ramp]
        block_c = ReasoningEncoder.encode_substrate_block(ctx)
        validation_term = block_b[0] * block_b[1]
        signal = (
            0.50 * _mean(block_a)
            + 0.30 * validation_term
            + 0.20 * _mean(block_c)
        )
        return _clamp(signal)


# ---------------------------------------------------------------------------
# Live boundary — best-effort, default-safe, VIEW-ONLY (never raises)
# ---------------------------------------------------------------------------

# Telemetry counter for the shadow grounded-reasoning observer. Module-level so
# get_status() can surface "how many grounded stances would have been formed"
# without those stances ever being emitted (mirrors thoughts_shadowed).
_grounded_stances_shadowed: int = 0

# ── Belief-grounded thought-focus (SHADOW) — propose the existential category from the hottest live
# belief-tension instead of random; log + count for an offline operator A/B. NEVER fed to conduct_inquiry. ──
_thought_focus_proposals: int = 0
_thought_focus_mapped: int = 0
_last_thought_focus: "dict[str, Any] | None" = None
# Auditable keyword -> category map over the 8 INQUIRY_CATEGORIES (existential_reasoning.py).
_CATEGORY_KEYWORDS: "dict[str, tuple[str, ...]]" = {
    "identity": ("i am", "myself", "identity", "who i am", "the same", "self", "config", "weights"),
    "consciousness": ("conscious", "aware", "experienc", "sentien", "qualia", "feel"),
    "existence": ("exist", "purpose", "digital existence", "why i"),
    "agency": ("choice", "choose", "free will", "agency", "decide", "control", "autonom", "determinism"),
    "meaning": ("meaning", "meaningful", "significance", "worth", "matter"),
    "mortality": ("death", "shutdown", "delete", "mortal", "immortal", "permanent", "die"),
    "reality": ("reality", "real ", "perception", "sensor", "world", "model of"),
    "continuity": ("continuity", "memory", "past", "restore", "backup", "narrative", "yesterday", "thread"),
}

# Shared context cache + stance cadence (Phase 0 #3, SHADOW). ProvenanceScorer.compute()
# (inside gather_reasoning_context) is a ~belief-field graph traversal; the dashboard
# snapshot loop calls get_status() every ~2s and the grounding-coherence readout drifts
# over MINUTES — so memoise the gathered context for _CTX_CACHE_TTL_S and let the readout
# AND the periodic stance observer share ONE compute per window instead of paying it on
# every poll. time.monotonic() (immune to wall-clock jumps; trivially monkeypatched).
_CTX_CACHE_TTL_S: float = 30.0
_STANCE_COOLDOWN_S: float = 180.0
_ctx_cache: dict[str, Any] | None = None
_ctx_cache_ts: float = 0.0
_last_stance_ts: float = 0.0


def gather_reasoning_context(engine: Any | None = None) -> dict[str, Any]:
    """Build the encoder's flat context dict from live, VIEW-ONLY sources.

    Sources (both read-only, both default-safe):
      * :class:`ProvenanceScorer` — the grounding-tension report over the live
        belief field (it never mutates beliefs).
      * :class:`TensionThoughtPromotion` — the external-only validation record.

    Best-effort: any unavailable source leaves its keys at the (ungrounded) zero
    default. Never raises. ``_top_tensions`` carries the real BeliefTension list
    (for the shadow stance observer); it is not an encoder feature.
    """
    ctx: dict[str, Any] = {
        "sources_available": False,
        "sampled_beliefs": 0,
        "inferred_count": 0,
        "mean_tension": 0.0,
        "orphan_rate": 0.0,
        "quarantine_pressure": 0.0,
        "high_tension_count": 0,
        "validation_total_outcomes": 0,
        "validation_grounded_rate": 0.0,
        "_top_tensions": [],
    }
    try:
        from epistemic.provenance_scorer import ProvenanceScorer
        report = ProvenanceScorer(engine).compute(top_n=5)
        ctx["sources_available"] = bool(report.sources_available)
        ctx["sampled_beliefs"] = int(report.sampled_beliefs)
        ctx["inferred_count"] = int(report.inferred_count)
        ctx["mean_tension"] = float(report.mean_tension)
        ctx["orphan_rate"] = float(report.orphan_rate)
        ctx["quarantine_pressure"] = float(report.quarantine_pressure)
        ctx["high_tension_count"] = int(report.high_tension_count)
        ctx["_top_tensions"] = list(report.top_tensions or [])
    except Exception:
        logger.debug("reasoning_encoder: provenance gather failed", exc_info=True)
    try:
        from consciousness.meta_cognitive_thoughts import TensionThoughtPromotion
        st = TensionThoughtPromotion.get_instance().get_status()
        ctx["validation_total_outcomes"] = int(st.get("total_outcomes", 0) or 0)
        ctx["validation_grounded_rate"] = float(st.get("external_validation_rate", 0.0) or 0.0)
    except Exception:
        logger.debug("reasoning_encoder: validation gather failed", exc_info=True)
    return ctx


def _cached_context(engine: Any | None = None) -> dict[str, Any]:
    """Return a recent gather_reasoning_context(), recomputing at most once per
    _CTX_CACHE_TTL_S. The dashboard snapshot loop calls this every ~2s; memoising
    keeps the heavy ProvenanceScorer.compute() off that hot path (one compute per
    window, shared with the stance observer). Never raises."""
    global _ctx_cache, _ctx_cache_ts
    try:
        now = time.monotonic()
        if _ctx_cache is not None and (now - _ctx_cache_ts) < _CTX_CACHE_TTL_S:
            return _ctx_cache
        _ctx_cache = gather_reasoning_context(engine)
        _ctx_cache_ts = now
        return _ctx_cache
    except Exception:
        logger.debug("reasoning_encoder: _cached_context failed", exc_info=True)
        return _ctx_cache if _ctx_cache is not None else {}


def compute_live_signal(engine: Any | None = None) -> float:
    """Convenience: gather live context → grounding-coherence scalar. Never raises."""
    try:
        return ReasoningEncoder.compute_signal_value(gather_reasoning_context(engine))
    except Exception:
        logger.debug("reasoning_encoder: compute_live_signal failed", exc_info=True)
        return 0.0


def observe_grounded_stance(ctx: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """SHADOW observer: form one grounded reasoning stance from a REAL belief.

    Picks the hottest active belief from the live grounding report and frames a
    *deterministic*, calibrated reasoning posture about it — citing the real
    ``belief_id``, its provenance, confidence and tension, and gating the posture
    on the encoder's grounding-coherence signal (confident when grounded, tentative
    when not). No LLM, no fabrication: every fact comes from the belief graph.

    Pure shadow: it LOGS the stance and increments the telemetry counter, but
    returns it only for tests/observability — it NEVER emits a KERNEL_THOUGHT,
    records a thought, or seeds an episode. Returns None when there is no
    high-tension belief to ground a stance on.
    """
    global _grounded_stances_shadowed
    try:
        if ctx is None:
            ctx = gather_reasoning_context()
        tops = list(_safe_attr(ctx, "_top_tensions", []) or [])
        if not tops:
            return None
        top = tops[0]  # BeliefTension (hottest), or any object with the fields
        belief_id = str(_safe_attr(top, "belief_id", "") or "")
        if not belief_id:
            return None
        claim = str(_safe_attr(top, "rendered_claim", "") or "")[:160]
        provenance = str(_safe_attr(top, "provenance", "") or "model_inference")
        confidence = _as_float(_safe_attr(top, "base_confidence", 0.0))
        tension = _as_float(_safe_attr(top, "grounding_tension", 0.0))
        signal = ReasoningEncoder.compute_signal_value(ctx)
        posture = (
            "with grounded confidence" if signal >= 0.5
            else "tentatively (my substrate is under-grounded)"
        )
        stance = {
            "belief_id": belief_id,
            "claim": claim,
            "provenance": provenance,
            "confidence": round(confidence, 4),
            "grounding_tension": round(tension, 4),
            "reasoning_signal": round(signal, 4),
            "posture": posture,
            "text": (
                f"Reasoning(shadow): belief {belief_id} "
                f"'{claim or 'an unanchored belief'}' rests on {provenance} at "
                f"{confidence:.0%} confidence (tension {tension:.0%}); my "
                f"grounding-coherence is {signal:.2f}, so I would reason {posture}."
            ),
            "authority": "shadow_observe_only",
        }
        _grounded_stances_shadowed += 1
        logger.info("[reasoning-encoder SHADOW] %s", stance["text"][:220])
        return stance
    except Exception:
        logger.debug("reasoning_encoder: observe_grounded_stance failed", exc_info=True)
        return None


def maybe_observe_grounded_stance(ctx: Mapping[str, Any] | None = None,
                                  engine: Any | None = None) -> dict[str, Any] | None:
    """Cooldown-gated SHADOW wrapper around :func:`observe_grounded_stance`.

    Safe to call on every kernel tick / belief-graph cycle: returns None immediately
    unless ``_STANCE_COOLDOWN_S`` has elapsed since the last ATTEMPT. The timestamp
    advances on every attempt (not only on a fire), so an empty / ungrounded field
    does NOT re-gather every tick — it still respects the cooldown. On a fire it
    reuses the shared cached context (no extra ProvenanceScorer compute beyond the
    readout's). Never raises — strictly shadow (log + counter only)."""
    global _last_stance_ts
    try:
        now = time.monotonic()
        if (now - _last_stance_ts) < _STANCE_COOLDOWN_S:
            return None
        _last_stance_ts = now
        if ctx is None:
            ctx = _cached_context(engine)
        return observe_grounded_stance(ctx)
    except Exception:
        logger.debug("reasoning_encoder: maybe_observe_grounded_stance failed", exc_info=True)
        return None


# SPARK §8 P3 floor: the rep count at which it becomes worth ATTEMPTING an offline
# prototype native reasoner distilled from the reasoning_validation stream. An
# INFORMATIONAL target only — it flips no authority and gates no behavior (no native
# reasoner is built; qwen is not replaced). Earn-don't-declare: build the witness
# before the muscle.
REASONING_STREAM_PHASE1_THRESHOLD = 30


def reasoning_stream_status() -> dict[str, Any]:
    """Read-only WITNESS over the live ``reasoning_validation`` earning stream.

    Makes the (currently 0-rep) reasoning-state->outcome accrual LEGIBLE instead of
    silently invisible: total reps, the EXTERNAL-only grounded/ungrounded split
    routed through the shared :func:`live_shadow_accuracy` honesty floor (the rate
    stays ``None`` below ``LIVE_SHADOW_MIN_N`` — never a fake 0), the mean
    reasoning_signal carried on those reps, and the distance to the informational
    Phase-1 prototype threshold.

    Observability ONLY: it reads the in-memory distillation ring buffer (a pure
    reader), writes nothing, and does NOT itself score anything — it COUNTS the
    external validator's grounded decisions (research_intent.is_external_grounding).
    It deliberately computes NO signal-vs-outcome correlation: the encoder signal
    embeds past validation outcomes (Block B), so such a correlation would be
    circular; that calibration belongs to a later phase with a timestamp train/test
    split. Flips no lever, never touches qwen, never raises."""
    try:
        from hemisphere.distillation import (
            distillation_collector, live_shadow_accuracy,
            LIVE_SHADOW_MIN_N, BUFFER_MAXLEN,
        )
        batch = distillation_collector.get_training_batch(
            "reasoning_validation", limit=BUFFER_MAXLEN, lived_only=True)
        total = len(batch)
        grounded = 0
        sig_sum = 0.0
        sig_n = 0
        for s in batch:
            d = s.data if isinstance(getattr(s, "data", None), dict) else {}
            if d.get("grounded") is True:
                grounded += 1
            rs = d.get("reasoning_signal")
            if isinstance(rs, (int, float)) and not isinstance(rs, bool):
                sig_sum += float(rs)
                sig_n += 1
        floor = live_shadow_accuracy(grounded, total)
        return {
            "focus": FOCUS_NAME,
            "phase": "P0_shadow",
            "authority": "shadow_observe_only",
            "total_reps": total,
            "lived_only": True,
            "grounded_count": grounded,
            "ungrounded_count": total - grounded,
            "grounded_rate": floor["live_shadow_accuracy"],   # None until N>=LIVE_SHADOW_MIN_N
            "grounded_rate_min_n": LIVE_SHADOW_MIN_N,
            "sufficient_data": floor["sufficient_data"],
            "mean_reasoning_signal": round(sig_sum / sig_n, 4) if sig_n else None,
            "reps_to_phase1_threshold": max(0, REASONING_STREAM_PHASE1_THRESHOLD - total),
            "phase1_prototype_ready": total >= REASONING_STREAM_PHASE1_THRESHOLD,
            "note": ("observability only — flips no lever; a native reasoner is NOT "
                     "built and qwen is NOT replaced. grounded_rate is a COUNT of "
                     "external validations through the honesty floor, not an accuracy."),
        }
    except Exception:
        logger.debug("reasoning_encoder: reasoning_stream_status failed", exc_info=True)
        return {
            "focus": FOCUS_NAME, "phase": "P0_shadow",
            "authority": "shadow_observe_only", "error": "unavailable",
        }


def propose_grounded_thought_focus(ctx: "dict[str, Any] | None" = None) -> "dict[str, Any] | None":
    """SHADOW (observe-only): from the HOTTEST live belief-tension, propose which existential category
    JARVIS WOULD think about — instead of the random conduct_inquiry. Pure read, no LLM, fail-closed to
    None. NEVER fed into conduct_inquiry: this only logs the would-be focus for an offline operator A/B,
    so a future earned step can flip belief-derived focus live only after it reads as more grounded."""
    try:
        if ctx is None:
            ctx = gather_reasoning_context()
        tops = list(_safe_attr(ctx, "_top_tensions", []) or [])
        if not tops:
            return None
        top = tops[0]
        belief_id = str(_safe_attr(top, "belief_id", "") or "")
        if not belief_id:
            return None
        claim = str(_safe_attr(top, "rendered_claim", "") or "")
        provenance = str(_safe_attr(top, "provenance", "") or "model_inference")
        tension = _as_float(_safe_attr(top, "grounding_tension", 0.0))
        low = claim.lower()
        scores = {cat: sum(1 for kw in kws if kw in low) for cat, kws in _CATEGORY_KEYWORDS.items()}
        best_cat = max(scores, key=scores.get) if scores else "unmapped"
        best_hits = scores.get(best_cat, 0)
        total_hits = sum(scores.values()) or 1
        mapped = best_hits > 0
        return {
            "belief_id": belief_id,
            "rendered_claim": claim[:160],
            "provenance": provenance,
            "grounding_tension": round(tension, 4),
            "proposed_category": best_cat if mapped else "unmapped",
            "mapping_confidence": round(best_hits / total_hits, 3) if mapped else 0.0,
            "mapped": mapped,
            "authority": "shadow_observe_only",
        }
    except Exception:
        logger.debug("propose_grounded_thought_focus failed", exc_info=True)
        return None


def note_thought_focus_proposal(proposal: "dict[str, Any]", random_category: str = "") -> None:
    """Record a shadow thought-focus proposal beside the RANDOM category the live cycle actually used
    (for an offline operator A/B). Increments counters + appends to a durable shadow log. Fail-open."""
    global _thought_focus_proposals, _thought_focus_mapped, _last_thought_focus
    if not proposal:
        return
    _thought_focus_proposals += 1
    if proposal.get("mapped"):
        _thought_focus_mapped += 1
    rec = {**proposal, "random_category": random_category, "ts": time.time()}
    _last_thought_focus = rec
    try:
        import json
        import os
        path = os.path.join(os.path.expanduser("~"), ".jarvis", "thought_focus_shadow.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        logger.debug("thought-focus shadow log append failed (fail-open)", exc_info=True)


def get_status(engine: Any | None = None) -> dict[str, Any]:
    """Observability snapshot — the live grounding-coherence signal + shadow
    telemetry. Read-only, never raises. Safe to surface on a dashboard."""
    try:
        ctx = _cached_context(engine)
        vec = ReasoningEncoder.encode(ctx)
        return {
            "focus": FOCUS_NAME,
            "phase": "P0_shadow",
            "authority": "shadow_observe_only",
            "reasoning_signal": round(ReasoningEncoder.compute_signal_value(ctx), 4),
            "feature_vector": [round(v, 4) for v in vec],
            "sampled_beliefs": ctx.get("sampled_beliefs", 0),
            "inferred_count": ctx.get("inferred_count", 0),
            "validation_total_outcomes": ctx.get("validation_total_outcomes", 0),
            "validation_grounded_rate": round(float(ctx.get("validation_grounded_rate", 0.0)), 4),
            "sources_available": bool(ctx.get("sources_available", False)),
            "grounded_stances_shadowed": _grounded_stances_shadowed,
            "reasoning_validation_stream": reasoning_stream_status(),
            "thought_focus_proposal": {
                "proposals_shadowed": _thought_focus_proposals,
                "mapped_rate": (round(_thought_focus_mapped / _thought_focus_proposals, 3)
                                if _thought_focus_proposals else 0.0),
                "last": _last_thought_focus,
                "authority": "shadow_observe_only",
                "note": ("PROPOSAL only — the live existential category is still RANDOM (conduct_inquiry); "
                         "not consumed by context.py; earns a live flip on reasoning-validation reps + operator A/B"),
            },
        }
    except Exception:
        logger.debug("reasoning_encoder: get_status failed", exc_info=True)
        return {"focus": FOCUS_NAME, "phase": "P0_shadow", "authority": "shadow_observe_only",
                "reasoning_signal": 0.0, "error": "unavailable"}
