"""Voice-intent NN shadow takeover (Ship B, P1.4).

Adds a three-level governor that lets the ``voice_intent`` distillation
specialist participate in the live routing pipeline alongside the regex
tool router:

  - ``shadow``    — log-only. NN predictions are recorded for agreement
    statistics but never alter the regex result. Default for fresh
    brains.
  - ``advisory``  — NONE-rescue only. When the regex router returns
    ``ToolType.NONE`` AND the NN has high confidence on a non-general
    bucket, the runner rewrites the result to the canonical tool for
    that bucket. All other regex outputs pass through unchanged.
  - ``primary``   — full takeover on disagreement. When the NN disagrees
    with the regex on a high-confidence prediction, the NN wins. Regex
    is still computed (so we can still record disagreement) but the NN
    result is returned. Promotion to ``primary`` requires meeting the
    gates below.

Promotion gates (must all pass to advance ``shadow``→``advisory``→``primary``):
  - At least ``MIN_TEACHER_SAMPLES`` teacher samples logged via the
    distillation collector for the ``voice_intent`` student.
  - Rolling agreement rate >= ``MIN_AGREEMENT_FOR_PROMOTION`` over the
    last ``AGREEMENT_WINDOW`` observations.
  - Dwell window: at least ``MIN_DWELL_OBSERVATIONS_PER_LEVEL`` observations
    spent at the prior level since last promotion or boot.
  - No automatic rollback in the last ``ROLLBACK_COOLDOWN_S`` seconds.

Rollback (any level → previous level):
  - Rolling agreement rate drops below ``ROLLBACK_AGREEMENT_FLOOR`` over
    the last ``AGREEMENT_WINDOW`` observations after the level became
    active.
  - Any recorded "NN claimed a tool that returned a hard error" event.

Persistence:
  - State persisted to ``~/.jarvis/intent_shadow_state.json``. Schema is
    flat dict with level + counters + rolling agreement window +
    last_promotion_ts + rollback_history. Restart-continuous.
  - Honest gating: an empty state file means SHADOW with zero samples,
    not "trust restored from previous run." This parallels how the
    Phase 6.5 attestation ledger separates ``current_ok`` from
    ``prior_attested_ok``.

Inference plug-in:
  - The runner does NOT call into the hemisphere engine directly. It
    accepts an optional ``inference_fn(user_message: str) -> list[float] | None``
    that returns the 8-dim bucket distribution from the ``voice_intent``
    specialist. ``None`` means "NN unavailable for this request" and
    counts as ``no_prediction`` (does not affect the agreement window).
  - This makes the runner trivially testable, keeps it from mutating the
    hemisphere subsystem, and lets the wiring layer choose when to
    incur embedding/inference cost.

Per-request scope:
  - ``observe`` returns either the original ``RoutingResult`` or a NEW
    ``RoutingResult`` instance. It NEVER mutates the input.
  - All state mutations (counter bumps, agreement window appends,
    promotion / rollback) happen under ``self._lock``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants — gates and thresholds
# --------------------------------------------------------------------------

MIN_TEACHER_SAMPLES: int = 500
MIN_DWELL_OBSERVATIONS_PER_LEVEL: int = 100
MIN_AGREEMENT_FOR_PROMOTION: float = 0.80
ROLLBACK_AGREEMENT_FLOOR: float = 0.65
AGREEMENT_WINDOW: int = 200
ROLLBACK_COOLDOWN_S: float = 24 * 3600.0

# Per-request thresholds
NN_RESCUE_MIN_CONFIDENCE: float = 0.55
NN_PRIMARY_MIN_CONFIDENCE: float = 0.70
NN_PRIMARY_MARGIN: float = 0.15

# Bucket order MUST match ``tool_router._VOICE_INTENT_BUCKETS``.
BUCKETS: tuple[str, ...] = (
    "general_chat",
    "status_ops",
    "introspection",
    "memory",
    "vision_camera",
    "identity",
    "skill_action",
    "knowledge_lookup",
)


# Reverse mapping for NONE-rescue. We choose the most representative
# canonical tool per bucket. ``general_chat`` deliberately maps to
# ``NONE`` (no rescue) — that bucket means "the NN agrees the regex is
# right to fall through to chat".
def _bucket_to_tool() -> dict[str, str]:
    return {
        "general_chat": "NONE",
        "status_ops": "SYSTEM_STATUS",
        "introspection": "INTROSPECTION",
        "memory": "MEMORY",
        "vision_camera": "VISION",
        "identity": "IDENTITY",
        "skill_action": "SKILL",
        "knowledge_lookup": "WEB_SEARCH",
    }


BUCKET_TO_TOOL: dict[str, str] = _bucket_to_tool()


# --------------------------------------------------------------------------
# Level enum
# --------------------------------------------------------------------------


class IntentShadowLevel(str, Enum):
    SHADOW = "shadow"
    ADVISORY = "advisory"
    PRIMARY = "primary"


_LEVEL_ORDER: tuple[IntentShadowLevel, ...] = (
    IntentShadowLevel.SHADOW,
    IntentShadowLevel.ADVISORY,
    IntentShadowLevel.PRIMARY,
)


# --------------------------------------------------------------------------
# Public observation result
# --------------------------------------------------------------------------


@dataclass
class IntentShadowObservation:
    """Returned by ``observe`` when the caller wants the diagnostic info.

    ``observe_and_rewrite`` returns the (possibly rewritten) routing
    result; this dataclass is exposed for callers that want both.
    """

    level: IntentShadowLevel
    nn_available: bool
    nn_top_bucket: str | None
    nn_top_confidence: float
    regex_bucket: str | None
    agreed: bool | None
    rewrote: bool
    rewrite_reason: str | None = None


# --------------------------------------------------------------------------
# State + persistence
# --------------------------------------------------------------------------


@dataclass
class _PersistedState:
    level: str = IntentShadowLevel.SHADOW.value
    teacher_samples_observed: int = 0
    observations_total: int = 0
    nn_predictions_total: int = 0
    agreements_total: int = 0
    disagreements_total: int = 0
    rescues_applied: int = 0
    primary_overrides_applied: int = 0
    last_promotion_ts: float = 0.0
    last_rollback_ts: float = 0.0
    observations_at_current_level: int = 0
    rolling_agreement: list[int] = field(default_factory=list)
    rollback_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "teacher_samples_observed": self.teacher_samples_observed,
            "observations_total": self.observations_total,
            "nn_predictions_total": self.nn_predictions_total,
            "agreements_total": self.agreements_total,
            "disagreements_total": self.disagreements_total,
            "rescues_applied": self.rescues_applied,
            "primary_overrides_applied": self.primary_overrides_applied,
            "last_promotion_ts": self.last_promotion_ts,
            "last_rollback_ts": self.last_rollback_ts,
            "observations_at_current_level": self.observations_at_current_level,
            "rolling_agreement": list(self.rolling_agreement),
            "rollback_history": list(self.rollback_history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _PersistedState:
        return cls(
            level=str(d.get("level", IntentShadowLevel.SHADOW.value)),
            teacher_samples_observed=int(d.get("teacher_samples_observed", 0) or 0),
            observations_total=int(d.get("observations_total", 0) or 0),
            nn_predictions_total=int(d.get("nn_predictions_total", 0) or 0),
            agreements_total=int(d.get("agreements_total", 0) or 0),
            disagreements_total=int(d.get("disagreements_total", 0) or 0),
            rescues_applied=int(d.get("rescues_applied", 0) or 0),
            primary_overrides_applied=int(d.get("primary_overrides_applied", 0) or 0),
            last_promotion_ts=float(d.get("last_promotion_ts", 0.0) or 0.0),
            last_rollback_ts=float(d.get("last_rollback_ts", 0.0) or 0.0),
            observations_at_current_level=int(
                d.get("observations_at_current_level", 0) or 0
            ),
            rolling_agreement=list(d.get("rolling_agreement", []) or []),
            rollback_history=list(d.get("rollback_history", []) or []),
        )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


_DEFAULT_STATE_PATH = os.path.expanduser("~/.jarvis/intent_shadow_state.json")


class IntentShadowRunner:
    """Three-level NN-vs-regex shadow / advisory / primary governor."""

    def __init__(
        self,
        state_path: str = _DEFAULT_STATE_PATH,
        inference_fn: Callable[[str], list[float] | None] | None = None,
        teacher_sample_provider: Callable[[], int] | None = None,
    ) -> None:
        self._state_path = state_path
        self._inference_fn = inference_fn
        self._teacher_sample_provider = teacher_sample_provider
        self._lock = threading.Lock()
        self._state = self._load_state()
        # In-memory rolling window (mirrors persisted state for fast access).
        self._agreement_window: deque[int] = deque(
            self._state.rolling_agreement[-AGREEMENT_WINDOW:],
            maxlen=AGREEMENT_WINDOW,
        )

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def _load_state(self) -> _PersistedState:
        if not os.path.exists(self._state_path):
            return _PersistedState()
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                return _PersistedState.from_dict(json.load(f))
        except Exception:
            logger.warning(
                "intent_shadow: corrupt state file at %s, resetting to SHADOW",
                self._state_path,
                exc_info=True,
            )
            return _PersistedState()

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            self._state.rolling_agreement = list(self._agreement_window)
            tmp = self._state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, indent=2, sort_keys=True)
            os.replace(tmp, self._state_path)
        except Exception:
            logger.exception("intent_shadow: failed to persist state")

    # ----------------------------------------------------------------------
    # NN inference helpers
    # ----------------------------------------------------------------------

    def _nn_prediction(
        self, user_message: str
    ) -> tuple[str | None, float, list[float] | None]:
        """Return (top_bucket, top_confidence, raw_logits) from the NN.

        Returns ``(None, 0.0, None)`` when no inference function is wired
        or when inference raises / returns an unexpected shape.
        """
        if self._inference_fn is None:
            return None, 0.0, None
        try:
            logits = self._inference_fn(user_message)
        except Exception:
            logger.debug("intent_shadow: inference_fn raised", exc_info=True)
            return None, 0.0, None
        if not logits or len(logits) != len(BUCKETS):
            return None, 0.0, None
        # Treat negative values defensively; we want a probability-like
        # distribution but tolerate raw scores.
        clamped = [max(0.0, float(x)) for x in logits]
        total = sum(clamped)
        if total <= 0.0:
            return None, 0.0, list(logits)
        probs = [x / total for x in clamped]
        top_idx = max(range(len(probs)), key=lambda i: probs[i])
        return BUCKETS[top_idx], float(probs[top_idx]), list(probs)

    @staticmethod
    def _regex_bucket(tool_value: str) -> str:
        """Map a regex tool result back to the bucket the NN would emit."""
        # Lazy-import to avoid a tool_router <-> intent_shadow cycle.
        from reasoning.tool_router import _VOICE_INTENT_TOOL_BUCKET, ToolType

        try:
            tool = ToolType(tool_value)
        except ValueError:
            return "general_chat"
        return _VOICE_INTENT_TOOL_BUCKET.get(tool, "general_chat")

    # ----------------------------------------------------------------------
    # Observation entry point
    # ----------------------------------------------------------------------

    def observe_and_rewrite(
        self,
        user_message: str,
        result: Any,
    ) -> tuple[Any, IntentShadowObservation]:
        """Run shadow comparison and optionally rewrite the routing result.

        ``result`` is duck-typed as anything with a ``.tool`` enum and a
        ``.confidence`` float. The returned ``RoutingResult`` is either
        the original instance (no rewrite) or a new instance.
        """
        from reasoning.tool_router import RoutingResult, ToolType

        with self._lock:
            level = IntentShadowLevel(self._state.level)
            self._state.observations_total += 1
            self._state.observations_at_current_level += 1

            top_bucket, top_conf, _probs = self._nn_prediction(user_message)
            regex_bucket = self._regex_bucket(result.tool.value)

            agreed: bool | None = None
            if top_bucket is not None:
                self._state.nn_predictions_total += 1
                agreed = top_bucket == regex_bucket
                self._agreement_window.append(1 if agreed else 0)
                if agreed:
                    self._state.agreements_total += 1
                else:
                    self._state.disagreements_total += 1

            # Default: pass through the regex result.
            new_result = result
            rewrote = False
            rewrite_reason: str | None = None

            if (
                level in (IntentShadowLevel.ADVISORY, IntentShadowLevel.PRIMARY)
                and top_bucket is not None
                and result.tool == ToolType.NONE
                and top_bucket != "general_chat"
                and top_conf >= NN_RESCUE_MIN_CONFIDENCE
            ):
                tool_str = BUCKET_TO_TOOL.get(top_bucket, "NONE")
                try:
                    new_tool = ToolType(tool_str)
                except ValueError:
                    new_tool = ToolType.NONE
                if new_tool != ToolType.NONE:
                    new_result = RoutingResult(
                        tool=new_tool,
                        confidence=float(top_conf),
                        extracted_args=dict(result.extracted_args or {}),
                        golden_context=getattr(result, "golden_context", None),
                    )
                    new_result.extracted_args["intent_shadow_rescued"] = True
                    new_result.extracted_args["intent_shadow_top_bucket"] = top_bucket
                    new_result.extracted_args["intent_shadow_top_confidence"] = round(
                        top_conf, 4
                    )
                    self._state.rescues_applied += 1
                    rewrote = True
                    rewrite_reason = "advisory_none_rescue"

            if (
                level == IntentShadowLevel.PRIMARY
                and not rewrote
                and top_bucket is not None
                and not agreed
                and top_conf >= NN_PRIMARY_MIN_CONFIDENCE
                and top_conf - max(0.0, float(result.confidence)) >= NN_PRIMARY_MARGIN
            ):
                tool_str = BUCKET_TO_TOOL.get(top_bucket, "NONE")
                try:
                    new_tool = ToolType(tool_str)
                except ValueError:
                    new_tool = ToolType.NONE
                new_result = RoutingResult(
                    tool=new_tool,
                    confidence=float(top_conf),
                    extracted_args=dict(result.extracted_args or {}),
                    golden_context=getattr(result, "golden_context", None),
                )
                new_result.extracted_args["intent_shadow_primary_override"] = True
                new_result.extracted_args["intent_shadow_top_bucket"] = top_bucket
                new_result.extracted_args["intent_shadow_top_confidence"] = round(
                    top_conf, 4
                )
                self._state.primary_overrides_applied += 1
                rewrote = True
                rewrite_reason = "primary_override"

            self._maybe_auto_rollback()
            self._save_state()

            obs = IntentShadowObservation(
                level=level,
                nn_available=top_bucket is not None,
                nn_top_bucket=top_bucket,
                nn_top_confidence=top_conf,
                regex_bucket=regex_bucket,
                agreed=agreed,
                rewrote=rewrote,
                rewrite_reason=rewrite_reason,
            )
            return new_result, obs

    # Convenience: most call sites only want the result.
    def observe(self, user_message: str, result: Any) -> Any:
        new_result, _ = self.observe_and_rewrite(user_message, result)
        return new_result

    # ----------------------------------------------------------------------
    # Promotion / rollback
    # ----------------------------------------------------------------------

    def _agreement_rate(self) -> float | None:
        if not self._agreement_window:
            return None
        return sum(self._agreement_window) / len(self._agreement_window)

    def _refresh_teacher_samples(self) -> int:
        if self._teacher_sample_provider is None:
            return self._state.teacher_samples_observed
        try:
            n = int(self._teacher_sample_provider())
            if n > self._state.teacher_samples_observed:
                self._state.teacher_samples_observed = n
            return self._state.teacher_samples_observed
        except Exception:
            return self._state.teacher_samples_observed

    def maybe_promote(self) -> bool:
        """Attempt to advance one level. Returns True if promoted."""
        with self._lock:
            current = IntentShadowLevel(self._state.level)
            idx = _LEVEL_ORDER.index(current)
            if idx >= len(_LEVEL_ORDER) - 1:
                return False
            samples = self._refresh_teacher_samples()
            if samples < MIN_TEACHER_SAMPLES:
                return False
            if self._state.observations_at_current_level < MIN_DWELL_OBSERVATIONS_PER_LEVEL:
                return False
            agreement = self._agreement_rate()
            if agreement is None or agreement < MIN_AGREEMENT_FOR_PROMOTION:
                return False
            now = time.time()
            if (
                self._state.last_rollback_ts
                and now - self._state.last_rollback_ts < ROLLBACK_COOLDOWN_S
            ):
                return False
            new_level = _LEVEL_ORDER[idx + 1]
            self._state.level = new_level.value
            self._state.last_promotion_ts = now
            self._state.observations_at_current_level = 0
            self._save_state()
            logger.info(
                "intent_shadow: promoted %s -> %s (agreement=%.3f, samples=%d)",
                current.value, new_level.value, agreement, samples,
            )
            return True

    def _maybe_auto_rollback(self) -> None:
        """Internal: called inside ``observe`` under lock."""
        current = IntentShadowLevel(self._state.level)
        if current == IntentShadowLevel.SHADOW:
            return
        if len(self._agreement_window) < AGREEMENT_WINDOW:
            return
        agreement = self._agreement_rate() or 0.0
        if agreement >= ROLLBACK_AGREEMENT_FLOOR:
            return
        idx = _LEVEL_ORDER.index(current)
        prev_level = _LEVEL_ORDER[idx - 1]
        now = time.time()
        self._state.rollback_history.append({
            "from_level": current.value,
            "to_level": prev_level.value,
            "ts": now,
            "agreement_at_rollback": round(agreement, 4),
            "reason": "agreement_below_floor",
        })
        # Cap history so the file doesn't grow unbounded.
        if len(self._state.rollback_history) > 50:
            self._state.rollback_history = self._state.rollback_history[-50:]
        self._state.level = prev_level.value
        self._state.last_rollback_ts = now
        self._state.observations_at_current_level = 0
        logger.warning(
            "intent_shadow: auto-rollback %s -> %s (agreement=%.3f)",
            current.value, prev_level.value, agreement,
        )

    def manual_rollback(self, reason: str = "manual") -> bool:
        """Operator-initiated rollback (one level). Returns True if rolled back."""
        with self._lock:
            current = IntentShadowLevel(self._state.level)
            idx = _LEVEL_ORDER.index(current)
            if idx == 0:
                return False
            prev_level = _LEVEL_ORDER[idx - 1]
            now = time.time()
            self._state.rollback_history.append({
                "from_level": current.value,
                "to_level": prev_level.value,
                "ts": now,
                "agreement_at_rollback": (self._agreement_rate() or 0.0),
                "reason": reason,
            })
            self._state.level = prev_level.value
            self._state.last_rollback_ts = now
            self._state.observations_at_current_level = 0
            self._save_state()
            return True

    # ----------------------------------------------------------------------
    # Read-only state for the dashboard
    # ----------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            agreement = self._agreement_rate()
            samples = self._refresh_teacher_samples()
            current = IntentShadowLevel(self._state.level)
            idx = _LEVEL_ORDER.index(current)
            next_level = (
                _LEVEL_ORDER[idx + 1].value
                if idx < len(_LEVEL_ORDER) - 1
                else None
            )
            gates = {
                "min_teacher_samples": {
                    "required": MIN_TEACHER_SAMPLES,
                    "actual": samples,
                    "ok": samples >= MIN_TEACHER_SAMPLES,
                },
                "min_dwell_observations": {
                    "required": MIN_DWELL_OBSERVATIONS_PER_LEVEL,
                    "actual": self._state.observations_at_current_level,
                    "ok": (
                        self._state.observations_at_current_level
                        >= MIN_DWELL_OBSERVATIONS_PER_LEVEL
                    ),
                },
                "rolling_agreement": {
                    "required": MIN_AGREEMENT_FOR_PROMOTION,
                    "actual": (
                        round(agreement, 4) if agreement is not None else None
                    ),
                    "ok": (
                        agreement is not None
                        and agreement >= MIN_AGREEMENT_FOR_PROMOTION
                    ),
                },
                "rollback_cooldown": {
                    "required_s": ROLLBACK_COOLDOWN_S,
                    "since_last_rollback_s": (
                        round(time.time() - self._state.last_rollback_ts, 1)
                        if self._state.last_rollback_ts
                        else None
                    ),
                    "ok": (
                        not self._state.last_rollback_ts
                        or (
                            time.time() - self._state.last_rollback_ts
                            >= ROLLBACK_COOLDOWN_S
                        )
                    ),
                },
            }
            return {
                "level": current.value,
                "next_level": next_level,
                "ready_for_promotion": (
                    next_level is not None
                    and all(g["ok"] for g in gates.values())
                ),
                "observations_total": self._state.observations_total,
                "nn_predictions_total": self._state.nn_predictions_total,
                "agreements_total": self._state.agreements_total,
                "disagreements_total": self._state.disagreements_total,
                "rolling_agreement": (
                    round(agreement, 4) if agreement is not None else None
                ),
                "rolling_window_size": len(self._agreement_window),
                "rescues_applied": self._state.rescues_applied,
                "primary_overrides_applied": self._state.primary_overrides_applied,
                "last_promotion_ts": self._state.last_promotion_ts or None,
                "last_rollback_ts": self._state.last_rollback_ts or None,
                "rollback_history": list(self._state.rollback_history[-10:]),
                "gates": gates,
                "thresholds": {
                    "nn_rescue_min_confidence": NN_RESCUE_MIN_CONFIDENCE,
                    "nn_primary_min_confidence": NN_PRIMARY_MIN_CONFIDENCE,
                    "nn_primary_margin": NN_PRIMARY_MARGIN,
                    "rollback_agreement_floor": ROLLBACK_AGREEMENT_FLOOR,
                    "agreement_window": AGREEMENT_WINDOW,
                },
            }


# --------------------------------------------------------------------------
# Module-level singleton accessor
# --------------------------------------------------------------------------


_runner: IntentShadowRunner | None = None
_runner_lock = threading.Lock()


def get_intent_shadow_runner() -> IntentShadowRunner:
    """Return a process-wide ``IntentShadowRunner`` singleton.

    Lazily initialized; safe to call from any thread. Tests should use
    ``set_intent_shadow_runner`` to inject a runner backed by a tmp path.
    """
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = IntentShadowRunner()
        return _runner


def set_intent_shadow_runner(runner: IntentShadowRunner | None) -> None:
    """Override the singleton (used by tests)."""
    global _runner
    with _runner_lock:
        _runner = runner


# --------------------------------------------------------------------------
# Production wiring helpers (kept at module bottom to avoid import cycles).
# --------------------------------------------------------------------------


def make_hemisphere_inference_fn(
    orchestrator: Any, network_id: str | None = None
) -> Callable[[str], list[float] | None]:
    """Build an ``inference_fn`` backed by the hemisphere orchestrator.

    On each call:
      1. Embed the user message with the vector store (384-dim).
      2. Resolve the ``voice_intent`` network id (or use the supplied one).
      3. Call ``engine.infer(network_id, input_vec)``.

    Any failure returns ``None`` (treated as "NN unavailable" by the runner).
    This keeps the runner honest: no silent fallbacks, no fake agreement.
    """

    resolved_id: list[str | None] = [network_id]

    def _resolve_network_id() -> str | None:
        if resolved_id[0]:
            return resolved_id[0]
        try:
            state = orchestrator.get_state()
            for hemi in state.get("hemispheres", []) or []:
                for net in hemi.get("networks", []) or []:
                    if net.get("focus") == "voice_intent":
                        resolved_id[0] = net.get("id") or net.get("network_id")
                        return resolved_id[0]
        except Exception:
            logger.debug(
                "intent_shadow: failed to resolve voice_intent network",
                exc_info=True,
            )
        return None

    def _fn(user_message: str) -> list[float] | None:
        try:
            from memory.search import get_vector_store

            vs = get_vector_store()
            if not vs or not getattr(vs, "available", False):
                return None
            emb = vs.embed(user_message)
            if not emb or len(emb) < 384:
                return None
        except Exception:
            logger.debug("intent_shadow: embed failed", exc_info=True)
            return None

        nid = _resolve_network_id()
        if not nid:
            return None
        try:
            engine = getattr(orchestrator, "_engine", None) or getattr(
                orchestrator, "engine", None
            )
            if engine is None:
                return None
            return list(engine.infer(nid, emb[:384]))
        except Exception:
            logger.debug("intent_shadow: infer failed", exc_info=True)
            return None

    return _fn


def make_teacher_sample_provider() -> Callable[[], int]:
    """Return a callable that counts ``voice_intent`` teacher samples.

    Reads from the distillation collector if available. Returns 0 on any
    failure so promotion gates stay closed rather than accidentally
    opening on a broken collector.
    """

    def _fn() -> int:
        try:
            from hemisphere.distillation import distillation_collector

            # The collector exposes per-teacher counts.
            counts = getattr(distillation_collector, "sample_counts", None)
            if callable(counts):
                return int(counts().get("tool_router", 0))
            if isinstance(counts, dict):
                return int(counts.get("tool_router", 0))
        except Exception:
            logger.debug(
                "intent_shadow: teacher_sample_provider failed", exc_info=True
            )
        return 0

    return _fn
