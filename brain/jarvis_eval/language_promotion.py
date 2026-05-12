"""Language promotion governor for Phase D eval gates.

Manages per-response-class promotion state: shadow → canary → live.
Promotion requires sustained green gate scores. Regression triggers
automatic rollback to shadow.

State persisted to ~/.jarvis/language_promotion.json.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from jarvis_eval.language_scorers import (
    BOUNDED_RESPONSE_CLASSES,
    compute_gate_scores,
    classify_gate,
    classify_gate_reason,
)

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
PROMOTION_STATE_PATH = JARVIS_DIR / "language_promotion.json"

PromotionLevel = Literal["shadow", "canary", "live"]

# How many consecutive green evaluations needed for promotion
SHADOW_TO_CANARY_THRESHOLD = 5
CANARY_TO_LIVE_THRESHOLD = 10

# How many consecutive red evaluations trigger rollback
ROLLBACK_THRESHOLD = 3

# Minimum time in each level before promotion (seconds)
MIN_SHADOW_DWELL_S = 3600.0     # 1 hour
MIN_CANARY_DWELL_S = 7200.0     # 2 hours


@dataclass
class ClassPromotionState:
    """Tracks promotion state for a single response class."""

    response_class: str
    level: PromotionLevel = "shadow"
    consecutive_green: int = 0
    consecutive_red: int = 0
    last_evaluation_time: float = 0.0
    level_entered_at: float = 0.0
    total_evaluations: int = 0
    last_scores: dict[str, float] = field(default_factory=dict)
    last_gate_color: str = "red"
    last_gate_reason: str = ""
    last_transition_at: float = 0.0
    last_transition_from: str = ""
    last_transition_to: str = ""
    last_transition_reason: str = ""
    last_transition_scores: dict[str, float] = field(default_factory=dict)
    last_rollback_reason: str = ""
    promotion_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LanguagePromotionGovernor:
    """Manages promotion state for all bounded response classes."""

    _instance: LanguagePromotionGovernor | None = None

    def __init__(self) -> None:
        self._states: dict[str, ClassPromotionState] = {}
        self._loaded = False
        self._last_eval_signature = ""

        for rc in BOUNDED_RESPONSE_CLASSES:
            self._states[rc] = ClassPromotionState(
                response_class=rc,
                level_entered_at=time.time(),
            )

    @classmethod
    def get_instance(cls) -> LanguagePromotionGovernor:
        if cls._instance is None:
            cls._instance = LanguagePromotionGovernor()
            cls._instance._load()
        return cls._instance

    def evaluate(
        self,
        corpus_stats: dict[str, Any],
        telemetry_stats: dict[str, Any],
        *,
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate all response classes and update promotion state.

        Returns a summary dict per response class with level, scores, color.
        """
        self._ensure_loaded()
        signature = self._build_signature(corpus_stats, telemetry_stats)
        if not force and signature and signature == self._last_eval_signature:
            return self.get_summary()

        now = time.time()
        results: dict[str, dict[str, Any]] = {}

        for rc in BOUNDED_RESPONSE_CLASSES:
            state = self._states.get(rc)
            if state is None:
                state = ClassPromotionState(
                    response_class=rc, level_entered_at=now,
                )
                self._states[rc] = state

            scores = compute_gate_scores(corpus_stats, telemetry_stats, rc)
            color = classify_gate(scores)
            reason = classify_gate_reason(scores)

            state.last_scores = scores
            state.last_gate_color = color
            state.last_gate_reason = reason
            state.last_evaluation_time = now
            state.total_evaluations += 1

            previous_level = state.level
            if color == "green":
                state.consecutive_green += 1
                state.consecutive_red = 0
                self._maybe_promote(state, now)
            elif color == "red":
                state.consecutive_red += 1
                state.consecutive_green = 0
                self._maybe_rollback(state, now)
            else:  # yellow
                state.consecutive_red = 0
                # Don't reset green — yellow is OK, just not promotable

            row = self._state_summary(state, now)
            row["level_changed"] = previous_level != state.level
            results[rc] = row

        self._last_eval_signature = signature
        self._save()
        return results

    def get_level(self, response_class: str) -> PromotionLevel:
        """Get current promotion level for a response class."""
        self._ensure_loaded()
        state = self._states.get(response_class)
        if state is None:
            return "shadow"
        return state.level

    def get_summary(self) -> dict[str, Any]:
        """Return summary of all promotion states for dashboard."""
        self._ensure_loaded()
        now = time.time()
        summary: dict[str, Any] = {}
        for rc, state in self._states.items():
            summary[rc] = self._state_summary(state, now)
        return summary

    def get_live_artifact_id(self) -> str | None:
        """Return the Phase E live artifact id (P1.5 wiring).

        Delegates to :class:`LanguageKernelRegistry` so the governor
        can surface "which checkpoint identity is live" without
        introducing a new activation path. This is a read-only cross
        reference; the governor itself does not own artifact identity.
        """
        try:
            from language.kernel import get_language_kernel_registry

            live = get_language_kernel_registry().get_live_artifact()
            return live.artifact_id if live is not None else None
        except Exception:
            logger.debug(
                "language_promotion: failed to read kernel registry",
                exc_info=True,
            )
            return None

    def _state_summary(self, state: ClassPromotionState, now: float) -> dict[str, Any]:
        return {
            "level": state.level,
            "color": state.last_gate_color,
            "gate_reason": state.last_gate_reason,
            "consecutive_green": state.consecutive_green,
            "consecutive_red": state.consecutive_red,
            "total_evaluations": state.total_evaluations,
            "scores": state.last_scores,
            "last_evaluation_time": state.last_evaluation_time,
            "level_entered_at": state.level_entered_at,
            "dwell_s": max(0.0, now - state.level_entered_at) if state.level_entered_at > 0 else 0.0,
            "last_transition_at": state.last_transition_at,
            "last_transition_from": state.last_transition_from,
            "last_transition_to": state.last_transition_to,
            "last_transition_reason": state.last_transition_reason,
            "last_transition_scores": state.last_transition_scores,
            "last_rollback_reason": state.last_rollback_reason,
            "promotion_history_len": len(state.promotion_history),
        }

    def _maybe_promote(self, state: ClassPromotionState, now: float) -> None:
        dwell = now - state.level_entered_at

        if state.level == "shadow":
            if (state.consecutive_green >= SHADOW_TO_CANARY_THRESHOLD
                    and dwell >= MIN_SHADOW_DWELL_S):
                self._set_level(
                    state,
                    "canary",
                    now,
                    reason=(
                        f"green_streak={state.consecutive_green} "
                        f"dwell_s={dwell:.1f} "
                        f"thresholds=({SHADOW_TO_CANARY_THRESHOLD},{MIN_SHADOW_DWELL_S:.0f})"
                    ),
                    transition_type="promotion",
                )
        elif state.level == "canary":
            if (state.consecutive_green >= CANARY_TO_LIVE_THRESHOLD
                    and dwell >= MIN_CANARY_DWELL_S):
                self._set_level(
                    state,
                    "live",
                    now,
                    reason=(
                        f"green_streak={state.consecutive_green} "
                        f"dwell_s={dwell:.1f} "
                        f"thresholds=({CANARY_TO_LIVE_THRESHOLD},{MIN_CANARY_DWELL_S:.0f})"
                    ),
                    transition_type="promotion",
                )

    def _maybe_rollback(self, state: ClassPromotionState, now: float) -> None:
        if state.consecutive_red >= ROLLBACK_THRESHOLD:
            if state.level != "shadow":
                old_level = state.level
                rollback_reason = (
                    f"red_streak={state.consecutive_red} "
                    f"threshold={ROLLBACK_THRESHOLD} "
                    f"last_gate_color={state.last_gate_color}"
                )
                self._set_level(
                    state,
                    "shadow",
                    now,
                    reason=rollback_reason,
                    transition_type="rollback",
                )
                logger.warning(
                    "Language gate rollback: %s %s → shadow "
                    "(%d consecutive red evaluations, reason=%s)",
                    state.response_class, old_level,
                    state.consecutive_red,
                    rollback_reason,
                )

    def _set_level(
        self,
        state: ClassPromotionState,
        new_level: PromotionLevel,
        now: float,
        *,
        reason: str = "",
        transition_type: str = "promotion",
    ) -> None:
        old_level = state.level
        if old_level == new_level:
            return
        dwell_before = max(0.0, now - state.level_entered_at)
        state.level = new_level
        state.level_entered_at = now
        state.last_transition_at = now
        state.last_transition_from = old_level
        state.last_transition_to = new_level
        state.last_transition_reason = reason
        state.last_transition_scores = dict(state.last_scores)
        if transition_type == "rollback":
            state.last_rollback_reason = reason
        state.consecutive_green = 0
        state.consecutive_red = 0
        state.promotion_history.append({
            "type": transition_type,
            "from": old_level,
            "to": new_level,
            "at": now,
            "reason": reason,
            "gate_color": state.last_gate_color,
            "dwell_s_before": round(dwell_before, 3),
            "scores": dict(state.last_scores),
        })
        # Keep history bounded
        if len(state.promotion_history) > 50:
            state.promotion_history = state.promotion_history[-50:]

        logger.info(
            "Language gate transition: %s %s → %s (%s, reason=%s)",
            state.response_class, old_level, new_level, transition_type, reason or "--",
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _build_signature(
        self,
        corpus_stats: dict[str, Any],
        telemetry_stats: dict[str, Any],
    ) -> str:
        parts = [
            str(int(corpus_stats.get("total_examples", 0) or 0)),
            str(float(corpus_stats.get("last_capture_ts", 0.0) or 0.0)),
            str(int(telemetry_stats.get("total_events", 0) or 0)),
            str(float(telemetry_stats.get("last_event_ts", 0.0) or 0.0)),
        ]
        return "|".join(parts)

    def _load(self) -> None:
        self._loaded = True
        if not PROMOTION_STATE_PATH.exists():
            return
        try:
            data = json.loads(PROMOTION_STATE_PATH.read_text(encoding="utf-8"))
            meta = data.pop("__meta__", {}) if isinstance(data, dict) else {}
            if isinstance(meta, dict):
                self._last_eval_signature = str(meta.get("last_eval_signature", "") or "")
            for rc, raw in data.items():
                if rc in self._states:
                    state = self._states[rc]
                    level = str(raw.get("level", "shadow") or "shadow")
                    if level not in ("shadow", "canary", "live"):
                        level = "shadow"
                    state.level = level  # type: ignore[assignment]
                    state.consecutive_green = int(raw.get("consecutive_green", 0) or 0)
                    state.consecutive_red = int(raw.get("consecutive_red", 0) or 0)
                    state.last_evaluation_time = float(raw.get("last_evaluation_time", 0.0) or 0.0)
                    state.level_entered_at = float(raw.get("level_entered_at", 0.0) or 0.0)
                    if state.level_entered_at <= 0:
                        state.level_entered_at = time.time()
                    state.total_evaluations = int(raw.get("total_evaluations", 0) or 0)
                    state.last_scores = raw.get("last_scores", {}) if isinstance(raw.get("last_scores"), dict) else {}
                    state.last_gate_color = str(raw.get("last_gate_color", "red") or "red")
                    state.last_gate_reason = str(raw.get("last_gate_reason", "") or "")
                    state.last_transition_at = float(raw.get("last_transition_at", 0.0) or 0.0)
                    state.last_transition_from = str(raw.get("last_transition_from", "") or "")
                    state.last_transition_to = str(raw.get("last_transition_to", "") or "")
                    state.last_transition_reason = str(raw.get("last_transition_reason", "") or "")
                    state.last_transition_scores = (
                        raw.get("last_transition_scores", {})
                        if isinstance(raw.get("last_transition_scores"), dict)
                        else {}
                    )
                    state.last_rollback_reason = str(raw.get("last_rollback_reason", "") or "")
                    hist = raw.get("promotion_history", [])
                    state.promotion_history = hist if isinstance(hist, list) else []
            logger.info("Loaded language promotion state for %d classes", len(data))
        except Exception:
            logger.warning("Failed to load language promotion state", exc_info=True)

    def _save(self) -> None:
        try:
            PROMOTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {rc: state.to_dict() for rc, state in self._states.items()}
            data["__meta__"] = {
                "schema_version": 2,
                "saved_at": time.time(),
                "last_eval_signature": self._last_eval_signature,
            }
            tmp: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=str(PROMOTION_STATE_PATH.parent),
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    json.dump(data, f, indent=2, default=str)
                    tmp = f.name
                os.replace(tmp, str(PROMOTION_STATE_PATH))
            except Exception:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except Exception:
            logger.warning("Failed to save language promotion state", exc_info=True)
