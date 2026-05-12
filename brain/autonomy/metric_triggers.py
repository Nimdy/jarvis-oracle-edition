"""Metric Triggers — drives autonomous research from measurable system deficits.

Watches consciousness analytics, policy telemetry, and health counters for
sustained degradation. Generates ResearchIntents with strong evidence backing,
replacing noisy thought-based triggers as the primary autonomy driver.

Philosophy: metrics are hard to generate and easy to validate — the opposite
of thoughts. Making metrics the primary trigger eliminates the "spam cannon"
problem while keeping thought-based triggers as secondary flavor.

Before firing, consults AutonomyPolicyMemory to avoid repeating historically
failed research. When the default tool for a metric has a low win rate,
rotates to an alternative tool hint (codebase ↔ web) instead of giving up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from autonomy.research_intent import ResearchIntent

logger = logging.getLogger(__name__)

EVAL_INTERVAL_S = 60.0
DEFICIT_SUSTAIN_S = 120.0
TRIGGER_COOLDOWN_S = 1800.0
VETO_MIN_OUTCOMES = 3
VETO_WIN_RATE_THRESHOLD = 0.15
VETO_EXTENDED_COOLDOWN_S = 14400.0
SEVERITY_OVERRIDE_DURATION_S = 480.0
TOOL_ROTATION_WIN_RATE = 0.25
_TOOL_ALTERNATIVES: dict[str, str] = {
    "web": "codebase",
    "codebase": "web",
    "memory": "codebase",
    "introspection": "codebase",
}


@dataclass
class _TriggerState:
    metric_name: str
    in_deficit: bool = False
    deficit_start: float = 0.0
    last_triggered: float = 0.0
    trigger_count: int = 0
    veto_count: int = 0
    last_veto_reason: str = ""


@dataclass
class MetricDeficit:
    metric: str
    current_value: float
    threshold: float
    deficit_duration_s: float
    severity: str


# Each trigger maps a metric to a concrete, actionable research question.
_TRIGGER_DEFS: dict[str, dict[str, Any]] = {
    "confidence_volatility": {
        "threshold": 0.15, "direction": "above",
        "question": "What techniques reduce confidence volatility in adaptive decision systems?",
        "tool_hint": "web", "scope": "external_ok",
        "tags": ("confidence", "stability"),
    },
    "tick_p95_ms": {
        "threshold": 15.0, "direction": "above",
        "question": "What optimizations reduce tick processing latency in event-driven kernel loops?",
        "tool_hint": "codebase", "scope": "local_only",
        "tags": ("performance", "kernel"),
    },
    "barge_in_rate": {
        "threshold": 0.2, "direction": "above",
        "question": "What pacing heuristics reduce barge-in interruptions during speech synthesis?",
        "tool_hint": "web", "scope": "external_ok",
        "tags": ("barge_in", "pacing", "speech"),
    },
    "reasoning_coherence": {
        "threshold": 0.5, "direction": "below",
        "question": "How can reasoning coherence be improved in multi-modal consciousness systems?",
        "tool_hint": "codebase", "scope": "local_only",
        "tags": ("reasoning", "coherence"),
    },
    "processing_health": {
        "threshold": 0.5, "direction": "below",
        "question": "What patterns improve processing health in budget-constrained kernel architectures?",
        "tool_hint": "codebase", "scope": "local_only",
        "tags": ("health", "processing"),
    },
    "shadow_default_win_rate": {
        "threshold": 0.95, "direction": "above",
        "question": "How should shadow evaluation differentiate true wins from default-safe no-ops in policy learning?",
        "tool_hint": "web", "scope": "external_ok",
        "tags": ("policy", "evaluation", "shadow"),
    },
    "memory_recall_miss_rate": {
        "threshold": 0.3, "direction": "above",
        "question": "What indexing strategies improve semantic memory recall accuracy?",
        "tool_hint": "codebase", "scope": "local_only",
        "tags": ("memory", "recall"),
    },
    "friction_rate": {
        "threshold": 0.15, "direction": "above",
        "question": "What conversation patterns cause repeated user corrections, rephrases, or dissatisfaction?",
        "tool_hint": "introspection", "scope": "local_only",
        "tags": ("conversation_quality", "friction"),
    },
}


class MetricTriggers:
    """Watches system metrics and generates research intents from sustained deficits.

    Called periodically from the autonomy orchestrator with the latest metric
    snapshot. When a metric stays in deficit for ``DEFICIT_SUSTAIN_S``, a
    single research intent is emitted (with a per-metric cooldown to prevent
    spam). This makes autonomy *metric-driven, not thought-driven*.
    """

    def __init__(self) -> None:
        self._states: dict[str, _TriggerState] = {
            k: _TriggerState(metric_name=k) for k in _TRIGGER_DEFS
        }
        self._last_eval_time: float = 0.0
        self._total_triggers: int = 0
        self._total_vetoed: int = 0
        self._total_rotated: int = 0
        self._active_deficits: dict[str, MetricDeficit] = {}
        self._policy_memory: Any = None

    def set_policy_memory(self, policy_memory: Any) -> None:
        """Wire the policy memory for pre-fire veto checks."""
        self._policy_memory = policy_memory

    def evaluate(
        self,
        metrics: dict[str, float],
        enqueue_cb: Callable[[ResearchIntent], bool],
    ) -> list[ResearchIntent]:
        """Evaluate current metrics; return any newly generated intents."""
        now = time.time()
        if now - self._last_eval_time < EVAL_INTERVAL_S:
            return []
        self._last_eval_time = now

        generated: list[ResearchIntent] = []

        for metric_name, defn in _TRIGGER_DEFS.items():
            value = metrics.get(metric_name)
            if value is None:
                continue

            state = self._states[metric_name]
            threshold = defn["threshold"]
            in_deficit = (
                value > threshold if defn["direction"] == "above" else value < threshold
            )

            if in_deficit:
                if not state.in_deficit:
                    state.in_deficit = True
                    state.deficit_start = now

                duration = now - state.deficit_start
                severity = (
                    "high" if duration > DEFICIT_SUSTAIN_S * 2
                    else "medium" if duration > DEFICIT_SUSTAIN_S
                    else "low"
                )
                self._active_deficits[metric_name] = MetricDeficit(
                    metric=metric_name, current_value=value,
                    threshold=threshold, deficit_duration_s=duration,
                    severity=severity,
                )

                if duration >= DEFICIT_SUSTAIN_S and now - state.last_triggered >= TRIGGER_COOLDOWN_S:
                    veto = self._check_policy_veto(metric_name, defn, duration)
                    if veto:
                        state.veto_count += 1
                        state.last_veto_reason = veto
                        state.last_triggered = now
                        self._total_vetoed += 1
                        logger.info(
                            "Metric trigger vetoed: %s — %s", metric_name, veto,
                        )
                        continue

                    tool_hint = self._maybe_rotate_tool(metric_name, defn)
                    intent = self._create_intent(metric_name, defn, value, duration, tool_hint)
                    if enqueue_cb(intent):
                        state.last_triggered = now
                        state.trigger_count += 1
                        self._total_triggers += 1
                        generated.append(intent)
                        logger.info(
                            "Metric trigger: %s=%.3f (thresh=%.3f, deficit=%.0fs, tool=%s)",
                            metric_name, value, threshold, duration, tool_hint,
                        )
            else:
                state.in_deficit = False
                state.deficit_start = 0.0
                self._active_deficits.pop(metric_name, None)

        return generated

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_triggers": self._total_triggers,
            "total_vetoed": self._total_vetoed,
            "total_rotated": self._total_rotated,
            "active_deficits": {
                k: {"value": round(v.current_value, 3), "threshold": v.threshold,
                     "duration_s": round(v.deficit_duration_s, 1), "severity": v.severity}
                for k, v in self._active_deficits.items()
            },
            "trigger_counts": {
                k: v.trigger_count for k, v in self._states.items() if v.trigger_count > 0
            },
            "veto_counts": {
                k: v.veto_count for k, v in self._states.items() if v.veto_count > 0
            },
        }

    def get_active_deficit_count(self) -> int:
        return len(self._active_deficits)

    # -- Phase 6.5: L3 escalation candidates --------------------------------

    # Deficit must be sustained beyond 8 minutes and L1 research
    # exhausted (>= 5 attempts with < 15% win rate) before this module
    # will nominate a metric for human escalation. These numbers come
    # from the Phase 6.5 plan and are intentionally strict — the whole
    # point of the escalation channel is that it represents *rare*
    # situations where L1 has provably failed.
    _ESCALATION_MIN_DEFICIT_S = 480.0
    _ESCALATION_MIN_L1_ATTEMPTS = 5
    _ESCALATION_MAX_WIN_RATE = 0.15

    def get_escalation_candidates(
        self, live_autonomy_level: int,
    ) -> list["EscalationRequest"]:
        """Return escalation-ready requests for metrics that meet every gate.

        Gates (all must hold for a metric to appear):
        - Metric is currently in deficit with duration > 480s (8 min).
        - The policy veto has fired at least once for this metric (i.e.
          ``state.veto_count >= 1``), indicating live L1 research has
          been blocked by historical failure.
        - Historical L1 attempts for the metric's tag cluster is
          ``>= 5`` (pulled from policy memory).
        - Historical win rate for the tag cluster is ``< 0.15``.
        - ``live_autonomy_level >= 3`` — auto-generated escalation
          requests require live L3 per the Phase 6.5 "two escalation
          generation modes" invariant. Attestation does NOT satisfy
          this gate; the operator must first manually promote to L3.

        This function is a pure getter — it does not enqueue, does not
        dedupe, and does not mutate state. The caller is expected to
        pass each returned request through
        :meth:`autonomy.escalation.EscalationStore.submit`, which
        handles rate limiting, duplicate-protection, and persistence.
        """
        from autonomy.escalation import (  # late import to avoid cycle
            EscalationRequest,
            METRIC_ESCALATION_POLICY,
            build_request_from_metric_deficit,
        )
        if live_autonomy_level < 3:
            return []
        if self._policy_memory is None:
            return []

        out: list[EscalationRequest] = []
        for metric, defn in _TRIGGER_DEFS.items():
            if metric not in METRIC_ESCALATION_POLICY:
                continue
            deficit = self._active_deficits.get(metric)
            if deficit is None:
                continue
            if deficit.deficit_duration_s <= self._ESCALATION_MIN_DEFICIT_S:
                continue
            state = self._states[metric]
            if state.veto_count < 1:
                continue
            try:
                prior = self._policy_memory.get_topic_prior(defn.get("tags", ()))
            except Exception:
                continue
            total = int(getattr(prior, "total", 0))
            win_rate = float(getattr(prior, "win_rate", 0.0))
            if total < self._ESCALATION_MIN_L1_ATTEMPTS:
                continue
            if win_rate >= self._ESCALATION_MAX_WIN_RATE:
                continue

            try:
                out.append(build_request_from_metric_deficit(
                    metric=metric,
                    current_value=deficit.current_value,
                    threshold=deficit.threshold,
                    deficit_duration_s=deficit.deficit_duration_s,
                    l1_attempts=total,
                    win_rate=win_rate,
                    live_autonomy_level=live_autonomy_level,
                ))
            except Exception:
                logger.exception(
                    "Failed to build escalation request for metric %s", metric,
                )
        return out

    def _check_policy_veto(
        self, metric_name: str, defn: dict[str, Any], duration: float,
    ) -> str:
        """Consult policy memory to decide if this trigger should be vetoed.

        Returns empty string if allowed, or a reason string if vetoed.
        Severity override: "high" deficits (sustained > 8 min) always fire,
        even if historically bad — the system is in real trouble.
        """
        if self._policy_memory is None:
            return ""

        if duration >= SEVERITY_OVERRIDE_DURATION_S:
            return ""

        try:
            tags = defn.get("tags", ())
            prior = self._policy_memory.get_topic_prior(tags)

            if prior.total < VETO_MIN_OUTCOMES:
                return ""

            if prior.win_rate < VETO_WIN_RATE_THRESHOLD:
                return (
                    f"policy_veto: {prior.total} attempts, "
                    f"{prior.win_rate:.0%} win rate (below {VETO_WIN_RATE_THRESHOLD:.0%})"
                )
        except Exception:
            pass

        return ""

    def _maybe_rotate_tool(self, metric_name: str, defn: dict[str, Any]) -> str:
        """If the default tool for this metric has a low win rate, try the alternative.

        Returns the tool hint to use (may be the original or a rotation).
        """
        default_tool = defn["tool_hint"]
        if self._policy_memory is None:
            return default_tool

        try:
            tool_prior = self._policy_memory.get_tool_prior(default_tool)
            if tool_prior.total < VETO_MIN_OUTCOMES:
                return default_tool

            if tool_prior.win_rate < TOOL_ROTATION_WIN_RATE:
                alt = _TOOL_ALTERNATIVES.get(default_tool, default_tool)
                if alt != default_tool:
                    self._total_rotated += 1
                    logger.info(
                        "Tool rotation for %s: %s → %s (win_rate=%.0f%%)",
                        metric_name, default_tool, alt, tool_prior.win_rate,
                    )
                    return alt
        except Exception:
            pass

        return default_tool

    @staticmethod
    def _create_intent(
        metric_name: str, defn: dict[str, Any],
        current_value: float, deficit_duration: float,
        tool_hint: str | None = None,
    ) -> ResearchIntent:
        if deficit_duration > DEFICIT_SUSTAIN_S * 3:
            priority = 0.85
        elif deficit_duration > DEFICIT_SUSTAIN_S * 2:
            priority = 0.70
        else:
            priority = 0.55

        hint = tool_hint or defn["tool_hint"]
        scope = defn["scope"]
        if hint != defn["tool_hint"]:
            scope = "external_ok" if hint == "web" else "local_only"

        return ResearchIntent(
            question=defn["question"],
            source_event=f"metric:{metric_name}",
            source_hint=hint,
            priority=priority,
            scope=scope,
            tag_cluster=defn["tags"],
            trigger_count=1,
            reason=(
                f"Sustained metric deficit: {metric_name}={current_value:.3f} "
                f"(threshold={defn['threshold']}, duration={deficit_duration:.0f}s)"
            ),
        )
