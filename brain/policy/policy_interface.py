"""Policy Interface — stable boundary the kernel calls.

The kernel calls policy.decide(state) -> PolicyDecision.
If policy is disabled/unsafe: returns defaults (kernel behavior unchanged).

All decisions update PolicyTelemetry on the hot path (counters + EMAs only).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    thought_weights_delta: dict[str, float] = field(default_factory=dict)
    budget_ms: int | None = None
    run_tasks: list[str] = field(default_factory=list)
    mutation_rank: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "kernel_default"
    # Behavioral params the policy can control (Tier 5)
    response_length_hint: str = ""  # "brief", "normal", "detailed"
    proactivity_cooldown_s: float | None = None
    interruption_threshold: float | None = None
    attention_decay_rate: float | None = None
    memory_reinforcement_multiplier: float | None = None
    suggested_mode: str = ""  # operational mode suggestion


DEFAULT_DECISION = PolicyDecision(confidence=0.0, source="kernel_default")


class PolicyInterface:
    """Top-level policy interface. Delegates to NN when enabled, else returns defaults."""

    def __init__(self) -> None:
        self._enabled = False
        self._nn_controller: Any = None
        self._governor: Any = None
        self._feature_flags: dict[str, bool] = {
            "budget_allocation": False,
            "task_scheduling": False,
            "thought_weight_delta": False,
            "mutation_ranking": False,
            "response_length": False,
            "proactivity_control": False,
            "interruption_control": False,
            "mode_suggestion": False,
        }
        self._decision_budget_ms = 5.0
        self._last_feature_enable_time: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def has_active_features(self) -> bool:
        return any(self._feature_flags.values())

    def set_nn_controller(self, controller: Any) -> None:
        self._nn_controller = controller

    def set_governor(self, governor: Any) -> None:
        self._governor = governor

    def enable(self) -> None:
        self._enabled = True
        from policy.telemetry import policy_telemetry
        policy_telemetry.active = True
        policy_telemetry.feature_flags = dict(self._feature_flags)
        policy_telemetry.log_event("enable", "Policy layer enabled")
        logger.info("Policy layer enabled")

    def disable(self) -> None:
        self._enabled = False
        from policy.telemetry import policy_telemetry
        policy_telemetry.active = False
        policy_telemetry.log_event("disable", "Policy layer disabled")
        logger.info("Policy layer disabled (kernel defaults)")

    def enable_feature(self, feature: str) -> None:
        if feature in self._feature_flags:
            self._feature_flags[feature] = True
            from policy.telemetry import policy_telemetry
            policy_telemetry.feature_flags = dict(self._feature_flags)
            policy_telemetry.log_event("feature_on", f"{feature} enabled")
            logger.info("Policy feature enabled: %s", feature)

    def disable_feature(self, feature: str) -> None:
        if feature in self._feature_flags:
            self._feature_flags[feature] = False
            from policy.telemetry import policy_telemetry
            policy_telemetry.feature_flags = dict(self._feature_flags)
            policy_telemetry.log_event("feature_off", f"{feature} disabled")
            logger.info("Policy feature disabled: %s", feature)

    def decide(self, state: dict[str, Any]) -> PolicyDecision:
        """Make a policy decision based on current consciousness state."""
        from policy.telemetry import policy_telemetry

        if not self._enabled or self._nn_controller is None:
            return PolicyDecision(confidence=0.0, source="kernel_default")

        t0 = time.perf_counter()

        try:
            raw_decision = self._nn_controller.forward(state)
            t_encode = (time.perf_counter() - t0) * 1000.0

            decision = self._apply_feature_flags(raw_decision)

            if self._governor:
                decision = self._governor.gate(decision, state)

            total_ms = (time.perf_counter() - t0) * 1000.0

            # Hot-path telemetry: counters + EMAs only
            policy_telemetry.record_decision(total_ms, t_encode)
            if total_ms > self._decision_budget_ms:
                policy_telemetry.record_overrun()

            return decision
        except Exception:
            logger.exception("Policy decision failed, returning defaults")
            policy_telemetry.log_event("error", "Decision failed, fallback to kernel")
            return PolicyDecision(confidence=0.0, source="fallback")

    SHADOW_EXPLORATION_SIGMA = 0.03

    def decide_raw(self, state: dict[str, Any]) -> PolicyDecision | None:
        """Return the raw NN decision WITHOUT governor gating, with exploration noise.

        For shadow evaluation only. Feature flags are NOT applied here so the
        evaluator can compare the NN's actual proposal against the kernel.
        Feature flags are only applied in decide() (the live path).
        """
        if not self._enabled or self._nn_controller is None:
            return None
        try:
            t0 = time.perf_counter()
            raw_decision = self._nn_controller.forward(state)
            self._add_exploration_noise(raw_decision)
            total_ms = (time.perf_counter() - t0) * 1000.0
            from policy.telemetry import policy_telemetry
            policy_telemetry.record_decision(total_ms, total_ms)
            return raw_decision
        except Exception:
            return None

    def _add_exploration_noise(self, decision: PolicyDecision) -> None:
        """Add small Gaussian noise to shadow-only proposals for diversity."""
        import random
        sigma = self.SHADOW_EXPLORATION_SIGMA
        for key in list(decision.thought_weights_delta):
            decision.thought_weights_delta[key] += random.gauss(0, sigma)
        if decision.budget_ms is not None:
            decision.budget_ms = max(8, min(24, decision.budget_ms + int(random.gauss(0, 1.5))))

    _FEATURE_THRESHOLDS: list[tuple[int, float, int, str]] = [
        #  (min_exp, min_wr, min_shadow_ab, feature_name)
        (200,  0.40, 100, "budget_allocation"),
        (500,  0.50, 200, "thought_weight_delta"),
        (500,  0.55, 200, "mode_suggestion"),
        (1000, 0.55, 300, "response_length"),
        (1000, 0.60, 300, "proactivity_control"),
    ]
    _FEATURE_COOLDOWN_S: float = 300.0

    def auto_enable_features(
        self, experience_count: int, win_rate: float, shadow_ab_total: int = 0,
    ) -> None:
        """Progressively enable NN features based on demonstrated competence.

        Called periodically from the engine. Features are earned, not scripted.
        At most one feature can be enabled per call to prevent mode/budget thrash.
        A cooldown of _FEATURE_COOLDOWN_S is enforced between consecutive enables.
        """
        now = time.time()
        if now - self._last_feature_enable_time < self._FEATURE_COOLDOWN_S:
            return

        for min_exp, min_wr, min_shadow, feature in self._FEATURE_THRESHOLDS:
            if self._feature_flags.get(feature):
                continue
            cooldown_ok = (now - self._last_feature_enable_time) >= self._FEATURE_COOLDOWN_S
            if (experience_count >= min_exp
                    and win_rate >= min_wr
                    and shadow_ab_total >= min_shadow
                    and cooldown_ok):
                self.enable_feature(feature)
                self._last_feature_enable_time = now
                logger.info(
                    "Policy feature enabled: %s (exp=%d shadow=%d wr=%.2f cooldown_ok=%s)",
                    feature, experience_count, shadow_ab_total, win_rate, cooldown_ok,
                )
                return  # at most 1 per tick

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "feature_flags": dict(self._feature_flags),
            "has_nn": self._nn_controller is not None,
            "has_governor": self._governor is not None,
        }

    def _apply_feature_flags(self, decision: PolicyDecision) -> PolicyDecision:
        if not self._feature_flags.get("budget_allocation"):
            decision.budget_ms = None
        if not self._feature_flags.get("task_scheduling"):
            decision.run_tasks = []
        if not self._feature_flags.get("thought_weight_delta"):
            decision.thought_weights_delta = {}
        if not self._feature_flags.get("mutation_ranking"):
            decision.mutation_rank = []
        if not self._feature_flags.get("response_length"):
            decision.response_length_hint = ""
        if not self._feature_flags.get("proactivity_control"):
            decision.proactivity_cooldown_s = None
        if not self._feature_flags.get("interruption_control"):
            decision.interruption_threshold = None
        if not self._feature_flags.get("mode_suggestion"):
            decision.suggested_mode = ""
        return decision
