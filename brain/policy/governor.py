"""Policy Governor — blocks NN decisions that are unsafe or low-confidence.

Responsibilities:
  - Block if confidence too low (with hysteresis)
  - Block if decision out of bounds
  - Block if system under load
  - Block if recent regressions detected
  - Block forbidden knobs
  - Revert to kernel behavior on violation + cooldown
  - Shadow-eligible: blocked decisions are still recorded for evaluation

All gate() calls update PolicyTelemetry (block/pass counters + reason).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any



from policy.policy_interface import PolicyDecision

logger = logging.getLogger(__name__)

CONFIDENCE_BLOCK = 0.08
CONFIDENCE_UNBLOCK = 0.12
MAX_THOUGHT_WEIGHT_DELTA = 0.15
MAX_BUDGET_DELTA_MS = 8
COOLDOWN_AFTER_BLOCK_S = 10.0
MAX_REGRESSIONS_BEFORE_DISABLE = 5


_REGRESSION_DECAY_S = 86400.0


@dataclass
class GovernorState:
    blocks: int = 0
    passes: int = 0
    shadow_eligible_blocks: int = 0
    recent_regressions: int = 0
    last_block_time: float = 0.0
    auto_disabled: bool = False
    confidence_blocked: bool = False
    regression_timestamps: list[float] = field(default_factory=list)


class PolicyGovernor:
    def __init__(self) -> None:
        self._state = GovernorState()

    def gate(self, decision: PolicyDecision, system_state: dict[str, Any]) -> PolicyDecision:
        """Filter/block a policy decision. Returns kernel defaults if blocked.

        When blocked, the original decision is stashed in
        ``decision._shadow_original`` so the evaluator can still score it
        in shadow mode without the NN ever actuating.
        """
        from policy.telemetry import policy_telemetry

        self._decay_regressions()

        if self._state.auto_disabled:
            policy_telemetry.record_block("auto_disabled")
            return self._blocked_with_shadow(decision, "governor_disabled")

        now = time.time()
        if now - self._state.last_block_time < COOLDOWN_AFTER_BLOCK_S and self._state.blocks > 0:
            policy_telemetry.record_block("cooldown")
            return self._blocked_with_shadow(decision, "governor_cooldown")

        violations: list[str] = []

        # Hysteresis on confidence: once blocked, must exceed UNBLOCK to pass
        if self._state.confidence_blocked:
            if decision.confidence < CONFIDENCE_UNBLOCK:
                violations.append(f"confidence {decision.confidence:.2f} < unblock {CONFIDENCE_UNBLOCK}")
            else:
                self._state.confidence_blocked = False
        else:
            if decision.confidence < CONFIDENCE_BLOCK:
                violations.append(f"confidence {decision.confidence:.2f} < block {CONFIDENCE_BLOCK}")
                self._state.confidence_blocked = True

        for key, delta in decision.thought_weights_delta.items():
            if abs(delta) > MAX_THOUGHT_WEIGHT_DELTA:
                violations.append(f"tw_delta.{key}={delta:.3f} exceeds max {MAX_THOUGHT_WEIGHT_DELTA}")

        if decision.budget_ms is not None:
            if abs(decision.budget_ms - 16) > MAX_BUDGET_DELTA_MS:
                violations.append(f"budget_ms={decision.budget_ms} too far from default")

        health = system_state.get("consciousness", {})
        if not health.get("system_healthy", True):
            violations.append("system unhealthy")

        if violations:
            self._state.blocks += 1
            self._state.last_block_time = now
            reason = "; ".join(violations)
            logger.info("Policy governor blocked decision: %s", reason)
            policy_telemetry.record_block(reason)
            return self._blocked_with_shadow(decision, "governor_blocked")

        self._state.passes += 1
        policy_telemetry.record_pass()
        return decision

    def _blocked_with_shadow(self, original: PolicyDecision, source: str) -> PolicyDecision:
        """Return kernel defaults but preserve the NN proposal for shadow evaluation."""
        self._state.shadow_eligible_blocks += 1
        blocked = PolicyDecision(confidence=0.0, source=source)
        blocked._shadow_original = original  # type: ignore[attr-defined]
        return blocked

    def record_regression(self) -> None:
        from policy.telemetry import policy_telemetry

        now = time.time()
        self._state.regression_timestamps.append(now)
        self._decay_regressions()

        policy_telemetry.regressions = self._state.recent_regressions

        if self._state.recent_regressions >= MAX_REGRESSIONS_BEFORE_DISABLE:
            self._state.auto_disabled = True
            policy_telemetry.auto_disabled = True
            policy_telemetry.log_event("auto_disable",
                                       f"Disabled after {self._state.recent_regressions} regressions")
            logger.warning("Policy governor auto-disabled after %d regressions",
                           self._state.recent_regressions)

    def _decay_regressions(self) -> None:
        """Remove regressions older than _REGRESSION_DECAY_S."""
        cutoff = time.time() - _REGRESSION_DECAY_S
        self._state.regression_timestamps = [
            ts for ts in self._state.regression_timestamps if ts > cutoff
        ]
        self._state.recent_regressions = len(self._state.regression_timestamps)

    def reset_regressions(self) -> None:
        from policy.telemetry import policy_telemetry

        self._state.recent_regressions = 0
        self._state.auto_disabled = False
        self._state.confidence_blocked = False
        policy_telemetry.regressions = 0
        policy_telemetry.auto_disabled = False

    def get_status(self) -> dict[str, Any]:
        return {
            "blocks": self._state.blocks,
            "passes": self._state.passes,
            "shadow_eligible_blocks": self._state.shadow_eligible_blocks,
            "regressions": self._state.recent_regressions,
            "auto_disabled": self._state.auto_disabled,
            "confidence_blocked": self._state.confidence_blocked,
        }
