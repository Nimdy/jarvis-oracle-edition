"""Mutation Governor — single gatekeeper for all kernel config changes.

Flow: proposal → validate bounds → risk score → cooldown check → stability gate
→ snapshot → apply → monitor → rollback if degraded.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from consciousness.events import event_bus, MUTATION_ROLLBACK

if TYPE_CHECKING:
    from consciousness.kernel_config import ConfigPatch, ConfigSnapshot, KernelConfig, Violation

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 180.0  # 3 minutes between mutations
MAX_RISK_SCORE = 0.7      # reject proposals above this
P95_REJECT_MS = 50.0      # reject if tick p95 exceeds this
BACKLOG_REJECT = 15       # reject if deferred queue exceeds this
MIN_AWARENESS_LEVEL = 0.4 # reject mutations when observer awareness is too low
MONITOR_WINDOW_S = 30.0   # post-mutation observation window
REGRESSION_THRESHOLD = 0.15  # metric degradation that triggers rollback
MAX_MUTATIONS_PER_HOUR = 12   # mutation fatigue: hard cap per rolling hour
MAX_MUTATIONS_PER_SESSION = 400  # absolute cap per session (prevents oscillation)


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


BOOT_STABILIZATION_S = _env_seconds("JARVIS_BOOT_STABILIZATION_S", 600.0)


@dataclass
class RiskAssessment:
    identity_risk: float = 0.0
    functional_risk: float = 0.0
    reversibility: float = 1.0
    benefit_potential: float = 0.0

    @property
    def overall(self) -> float:
        return (self.identity_risk * 0.4 + self.functional_risk * 0.4
                + (1.0 - self.reversibility) * 0.2)


@dataclass
class GovernorDecision:
    approved: bool
    risk: RiskAssessment
    violations: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SystemHealth:
    tick_p95_ms: float = 0.0
    deferred_backlog: int = 0
    avg_tick_ms: float = 0.0


@dataclass
class ApplyResult:
    snapshot_id: str
    applied_at: float
    monitor_until: float
    metrics_before: dict[str, float] = field(default_factory=dict)


class MutationGovernor:
    def __init__(self) -> None:
        self._boot_time: float = time.time()
        self._boot_stabilization_s: float = BOOT_STABILIZATION_S
        self._last_mutation_time: float = 0.0
        self._mutation_count: int = 0
        self._active_monitor: ApplyResult | None = None
        self._rollback_count: int = 0
        self._recent_rejections: list[dict[str, Any]] = []
        self._mutation_timestamps: list[float] = []  # rolling window for hourly rate
        self._session_mutation_count: int = 0  # resets each session (not persisted)
        self._total_rejections: int = 0

    @property
    def last_mutation_time(self) -> float:
        return self._last_mutation_time

    @property
    def mutation_count(self) -> int:
        return self._mutation_count

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    # -- evaluate proposal ---------------------------------------------------

    def evaluate(
        self,
        proposal_changes: dict[str, Any],
        proposal_confidence: float,
        current_config: KernelConfig,
        health: SystemHealth,
        awareness_level: float = 1.0,
    ) -> GovernorDecision:
        from consciousness.kernel_config import KernelConfig as KC

        violations: list[str] = []

        # 0. Awareness gate — if the observer hasn't accumulated enough
        # observations, the system doesn't have sufficient self-model to
        # safely self-modify.
        if awareness_level < MIN_AWARENESS_LEVEL:
            return GovernorDecision(
                approved=False,
                risk=RiskAssessment(),
                violations=[f"Low awareness: {awareness_level:.2f} < {MIN_AWARENESS_LEVEL}"],
                reasoning="Mutation rejected: insufficient observer awareness for safe self-modification",
            )

        # 1. Cooldown check (modulated by quarantine pressure)
        if self._boot_stabilization_s > 0.0:
            boot_elapsed = time.time() - self._boot_time
            if boot_elapsed < self._boot_stabilization_s:
                remaining = self._boot_stabilization_s - boot_elapsed
                return GovernorDecision(
                    approved=False,
                    risk=RiskAssessment(),
                    violations=[f"Boot stabilization active: {remaining:.0f}s remaining"],
                    reasoning="Mutation rejected: startup stabilization window",
                )

        effective_cd = COOLDOWN_SECONDS
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            _, cd_override = get_quarantine_pressure().mutation_rate_factor()
            if cd_override is not None:
                effective_cd = cd_override
        except Exception:
            pass
        elapsed = time.time() - self._last_mutation_time
        if elapsed < effective_cd:
            remaining = effective_cd - elapsed
            return GovernorDecision(
                approved=False,
                risk=RiskAssessment(),
                violations=[f"Cooldown active: {remaining:.0f}s remaining"],
                reasoning="Mutation rejected: cooldown period not elapsed",
            )

        # 1b. Mutation fatigue — hourly rate limit (modulated by quarantine pressure)
        now = time.time()
        hour_ago = now - 3600.0
        self._mutation_timestamps = [t for t in self._mutation_timestamps if t > hour_ago]
        effective_hourly_cap = self._graduated_hourly_cap(now)
        effective_cooldown = COOLDOWN_SECONDS
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            cap_override, cd_override = get_quarantine_pressure().mutation_rate_factor()
            if cap_override is not None:
                effective_hourly_cap = min(effective_hourly_cap, cap_override)
            if cd_override is not None:
                effective_cooldown = cd_override
        except Exception:
            pass
        if len(self._mutation_timestamps) >= effective_hourly_cap:
            return GovernorDecision(
                approved=False,
                risk=RiskAssessment(),
                violations=[f"Mutation fatigue: {len(self._mutation_timestamps)} mutations in last hour (max {effective_hourly_cap})"],
                reasoning="Mutation rejected: hourly rate limit reached — parameter oscillation prevention",
            )

        # 1c. Session cap
        if self._session_mutation_count >= MAX_MUTATIONS_PER_SESSION:
            return GovernorDecision(
                approved=False,
                risk=RiskAssessment(),
                violations=[f"Session cap: {self._session_mutation_count} mutations (max {MAX_MUTATIONS_PER_SESSION})"],
                reasoning="Mutation rejected: session mutation cap reached",
            )

        # 2. Stability gate
        if health.tick_p95_ms > P95_REJECT_MS:
            return GovernorDecision(
                approved=False,
                risk=RiskAssessment(),
                violations=[f"System under load: p95={health.tick_p95_ms:.1f}ms > {P95_REJECT_MS}ms"],
                reasoning="Mutation rejected: system performance degraded",
            )
        if health.deferred_backlog > BACKLOG_REJECT:
            return GovernorDecision(
                approved=False,
                risk=RiskAssessment(),
                violations=[f"Deferred backlog={health.deferred_backlog} > {BACKLOG_REJECT}"],
                reasoning="Mutation rejected: operation backlog too large",
            )

        # 3. Validate bounds (simulate apply)
        import copy
        test_config = copy.deepcopy(current_config)
        flat = test_config._to_flat()
        flat.update(proposal_changes)
        test_config._from_flat(flat)
        config_violations = test_config.validate()
        for v in config_violations:
            violations.append(v.message)

        # 4. Risk assessment
        risk = self._assess_risk(proposal_changes, proposal_confidence, current_config)

        # 4b. Layer 8 active-lite quarantine friction
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            qp = get_quarantine_pressure()
            p = qp.current
            risk_addon = qp.mutation_risk_addon()
            if risk_addon > 0.0:
                risk.functional_risk = min(1.0, risk.functional_risk + risk_addon)
            if qp.should_reject_identity_mutation(risk.identity_risk):
                violations.append(
                    f"Quarantine pressure high+chronic: identity_risk {risk.identity_risk:.2f} > 0.3"
                )
                qp.record_promotion_blocked()
        except Exception:
            pass

        if risk.overall > MAX_RISK_SCORE:
            violations.append(f"Risk score {risk.overall:.2f} exceeds max {MAX_RISK_SCORE}")

        approved = len(violations) == 0
        reasoning = "Approved" if approved else f"Rejected: {'; '.join(violations)}"

        if approved:
            logger.info("Governor approved mutation (risk=%.2f, confidence=%.2f)",
                        risk.overall, proposal_confidence)
        else:
            logger.info("Governor rejected mutation: %s", reasoning)
            self._total_rejections += 1
            changes_summary = ", ".join(f"{k}={v}" for k, v in proposal_changes.items())[:80]
            self._recent_rejections.append({
                "changes": changes_summary,
                "reason": reasoning[:120],
                "time": time.time(),
            })
            if len(self._recent_rejections) > 10:
                self._recent_rejections = self._recent_rejections[-10:]

        return GovernorDecision(
            approved=approved,
            risk=risk,
            violations=violations,
            reasoning=reasoning,
        )

    # -- apply mutation ------------------------------------------------------

    def apply(
        self,
        patch: ConfigPatch,
        config: KernelConfig,
        health: SystemHealth,
    ) -> tuple[KernelConfig, ApplyResult]:
        snap = config.snapshot()
        config.save_snapshot(snap)

        new_config = config.apply_patch(patch)
        new_config.clamp()

        new_config.evolution.last_mutation = time.time()
        new_config.evolution.mutation_history.append(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}: {patch.description}"
        )
        if len(new_config.evolution.mutation_history) > 15:
            new_config.evolution.mutation_history = new_config.evolution.mutation_history[-15:]
        new_config.metadata.mutation_count += 1

        self._last_mutation_time = time.time()
        self._mutation_count += 1
        self._session_mutation_count += 1
        self._mutation_timestamps.append(self._last_mutation_time)

        result = ApplyResult(
            snapshot_id=snap.id,
            applied_at=time.time(),
            monitor_until=time.time() + MONITOR_WINDOW_S,
            metrics_before={
                "tick_p95_ms": health.tick_p95_ms,
                "avg_tick_ms": health.avg_tick_ms,
                "deferred_backlog": health.deferred_backlog,
            },
        )
        self._active_monitor = result

        new_config.save()
        logger.info("Mutation applied: %s (snapshot=%s)", patch.description, snap.id)
        return new_config, result

    # -- post-mutation monitoring -------------------------------------------

    def check_post_mutation(self, health: SystemHealth) -> bool:
        """Returns True if system is healthy, False if rollback needed."""
        if self._active_monitor is None:
            return True

        if time.time() < self._active_monitor.monitor_until:
            return True  # still in observation window

        before = self._active_monitor.metrics_before
        p95_before = before.get("tick_p95_ms", 0.0)
        p95_now = health.tick_p95_ms

        if p95_before > 0 and p95_now > p95_before * (1.0 + REGRESSION_THRESHOLD):
            logger.warning(
                "Post-mutation regression: p95 %.1fms -> %.1fms (%.0f%% increase)",
                p95_before, p95_now,
                ((p95_now - p95_before) / p95_before) * 100,
            )
            self._active_monitor = None
            return False

        backlog_before = before.get("deferred_backlog", 0)
        if health.deferred_backlog > max(backlog_before + 5, BACKLOG_REJECT):
            logger.warning("Post-mutation backlog regression: %d -> %d",
                           backlog_before, health.deferred_backlog)
            self._active_monitor = None
            return False

        logger.info("Post-mutation monitoring complete: system healthy")
        self._active_monitor = None
        return True

    # -- rollback -----------------------------------------------------------

    def rollback(self, snapshot_id: str) -> KernelConfig | None:
        from consciousness.kernel_config import KernelConfig

        snap = KernelConfig.load_snapshot(snapshot_id)
        if snap is None:
            logger.error("Cannot rollback: snapshot %s not found", snapshot_id)
            return None

        config = KernelConfig.from_snapshot(snap)
        config.save()
        self._rollback_count += 1
        self._active_monitor = None
        event_bus.emit(MUTATION_ROLLBACK, snapshot_id=snapshot_id,
                       rollback_count=self._rollback_count)
        logger.info("Rolled back to snapshot %s (total rollbacks: %d)",
                     snapshot_id, self._rollback_count)
        return config

    def get_active_monitor(self) -> ApplyResult | None:
        return self._active_monitor

    @property
    def mutations_this_hour(self) -> int:
        now = time.time()
        return sum(1 for t in self._mutation_timestamps if t > now - 3600.0)

    @property
    def recent_rejections(self) -> list[dict[str, Any]]:
        return list(self._recent_rejections)

    @property
    def total_rejections(self) -> int:
        return self._total_rejections

    # -- graduated hourly cap ------------------------------------------------

    def _graduated_hourly_cap(self, now: float) -> int:
        """Return the hourly mutation cap based on uptime.

        Fresh brains ramp up gradually:
          0–2 hours:  3/hour  (conservative exploration)
          2–8 hours:  6/hour  (moderate)
          8+ hours:  12/hour  (full rate)
        """
        uptime_h = (now - self._boot_time) / 3600.0
        if uptime_h < 2.0:
            return 3
        if uptime_h < 8.0:
            return 6
        return MAX_MUTATIONS_PER_HOUR

    # -- risk computation ---------------------------------------------------

    def _assess_risk(
        self,
        changes: dict[str, Any],
        confidence: float,
        config: KernelConfig,
    ) -> RiskAssessment:
        identity_risk = 0.0
        functional_risk = 0.0

        identity_keys = {"tw.philosophical", "tw.introspective", "ev.exploration_drive", "ev.stability_desire"}
        functional_keys = {"tw.reactive", "tw.contextual", "mp.decay_bias", "mp.association_threshold"}

        for key in changes:
            if key in identity_keys:
                identity_risk += 0.15
            if key in functional_keys:
                functional_risk += 0.1

        change_magnitude = len(changes)
        if change_magnitude > 4:
            identity_risk += 0.1
            functional_risk += 0.1

        reversibility = 0.9 if confidence > 0.8 else 0.6 if confidence > 0.5 else 0.3
        benefit_potential = min(1.0, confidence * 0.7 + (change_magnitude / 10) * 0.3)

        return RiskAssessment(
            identity_risk=min(1.0, identity_risk),
            functional_risk=min(1.0, functional_risk),
            reversibility=reversibility,
            benefit_potential=benefit_potential,
        )


mutation_governor = MutationGovernor()
