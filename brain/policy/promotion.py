"""Policy Promotion Pipeline — automated train -> shadow eval -> promote cycle.

Orchestrates:
1. Imitation learning from collected experience buffer
2. Shadow A/B evaluation (NN vs kernel)
3. Promotion when NN consistently wins
4. Staged feature flag rollout (budget -> scheduling -> thought weights)
5. Rollback on regression
6. M6 broadcast expansion: shadow dual-encoder migration (20→22 dim)

All promotion decisions flow through PolicyTelemetry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from policy.experience_buffer import ExperienceBuffer
from policy.evaluator import PolicyEvaluator
from policy.trainer import PolicyTrainer
from policy.registry import ModelRegistry
from policy.policy_nn import PolicyNNController, ARCHITECTURE_REGISTRY, TORCH_AVAILABLE
from policy.governor import PolicyGovernor
from policy.policy_interface import PolicyInterface
from policy.telemetry import policy_telemetry

logger = logging.getLogger(__name__)

TRAIN_MIN_EXPERIENCES = 30
MIN_NEW_EXPERIENCES = 15         # require new data since last train
TRAIN_INTERVAL_S = 1800.0       # retrain at most every 30 min
EVAL_INTERVAL_S = 30.0          # push eval stats every 30s
PROMOTION_WIN_RATE = 0.55
PROMOTION_MIN_DECISIONS = 100
FEATURE_ADVANCE_INTERVAL_S = 600.0   # min 10 min between feature advances
FEATURE_ADVANCE_MIN_DECISIONS = 50   # min shadow A/B decisions before advancing
ACTIVE_RETRAIN_INTERVAL_S = 3600.0   # retrain active model every 60 min
ACTIVE_RETRAIN_MIN_NEW = 30          # require 30 new experiences before retraining

STAGED_FEATURES = [
    "budget_allocation",
    "task_scheduling",
    "thought_weight_delta",
    "mutation_ranking",
]


@dataclass
class PromotionStatus:
    phase: str = "collecting"  # collecting, training, evaluating, promoting, active
    last_train_time: float = 0.0
    last_eval_time: float = 0.0
    last_train_experience_count: int = 0
    current_feature_index: int = 0
    total_promotions: int = 0
    total_rollbacks: int = 0
    last_feature_advance_time: float = 0.0


class PromotionPipeline:
    """Automated pipeline: collect experience -> train -> evaluate -> promote."""

    def __init__(
        self,
        buffer: ExperienceBuffer,
        evaluator: PolicyEvaluator,
        registry: ModelRegistry,
        governor: PolicyGovernor,
        interface: PolicyInterface,
    ):
        self._buffer = buffer
        self._evaluator = evaluator
        self._trainer = PolicyTrainer(buffer)
        self._registry = registry
        self._governor = governor
        self._interface = interface
        self._status = PromotionStatus(last_train_time=time.time())
        self._candidates: list[PolicyNNController] = []

        # M6 expansion: shadow dual-encoder runner
        self._shadow_runner: Any = None
        self._hemisphere_orchestrator: Any = None
        self._expansion_subscribed = False
        self._subscribe_expansion_event()
        self._resume_expansion_on_boot()

    @property
    def status(self) -> PromotionStatus:
        return self._status

    @property
    def shadow_runner(self) -> Any:
        return self._shadow_runner

    async def tick(self, now: float) -> None:
        """Called periodically from the main loop. Drives the promotion state machine."""
        if not TORCH_AVAILABLE:
            return

        if not self._expansion_subscribed:
            self._subscribe_expansion_event()

        if now - self._status.last_eval_time >= EVAL_INTERVAL_S:
            self._evaluator.update_telemetry()
            self._status.last_eval_time = now

        await self._tick_expansion()

        if self._status.phase == "collecting":
            buf_size = len(self._buffer)
            new_since_last = buf_size - self._status.last_train_experience_count
            if buf_size >= TRAIN_MIN_EXPERIENCES and new_since_last >= MIN_NEW_EXPERIENCES:
                elapsed = now - self._status.last_train_time
                if elapsed >= TRAIN_INTERVAL_S:
                    logger.info("Promotion: %d experiences (%d new), starting training (%.0fs since last)",
                                buf_size, new_since_last, elapsed)
                    self._status.phase = "training"
                    await self._train_candidates()

        elif self._status.phase == "evaluating":
            if self._should_promote_with_friction():
                self._status.phase = "promoting"
                await self._promote_best()
            elif self._evaluator.evaluate().total_decisions >= PROMOTION_MIN_DECISIONS * 2:
                report = self._evaluator.evaluate()
                if report.nn_win_rate < 0.45:
                    logger.info("NN underperforming (%.1f%% wins), back to collecting",
                                report.nn_win_rate * 100)
                    self._status.phase = "collecting"

        elif self._status.phase == "active":
            self._check_quarantine_rollback()
            if self._status.phase == "active":
                self._maybe_advance_feature(now)
                await self._maybe_retrain_active(now)

    async def _train_candidates(self) -> None:
        """Train candidate models from the experience buffer."""
        self._status.last_train_time = time.time()
        self._status.last_train_experience_count = len(self._buffer)
        self._candidates.clear()

        for arch_name in ["mlp2", "mlp3", "gru"]:
            if arch_name not in ARCHITECTURE_REGISTRY:
                continue
            try:
                from policy.state_encoder import STATE_DIM
                candidate = PolicyNNController(arch=arch_name, input_dim=STATE_DIM)
                if candidate.model is None:
                    continue

                result = self._trainer.train_imitation(candidate.model)
                if result.samples_used < 10:
                    continue

                from policy.state_encoder import ENCODER_VERSION
                arch_label = f"{arch_name}_enc{ENCODER_VERSION}"
                version = self._registry.register(
                    arch=arch_label,
                    validation_loss=result.best_loss,
                    shadow_win_rate=0.0,
                    model_saver=lambda p, m=candidate: m.save(p),
                )

                candidate._last_train_loss = result.best_loss
                candidate._registry_version = version.version
                self._candidates.append(candidate)
                logger.info("Trained candidate: %s v%d (loss=%.4f)",
                            arch_name, version.version, result.best_loss)
            except Exception as exc:
                logger.error("Training %s failed: %s", arch_name, exc)

        if self._candidates:
            best = min(self._candidates, key=lambda c: getattr(c, '_last_train_loss', float('inf')))
            self._interface.set_nn_controller(best)
            self._evaluator.set_mode("shadow")
            self._status.phase = "evaluating"
            self._status.candidate_count = len(self._candidates)
            policy_telemetry.train_enabled = True
            policy_telemetry.log_event("training_complete",
                                       f"{len(self._candidates)} candidates trained")
        else:
            self._status.phase = "collecting"
            logger.info("No candidates trained, back to collecting")

    async def _promote_best(self) -> None:
        """Promote the best candidate to active control."""
        report = self._evaluator.evaluate()

        if not report.eligible_for_control:
            self._status.phase = "evaluating"
            return

        report = self._evaluator.evaluate()
        if report.nn_decisive_win_rate > 0 and len(self._candidates) > 1:
            best = max(
                self._candidates,
                key=lambda c: getattr(c, '_shadow_win_rate', 0.0),
                default=self._candidates[0],
            )
        else:
            best = min(self._candidates, key=lambda c: getattr(c, '_last_train_loss', float('inf')))
        best_version = getattr(best, '_registry_version', None)
        if best_version:
            self._registry.promote(best_version)

        self._evaluator.set_mode("partial")
        feature = STAGED_FEATURES[self._status.current_feature_index]
        self._interface.enable_feature(feature)

        self._status.total_promotions += 1
        self._status.phase = "active"
        self._status.last_feature_advance_time = time.time()
        policy_telemetry.log_event("promotion",
                                   f"NN promoted with feature={feature}, win_rate={report.nn_win_rate:.1%}")

        logger.info(
            "NN promoted! Feature=%s, win_rate=%.1f%%, decisive_wr=%.1f%%, "
            "nn_wins=%d kernel_wins=%d ties=%d total=%d",
            feature, report.nn_win_rate * 100, report.nn_decisive_win_rate * 100,
            report.nn_wins, report.kernel_wins, report.ties, report.total_decisions,
        )

    def _should_promote_with_friction(self) -> bool:
        """Quarantine-aware promotion check: elevated raises thresholds, high blocks."""
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            friction = get_quarantine_pressure().policy_promotion_friction()
            if friction.get("block"):
                logger.info("Policy promotion blocked by quarantine pressure")
                get_quarantine_pressure().record_promotion_blocked()
                return False
        except Exception:
            friction = {}
        return self._evaluator.should_promote()

    def _check_quarantine_rollback(self) -> None:
        """If quarantine pressure is high+chronic during active phase, rollback."""
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            friction = get_quarantine_pressure().policy_promotion_friction()
            if friction.get("allow_rollback") and self._status.phase == "active":
                logger.warning("Quarantine pressure high+chronic: rolling back policy")
                self.rollback("quarantine_pressure")
        except Exception:
            pass

    def _maybe_advance_feature(self, now: float) -> None:
        """Check if the NN merits advancing to the next staged feature."""
        if self._status.current_feature_index >= len(STAGED_FEATURES) - 1:
            return
        if now - self._status.last_feature_advance_time < FEATURE_ADVANCE_INTERVAL_S:
            return
        report = self._evaluator.evaluate()
        if report.total_decisions < FEATURE_ADVANCE_MIN_DECISIONS:
            return
        if report.nn_decisive_win_rate < PROMOTION_WIN_RATE:
            return
        if report.nn_wins <= report.kernel_wins:
            return
        feature = self.advance_feature()
        if feature:
            self._status.last_feature_advance_time = now
            logger.info(
                "Feature advanced to %s after %d decisions (decisive_wr=%.1f%%)",
                feature, report.total_decisions, report.nn_decisive_win_rate * 100,
            )

    async def _maybe_retrain_active(self, now: float) -> None:
        """Periodically retrain the active NN on fresh experience data."""
        elapsed = now - self._status.last_train_time
        if elapsed < ACTIVE_RETRAIN_INTERVAL_S:
            return
        buf_size = len(self._buffer)
        new_since_last = buf_size - self._status.last_train_experience_count
        if new_since_last < ACTIVE_RETRAIN_MIN_NEW:
            return
        logger.info("Active retrain: %d new experiences since last train (%.0fs ago)",
                     new_since_last, elapsed)
        self._status.last_train_time = now
        self._status.last_train_experience_count = buf_size
        current = self._interface._nn_controller
        if current is None or current.model is None:
            return
        try:
            result = self._trainer.train_imitation(current.model)
            if result.samples_used >= 10:
                policy_telemetry.log_event("active_retrain",
                                           f"loss={result.best_loss:.4f} samples={result.samples_used}")
                logger.info("Active retrain complete: loss=%.4f samples=%d",
                            result.best_loss, result.samples_used)
        except Exception as exc:
            logger.error("Active retrain failed: %s", exc)

    def advance_feature(self) -> str | None:
        """Advance to the next staged feature. Returns the feature name or None if done."""
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            if get_quarantine_pressure().current.high:
                logger.info("Feature advance blocked by quarantine high pressure")
                return None
        except Exception:
            pass
        idx = self._status.current_feature_index + 1
        if idx >= len(STAGED_FEATURES):
            self._evaluator.set_mode("full")
            self._interface.enable()
            policy_telemetry.log_event("full_control", "All features enabled")
            return None

        self._status.current_feature_index = idx
        feature = STAGED_FEATURES[idx]
        self._interface.enable_feature(feature)
        policy_telemetry.log_event("feature_advance", f"Enabled {feature}")
        return feature

    def rollback(self, reason: str = "regression") -> None:
        """Rollback to kernel defaults."""
        self._status.total_rollbacks += 1
        self._status.phase = "collecting"
        self._status.current_feature_index = 0

        for feat in STAGED_FEATURES:
            self._interface.disable_feature(feat)
        self._interface.disable()
        self._evaluator.set_mode("shadow")
        self._governor.record_regression()

        policy_telemetry.log_event("rollback", f"Rolled back: {reason}")
        logger.warning("Policy rollback: %s", reason)

    # ------------------------------------------------------------------
    # M6: Broadcast expansion — shadow dual-encoder migration
    # ------------------------------------------------------------------

    def _resume_expansion_on_boot(self) -> None:
        """Resume shadow evaluation if it was active before brain restart."""
        try:
            from policy.shadow_runner import ShadowPolicyRunner, EXPANSION_STATE_FILE
            if not EXPANSION_STATE_FILE.exists():
                return
            runner = ShadowPolicyRunner()
            if runner.expansion_state.phase == "shadow":
                self._shadow_runner = runner
                arch_name = "mlp2"
                current = self._interface._nn_controller
                if current is not None:
                    arch_name = current.arch_name
                if runner.start_shadow(arch=arch_name):
                    runner.train_shadow_from_buffer(self._buffer)
                    logger.info(
                        "Resumed M6 shadow evaluation from boot "
                        "(decisions=%d, win_rate=%.2f%%)",
                        runner.expansion_state.shadow_decisions,
                        runner.expansion_state.shadow_win_rate * 100,
                    )
                else:
                    self._shadow_runner = None
        except Exception:
            logger.debug("No expansion state to resume on boot")

    def _subscribe_expansion_event(self) -> None:
        if self._expansion_subscribed:
            return
        try:
            from consciousness.events import MATRIX_EXPANSION_TRIGGERED, event_bus
            event_bus.on(MATRIX_EXPANSION_TRIGGERED, self._on_expansion_triggered)
            self._expansion_subscribed = True
        except Exception:
            logger.debug("Could not subscribe to expansion event (early boot)")

    def _on_expansion_triggered(self, **kwargs: Any) -> None:
        """Handle MATRIX_EXPANSION_TRIGGERED: start shadow dual-encoder evaluation."""
        if self._shadow_runner is not None:
            sr = self._shadow_runner
            if sr.expansion_state.phase in ("shadow", "promoted"):
                logger.info("Expansion already in phase=%s, ignoring re-trigger",
                            sr.expansion_state.phase)
                return

        logger.info("M6 expansion triggered — initializing shadow runner")
        try:
            from policy.shadow_runner import ShadowPolicyRunner
            self._shadow_runner = ShadowPolicyRunner()

            self._expand_hemisphere_slots()

            current_arch = self._interface._nn_controller
            arch_name = current_arch.arch_name if current_arch else "mlp2"
            if not self._shadow_runner.start_shadow(arch=arch_name):
                logger.error("Shadow runner failed to start")
                self._shadow_runner = None
                return

            if current_arch is not None and current_arch.model is not None:
                from policy.policy_nn import migrate_weights
                from policy.state_encoder import STATE_DIM, SHADOW_STATE_DIM
                migrate_weights(
                    current_arch.model,
                    self._shadow_runner._shadow_controller.model,
                    old_dim=STATE_DIM,
                    new_dim=SHADOW_STATE_DIM,
                )

            self._shadow_runner.train_shadow_from_buffer(self._buffer)

            policy_telemetry.log_event(
                "expansion_shadow_started",
                f"Shadow dual-encoder A/B started (arch={arch_name})",
            )
        except Exception:
            logger.exception("Failed to start M6 expansion shadow")
            self._shadow_runner = None

    def set_hemisphere_orchestrator(self, orchestrator: Any) -> None:
        """Wire the hemisphere orchestrator for slot expansion/contraction."""
        self._hemisphere_orchestrator = orchestrator

    def _expand_hemisphere_slots(self) -> None:
        """Tell the hemisphere orchestrator to expand broadcast slots."""
        orch = getattr(self, "_hemisphere_orchestrator", None)
        if orch is not None:
            try:
                orch.expand_broadcast_slots()
            except Exception:
                logger.debug("Could not expand hemisphere slots")

    def _contract_hemisphere_slots(self) -> None:
        """Tell the hemisphere orchestrator to contract broadcast slots back to 4."""
        orch = getattr(self, "_hemisphere_orchestrator", None)
        if orch is not None:
            try:
                orch.contract_broadcast_slots()
            except Exception:
                logger.debug("Could not contract hemisphere slots")

    def expansion_shadow_forward(self, state: dict[str, Any]) -> Any:
        """Run shadow forward pass for M6 A/B. Returns PolicyDecision or None."""
        if self._shadow_runner is None or not self._shadow_runner.active:
            return None
        return self._shadow_runner.shadow_forward(state)

    def record_expansion_outcome(
        self,
        kernel_reward: float,
        shadow_decision: Any,
        kernel_decision: Any,
    ) -> None:
        """Record A/B outcome for expansion shadow evaluation."""
        if self._shadow_runner is None or not self._shadow_runner.active:
            return
        self._shadow_runner.record_shadow_outcome(
            kernel_reward, shadow_decision, kernel_decision,
        )

    async def _tick_expansion(self) -> None:
        """Check expansion shadow A/B status and promote/rollback if ready."""
        if self._shadow_runner is None:
            return
        sr = self._shadow_runner
        if sr.expansion_state.phase != "shadow":
            return

        if sr.should_promote():
            await self._promote_expansion()
        elif sr.should_rollback():
            self._rollback_expansion()

    async def _promote_expansion(self) -> None:
        """Promote the 22-dim encoding to live, replacing the 20-dim encoder."""
        sr = self._shadow_runner
        if sr is None:
            return

        logger.info("M6 expansion: promoting 22-dim encoding to live")

        try:
            from policy.state_encoder import ShadowStateEncoder, SHADOW_STATE_DIM

            new_encoder = ShadowStateEncoder()
            new_encoder.set_hemisphere_signals(
                sr._shadow_encoder._hemisphere_signals
                if sr._shadow_encoder else {}
            )

            current_controller = self._interface._nn_controller
            arch_name = current_controller.arch_name if current_controller else "mlp2"
            new_controller = PolicyNNController(
                arch=arch_name, input_dim=SHADOW_STATE_DIM,
            )

            if (sr._shadow_controller is not None
                    and sr._shadow_controller.model is not None
                    and new_controller.model is not None):
                new_controller.model.load_state_dict(
                    sr._shadow_controller.model.state_dict()
                )

            new_controller.set_encoder(new_encoder)
            self._interface.set_nn_controller(new_controller)

            sr.mark_promoted()

            policy_telemetry.log_event(
                "expansion_promoted",
                f"22-dim encoding promoted to live (win_rate={sr.expansion_state.shadow_win_rate:.1%})",
            )
            logger.info(
                "M6 expansion promoted: 22-dim encoding now live "
                "(decisions=%d, win_rate=%.1f%%)",
                sr.expansion_state.shadow_decisions,
                sr.expansion_state.shadow_win_rate * 100,
            )
        except Exception:
            logger.exception("M6 expansion promotion failed")
            self._rollback_expansion()

    def _rollback_expansion(self) -> None:
        """Rollback M6 expansion — revert to 20-dim encoding."""
        sr = self._shadow_runner
        if sr is None:
            return

        logger.info("M6 expansion: rolling back to 20-dim encoding")
        sr.mark_rolled_back()
        self._contract_hemisphere_slots()

        policy_telemetry.log_event(
            "expansion_rolled_back",
            f"22-dim encoding rolled back (decisions={sr.expansion_state.shadow_decisions}, "
            f"win_rate={sr.expansion_state.shadow_win_rate:.1%})",
        )

    def get_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "phase": self._status.phase,
            "total_promotions": self._status.total_promotions,
            "total_rollbacks": self._status.total_rollbacks,
            "candidates_count": len(self._candidates),
            "buffer_size": len(self._buffer),
            "current_feature": STAGED_FEATURES[self._status.current_feature_index] if self._status.current_feature_index < len(STAGED_FEATURES) else "all",
            "evaluator": self._evaluator.get_status(),
            "registry": self._registry.get_status(),
        }
        if self._shadow_runner is not None:
            status["expansion"] = self._shadow_runner.get_status()
        return status
