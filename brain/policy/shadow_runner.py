"""Shadow Policy Runner — dual-encoder A/B evaluation for M6 expansion.

Runs a shadow PolicyNNController with SHADOW_STATE_DIM (22) input alongside
the live 20-dim controller. Collects parallel decisions and feeds them into
a dedicated PolicyEvaluator for shadow A/B comparison. The live policy NN
continues driving actual decisions throughout.

Persistence: expansion state saved to ~/.jarvis/expansion_state.json so
shadow evaluation resumes across brain restarts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
EXPANSION_STATE_FILE = JARVIS_DIR / "expansion_state.json"

SHADOW_MIN_DECISIONS = 100
SHADOW_PROMOTE_WIN_RATE = 0.55
SHADOW_ROLLBACK_WIN_RATE = 0.45
SHADOW_ROLLBACK_DECISIONS_MULT = 2


@dataclass
class ExpansionState:
    triggered: bool = False
    triggered_at: float = 0.0
    phase: str = "inactive"   # inactive / shadow / promoted / rolled_back
    shadow_decisions: int = 0
    shadow_win_rate: float = 0.0
    promoted_at: float = 0.0
    rolled_back_at: float = 0.0


class ShadowPolicyRunner:
    """Parallel shadow NN with expanded state encoding for A/B evaluation."""

    def __init__(self) -> None:
        self._state = ExpansionState()
        self._shadow_controller: Any = None
        self._shadow_encoder: Any = None
        self._shadow_evaluator: Any = None
        self._active = False
        self._load_state()

    @property
    def expansion_state(self) -> ExpansionState:
        return self._state

    @property
    def active(self) -> bool:
        return self._active and self._state.phase == "shadow"

    def start_shadow(self, arch: str = "mlp2") -> bool:
        """Initialize shadow NN + encoder and begin A/B collection.

        Called when MATRIX_EXPANSION_TRIGGERED fires. Returns False if
        PyTorch is unavailable or shadow is already running.
        """
        if self._active:
            logger.info("Shadow runner already active, skipping start")
            return False

        try:
            from policy.state_encoder import ShadowStateEncoder, SHADOW_STATE_DIM
            from policy.policy_nn import PolicyNNController, TORCH_AVAILABLE
            from policy.evaluator import PolicyEvaluator

            if not TORCH_AVAILABLE:
                logger.warning("Shadow runner: PyTorch unavailable, cannot start")
                return False

            self._shadow_encoder = ShadowStateEncoder()
            self._shadow_controller = PolicyNNController(
                arch=arch, input_dim=SHADOW_STATE_DIM,
            )
            self._shadow_controller.set_encoder(self._shadow_encoder)
            self._shadow_evaluator = PolicyEvaluator()

            self._state.phase = "shadow"
            if self._state.triggered_at == 0.0:
                self._state.triggered = True
                self._state.triggered_at = time.time()
            self._active = True
            self._save_state()

            logger.info(
                "Shadow runner started: arch=%s, dim=%d",
                arch, SHADOW_STATE_DIM,
            )
            return True

        except Exception:
            logger.exception("Failed to start shadow runner")
            return False

    def train_shadow_from_buffer(self, buffer: Any) -> bool:
        """Train the shadow NN from the existing experience buffer.

        Re-encodes experiences with the 22-dim shadow encoder (zero-padding
        dims 20-21 for historical data).
        """
        if not self._active or self._shadow_controller is None:
            return False

        try:
            from policy.trainer import PolicyTrainer
            trainer = PolicyTrainer(buffer)
            result = trainer.train_imitation(self._shadow_controller.model)
            if result.samples_used >= 10:
                logger.info(
                    "Shadow NN trained: loss=%.4f, samples=%d",
                    result.best_loss, result.samples_used,
                )
                return True
            logger.info("Shadow NN training: insufficient samples (%d)", result.samples_used)
            return False
        except Exception:
            logger.exception("Shadow NN training failed")
            return False

    def shadow_forward(self, state: dict[str, Any]) -> Any:
        """Run shadow NN forward pass. Returns PolicyDecision or None."""
        if not self._active or self._shadow_controller is None:
            return None
        try:
            return self._shadow_controller.forward(state)
        except Exception:
            logger.warning("Shadow forward pass failed", exc_info=True)
            return None

    def set_hemisphere_signals(self, signals: dict[str, float]) -> None:
        """Forward hemisphere broadcast signals into the shadow encoder.

        Mirror of `StateEncoder.set_hemisphere_signals` called by the engine
        tick handler. Without this, dims 16-21 of `ShadowStateEncoder` stay
        at 0.0 for the entire A/B window and the shadow NN cannot learn
        whether M6 expanded slots (slot_4, slot_5) help the policy.
        """
        if not self._active or self._shadow_encoder is None:
            return
        try:
            if hasattr(self._shadow_encoder, "set_hemisphere_signals"):
                self._shadow_encoder.set_hemisphere_signals(signals)
        except Exception:
            logger.warning(
                "Shadow encoder set_hemisphere_signals failed", exc_info=True,
            )

    def record_shadow_outcome(
        self,
        kernel_reward: float,
        shadow_decision: Any,
        kernel_decision: Any,
    ) -> None:
        """Record A/B outcome between live (20-dim) and shadow (22-dim)."""
        if not self._active or self._shadow_evaluator is None:
            return

        try:
            if shadow_decision is not None and kernel_decision is not None:
                nn_proposed = {
                    "thought_weights_delta": shadow_decision.thought_weights_delta,
                    "budget_ms": shadow_decision.budget_ms,
                    "suggested_mode": shadow_decision.suggested_mode,
                    "response_length_hint": shadow_decision.response_length_hint,
                }
                kernel_actual = {
                    "thought_weights_delta": kernel_decision.thought_weights_delta,
                    "budget_ms": kernel_decision.budget_ms,
                    "suggested_mode": kernel_decision.suggested_mode,
                    "response_length_hint": kernel_decision.response_length_hint,
                }
                from policy.evaluator import PendingShadow
                self._shadow_evaluator._pending_shadow = PendingShadow(
                    nn_proposed=nn_proposed,
                    kernel_actual=kernel_actual,
                    timestamp=time.time(),
                )
                self._shadow_evaluator.score_retrospective(kernel_reward)
            else:
                self._shadow_evaluator.record_shadow(
                    kernel_reward, kernel_reward, nn_is_noop=True,
                )

            report = self._shadow_evaluator.evaluate()
            self._state.shadow_decisions = report.total_decisions
            self._state.shadow_win_rate = report.nn_win_rate

            if report.total_decisions % 25 == 0 and report.total_decisions > 0:
                self._save_state()
                logger.info(
                    "Shadow A/B: %d decisions, win_rate=%.2f%%, decisive_wr=%.2f%%",
                    report.total_decisions,
                    report.nn_win_rate * 100,
                    report.nn_decisive_win_rate * 100,
                )
        except Exception:
            logger.exception("Shadow outcome recording failed")

    def should_promote(self) -> bool:
        """Check if shadow encoding should be promoted to live."""
        if not self._active or self._shadow_evaluator is None:
            return False
        report = self._shadow_evaluator.evaluate()
        return (
            report.total_decisions >= SHADOW_MIN_DECISIONS
            and report.nn_win_rate >= SHADOW_PROMOTE_WIN_RATE
            and report.nn_wins > report.kernel_wins
        )

    def should_rollback(self) -> bool:
        """Check if shadow evaluation should be abandoned."""
        if not self._active or self._shadow_evaluator is None:
            return False
        report = self._shadow_evaluator.evaluate()
        return (
            report.total_decisions >= SHADOW_MIN_DECISIONS * SHADOW_ROLLBACK_DECISIONS_MULT
            and report.nn_win_rate < SHADOW_ROLLBACK_WIN_RATE
        )

    def mark_promoted(self) -> None:
        """Mark expansion as promoted — live encoding switches to 22-dim."""
        self._state.phase = "promoted"
        self._state.promoted_at = time.time()
        self._active = False
        self._save_state()
        logger.info("M6 expansion promoted to live")

    def mark_rolled_back(self) -> None:
        """Mark expansion as rolled back — revert to 20-dim."""
        self._state.phase = "rolled_back"
        self._state.rolled_back_at = time.time()
        self._active = False
        self._shadow_controller = None
        self._shadow_encoder = None
        self._shadow_evaluator = None
        self._save_state()
        logger.info("M6 expansion rolled back")

    def get_status(self) -> dict[str, Any]:
        report_data: dict[str, Any] = {}
        if self._shadow_evaluator is not None:
            try:
                report = self._shadow_evaluator.evaluate()
                report_data = {
                    "total_decisions": report.total_decisions,
                    "nn_win_rate": round(report.nn_win_rate, 3),
                    "nn_decisive_win_rate": round(report.nn_decisive_win_rate, 3),
                    "eligible": report.eligible_for_control,
                }
            except Exception:
                pass

        return {
            "phase": self._state.phase,
            "triggered": self._state.triggered,
            "triggered_at": self._state.triggered_at,
            "shadow_decisions": self._state.shadow_decisions,
            "shadow_win_rate": round(self._state.shadow_win_rate, 3),
            "promoted_at": self._state.promoted_at,
            "rolled_back_at": self._state.rolled_back_at,
            "active": self._active,
            "shadow_eval": report_data,
        }

    # -- persistence -----------------------------------------------------------

    def _save_state(self) -> None:
        try:
            from memory.persistence import atomic_write_json
            data = {
                "triggered": self._state.triggered,
                "triggered_at": self._state.triggered_at,
                "phase": self._state.phase,
                "shadow_decisions": self._state.shadow_decisions,
                "shadow_win_rate": self._state.shadow_win_rate,
                "promoted_at": self._state.promoted_at,
                "rolled_back_at": self._state.rolled_back_at,
            }
            atomic_write_json(EXPANSION_STATE_FILE, data, indent=2)
        except Exception:
            logger.exception("Failed to save expansion state")

    def _load_state(self) -> None:
        if not EXPANSION_STATE_FILE.exists():
            return
        try:
            data = json.loads(EXPANSION_STATE_FILE.read_text())
            self._state.triggered = data.get("triggered", False)
            self._state.triggered_at = data.get("triggered_at", 0.0)
            self._state.phase = data.get("phase", "inactive")
            self._state.shadow_decisions = data.get("shadow_decisions", 0)
            self._state.shadow_win_rate = data.get("shadow_win_rate", 0.0)
            self._state.promoted_at = data.get("promoted_at", 0.0)
            self._state.rolled_back_at = data.get("rolled_back_at", 0.0)
            logger.info(
                "Loaded expansion state: phase=%s, decisions=%d",
                self._state.phase, self._state.shadow_decisions,
            )
        except Exception:
            logger.exception("Failed to load expansion state")
