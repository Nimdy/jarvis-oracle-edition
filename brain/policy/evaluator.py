"""Evaluator — shadow A/B comparison between kernel and NN decisions.

Runs NN in parallel (no apply), scores "would NN have improved outcome?"
using retrospective outcome measurement. The NN "wins" by proposing
meaningfully different configs that correlate with better outcomes.

Key design:
  - Scoring is outcome-based: NN gets credit when its divergent proposal
    coincides with good outcomes.
  - Noise-aware noop detection: deviations within exploration sigma are
    classified as noops (the NN didn't really propose something different).
  - Minimum margin: outcomes within MIN_OUTCOME_MARGIN are ties, preventing
    systematic false-credit from micro-wins (exploration noise + stable system).
  - Deviation bonus only for genuinely different proposals (above noise floor).
  - Promotion requires real decisive wins, not just volume.

All shadow results flow into PolicyTelemetry (counters only on hot path).
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SHADOW_WINDOW_SIZE = 500
WIN_THRESHOLD_PCT = 0.55
MIN_SHADOW_DECISIONS = 100
TIE_MARGIN = 0.03
MIN_OUTCOME_MARGIN = 0.01
NOOP_PENALTY = -0.01
DEVIATION_BONUS = 0.08
NOISE_NOOP_SIGMA_MULT = 2.5
EXPLORATION_SIGMA = 0.03
EMA_ALPHA = 0.1
_DIVERSITY_BONUS_FRAC = 0.3


@dataclass
class ShadowResult:
    timestamp: float
    kernel_reward: float
    nn_reward: float
    nn_won: bool
    margin: float = 0.0
    classification: str = "tie"
    is_noop: bool = False


@dataclass
class EvaluationReport:
    total_decisions: int = 0
    nn_wins: int = 0
    kernel_wins: int = 0
    ties: int = 0
    nn_win_rate: float = 0.0
    nn_decisive_win_rate: float = 0.0
    eligible_for_control: bool = False
    avg_nn_reward: float = 0.0
    avg_kernel_reward: float = 0.0
    avg_win_margin: float = 0.0
    noop_rate: float = 0.0


@dataclass
class PendingShadow:
    """An NN proposal awaiting retrospective outcome scoring."""
    timestamp: float
    nn_proposed: dict
    kernel_actual: dict


class PolicyEvaluator:
    def __init__(self) -> None:
        self._shadow_results: deque[ShadowResult] = deque(maxlen=SHADOW_WINDOW_SIZE)
        self._mode: str = "shadow"
        self._pending_shadow: PendingShadow | None = None
        self._win_margin_ema: float = 0.0
        self._noop_count: int = 0
        self._prev_reward: float | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def record_shadow(
        self,
        kernel_reward: float,
        nn_reward: float,
        nn_is_noop: bool = False,
    ) -> ShadowResult:
        from policy.telemetry import policy_telemetry

        if nn_is_noop:
            nn_reward += NOOP_PENALTY
            self._noop_count += 1

        margin = nn_reward - kernel_reward

        if abs(margin) < MIN_OUTCOME_MARGIN:
            classification = "tie"
            nn_won = False
        elif margin > TIE_MARGIN:
            classification = "nn_win"
            nn_won = True
        elif margin < -TIE_MARGIN:
            classification = "kernel_win"
            nn_won = False
        else:
            classification = "tie"
            nn_won = False

        result = ShadowResult(
            timestamp=time.time(),
            kernel_reward=kernel_reward,
            nn_reward=nn_reward,
            nn_won=nn_won,
            margin=margin,
            classification=classification,
            is_noop=nn_is_noop,
        )
        self._shadow_results.append(result)

        self._win_margin_ema = (
            margin if self._win_margin_ema == 0.0
            else EMA_ALPHA * margin + (1.0 - EMA_ALPHA) * self._win_margin_ema
        )

        if classification == "tie":
            policy_telemetry.record_shadow_tie()
        else:
            policy_telemetry.record_shadow(nn_won)

        policy_telemetry.record_win_margin(margin)

        return result

    def evaluate(self) -> EvaluationReport:
        results = list(self._shadow_results)
        total = len(results)
        if total == 0:
            return EvaluationReport()

        nn_wins = sum(1 for r in results if r.classification == "nn_win")
        kernel_wins = sum(1 for r in results if r.classification == "kernel_win")
        ties = sum(1 for r in results if r.classification == "tie")

        win_rate = nn_wins / total
        decisive = nn_wins + kernel_wins
        decisive_win_rate = nn_wins / decisive if decisive > 0 else 0.0

        avg_nn = sum(r.nn_reward for r in results) / total
        avg_kernel = sum(r.kernel_reward for r in results) / total
        margins = [r.margin for r in results if r.classification == "nn_win"]
        avg_margin = sum(margins) / len(margins) if margins else 0.0

        windowed_noops = sum(1 for r in results if r.is_noop)
        noop_rate = windowed_noops / total if total > 0 else 0.0

        eligible = (
            total >= MIN_SHADOW_DECISIONS
            and decisive_win_rate > WIN_THRESHOLD_PCT
            and decisive >= max(30, int(total * 0.15))
            and avg_margin > TIE_MARGIN
            and nn_wins > kernel_wins
        )

        return EvaluationReport(
            total_decisions=total,
            nn_wins=nn_wins,
            kernel_wins=kernel_wins,
            ties=ties,
            nn_win_rate=win_rate,
            nn_decisive_win_rate=decisive_win_rate,
            eligible_for_control=eligible,
            avg_nn_reward=avg_nn,
            avg_kernel_reward=avg_kernel,
            avg_win_margin=avg_margin,
            noop_rate=noop_rate,
        )

    def should_promote(self) -> bool:
        report = self.evaluate()
        return report.eligible_for_control

    def record_pending_shadow(self, nn_proposed: dict, kernel_actual: dict) -> None:
        """Record the NN's proposal alongside the kernel's actual values
        for later retrospective scoring when conversation outcome arrives."""
        self._pending_shadow = PendingShadow(
            timestamp=time.time(),
            nn_proposed=nn_proposed,
            kernel_actual=kernel_actual,
        )

    def score_retrospective(self, actual_reward: float) -> None:
        """Score the pending shadow using reward level + improvement signal.

        Scoring logic:
        - Base: both kernel and NN start at actual_reward.
        - If NN proposed something meaningfully different (not noise-level noop):
          * reward improved (delta > 0.02): NN gets full deviation bonus
          * reward degraded (delta < -0.02): kernel gets credit
          * reward stable AND system healthy (reward > 0.7): NN gets a
            proportional diversity bonus so it can earn wins even during
            stable operation. This is critical for learning — a stable system
            should still explore, and novel proposals that don't hurt should
            be rewarded.
          * reward stable AND system struggling (reward <= 0.7): smaller
            diversity bonus (NN should prove itself more when things aren't great)
        - No-ops (including noise-level deviations) are scored as exact ties.
        """
        pending = self._pending_shadow
        if pending is None:
            self._prev_reward = actual_reward
            return
        self._pending_shadow = None

        if time.time() - pending.timestamp > 120.0:
            self._prev_reward = actual_reward
            return

        nn_is_noop = self._is_noop(pending.nn_proposed, pending.kernel_actual)

        nn_reward = actual_reward
        kernel_reward = actual_reward

        if nn_is_noop:
            self.record_shadow(kernel_reward, nn_reward, nn_is_noop=True)
            self._prev_reward = actual_reward
            return

        deviation_magnitude = self._compute_deviation(pending.nn_proposed, pending.kernel_actual)
        scaled_bonus = DEVIATION_BONUS * min(1.0, deviation_magnitude / 0.15)

        reward_delta = (actual_reward - self._prev_reward) if self._prev_reward is not None else 0.0

        if reward_delta > 0.02:
            nn_reward += scaled_bonus
        elif reward_delta < -0.02:
            kernel_reward += scaled_bonus * 0.5
        elif actual_reward > 0.7:
            nn_reward += scaled_bonus * 0.5
        elif actual_reward > 0.1:
            nn_reward += scaled_bonus * _DIVERSITY_BONUS_FRAC

        self.record_shadow(kernel_reward, nn_reward, nn_is_noop=False)
        self._prev_reward = actual_reward

    @staticmethod
    def _compute_deviation(nn_proposed: dict, kernel_actual: dict) -> float:
        """Compute the magnitude of deviation between NN and kernel proposals."""
        diffs: list[float] = []
        budget_diff = abs((nn_proposed.get("budget_ms") or 16) - (kernel_actual.get("budget_ms") or 16))
        diffs.append(budget_diff / 16.0)

        tw_nn = nn_proposed.get("thought_weights_delta", {})
        tw_k = kernel_actual.get("thought_weights_delta", {})
        for k in set(tw_nn) | set(tw_k):
            diffs.append(abs(tw_nn.get(k, 0) - tw_k.get(k, 0)))

        if nn_proposed.get("suggested_mode", "") != kernel_actual.get("suggested_mode", ""):
            diffs.append(0.2)
        if nn_proposed.get("response_length_hint", "") != kernel_actual.get("response_length_hint", ""):
            diffs.append(0.15)

        return math.sqrt(sum(d * d for d in diffs)) if diffs else 0.0

    @staticmethod
    def _is_noop(nn_proposed: dict, kernel_actual: dict) -> bool:
        """Detect if the NN proposed essentially the same thing as the kernel.

        Uses a noise-aware threshold: deviations within NOISE_NOOP_SIGMA_MULT * sigma
        are treated as exploration noise, not real proposals. This prevents the NN
        from getting unearned credit for random jitter.
        """
        noise_thresh = NOISE_NOOP_SIGMA_MULT * EXPLORATION_SIGMA

        budget_diff = abs((nn_proposed.get("budget_ms") or 16) - (kernel_actual.get("budget_ms") or 16))
        if budget_diff > 3:
            return False
        if nn_proposed.get("suggested_mode", "") != kernel_actual.get("suggested_mode", ""):
            return False
        if nn_proposed.get("response_length_hint", "") != kernel_actual.get("response_length_hint", ""):
            return False

        tw_nn = nn_proposed.get("thought_weights_delta", {})
        tw_k = kernel_actual.get("thought_weights_delta", {})
        all_keys = set(tw_nn) | set(tw_k)
        if all_keys:
            diffs = [abs(tw_nn.get(k, 0) - tw_k.get(k, 0)) for k in all_keys]
            max_diff = max(diffs) if diffs else 0.0
            l2_diff = math.sqrt(sum(d * d for d in diffs))
            if max_diff > noise_thresh or l2_diff > noise_thresh * 2:
                return False

        return True

    def update_telemetry(self) -> None:
        """Push current eval stats to telemetry (call infrequently, e.g. every 30s)."""
        from policy.telemetry import policy_telemetry

        report = self.evaluate()
        policy_telemetry.update_eval(
            score=report.nn_win_rate,
            eligible=report.eligible_for_control,
            candidates=0,
        )
        policy_telemetry.nn_decisive_win_rate = report.nn_decisive_win_rate
        policy_telemetry.noop_count = self._noop_count
        policy_telemetry._windowed_nn_win_rate = report.nn_win_rate
        policy_telemetry.record_win_rate_snapshot(win_rate_override=report.nn_win_rate)

    def set_mode(self, mode: str) -> None:
        if mode in ("shadow", "partial", "full"):
            old = self._mode
            self._mode = mode
            if old != mode:
                from policy.telemetry import policy_telemetry
                policy_telemetry.mode = mode
                policy_telemetry.log_event("mode_change", f"{old} → {mode}")
            logger.info("Policy evaluator mode: %s", mode)

    def get_status(self) -> dict[str, Any]:
        report = self.evaluate()
        return {
            "mode": self._mode,
            "total_decisions": report.total_decisions,
            "nn_win_rate": round(report.nn_win_rate, 3),
            "nn_decisive_win_rate": round(report.nn_decisive_win_rate, 3),
            "eligible_for_control": report.eligible_for_control,
            "avg_win_margin": round(report.avg_win_margin, 4),
            "noop_rate": round(report.noop_rate, 3),
            "win_margin_ema": round(self._win_margin_ema, 4),
            "tie_margin_threshold": TIE_MARGIN,
        }
