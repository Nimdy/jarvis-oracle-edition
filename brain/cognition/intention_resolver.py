"""IntentionResolver — Stage 1 relevance predictor for resolved intentions.

Shadow-first, governance-gated proactive delivery candidate generator.

Contract (mirrors PolicyNN shadow protocol):
  - Read-only against WorldModel and IntentionRegistry.
  - NEVER calls TTS or emits CONVERSATION_RESPONSE.
  - Emits candidates through ProactiveGovernor ONLY after its shadow
    accuracy gate clears (see promotion path).
  - All verdicts are logged to an append-only JSONL so a future
    `intention_delivery` hemisphere specialist can train on them.

Design frozen at: docs/INTENTION_STAGE_1_DESIGN.md
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
VERDICTS_PATH = JARVIS_DIR / "intention_resolver_verdicts.jsonl"
_VERDICTS_MAX_FILE_MB = 10


# ---------------------------------------------------------------------------
# Controlled vocabulary — reason codes (from design doc §3)
# ---------------------------------------------------------------------------

ResolverDecision = Literal[
    "deliver_now",
    "deliver_on_next_turn",
    "suppress",
    "defer",
]

REASON_CODES = frozenset({
    "fresh_actionable_result",
    "fresh_speaker_gone",
    "stale_low_relevance",
    "failed_result_informational",
    "failed_result_noisy",
    "low_confidence_suppress",
    "governance_blocked",
    "conversation_inactive_wait",
    "cooldown_defer",
    "duplicate_of_earlier_delivery",
})


# ---------------------------------------------------------------------------
# Promotion stages (design doc §4)
# ---------------------------------------------------------------------------

ResolverStage = Literal[
    "shadow_only",
    "shadow_advisory",
    "advisory_canary",
    "advisory",
    "active",
]

STAGE_ORDER: list[str] = [
    "shadow_only",
    "shadow_advisory",
    "advisory_canary",
    "advisory",
    "active",
]


# ---------------------------------------------------------------------------
# Frozen dataclasses (design doc §1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolverSignal:
    """Input to the resolver: one resolved intention + world context."""

    intention_id: str
    backing_job_id: str
    commitment_type: str
    outcome: str
    age_s: float
    result_summary: str = ""
    speaker_present: bool = False
    active_conversation: bool = False
    topic_overlap: float = 0.0
    quarantine_pressure: float = 0.0
    soul_integrity: float = 1.0
    proactive_cooldown_remaining: float = 0.0
    friction_rate: float = 0.0
    same_speaker_present: bool = False


@dataclass(frozen=True)
class ResolverVerdict:
    """Output from the resolver: a decision + reasoning."""

    intention_id: str
    decision: str  # ResolverDecision
    score: float
    reason_code: str
    candidate_text: str | None = None

    def __post_init__(self) -> None:
        if self.reason_code not in REASON_CODES:
            raise ValueError(
                f"Unknown reason_code {self.reason_code!r}; "
                f"must be one of {sorted(REASON_CODES)}"
            )


# ---------------------------------------------------------------------------
# IntentionResolver
# ---------------------------------------------------------------------------

_STALE_THRESHOLD_S = 7200.0  # 2 hours
_FRESH_THRESHOLD_S = 300.0   # 5 minutes
_MEDIUM_THRESHOLD_S = 1800.0 # 30 minutes


class IntentionResolver:
    """Heuristic relevance predictor for resolved intentions.

    Stage 1 uses hand-written rules. A future Stage 2 replaces these with
    a trained NN via the intention_delivery hemisphere specialist.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stage: str = "shadow_only"
        self._total_evaluated = 0
        self._verdict_counts: dict[str, int] = {
            "deliver_now": 0,
            "deliver_on_next_turn": 0,
            "suppress": 0,
            "defer": 0,
        }
        self._reason_counts: dict[str, int] = {code: 0 for code in REASON_CODES}
        self._delivered_ids: deque[str] = deque(maxlen=200)
        self._recent_verdicts: deque[dict[str, Any]] = deque(maxlen=50)
        self._shadow_correct = 0
        self._shadow_total = 0
        self._created_at = time.time()
        JARVIS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evaluate(self, signal: ResolverSignal) -> ResolverVerdict:
        """Evaluate a resolved intention and produce a delivery verdict."""
        verdict = self._heuristic_evaluate(signal)

        with self._lock:
            self._total_evaluated += 1
            self._verdict_counts[verdict.decision] = (
                self._verdict_counts.get(verdict.decision, 0) + 1
            )
            self._reason_counts[verdict.reason_code] = (
                self._reason_counts.get(verdict.reason_code, 0) + 1
            )
            if verdict.decision != "suppress":
                self._delivered_ids.append(signal.intention_id)
            self._recent_verdicts.append({
                "intention_id": verdict.intention_id,
                "decision": verdict.decision,
                "score": verdict.score,
                "reason_code": verdict.reason_code,
                "ts": time.time(),
            })

        self._log_verdict(signal, verdict)
        return verdict

    def can_deliver(self) -> bool:
        """Whether the resolver's current stage allows actual delivery."""
        return self._stage in ("advisory_canary", "advisory", "active")

    def get_stage(self) -> str:
        return self._stage

    def set_stage(self, stage: str) -> bool:
        """Manual stage change (operator-initiated via API)."""
        if stage not in STAGE_ORDER:
            return False
        with self._lock:
            self._stage = stage
        logger.info("IntentionResolver stage set to %s", stage)
        return True

    def rollback(self) -> str:
        """Demote one rung in the promotion ladder. Returns new stage."""
        with self._lock:
            idx = STAGE_ORDER.index(self._stage)
            if idx > 0:
                self._stage = STAGE_ORDER[idx - 1]
            logger.info("IntentionResolver rolled back to %s", self._stage)
            return self._stage

    def record_shadow_outcome(self, correct: bool) -> None:
        """Record whether a shadow verdict was retroactively correct."""
        with self._lock:
            self._shadow_total += 1
            if correct:
                self._shadow_correct += 1

    # ------------------------------------------------------------------
    # Status / metrics
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stage": self._stage,
                "total_evaluated": self._total_evaluated,
                "verdict_counts": dict(self._verdict_counts),
                "reason_counts": {k: v for k, v in self._reason_counts.items() if v > 0},
                "delivered_count": len(self._delivered_ids),
                "recent_verdicts": list(self._recent_verdicts),
                "shadow_metrics": self._shadow_metrics_locked(),
                "uptime_s": round(time.time() - self._created_at, 1),
            }

    def get_shadow_metrics(self) -> dict[str, Any]:
        with self._lock:
            return self._shadow_metrics_locked()

    def _shadow_metrics_locked(self) -> dict[str, Any]:
        accuracy = 0.0
        if self._shadow_total >= 10:
            accuracy = self._shadow_correct / self._shadow_total
        return {
            "shadow_correct": self._shadow_correct,
            "shadow_total": self._shadow_total,
            "shadow_accuracy": round(accuracy, 4),
            "sufficient_data": self._shadow_total >= 50,
        }

    # ------------------------------------------------------------------
    # Heuristic logic (Stage 1 — no NN)
    # ------------------------------------------------------------------

    def _heuristic_evaluate(self, s: ResolverSignal) -> ResolverVerdict:
        if s.intention_id in self._delivered_ids:
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="suppress",
                score=0.0,
                reason_code="duplicate_of_earlier_delivery",
            )

        if s.quarantine_pressure > 0.6 or s.soul_integrity < 0.50:
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="defer",
                score=0.2,
                reason_code="governance_blocked",
            )

        if s.proactive_cooldown_remaining > 0:
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="defer",
                score=0.3,
                reason_code="cooldown_defer",
            )

        if s.age_s > _STALE_THRESHOLD_S:
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="suppress",
                score=0.1,
                reason_code="stale_low_relevance",
            )

        if s.outcome == "failed":
            if s.friction_rate > 0.15:
                return ResolverVerdict(
                    intention_id=s.intention_id,
                    decision="suppress",
                    score=0.15,
                    reason_code="failed_result_noisy",
                )
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="deliver_on_next_turn",
                score=0.5,
                reason_code="failed_result_informational",
            )

        score = self._compute_relevance_score(s)

        if score < 0.3:
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="suppress",
                score=score,
                reason_code="low_confidence_suppress",
            )

        if s.age_s <= _FRESH_THRESHOLD_S and s.speaker_present and s.active_conversation:
            if s.topic_overlap > 0.3 or s.same_speaker_present:
                return ResolverVerdict(
                    intention_id=s.intention_id,
                    decision="deliver_now",
                    score=score,
                    reason_code="fresh_actionable_result",
                    candidate_text=self._build_candidate_text(s),
                )

        if not s.speaker_present:
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="deliver_on_next_turn",
                score=score,
                reason_code="fresh_speaker_gone",
            )

        if not s.active_conversation:
            return ResolverVerdict(
                intention_id=s.intention_id,
                decision="deliver_on_next_turn",
                score=score,
                reason_code="conversation_inactive_wait",
            )

        return ResolverVerdict(
            intention_id=s.intention_id,
            decision="deliver_on_next_turn",
            score=score,
            reason_code="fresh_speaker_gone",
        )

    def _compute_relevance_score(self, s: ResolverSignal) -> float:
        """Compute a 0..1 relevance score from signal features."""
        score = 0.5

        import math
        age_factor = 1.0 - min(1.0, math.log1p(s.age_s) / math.log1p(86400))
        score += 0.2 * age_factor

        if s.topic_overlap > 0.3:
            score += 0.15

        if s.speaker_present:
            score += 0.1
        if s.active_conversation:
            score += 0.05
        if s.same_speaker_present:
            score += 0.05

        score -= 0.1 * s.friction_rate
        score -= 0.1 * s.quarantine_pressure

        return max(0.0, min(1.0, score))

    def _build_candidate_text(self, s: ResolverSignal) -> str:
        """Build a minimal delivery candidate from the resolution summary."""
        summary = (s.result_summary or "").strip()
        if not summary:
            return ""
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary

    # ------------------------------------------------------------------
    # JSONL logging
    # ------------------------------------------------------------------

    def _log_verdict(self, signal: ResolverSignal, verdict: ResolverVerdict) -> None:
        try:
            self._maybe_rotate_verdicts()
            entry = {
                "type": "resolver_verdict",
                "ts": time.time(),
                "stage": self._stage,
                "signal": asdict(signal),
                "verdict": asdict(verdict),
            }
            with open(VERDICTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":"), default=str) + "\n")
        except Exception as exc:
            logger.warning("IntentionResolver verdict log failed: %s", exc)

    def _maybe_rotate_verdicts(self) -> None:
        try:
            if not VERDICTS_PATH.exists():
                return
            size = VERDICTS_PATH.stat().st_size
            if size < _VERDICTS_MAX_FILE_MB * 1024 * 1024:
                return
            with open(VERDICTS_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            half = len(lines) // 2
            with open(VERDICTS_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines[half:])
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_resolver_instance: IntentionResolver | None = None
_resolver_lock = threading.Lock()


def get_intention_resolver() -> IntentionResolver:
    global _resolver_instance
    if _resolver_instance is None:
        with _resolver_lock:
            if _resolver_instance is None:
                _resolver_instance = IntentionResolver()
    return _resolver_instance


__all__ = [
    "ResolverDecision",
    "ResolverSignal",
    "ResolverVerdict",
    "IntentionResolver",
    "get_intention_resolver",
    "REASON_CODES",
    "STAGE_ORDER",
    "VERDICTS_PATH",
]
