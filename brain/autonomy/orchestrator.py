"""Autonomy Orchestrator — manages the research queue and coordinates all components.

The queue is explicit and inspectable:
  - Each item has: reason, tool route, budgets, status, result summary + provenance
  - The dashboard "Autonomy Feed" panel reads from get_status()
  - Priority queue uses OpportunityScorer composite scores, not raw priority
  - One research job runs at a time (governor-enforced)
  - DeltaTracker records before/after metrics for credit assignment
  - MetricTriggers inject research intents from sustained system deficits
  - Autonomy levels gate what actions are allowed:
      L0 propose  — generate intents, show in dashboard, never execute
      L1 research — execute research (web/codebase/memory), never apply code
      L2 safe-apply — auto-apply docs/tests/dashboard patches only
      L3 full     — with escalation approval for new capabilities

Called from the consciousness engine's tick cycle in non-conversational modes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from consciousness.events import (
    event_bus,
    AUTONOMY_INTENT_QUEUED,
    AUTONOMY_INTENT_BLOCKED,
    AUTONOMY_RESEARCH_STARTED,
    AUTONOMY_RESEARCH_COMPLETED,
    AUTONOMY_RESEARCH_FAILED,
    AUTONOMY_LEVEL_CHANGED,
    AUTONOMY_DELTA_MEASURED,
    AUTONOMY_RESEARCH_SKIPPED,
    AUTONOMY_L3_ACTIVATION_DENIED,
    AUTONOMY_L3_ELIGIBLE,
    AUTONOMY_L3_PROMOTED,
)

from autonomy.calibrator import AutonomyCalibrator
from autonomy.curiosity_detector import CuriosityDetector
from autonomy.delta_tracker import DeltaTracker, MetricSnapshot
from autonomy.metric_history import MetricHistoryTracker
from autonomy.eval_harness import EpisodeRecorder, TraceEpisode
from autonomy.event_bridge import AutonomyEventBridge
from autonomy.knowledge_integrator import KnowledgeIntegrator
from autonomy.metric_triggers import MetricTriggers
from autonomy.constants import MIN_MEANINGFUL_DELTA
from autonomy.opportunity_scorer import OpportunityScorer
from autonomy.policy_memory import AutonomyPolicyMemory, PolicyOutcome
from autonomy.query_interface import InternalQueryInterface
from autonomy.research_governor import ResearchGovernor
from autonomy.research_intent import ResearchIntent

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = 20
PROCESS_INTERVAL_S = 30.0
METRICS_FEED_INTERVAL_S = 5.0
DRIVE_EVAL_INTERVAL_S = 60.0
PROMOTION_CHECK_INTERVAL_S = 300.0
QUESTION_DEDUP_EPISODE_LOOKBACK = 100
QUESTION_DEDUP_COOLDOWN_S = 6 * 3600.0

AUTONOMY_LEVELS = {
    0: "propose",
    1: "research",
    2: "safe_apply",
    3: "full",
}

_JARVIS_DIR = Path(os.path.expanduser("~/.jarvis"))
_AUTONOMY_STATE_PATH = _JARVIS_DIR / "autonomy_state.json"


class AutonomyOrchestrator:
    """Coordinates the full autonomous research pipeline.

    Lifecycle:
      1. AutonomyEventBridge detects curiosity → enqueue(intent)
      2. MetricTriggers detect sustained deficits → enqueue(intent)
      3. OpportunityScorer.score() replaces raw priority with composite score
      4. on_tick() called from engine → process_queue()
      5. DeltaTracker.start_tracking() captures baseline
      6. Governor gates the intent (+ level check)
      7. QueryInterface executes headlessly
      8. KnowledgeIntegrator stores findings as memories + evidence
      9. DeltaTracker.mark_completed() → post-window delta measurement
     10. Dashboard shows the full queue with scores, deltas, and provenance
    """

    def __init__(self, autonomy_level: int = 1) -> None:
        self._detector = CuriosityDetector()
        self._governor = ResearchGovernor()
        self._query_interface = InternalQueryInterface()
        self._integrator = KnowledgeIntegrator()
        self._scorer = OpportunityScorer()
        self._metric_triggers = MetricTriggers()
        self._metric_history = MetricHistoryTracker()
        self._delta_tracker = DeltaTracker(metric_history=self._metric_history)
        self._rehydrate_delta_tracker()
        self._policy_memory = AutonomyPolicyMemory()
        self._calibrator = AutonomyCalibrator()
        self._episode_recorder = EpisodeRecorder()
        self._bridge = AutonomyEventBridge(
            detector=self._detector,
            enqueue_cb=self.enqueue,
        )

        self._scorer.set_policy_memory(self._policy_memory)
        self._metric_triggers.set_policy_memory(self._policy_memory)

        self._drive_manager = None
        try:
            from autonomy.drives import DriveManager
            self._drive_manager = DriveManager()
            try:
                self._drive_manager.load_state()
            except Exception:
                logger.warning("DriveManager state load failed (non-critical)", exc_info=True)
        except Exception:
            logger.warning("DriveManager not available, drive-based autonomy disabled")

        self._queue: deque[ResearchIntent] = deque(maxlen=MAX_QUEUE_SIZE)
        self._completed: deque[ResearchIntent] = deque(maxlen=200)
        self._intent_metadata: dict[str, dict[str, Any]] = {}
        self._planner_shadow_previews: deque[dict[str, Any]] = deque(maxlen=50)
        self._intent_ledger_ids: dict[str, str] = {}
        self._metadata_prune_counter: int = 0
        self._last_process_time: float = 0.0
        self._last_metrics_feed_time: float = 0.0
        self._last_drive_eval_time: float = 0.0
        self._saturated_topics: set[str] = set()
        self._topic_recall_misses: dict[str, int] = {}
        self._last_saturation_clear: float = time.time()
        self._enabled: bool = True
        self._started: bool = False
        self._engine_ref: Any = None
        self._autonomy_level: int = max(0, min(3, autonomy_level))
        self._level_restored_from_disk: bool = False
        self._persisted_autonomy_data: dict | None = None
        self._current_mode: str = ""
        self._goal_callback: Any = None
        self._goal_manager: Any = None
        self._last_promotion_check: float = 0.0
        self._last_eval_replay_time: float = 0.0
        self._boot_time: float = time.time()
        self._last_enqueue_block_reason: str = ""
        # Phase 6.5: L3 eligibility is announced at most once per session.
        # Emitting AUTONOMY_L3_ELIGIBLE every tick would create log/event
        # spam; the event represents "this session has earned eligibility"
        # and downstream consumers (dashboard, validation pack) can read
        # live state for polling needs.
        self._l3_eligibility_announced: bool = False
        self._escalation_store: Any = None
        self._escalation_wire_last_error_log_ts: float = 0.0

        self._restore_autonomy_level(config_level=autonomy_level)

    def set_goal_callback(self, callback: Any) -> None:
        """Register callback for forwarding delta outcomes to Goal Continuity Layer.

        Signature: callback(goal_id, task_id, intent_id, delta_result_dict)
        """
        self._goal_callback = callback

    def set_goal_manager(self, mgr: Any) -> None:
        """Register goal manager for intent annotation and suppression."""
        self._goal_manager = mgr

    def _rehydrate_delta_tracker(self) -> None:
        """Reload pending delta windows and cumulative counters from disk."""
        try:
            self._delta_tracker.load_counters()
        except Exception:
            logger.warning("Delta tracker counter load failed", exc_info=True)
        try:
            restored, interrupted = self._delta_tracker.load_pending()
            if interrupted:
                logger.info("Delta tracker: %d interrupted outcomes from prior session", len(interrupted))
        except Exception:
            logger.warning("Delta tracker rehydration failed", exc_info=True)

    def _restore_autonomy_level(self, config_level: int) -> None:
        """Load persisted autonomy state for reconcile_on_boot() to evaluate.

        Does NOT apply the level — that decision is deferred to reconcile_on_boot()
        where quarantine, contradiction engine, and other safety subsystems are
        fully initialized.
        """
        try:
            if not _AUTONOMY_STATE_PATH.exists():
                return
            data = json.loads(_AUTONOMY_STATE_PATH.read_text())
            persisted_level = data.get("autonomy_level")
            if persisted_level is not None and isinstance(persisted_level, int):
                self._persisted_autonomy_data = data
                logger.info(
                    "Autonomy persisted state loaded: L%d (promoted_at=%s)",
                    persisted_level, data.get("promoted_at", "?"),
                )
        except Exception:
            logger.warning("Autonomy state load failed", exc_info=True)

    _DEBT_VETO_THRESHOLD = 0.20
    _REGRESSION_WINDOW = 5
    _STRONG_REGRESSION_MULTIPLIER = 2.0

    def reconcile_on_boot(self) -> dict[str, Any]:
        """Cross-check persisted autonomy state and attempt gated auto-restore.

        Auto-restore rules:
          - Persisted L2 + all 5 gates pass  -> restore to L2
          - Persisted L3 + all gates pass    -> warn only (manual restore)
          - Any gate fails                   -> stay at current level

        Returns a structured reconciliation report.
        """
        level_before = self._autonomy_level

        report: dict[str, Any] = {
            "auto_restored": False,
            "restored_level": None,
            "requested_level": None,
            "vetoed_by": [],
            "l3_eligible_but_manual": False,
            "gate_results": {
                "persisted_state_valid": False,
                "policy_qualifies": False,
                "regression_clear": False,
                "quarantine_clear": False,
                "debt_clear": False,
            },
            "current_level_before": level_before,
            "current_level_after": level_before,
            "persisted_level": None,
            "policy_eligible_level": 1,
            "persisted_state_age_s": None,
            "persisted_snapshot_matches_policy": None,
            "recent_regression_count": 0,
            "contradiction_debt": 0.0,
            "quarantine_high": False,
            "disagreements": [],
        }

        elig = self.check_promotion_eligibility()
        stats = self._policy_memory.get_stats()
        report["policy_eligible_level"] = (
            3 if elig.get("eligible_for_l3") else
            2 if elig.get("eligible_for_l2") else 1
        )

        persisted = self._persisted_autonomy_data
        if persisted is None:
            logger.info("Boot reconciliation: no persisted autonomy state")
            return report

        persisted_level = max(0, min(3, persisted.get("autonomy_level", 0)))
        report["requested_level"] = persisted_level
        report["persisted_level"] = persisted_level

        promoted_at = persisted.get("promoted_at")
        if promoted_at and isinstance(promoted_at, (int, float)):
            report["persisted_state_age_s"] = round(time.time() - promoted_at, 1)

        persisted_win_rate = persisted.get("restored_from_policy_win_rate")
        persisted_outcomes = persisted.get("restored_from_policy_outcomes")
        current_win_rate = stats.get("overall_win_rate", 0.0)
        current_outcomes = stats.get("total_outcomes", 0)
        if persisted_win_rate is not None and persisted_outcomes is not None:
            rate_close = abs(persisted_win_rate - current_win_rate) <= 0.05
            outcomes_close = abs(persisted_outcomes - current_outcomes) <= 10
            report["persisted_snapshot_matches_policy"] = rate_close and outcomes_close

        if persisted_level <= self._autonomy_level:
            logger.info(
                "Boot reconciliation: persisted L%d <= current L%d, no restore needed",
                persisted_level, self._autonomy_level,
            )
            return report

        # -- Gate 1: persisted state valid --
        if persisted_level >= 2 and promoted_at:
            report["gate_results"]["persisted_state_valid"] = True
        else:
            report["vetoed_by"].append("persisted_state_invalid")

        # -- Gate 2: policy memory still qualifies --
        if persisted_level == 2 and elig.get("eligible_for_l2"):
            report["gate_results"]["policy_qualifies"] = True
        elif persisted_level == 3 and elig.get("eligible_for_l3"):
            report["gate_results"]["policy_qualifies"] = True
        else:
            report["vetoed_by"].append("policy_no_longer_qualifies")

        # -- Gate 3: no significant recent regressions --
        regression_count = self._count_recent_regressions()
        report["recent_regression_count"] = regression_count["count"]
        if regression_count["vetoed"]:
            report["vetoed_by"].append(
                f"recent_regressions ({regression_count['reason']})"
            )
        else:
            report["gate_results"]["regression_clear"] = True

        # -- Gate 4: quarantine not high --
        quarantine_high = self._check_quarantine_high()
        report["quarantine_high"] = quarantine_high
        if quarantine_high:
            report["vetoed_by"].append("quarantine_pressure_high")
        else:
            report["gate_results"]["quarantine_clear"] = True

        # -- Gate 5: contradiction debt below threshold --
        debt = self._check_contradiction_debt()
        report["contradiction_debt"] = round(debt, 4)
        if debt >= self._DEBT_VETO_THRESHOLD:
            report["vetoed_by"].append(
                f"contradiction_debt_high ({debt:.4f} >= {self._DEBT_VETO_THRESHOLD})"
            )
        else:
            report["gate_results"]["debt_clear"] = True

        # -- Decision --
        all_gates_pass = not report["vetoed_by"]

        if all_gates_pass and persisted_level == 2:
            self.set_autonomy_level(2)
            self._level_restored_from_disk = True
            report["auto_restored"] = True
            report["restored_level"] = 2
            report["current_level_after"] = 2
            age_tag = ""
            if report["persisted_state_age_s"] is not None:
                age_tag = f", age={report['persisted_state_age_s']:.0f}s"
            match_tag = ""
            if report["persisted_snapshot_matches_policy"] is not None:
                match_tag = f", snapshot_match={report['persisted_snapshot_matches_policy']}"
            logger.info(
                "Boot auto-restore: L%d → L2 (all gates passed, "
                "policy_win_rate=%.3f, outcomes=%d%s%s)",
                level_before, current_win_rate, current_outcomes,
                age_tag, match_tag,
            )
        elif all_gates_pass and persisted_level == 3:
            report["l3_eligible_but_manual"] = True
            report["current_level_after"] = self._autonomy_level
            logger.info(
                "Boot reconciliation: L3 eligible for restore (all gates passed) "
                "but L3 auto-restore is manual-only — staying at L%d",
                self._autonomy_level,
            )
        elif report["vetoed_by"]:
            report["current_level_after"] = self._autonomy_level
            logger.warning(
                "Boot reconciliation: restore to L%d vetoed by %s — staying at L%d",
                persisted_level, report["vetoed_by"], self._autonomy_level,
            )

        # -- Disagreement detection (backward compat + informational) --
        if self._autonomy_level < 2 and elig.get("eligible_for_l2"):
            report["disagreements"].append(
                f"autonomy_level=L{self._autonomy_level} but policy memory qualifies for L2"
            )
        if self._autonomy_level == 2 and elig.get("eligible_for_l3"):
            report["disagreements"].append(
                "autonomy_level=L2 but policy memory qualifies for L3"
            )

        try:
            from cognition.promotion import WorldModelPromotion
            wm_promo = WorldModelPromotion()
            wm_level = wm_promo._state.level
            if wm_level >= 2 and self._autonomy_level < 2:
                report["disagreements"].append(
                    f"world_model promotion=active (level {wm_level}) "
                    f"but autonomy only L{self._autonomy_level}"
                )
        except Exception:
            pass

        for d in report["disagreements"]:
            logger.warning("Boot reconciliation: %s", d)

        return report

    def _count_recent_regressions(self) -> dict[str, Any]:
        """Examine last N non-warmup outcomes for regression signals.

        Veto if: 2+ meaningful regressions, OR most recent is a strong regression.
        """
        outcomes = [o for o in self._policy_memory._outcomes if not o.warmup]
        recent = outcomes[-self._REGRESSION_WINDOW:] if outcomes else []

        meaningful_threshold = MIN_MEANINGFUL_DELTA
        strong_threshold = self._STRONG_REGRESSION_MULTIPLIER * MIN_MEANINGFUL_DELTA

        regression_count = sum(
            1 for o in recent if o.net_delta < -meaningful_threshold
        )

        most_recent_strong = False
        if recent and recent[-1].net_delta < -strong_threshold:
            most_recent_strong = True

        vetoed = False
        reason = ""
        if regression_count >= 2:
            vetoed = True
            reason = f"{regression_count} regressions in last {len(recent)}"
        elif most_recent_strong:
            vetoed = True
            reason = (
                f"most recent outcome is strong regression "
                f"(delta={recent[-1].net_delta:.4f})"
            )

        return {"count": regression_count, "vetoed": vetoed, "reason": reason}

    def _check_quarantine_high(self) -> bool:
        """Check if quarantine pressure is in 'high' band."""
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            return get_quarantine_pressure().current.high
        except Exception:
            return False

    def _check_contradiction_debt(self) -> float:
        """Read current contradiction debt from the engine."""
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            return ContradictionEngine.get_instance().contradiction_debt
        except Exception:
            return 0.0

    # -- lifecycle -----------------------------------------------------------

    def start(self, engine: Any = None) -> None:
        if self._started:
            return
        self._engine_ref = engine
        self._integrator.set_engine(engine)
        self._bridge.wire()
        self._started = True
        logger.info(
            "Autonomy orchestrator started (level=%d/%s, queue=%d)",
            self._autonomy_level, AUTONOMY_LEVELS.get(self._autonomy_level, "?"),
            MAX_QUEUE_SIZE,
        )

    def stop(self) -> None:
        self._bridge.unwire()
        self._started = False
        logger.info("Autonomy orchestrator stopped")

    def set_strict_provenance(self, strict: bool) -> None:
        """Toggle strict provenance gating on the knowledge integrator."""
        self._integrator.set_strict_provenance(strict)

    @property
    def autonomy_level(self) -> int:
        return self._autonomy_level

    def set_autonomy_level(
        self,
        level: int,
        *,
        evidence_path: str | None = None,
        approval_source: str = "",
        caller_id: str = "",
    ) -> None:
        """Change the live autonomy level.

        Phase 6.5 invariant: transitions to L3 are only permitted with an
        explicit ``evidence_path`` from the manual-promotion API. Callers
        without an evidence_path (e.g. the internal promotion loop) are
        refused when attempting L3. L0/L1/L2 transitions are unchanged.

        Parameters
        ----------
        level
            Target autonomy level (clamped to [0, 3]).
        evidence_path
            One of ``"current_eligible"``, ``"prior_attested"``,
            ``"operator_override"``, ``"l3_emergency_override"``. Required
            when transitioning into L3; ignored otherwise.
        approval_source, caller_id
            Audit fields; carried through to the ``AUTONOMY_L3_PROMOTED``
            event payload when applicable.
        """
        level = max(0, min(3, level))
        if level == self._autonomy_level:
            return

        if level == 3 and evidence_path is None:
            logger.error(
                "Refusing to set autonomy L3 without evidence_path "
                "(caller=%r approval_source=%r)",
                caller_id, approval_source,
            )
            self._emit_event(
                AUTONOMY_L3_ACTIVATION_DENIED,
                reason="missing_evidence_path",
                caller_id=caller_id,
                approval_source=approval_source,
                current_level=self._autonomy_level,
                denied_at=time.time(),
            )
            raise PermissionError(
                "set_autonomy_level(3) requires explicit evidence_path; "
                "L3 auto-promotion is forbidden by Phase 6.5 invariant"
            )

        old = self._autonomy_level
        self._autonomy_level = level
        logger.info(
            "Autonomy level changed: L%d (%s) → L%d (%s)",
            old, AUTONOMY_LEVELS.get(old, "?"),
            level, AUTONOMY_LEVELS.get(level, "?"),
        )
        self._emit_event(AUTONOMY_LEVEL_CHANGED, old_level=old, new_level=level)

        if level == 3 and old == 2:
            # Single authoritative record that L3 became active this
            # session. This event fires ONLY on a successful 2→3
            # transition. Denials live on AUTONOMY_L3_ACTIVATION_DENIED;
            # rollbacks of the triggering escalation live on
            # AUTONOMY_ESCALATION_ROLLED_BACK. ``outcome`` is always
            # the literal string "clean" for this event; any other
            # value is a taxonomy bug.
            self._emit_event(
                AUTONOMY_L3_PROMOTED,
                evidence_path=evidence_path,
                approval_source=approval_source,
                caller_id=caller_id,
                outcome="clean",
                prior_level=old,
                promoted_at=time.time(),
            )

    _WARMUP_BEFORE_PROMOTION_S = 1800.0

    def _check_and_apply_promotion(self) -> None:
        """Evaluate earned promotion and apply one level at a time.

        Gates: no promotion during warmup (first 30 min of session) unless
        the current level was restored from disk, no promotion during
        gestation, one level per check.
        """
        if self._autonomy_level >= 3:
            return
        if not self._level_restored_from_disk and \
                time.time() - self._boot_time < self._WARMUP_BEFORE_PROMOTION_S:
            return
        if self._current_mode == "gestation":
            return

        elig = self.check_promotion_eligibility()

        if self._autonomy_level < 2 and elig.get("eligible_for_l2"):
            logger.info(
                "Autonomy promotion earned: L%d → L2 (%s)",
                self._autonomy_level, elig.get("l2_reason", ""),
            )
            self.set_autonomy_level(2)
        elif self._autonomy_level == 2 and elig.get("eligible_for_l3"):
            # Phase 6.5 invariant: L3 is never auto-promoted. Live-runtime
            # eligibility only emits AUTONOMY_L3_ELIGIBLE; actual promotion
            # requires POST /api/autonomy/level with explicit evidence_path.
            # See docs/plans/phase_6_5_l3_escalation.plan.md.
            if not self._l3_eligibility_announced:
                self._l3_eligibility_announced = True
                logger.info(
                    "Autonomy L3 eligibility earned: %s (manual promotion required)",
                    elig.get("l3_reason", ""),
                )
                self._emit_event(
                    AUTONOMY_L3_ELIGIBLE,
                    reason=elig.get("l3_reason", ""),
                    wins=elig.get("wins", 0),
                    win_rate=elig.get("win_rate", 0.0),
                    regressions_in_last_10=elig.get("recent_regressions", 0),
                )
            return

    # -- queue management ----------------------------------------------------

    def enqueue(self, intent: ResearchIntent) -> bool:
        self._last_enqueue_block_reason = ""
        if not self._enabled:
            self._last_enqueue_block_reason = "autonomy_disabled"
            return False
        # Golden authorization is required only for non-local code operations.
        # Local-only codebase intents are read-only research queries and do not
        # mutate files; they should not be blocked at enqueue time.
        intent_scope = str(getattr(intent, "scope", "local_only") or "local_only").lower()
        if (
            getattr(intent, "goal_id", "")
            and str(getattr(intent, "source_hint", "")).lower() == "codebase"
            and intent_scope != "local_only"
            and str(getattr(intent, "golden_status", "none")).lower() != "executed"
        ):
            intent.status = "blocked"
            intent.blocked_reason = "golden_required:goal_apply"
            self._last_enqueue_block_reason = intent.blocked_reason
            self._emit_event(
                AUTONOMY_INTENT_BLOCKED,
                intent_id=intent.id,
                goal_id=str(getattr(intent, "goal_id", "")),
                task_id=str(getattr(intent, "task_id", "")),
                golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                golden_command_id=str(getattr(intent, "golden_command_id", "")),
                reason=intent.blocked_reason,
            )
            return False
        if self._question_already_queued(intent.question):
            logger.debug("Autonomy enqueue skipped: duplicate question already queued/completed")
            self._last_enqueue_block_reason = "duplicate_question"
            return False
        if len(self._queue) >= MAX_QUEUE_SIZE:
            lowest = min(self._queue, key=lambda i: i.priority)
            if intent.priority <= lowest.priority:
                self._last_enqueue_block_reason = "queue_full_low_priority"
                return False
            self._queue.remove(lowest)
            lowest.status = "cancelled"

        # Annotate: promote goal-adjacent intents to goal-linked
        if self._goal_manager:
            try:
                self._goal_manager.annotate_intent(intent)
            except Exception:
                pass

        opp = self._scorer.score(intent)
        intent.priority = opp.total
        score_detail = (
            f"[score={opp.total:.2f} "
            f"I={opp.impact:.2f} E={opp.evidence:.2f} C={opp.confidence:.2f} "
            f"R={opp.risk:.2f} $={opp.cost:.2f}"
        )
        if opp.policy_adjustment:
            score_detail += f" pol={opp.policy_adjustment:+.2f}"
        if opp.diminishing_penalty > 0.01:
            score_detail += f" dim={opp.diminishing_penalty:.2f}"
        if opp.action_rate_penalty > 0.01:
            score_detail += f" rate={opp.action_rate_penalty:.2f}"
        score_detail += "]"
        intent.reason = f"{score_detail} {intent.reason}"

        shadow_preview = self._build_shadow_policy_preview(intent)
        if shadow_preview is not None:
            self._planner_shadow_previews.append(shadow_preview)
            intent.reason = (
                f"[planner_shadow proposed={shadow_preview['proposed_priority']:.2f} "
                f"delta={shadow_preview['proposed_delta']:.2f} applied=false] "
                f"{intent.reason}"
            )

        self._queue.append(intent)
        self._emit_event(
            AUTONOMY_INTENT_QUEUED,
            intent_id=intent.id,
            goal_id=str(getattr(intent, "goal_id", "")),
            task_id=str(getattr(intent, "task_id", "")),
            golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
            golden_command_id=str(getattr(intent, "golden_command_id", "")),
            question=intent.question[:60],
            priority=intent.priority,
            score=opp.to_dict(),
        )
        return True

    @staticmethod
    def _build_shadow_policy_preview(intent: Any) -> dict[str, Any] | None:
        """Build a non-authoritative policy preview from planner shadow metadata."""
        event = str(getattr(intent, "shadow_planner_event", "") or "")
        recommendation = str(getattr(intent, "shadow_planner_recommendation", "") or "")
        if not event and not recommendation:
            return None

        base_priority = float(getattr(intent, "priority", 0.0) or 0.0)
        utility = max(0.0, float(getattr(intent, "shadow_planner_utility", 0.0) or 0.0))
        alignment = max(0.0, float(getattr(intent, "shadow_planner_goal_alignment", 0.0) or 0.0))

        proposed_delta = min(0.20, (utility * 0.10) + max(0.0, alignment - 1.0) * 0.15)
        proposed_priority = base_priority + proposed_delta
        return {
            "intent_id": str(getattr(intent, "id", "")),
            "goal_id": str(getattr(intent, "goal_id", "")),
            "task_id": str(getattr(intent, "task_id", "")),
            "source_event": event,
            "recommendation": recommendation,
            "base_priority": round(base_priority, 3),
            "proposed_priority": round(proposed_priority, 3),
            "proposed_delta": round(proposed_delta, 3),
            "utility": round(utility, 3),
            "goal_alignment": round(alignment, 3),
            "applied": False,
            "mode": "shadow_only",
        }

    @staticmethod
    def _intent_trace_fields(intent: Any) -> dict[str, str]:
        """Canonical correlation fields for autonomy intent ledger records."""
        return {
            "intent_id": str(getattr(intent, "id", "")),
            "goal_id": str(getattr(intent, "goal_id", "")),
            "task_id": str(getattr(intent, "task_id", "")),
            "golden_trace_id": str(getattr(intent, "golden_trace_id", "")),
            "golden_command_id": str(getattr(intent, "golden_command_id", "")),
            "golden_status": str(getattr(intent, "golden_status", "none")),
            "source_event": str(getattr(intent, "source_event", "")),
        }

    @classmethod
    def _intent_evidence_refs(cls, intent: Any) -> list[dict[str, str]]:
        """Evidence refs used for deterministic intent -> goal/task joins."""
        trace = cls._intent_trace_fields(intent)
        refs: list[dict[str, str]] = []
        if trace["intent_id"]:
            refs.append({"kind": "intent", "id": trace["intent_id"]})
        if trace["goal_id"]:
            refs.append({"kind": "goal", "id": trace["goal_id"]})
        if trace["task_id"]:
            refs.append({"kind": "goal_task", "id": trace["task_id"]})
        if trace["golden_trace_id"]:
            refs.append({"kind": "golden_trace", "id": trace["golden_trace_id"]})
        if trace["golden_command_id"]:
            refs.append({"kind": "golden_command", "id": trace["golden_command_id"]})
        return refs

    def on_tick(self, current_mode: str) -> None:
        if not self._enabled or not self._started:
            return
        self._current_mode = current_mode

        now = time.time()

        if now - self._last_metrics_feed_time >= METRICS_FEED_INTERVAL_S:
            self._last_metrics_feed_time = now
            self._feed_metrics()

        delta_results = self._delta_tracker.check_pending()
        for dr in delta_results:
            self._process_delta_outcome(dr)
            self._emit_event(AUTONOMY_DELTA_MEASURED,
                             intent_id=dr.intent_id,
                             net_improvement=dr.net_improvement,
                             net_attribution=dr.net_attribution,
                             stable=dr.stable)

        try:
            from autonomy.source_ledger import get_source_ledger
            get_source_ledger().compute_verdicts()
        except Exception:
            pass

        try:
            from autonomy.intervention_runner import get_intervention_runner
            runner = get_intervention_runner()
            current_metrics = self._collect_metrics()
            runner.auto_activate_proposed(metrics=current_metrics)
            shadow_results = runner.check_shadow_results(metrics=current_metrics)
            for iv in shadow_results:
                if iv.measured_delta > 0.01:
                    runner.promote(iv.intervention_id)
                else:
                    runner.discard(iv.intervention_id, reason="no_measurable_improvement")
        except Exception:
            pass

        if self._drive_manager and now - self._last_drive_eval_time >= DRIVE_EVAL_INTERVAL_S:
            self._last_drive_eval_time = now
            self._evaluate_drives()

        if now - self._last_promotion_check >= PROMOTION_CHECK_INTERVAL_S:
            self._last_promotion_check = now
            self._check_and_apply_promotion()

        if (
            self._current_mode in ("dreaming", "reflective", "sleep", "deep_learning")
            and now - self._last_eval_replay_time >= 3600.0
        ):
            self._last_eval_replay_time = now
            self._run_eval_replay()

        if now - self._last_process_time < PROCESS_INTERVAL_S:
            return

        self._last_process_time = now

        if self._autonomy_level >= 1:
            self._process_next(current_mode)

    # -- processing ----------------------------------------------------------

    def _process_next(self, current_mode: str) -> None:
        if not self._queue:
            return

        # Hard gate: check if a user goal is stalled
        stalled_goal = None
        if self._goal_manager:
            try:
                stalled_goal = self._goal_manager.get_stalled_user_goal()
            except Exception:
                pass

        sorted_queue = sorted(self._queue, key=lambda i: i.priority, reverse=True)
        for intent in sorted_queue:
            if intent.status != "queued":
                continue

            # Hard gate: block non-goal, non-metric, non-adjacent research
            if stalled_goal and self._goal_manager:
                source_prefix = (intent.source_event or "").split(":")[0]
                is_metric = source_prefix == "metric"
                is_goal_linked = bool(getattr(intent, "goal_id", ""))
                is_adjacent = False
                if not is_goal_linked and not is_metric:
                    try:
                        is_adjacent = self._goal_manager.classify_intent_alignment(intent) == "adjacent"
                    except Exception:
                        pass
                if not is_metric and not is_goal_linked and not is_adjacent:
                    logger.info(
                        "Hard gate evicted intent %s (stalled goal %s): %s",
                        intent.id, stalled_goal.goal_id, intent.question[:60],
                    )
                    intent.status = "blocked"
                    try:
                        self._queue.remove(intent)
                    except ValueError:
                        pass
                    self._emit_event(
                        AUTONOMY_INTENT_BLOCKED,
                        intent_id=intent.id,
                        goal_id=str(getattr(intent, "goal_id", "")),
                        task_id=str(getattr(intent, "task_id", "")),
                        golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                        golden_command_id=str(getattr(intent, "golden_command_id", "")),
                        reason="hard_gate:user_goal_stalled",
                    )
                    self._requeue_gestation_directive(intent)
                    continue

            # Soft suppression: existential + no goal link + stalled user goal
            if self._goal_manager:
                try:
                    if self._goal_manager.should_suppress(intent, mode=current_mode):
                        intent.status = "blocked"
                        try:
                            self._queue.remove(intent)
                        except ValueError:
                            pass
                        self._emit_event(
                            AUTONOMY_INTENT_BLOCKED,
                            intent_id=intent.id,
                            goal_id=str(getattr(intent, "goal_id", "")),
                            task_id=str(getattr(intent, "task_id", "")),
                            golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                            golden_command_id=str(getattr(intent, "golden_command_id", "")),
                            reason="goal_suppressed:existential",
                        )
                        self._requeue_gestation_directive(intent)
                        continue
                except Exception:
                    pass

            decision = self._governor.evaluate(intent, current_mode)
            if not decision.allowed:
                intent.status = "blocked"
                try:
                    self._queue.remove(intent)
                except ValueError:
                    pass
                self._emit_event(
                    AUTONOMY_INTENT_BLOCKED,
                    intent_id=intent.id,
                    goal_id=str(getattr(intent, "goal_id", "")),
                    task_id=str(getattr(intent, "task_id", "")),
                    golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                    golden_command_id=str(getattr(intent, "golden_command_id", "")),
                    reason=decision.reason,
                )
                self._requeue_gestation_directive(intent)
                continue

            baseline = self._delta_tracker.start_tracking(
                intent.id, source_event=intent.source_event or "",
            )
            self._governor.record_start(intent)
            self._emit_event(
                AUTONOMY_RESEARCH_STARTED,
                intent_id=intent.id,
                goal_id=str(getattr(intent, "goal_id", "")),
                task_id=str(getattr(intent, "task_id", "")),
                golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                golden_command_id=str(getattr(intent, "golden_command_id", "")),
                question=intent.question[:60],
                tool_hint=intent.source_hint,
                baseline=baseline.to_dict(),
            )
            try:
                from consciousness.attribution_ledger import attribution_ledger
                _intent_trace = self._intent_trace_fields(intent)
                _started_eid = attribution_ledger.record(
                    subsystem="autonomy",
                    event_type="research_started",
                    source=getattr(intent, "source_event", ""),
                    data={
                        "question": intent.question[:120],
                        "tool_hint": intent.source_hint or "",
                        **_intent_trace,
                    },
                    evidence_refs=self._intent_evidence_refs(intent),
                )
                if _started_eid:
                    self._intent_ledger_ids[intent.id] = _started_eid
            except Exception:
                pass

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._execute_and_integrate(intent))
                else:
                    loop.run_until_complete(self._execute_and_integrate(intent))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._execute_and_integrate(intent))
                finally:
                    loop.close()

            return

    async def _execute_and_integrate(self, intent: ResearchIntent) -> None:
        start_ts = time.time()
        try:
            _intent_trace = self._intent_trace_fields(intent)
            _intent_evidence_refs = self._intent_evidence_refs(intent)
            prior = self._integrator.check_prior_knowledge(intent.question)
            if prior["recommendation"] == "skip":
                mem_ids = [m.get("id", "?")[:12] for m in prior.get("existing_memories", [])]
                sig = prior.get("match_signature", "")
                logger.info(
                    "Skipping research (already known, confidence=%.2f, sig=%s, memories=%s): %s",
                    prior["best_confidence"], sig, mem_ids, intent.question[:60],
                )
                self._governor.record_complete(intent)
                self._delta_tracker.mark_completed(intent.id)
                intent.status = "completed"
                try:
                    self._queue.remove(intent)
                except ValueError:
                    pass
                self._completed.append(intent)
                self._intent_metadata[intent.id] = {
                    "tool_used": "skip",
                    "tag_cluster": tuple(intent.tag_cluster) if intent.tag_cluster else (),
                    "source_event": intent.source_event or "",
                    "question": intent.question[:80],
                    "goal_id": intent.goal_id or "",
                    "task_id": intent.task_id or "",
                    "golden_trace_id": intent.golden_trace_id or "",
                    "golden_command_id": intent.golden_command_id or "",
                    "golden_status": intent.golden_status or "none",
                }
                self._prune_intent_metadata()
                self._record_episode(intent, "allowed", "skipped:already_known", 0.0)
                self._emit_event(
                    AUTONOMY_RESEARCH_SKIPPED,
                    intent_id=intent.id,
                    goal_id=str(getattr(intent, "goal_id", "")),
                    task_id=str(getattr(intent, "task_id", "")),
                    golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                    golden_command_id=str(getattr(intent, "golden_command_id", "")),
                    question=intent.question[:60],
                    reason="already_known",
                    existing_confidence=prior["best_confidence"],
                    existing_memories=prior.get("existing_memories", []),
                    match_signature=sig,
                )
                self._record_drive_soft_outcome(intent, None, 0, already_known=True)
                return
            if prior["recommendation"] == "verify":
                age_h = prior.get("age_hours", 0)
                logger.info(
                    "Research will verify stale knowledge (%.0fh old): %s",
                    age_h, intent.question[:60],
                )

            result = await self._query_interface.execute(intent)
            self._governor.record_complete(intent)
            self._delta_tracker.mark_completed(intent.id)
            self._scorer.record_action(self._scorer._intent_category(intent))

            _root_eid = self._intent_ledger_ids.get(intent.id, "")
            _last_chain_eid = _root_eid
            try:
                from consciousness.attribution_ledger import attribution_ledger as _al
                _query_eid = _al.record(
                    subsystem="autonomy",
                    event_type="query_executed",
                    source=result.tool_used,
                    data={
                        "question": intent.question[:120],
                        "findings": len(result.findings),
                        "success": result.success,
                        "tool": result.tool_used,
                        **_intent_trace,
                    },
                    evidence_refs=_intent_evidence_refs,
                    parent_entry_id=_root_eid,
                    root_entry_id=_root_eid,
                )
                _last_chain_eid = _query_eid
            except Exception:
                pass

            memories_created = 0
            if result.success and result.findings:
                conflicts = self._integrator.detect_conflicts(
                    intent.question, result.findings,
                )
                if conflicts:
                    upgrades = sum(1 for c in conflicts if c["is_upgrade"])
                    logger.info(
                        "Knowledge conflicts detected for '%s': %d conflicts "
                        "(upgrades: %d)",
                        intent.question[:40],
                        len(conflicts),
                        upgrades,
                    )
                    if upgrades > 0:
                        self._integrator.apply_upgrades(conflicts)
                memories_created = self._integrator.integrate(intent, result)

                try:
                    from consciousness.attribution_ledger import attribution_ledger as _al2
                    _integrate_eid = _al2.record(
                        subsystem="autonomy",
                        event_type="knowledge_integrated",
                        source=result.tool_used,
                        data={
                            "memories_created": memories_created,
                            "conflicts": len(conflicts) if conflicts else 0,
                            "upgrades": upgrades if conflicts else 0,
                            **_intent_trace,
                        },
                        evidence_refs=_intent_evidence_refs,
                        parent_entry_id=_last_chain_eid,
                        root_entry_id=_root_eid,
                    )
                    _last_chain_eid = _integrate_eid
                except Exception:
                    pass

            try:
                self._queue.remove(intent)
            except ValueError:
                pass
            self._completed.append(intent)
            self._intent_metadata[intent.id] = {
                "tool_used": result.tool_used,
                "tag_cluster": tuple(intent.tag_cluster) if intent.tag_cluster else (),
                "source_event": intent.source_event or "",
                "question": intent.question[:80],
                "memories_created": memories_created,
                "immediate_success": result.success and memories_created > 0,
                "goal_id": intent.goal_id or "",
                "task_id": intent.task_id or "",
                "golden_trace_id": intent.golden_trace_id or "",
                "golden_command_id": intent.golden_command_id or "",
                "golden_status": intent.golden_status or "none",
            }
            self._prune_intent_metadata()

            self._record_episode(intent, "allowed", "success", time.time() - start_ts)

            self._emit_event(
                AUTONOMY_RESEARCH_COMPLETED,
                intent_id=intent.id,
                goal_id=str(getattr(intent, "goal_id", "")),
                task_id=str(getattr(intent, "task_id", "")),
                golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                golden_command_id=str(getattr(intent, "golden_command_id", "")),
                question=intent.question[:60],
                tool_used=result.tool_used,
                findings=len(result.findings),
                memories_created=memories_created,
                success=result.success,
            )
            try:
                from cognition.intention_registry import intention_registry
                intention_registry.resolve(
                    backing_job_id=intent.id,
                    outcome="resolved" if result.success else "failed",
                    reason=f"research_{result.tool_used}" if result.success else "research_no_findings",
                    result_summary=intent.question[:200],
                    metadata={
                        "memories_created": memories_created,
                        "findings": len(result.findings),
                        "tool_used": result.tool_used,
                    },
                )
            except Exception:
                logger.debug("intention_registry.resolve (autonomy completed) failed", exc_info=True)
            try:
                from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
                attribution_ledger.record(
                    subsystem="autonomy",
                    event_type="research_completed",
                    source=result.tool_used,
                    data={
                        "question": intent.question[:120],
                        "findings": len(result.findings),
                        "memories_created": memories_created,
                        "success": result.success,
                        **_intent_trace,
                    },
                    evidence_refs=_intent_evidence_refs,
                    parent_entry_id=_last_chain_eid,
                    root_entry_id=_root_eid,
                )
                if _root_eid:
                    _immediate_outcome = "success" if result.success and memories_created > 0 else (
                        "partial" if result.success else "failure"
                    )
                    attribution_ledger.record_outcome(_root_eid, _immediate_outcome, build_outcome_data(
                        confidence=0.6,
                        latency_s=round(time.time() - start_ts, 2),
                        source="system_metric",
                        tier="immediate",
                        scope="autonomy_policy",
                        blame_target="intent_selection",
                        findings=len(result.findings),
                        memories_created=memories_created,
                        tool_used=result.tool_used,
                        intent_id=_intent_trace["intent_id"],
                        goal_id=_intent_trace["goal_id"],
                        task_id=_intent_trace["task_id"],
                        golden_trace_id=_intent_trace["golden_trace_id"],
                        golden_command_id=_intent_trace["golden_command_id"],
                    ))
            except Exception:
                pass

            # Phase 5.2: extract candidate interventions from research results
            try:
                from autonomy.intervention_runner import get_intervention_runner
                from autonomy.interventions import CandidateIntervention, make_no_action
                runner = get_intervention_runner()
                interventions = self._extract_interventions(intent, result, memories_created)
                if not interventions:
                    source_ids = self._integrator.get_source_ids_for_intent(intent.id, limit=5)
                    if not source_ids:
                        source_ids = [intent.id]
                    interventions = [make_no_action(
                        trigger_deficit=intent.source_event.split(":")[0] if intent.source_event else "",
                        source_ids=source_ids,
                        evidence_summary=f"Research on '{intent.question[:60]}' produced {memories_created} memories but no actionable changes.",
                        falsifier="future deficit recurrence would justify revisiting",
                    )]
                for iv in interventions:
                    runner.propose(iv)
            except Exception:
                pass

            if memories_created > 0:
                logger.info(
                    "Autonomous learning: '%s' → %d findings, %d memories via %s",
                    intent.question[:50], len(result.findings), memories_created, result.tool_used,
                )

            # Notify gestation manager if active
            if intent.source_event and intent.source_event.startswith("gestation:"):
                self._notify_gestation_complete(intent.id, result, intent.source_event)

            # Immediate goal-task closure so dispatch unblocks without waiting
            # for the 600s delta measurement window.
            # execution_ok=True (no exception), worked = produced useful output.
            self._notify_goal_immediate(
                intent,
                execution_ok=True,
                worked=result.success and memories_created > 0,
                summary=intent.question[:80],
            )

            self._record_drive_soft_outcome(intent, result, memories_created)

        except Exception as exc:
            self._governor.record_complete(intent)
            self._delta_tracker.mark_completed(intent.id)
            intent.status = "failed"
            try:
                self._queue.remove(intent)
            except ValueError:
                pass
            self._completed.append(intent)
            self._intent_metadata[intent.id] = {
                "tool_used": "failed",
                "tag_cluster": tuple(intent.tag_cluster) if intent.tag_cluster else (),
                "source_event": intent.source_event or "",
                "question": intent.question[:80],
                "goal_id": intent.goal_id or "",
                "task_id": intent.task_id or "",
                "golden_trace_id": intent.golden_trace_id or "",
                "golden_command_id": intent.golden_command_id or "",
                "golden_status": intent.golden_status or "none",
            }
            self._prune_intent_metadata()
            self._record_episode(intent, "allowed", f"failed:{exc}", time.time() - start_ts)
            logger.warning("Autonomous research failed: %s — %s", intent.question[:40], exc)
            self._emit_event(
                AUTONOMY_RESEARCH_FAILED,
                intent_id=intent.id,
                goal_id=str(getattr(intent, "goal_id", "")),
                task_id=str(getattr(intent, "task_id", "")),
                golden_trace_id=str(getattr(intent, "golden_trace_id", "")),
                golden_command_id=str(getattr(intent, "golden_command_id", "")),
                error=str(exc)[:100],
            )
            try:
                from cognition.intention_registry import intention_registry
                intention_registry.resolve(
                    backing_job_id=intent.id,
                    outcome="failed",
                    reason=f"research_exception: {type(exc).__name__}: {str(exc)[:140]}",
                )
            except Exception:
                logger.debug("intention_registry.resolve (autonomy failed) failed", exc_info=True)
            # Execution failed — task gets status=failed, effect=inconclusive
            self._notify_goal_immediate(
                intent, execution_ok=False, worked=False,
                summary=f"failed: {exc}"[:80],
            )
            self._record_drive_soft_outcome(intent, None, 0, failed=True)

    # -- metadata housekeeping --------------------------------------------------

    _METADATA_PRUNE_EVERY = 20

    def _prune_intent_metadata(self) -> None:
        """Remove metadata/ledger entries for intents evicted from _completed."""
        self._metadata_prune_counter += 1
        if self._metadata_prune_counter < self._METADATA_PRUNE_EVERY:
            return
        self._metadata_prune_counter = 0

        live_ids = {i.id for i in self._queue}
        live_ids.update(i.id for i in self._completed)
        stale_meta = [k for k in self._intent_metadata if k not in live_ids]
        stale_ledger = [k for k in self._intent_ledger_ids if k not in live_ids]
        for k in stale_meta:
            del self._intent_metadata[k]
        for k in stale_ledger:
            del self._intent_ledger_ids[k]
        if stale_meta or stale_ledger:
            logger.debug("Pruned intent metadata: %d meta, %d ledger entries",
                         len(stale_meta), len(stale_ledger))

    # -- credit assignment + policy memory ------------------------------------

    def _process_delta_outcome(self, dr: Any) -> None:
        """When a delta result is measured, record in policy memory + calibrator."""
        if dr.status not in ("measured", "interrupted_by_restart"):
            return

        if dr.status == "interrupted_by_restart":
            logger.info("Delta attribution interrupted by restart: intent=%s", dr.intent_id)
            return

        meta = self._intent_metadata.pop(dr.intent_id, None)

        completed_intent = None
        for c in self._completed:
            if c.id == dr.intent_id:
                completed_intent = c
                break

        net = dr.net_attribution if dr.net_attribution != 0.0 else dr.net_improvement
        worked = net > MIN_MEANINGFUL_DELTA and dr.stable

        # Knowledge-aware signal: research that created memories is valuable
        # even when system health metrics don't visibly change within the
        # 600s measurement window.  Memories/beliefs are the primary output
        # of research — the delta tracker's 8 health metrics don't capture
        # knowledge gains.
        if not worked and meta and meta.get("immediate_success"):
            worked = True
            net = max(net, MIN_MEANINGFUL_DELTA * 2)

        if meta:
            intent_type = meta["source_event"]
            tool_used = meta["tool_used"]
            topic_tags = meta["tag_cluster"]
        elif completed_intent:
            intent_type = completed_intent.source_event or ""
            tool_used = (completed_intent.result.tool_used
                         if completed_intent.result else "")
            topic_tags = completed_intent.tag_cluster or ()
        else:
            intent_type = ""
            tool_used = ""
            topic_tags = ()

        question_summary = ""
        if meta:
            question_summary = meta.get("question", "")
        elif completed_intent:
            question_summary = completed_intent.question[:80]

        goal_id = ""
        task_id = ""
        golden_trace_id = ""
        golden_command_id = ""
        if meta:
            goal_id = str(meta.get("goal_id", "") or "")
            task_id = str(meta.get("task_id", "") or "")
            golden_trace_id = str(meta.get("golden_trace_id", "") or "")
            golden_command_id = str(meta.get("golden_command_id", "") or "")
        elif completed_intent:
            goal_id = str(getattr(completed_intent, "goal_id", "") or "")
            task_id = str(getattr(completed_intent, "task_id", "") or "")
            golden_trace_id = str(getattr(completed_intent, "golden_trace_id", "") or "")
            golden_command_id = str(getattr(completed_intent, "golden_command_id", "") or "")

        outcome = PolicyOutcome(
            intent_id=dr.intent_id,
            intent_type=intent_type,
            tool_used=tool_used,
            topic_tags=topic_tags,
            question_summary=question_summary,
            net_delta=net,
            stable=dr.stable,
            confidence=float(dr.sample_count) / 10.0,
            worked=worked,
        )
        self._policy_memory.record_outcome(outcome)

        _started_eid = self._intent_ledger_ids.pop(dr.intent_id, "")
        if _started_eid:
            try:
                from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
                _delta_outcome = "success" if worked else ("regressed" if net < -MIN_MEANINGFUL_DELTA else "inconclusive")
                attribution_ledger.record_outcome(_started_eid, _delta_outcome, build_outcome_data(
                    confidence=min(float(dr.sample_count) / 10.0, 1.0),
                    latency_s=round(getattr(dr, 'elapsed_s', 600.0), 2),
                    source="delta_tracker",
                    tier="delayed",
                    scope="autonomy_policy",
                    blame_target="intent_selection" if not worked else "general",
                    net_attribution=round(net, 4),
                    net_improvement=round(dr.net_improvement, 4),
                    stable=dr.stable,
                    sample_count=dr.sample_count,
                    intent_id=dr.intent_id,
                    goal_id=goal_id,
                    task_id=task_id,
                    golden_trace_id=golden_trace_id,
                    golden_command_id=golden_command_id,
                ))
            except Exception:
                pass

        metric_name = ""
        if intent_type and ":" in intent_type:
            metric_name = intent_type.split(":", 1)[1]
        import hashlib as _hl
        cluster_hash = _hl.md5(
            "|".join(sorted(topic_tags)).encode()
        ).hexdigest()[:8] if topic_tags else ""

        self._calibrator.record_outcome(
            intent_id=dr.intent_id,
            metric=metric_name,
            tool=tool_used,
            cluster_hash=cluster_hash,
            delta=net,
            is_win=worked,
            warmup=False,
            stable=dr.stable,
            errored=(dr.status == "error"),
            sample_count=dr.sample_count,
        )

        # Hard (authoritative) outcome to drive manager from measured delta.
        # Fallback chain: completed_intent → meta → delta result (persisted
        # across restarts) → intent_type (derived from meta/intent above).
        if self._drive_manager:
            se = ""
            if completed_intent:
                se = completed_intent.source_event or ""
            if not se and meta:
                se = meta.get("source_event", "")
            if not se:
                se = getattr(dr, "source_event", "") or intent_type
            if se.startswith("drive:"):
                drive_type = se.split(":", 1)[1]
                self._drive_manager.record_outcome(drive_type, worked)
                logger.info("Drive outcome: %s worked=%s (net=%.4f stable=%s)",
                            drive_type, worked, net, dr.stable)

        # Forward refined delta metrics to Goal Continuity Layer if goal-linked.
        # The task lifecycle was already closed by _notify_goal_immediate();
        # this second call is idempotent and only enriches with delta data.
        if completed_intent and self._goal_callback:
            g_id = getattr(completed_intent, "goal_id", "")
            t_id = getattr(completed_intent, "task_id", "")
            if g_id:
                try:
                    self._goal_callback(
                        g_id, t_id, dr.intent_id,
                        {"worked": worked, "net_delta": net, "stable": dr.stable,
                         "summary": outcome.question_summary},
                    )
                except Exception:
                    logger.debug("Goal callback error (delta)", exc_info=True)

    def _notify_goal_immediate(
        self, intent: ResearchIntent, *,
        execution_ok: bool, worked: bool, summary: str,
    ) -> None:
        """Close goal task lifecycle immediately after research finishes.

        Args:
            execution_ok: True if the task ran without exceptions.
            worked: True if the task produced useful output for the goal.

        The deferred delta callback in _process_delta_outcome() will fire later
        with refined metrics; record_task_outcome() is idempotent so the second
        call only enriches — it won't double-count tasks_attempted.
        """
        if not self._goal_callback:
            return
        g_id = getattr(intent, "goal_id", "")
        t_id = getattr(intent, "task_id", "")
        if not g_id:
            return
        try:
            self._goal_callback(
                g_id, t_id, intent.id,
                {"execution_ok": execution_ok, "worked": worked,
                 "net_delta": 0.0, "stable": False, "summary": summary},
            )
            logger.info(
                "Goal task closed immediately: goal=%s task=%s exec=%s worked=%s",
                g_id, t_id, execution_ok, worked,
            )
        except Exception:
            logger.debug("Goal callback error (immediate)", exc_info=True)

    def _extract_interventions(
        self, intent: ResearchIntent, result: Any, memories_created: int,
    ) -> list[Any]:
        """Lightweight extraction of candidate interventions from research.

        Builds candidate summaries and queues them — does not perform heavy
        analysis inline. Returns empty list if no actionable changes found.
        """
        from autonomy.interventions import CandidateIntervention
        interventions: list[CandidateIntervention] = []

        if not result or not result.success or not result.findings:
            return interventions

        trigger = intent.source_event.split(":")[0] if intent.source_event else ""
        question_lower = (intent.question or "").lower()
        source_ids = self._integrator.get_source_ids_for_intent(intent.id, limit=5)
        if not source_ids:
            source_ids = [intent.id]
        seen_interventions: set[tuple[str, str]] = set()

        for finding in result.findings[:5]:
            claims = []
            if hasattr(finding, "claims") and finding.claims:
                claims = finding.claims[:3]
            elif hasattr(finding, "content") and finding.content:
                claims = [finding.content[:200]]

            for claim in claims:
                claim_lower = (claim or "").lower()
                candidates: list[tuple[str, str, str, str, str]] = []
                inferred_subsystem = self._infer_subsystem(intent)
                inferred_metric = self._infer_metric(intent)
                inferred_direction = self._infer_direction(inferred_metric)

                if any(kw in claim_lower for kw in ("threshold", "parameter", "tuning", "adjust", "calibrate")):
                    candidates.append((
                        "threshold_change",
                        inferred_subsystem,
                        inferred_metric,
                        inferred_direction,
                        "Tune threshold/parameter for the target subsystem",
                    ))
                if any(kw in claim_lower for kw in (
                    "route", "routing", "intent", "disambigu", "classifier", "follow-up", "anaphora",
                )):
                    candidates.append((
                        "routing_rule",
                        "routing",
                        "friction_rate",
                        "down",
                        "Add or tighten routing disambiguation guidance",
                    ))
                if any(kw in claim_lower for kw in (
                    "prompt", "response format", "verbosity", "concise", "clarity", "tone", "instruction",
                )):
                    candidates.append((
                        "prompt_frame",
                        "conversation",
                        "friction_rate",
                        "down",
                        "Refine response framing and style constraints",
                    ))
                if any(kw in claim_lower for kw in (
                    "retrieval", "ranker", "salience", "memory weight", "decay", "association",
                )):
                    candidates.append((
                        "memory_weighting_rule",
                        "memory",
                        "retrieval_hit_rate",
                        "up",
                        "Adjust memory weighting/retrieval heuristics",
                    ))
                if any(kw in claim_lower for kw in (
                    "contract", "validation", "coverage", "assert", "regression test",
                )):
                    candidates.append((
                        "eval_contract",
                        "eval",
                        "processing_health",
                        "up",
                        "Add/adjust eval contract coverage for the surfaced weakness",
                    ))

                for change_type, subsystem, metric, direction, plan in candidates:
                    key = (change_type, subsystem)
                    if key in seen_interventions:
                        continue
                    interventions.append(CandidateIntervention(
                        change_type=change_type,
                        target_subsystem=subsystem,
                        target_symbol="",
                        trigger_deficit=trigger,
                        source_ids=source_ids,
                        evidence_summary=claim[:200],
                        proposed_change=f"{plan}: {claim[:100]}",
                        expected_metric=metric,
                        expected_direction=direction,
                        falsifier="no improvement in target metric after shadow window",
                        risk_level="low",
                    ))
                    seen_interventions.add(key)
                    if len(interventions) >= 3:
                        break
                if len(interventions) >= 3:
                    break
            if len(interventions) >= 3:
                break

        # Fallback: some budgeted research results are semantically relevant but
        # too terse for keyword extraction; use intent context conservatively.
        if not interventions and trigger == "metric":
            if any(kw in question_lower for kw in (
                "friction", "correction", "rephrase", "dissatisfaction", "follow-up",
            )):
                interventions.append(CandidateIntervention(
                    change_type="routing_rule",
                    target_subsystem="routing",
                    target_symbol="",
                    trigger_deficit=trigger,
                    source_ids=source_ids,
                    evidence_summary=f"Intent-level friction signal: {intent.question[:180]}",
                    proposed_change="Tighten routing disambiguation for follow-up and correction patterns.",
                    expected_metric="friction_rate",
                    expected_direction="down",
                    falsifier="no reduction in friction_rate after shadow window",
                    risk_level="low",
                ))
                interventions.append(CandidateIntervention(
                    change_type="prompt_frame",
                    target_subsystem="conversation",
                    target_symbol="",
                    trigger_deficit=trigger,
                    source_ids=source_ids,
                    evidence_summary=f"Intent-level friction signal: {intent.question[:180]}",
                    proposed_change="Constrain response framing to concise corrective format for friction-heavy turns.",
                    expected_metric="friction_rate",
                    expected_direction="down",
                    falsifier="no reduction in correction/rephrase friction events after shadow window",
                    risk_level="low",
                ))

        return interventions[:3]

    @staticmethod
    def _infer_subsystem(intent: ResearchIntent) -> str:
        tags = set(intent.tag_cluster) if intent.tag_cluster else set()
        q = intent.question.lower()
        if "memory" in q or "memory" in tags or "recall" in tags:
            return "memory"
        if "calibration" in q or "calibration" in tags or "truth" in tags:
            return "calibration"
        if "routing" in q or "route" in tags:
            return "routing"
        if "world" in q or "simulator" in tags or "scene" in tags:
            return "world_model"
        if "autonomy" in q or "curiosity" in tags:
            return "autonomy"
        return "conversation"

    @staticmethod
    def _infer_metric(intent: ResearchIntent) -> str:
        q = intent.question.lower()
        if "memory" in q or "recall" in q:
            return "retrieval_hit_rate"
        if "calibration" in q or "truth" in q:
            return "contradiction_resolution_rate"
        if "friction" in q or "conversation" in q:
            return "friction_rate"
        return "processing_health"

    @staticmethod
    def _infer_direction(metric_name: str) -> str:
        lower = (metric_name or "").lower()
        if lower in {"friction_rate", "tick_p95_ms", "contradiction_debt"}:
            return "down"
        return "up"

    def _run_eval_replay(self) -> None:
        """Automatic policy comparison during dream/reflection cycles.

        Loads recent episodes, compares current scorer against a baseline
        heuristic, and persists comparison results.
        """
        try:
            from autonomy.eval_harness import compare_policies, TraceEpisode

            episodes = self._episode_recorder.load_recent(200)
            if len(episodes) < 5:
                return

            def current_scorer(ep: TraceEpisode) -> float:
                return ep.score_breakdown.get("total", 0.0)

            def baseline_scorer(ep: TraceEpisode) -> float:
                s = ep.score_breakdown.get("total", 0.0)
                return max(0.0, s * 0.9)

            report = compare_policies(
                episodes, current_scorer, baseline_scorer,
                name_a="current", name_b="baseline_heuristic",
            )

            comparison = {
                "timestamp": time.time(),
                "episodes_compared": report.episodes_compared,
                "current_avg_score": report.a_total_score,
                "baseline_avg_score": report.b_total_score,
                "current_would_execute": report.a_would_execute,
                "baseline_would_execute": report.b_would_execute,
                "agreement_rate": report.agreement_rate,
                "current_predicted_delta": round(report.a_predicted_delta, 4),
                "baseline_predicted_delta": round(report.b_predicted_delta, 4),
            }

            import json as _json
            comp_path = os.path.join(os.path.expanduser("~/.jarvis"), "eval_comparisons.jsonl")
            os.makedirs(os.path.dirname(comp_path), exist_ok=True)
            with open(comp_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(comparison, default=str) + "\n")

            logger.info(
                "Eval replay: %d episodes, current=%.3f baseline=%.3f agreement=%.1f%%",
                report.episodes_compared, report.a_total_score,
                report.b_total_score, report.agreement_rate * 100,
            )
        except Exception:
            logger.warning("Eval replay failed", exc_info=True)

    def _route_to_self_improve(self, intervention: Any) -> bool:
        """L2 bridge — routes code-changing interventions to self-improve.

        Stage-aware: consults the self-improvement orchestrator's stage
        rather than reading FREEZE_AUTO_IMPROVE from os.environ.  Requires
        stage >= 2 (human-approval) to allow routing.
        """
        if not hasattr(intervention, "change_type"):
            return False
        if intervention.change_type != "code_patch":
            return False
        if self._autonomy_level < 2:
            logger.debug("L2 bridge: autonomy level %d < 2, skipping", self._autonomy_level)
            return False

        si_orch = getattr(self, "_self_improve_orchestrator", None)
        if si_orch is None:
            logger.info("L2 bridge: no self-improve orchestrator available")
            return False
        si_status = si_orch.get_status() if hasattr(si_orch, "get_status") else {}
        si_stage = si_status.get("stage", 0)
        if si_stage < 2:
            logger.info("L2 bridge: self-improve stage=%d < 2, skipping code_patch intervention", si_stage)
            return False

        logger.info("L2 bridge: would route intervention %s to self_improve (not yet active)",
                     getattr(intervention, "intervention_id", "?"))
        return False

    def _record_drive_soft_outcome(
        self, intent: ResearchIntent, result: Any,
        memories_created: int, *,
        failed: bool = False,
        already_known: bool = False,
    ) -> None:
        """Track relevance saturation immediately after intent execution.

        Drive outcome accounting is handled exclusively by
        _process_delta_outcome() after the delta measurement window.
        This method only tracks topic saturation for the relevance drive.
        """
        if not self._drive_manager:
            return
        se = intent.source_event or ""
        if not se.startswith("drive:"):
            return

        if failed or already_known:
            self._track_relevance_saturation(intent, has_findings=False)
            return
        if result is None:
            return
        has_findings = (memories_created > 0 or
                        (hasattr(result, 'findings') and bool(result.findings)))
        if not has_findings:
            self._track_relevance_saturation(intent, has_findings=False)
        else:
            self._track_relevance_saturation(intent, has_findings=True)

    def _track_relevance_saturation(self, intent: ResearchIntent, *, has_findings: bool) -> None:
        """Track topic saturation for relevance drive recall actions."""
        se = intent.source_event or ""
        if "relevance" not in se:
            return
        topic = intent.question.replace("Recall: ", "").strip().lower()
        if not topic:
            return
        if has_findings:
            self._topic_recall_misses.pop(topic, None)
            self._saturated_topics.discard(topic)
        else:
            misses = self._topic_recall_misses.get(topic, 0) + 1
            self._topic_recall_misses[topic] = misses
            if misses >= 3:
                self._saturated_topics.add(topic)
                logger.info("Topic saturated after %d misses: '%s'", misses, topic[:40])

    def _clear_stale_saturation(self) -> None:
        """Clear topic saturation after 1 hour."""
        now = time.time()
        if now - self._last_saturation_clear > 3600:
            self._saturated_topics.clear()
            self._topic_recall_misses.clear()
            self._last_saturation_clear = now

    def _record_episode(
        self, intent: ResearchIntent, gov_decision: str, exec_result: str,
        duration_s: float,
    ) -> None:
        """Record a trace episode for offline replay."""
        try:
            episode = TraceEpisode(
                episode_id=intent.id,
                intent_id=intent.id,
                question=intent.question,
                source_event=intent.source_event,
                tool_hint=intent.source_hint,
                tag_cluster=intent.tag_cluster,
                trigger_count=intent.trigger_count,
                scope=intent.scope,
                governor_decision=gov_decision,
                execution_result=exec_result,
                autonomy_level=self._autonomy_level,
                duration_s=duration_s,
            )
            self._episode_recorder.record(episode)
        except Exception:
            pass

    def check_promotion_eligibility(self) -> dict[str, Any]:
        """Check if autonomy has earned promotion to a higher level.

        L1→L2: requires l2_required_positive_deltas positive attributions
                with win_rate >= l2_min_win_rate.
        L2→L3: requires l3_required_positive_deltas + l3_min_win_rate,
                plus no regressions in last 10 jobs.
        """
        stats = self._policy_memory.get_stats()
        total = stats["total_outcomes"]
        wins = stats["total_wins"]
        win_rate = stats["overall_win_rate"]

        recent_outcomes = list(self._policy_memory._outcomes)[-10:]
        recent_regressions = sum(
            1 for o in recent_outcomes if o.net_delta < -MIN_MEANINGFUL_DELTA
        )

        result: dict[str, Any] = {
            "current_level": self._autonomy_level,
            "total_outcomes": total,
            "wins": wins,
            "win_rate": round(win_rate, 3),
            "recent_regressions": recent_regressions,
            "eligible_for_l2": False,
            "eligible_for_l3": False,
            "l2_reason": "",
            "l3_reason": "",
        }

        try:
            from config import BrainConfig
            cfg = BrainConfig()
            l2_req = cfg.autonomy.l2_required_positive_deltas
            l2_wr = cfg.autonomy.l2_min_win_rate
            l3_req = cfg.autonomy.l3_required_positive_deltas
            l3_wr = cfg.autonomy.l3_min_win_rate
        except Exception:
            l2_req, l2_wr = 10, 0.4
            l3_req, l3_wr = 25, 0.5

        if wins >= l2_req and win_rate >= l2_wr:
            result["eligible_for_l2"] = True
            result["l2_reason"] = f"Earned: {wins}/{l2_req} wins, {win_rate:.0%}/{l2_wr:.0%} rate"
        else:
            result["l2_reason"] = (
                f"Need {l2_req} wins (have {wins}) "
                f"and {l2_wr:.0%} win rate (have {win_rate:.0%})"
            )

        if wins >= l3_req and win_rate >= l3_wr:
            if recent_regressions == 0:
                result["eligible_for_l3"] = True
                result["l3_reason"] = (
                    f"Earned: {wins}/{l3_req} wins, {win_rate:.0%}/{l3_wr:.0%} rate, "
                    f"0 regressions in last 10"
                )
            else:
                result["l3_reason"] = (
                    f"Meets thresholds but {recent_regressions} regressions in last 10 jobs"
                )
        else:
            result["l3_reason"] = (
                f"Need {l3_req} wins (have {wins}) "
                f"and {l3_wr:.0%} win rate (have {win_rate:.0%})"
            )

        return result

    # -- metric feed ----------------------------------------------------------

    def _feed_metrics(self) -> None:
        """Collect current system metrics and feed them to scorer + triggers."""
        metrics = self._collect_metrics()
        if not metrics:
            return

        self._scorer.record_metrics(metrics)

        snapshot = MetricSnapshot(
            timestamp=time.time(),
            confidence_avg=metrics.get("confidence_avg", 0.0),
            confidence_volatility=metrics.get("confidence_volatility", 0.0),
            tick_p95_ms=metrics.get("tick_p95_ms", 0.0),
            reasoning_coherence=metrics.get("reasoning_coherence", 0.0),
            processing_health=metrics.get("processing_health", 0.0),
            memory_count=int(metrics.get("memory_count", 0)),
            barge_in_count=int(metrics.get("barge_in_count", 0)),
            error_count=int(metrics.get("error_count", 0)),
            retrieval_hit_rate=metrics.get("retrieval_hit_rate", 0.0),
            belief_graph_coverage=metrics.get("belief_graph_coverage", 0.0),
            contradiction_resolution_rate=metrics.get("contradiction_resolution_rate", 0.0),
            friction_rate=metrics.get("friction_rate", 0.0),
        )
        self._delta_tracker.record_metrics(snapshot)
        self._metric_history.record(metrics)

        if self._autonomy_level >= 1:
            self._metric_triggers.evaluate(metrics, self.enqueue)

        if self._autonomy_level >= 3:
            self._evaluate_l3_escalations()

    def _evaluate_l3_escalations(self) -> None:
        """Phase 6.5: auto-generate L3 escalation requests from sustained deficits.

        Called on every tick when live autonomy level is >= 3. Consults
        ``MetricTriggers.get_escalation_candidates`` (which gates on
        deficit duration, L1 attempts, win rate, policy veto) and
        submits each candidate through the escalation store. The store
        itself enforces per-metric rate limiting, duplicate pending
        protection, and policy validation, so all gates are honored
        even if this method is called on every tick.
        """
        if self._autonomy_level < 3:
            return
        try:
            candidates = self._metric_triggers.get_escalation_candidates(
                live_autonomy_level=self._autonomy_level,
            )
        except Exception:
            now = time.time()
            if now - self._escalation_wire_last_error_log_ts > 60.0:
                logger.exception("get_escalation_candidates failed")
                self._escalation_wire_last_error_log_ts = now
            return

        if not candidates:
            return

        try:
            from autonomy.escalation import (
                EscalationStore,
                EscalationStoreError,
                submit_and_emit,
            )
        except Exception:
            logger.exception("Failed to import escalation module")
            return

        if self._escalation_store is None:
            try:
                self._escalation_store = EscalationStore()
            except Exception:
                logger.exception("EscalationStore init failed")
                return

        for req in candidates:
            try:
                submit_and_emit(
                    self._escalation_store,
                    req,
                    event_emit=lambda et, **kw: self._emit_event(et, **kw),
                )
            except EscalationStoreError as exc:
                logger.debug("Escalation submit rejected: %s", exc)
            except Exception:
                logger.exception("Escalation submit failed for metric %s", req.metric)

    def _collect_metrics(self) -> dict[str, float]:
        """Pull current metrics from engine subsystems. Best-effort, never crash."""
        engine = self._engine_ref
        if not engine:
            return {}

        m: dict[str, float] = {}
        try:
            cs = engine.consciousness
            if cs:
                analytics = cs.analytics
                conf = analytics.get_confidence()
                reasoning = analytics.get_reasoning_quality()
                health = analytics.get_system_health()
                m["confidence_avg"] = conf.avg
                m["confidence_volatility"] = conf.volatility
                m["tick_p95_ms"] = health.tick_p95_ms
                m["reasoning_coherence"] = reasoning.coherence
                m["memory_count"] = health.memory_count

                ch = analytics.get_health_report()
                m["processing_health"] = ch.get("components", {}).get("processing_health", 0.5)
        except Exception:
            pass

        try:
            from policy.telemetry import policy_telemetry
            snap = policy_telemetry.snapshot()
            total = snap.get("shadow_ab_total", 0)
            if total > 0:
                wins = snap.get("shadow_nn_wins", 0)
                m["shadow_default_win_rate"] = wins / total
        except Exception:
            pass

        try:
            from dashboard.app import _health
            if _health:
                hsnap = _health.snapshot()
                m["barge_in_count"] = hsnap.get("barge_in_count", 0)
                m["error_count"] = hsnap.get("error_count", 0)
                resp_count = hsnap.get("response_count", 0)
                if resp_count > 0:
                    m["barge_in_rate"] = hsnap.get("barge_in_count", 0) / resp_count
        except Exception:
            pass

        try:
            from autonomy.friction_miner import get_friction_miner
            m["friction_rate"] = get_friction_miner().get_friction_rate(3600)
        except Exception:
            pass

        try:
            from memory.retrieval_log import memory_retrieval_log
            eval_metrics = memory_retrieval_log.get_eval_metrics(window=100)
            lift = eval_metrics.get("lift")
            if lift is not None:
                m["retrieval_hit_rate"] = max(0.0, min(1.0, 0.5 + lift))
        except Exception:
            pass

        try:
            cs = engine.consciousness if engine else None
            ce = getattr(cs, "_contradiction_engine", None) if cs else None
            if ce:
                debt_stats = ce.get_stats() if hasattr(ce, "get_stats") else {}
                resolved = debt_stats.get("resolved_count", 0)
                open_debt = debt_stats.get("open_debt_count", debt_stats.get("debt_count", 0))
                total = resolved + open_debt
                if total > 0:
                    m["contradiction_resolution_rate"] = resolved / total
        except Exception:
            pass

        try:
            cs = engine.consciousness if engine else None
            bg = getattr(cs, "_belief_graph", None) if cs else None
            if bg and hasattr(bg, "_edge_store"):
                from memory.storage import memory_storage
                recent = memory_storage.get_recent(100)
                if recent:
                    with_edge = 0
                    for mem in recent:
                        mid = getattr(mem, "id", "")
                        if mid and hasattr(bg, "_edge_store"):
                            edges = bg._edge_store.get_edges_for(mid) if hasattr(bg._edge_store, "get_edges_for") else []
                            if edges:
                                with_edge += 1
                    m["belief_graph_coverage"] = with_edge / len(recent)
        except Exception:
            pass

        return m

    # -- drive-based exploration ---------------------------------------------

    @staticmethod
    def _normalize_question(q: str) -> str:
        """Normalize a question for dedup: lowercase, strip punctuation, collapse whitespace."""
        import re as _re
        return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]", "", q.lower())).strip()[:120]

    def _question_already_queued(self, question: str) -> bool:
        """Check if a similar question is already in recent autonomy history.

        Sources consulted:
        - active queue
        - recent in-memory completions
        - recent persisted trace episodes on disk (survives restart)
        """
        norm = self._normalize_question(question)
        if not norm:
            return False
        for intent in self._queue:
            if self._normalize_question(intent.question) == norm:
                return True
        for intent in list(self._completed)[-10:]:
            if self._normalize_question(intent.question) == norm:
                return True
        try:
            now = time.time()
            for ep in reversed(self._episode_recorder.load_recent(QUESTION_DEDUP_EPISODE_LOOKBACK)):
                if self._normalize_question(ep.question) != norm:
                    continue
                age_s = max(0.0, now - float(getattr(ep, "timestamp", 0.0) or 0.0))
                if age_s <= QUESTION_DEDUP_COOLDOWN_S:
                    return True
        except Exception:
            pass
        return False

    def _get_recent_completed_unique(self, limit: int = 10) -> list[ResearchIntent]:
        """Return most recent completed intents with duplicate questions collapsed."""
        unique: list[ResearchIntent] = []
        seen: set[str] = set()
        for intent in reversed(list(self._completed)):
            norm = self._normalize_question(intent.question)
            key = norm or intent.id
            if key in seen:
                continue
            unique.append(intent)
            seen.add(key)
            if len(unique) >= limit:
                break
        return list(reversed(unique))

    def _evaluate_drives(self) -> None:
        """Run the drive manager and enqueue the top drive's action as an intent."""
        if not self._drive_manager:
            return
        try:
            self._clear_stale_saturation()
            signals = self._collect_drive_signals()
            drives = self._drive_manager.evaluate(signals)
            if not drives:
                return
            action = self._drive_manager.select_action(drives[0], signals)
            if action is None:
                return
            if action.action_type == "noop":
                return

            if action.action_type == "learn":
                self._handle_learn_action(action)
                return

            if action.action_type == "recall" and self._current_mode in ("conversational", "focused"):
                logger.debug("Suppressing drive recall '%s' during %s mode — "
                             "conversation retrieval handles this",
                             action.drive_type, self._current_mode)
                return

            if action.action_type == "recall":
                try:
                    from reasoning.response import get_last_memory_route
                    last_route = get_last_memory_route()
                    if last_route and last_route.route_type in ("referenced_person", "self_preference"):
                        logger.debug("Suppressing drive recall '%s' — active %s route handles this",
                                     action.drive_type, last_route.route_type)
                        return
                except Exception:
                    pass

            if self._question_already_queued(action.question):
                logger.debug("Drive '%s' skipped: duplicate question already queued/completed",
                             action.drive_type)
                self._drive_manager.record_outcome(action.drive_type, False)
                return

            intent = ResearchIntent(
                question=action.question,
                source_event=f"drive:{action.drive_type}",
                source_hint=action.tool_hint,
                scope=action.scope,
                tag_cluster=action.tags,
                priority=action.urgency,
            )
            if self.enqueue(intent):
                logger.info("Drive '%s' enqueued intent: %s (urgency=%.2f, action=%s)",
                            action.drive_type, action.question[:60],
                            action.urgency, action.action_type)
        except Exception:
            logger.warning("Drive evaluation failed", exc_info=True)

    def _handle_learn_action(self, action: Any) -> None:
        """Route a mastery-drive learn action to the LearningJobOrchestrator."""
        try:
            from autonomy.drives import _DEFICIT_CAPABILITY_MAP
            skill_tag = None
            for tag in (action.tags or ()):
                if tag.startswith("skill:"):
                    skill_tag = tag.split(":", 1)[1]
                    break
            if not skill_tag:
                logger.debug("Learn action has no skill tag — skipping")
                return

            mapping = None
            for _dk, m in _DEFICIT_CAPABILITY_MAP.items():
                if m["skill_id"] == skill_tag:
                    mapping = m
                    break
            if not mapping:
                logger.debug("No deficit mapping for skill %s — skipping", skill_tag)
                return

            try:
                from tools.skill_tool import _learning_job_orch, _skill_registry
                if _learning_job_orch is None or _skill_registry is None:
                    logger.debug("Learning system not initialized — cannot create job from mastery drive")
                    return

                existing = _skill_registry.get(mapping["skill_id"])
                if existing is None:
                    from skills.registry import SkillRecord
                    rec = SkillRecord(
                        skill_id=mapping["skill_id"],
                        name=mapping["description"],
                        status="unknown",
                        capability_type=mapping["capability_type"],
                    )
                    _skill_registry.register(rec)

                job = _learning_job_orch.create_job(
                    skill_id=mapping["skill_id"],
                    capability_type=mapping["capability_type"],
                    requested_by={"source": "mastery_drive", "action": action.to_dict()},
                    risk_level="low",
                    priority=action.urgency,
                )
                if job is not None:
                    job.matrix_protocol = True
                    job.protocol_id = mapping["protocol_id"]
                    job.matrix_target = mapping["description"]
                    job.verification_profile = mapping["protocol_id"]
                    job.events.append({
                        "ts": job.created_at,
                        "type": "matrix_protocol_activated",
                        "msg": f"Mastery drive auto-created Matrix job — "
                               f"protocol {mapping['protocol_id']}",
                    })
                    _learning_job_orch.store.save(job)
                    logger.info(
                        "Mastery drive created learning job %s for %s (protocol %s)",
                        job.job_id, mapping["skill_id"], mapping["protocol_id"],
                    )
                    self._drive_manager.record_outcome("mastery", True)
                else:
                    self._drive_manager.record_outcome("mastery", False)
            except Exception:
                logger.exception("Failed to create learning job from mastery drive")
                self._drive_manager.record_outcome("mastery", False)
        except Exception:
            logger.warning("_handle_learn_action failed", exc_info=True)

    def _collect_drive_signals(self) -> Any:
        """Build DriveSignals from current system state."""
        from autonomy.drives import DriveSignals

        signals = DriveSignals()

        # Gate blocks + reasons
        try:
            from skills.capability_gate import capability_gate
            stats = capability_gate.get_stats()
            signals.gate_blocks_recent = stats.get("claims_blocked", 0)
            signals.gate_block_reasons = stats.get("recent_block_reasons", [])
        except Exception:
            pass

        # Actual contradictions from Layer 5
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            ce = ContradictionEngine.get_instance()
            ce_state = ce.get_state()
            signals.contradictions = ce_state.get("resolved_count", 0)
        except Exception:
            pass

        # Metric deficits
        metrics = self._collect_metrics()
        deficits = {}
        try:
            from autonomy.metric_triggers import _TRIGGER_DEFS
            for name, val in metrics.items():
                trigger_def = _TRIGGER_DEFS.get(name)
                if trigger_def:
                    thresh = trigger_def.get("threshold", 0)
                    direction = trigger_def.get("direction", "above")
                    if direction == "above" and val > thresh:
                        deficits[name] = val - thresh
                    elif direction == "below" and val < thresh:
                        deficits[name] = thresh - val
        except Exception:
            pass

        # Inject cortex training readiness as a deficit for mastery drive
        try:
            from memory.ranker import MemoryRanker
            from memory.salience import SalienceModel
            ranker = MemoryRanker.get_instance() if hasattr(MemoryRanker, 'get_instance') else None
            salience = SalienceModel.get_instance() if hasattr(SalienceModel, 'get_instance') else None
            if ranker and not getattr(ranker, '_enabled', True):
                deficits["ranker_not_ready"] = 0.5
            if salience and not getattr(salience, '_enabled', True):
                deficits["salience_not_ready"] = 0.4
        except Exception:
            pass
        signals.metric_deficits = deficits

        signals.system_health = metrics.get("processing_health", 0.7)
        signals.tick_p95_ms = metrics.get("tick_p95_ms", 0)

        # Novelty from emergent behaviors via consciousness system
        try:
            engine = self._engine_ref
            if engine and hasattr(engine, 'consciousness') and engine.consciousness:
                cs = engine.consciousness
                if hasattr(cs, 'evolution'):
                    evo_state = cs.evolution.get_state()
                    signals.novelty_events = min(evo_state.total_emergent_count, 20)
        except Exception:
            pass

        signals.open_threads = len([i for i in self._queue if i.status == "queued"])
        signals.pending_jobs = signals.open_threads

        # Pending delta windows count as continuity pressure
        try:
            if self._delta_tracker:
                pending = len(getattr(self._delta_tracker, '_active_windows', {}))
                signals.open_threads += pending
                signals.pending_jobs += pending
        except Exception:
            pass

        try:
            signals.user_topics = self._collect_user_topics()
        except Exception:
            pass

        # Wire relevance drive success rate + saturated topics
        try:
            if self._drive_manager:
                rel_state = self._drive_manager._states.get("relevance")
                if rel_state:
                    signals.relevance_success_rate = rel_state.success_rate
                signals.saturated_topics = self._saturated_topics
        except Exception:
            pass

        # Surface recognition gap as mastery deficit
        try:
            from perception.identity_fusion import _active_instance as _fusion_instance
            if _fusion_instance:
                status = _fusion_instance.get_status()
                if status.get("user_present") and not status.get("is_known"):
                    rec_state = status.get("recognition_state", "")
                    if rec_state in ("unknown_present", "tentative_match"):
                        signals.metric_deficits["recognition_confidence"] = 0.5
        except Exception:
            pass

        # Active user goals count for curiosity dampening
        if self._goal_manager:
            try:
                signals.active_user_goals = self._goal_manager.get_active_user_goal_count()
            except Exception:
                pass

        return signals

    _TOPIC_STOPWORDS: frozenset[str] = frozenset({
        "about", "after", "again", "being", "could", "doing", "every",
        "going", "their", "there", "these", "thing", "think", "those",
        "would", "which", "where", "while", "right", "really", "should",
        "something", "because", "between", "through", "before",
        # Pronouns, determiners, prepositions, auxiliaries (4+ chars, pass isalpha)
        "your", "yours", "they", "them", "that", "this", "what", "when",
        "with", "from", "into", "have", "been", "were", "will", "does",
        "also", "just", "very", "much", "more", "some", "only", "each",
        "than", "then", "both", "many", "most", "such", "same", "well",
        "even", "over", "know", "tell", "told", "said", "want", "need",
        "come", "came", "make", "made", "take", "took", "give", "gave",
        # Schema / structural words from memory payloads
        "response", "complexity", "confidence", "speaker", "method",
        "latency", "outcome", "neutral", "completed", "unknown",
        "message", "payload", "status", "result", "timestamp",
        # Common conversational filler
        "actually", "currently", "working", "feeling", "looking",
        "trying", "pretty", "always", "still", "like",
    })

    _TOPIC_DISCARD_RE = re.compile(
        r"^(?:no[,.]?\s*|nah|nope|not really|okay|ok|sure|yes|yeah|yep|"
        r"hi|hello|hey|thanks|thank you|bye|goodbye|good night|"
        r"i(?:'m| am) (?:just|not|here)|forget it|never\s?mind|stop|"
        r"that'?s (?:fine|okay|good|it|all)|what(?:'s| is) (?:up|going on)|"
        r"how (?:are|about) you)",
        re.IGNORECASE,
    )

    def _collect_user_topics(self) -> dict[str, float]:
        """Extract recent user conversation topics as clean phrases.

        Returns a dict of topic_phrase -> weight (0..1), extracted from
        recent user messages using unigram and bigram counting with
        schema-word filtering. Discards negations, greetings, and
        low-semantic-value filler utterances.
        """
        try:
            from memory.storage import memory_storage
            recent = memory_storage.get_by_type("conversation")
            if not recent:
                return {}
            recent = recent[-20:]

            phrase_counts: dict[str, int] = {}
            for m in recent:
                text = ""
                if isinstance(m.payload, dict):
                    text = m.payload.get("user_message", "")
                elif isinstance(m.payload, str):
                    text = m.payload
                if not text or not isinstance(text, str):
                    continue
                if len(text) < 8 or self._TOPIC_DISCARD_RE.match(text.strip()):
                    continue

                words = [
                    w for w in text.lower().split()
                    if len(w) > 3 and w.isalpha()
                    and w not in self._TOPIC_STOPWORDS
                ]

                for w in words:
                    phrase_counts[w] = phrase_counts.get(w, 0) + 1
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 2

            if not phrase_counts:
                return {}
            # Absolute frequency weighting: 1 mention=0.2, 5+=1.0
            # Reject single-word topics shorter than 6 chars (low specificity)
            return {
                p: min(1.0, c * 0.2)
                for p, c in sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                if " " in p or len(p) >= 6
            }
        except Exception:
            return {}

    # -- introspection for dashboard -----------------------------------------

    def save_calibration(self) -> None:
        """Persist calibrator state, pending delta windows, drive state, and autonomy level."""
        try:
            self._calibrator.save()
        except Exception:
            logger.warning("Calibration save failed", exc_info=True)
        try:
            self._delta_tracker.save_pending()
        except Exception:
            logger.warning("Delta tracker save_pending failed", exc_info=True)
        try:
            self._delta_tracker.save_counters()
        except Exception:
            logger.warning("Delta tracker save_counters failed", exc_info=True)
        try:
            if self._drive_manager is not None:
                self._drive_manager.save_state()
        except Exception:
            logger.warning("Drive manager save failed", exc_info=True)
        self._save_autonomy_state()

    def _save_autonomy_state(self) -> None:
        """Persist autonomy level with full audit trail for boot reconciliation."""
        try:
            _JARVIS_DIR.mkdir(parents=True, exist_ok=True)
            stats = self._policy_memory.get_stats()
            delta_stats = self._delta_tracker.get_stats() if hasattr(self, "_delta_tracker") else {}
            state = {
                "autonomy_level": self._autonomy_level,
                "promoted_at": time.time(),
                "restored_from_policy_win_rate": round(stats.get("overall_win_rate", 0.0), 4),
                "restored_from_policy_outcomes": stats.get("total_outcomes", 0),
                "restored_from_delta_improvement_rate": round(delta_stats.get("improvement_rate", 0.0) or 0.0, 4),
            }
            tmp = _AUTONOMY_STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(_AUTONOMY_STATE_PATH)
        except Exception:
            logger.warning("Autonomy state save failed", exc_info=True)

    def get_status(self) -> dict[str, Any]:
        episode_stats = self._episode_recorder.get_stats()
        session_completed_total = len(self._completed)
        lifetime_completed_total = max(
            session_completed_total,
            int(episode_stats.get("successful_episodes", 0)),
        )
        intervention_stats: dict[str, Any] = {}
        source_ledger_stats: dict[str, Any] = {}
        friction_stats: dict[str, Any] = {}
        try:
            from autonomy.intervention_runner import get_intervention_runner
            runner = get_intervention_runner()
            intervention_stats = runner.get_stats()
            intervention_stats["recent"] = runner.get_recent(10)
        except Exception:
            pass
        try:
            from autonomy.source_ledger import get_source_ledger
            source_ledger = get_source_ledger()
            source_ledger_stats = source_ledger.get_stats()
            source_ledger_stats["recent"] = source_ledger.get_recent_records(10)
        except Exception:
            pass
        try:
            from autonomy.friction_miner import get_friction_miner
            friction = get_friction_miner()
            friction_stats = friction.get_stats()
            friction_stats["recent"] = friction.get_recent_events(10)
            friction_stats["active_cluster_rows"] = friction.get_active_clusters()[:10]
        except Exception:
            pass
        return {
            "enabled": self._enabled,
            "started": self._started,
            "autonomy_level": self._autonomy_level,
            "autonomy_level_name": AUTONOMY_LEVELS.get(self._autonomy_level, "unknown"),
            "queue": [i.to_dict() for i in self._queue],
            "queue_size": len(self._queue),
            "completed": [i.to_dict() for i in self._get_recent_completed_unique(10)],
            "completed_total": lifetime_completed_total,
            "completed_total_session": session_completed_total,
            "detector": self._detector.get_stats(),
            "governor": self._governor.get_stats(current_mode=self._current_mode),
            "query_interface": self._query_interface.get_stats(),
            "integrator": self._integrator.get_stats(),
            "bridge": self._bridge.get_stats(),
            "scorer": self._scorer.get_stats(),
            "metric_triggers": self._metric_triggers.get_stats(),
            "delta_tracker": self._delta_tracker.get_stats(),
            "recent_deltas": self._delta_tracker.get_recent_deltas(5),
            "metric_history": self._metric_history.get_coverage(),
            "policy_memory": self._policy_memory.get_stats(),
            "calibration": self._calibrator.get_readiness(),
            "promotion": self.check_promotion_eligibility(),
            "episode_recorder": episode_stats,
            "last_process_time": self._last_process_time,
            "recent_learnings": self._integrator.get_recent_learnings(5),
            "drives": self._drive_manager.get_status() if self._drive_manager else {},
            "interventions": intervention_stats,
            "source_ledger": source_ledger_stats,
            "friction": friction_stats,
            "planner_shadow_policy": {
                "enabled": True,
                "applied": False,
                "recent_previews": list(self._planner_shadow_previews)[-10:],
                "preview_count": len(self._planner_shadow_previews),
            },
        }

    def get_evidence_summary(self) -> str:
        return self._integrator.get_evidence_summary()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        logger.info("Autonomy %s", "enabled" if enabled else "disabled")

    # -- helpers -------------------------------------------------------------

    def _notify_gestation_complete(self, intent_id: str, result: Any,
                                    source_event: str = "") -> None:
        """Notify the gestation manager that one of its directives completed."""
        try:
            if self._engine_ref and hasattr(self._engine_ref, '_gestation_manager'):
                gm = self._engine_ref._gestation_manager
                if gm and gm.is_active:
                    gm.on_directive_completed(intent_id, result, source_event=source_event)
        except Exception as exc:
            logger.debug("Gestation notification failed: %s", exc)

    def _requeue_gestation_directive(self, intent: Any) -> None:
        """Re-queue a blocked gestation directive so it can be retried later."""
        source_event = getattr(intent, "source_event", "") or ""
        if not source_event.startswith("gestation:"):
            return
        try:
            if self._engine_ref and hasattr(self._engine_ref, '_gestation_manager'):
                gm = self._engine_ref._gestation_manager
                if gm and gm.is_active:
                    gm.on_directive_blocked(
                        intent_id=intent.id,
                        question=intent.question,
                        source_event=source_event,
                        tool_hint=getattr(intent, "source_hint", "any"),
                        priority=getattr(intent, "priority", 50),
                        tag_cluster=getattr(intent, "tag_cluster", set()),
                    )
        except Exception as exc:
            logger.debug("Gestation requeue failed: %s", exc)

    @staticmethod
    def _emit_event(event_type: str, **kwargs: Any) -> None:
        try:
            event_bus.emit(event_type, **kwargs)
        except Exception:
            pass
