"""Drive-based exploration system for Jarvis autonomy.

Instead of generating research questions first, drives represent
underlying motives (truth, curiosity, mastery, relevance, coherence,
continuity, play).  Each drive competes for attention based on urgency
signals.  The winning drive selects the cheapest meaningful action —
preferring local tools (introspection, memory, codebase) before
escalating to external research.

Drives emit not just ResearchIntents but also non-research actions:
  - audit: inspect recent gate rewrites, subsystem disagreements
  - recall: search memory for existing knowledge
  - experiment: run bounded local test (prompt mutation, routing compare)
  - reprioritize: adjust active topic weights toward user goals
  - noop: defer when no drive has sufficient urgency

The DriveManager is called from AutonomyOrchestrator.on_tick() as an
additional intent source alongside MetricTriggers and EventBridge.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DRIVE_STATE_PATH = Path("~/.jarvis/drive_state.json").expanduser()

DriveType = Literal[
    "truth", "curiosity", "mastery", "relevance",
    "coherence", "continuity", "play",
]

ActionType = Literal[
    "research", "audit", "recall", "experiment",
    "mutation", "reprioritize", "learn", "noop",
]

DRIVE_TYPES: tuple[DriveType, ...] = (
    "truth", "curiosity", "mastery", "relevance",
    "coherence", "continuity", "play",
)


@dataclass
class DriveSignals:
    """Aggregated signals that feed drive urgency computation."""
    gate_blocks_recent: int = 0
    gate_block_reasons: list[str] = field(default_factory=list)
    metric_deficits: dict[str, float] = field(default_factory=dict)
    user_topics: dict[str, float] = field(default_factory=dict)
    open_threads: int = 0
    pending_jobs: int = 0
    contradictions: int = 0
    novelty_events: int = 0
    system_health: float = 1.0
    tick_p95_ms: float = 0.0
    error_rate: float = 0.0
    total_urgency: float = 0.0
    uptime_s: float = 0.0
    relevance_success_rate: float = 0.5
    saturated_topics: set[str] = field(default_factory=set)
    active_user_goals: int = 0


@dataclass
class DriveState:
    """Per-drive state tracked across ticks."""
    drive_type: DriveType
    urgency: float = 0.0
    last_acted: float = 0.0
    last_outcome_positive: bool = False
    action_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.action_count == 0:
            return 0.5
        return self.success_count / self.action_count


@dataclass
class DriveAction:
    """Output of drive evaluation — what the system should do next."""
    drive_type: DriveType
    action_type: ActionType
    urgency: float
    question: str = ""
    tool_hint: str = "any"
    scope: str = "local_only"
    tags: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "drive": self.drive_type,
            "action": self.action_type,
            "urgency": round(self.urgency, 3),
            "question": self.question,
            "tool_hint": self.tool_hint,
            "scope": self.scope,
            "tags": list(self.tags),
            "detail": self.detail,
        }


# ── Drive urgency computation ────────────────────────────────────────

_URGENCY_THRESHOLD = 0.25
_COOLDOWN_S = 120.0
_DECAY_PER_S = 0.001
_BOOT_FAILURE_CAP = 10

_DRIVE_FLOOR: dict[str, float] = {
    "truth":      0.10,
    "curiosity":  0.15,
    "mastery":    0.08,
    "relevance":  0.12,
    "coherence":  0.10,
    "continuity": 0.08,
    "play":       0.03,
}

# Per-drive: preferred strategy, cheapest tool, escalation tool
_DRIVE_STRATEGY: dict[DriveType, dict[str, Any]] = {
    "truth": {
        "action": "audit",
        "tools": ("introspection", "codebase", "memory"),
        "escalation": "web",
        "scope": "local_only",
    },
    "curiosity": {
        "action": "research",
        "tools": ("memory", "codebase", "academic"),
        "escalation": "web",
        "scope": "external_ok",
    },
    "mastery": {
        "action": "experiment",
        "tools": ("codebase", "introspection"),
        "escalation": "academic",
        "scope": "local_only",
    },
    "relevance": {
        "action": "recall",
        "tools": ("memory", "introspection"),
        "escalation": "web",
        "scope": "local_only",
    },
    "coherence": {
        "action": "audit",
        "tools": ("introspection", "memory"),
        "escalation": "codebase",
        "scope": "local_only",
    },
    "continuity": {
        "action": "recall",
        "tools": ("memory", "introspection"),
        "escalation": None,
        "scope": "local_only",
    },
    "play": {
        "action": "experiment",
        "tools": ("introspection", "codebase"),
        "escalation": None,
        "scope": "local_only",
    },
}


def _compute_truth_urgency(signals: DriveSignals) -> float:
    """Truth drive: gate blocks and response contradictions."""
    blocks = min(signals.gate_blocks_recent, 10) / 10.0
    contradictions = min(signals.contradictions, 5) / 5.0
    return min(1.0, blocks * 0.6 + contradictions * 0.4)


def _compute_curiosity_urgency(signals: DriveSignals) -> float:
    """Curiosity drive: novel events and surprise spikes.

    Dampened by 0.5x when active user goals exist to reduce
    existential/philosophical research that competes with user work.
    """
    novelty = min(signals.novelty_events, 10) / 10.0
    stability_bonus = max(0, signals.system_health - 0.7) * 0.5
    urgency = min(1.0, novelty * 0.7 + stability_bonus * 0.3)
    if signals.active_user_goals > 0:
        urgency *= 0.5
    return urgency


def _compute_mastery_urgency(signals: DriveSignals) -> float:
    """Mastery drive: repeated failures and metric deficits.

    Only actionable deficits contribute to urgency.  Time-gated, external,
    and systemic deficits are acknowledged but not acted on.
    """
    if not signals.metric_deficits:
        return 0.0
    actionable = _filter_actionable_deficits(signals.metric_deficits)
    if not actionable:
        return 0.0
    worst = max(actionable.values())
    num_deficits = len(actionable)
    return min(1.0, worst * 0.5 + min(num_deficits, 5) / 5.0 * 0.5)


def _compute_relevance_urgency(signals: DriveSignals) -> float:
    """Relevance drive: user emphasis, outcome-weighted, specificity-gated."""
    if not signals.user_topics:
        return 0.0
    # Filter out saturated topics before scoring
    live_topics = {t: w for t, w in signals.user_topics.items()
                   if t not in signals.saturated_topics}
    if not live_topics:
        return 0.0
    top_weight = max(live_topics.values())
    # Outcome dampening: poor success rate reduces urgency (floor at 30%)
    outcome_factor = 0.3 + 0.7 * signals.relevance_success_rate
    # Specificity: bigrams (containing spaces) are higher quality than single words
    bigram_ratio = sum(1 for t in live_topics if " " in t) / max(len(live_topics), 1)
    specificity_factor = 0.5 + 0.5 * min(1.0, bigram_ratio)
    return min(1.0, top_weight * outcome_factor * specificity_factor)


def _compute_coherence_urgency(signals: DriveSignals) -> float:
    """Coherence drive: internal contradictions and disagreements."""
    c = min(signals.contradictions, 5) / 5.0
    health_penalty = max(0, 0.7 - signals.system_health) * 2.0
    return min(1.0, c * 0.5 + health_penalty * 0.5)


def _compute_continuity_urgency(signals: DriveSignals) -> float:
    """Continuity drive: unresolved threads and pending jobs."""
    threads = min(signals.open_threads, 10) / 10.0
    jobs = min(signals.pending_jobs, 5) / 5.0
    return min(1.0, threads * 0.6 + jobs * 0.4)


def _compute_play_urgency(signals: DriveSignals) -> float:
    """Play drive: inverse of total urgency — explore when nothing is pressing."""
    if signals.total_urgency > 0.6:
        return 0.0
    stability = signals.system_health
    calm = max(0, 1.0 - signals.total_urgency)
    if signals.uptime_s < 600:
        return 0.0
    return min(0.5, calm * 0.6 + stability * 0.2)


_URGENCY_FNS: dict[DriveType, Any] = {
    "truth": _compute_truth_urgency,
    "curiosity": _compute_curiosity_urgency,
    "mastery": _compute_mastery_urgency,
    "relevance": _compute_relevance_urgency,
    "coherence": _compute_coherence_urgency,
    "continuity": _compute_continuity_urgency,
    "play": _compute_play_urgency,
}


# ── Question templates (parameterized, not static) ───────────────────

def _truth_question(signals: DriveSignals) -> str:
    if signals.gate_block_reasons:
        reason = signals.gate_block_reasons[0]
        return (f"The capability gate recently blocked '{reason}'. "
                f"What distinguishes grounded sensor observations from "
                f"unverified capability claims in AI systems?")
    return ("What patterns cause truth-layer conflicts between perception "
            "evidence and capability gating in embodied AI systems?")


def _curiosity_question(signals: DriveSignals) -> str:
    return ("What unexpected pattern or novel signal in recent system behavior "
            "warrants further investigation?")


def _mastery_question(signals: DriveSignals) -> str:
    actionable = _filter_actionable_deficits(signals.metric_deficits)
    if actionable:
        worst_metric = max(actionable, key=actionable.get)
        return (f"The metric '{worst_metric}' has been in deficit. "
                f"What techniques could reduce this deficit in an "
                f"adaptive decision system?")
    return "What optimization would address the most impactful current deficit?"


def _relevance_question(signals: DriveSignals) -> str:
    if signals.user_topics:
        # Filter out saturated topics
        live = {t: w for t, w in signals.user_topics.items()
                if t not in signals.saturated_topics}
        if live:
            top = max(live, key=live.get)
            return f"Recall: {top}"
    return "Recall: recent user conversation topics"


def _coherence_question(signals: DriveSignals) -> str:
    return ("What internal contradictions exist between recent responses "
            "and current subsystem states?")


def _continuity_question(signals: DriveSignals) -> str:
    return ("What unresolved threads from recent conversations or learning "
            "jobs should be revisited?")


def _play_question(signals: DriveSignals) -> str:
    return ("What bounded local experiment could yield new insight about "
            "system behavior or response quality?")


_QUESTION_FNS: dict[DriveType, Any] = {
    "truth": _truth_question,
    "curiosity": _curiosity_question,
    "mastery": _mastery_question,
    "relevance": _relevance_question,
    "coherence": _coherence_question,
    "continuity": _continuity_question,
    "play": _play_question,
}


# ── Deficit Actionability Classification ──────────────────────────────
#
# Classifies each known deficit metric by whether the mastery drive can
# meaningfully improve it.  Unknown deficits default to "actionable"
# (safe fallback — new metrics get attention until explicitly classified).

DeficitCategory = Literal["actionable", "time_gated", "external", "systemic"]

_DEFICIT_ACTIONABILITY: dict[str, DeficitCategory] = {
    "recognition_confidence": "actionable",
    "emotion_accuracy":       "actionable",
    "ranker_not_ready":       "actionable",
    "salience_not_ready":     "actionable",
    "memory_recall_miss_rate": "actionable",
    "tick_p95_ms":            "actionable",
    "confidence_volatility":  "time_gated",
    "shadow_default_win_rate": "time_gated",
    "barge_in_rate":          "external",
    "friction_rate":          "external",
    "reasoning_coherence":    "systemic",
    "processing_health":      "systemic",
}

_ACTIONABLE_CATEGORIES: frozenset[str] = frozenset({"actionable"})


def _filter_actionable_deficits(
    deficits: dict[str, float],
) -> dict[str, float]:
    """Return only deficits the mastery drive can meaningfully act on."""
    return {
        k: v for k, v in deficits.items()
        if _DEFICIT_ACTIONABILITY.get(k, "actionable") in _ACTIONABLE_CATEGORIES
    }


# ── Mastery → Learning Bridge ─────────────────────────────────────────
#
# Map from a measured deficit to a capability that CAN be improved via an
# orchestrated learning job (assess → collect → train → verify → register).
#
# Intentionally NOT in this map:
#   recognition_confidence → speaker_identification_v1 (retired 2026-04-18)
#   emotion_accuracy       → emotion_detection_v1      (retired 2026-04-18)
#
# Reason: these perceptual capabilities already self-improve continuously via
# the Tier-1 distillation loop (hemisphere/orchestrator.py). That loop retrains
# speaker_repr / emotion_depth specialists every ~2 minutes against fresh
# teacher signals, with a regression gate protecting already-good models.
# Creating a parallel learning job for the same capability produced redundant
# bookkeeping AND historically blocked on a verifier that read the wrong
# telemetry fields. See BUILD_HISTORY 2026-04-18 entry.
#
# With these deficits removed from the map, the mastery drive still tracks
# them for telemetry (_DEFICIT_ACTIONABILITY keeps them "actionable") and
# still acts — but via the default strategy (experiment + codebase/introspection
# tools), which is a safe investigation path that cannot create blocked skills.
#
# If a user EXPLICITLY asks to "train speaker identification" or similar, the
# SkillResolver path in skills/resolver.py still creates a learning job with
# a guided-collect flow. This only retires the autonomous auto-creation.
#
# Verifier fields (fixed 2026-04-18, covered by tests in
# test_skill_baseline.py::TestVerifierBugRegressions):
#   * skills/baseline.py now reads h["best_accuracy"] (was
#     "migration_readiness" — only populates during substrate migration).
#   * Per-teacher distillation counts read teachers[t]["total"] (was
#     "total_signals" — that key only exists at the outer aggregate level).
#   * Speaker enrollment metrics derive from the real profile schema
#     (enrollment_clips + _score_ema), not the never-populated
#     Profile.confidence field.
#   * skills/executors/perceptual.py verify sites now read "best_accuracy"
#     at all three call sites as well.
# Re-adding a deficit→learning-job mapping is now safe from a verifier
# standpoint. Only re-add one if that specific capability does NOT already
# self-improve via Tier-1 distillation.

_DEFICIT_CAPABILITY_MAP: dict[str, dict[str, str]] = {
    "ranker_not_ready": {
        "skill_id": "memory_ranking_v1",
        "capability_type": "procedural",
        "protocol_id": "SK-001",
        "description": "Improve memory retrieval ranking",
    },
}

_LEARN_COOLDOWN_S = 3600.0


def _has_protocol(protocol_id: str) -> bool:
    """Check whether a verification protocol exists for a given ID."""
    try:
        from skills.verification_protocols import get_protocol
        return get_protocol(protocol_id) is not None
    except ImportError:
        return False


def _is_already_learning(skill_id: str) -> bool:
    """Check if there's already an active learning job for this skill."""
    try:
        from skills.learning_jobs import LearningJobStore
        store = LearningJobStore()
        for job in store.load_all():
            if job.skill_id == skill_id and job.status in ("active", "paused"):
                return True
    except Exception:
        pass
    return False


def _is_already_verified(skill_id: str) -> bool:
    """Check if the skill is already verified in the registry."""
    try:
        from skills.registry import skill_registry
        if skill_registry:
            rec = skill_registry.get(skill_id)
            if rec and rec.status == "verified":
                return True
    except Exception:
        pass
    return False


# ── DriveManager ─────────────────────────────────────────────────────

class DriveManager:
    """Evaluates drives, selects the winning motive, and produces an action.

    Called from AutonomyOrchestrator.on_tick() every ~60s as an additional
    intent source.  Does not replace MetricTriggers or EventBridge — it
    supplements them with a higher-level motivational layer.
    """

    def __init__(self) -> None:
        self._states: dict[DriveType, DriveState] = {
            dt: DriveState(drive_type=dt) for dt in DRIVE_TYPES
        }
        self._last_eval: float = 0.0
        self._boot_ts: float = time.time()
        self._learn_cooldowns: dict[str, float] = {}  # skill_id -> last_proposed_ts
        self._last_signals: DriveSignals | None = None

    def evaluate(self, signals: DriveSignals) -> list[DriveState]:
        """Score all drives, return sorted by urgency (highest first).

        Applies cooldown and decay to prevent stale drives from dominating.
        """
        now = time.time()
        signals.uptime_s = now - self._boot_ts

        # First pass: compute raw urgency
        raw_total = 0.0
        for dt in DRIVE_TYPES:
            fn = _URGENCY_FNS[dt]
            raw = fn(signals)
            self._states[dt].urgency = raw
            raw_total += raw
        signals.total_urgency = raw_total

        # Recompute play (depends on total urgency)
        self._states["play"].urgency = _compute_play_urgency(signals)

        # Apply cooldown, graduated failure dampening, decay, and floor
        for dt in DRIVE_TYPES:
            state = self._states[dt]
            floor = _DRIVE_FLOOR.get(dt, 0.05)
            if state.consecutive_failures >= 3:
                dampening = max(0.05, 0.3 ** (1 + (state.consecutive_failures - 3) / 10))
                state.urgency *= dampening
            if state.last_acted > 0:
                since_acted = now - state.last_acted
                if since_acted < _COOLDOWN_S:
                    state.urgency *= 0.2
                if since_acted > 300:
                    elapsed = since_acted - 300
                    state.urgency = max(floor, state.urgency * math.exp(-0.002 * elapsed))
            elif state.consecutive_failures >= 3:
                state.urgency = max(floor, state.urgency)

        self._last_eval = now
        self._last_signals = signals
        return sorted(self._states.values(), key=lambda s: s.urgency, reverse=True)

    def select_action(self, top_drive: DriveState, signals: DriveSignals) -> DriveAction | None:
        """Generate an action from the winning drive.

        Returns None if urgency is below threshold.
        When the mastery drive wins and a learnable capability mapping exists,
        produces a ``learn`` action instead of the default ``experiment``.
        """
        if top_drive.urgency < _URGENCY_THRESHOLD:
            return None

        dt = top_drive.drive_type

        # Mastery drive → check if we should propose a learning job
        if dt == "mastery" and signals.metric_deficits:
            learn_action = self._try_mastery_learn(top_drive, signals)
            if learn_action is not None:
                return learn_action

            # All remaining deficits are non-actionable → noop (no failure recorded)
            actionable = _filter_actionable_deficits(signals.metric_deficits)
            if not actionable:
                non_actionable = {
                    k: _DEFICIT_ACTIONABILITY.get(k, "unknown")
                    for k in signals.metric_deficits
                }
                detail = (
                    f"Mastery noop: {len(signals.metric_deficits)} deficit(s) "
                    f"present but none actionable — {non_actionable}"
                )
                logger.debug(detail)
                return DriveAction(
                    drive_type="mastery",
                    action_type="noop",
                    urgency=top_drive.urgency,
                    detail=detail,
                    tags=("drive:mastery", "action:noop"),
                )

        strategy = _DRIVE_STRATEGY[dt]
        action_type: ActionType = strategy["action"]
        tools = strategy["tools"]
        tool_hint = tools[0] if tools else "any"

        # Escalate if local tools have been failing
        if top_drive.consecutive_failures >= 2 and strategy.get("escalation"):
            tool_hint = strategy["escalation"]
            if strategy.get("scope") == "local_only":
                scope = "external_ok"
            else:
                scope = strategy["scope"]
        else:
            scope = strategy["scope"]

        question_fn = _QUESTION_FNS.get(dt)
        question = question_fn(signals) if question_fn else ""

        tags = (f"drive:{dt}", f"action:{action_type}")

        top_drive.last_acted = time.time()
        top_drive.action_count += 1

        return DriveAction(
            drive_type=dt,
            action_type=action_type,
            urgency=top_drive.urgency,
            question=question,
            tool_hint=tool_hint,
            scope=scope,
            tags=tags,
            detail=f"Drive '{dt}' urgency={top_drive.urgency:.2f}",
        )

    def _try_mastery_learn(
        self, top_drive: DriveState, signals: DriveSignals,
    ) -> DriveAction | None:
        """If a deficit maps to a learnable capability with a protocol, emit learn action.

        Guards:
          - protocol must exist for the mapped capability
          - skill must not already be learning or verified
          - cooldown per skill_id
        """
        now = time.time()
        for deficit_key in sorted(
            signals.metric_deficits, key=signals.metric_deficits.get, reverse=True
        ):
            mapping = _DEFICIT_CAPABILITY_MAP.get(deficit_key)
            if mapping is None:
                continue

            skill_id = mapping["skill_id"]
            protocol_id = mapping["protocol_id"]

            # Protocol-exists gate
            if not _has_protocol(protocol_id):
                logger.debug("Mastery learn skipped: no protocol %s for %s",
                             protocol_id, skill_id)
                continue

            # Already learning/verified suppression
            if _is_already_learning(skill_id):
                logger.debug("Mastery learn skipped: %s already learning", skill_id)
                continue
            if _is_already_verified(skill_id):
                logger.debug("Mastery learn skipped: %s already verified", skill_id)
                continue

            # Cooldown/dedup
            last_proposed = self._learn_cooldowns.get(skill_id, 0.0)
            if now - last_proposed < _LEARN_COOLDOWN_S:
                logger.debug("Mastery learn cooldown: %s (%.0fs remaining)",
                             skill_id, _LEARN_COOLDOWN_S - (now - last_proposed))
                continue

            self._learn_cooldowns[skill_id] = now
            top_drive.last_acted = now
            top_drive.action_count += 1

            return DriveAction(
                drive_type="mastery",
                action_type="learn",
                urgency=top_drive.urgency,
                question=mapping["description"],
                tool_hint="learning_jobs",
                scope="local_only",
                tags=("drive:mastery", "action:learn", f"skill:{skill_id}"),
                detail=(
                    f"Mastery drive proposes learning job: "
                    f"{skill_id} (protocol {protocol_id}, "
                    f"deficit {deficit_key}={signals.metric_deficits[deficit_key]:.2f})"
                ),
            )

        return None

    def record_outcome(self, drive_type: DriveType, worked: bool) -> None:
        """Feed back into drive urgency calibration.

        On success, decrement consecutive_failures by 3 (fast recovery from
        dampening).  A drive dampened at 8 failures recovers in 2 wins
        (8→5→2) rather than 6 wins under the old -1 rule.
        """
        state = self._states.get(drive_type)
        if not state:
            return
        state.last_outcome_positive = worked
        if worked:
            state.success_count += 1
            state.consecutive_failures = max(0, state.consecutive_failures - 3)
        else:
            state.consecutive_failures += 1

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        drives_out: dict[str, dict[str, Any]] = {}
        for dt, s in self._states.items():
            strategy = _DRIVE_STRATEGY.get(dt, {})
            suppression = ""
            if s.last_acted > 0 and (now - s.last_acted) < _COOLDOWN_S:
                suppression = f"cooldown ({int(_COOLDOWN_S - (now - s.last_acted))}s left)"
            elif s.consecutive_failures >= 3:
                dampening = max(0.05, 0.3 ** (1 + (s.consecutive_failures - 3) / 10))
                suppression = (
                    f"dampened ({s.consecutive_failures} failures, "
                    f"{dampening:.2f}x multiplier)"
                )
            drives_out[dt] = {
                "urgency": round(s.urgency, 3),
                "strategy": strategy.get("action", ""),
                "tools": list(strategy.get("tools", ())),
                "action_count": s.action_count,
                "success_rate": round(s.success_rate, 3),
                "consecutive_failures": s.consecutive_failures,
                "last_acted": s.last_acted,
                "last_acted_ago_s": round(now - s.last_acted, 1) if s.last_acted > 0 else None,
                "last_outcome": "positive" if s.last_outcome_positive else ("negative" if s.action_count > 0 else "none"),
                "suppression": suppression,
            }

        status: dict[str, Any] = {
            "drives": drives_out,
            "last_eval": self._last_eval,
            "boot_ts": self._boot_ts,
        }

        signals = self._last_signals
        if signals is not None and signals.metric_deficits:
            deficit_detail: dict[str, dict[str, Any]] = {}
            for k, v in signals.metric_deficits.items():
                category = _DEFICIT_ACTIONABILITY.get(k, "actionable")
                deficit_detail[k] = {
                    "magnitude": round(v, 3),
                    "category": category,
                    "actionable": category in _ACTIONABLE_CATEGORIES,
                }
            status["deficit_actionability"] = deficit_detail

        return status

    def save_state(self) -> None:
        """Persist per-drive counters (not urgency — computed fresh each tick)."""
        data: dict[str, Any] = {}
        for dt, s in self._states.items():
            data[dt] = {
                "action_count": s.action_count,
                "success_count": s.success_count,
                "consecutive_failures": s.consecutive_failures,
                "last_acted": s.last_acted,
                "last_outcome_positive": s.last_outcome_positive,
            }
        try:
            DRIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=DRIVE_STATE_PATH.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, separators=(",", ":"))
                os.replace(tmp, DRIVE_STATE_PATH)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            logger.debug("DriveManager save_state failed", exc_info=True)

    def load_state(self) -> int:
        """Restore per-drive counters from disk. Returns count of drives restored."""
        if not DRIVE_STATE_PATH.exists():
            return 0
        try:
            data = json.loads(DRIVE_STATE_PATH.read_text())
        except Exception:
            logger.warning("DriveManager load_state: corrupt file, skipping")
            return 0

        restored = 0
        for dt, d in data.items():
            state = self._states.get(dt)  # type: ignore[arg-type]
            if state is None:
                continue
            state.action_count = d.get("action_count", 0)
            state.success_count = d.get("success_count", 0)
            state.consecutive_failures = d.get("consecutive_failures", 0)
            state.last_acted = d.get("last_acted", 0.0)
            if state.last_acted > 0 and (time.time() - state.last_acted) > 600:
                state.last_acted = 0.0
                if state.consecutive_failures > _BOOT_FAILURE_CAP:
                    logger.info(
                        "Drive %s: capping consecutive_failures %d → %d on boot",
                        dt, state.consecutive_failures, _BOOT_FAILURE_CAP,
                    )
                    state.consecutive_failures = _BOOT_FAILURE_CAP
            state.last_outcome_positive = d.get("last_outcome_positive", False)
            restored += 1

        if restored:
            logger.info("DriveManager restored state for %d drives", restored)
        return restored
