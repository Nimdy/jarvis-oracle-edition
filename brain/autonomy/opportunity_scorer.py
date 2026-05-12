"""Opportunity Scorer — ranks research intents by measurable value.

Score = (Impact × Evidence × Confidence − 0.3×Risk − 0.3×Cost)
        + policy_adjustment
        − diminishing_returns_penalty
        − action_rate_penalty

Where:
  Impact:      expected improvement to a real metric
  Evidence:    sustained degradation window + sample count
  Confidence:  codebase grounding quality + reproducibility
  Risk:        touches sensitive subsystems / introduces new capability
  Cost:        estimated files/lines/iterations (token budget proxy)

Anti-gaming guardrails:
  - minimum meaningful delta threshold to count as a "win" in policy memory
  - diminishing returns on repeated intents in same category
  - action count penalty over a time window (prevents spamming cheap actions)
  - policy memory adjustment boosts/penalizes based on historical outcomes

Replaces the simple priority float with a principled composite score so
the orchestrator picks "most painful thing first" instead of "most
interesting thought."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from autonomy.constants import MIN_MEANINGFUL_DELTA
from autonomy.research_intent import ResearchIntent

logger = logging.getLogger(__name__)

DEFICIT_THRESHOLDS: dict[str, tuple[float, str]] = {
    # metric_name: (threshold, direction)  — direction: "above" means high=bad
    "confidence_volatility":    (0.15, "above"),
    "tick_p95_ms":              (15.0, "above"),
    "memory_recall_miss_rate":  (0.30, "above"),
    "barge_in_rate":            (0.20, "above"),
    "tool_failure_rate":        (0.20, "above"),
    "reasoning_coherence":      (0.50, "below"),
    "processing_health":        (0.50, "below"),
}

EVIDENCE_STRONG_WINDOW_S = 300.0
EVIDENCE_WEAK_WINDOW_S = 60.0

DIMINISHING_RETURNS_WINDOW_S = 3600.0
DIMINISHING_RETURNS_DECAY = 0.15
ACTION_RATE_WINDOW_S = 1800.0
ACTION_RATE_PENALTY_PER_FAMILY = 0.04
ACTION_RATE_MAX_FAMILY_PENALTY = 0.12
ACTION_RATE_GLOBAL_PER = 0.01
ACTION_RATE_MAX_GLOBAL_PENALTY = 0.06


@dataclass
class OpportunityScore:
    """Breakdown of a research intent's opportunity value."""

    impact: float = 0.0
    evidence: float = 0.0
    confidence: float = 0.0
    risk: float = 0.0
    cost: float = 0.0
    policy_adjustment: float = 0.0
    diminishing_penalty: float = 0.0
    action_rate_penalty: float = 0.0
    total: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "impact": round(self.impact, 3),
            "evidence": round(self.evidence, 3),
            "confidence": round(self.confidence, 3),
            "risk": round(self.risk, 3),
            "cost": round(self.cost, 3),
            "policy_adjustment": round(self.policy_adjustment, 3),
            "diminishing_penalty": round(self.diminishing_penalty, 3),
            "action_rate_penalty": round(self.action_rate_penalty, 3),
            "total": round(self.total, 3),
            "reason": self.reason,
        }


@dataclass
class _MetricReading:
    value: float
    timestamp: float
    in_deficit: bool = False


class OpportunityScorer:
    """Consumes system metrics and produces a composite score for each ResearchIntent.

    The orchestrator uses ``score.total`` to replace the simple ``priority`` field,
    ensuring the queue is always sorted by measured value, not interest.
    """

    def __init__(self) -> None:
        self._metric_history: dict[str, list[_MetricReading]] = {
            k: [] for k in DEFICIT_THRESHOLDS
        }
        self._total_scored: int = 0
        self._deficit_durations: dict[str, float] = {k: 0.0 for k in DEFICIT_THRESHOLDS}
        self._policy_memory: Any = None
        self._recent_category_actions: list[tuple[float, str]] = []
        self._recent_family_actions: list[tuple[float, str]] = []

    # -- metric ingestion (called every ~5s from orchestrator) -----------------

    def record_metrics(self, metrics: dict[str, float]) -> None:
        """Ingest a snapshot of current system metrics."""
        now = time.time()
        for key, (threshold, direction) in DEFICIT_THRESHOLDS.items():
            value = metrics.get(key)
            if value is None:
                continue

            in_deficit = (
                value > threshold if direction == "above" else value < threshold
            )

            history = self._metric_history[key]
            history.append(_MetricReading(value=value, timestamp=now, in_deficit=in_deficit))
            if len(history) > 60:
                self._metric_history[key] = history[-60:]

            if in_deficit and history:
                deficit_start = now
                for r in reversed(history):
                    if r.in_deficit:
                        deficit_start = r.timestamp
                    else:
                        break
                self._deficit_durations[key] = now - deficit_start
            else:
                self._deficit_durations[key] = 0.0

    # -- scoring ---------------------------------------------------------------

    def set_policy_memory(self, policy_memory: Any) -> None:
        """Wire the policy memory for experience-based score adjustments."""
        self._policy_memory = policy_memory

    def record_action(self, category: str) -> None:
        """Record that an action was taken in a category (for rate penalty)."""
        now = time.time()
        family = category.split(":")[0] if ":" in category else category
        self._recent_category_actions.append((now, category))
        self._recent_family_actions.append((now, family))
        cutoff = now - max(DIMINISHING_RETURNS_WINDOW_S, ACTION_RATE_WINDOW_S)
        self._recent_category_actions = [
            (t, c) for t, c in self._recent_category_actions if t > cutoff
        ]
        self._recent_family_actions = [
            (t, f) for t, f in self._recent_family_actions if t > cutoff
        ]

    def score(self, intent: ResearchIntent) -> OpportunityScore:
        """Compute composite score for a research intent."""
        impact = self._compute_impact(intent)
        evidence = self._compute_evidence(intent)
        confidence = self._compute_confidence(intent)
        risk = self._compute_risk(intent)
        cost = self._compute_cost(intent)

        base = (impact * 0.4 + evidence * 0.3 + confidence * 0.2) - (risk * 0.25) - (cost * 0.15)

        policy_adj = self._compute_policy_adjustment(intent)
        dim_penalty = self._compute_diminishing_returns(intent)
        rate_penalty = self._compute_action_rate_penalty(intent)

        total = max(0.0, min(1.0, base + policy_adj - dim_penalty - rate_penalty))

        parts: list[str] = []
        if impact > 0.5:
            parts.append(f"high-impact({impact:.2f})")
        if evidence > 0.5:
            parts.append(f"strong-evidence({evidence:.2f})")
        if risk > 0.3:
            parts.append(f"elevated-risk({risk:.2f})")
        if policy_adj > 0.05:
            parts.append(f"history-boost({policy_adj:+.2f})")
        elif policy_adj < -0.05:
            parts.append(f"history-penalty({policy_adj:+.2f})")
        if dim_penalty > 0.01:
            parts.append(f"diminishing({dim_penalty:.2f})")

        self._total_scored += 1
        return OpportunityScore(
            impact=impact, evidence=evidence, confidence=confidence,
            risk=risk, cost=cost, policy_adjustment=policy_adj,
            diminishing_penalty=dim_penalty, action_rate_penalty=rate_penalty,
            total=total,
            reason=", ".join(parts) if parts else "baseline",
        )

    # -- anti-gaming -----------------------------------------------------------

    def _compute_policy_adjustment(self, intent: ResearchIntent) -> float:
        """Boost or penalize based on historical outcomes for similar topics."""
        adj = 0.0
        if self._policy_memory is not None:
            try:
                adj += self._policy_memory.score_adjustment(
                    intent.tag_cluster, intent.source_hint,
                )
            except Exception:
                pass

        try:
            from autonomy.source_ledger import get_source_ledger
            usefulness = get_source_ledger().get_topic_usefulness(intent.tag_cluster)
            # usefulness is 0.0-1.0 (0.5 = neutral). Shift to ±0.15 adjustment.
            adj += (usefulness - 0.5) * 0.3
        except Exception:
            pass

        return max(-0.3, min(0.3, adj))

    def _compute_diminishing_returns(self, intent: ResearchIntent) -> float:
        """Penalize repeated actions in the same category within the window.

        Each recent action in the same category adds DIMINISHING_RETURNS_DECAY
        to the penalty, capped at 0.3.
        """
        now = time.time()
        cutoff = now - DIMINISHING_RETURNS_WINDOW_S
        category = self._intent_category(intent)
        count = sum(
            1 for t, c in self._recent_category_actions
            if t > cutoff and c == category
        )
        return min(0.3, count * DIMINISHING_RETURNS_DECAY)

    def _compute_action_rate_penalty(self, intent: ResearchIntent) -> float:
        """Per-family rate penalty + small global cap.

        Penalizes high action rate within the intent's source family
        (metric:*, thought:*, gap:*, etc.) so one busy domain doesn't
        shut down all autonomy. A smaller global penalty prevents any
        single hour from going wild regardless of diversity.
        """
        now = time.time()
        cutoff = now - ACTION_RATE_WINDOW_S
        family = (intent.source_event.split(":")[0]
                  if intent.source_event and ":" in intent.source_event
                  else intent.source_hint or "unknown")

        family_count = sum(
            1 for t, f in self._recent_family_actions
            if t > cutoff and f == family
        )
        global_count = sum(1 for t, _ in self._recent_family_actions if t > cutoff)

        family_penalty = min(ACTION_RATE_MAX_FAMILY_PENALTY,
                             family_count * ACTION_RATE_PENALTY_PER_FAMILY)
        global_penalty = min(ACTION_RATE_MAX_GLOBAL_PENALTY,
                             global_count * ACTION_RATE_GLOBAL_PER)

        return family_penalty + global_penalty

    @staticmethod
    def _intent_category(intent: ResearchIntent) -> str:
        """Derive a broad category from the intent for diminishing returns."""
        source = intent.source_event.split(":")[0] if intent.source_event else "unknown"
        tool = intent.source_hint or "any"
        tags = sorted(intent.tag_cluster)[:2]
        return f"{source}:{tool}:{','.join(tags)}" if tags else f"{source}:{tool}"

    # -- introspection ---------------------------------------------------------

    def get_active_deficits(self) -> dict[str, float]:
        return {k: v for k, v in self._deficit_durations.items() if v > 0}

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_scored": self._total_scored,
            "active_deficits": self.get_active_deficits(),
            "deficit_count": sum(1 for v in self._deficit_durations.values() if v > 0),
        }

    # -- component scorers ----------------------------------------------------

    _EXISTENTIAL_TAG_KEYWORDS = frozenset({
        "consciousness", "meaning", "philosophical", "reality",
        "existence", "agency",
    })

    def _compute_impact(self, intent: ResearchIntent) -> float:
        source = intent.source_event.split(":")[0] if intent.source_event else ""
        tag_set = {t.lower() for t in intent.tag_cluster}

        # Goal-linked intents outrank everything
        if getattr(intent, "goal_id", ""):
            return 0.85

        if source == "metric":
            return 0.8

        base = {"thought": 0.4, "existential": 0.3, "emergence": 0.5,
                "gap": 0.7, "learning_protocol": 0.4}.get(source, 0.3)

        # Existential penalty: unfocused philosophical research without goal alignment
        is_existential = source == "existential"
        if not is_existential:
            full_source = intent.source_event or ""
            is_existential = full_source in (
                "drive:curiosity", "drive:coherence", "drive:play",
            )
        if not is_existential:
            tag_words = set()
            for t in tag_set:
                tag_words.update(t.split(":"))
            is_existential = bool(tag_words & self._EXISTENTIAL_TAG_KEYWORDS)

        if is_existential:
            base = 0.15

        deficit_tags = {
            "memory": "memory_recall_miss_rate",
            "response": "barge_in_rate",
            "performance": "tick_p95_ms",
            "coherence": "reasoning_coherence",
        }
        for tag in tag_set:
            for dtag, metric in deficit_tags.items():
                if dtag in tag and self._deficit_durations.get(metric, 0) > 0:
                    base += 0.2
                    break

        return min(1.0, base)

    def _compute_evidence(self, intent: ResearchIntent) -> float:
        source = intent.source_event.split(":")[0] if intent.source_event else ""

        if source == "metric":
            metric_name = intent.source_event.split(":", 1)[1] if ":" in intent.source_event else ""
            duration = self._deficit_durations.get(metric_name, 0)
            if duration >= EVIDENCE_STRONG_WINDOW_S:
                return 0.9
            if duration >= EVIDENCE_WEAK_WINDOW_S:
                return 0.6
            return 0.3

        reps = intent.trigger_count
        if reps >= 5:
            return 0.8
        if reps >= 3:
            return 0.5
        return 0.3

    @staticmethod
    def _compute_confidence(intent: ResearchIntent) -> float:
        return {"codebase": 0.9, "memory": 0.7, "introspection": 0.8,
                "web": 0.5}.get(intent.source_hint, 0.6)

    @staticmethod
    def _compute_risk(intent: ResearchIntent) -> float:
        risk = 0.0
        q = intent.question.lower()
        tags = {t.lower() for t in intent.tag_cluster}
        for term in ("mutation", "governor", "self_improve", "kernel", "policy"):
            if term in q or term in tags:
                risk += 0.1
        if intent.scope == "external_ok":
            risk += 0.1
        if intent.source_hint == "web":
            risk += 0.05
        return min(0.5, risk)

    @staticmethod
    def _compute_cost(intent: ResearchIntent) -> float:
        token_cost = intent.max_tokens / 10000.0
        time_cost = intent.timeout_s / 120.0
        return min(0.3, (token_cost + time_cost) / 2.0)
