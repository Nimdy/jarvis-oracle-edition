"""Counterfactual Evaluation Engine — Phase 10 (#17).

After important autonomy decisions, evaluate the alternatives we DIDN'T take:
for each lived decision, estimate what the best historically-known alternative
(conditioned on the decision's topic) would have yielded, and flag a
"missed opportunity" when an alternative would have meaningfully beaten the
actual outcome.

Discipline:
  - read-only: reads the persisted autonomy policy history; mutates nothing
    canonical; no LLM; deterministic given the same history.
  - dream/sleep-bound: invoked by the Layer-9 reflective audit, which only runs
    during dream/sleep modes — so counterfactuals are computed "during dream
    cycles" per the spec.
  - DATA-GATED: dormant until >= MIN_OUTCOMES lived outcomes exist AND
    > MIN_BUFFER distinct decisions have been evaluated. Until then it silently
    accumulates toward the gate and emits nothing — gate-blocked by design.
  - lived-before-synthetic: a counterfactual regret is a SYNTHETIC estimate, so
    it NEVER writes into the live policy reward. The shadow_reward accumulator is
    observability only (``live_influence=False``). Promoting it to live policy
    influence is an authority step that must be earned + explicitly enabled.

The counterfactual is grounded, not invented: the estimated value of an
alternative tool is the *measured* average net_delta of past decisions that
shared a topic tag and used that tool. With no lived history there is nothing to
estimate from — which is exactly why the engine is data-gated.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.counterfactual")

# ── Data gate (#17: "200+ outcomes, buffer >500") ──
MIN_OUTCOMES = 200            # lived non-warmup outcomes that must exist
MIN_BUFFER = 500              # distinct decisions that must have been evaluated

# ── Counterfactual credibility thresholds ──
MIN_ALT_SAMPLES = 5                 # peer samples required for an alternative to count
SAMPLES_FOR_FULL_CONFIDENCE = 20    # peer samples that map to confidence 1.0
MIN_CONFIDENCE = 0.6                 # below this, the counterfactual is too thin to emit
REGRET_THRESHOLD = 0.05             # min (alt_avg - actual) to call it a missed opportunity

BUFFER_MAXLEN = 1000
WARNING_REGRET = 0.15               # findings at/above this regret escalate info -> warning

STATE_PATH = os.path.join(os.path.expanduser("~"), ".jarvis", "counterfactual_state.json")


@dataclass(frozen=True)
class CounterfactualFinding:
    """One 'missed opportunity': an alternative that historically beats the choice made."""

    decision_intent_id: str
    decision_intent_type: str
    actual_tool: str
    actual_delta: float
    alternative_tool: str
    alternative_estimated_delta: float
    regret: float
    confidence: float
    topic_tags: tuple[str, ...]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_intent_id": self.decision_intent_id,
            "decision_intent_type": self.decision_intent_type,
            "actual_tool": self.actual_tool,
            "actual_delta": round(self.actual_delta, 4),
            "alternative_tool": self.alternative_tool,
            "alternative_estimated_delta": round(self.alternative_estimated_delta, 4),
            "regret": round(self.regret, 4),
            "confidence": round(self.confidence, 3),
            "topic_tags": list(self.topic_tags),
            "timestamp": self.timestamp,
        }


class CounterfactualEngine:
    """Singleton counterfactual evaluator. Read-only, no-LLM, data-gated."""

    _instance: CounterfactualEngine | None = None

    def __init__(self) -> None:
        self._total_evaluations: int = 0       # distinct decisions ever evaluated
        self._last_evaluated_ts: float = 0.0   # dedup watermark (persisted)
        self._missed_opportunity_count: int = 0
        self._shadow_reward_sum: float = 0.0   # synthetic; live_influence=False
        self._lived_outcomes: int = 0          # last observed non-warmup count
        self._findings: deque[CounterfactualFinding] = deque(maxlen=BUFFER_MAXLEN)
        self._last_run_ts: float = 0.0
        self._load()

    @classmethod
    def get_instance(cls) -> CounterfactualEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- Gate ---------------------------------------------------------------

    def is_active(self) -> bool:
        """True once both halves of the data gate are satisfied."""
        return (self._total_evaluations > MIN_BUFFER
                and self._lived_outcomes >= MIN_OUTCOMES)

    # -- Main entry ---------------------------------------------------------

    def evaluate(self, policy_memory: Any = None, now: float | None = None) -> list[CounterfactualFinding]:
        """Evaluate decisions recorded since the last watermark.

        Returns the list of qualifying missed-opportunity findings — but only
        when the data gate is open. While dormant it still advances the
        evaluation counter (earning toward the gate) and returns ``[]``.

        ``policy_memory`` is injectable for tests; otherwise a read-only
        ``AutonomyPolicyMemory`` is loaded from its persisted JSONL.
        """
        self._last_run_ts = now if now is not None else time.time()

        if policy_memory is None:
            try:
                from autonomy.policy_memory import AutonomyPolicyMemory
                policy_memory = AutonomyPolicyMemory()
            except Exception:
                logger.debug("counterfactual: policy memory unavailable", exc_info=True)
                return []

        outcomes = [o for o in getattr(policy_memory, "_outcomes", []) if not o.warmup]
        self._lived_outcomes = len(outcomes)

        new = [o for o in outcomes if o.timestamp > self._last_evaluated_ts]
        new.sort(key=lambda o: o.timestamp)

        qualifying: list[CounterfactualFinding] = []
        for d in new:
            self._total_evaluations += 1
            if d.timestamp > self._last_evaluated_ts:
                self._last_evaluated_ts = d.timestamp
            cf = self._evaluate_decision(d, outcomes)
            if cf is not None:
                qualifying.append(cf)

        emitted: list[CounterfactualFinding] = []
        if self.is_active():
            for cf in qualifying:
                self._findings.append(cf)
                self._missed_opportunity_count += 1
                # SHADOW signal only: regret is synthetic, so it informs
                # observability, never the live policy reward (live_influence=False).
                self._shadow_reward_sum += -cf.regret
                emitted.append(cf)

        self._save()
        return emitted

    # -- Counterfactual logic ----------------------------------------------

    def _evaluate_decision(
        self, d: Any, outcomes: list[Any],
    ) -> CounterfactualFinding | None:
        """Estimate the best alternative tool for *d*'s topic from lived history."""
        if not d.topic_tags:
            return None
        tagset = set(d.topic_tags)

        peers = [
            o for o in outcomes
            if (set(o.topic_tags) & tagset)
            and not (o.intent_id == d.intent_id and o.timestamp == d.timestamp)
        ]
        if len(peers) < MIN_ALT_SAMPLES:
            return None

        by_tool: dict[str, list[float]] = defaultdict(list)
        for o in peers:
            by_tool[o.tool_used].append(o.net_delta)

        best_tool: str | None = None
        best_avg: float = 0.0
        best_n: int = 0
        for tool, deltas in by_tool.items():
            if tool == d.tool_used or len(deltas) < MIN_ALT_SAMPLES:
                continue
            avg = sum(deltas) / len(deltas)
            if best_tool is None or avg > best_avg:
                best_tool, best_avg, best_n = tool, avg, len(deltas)

        if best_tool is None:
            return None

        regret = best_avg - d.net_delta
        if regret <= REGRET_THRESHOLD:
            return None

        confidence = min(1.0, best_n / SAMPLES_FOR_FULL_CONFIDENCE)
        if confidence < MIN_CONFIDENCE:
            return None

        return CounterfactualFinding(
            decision_intent_id=d.intent_id,
            decision_intent_type=d.intent_type,
            actual_tool=d.tool_used,
            actual_delta=d.net_delta,
            alternative_tool=best_tool,
            alternative_estimated_delta=best_avg,
            regret=regret,
            confidence=confidence,
            topic_tags=tuple(d.topic_tags),
            timestamp=d.timestamp,
        )

    # -- Observability ------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        active = self.is_active()
        reason = "ok" if active else (
            f"data_gated (evaluated {self._total_evaluations}/{MIN_BUFFER}, "
            f"outcomes {self._lived_outcomes}/{MIN_OUTCOMES})"
        )
        return {
            "active": active,
            "reason": reason,
            "total_evaluations": self._total_evaluations,
            "min_buffer": MIN_BUFFER,
            "lived_outcomes": self._lived_outcomes,
            "min_outcomes": MIN_OUTCOMES,
            "missed_opportunity_count": self._missed_opportunity_count,
            "shadow_reward_sum": round(self._shadow_reward_sum, 4),
            "live_influence": False,  # synthetic — never drives live policy until earned + enabled
            "recent_findings": [f.to_dict() for f in list(self._findings)[-5:]],
            "last_evaluated_ts": self._last_evaluated_ts,
            "last_run_ts": self._last_run_ts,
        }

    # -- Persistence --------------------------------------------------------

    def _save(self) -> None:
        data = {
            "total_evaluations": self._total_evaluations,
            "last_evaluated_ts": self._last_evaluated_ts,
            "missed_opportunity_count": self._missed_opportunity_count,
            "shadow_reward_sum": self._shadow_reward_sum,
        }
        try:
            path = Path(STATE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to save counterfactual state", exc_info=True)

    def _load(self) -> None:
        try:
            path = Path(STATE_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            self._total_evaluations = int(data.get("total_evaluations", 0))
            self._last_evaluated_ts = float(data.get("last_evaluated_ts", 0.0))
            self._missed_opportunity_count = int(data.get("missed_opportunity_count", 0))
            self._shadow_reward_sum = float(data.get("shadow_reward_sum", 0.0))
            if self._total_evaluations > 0:
                logger.info(
                    "Counterfactual state restored: evaluated=%d, missed_opps=%d",
                    self._total_evaluations, self._missed_opportunity_count,
                )
        except Exception:
            logger.debug("Failed to load counterfactual state", exc_info=True)


def get_counterfactual_engine() -> CounterfactualEngine:
    return CounterfactualEngine.get_instance()
