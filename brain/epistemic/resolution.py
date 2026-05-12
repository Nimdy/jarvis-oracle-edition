"""Per-class resolution strategies for the Contradiction Engine.

Phase 1: FactualResolution, TemporalResolution, IdentityTensionResolution
Phase 2: ProvenanceResolution, PolicyResolution, MultiPerspectiveResolution

Sacred Invariants:
- IdentityTensionResolution NEVER reduces belief_confidence
- IdentityTensionResolution ALWAYS produces debt_delta <= 0
- ProvenanceResolution NEVER merges or overwrites source distinction
- PolicyResolution outcome path NEVER touches factual belief confidence
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from epistemic.belief_record import (
    BeliefRecord,
    BeliefStore,
    ConflictClassification,
    ResolutionOutcome,
    TensionRecord,
    infer_tension_topic,
    DEBT_CRITICAL_UNRESOLVED,
    DEBT_FACTUAL_RESOLVED,
    DEBT_IDENTITY_TENSION,
    DEBT_MODERATE_UNRESOLVED,
    DEBT_MULTI_PERSPECTIVE,
    DEBT_POLICY_NORM,
    DEBT_TEMPORAL_VERSION,
    DEBT_TENSION_MATURED,
    MATURATION_REVISIT_DUAL_SIDED,
    MATURATION_REVISIT_NEW_EVIDENCE,
    MATURATION_REVISIT_PLAIN,
    TENSION_MAX_BELIEF_IDS,
)

logger = logging.getLogger(__name__)


def _record_ledger(
    event_type: str,
    belief_a: BeliefRecord,
    belief_b: BeliefRecord,
    classification: ConflictClassification,
    extra_data: dict[str, Any] | None = None,
) -> str:
    """Record to attribution ledger. Returns entry_id or empty string."""
    try:
        from consciousness.attribution_ledger import attribution_ledger
        data = {
            "conflict_type": classification.conflict_type,
            "severity": classification.severity,
            "is_pathological": classification.is_pathological,
            **(extra_data or {}),
        }
        return attribution_ledger.record(
            subsystem="epistemic",
            event_type=event_type,
            source="contradiction_engine",
            confidence=classification.confidence,
            evidence_refs=[
                {"type": "belief", "id": belief_a.belief_id},
                {"type": "belief", "id": belief_b.belief_id},
                {"type": "memory", "id": belief_a.source_memory_id},
                {"type": "memory", "id": belief_b.source_memory_id},
            ],
            data=data,
        )
    except Exception:
        logger.debug("Attribution ledger unavailable for epistemic event")
        return ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ResolutionStrategy(ABC):
    @abstractmethod
    def resolve(
        self,
        classification: ConflictClassification,
        belief_a: BeliefRecord,
        belief_b: BeliefRecord,
        belief_store: BeliefStore,
    ) -> ResolutionOutcome:
        ...


# ---------------------------------------------------------------------------
# Phase 1 strategies
# ---------------------------------------------------------------------------


class FactualResolution(ResolutionStrategy):
    """Compare provenance, evidence count, recency, belief_confidence.
    Downgrade the weaker. If both strong -> tension pending evidence."""

    def resolve(
        self, classification: ConflictClassification,
        belief_a: BeliefRecord, belief_b: BeliefRecord,
        belief_store: BeliefStore,
    ) -> ResolutionOutcome:
        stronger, weaker = self._rank(belief_a, belief_b)
        confidence_gap = stronger.belief_confidence - weaker.belief_confidence

        beliefs_modified = []
        confidence_deltas: dict[str, float] = {}

        if confidence_gap > 0.1:
            penalty = min(0.2, confidence_gap * 0.5)
            new_conf = max(0.05, weaker.belief_confidence - penalty)
            belief_store.update_belief_confidence(weaker.belief_id, new_conf)
            belief_store.update_resolution(weaker.belief_id, "superseded")
            beliefs_modified.append(weaker.belief_id)
            confidence_deltas[weaker.belief_id] = -penalty
            debt_delta = DEBT_FACTUAL_RESOLVED
        else:
            belief_store.update_resolution(belief_a.belief_id, "tension")
            belief_store.update_resolution(belief_b.belief_id, "tension")
            beliefs_modified = [belief_a.belief_id, belief_b.belief_id]
            debt_delta = DEBT_CRITICAL_UNRESOLVED

        belief_store.add_contradiction_link(belief_a.belief_id, belief_b.belief_id)
        belief_store.add_contradiction_link(belief_b.belief_id, belief_a.belief_id)

        ledger_id = _record_ledger(
            "contradiction_detected", belief_a, belief_b, classification,
            {"confidence_deltas": confidence_deltas, "resolution": "factual"},
        )
        return ResolutionOutcome(
            action_taken="confidence_adjusted" if confidence_gap > 0.1 else "tension_held",
            beliefs_modified=beliefs_modified,
            confidence_deltas=confidence_deltas,
            tension_id=None,
            ledger_entry_id=ledger_id,
            needs_user_clarification=False,
            clarification_prompt=None,
            debt_delta=debt_delta,
        )

    def _rank(self, a: BeliefRecord, b: BeliefRecord) -> tuple[BeliefRecord, BeliefRecord]:
        score_a = a.belief_confidence + len(a.evidence_refs) * 0.05 + (0.1 if a.timestamp > b.timestamp else 0)
        score_b = b.belief_confidence + len(b.evidence_refs) * 0.05 + (0.1 if b.timestamp > a.timestamp else 0)
        return (a, b) if score_a >= score_b else (b, a)


class TemporalResolution(ResolutionStrategy):
    """State change -> version with debt=0.
    Stable fact with overlap -> delegate to FactualResolution."""

    def __init__(self) -> None:
        self._factual = FactualResolution()

    def resolve(
        self, classification: ConflictClassification,
        belief_a: BeliefRecord, belief_b: BeliefRecord,
        belief_store: BeliefStore,
    ) -> ResolutionOutcome:
        if belief_a.is_state_belief or belief_b.is_state_belief:
            older, newer = (belief_a, belief_b) if belief_a.timestamp <= belief_b.timestamp else (belief_b, belief_a)
            belief_store.update_resolution(older.belief_id, "versioned")

            ledger_id = _record_ledger(
                "contradiction_resolved", belief_a, belief_b, classification,
                {"resolution": "temporal_version"},
            )
            return ResolutionOutcome(
                action_taken="belief_versioned",
                beliefs_modified=[older.belief_id],
                confidence_deltas={},
                tension_id=None,
                ledger_entry_id=ledger_id,
                needs_user_clarification=False,
                clarification_prompt=None,
                debt_delta=DEBT_TEMPORAL_VERSION,
            )

        return self._factual.resolve(classification, belief_a, belief_b, belief_store)


_STABLE_PARADOX_MIN_REVISITS = 50
_STABLE_PARADOX_MIN_MATURATION = 0.90

class IdentityTensionResolution(ResolutionStrategy):
    """Create or update TensionRecord. NEVER downgrade confidence. debt_delta=0."""

    def resolve(
        self, classification: ConflictClassification,
        belief_a: BeliefRecord, belief_b: BeliefRecord,
        belief_store: BeliefStore,
    ) -> ResolutionOutcome:
        topic = infer_tension_topic(belief_a, belief_b)
        existing = belief_store.get_tension_by_topic(topic)

        if existing is not None:
            if self._is_stable_paradox(existing):
                return ResolutionOutcome(
                    action_taken="stable_paradox",
                    beliefs_modified=[],
                    confidence_deltas={},
                    tension_id=existing.tension_id,
                    ledger_entry_id="",
                    needs_user_clarification=False,
                    clarification_prompt=None,
                    debt_delta=0.0,
                )

            if belief_a.belief_id not in existing.belief_ids:
                existing.belief_ids.append(belief_a.belief_id)
            if belief_b.belief_id not in existing.belief_ids:
                existing.belief_ids.append(belief_b.belief_id)

            mat_delta = self._update_maturation(existing, classification)
            self._enforce_fan_in_cap(existing, belief_store)
            belief_store.update_tension(existing)
            tension_id = existing.tension_id
            debt_delta = DEBT_TENSION_MATURED if mat_delta > 0 else DEBT_IDENTITY_TENSION
        else:
            from nanoid import generate as nanoid
            tension = TensionRecord(
                tension_id=f"ten_{nanoid(size=12)}",
                topic=topic,
                belief_ids=[belief_a.belief_id, belief_b.belief_id],
                conflict_key=classification.conflict_key,
                created_at=time.time(),
                last_revisited=time.time(),
                revisit_count=0,
                stability=0.5,
                maturation_score=0.0,
            )
            belief_store.add_tension(tension)
            tension_id = tension.tension_id
            debt_delta = DEBT_IDENTITY_TENSION

        belief_store.update_resolution(belief_a.belief_id, "tension")
        belief_store.update_resolution(belief_b.belief_id, "tension")
        belief_store.add_contradiction_link(belief_a.belief_id, belief_b.belief_id)
        belief_store.add_contradiction_link(belief_b.belief_id, belief_a.belief_id)

        ledger_id = _record_ledger(
            "tension_held", belief_a, belief_b, classification,
            {"tension_id": tension_id, "topic": topic},
        )
        return ResolutionOutcome(
            action_taken="tension_held",
            beliefs_modified=[belief_a.belief_id, belief_b.belief_id],
            confidence_deltas={},
            tension_id=tension_id,
            ledger_entry_id=ledger_id,
            needs_user_clarification=False,
            clarification_prompt=None,
            debt_delta=debt_delta,
        )

    def _update_maturation(self, tension: TensionRecord, classification: ConflictClassification) -> float:
        has_new_evidence = len(classification.reasoning) > 100
        has_dual_sided = any(
            kw in classification.reasoning.lower()
            for kw in ("both", "tension", "perspectives", "sides", "coexist")
        )

        if has_dual_sided:
            delta = MATURATION_REVISIT_DUAL_SIDED
        elif has_new_evidence:
            delta = MATURATION_REVISIT_NEW_EVIDENCE
        else:
            delta = MATURATION_REVISIT_PLAIN

        tension.maturation_score = min(1.0, tension.maturation_score + delta)
        tension.revisit_count += 1
        tension.last_revisited = time.time()
        return delta

    def _enforce_fan_in_cap(self, tension: TensionRecord, belief_store: BeliefStore) -> None:
        if len(tension.belief_ids) <= TENSION_MAX_BELIEF_IDS:
            return
        beliefs = [belief_store.get(bid) for bid in tension.belief_ids]
        beliefs = [b for b in beliefs if b is not None]
        beliefs.sort(key=lambda b: b.timestamp)
        overflow = beliefs[:len(beliefs) - TENSION_MAX_BELIEF_IDS]
        for b in overflow:
            belief_store.update_resolution(b.belief_id, "superseded")
            if b.belief_id in tension.belief_ids:
                tension.belief_ids.remove(b.belief_id)

    @staticmethod
    def _is_stable_paradox(tension: TensionRecord) -> bool:
        """A tension that has been thoroughly explored and is now stable."""
        return (
            tension.revisit_count >= _STABLE_PARADOX_MIN_REVISITS
            and tension.maturation_score >= _STABLE_PARADOX_MIN_MATURATION
        )


# ---------------------------------------------------------------------------
# Phase 2 strategies
# ---------------------------------------------------------------------------


class ProvenanceResolution(ResolutionStrategy):
    """Keep both. Mark source disagreement. Never merge sources.
    user_claim vs observed -> clarification flag above threshold."""

    def resolve(
        self, classification: ConflictClassification,
        belief_a: BeliefRecord, belief_b: BeliefRecord,
        belief_store: BeliefStore,
    ) -> ResolutionOutcome:
        needs_clarification = (
            {belief_a.provenance, belief_b.provenance} & {"user_claim", "observed"} == {"user_claim", "observed"}
            and min(belief_a.belief_confidence, belief_b.belief_confidence) > 0.4
        )
        prompt = None
        if needs_clarification:
            prompt = (
                f"I have two signals about {belief_a.canonical_subject}: "
                f"you said \"{belief_a.rendered_claim}\" but I observed \"{belief_b.rendered_claim}\". "
                "Would you like me to update my understanding?"
            )

        belief_store.add_contradiction_link(belief_a.belief_id, belief_b.belief_id)
        belief_store.add_contradiction_link(belief_b.belief_id, belief_a.belief_id)

        ledger_id = _record_ledger(
            "contradiction_detected", belief_a, belief_b, classification,
            {"resolution": "provenance_separation", "needs_clarification": needs_clarification},
        )
        return ResolutionOutcome(
            action_taken="source_separated",
            beliefs_modified=[],
            confidence_deltas={},
            tension_id=None,
            ledger_entry_id=ledger_id,
            needs_user_clarification=needs_clarification,
            clarification_prompt=prompt,
            debt_delta=DEBT_CRITICAL_UNRESOLVED,
        )


class PolicyResolution(ResolutionStrategy):
    """Split by subtype. Outcome -> policy memory penalty. Norm -> record shift.
    Hard wall: outcome never penalizes norms, norm never penalizes actions."""

    def resolve(
        self, classification: ConflictClassification,
        belief_a: BeliefRecord, belief_b: BeliefRecord,
        belief_store: BeliefStore,
    ) -> ResolutionOutcome:
        is_norm = classification.conflict_type == "policy_norm"

        belief_store.add_contradiction_link(belief_a.belief_id, belief_b.belief_id)
        belief_store.add_contradiction_link(belief_b.belief_id, belief_a.belief_id)

        if is_norm:
            ledger_id = _record_ledger(
                "contradiction_detected", belief_a, belief_b, classification,
                {"resolution": "policy_norm_noted"},
            )
            return ResolutionOutcome(
                action_taken="norm_noted",
                beliefs_modified=[],
                confidence_deltas={},
                tension_id=None,
                ledger_entry_id=ledger_id,
                needs_user_clarification=False,
                clarification_prompt=None,
                debt_delta=DEBT_POLICY_NORM,
            )
        else:
            weaker = belief_a if belief_a.belief_confidence <= belief_b.belief_confidence else belief_b
            penalty = min(0.15, weaker.belief_confidence * 0.3)
            new_conf = max(0.05, weaker.belief_confidence - penalty)
            belief_store.update_belief_confidence(weaker.belief_id, new_conf)

            ledger_id = _record_ledger(
                "contradiction_detected", belief_a, belief_b, classification,
                {"resolution": "policy_outcome_penalty", "penalized": weaker.belief_id},
            )
            return ResolutionOutcome(
                action_taken="policy_penalized",
                beliefs_modified=[weaker.belief_id],
                confidence_deltas={weaker.belief_id: -penalty},
                tension_id=None,
                ledger_entry_id=ledger_id,
                needs_user_clarification=False,
                clarification_prompt=None,
                debt_delta=DEBT_MODERATE_UNRESOLVED,
            )


class MultiPerspectiveResolution(ResolutionStrategy):
    """Mark both as tension. Track evidence. debt_delta=0."""

    def resolve(
        self, classification: ConflictClassification,
        belief_a: BeliefRecord, belief_b: BeliefRecord,
        belief_store: BeliefStore,
    ) -> ResolutionOutcome:
        belief_store.update_resolution(belief_a.belief_id, "tension")
        belief_store.update_resolution(belief_b.belief_id, "tension")
        belief_store.add_contradiction_link(belief_a.belief_id, belief_b.belief_id)
        belief_store.add_contradiction_link(belief_b.belief_id, belief_a.belief_id)

        ledger_id = _record_ledger(
            "tension_held", belief_a, belief_b, classification,
            {"resolution": "multi_perspective"},
        )
        return ResolutionOutcome(
            action_taken="tension_held",
            beliefs_modified=[belief_a.belief_id, belief_b.belief_id],
            confidence_deltas={},
            tension_id=None,
            ledger_entry_id=ledger_id,
            needs_user_clarification=False,
            clarification_prompt=None,
            debt_delta=DEBT_MULTI_PERSPECTIVE,
        )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_MAP: dict[str, ResolutionStrategy] = {
    "factual": FactualResolution(),
    "provenance": ProvenanceResolution(),
    "policy_outcome": PolicyResolution(),
    "policy_norm": PolicyResolution(),
    "temporal": TemporalResolution(),
    "identity_tension": IdentityTensionResolution(),
    "multi_perspective": MultiPerspectiveResolution(),
}


def resolve_conflict(
    classification: ConflictClassification,
    belief_a: BeliefRecord,
    belief_b: BeliefRecord,
    belief_store: BeliefStore,
) -> ResolutionOutcome:
    strategy = STRATEGY_MAP.get(classification.conflict_type)
    if strategy is None:
        logger.warning("No strategy for conflict type: %s", classification.conflict_type)
        return ResolutionOutcome(
            action_taken="unhandled",
            beliefs_modified=[],
            confidence_deltas={},
            tension_id=None,
            ledger_entry_id="",
            needs_user_clarification=False,
            clarification_prompt=None,
            debt_delta=0.0,
        )
    return strategy.resolve(classification, belief_a, belief_b, belief_store)
