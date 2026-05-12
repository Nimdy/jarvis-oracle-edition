"""6-class conflict classifier with modality-aware pre-gates and near-miss logging.

Classifies belief pairs into: factual, provenance, policy_outcome, policy_norm,
temporal, identity_tension, or multi_perspective. Returns None for near-misses.
"""

from __future__ import annotations

import time
from collections import deque

from epistemic.belief_record import (
    BeliefRecord,
    ConflictClassification,
    NearMiss,
    EXTRACTION_NEAR_MISS_THRESHOLD,
    NEAR_MISS_RING_BUFFER_SIZE,
)

# Opposing predicate families — predicates that semantically contradict
_OPPOSING_PREDICATES: dict[str, str] = {
    "improves": "degrades",
    "degrades": "improves",
    "caused": "prevented",
    "prevented": "caused",
    "is": "is",
    "prefers": "prefers",
}


class ConflictClassifier:
    def __init__(self) -> None:
        self._near_misses: deque[NearMiss] = deque(maxlen=NEAR_MISS_RING_BUFFER_SIZE)
        self._total_checks: int = 0
        self._total_classifications: int = 0
        self._type_counts: dict[str, int] = {}

    def classify(self, a: BeliefRecord, b: BeliefRecord) -> ConflictClassification | None:
        """Classify a pair of beliefs. Returns None for near-misses."""
        self._total_checks += 1
        severity_downgrade = 0

        # Pre-gate 0: Cross-identity boundary (Layer 3)
        a_subj = getattr(a, "identity_subject_id", "")
        b_subj = getattr(b, "identity_subject_id", "")
        if a_subj and b_subj and a_subj != b_subj:
            self._log_near_miss(a, b, "identity_boundary")
            return None

        # Pre-gate 1: Same source memory
        if a.source_memory_id and b.source_memory_id and a.source_memory_id == b.source_memory_id:
            self._log_near_miss(a, b, "same_source_memory")
            return None

        # Pre-gate 2: Different modality on same subject
        if a.canonical_subject == b.canonical_subject and a.modality != b.modality:
            self._log_near_miss(a, b, "different_modality")
            return None

        # Pre-gate 3: question stance vs any other
        if a.stance == "question" or b.stance == "question":
            self._log_near_miss(a, b, "question_vs_assertion")
            return None

        # Pre-gate 4: uncertain vs assert
        if (a.stance == "uncertain") != (b.stance == "uncertain"):
            if a.claim_type == "factual" and b.claim_type == "factual":
                severity_downgrade += 1
            else:
                self._log_near_miss(a, b, "uncertain_vs_assert")
                return None

        # Pre-gate 5: provisional epistemic status -> downgrade severity
        if a.epistemic_status == "provisional" or b.epistemic_status == "provisional":
            severity_downgrade += 1

        # Pre-gate 6: Low extraction confidence
        if a.extraction_confidence < EXTRACTION_NEAR_MISS_THRESHOLD or b.extraction_confidence < EXTRACTION_NEAR_MISS_THRESHOLD:
            self._log_near_miss(a, b, "low_extraction_confidence")
            return None

        # Type-specific classification
        result = self._classify_typed(a, b, severity_downgrade)
        if result is not None:
            self._total_classifications += 1
            self._type_counts[result.conflict_type] = self._type_counts.get(result.conflict_type, 0) + 1
        return result

    def _classify_typed(
        self, a: BeliefRecord, b: BeliefRecord, severity_downgrade: int,
    ) -> ConflictClassification | None:
        # Identity / philosophical always routes to identity_tension
        if a.claim_type in ("identity", "philosophical") or b.claim_type in ("identity", "philosophical"):
            return self._classify_identity(a, b, severity_downgrade)

        # Temporal gate: state beliefs or overlapping time ranges on same subject
        if a.is_state_belief or b.is_state_belief:
            return self._classify_temporal(a, b, severity_downgrade)

        if (a.canonical_subject == b.canonical_subject
                and a.time_range is not None and b.time_range is not None):
            return self._classify_temporal(a, b, severity_downgrade)

        # Policy
        if a.claim_type == "policy" and b.claim_type == "policy":
            return self._classify_policy(a, b, severity_downgrade)

        # Provenance conflict
        if (a.canonical_subject == b.canonical_subject
                and a.canonical_object == b.canonical_object
                and a.provenance != b.provenance
                and self._conclusions_incompatible(a, b)):
            return self._classify_provenance(a, b, severity_downgrade)

        # Factual
        if a.claim_type == "factual" and b.claim_type == "factual":
            return self._classify_factual(a, b, severity_downgrade)

        # Multi-perspective fallback
        if (a.claim_type == "factual" and b.claim_type == "factual"
                and a.scope and b.scope and a.scope != b.scope):
            return self._classify_multi_perspective(a, b, severity_downgrade)

        # Check for cross-type factual-like conflicts on same subject+object
        if (a.canonical_subject == b.canonical_subject
                and a.canonical_object == b.canonical_object
                and self._conclusions_incompatible(a, b)):
            if a.scope and b.scope and a.scope != b.scope:
                return self._classify_multi_perspective(a, b, severity_downgrade)
            return self._classify_factual(a, b, severity_downgrade)

        self._log_near_miss(a, b, "same_conclusion")
        return None

    def _classify_factual(
        self, a: BeliefRecord, b: BeliefRecord, severity_downgrade: int,
    ) -> ConflictClassification | None:
        same_subject = a.canonical_subject == b.canonical_subject
        same_object = a.canonical_object == b.canonical_object
        same_modality = a.modality == b.modality

        if not (same_subject and same_object and same_modality):
            # Different scope -> multi_perspective
            if same_subject and a.scope and b.scope and a.scope != b.scope:
                return self._classify_multi_perspective(a, b, severity_downgrade)
            self._log_near_miss(a, b, "different_scope")
            return None

        opposite_polarity = (a.polarity != 0 and b.polarity != 0 and a.polarity != b.polarity)
        opposing_family = _OPPOSING_PREDICATES.get(a.canonical_predicate) == b.canonical_predicate

        if not (opposite_polarity or opposing_family):
            self._log_near_miss(a, b, "same_conclusion")
            return None

        severity = _downgrade_severity("critical", severity_downgrade)
        conflict_key = a.conflict_key or b.conflict_key
        reasoning = (
            f"Factual contradiction: {a.rendered_claim} vs {b.rendered_claim}. "
            f"{'Opposite polarity' if opposite_polarity else 'Opposing predicate families'}."
        )
        return ConflictClassification(
            conflict_type="factual",
            severity=severity,
            is_pathological=True,
            confidence=min(a.belief_confidence, b.belief_confidence),
            reasoning=reasoning,
            conflict_key=conflict_key,
        )

    def _classify_provenance(
        self, a: BeliefRecord, b: BeliefRecord, severity_downgrade: int,
    ) -> ConflictClassification:
        severity = _downgrade_severity("critical", severity_downgrade)
        conflict_key = a.conflict_key or b.conflict_key
        return ConflictClassification(
            conflict_type="provenance",
            severity=severity,
            is_pathological=True,
            confidence=min(a.belief_confidence, b.belief_confidence),
            reasoning=f"Provenance conflict: {a.provenance} vs {b.provenance} on {a.canonical_subject}",
            conflict_key=conflict_key,
        )

    def _classify_policy(
        self, a: BeliefRecord, b: BeliefRecord, severity_downgrade: int,
    ) -> ConflictClassification | None:
        if a.canonical_subject != b.canonical_subject:
            self._log_near_miss(a, b, "different_action")
            return None

        # Distinguish outcome vs norm
        a_is_norm = a.modality == "should"
        b_is_norm = b.modality == "should"

        if a_is_norm and b_is_norm:
            severity = _downgrade_severity("moderate", severity_downgrade)
            return ConflictClassification(
                conflict_type="policy_norm",
                severity=severity,
                is_pathological=False,
                confidence=min(a.belief_confidence, b.belief_confidence),
                reasoning=f"Policy norm conflict on {a.canonical_subject}: {a.rendered_claim} vs {b.rendered_claim}",
                conflict_key=a.conflict_key or b.conflict_key,
            )

        if not a_is_norm and not b_is_norm:
            if not self._conclusions_incompatible(a, b):
                self._log_near_miss(a, b, "same_conclusion")
                return None
            severity = _downgrade_severity("critical", severity_downgrade)
            return ConflictClassification(
                conflict_type="policy_outcome",
                severity=severity,
                is_pathological=True,
                confidence=min(a.belief_confidence, b.belief_confidence),
                reasoning=f"Policy outcome conflict on {a.canonical_subject}: {a.rendered_claim} vs {b.rendered_claim}",
                conflict_key=a.conflict_key or b.conflict_key,
            )

        # Mixed: one norm, one outcome -> near-miss (they don't conflict across the wall)
        self._log_near_miss(a, b, "policy_norm_vs_outcome")
        return None

    def _classify_temporal(
        self, a: BeliefRecord, b: BeliefRecord, severity_downgrade: int,
    ) -> ConflictClassification | None:
        if a.is_state_belief or b.is_state_belief:
            severity = _downgrade_severity("moderate", severity_downgrade)
            return ConflictClassification(
                conflict_type="temporal",
                severity=severity,
                is_pathological=False,
                confidence=min(a.belief_confidence, b.belief_confidence),
                reasoning=f"State-belief temporal conflict on {a.canonical_subject}",
                conflict_key=a.conflict_key or b.conflict_key,
            )

        # Both stable facts with overlapping time ranges
        if a.time_range and b.time_range:
            if _time_ranges_overlap(a.time_range, b.time_range):
                severity = _downgrade_severity("critical", severity_downgrade)
                return ConflictClassification(
                    conflict_type="temporal",
                    severity=severity,
                    is_pathological=True,
                    confidence=min(a.belief_confidence, b.belief_confidence),
                    reasoning=f"Temporal stable-fact conflict on {a.canonical_subject}",
                    conflict_key=a.conflict_key or b.conflict_key,
                )
            else:
                self._log_near_miss(a, b, "non_overlapping_time")
                return None

        self._log_near_miss(a, b, "state_change_version")
        return None

    def _classify_identity(
        self, a: BeliefRecord, b: BeliefRecord, severity_downgrade: int,
    ) -> ConflictClassification:
        conflict_key = a.conflict_key or b.conflict_key
        return ConflictClassification(
            conflict_type="identity_tension",
            severity="informational",
            is_pathological=False,
            confidence=min(a.belief_confidence, b.belief_confidence),
            reasoning=(
                f"Identity tension between {a.rendered_claim} and {b.rendered_claim}. "
                "Both perspectives preserved as productive tension."
            ),
            conflict_key=conflict_key,
        )

    def _classify_multi_perspective(
        self, a: BeliefRecord, b: BeliefRecord, severity_downgrade: int,
    ) -> ConflictClassification:
        conflict_key = a.conflict_key or b.conflict_key
        return ConflictClassification(
            conflict_type="multi_perspective",
            severity="informational",
            is_pathological=False,
            confidence=min(a.belief_confidence, b.belief_confidence),
            reasoning=f"Multi-perspective: {a.scope or 'default'} vs {b.scope or 'default'} on {a.canonical_subject}",
            conflict_key=conflict_key,
        )

    # -- Helpers ------------------------------------------------------------

    def _conclusions_incompatible(self, a: BeliefRecord, b: BeliefRecord) -> bool:
        if a.polarity != 0 and b.polarity != 0 and a.polarity != b.polarity:
            return True
        if _OPPOSING_PREDICATES.get(a.canonical_predicate) == b.canonical_predicate:
            return True
        if a.canonical_object != b.canonical_object:
            return True
        return False

    def _log_near_miss(self, a: BeliefRecord, b: BeliefRecord, reason: str) -> None:
        self._near_misses.append(NearMiss(
            timestamp=time.time(),
            belief_a_id=a.belief_id,
            belief_b_id=b.belief_id,
            reason=reason,
            subject=a.canonical_subject,
        ))

    # -- State --------------------------------------------------------------

    def get_near_misses(self) -> list[NearMiss]:
        return list(self._near_misses)

    def get_stats(self) -> dict:
        total = self._total_checks
        nm = total - self._total_classifications
        return {
            "total_checks": total,
            "total_classifications": self._total_classifications,
            "near_miss_count": nm,
            "near_miss_rate": nm / total if total > 0 else 0.0,
            "by_type": dict(self._type_counts),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = ["critical", "moderate", "informational"]


def _downgrade_severity(base: str, levels: int) -> str:
    if levels <= 0:
        return base
    try:
        idx = _SEVERITY_ORDER.index(base)
    except ValueError:
        return base
    new_idx = min(idx + levels, len(_SEVERITY_ORDER) - 1)
    return _SEVERITY_ORDER[new_idx]


def _time_ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]
