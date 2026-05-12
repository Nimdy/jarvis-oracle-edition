"""Core Contradiction Engine — detect, classify, resolve, track debt.

Singleton that subscribes to MEMORY_WRITE, extracts claims, checks for
contradictions, and maintains contradiction_debt as a top-line epistemic
health signal.

Sacred Invariants:
- Hot-path (MEMORY_WRITE -> check) NEVER calls semantic_search
- scan_corpus() is only called from consciousness tick cycle
- Debt is always in [0.0, 1.0]
- Discarded beliefs (extraction_confidence < 0.2) are invisible
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from epistemic.belief_record import (
    BeliefStore,
    BeliefRecord,
    ResolutionOutcome,
    DEBT_MIN,
    DEBT_MAX,
    DEBT_PASSIVE_DECAY_PER_HOUR,
    DEBT_RECURRENCE_EXTRA,
    DEBT_TREND_WINDOW,
    EXTRACTION_DISCARD_THRESHOLD,
)
from epistemic.claim_extractor import extract_claims
from epistemic.conflict_classifier import ConflictClassifier
from epistemic.resolution import resolve_conflict

logger = logging.getLogger(__name__)

BELIEF_EXTRACTION_MIN_WEIGHT = 0.20
_BELIEF_INELIGIBLE_TYPES = frozenset({"error_recovery"})
_BELIEF_LOW_WEIGHT_TAGS = frozenset({
    "self_reflection", "interaction_review", "ambient_sound",
    "app_switch", "user_left",
})

# Dream-origin material is categorically non-belief-bearing, regardless of weight.
_DREAM_INELIGIBLE_TAGS = frozenset({
    "dream_insight",
    "dream_hypothesis",
    "sleep_candidate",
    "dream_artifact",
    "dream_consolidation",
})


class ContradictionEngine:
    _instance: ContradictionEngine | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._belief_store = BeliefStore()
        self._classifier = ConflictClassifier()

        self._contradiction_debt: float = 0.0
        self._debt_trend: deque[float] = deque(maxlen=DEBT_TREND_WINDOW)
        self._last_decay_time: float = time.time()

        self._extraction_discard_count: int = 0
        self._version_collapses: int = 0
        self._total_resolutions: int = 0
        self._seen_conflict_keys: set[str] = set()

        self._by_type: dict[str, int] = {}
        self._by_resolution_action: dict[str, int] = {}

        self._subscribed = False

    @classmethod
    def get_instance(cls) -> ContradictionEngine:
        if cls._instance is None:
            cls._instance = ContradictionEngine()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # -- Event subscription -------------------------------------------------

    def subscribe(self) -> None:
        if self._subscribed:
            return
        try:
            from consciousness.events import event_bus, MEMORY_WRITE
            event_bus.on(MEMORY_WRITE, self._on_memory_write)
            self._subscribed = True
            logger.info("ContradictionEngine subscribed to MEMORY_WRITE")
        except Exception:
            logger.debug("Could not subscribe to MEMORY_WRITE (bus not ready)")

    def _is_belief_eligible(self, memory: Any) -> bool:
        """Gate: only extract beliefs from memories with sufficient epistemic weight."""
        mem_type = getattr(memory, "type", "")
        if mem_type in _BELIEF_INELIGIBLE_TYPES:
            return False
        tags = set(getattr(memory, "tags", ()))
        if tags & _DREAM_INELIGIBLE_TAGS:
            return False
        if "quarantine:suspect" in tags:
            return False
        weight = getattr(memory, "weight", 0.0)
        if weight < BELIEF_EXTRACTION_MIN_WEIGHT:
            return False
        if tags & _BELIEF_LOW_WEIGHT_TAGS and weight < 0.30:
            return False
        return True

    def _on_memory_write(self, **kwargs: Any) -> None:
        memory = kwargs.get("memory")
        if memory is None:
            return
        if not self._is_belief_eligible(memory):
            return
        try:
            claims = extract_claims(memory)
            pending_emissions: list[tuple] = []
            pending_graph_notifications: list[BeliefRecord] = []
            pending_graph_versions: list[tuple[str, str]] = []

            with self._lock:
                for claim in claims:
                    if claim.extraction_confidence < EXTRACTION_DISCARD_THRESHOLD:
                        self._extraction_discard_count += 1
                        continue
                    collapsed, graph_notifs, version_notifs = self._try_version_collapse_deferred(claim)
                    if collapsed:
                        pending_graph_notifications.extend(graph_notifs)
                        pending_graph_versions.extend(version_notifs)
                        continue
                    self._belief_store.add(claim)
                    outcomes = self._check_new_belief_deferred(claim)
                    pending_emissions.extend(outcomes)
                    pending_graph_notifications.append(claim)

            for classification, outcome in pending_emissions:
                self._emit_event(classification, outcome)
            for belief in pending_graph_notifications:
                self._notify_belief_graph(belief)
            for old_id, new_id in pending_graph_versions:
                self._notify_belief_graph_version(old_id, new_id)
        except Exception:
            logger.exception("ContradictionEngine._on_memory_write error")

    def _try_version_collapse(self, claim: BeliefRecord) -> bool:
        """Try to collapse a new claim into an existing same-subject version.

        Returns True if the claim was collapsed (either superseded an old one
        or was itself suppressed). Returns False if the claim is genuinely new.

        NOTE: This method emits events inline. Only use from paths that do NOT
        hold self._lock. For the hot path (_on_memory_write), use
        _try_version_collapse_deferred() instead.
        """
        collapsed, graph_notifs, version_notifs = self._try_version_collapse_deferred(claim)
        for belief in graph_notifs:
            self._notify_belief_graph(belief)
        for old_id, new_id in version_notifs:
            self._notify_belief_graph_version(old_id, new_id)
        return collapsed

    def _try_version_collapse_deferred(
        self, claim: BeliefRecord,
    ) -> tuple[bool, list[BeliefRecord], list[tuple[str, str]]]:
        """Lock-safe version collapse that defers graph notifications.

        Returns (collapsed, graph_notifications, version_notifications).
        Must be called while self._lock is held.
        """
        graph_notifs: list[BeliefRecord] = []
        version_notifs: list[tuple[str, str]] = []

        existing = self._belief_store.find_by_subject(claim.canonical_subject)
        for old in existing:
            if old.resolution_state != "active":
                continue
            if old.canonical_predicate != claim.canonical_predicate:
                continue
            if old.provenance != claim.provenance:
                continue

            if old.canonical_object == claim.canonical_object:
                self._belief_store.update_resolution(old.belief_id, "superseded")
                self._belief_store.add(claim)
                graph_notifs.append(claim)
                version_notifs.append((old.belief_id, claim.belief_id))
                self._version_collapses += 1
                logger.debug(
                    "Version collapse (identical): superseded %s with %s [%s / %s]",
                    old.belief_id, claim.belief_id,
                    claim.canonical_subject, claim.canonical_predicate,
                )
                return True, graph_notifs, version_notifs

            if claim.extraction_confidence >= old.extraction_confidence:
                self._belief_store.update_resolution(old.belief_id, "superseded")
                self._belief_store.add(claim)
                outcomes = self._check_new_belief_deferred(claim)
                graph_notifs.append(claim)
                version_notifs.append((old.belief_id, claim.belief_id))
                self._version_collapses += 1
                logger.debug(
                    "Version collapse (updated): superseded %s with %s [%s / %s: %s -> %s]",
                    old.belief_id, claim.belief_id,
                    claim.canonical_subject, claim.canonical_predicate,
                    old.canonical_object, claim.canonical_object,
                )
                return True, graph_notifs, version_notifs
            else:
                self._version_collapses += 1
                logger.debug(
                    "Version collapse (suppressed): kept %s, discarded new [%s / %s] (conf %.2f < %.2f)",
                    old.belief_id, claim.canonical_subject, claim.canonical_predicate,
                    claim.extraction_confidence, old.extraction_confidence,
                )
                return True, graph_notifs, version_notifs

        return False, graph_notifs, version_notifs

    def _notify_belief_graph_version(self, old_belief_id: str, new_belief_id: str) -> None:
        """Notify Layer 7 that a belief was versioned (superseded by a newer claim)."""
        try:
            from epistemic.belief_graph import BeliefGraph
            graph = BeliefGraph.get_instance()
            if graph is not None:
                graph.on_belief_versioned(old_belief_id, new_belief_id)
        except Exception:
            pass

    def _notify_belief_graph(self, belief: BeliefRecord) -> None:
        """Notify Layer 7 of a new active belief for edge creation."""
        try:
            from epistemic.belief_graph import BeliefGraph
            graph = BeliefGraph.get_instance()
            if graph is not None:
                graph.on_new_belief(belief)
        except Exception:
            pass

    # -- Core API -----------------------------------------------------------

    def check_new_belief(self, belief: BeliefRecord) -> list[ResolutionOutcome]:
        """Hot-path check: subject-match only, no semantic search. < 5ms budget.

        NOTE: This method emits events inline. Only use from paths that do NOT
        hold self._lock. For the hot path (_on_memory_write), use
        _check_new_belief_deferred() instead.
        """
        deferred = self._check_new_belief_deferred(belief)
        for classification, outcome in deferred:
            self._emit_event(classification, outcome)
        return [outcome for _, outcome in deferred]

    def _check_new_belief_deferred(self, belief: BeliefRecord) -> list[tuple[Any, ResolutionOutcome]]:
        """Lock-safe belief check that defers event emission.

        Returns list of (classification, outcome) tuples for deferred emission.
        Must be called while self._lock is held.
        """
        candidates = self._belief_store.find_by_subject(belief.canonical_subject)
        pending: list[tuple[Any, ResolutionOutcome]] = []

        for candidate in candidates:
            if candidate.belief_id == belief.belief_id:
                continue
            if candidate.resolution_state in ("superseded", "resolved", "quarantined"):
                continue

            classification = self._classifier.classify(belief, candidate)
            if classification is None:
                continue

            outcome = resolve_conflict(classification, belief, candidate, self._belief_store)
            self._update_debt(outcome.debt_delta, classification.conflict_key)
            self._total_resolutions += 1
            self._by_type[classification.conflict_type] = self._by_type.get(classification.conflict_type, 0) + 1
            self._by_resolution_action[outcome.action_taken] = self._by_resolution_action.get(outcome.action_taken, 0) + 1

            pending.append((classification, outcome))

        return pending

    def scan_corpus(self) -> list[ResolutionOutcome]:
        """Cold-path full scan. Only called from consciousness tick cycle.

        Uses per-cycle topic collapse: once a topic has produced a resolution
        outcome in this scan, all further pairs sharing that topic are skipped.
        This prevents combinatorial explosion on subjects with many beliefs
        (e.g. identity-related beliefs producing O(n²) tension_held events).
        """
        outcomes: list[ResolutionOutcome] = []

        with self._lock:
            beliefs = list(self._belief_store._beliefs.values())

        active = [b for b in beliefs if b.resolution_state == "active"]
        checked_pairs: set[tuple[str, str]] = set()
        topics_resolved_this_cycle: set[str] = set()

        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if a.canonical_subject != b.canonical_subject:
                    continue

                if a.canonical_subject in topics_resolved_this_cycle:
                    continue

                pair_key = tuple(sorted([a.belief_id, b.belief_id]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                classification = self._classifier.classify(a, b)
                if classification is None:
                    continue

                outcome = resolve_conflict(classification, a, b, self._belief_store)

                if outcome.action_taken == "stable_paradox":
                    topics_resolved_this_cycle.add(a.canonical_subject)
                    continue

                with self._lock:
                    self._update_debt(outcome.debt_delta, classification.conflict_key)
                    self._total_resolutions += 1
                    self._by_type[classification.conflict_type] = self._by_type.get(classification.conflict_type, 0) + 1
                    self._by_resolution_action[outcome.action_taken] = self._by_resolution_action.get(outcome.action_taken, 0) + 1

                self._emit_event(classification, outcome)
                outcomes.append(outcome)

                if outcome.action_taken == "tension_held":
                    topics_resolved_this_cycle.add(a.canonical_subject)

        return outcomes

    def apply_passive_decay(self) -> None:
        """Called from the tick cycle. Applies hourly debt decay.

        Stable paradoxes (identity tensions with high revisit count and
        maturation) do NOT block debt decay — only genuinely unresolved
        active contradictions do.
        """
        with self._lock:
            now = time.time()
            elapsed_hours = (now - self._last_decay_time) / 3600.0
            self._last_decay_time = now

            if elapsed_hours < 0.001:
                return

            stable_tension_bids = self._get_stable_paradox_belief_ids()
            has_active_pathological = any(
                b.resolution_state == "active"
                and b.contradicts
                and b.belief_id not in stable_tension_bids
                for b in self._belief_store._beliefs.values()
            )
            if not has_active_pathological:
                decay = DEBT_PASSIVE_DECAY_PER_HOUR * elapsed_hours
                self._contradiction_debt = max(DEBT_MIN, self._contradiction_debt + decay)

            self._debt_trend.append(self._contradiction_debt)

    def apply_correction_debt(self) -> None:
        """Apply debt increase from a user correction (Layer 6 CorrectionDetector)."""
        from epistemic.belief_record import DEBT_USER_CORRECTION, DEBT_MAX
        with self._lock:
            self._contradiction_debt = min(DEBT_MAX, self._contradiction_debt + DEBT_USER_CORRECTION)
            self._debt_trend.append(self._contradiction_debt)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            debt = self._contradiction_debt
            debt_trend = list(self._debt_trend)
            total_res = self._total_resolutions
            by_type = dict(self._by_type)
            by_res = dict(self._by_resolution_action)
            disc = self._extraction_discard_count
            collapses = self._version_collapses
            beliefs_snapshot = list(self._belief_store._beliefs.values())

        stats = self._belief_store.get_stats()
        classifier_stats = self._classifier.get_stats()
        tensions = self._belief_store.get_active_tensions()

        recent_near_misses = [
            {
                "timestamp": nm.timestamp,
                "belief_a_id": nm.belief_a_id,
                "belief_b_id": nm.belief_b_id,
                "reason": nm.reason,
                "subject": nm.subject,
            }
            for nm in self._classifier.get_near_misses()[-20:]
        ]

        conflict_key_counts: dict[str, int] = {}
        for b in beliefs_snapshot:
            # Surface only currently active pathological conflict keys here.
            # Identity tensions and superseded/resolved beliefs have their own
            # dedicated views and should not pollute "recurring conflict keys".
            if b.resolution_state != "active":
                continue
            if not b.contradicts:
                continue
            conflict_key = (b.conflict_key or "").strip()
            if not conflict_key or conflict_key.startswith("identity::"):
                continue
            conflict_key_counts[conflict_key] = conflict_key_counts.get(conflict_key, 0) + 1

        stable_paradox_count = sum(
            1 for t in tensions
            if t.revisit_count >= 50 and t.maturation_score >= 0.90
        )

        return {
            "total_beliefs": stats["total_beliefs"],
            "active_beliefs": stats["active_beliefs"],
            "active_tensions": stats["active_tensions"],
            "stable_paradoxes": stable_paradox_count,
            "tension_records": [
                {
                    "tension_id": t.tension_id,
                    "topic": t.topic,
                    "belief_count": len(t.belief_ids),
                    "revisit_count": t.revisit_count,
                    "maturation_score": round(t.maturation_score, 3),
                    "stability": round(t.stability, 3),
                    "age_s": round(time.time() - t.created_at),
                }
                for t in tensions
            ],
            "resolved_count": total_res,
            "contradiction_debt": round(debt, 4),
            "debt_trend": [round(d, 4) for d in debt_trend],
            "by_type": by_type,
            "by_resolution": by_res,
            "by_conflict_key": conflict_key_counts,
            "near_miss_count": classifier_stats["near_miss_count"],
            "near_miss_rate": round(classifier_stats["near_miss_rate"], 3),
            "recent_near_misses": recent_near_misses,
            "extraction_discard_count": disc,
            "version_collapses": collapses,
        }

    def rehydrate(self) -> None:
        with self._lock:
            self._belief_store.rehydrate()
            for b in self._belief_store._beliefs.values():
                if b.contradicts:
                    self._seen_conflict_keys.add(b.conflict_key)

            floor = self._reconstruct_debt_floor()
            self._contradiction_debt = floor
            if floor > 0.0:
                self._debt_trend.append(floor)

            logger.info(
                "ContradictionEngine rehydrated: %d beliefs, %d tensions, "
                "reconstructed_debt_floor=%.4f",
                len(self._belief_store._beliefs),
                len(self._belief_store._tensions),
                floor,
            )

    # Per-active-contradiction estimate for debt floor reconstruction.
    # Not an exact replay of historical debt deltas — only a conservative lower bound.
    DEBT_RECONSTRUCT_PER_ACTIVE = 0.05
    DEBT_RECONSTRUCT_PER_TENSION = 0.02

    def _reconstruct_debt_floor(self) -> float:
        """Compute a conservative floor estimate of contradiction debt from
        rehydrated beliefs and tensions.

        This cannot reconstruct exact historical debt because resolution
        outcomes and passive decay history are not persisted.  It produces
        a lower bound so that restart does not wash real pressure to zero.
        """
        debt = 0.0

        for b in self._belief_store._beliefs.values():
            if b.contradicts and getattr(b, "resolution_state", "") == "active":
                debt += self.DEBT_RECONSTRUCT_PER_ACTIVE
                if b.conflict_key in self._seen_conflict_keys:
                    debt += DEBT_RECURRENCE_EXTRA

        for t in self._belief_store.get_active_tensions():
            if getattr(t, "resolution_state", "active") != "resolved":
                debt += self.DEBT_RECONSTRUCT_PER_TENSION

        return max(DEBT_MIN, min(DEBT_MAX, debt))

    def set_persisted_debt(self, persisted_debt: float) -> dict[str, Any]:
        """Apply persisted debt from consciousness_state.json, using the
        conservative max(persisted, reconstructed_floor) rule.

        Returns diagnostics dict for logging and dashboard exposure.
        """
        with self._lock:
            reconstructed = self._contradiction_debt  # already set by rehydrate
            effective = max(persisted_debt, reconstructed)
            mismatch = abs(persisted_debt - reconstructed) > 0.05

            self._contradiction_debt = max(DEBT_MIN, min(DEBT_MAX, effective))
            if effective > 0.0:
                self._debt_trend.append(self._contradiction_debt)

            source = "persisted" if persisted_debt >= reconstructed else "reconstructed"

            diagnostics = {
                "persisted_debt": round(persisted_debt, 4),
                "reconstructed_debt_floor": round(reconstructed, 4),
                "effective_debt": round(self._contradiction_debt, 4),
                "debt_restore_source": source,
                "restart_debt_mismatch_detected": mismatch,
            }

            if mismatch:
                logger.warning(
                    "RESTART_DEBT_MISMATCH: persisted=%.4f reconstructed_floor=%.4f "
                    "using=%s effective=%.4f",
                    persisted_debt, reconstructed, source, self._contradiction_debt,
                )
            else:
                logger.info(
                    "Contradiction debt restored: effective=%.4f source=%s",
                    self._contradiction_debt, source,
                )

            return diagnostics

    @property
    def contradiction_debt(self) -> float:
        return self._contradiction_debt

    @property
    def belief_store(self) -> BeliefStore:
        return self._belief_store

    # -- Internal -----------------------------------------------------------

    def _get_stable_paradox_belief_ids(self) -> set[str]:
        """Collect belief IDs belonging to tensions that are stable paradoxes."""
        stable_bids: set[str] = set()
        for tension in self._belief_store.get_active_tensions():
            if (tension.revisit_count >= 50
                    and tension.maturation_score >= 0.90):
                stable_bids.update(tension.belief_ids)
        return stable_bids

    def _update_debt(self, delta: float, conflict_key: str) -> None:
        if delta == 0.0:
            return
        effective_delta = delta
        if conflict_key in self._seen_conflict_keys:
            effective_delta += DEBT_RECURRENCE_EXTRA
        self._seen_conflict_keys.add(conflict_key)
        self._contradiction_debt = max(
            DEBT_MIN, min(DEBT_MAX, self._contradiction_debt + effective_delta)
        )

    def _emit_event(self, classification: Any, outcome: ResolutionOutcome) -> None:
        try:
            from consciousness.events import (
                event_bus,
                CONTRADICTION_TENSION_HELD,
                CONTRADICTION_DETECTED,
                CONTRADICTION_RESOLVED,
            )
            if classification.conflict_type == "identity_tension":
                event_bus.emit(
                    CONTRADICTION_TENSION_HELD,
                    conflict_type=classification.conflict_type,
                    tension_id=outcome.tension_id,
                    debt=self._contradiction_debt,
                )
            elif outcome.action_taken in ("confidence_adjusted", "policy_penalized", "source_separated"):
                event_bus.emit(
                    CONTRADICTION_DETECTED,
                    conflict_type=classification.conflict_type,
                    severity=classification.severity,
                    is_pathological=classification.is_pathological,
                    debt=self._contradiction_debt,
                )
            else:
                event_bus.emit(
                    CONTRADICTION_RESOLVED,
                    conflict_type=classification.conflict_type,
                    action=outcome.action_taken,
                    debt=self._contradiction_debt,
                )
        except Exception:
            pass
