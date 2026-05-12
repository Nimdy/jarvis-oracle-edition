"""Signal collector: gathers all subsystem metrics into a CalibrationSnapshot.

Read-only aggregation -- no locks, no writes, no side effects.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("jarvis.calibration.collector")


@dataclass
class CalibrationSnapshot:
    timestamp: float = 0.0

    # Retrieval domain
    reference_match_rate: float | None = None
    ranker_success_rate: float | None = None
    heuristic_success_rate: float | None = None
    lift: float | None = None
    provenance_weighted_success: float | None = None

    # Autonomy domain
    improvement_rate: float | None = None
    overall_win_rate: float | None = None
    research_conversion_rate: float | None = None

    # Salience domain
    wasted_rate: float | None = None
    useful_rate: float | None = None
    weight_error: float | None = None
    decay_error: float | None = None

    # Reasoning domain
    coherence: float | None = None
    consistency: float | None = None
    depth: float | None = None

    # Skill domain
    verified_skill_count: int = 0
    honesty_failures: int = 0

    # Epistemic domain
    contradiction_debt: float = 0.0
    near_miss_rate: float | None = None
    total_beliefs: int = 0
    resolved_count: int = 0
    graph_health_score: float | None = None

    # Confidence domain (populated externally by ConfidenceCalibrator)
    brier_score: float | None = None
    ece: float | None = None
    overconfidence_error: float | None = None
    underconfidence_error: float | None = None
    route_brier_scores: dict[str, float] = field(default_factory=dict)
    worst_route_brier: float | None = None
    worst_route_sample_count: int = 0

    # Correction penalty (populated externally by TruthCalibrationEngine)
    correction_penalty: float | None = None

    # Prediction domain (populated externally by PredictionValidator + WorldModel)
    prediction_accuracy: float | None = None
    wm_prediction_accuracy: float | None = None
    wm_prediction_count: int = 0

    # Ingestion quality domain
    ingestion_total_sources: int = 0
    ingestion_quality_score: float | None = None
    ingestion_chunk_count: int = 0

    # Spatial domains (Phase 1)
    spatial_calibration_valid: bool = False
    spatial_anchor_count: int = 0
    spatial_stable_tracks: int = 0
    spatial_promoted_deltas: int = 0
    spatial_rejected_promotions: int = 0
    spatial_contradiction_count: int = 0
    spatial_anchor_drift_score: float | None = None


class SignalCollector:
    """Reads metrics from all subsystems into a CalibrationSnapshot."""

    def __init__(self, engine: object | None = None) -> None:
        self._engine = engine

    def collect(self) -> CalibrationSnapshot:
        snap = CalibrationSnapshot(timestamp=time.time())
        self._collect_retrieval(snap)
        self._collect_lifecycle(snap)
        self._collect_autonomy(snap)
        self._collect_reasoning(snap)
        self._collect_epistemic(snap)
        self._collect_skills(snap)
        self._collect_ingestion(snap)
        self._collect_spatial(snap)
        return snap

    def _collect_retrieval(self, snap: CalibrationSnapshot) -> None:
        try:
            from memory.retrieval_log import memory_retrieval_log
            metrics = memory_retrieval_log.get_eval_metrics()
            snap.reference_match_rate = metrics.get("reference_match_rate")
            snap.ranker_success_rate = metrics.get("ranker_success_rate")
            snap.heuristic_success_rate = metrics.get("heuristic_success_rate")
            snap.lift = metrics.get("lift")
            snap.provenance_weighted_success = metrics.get("provenance_weighted_success_rate")
        except Exception as exc:
            logger.debug("Retrieval signal collection failed: %s", exc)

    def _collect_lifecycle(self, snap: CalibrationSnapshot) -> None:
        try:
            from memory.lifecycle_log import memory_lifecycle_log
            metrics = memory_lifecycle_log.get_effectiveness_metrics()
            snap.wasted_rate = metrics.get("wasted_rate")
            snap.useful_rate = metrics.get("useful_rate")
            snap.weight_error = metrics.get("weight_error")
            snap.decay_error = metrics.get("decay_error")
        except Exception as exc:
            logger.debug("Lifecycle signal collection failed: %s", exc)

    def _collect_autonomy(self, snap: CalibrationSnapshot) -> None:
        try:
            if not self._engine:
                return
            orch = getattr(self._engine, "_autonomy_orchestrator", None)
            if orch is None:
                return

            delta_stats = orch._delta_tracker.get_stats() if hasattr(orch, "_delta_tracker") else {}
            snap.improvement_rate = delta_stats.get("improvement_rate")

            pm_stats = orch._policy_memory.get_stats() if hasattr(orch, "_policy_memory") else {}
            snap.overall_win_rate = pm_stats.get("overall_win_rate")
        except Exception as exc:
            logger.debug("Autonomy signal collection failed: %s", exc)

        try:
            from autonomy.source_ledger import get_source_ledger
            ledger_stats = get_source_ledger().get_stats()
            by_verdict = ledger_stats.get("by_verdict", {})
            total_verdicted = sum(
                v for k, v in by_verdict.items() if k != "pending"
            )
            useful_count = by_verdict.get("useful", 0)
            if total_verdicted > 0:
                snap.research_conversion_rate = round(
                    useful_count / total_verdicted, 4,
                )
        except Exception as exc:
            logger.debug("Source ledger signal collection failed: %s", exc)

    def _collect_reasoning(self, snap: CalibrationSnapshot) -> None:
        try:
            if not self._engine:
                return
            # Primary runtime engine stores consciousness at `_consciousness`.
            # Keep `_consciousness_system` fallback for compatibility with tests
            # or alternative wrappers.
            cs = getattr(self._engine, "_consciousness", None)
            if cs is None:
                cs = getattr(self._engine, "_consciousness_system", None)
            if cs is None:
                cs = getattr(self._engine, "consciousness", None)
            if cs is None:
                return
            analytics = getattr(cs, "analytics", None)
            if analytics is None:
                return
            state = analytics.get_full_state()
            reasoning = state.get("reasoning", {})
            snap.coherence = reasoning.get("coherence")
            snap.consistency = reasoning.get("consistency")
            snap.depth = reasoning.get("depth")
        except Exception as exc:
            logger.debug("Reasoning signal collection failed: %s", exc)

    def _collect_epistemic(self, snap: CalibrationSnapshot) -> None:
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            engine = ContradictionEngine.get_instance()
            if engine is None:
                return
            state = engine.get_state()
            snap.contradiction_debt = state.get("contradiction_debt", 0.0)
            snap.near_miss_rate = state.get("near_miss_rate")
            snap.total_beliefs = state.get("total_beliefs", 0)
            snap.resolved_count = state.get("resolved_count", 0)
        except Exception as exc:
            logger.debug("Epistemic signal collection failed: %s", exc)

        try:
            from epistemic.belief_graph import BeliefGraph
            bg = BeliefGraph.get_instance()
            if bg and bg._initialized and bg._edge_store and bg._belief_store:
                from epistemic.belief_graph.integrity import compute_integrity
                metrics = compute_integrity(bg._edge_store, bg._belief_store)
                snap.graph_health_score = metrics.get("health_score")
        except Exception as exc:
            logger.debug("Graph health signal collection failed: %s", exc)

    def _collect_skills(self, snap: CalibrationSnapshot) -> None:
        try:
            from skills.registry import skill_registry
            from skills.capability_gate import capability_gate
            status = skill_registry.get_status_snapshot()
            by_status = status.get("by_status", {})
            snap.verified_skill_count = by_status.get("verified", 0)

            gate_stats = capability_gate.get_stats()
            snap.honesty_failures = gate_stats.get("honesty_failures", 0)
        except Exception as exc:
            logger.debug("Skills signal collection failed: %s", exc)

    def _collect_ingestion(self, snap: CalibrationSnapshot) -> None:
        try:
            from library.db import get_connection
            conn = get_connection()
            row = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
            snap.ingestion_total_sources = row[0] if row else 0
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            snap.ingestion_chunk_count = row[0] if row else 0
            row = conn.execute(
                "SELECT AVG(quality_score) FROM sources WHERE quality_score > 0"
            ).fetchone()
            snap.ingestion_quality_score = round(row[0], 3) if row and row[0] else None
        except Exception as exc:
            logger.debug("Ingestion signal collection failed: %s", exc)

    def _collect_spatial(self, snap: CalibrationSnapshot) -> None:
        """Collect spatial diagnostics from PerceptionOrchestrator."""
        try:
            if not self._engine:
                return
            po = getattr(self._engine, "_perception_orchestrator", None)
            if po is None or not hasattr(po, "get_spatial_state"):
                return

            spatial = po.get_spatial_state() or {}
            calibration = spatial.get("calibration", {}) or {}
            estimator = spatial.get("estimator", {}) or {}
            validation = spatial.get("validation", {}) or {}
            rejection_counts = validation.get("rejection_counts", {}) or {}

            state = calibration.get("state", "")
            anchor_consistency_ok = calibration.get("anchor_consistency_ok")
            snap.spatial_calibration_valid = (
                state == "valid"
                and anchor_consistency_ok is not False
            )
            snap.spatial_anchor_count = int(estimator.get("anchor_count", 0) or 0)
            snap.spatial_stable_tracks = int(estimator.get("stable_tracks", 0) or 0)
            snap.spatial_promoted_deltas = int(validation.get("total_promoted", 0) or 0)
            snap.spatial_rejected_promotions = int(validation.get("total_rejections", 0) or 0)
            snap.spatial_contradiction_count = int(
                rejection_counts.get("anchor_authority_conflict", 0) or 0,
            ) + int(rejection_counts.get("contradiction", 0) or 0)

            drift_score = validation.get("anchor_drift_score")
            if isinstance(drift_score, (int, float)):
                snap.spatial_anchor_drift_score = float(drift_score)
            elif anchor_consistency_ok is True and snap.spatial_anchor_count > 0:
                snap.spatial_anchor_drift_score = 0.0
            elif anchor_consistency_ok is False:
                snap.spatial_anchor_drift_score = 1.0
        except Exception as exc:
            logger.debug("Spatial signal collection failed: %s", exc)
