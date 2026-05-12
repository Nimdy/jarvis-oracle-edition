"""Epistemic Immune System -- Layer 6: Truth Calibration.

The epistemic cerebellum: continuously measures whether Jarvis's expectations
match reality across 8 calibration domains, producing a unified Truth Score
with maturity tracking.

Public API: TruthCalibrationEngine (orchestrator).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from epistemic.calibration.signal_collector import SignalCollector, CalibrationSnapshot
from epistemic.calibration.calibration_history import CalibrationHistory
from epistemic.calibration.domain_calibrator import DomainCalibrator, DomainScore, ALL_DOMAINS
from epistemic.calibration.truth_score import TruthScoreCalculator, TruthScoreReport
from epistemic.calibration.drift_detector import DriftDetector, DriftAlert

logger = logging.getLogger("jarvis.calibration")

_JARVIS_DIR = Path(os.path.expanduser("~/.jarvis"))
_CALIBRATION_LOG = _JARVIS_DIR / "calibration_truth.jsonl"
_MAX_LOG_ENTRIES = 200
_UNIFORM_OUTCOME_WARN_MIN = 100
_UNIFORM_OUTCOME_WARN_STEP = 100
_UNIFORM_OUTCOME_WARN_INTERVAL_S = 3600.0

_instance: TruthCalibrationEngine | None = None


class TruthCalibrationEngine:
    """Orchestrates the full Layer 6 calibration pipeline.

    Consciousness system calls on_tick() which runs:
      SignalCollector -> CalibrationHistory -> DomainCalibrator -> DriftDetector -> TruthScore
    """

    def __init__(self, engine: object | None = None) -> None:
        self._engine = engine
        self._collector = SignalCollector(engine=engine)
        self._history = CalibrationHistory()
        self._domain_calibrator = DomainCalibrator()
        self._truth_calculator = TruthScoreCalculator()
        self._drift_detector = DriftDetector()

        self._last_report: TruthScoreReport | None = None
        self._last_domain_scores: dict[str, DomainScore] = {}
        self._tick_count: int = 0
        self._total_drift_alerts: int = 0
        self._correction_count: int = 0
        self._total_response_count: int = 0

        self._confidence_calibrator = None
        self._correction_detector = None
        self._belief_adjuster = None
        self._prediction_validator = None
        self._skill_watchdog = None

        # Phase 6.1: outcome bridge counters (watchdog)
        self._bridge_prediction_validated: int = 0
        self._bridge_world_model_validated: int = 0
        self._bridge_outcome_resolved: int = 0
        self._bridge_positive_response: int = 0
        self._bridge_errors: int = 0
        self._last_watchdog_ts: float = 0.0
        self._last_uniform_outcome_warn_ts: float = 0.0
        self._last_uniform_outcome_warn_count: int = 0
        self._last_uniform_outcome_warn_dominant: str = ""

        # Phase 6.1b: track world model rolling accuracy locally so we can
        # use observed accuracy as confidence instead of a hardcoded default.
        self._wm_hits: int = 0
        self._wm_total: int = 0

        self._initialized = False
        global _instance
        _instance = self

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        _JARVIS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            from epistemic.calibration.confidence_calibrator import ConfidenceCalibrator
            self._confidence_calibrator = ConfidenceCalibrator()
        except Exception as exc:
            logger.debug("Confidence calibrator init failed: %s", exc)

        try:
            from epistemic.calibration.correction_detector import CorrectionDetector
            self._correction_detector = CorrectionDetector()
        except Exception as exc:
            logger.debug("Correction detector init failed: %s", exc)

        try:
            from epistemic.calibration.belief_adjuster import BeliefConfidenceAdjuster
            self._belief_adjuster = BeliefConfidenceAdjuster()
        except Exception as exc:
            logger.debug("Belief adjuster init failed: %s", exc)

        try:
            from epistemic.calibration.prediction_validator import PredictionValidator
            self._prediction_validator = PredictionValidator()
        except Exception as exc:
            logger.debug("Prediction validator init failed: %s", exc)

        try:
            from epistemic.calibration.skill_watchdog import SkillWatchdog
            self._skill_watchdog = SkillWatchdog()
        except Exception as exc:
            logger.debug("Skill watchdog init failed: %s", exc)

        try:
            loaded = self._history.rehydrate_from_log(_CALIBRATION_LOG)
            if loaded > 0:
                logger.info("Calibration history rehydrated: %d snapshots from log", loaded)
        except Exception as exc:
            logger.debug("Calibration history rehydration failed: %s", exc)

        self._subscribe_outcome_bridges()
        logger.info("TruthCalibrationEngine initialized (Layer 6)")

    def on_tick(self) -> TruthScoreReport | None:
        """Run one calibration cycle. Called from consciousness_system."""
        self._ensure_initialized()
        self._tick_count += 1
        now = time.time()

        snapshot = self._collector.collect()

        if self._confidence_calibrator:
            try:
                snapshot.brier_score = self._confidence_calibrator.get_brier_score()
                snapshot.ece = self._confidence_calibrator.get_ece()
                snapshot.overconfidence_error = self._confidence_calibrator.get_overconfidence_error()
                snapshot.underconfidence_error = self._confidence_calibrator.get_underconfidence_error()
                snapshot.route_brier_scores = self._confidence_calibrator.get_route_brier_scores()
                worst_brier, worst_count = self._confidence_calibrator.get_worst_route_brier()
                snapshot.worst_route_brier = worst_brier
                snapshot.worst_route_sample_count = worst_count
            except Exception as exc:
                logger.debug("Confidence calibrator read failed: %s", exc)

        if self._prediction_validator:
            try:
                snapshot.prediction_accuracy = self._prediction_validator.get_accuracy()
            except Exception as exc:
                logger.debug("Prediction validator read failed: %s", exc)

        if self._wm_total >= 20:
            snapshot.wm_prediction_accuracy = round(self._wm_hits / self._wm_total, 4)
            snapshot.wm_prediction_count = self._wm_total

        if self._total_response_count > 0:
            snapshot.correction_penalty = round(
                self._correction_count / self._total_response_count, 4,
            )

        self._history.record(snapshot)

        domain_scores = self._domain_calibrator.score_all(snapshot)
        self._last_domain_scores = domain_scores

        new_alerts = self._drift_detector.update(domain_scores)
        for alert in new_alerts:
            self._total_drift_alerts += 1
            self._emit_drift_event(alert)

        report = self._truth_calculator.compute(domain_scores, timestamp=now)
        self._last_report = report

        self._emit_calibration_event(report)
        self._persist_snapshot(report, snapshot)

        if self._prediction_validator:
            try:
                validated_preds = self._prediction_validator.tick()
                if validated_preds and self._confidence_calibrator:
                    for pred in validated_preds:
                        try:
                            self._confidence_calibrator.record_outcome(
                                belief_id=f"pred:{pred.prediction_id}",
                                confidence=pred.confidence,
                                correct=bool(pred.validation_result),
                                provenance="prediction",
                                route_class=f"prediction_{pred.prediction_type}",
                            )
                            self._bridge_prediction_validated += 1
                        except Exception:
                            self._bridge_errors += 1
            except Exception as exc:
                logger.debug("Prediction validator tick failed: %s", exc)

        if self._skill_watchdog:
            try:
                self._skill_watchdog.tick()
            except Exception as exc:
                logger.debug("Skill watchdog tick failed: %s", exc)

        if self._belief_adjuster and self._confidence_calibrator:
            try:
                from consciousness.modes import mode_manager
                if mode_manager.mode in ("dreaming", "deep_learning"):
                    prov_acc = self._confidence_calibrator.get_per_provenance_accuracy()
                    overconf = self._confidence_calibrator.get_overconfidence_error()
                    self._belief_adjuster.run_adjustment_cycle(prov_acc, overconf)
            except Exception as exc:
                logger.debug("Belief adjustment failed: %s", exc)

        self._run_outcome_watchdog()
        return report

    def get_state(self) -> dict:
        """Build dashboard-friendly state snapshot."""
        report = self._last_report
        if report is None:
            return {
                "truth_score": None,
                "maturity": 0.0,
                "provisional_count": len(ALL_DOMAINS),
                "route_brier_scores": {},
                "route_brier": {},
                "active_drift_alerts": [],
                "drift_alerts": [],
                "status": "not_started",
                "tick_count": self._tick_count,
            }

        active_alerts = self._drift_detector.get_active_alerts()
        resolved_alerts = self._drift_detector.get_resolved_alerts(5)

        trends: dict[str, list[float]] = {}
        for domain in ALL_DOMAINS:
            trends[domain] = self._history.get_domain_trend(domain, window=20)

        result: dict = {
            "truth_score": report.truth_score,
            "maturity": report.maturity,
            "provisional_count": report.provisional_count,
            "data_coverage": report.data_coverage,
            "domain_scores": report.domain_scores,
            "domain_provisional": report.domain_provisional,
            "domain_trends": trends,
            "tick_count": self._tick_count,
            "total_drift_alerts": self._total_drift_alerts,
            "correction_count": self._correction_count,
            "active_drift_alerts": [
                {
                    "domain": a.domain,
                    "severity": a.severity,
                    "score_drop": a.score_drop,
                    "readings_declining": a.readings_declining,
                    "peak_score": a.peak_score,
                    "current_score": a.current_score,
                    "triggered_at": a.triggered_at,
                }
                for a in active_alerts
            ],
            "resolved_drift_alerts": [
                {
                    "domain": a.domain,
                    "severity": a.severity,
                    "score_drop": a.score_drop,
                }
                for a in resolved_alerts
            ],
            "route_brier_scores": {},
            "route_sample_counts": {},
            "route_brier": {},
            "drift_alerts": [],
            "status": "calibrating" if report.truth_score is None else "active",
        }

        if self._confidence_calibrator:
            try:
                result["brier_score"] = self._confidence_calibrator.get_brier_score()
                result["ece"] = self._confidence_calibrator.get_ece()
                result["per_provenance_accuracy"] = self._confidence_calibrator.get_per_provenance_accuracy()
                result["overconfidence_error"] = self._confidence_calibrator.get_overconfidence_error()
                result["underconfidence_error"] = self._confidence_calibrator.get_underconfidence_error()
                result["calibration_curve"] = self._confidence_calibrator.get_calibration_curve()
                result["route_brier_scores"] = self._confidence_calibrator.get_route_brier_scores()
                result["route_sample_counts"] = self._confidence_calibrator.get_route_sample_counts()
                result["confidence_outcome_count"] = self._confidence_calibrator.outcome_count
            except Exception:
                pass

        # Backward-compatible aliases used by some dashboard renderers.
        result["route_brier"] = dict(result.get("route_brier_scores", {}) or {})
        result["drift_alerts"] = list(result.get("active_drift_alerts", []) or [])

        result["outcome_bridges"] = {
            "prediction_validated": self._bridge_prediction_validated,
            "world_model_validated": self._bridge_world_model_validated,
            "outcome_resolved": self._bridge_outcome_resolved,
            "positive_response": self._bridge_positive_response,
            "errors": self._bridge_errors,
            "total": (
                self._bridge_prediction_validated
                + self._bridge_world_model_validated
                + self._bridge_outcome_resolved
                + self._bridge_positive_response
            ),
            "wm_observed_accuracy": (
                round(self._wm_hits / self._wm_total, 3)
                if self._wm_total >= 20 else None
            ),
        }

        return result

    def get_truth_score(self) -> float | None:
        """Quick accessor for the truth score value."""
        return self._last_report.truth_score if self._last_report else None

    def get_maturity(self) -> float:
        """Quick accessor for truth score maturity."""
        return self._last_report.maturity if self._last_report else 0.0

    def record_correction(self) -> None:
        """Increment the user correction counter."""
        self._correction_count += 1

    def record_response(self) -> None:
        """Increment total response counter (used for correction_penalty ratio)."""
        self._total_response_count += 1

    def check_correction(
        self,
        user_text: str,
        is_negative: bool,
        last_response_text: str,
        last_tool_route: str,
        injected_memory_payloads: list[str],
        response_confidence: float | None = None,
    ) -> dict[str, Any] | None:
        """Check if the user is correcting a prior claim.

        Returns the structured correction record when detected, else ``None``.
        ``response_confidence`` is the system's actual stated confidence for
        the prior response being corrected — forwarded to the Brier bridge.
        """
        if not self._correction_detector:
            return None
        result = self._correction_detector.check(
            user_text, is_negative, last_response_text, last_tool_route, injected_memory_payloads,
        )
        if result:
            self._correction_count += 1
            try:
                from consciousness.events import event_bus, CALIBRATION_CORRECTION_DETECTED
                event_bus.emit(CALIBRATION_CORRECTION_DETECTED, **result)
            except Exception:
                pass
            try:
                from epistemic.contradiction_engine import ContradictionEngine
                from epistemic.belief_record import DEBT_USER_CORRECTION
                engine = ContradictionEngine.get_instance()
                if engine:
                    engine.apply_correction_debt()
            except Exception:
                pass
            self._record_correction_confidence_outcome(
                last_tool_route, result, response_confidence=response_confidence,
            )
            return result
        return None

    _ROUTE_CONFIDENCE_DEFAULTS: dict[str, float] = {
        "status": 0.80,
        "system_status": 0.80,
        "introspection": 0.70,
        "memory": 0.60,
        "identity": 0.70,
        "none": 0.50,
        "academic_search": 0.55,
        "web_search": 0.55,
        "codebase": 0.65,
    }

    def _record_correction_confidence_outcome(
        self, route: str, correction: dict,
        response_confidence: float | None = None,
    ) -> None:
        """Record a correction as a confidence outcome.

        ``response_confidence`` should be the system's actual stated confidence
        for the *prior* response that was corrected.  Falls back to route
        defaults only when no real confidence is available.
        """
        if not self._confidence_calibrator:
            return
        try:
            route_lower = (route or "").lower()
            if response_confidence is not None:
                confidence = max(0.0, min(1.0, response_confidence))
            else:
                confidence = self._ROUTE_CONFIDENCE_DEFAULTS.get(route_lower, 0.50)
            self._confidence_calibrator.record_outcome(
                belief_id=f"correction:{self._correction_count}",
                confidence=confidence,
                correct=False,
                provenance="user_correction",
                route_class=route_lower,
            )
        except Exception:
            logger.debug("Correction confidence outcome recording failed", exc_info=True)

    def set_confidence_calibrator(self, cal: object) -> None:
        self._confidence_calibrator = cal

    def set_correction_detector(self, det: object) -> None:
        self._correction_detector = det

    def set_belief_adjuster(self, adj: object) -> None:
        self._belief_adjuster = adj

    def set_prediction_validator(self, val: object) -> None:
        self._prediction_validator = val

    def set_skill_watchdog(self, watch: object) -> None:
        self._skill_watchdog = watch

    # ------------------------------------------------------------------
    # Phase 6.1: Outcome bridges — feed real data into ConfidenceCalibrator
    # ------------------------------------------------------------------

    def _subscribe_outcome_bridges(self) -> None:
        """Wire EventBus listeners that bridge validated outcomes into the calibrator."""
        try:
            from consciousness.events import (
                event_bus,
                WORLD_MODEL_PREDICTION_VALIDATED,
                OUTCOME_RESOLVED,
            )
            event_bus.on(WORLD_MODEL_PREDICTION_VALIDATED, self._on_world_model_prediction_validated)
            event_bus.on(OUTCOME_RESOLVED, self._on_outcome_resolved)
            logger.info("Phase 6.1: outcome bridges subscribed (world_model, outcome_resolved)")
        except Exception as exc:
            logger.warning("Phase 6.1: outcome bridge subscription failed: %s", exc)

    def _on_world_model_prediction_validated(self, **kwargs: Any) -> None:
        """Bridge 2: world model causal predictions → confidence outcomes.

        Uses the actual prediction confidence from the CausalEngine rule that
        generated the prediction.  Falls back to 0.70 only when the event
        doesn't carry the field (backward compat).
        """
        if not self._confidence_calibrator:
            return
        try:
            label = kwargs.get("prediction_label", "")
            outcome = kwargs.get("outcome", "")
            correct = outcome == "hit"

            self._wm_total += 1
            if correct:
                self._wm_hits += 1

            confidence = kwargs.get("prediction_confidence")
            if confidence is None:
                confidence = 0.70
            confidence = max(0.0, min(1.0, float(confidence)))

            self._confidence_calibrator.record_outcome(
                belief_id=f"wm_pred:{label}",
                confidence=confidence,
                correct=correct,
                provenance="world_model",
                route_class="world_model",
            )
            self._bridge_world_model_validated += 1
        except Exception:
            self._bridge_errors += 1
            logger.debug("Bridge world_model_prediction failed", exc_info=True)

    _SUBSYSTEM_CONFIDENCE_FALLBACK: dict[str, float] = {
        "conversation": 0.65,
        "autonomy": 0.60,
        "self_improve": 0.55,
        "perception": 0.70,
        "consciousness": 0.60,
    }

    def _on_outcome_resolved(self, **kwargs: Any) -> None:
        """Bridge 3: attribution ledger delayed outcomes → confidence outcomes.

        Uses the actual confidence from the outcome data when available.
        Determines correctness from the ``user_signal`` field:
        - ``"positive"`` or ``"correction"`` with ``outcome="success"`` → correct
        - ``"correction"`` or ``"negative"`` → incorrect
        - No user signal or bare ``"follow_up"`` → skip (no evidence either way)
        - Non-conversation subsystems use outcome directly (success/failure).
        """
        if not self._confidence_calibrator:
            return
        try:
            entry_id = kwargs.get("entry_id", "")
            subsystem = kwargs.get("subsystem", "")
            outcome = kwargs.get("outcome", "")
            if outcome in ("inconclusive",):
                return

            user_signal = kwargs.get("user_signal", "")
            event_confidence = kwargs.get("confidence")

            if subsystem == "conversation":
                if user_signal == "positive":
                    correct = True
                elif user_signal in ("correction", "negative"):
                    correct = False
                else:
                    return
            else:
                correct = outcome in ("success", "stable")

            if event_confidence is not None:
                confidence = max(0.0, min(1.0, float(event_confidence)))
            else:
                confidence = self._SUBSYSTEM_CONFIDENCE_FALLBACK.get(subsystem, 0.55)

            self._confidence_calibrator.record_outcome(
                belief_id=f"attr:{entry_id}",
                confidence=confidence,
                correct=correct,
                provenance="attribution",
                route_class=subsystem,
            )
            self._bridge_outcome_resolved += 1
        except Exception:
            self._bridge_errors += 1
            logger.debug("Bridge outcome_resolved failed", exc_info=True)

    def record_positive_response_outcome(
        self, route: str, response_confidence: float | None = None,
    ) -> None:
        """Bridge 5: explicitly positive conversational outcome → confidence outcome.

        Called from conversation_handler ONLY when the user gives an explicitly
        positive signal (e.g. "thanks", "great").  Bare follow-ups (user just
        continued talking) are NOT recorded — absence of complaint is not evidence
        of correctness.

        ``response_confidence`` should be the system's actual stated confidence
        for the response, from ``_language_example_seed["confidence"]``.  Falls
        back to route defaults only when no real confidence is available.
        """
        if not self._confidence_calibrator:
            return
        try:
            route_lower = (route or "").lower()
            if response_confidence is not None:
                confidence = max(0.0, min(1.0, response_confidence))
            else:
                confidence = self._ROUTE_CONFIDENCE_DEFAULTS.get(route_lower, 0.50)
            self._confidence_calibrator.record_outcome(
                belief_id=f"response_ok:{self._total_response_count}",
                confidence=confidence,
                correct=True,
                provenance="conversation",
                route_class=route_lower,
            )
            self._bridge_positive_response += 1
        except Exception:
            self._bridge_errors += 1
            logger.debug("Bridge positive_response failed", exc_info=True)

    def _run_outcome_watchdog(self) -> None:
        """Lightweight watchdog: warn if bridges are active but producing nothing."""
        now = time.time()
        if now - self._last_watchdog_ts < 600:
            return
        self._last_watchdog_ts = now

        if self._tick_count < 10:
            return

        total_bridged = (
            self._bridge_prediction_validated
            + self._bridge_world_model_validated
            + self._bridge_outcome_resolved
            + self._bridge_positive_response
        )

        cal = self._confidence_calibrator
        outcome_count = cal.outcome_count if cal else 0

        if self._total_response_count > 5 and total_bridged == 0 and outcome_count == 0:
            logger.warning(
                "Phase 6.1 WATCHDOG: %d responses processed, %d predictions validated, "
                "but 0 confidence outcomes recorded. Bridges may be broken.",
                self._total_response_count,
                self._bridge_prediction_validated + self._bridge_world_model_validated,
            )

        if total_bridged > 0 and outcome_count == 0:
            logger.warning(
                "Phase 6.1 WATCHDOG: %d events bridged but calibrator has 0 outcomes. "
                "Persistence may be broken.",
                total_bridged,
            )

        if outcome_count >= _UNIFORM_OUTCOME_WARN_MIN:
            correct_count = sum(1 for o in cal._outcomes if o.actual_correct)
            incorrect_count = outcome_count - correct_count
            if correct_count == 0 or incorrect_count == 0:
                dominant = "correct=True" if correct_count > 0 else "correct=False"
                if self._should_warn_uniform_outcomes(dominant, outcome_count, now):
                    provenance_counts: dict[str, int] = {}
                    try:
                        provenance_counts = cal.get_provenance_sample_counts()
                    except Exception:
                        provenance_counts = {}
                    top_prov = ""
                    top_count = 0
                    if provenance_counts:
                        top_prov, top_count = max(
                            provenance_counts.items(),
                            key=lambda item: item[1],
                        )
                    logger.warning(
                        "Phase 6.1 WATCHDOG: %d outcomes are 100%% %s. "
                        "Calibration will be skewed (top_provenance=%s:%d, "
                        "bridges=%d, throttle_step=%d).",
                        outcome_count,
                        dominant,
                        top_prov or "none",
                        int(top_count),
                        int(total_bridged),
                        _UNIFORM_OUTCOME_WARN_STEP,
                    )
                    self._last_uniform_outcome_warn_ts = now
                    self._last_uniform_outcome_warn_count = outcome_count
                    self._last_uniform_outcome_warn_dominant = dominant
            else:
                self._last_uniform_outcome_warn_ts = 0.0
                self._last_uniform_outcome_warn_count = 0
                self._last_uniform_outcome_warn_dominant = ""

    def _should_warn_uniform_outcomes(self, dominant: str, outcome_count: int, now: float) -> bool:
        """Throttle repetitive uniform-outcome warnings while preserving signal."""
        if self._last_uniform_outcome_warn_dominant != dominant:
            return True
        if (outcome_count - self._last_uniform_outcome_warn_count) >= _UNIFORM_OUTCOME_WARN_STEP:
            return True
        return (now - self._last_uniform_outcome_warn_ts) >= _UNIFORM_OUTCOME_WARN_INTERVAL_S

    def _emit_calibration_event(self, report: TruthScoreReport) -> None:
        try:
            from consciousness.events import event_bus, CALIBRATION_UPDATED
            event_bus.emit(CALIBRATION_UPDATED,
                           truth_score=report.truth_score,
                           maturity=report.maturity,
                           provisional_count=report.provisional_count)
        except Exception:
            pass

    def _emit_drift_event(self, alert: DriftAlert) -> None:
        try:
            from consciousness.events import event_bus, CALIBRATION_DRIFT_DETECTED
            event_bus.emit(CALIBRATION_DRIFT_DETECTED,
                           domain=alert.domain,
                           severity=alert.severity,
                           score_drop=alert.score_drop,
                           readings_declining=alert.readings_declining)
        except Exception:
            pass

    def _persist_snapshot(self, report: TruthScoreReport, snapshot: CalibrationSnapshot) -> None:
        try:
            entry = {
                "ts": round(snapshot.timestamp, 2),
                "truth_score": report.truth_score,
                "maturity": report.maturity,
                "domains": report.domain_scores,
                "debt": snapshot.contradiction_debt,
            }
            with open(_CALIBRATION_LOG, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")

            self._trim_log()
        except Exception as exc:
            logger.debug("Calibration persistence failed: %s", exc)

    def _trim_log(self) -> None:
        try:
            if not _CALIBRATION_LOG.exists():
                return
            lines = _CALIBRATION_LOG.read_text().splitlines()
            if len(lines) > _MAX_LOG_ENTRIES * 2:
                keep = lines[-_MAX_LOG_ENTRIES:]
                _CALIBRATION_LOG.write_text("\n".join(keep) + "\n")
        except Exception:
            pass

    @staticmethod
    def get_instance() -> TruthCalibrationEngine | None:
        return _instance
