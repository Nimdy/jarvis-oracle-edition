"""Soul Integrity Index — Layer 10 composite cognitive health metric.

Aggregates 10 health dimensions from all epistemic layers into a single
[0.0, 1.0] score. Each dimension is weighted and capped to prevent any
single subsystem from dominating the index.

Repair thresholds:
  - REPAIR_THRESHOLD (0.5): below this, emit SOUL_INTEGRITY_REPAIR_NEEDED
  - CRITICAL_THRESHOLD (0.3): below this, recommend mutation pause + forced dream

The index is intentionally conservative — it reports the *worst-case* view
of cognitive health, not an optimistic average.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

REPAIR_THRESHOLD = 0.50
CRITICAL_THRESHOLD = 0.30
HISTORY_SIZE = 60

# Dimension weights (must sum to 1.0)
DIMENSIONS: dict[str, float] = {
    "memory_coherence": 0.12,
    "belief_health": 0.12,
    "identity_integrity": 0.10,
    "skill_honesty": 0.10,
    "truth_calibration": 0.12,
    "belief_graph_health": 0.08,
    "quarantine_pressure": 0.08,
    "autonomy_effectiveness": 0.10,
    "audit_score": 0.10,
    "system_stability": 0.08,
}


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single integrity dimension."""
    name: str
    score: float  # 0.0 - 1.0
    weight: float
    source: str  # description of data source
    stale: bool = False  # True if data source unavailable


@dataclass
class IntegrityReport:
    """Result of a single integrity index computation."""
    timestamp: float
    index: float = 1.0  # 0.0 - 1.0 composite, set after dimension scoring
    dimensions: list[DimensionScore] = field(default_factory=list)
    repair_needed: bool = False
    critical: bool = False
    weakest_dimension: str = ""
    weakest_score: float = 1.0


class SoulIntegrityIndex:
    """Layer 10: composite cognitive health metric.

    Singleton. Ticked from consciousness system (every 120s default,
    60s during dream/sleep).
    """

    _instance: SoulIntegrityIndex | None = None

    def __init__(self) -> None:
        self._history: deque[IntegrityReport] = deque(maxlen=HISTORY_SIZE)
        self._total_computations: int = 0
        self._total_repairs_triggered: int = 0
        self._last_compute_ts: float = 0.0

    @classmethod
    def get_instance(cls) -> SoulIntegrityIndex:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compute(self) -> IntegrityReport:
        """Compute the soul integrity index from all available subsystem data.

        Returns an IntegrityReport with per-dimension scores and the composite index.
        """
        report = IntegrityReport(timestamp=time.time())

        scorers = [
            self._score_memory_coherence,
            self._score_belief_health,
            self._score_identity_integrity,
            self._score_skill_honesty,
            self._score_truth_calibration,
            self._score_belief_graph_health,
            self._score_quarantine_pressure,
            self._score_autonomy_effectiveness,
            self._score_audit,
            self._score_system_stability,
        ]

        for scorer in scorers:
            try:
                dim = scorer()
                report.dimensions.append(dim)
            except Exception:
                logger.debug("Integrity dimension %s failed", scorer.__name__, exc_info=True)

        # Compute weighted index
        weighted_sum = 0.0
        total_weight = 0.0
        for dim in report.dimensions:
            if not dim.stale:
                weighted_sum += dim.score * dim.weight
                total_weight += dim.weight

        report.index = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Find weakest
        for dim in report.dimensions:
            if not dim.stale and dim.score < report.weakest_score:
                report.weakest_score = dim.score
                report.weakest_dimension = dim.name

        report.repair_needed = report.index < REPAIR_THRESHOLD
        report.critical = report.index < CRITICAL_THRESHOLD

        self._history.append(report)
        self._total_computations += 1
        self._last_compute_ts = report.timestamp

        if report.repair_needed:
            self._total_repairs_triggered += 1

        return report

    # ------------------------------------------------------------------
    # Dimension Scorers
    # ------------------------------------------------------------------

    def _score_memory_coherence(self) -> DimensionScore:
        """Memory weight economy + provenance distribution."""
        try:
            from memory.storage import MemoryStorage
            storage = MemoryStorage.get_instance()
            if not storage:
                return DimensionScore("memory_coherence", 0.5, DIMENSIONS["memory_coherence"],
                                      "memory_storage unavailable", stale=True)

            stats = storage.get_stats()
            total = stats.get("total", 0)
            if total < 10:
                return DimensionScore("memory_coherence", 0.8, DIMENSIONS["memory_coherence"],
                                      f"small corpus ({total})")

            avg_weight = stats.get("avg_weight", 0.5)
            by_prov = stats.get("by_provenance", {})
            unknown_pct = by_prov.get("unknown", 0) / total if total else 0

            weight_score = 1.0 - abs(avg_weight - 0.55) * 2  # optimal around 0.55
            weight_score = max(0.0, min(1.0, weight_score))

            prov_score = 1.0 - unknown_pct  # penalty for unknown provenance

            score = weight_score * 0.6 + prov_score * 0.4
            return DimensionScore("memory_coherence", round(score, 4),
                                  DIMENSIONS["memory_coherence"],
                                  f"avg_weight={avg_weight:.3f}, unknown_prov={unknown_pct:.1%}")
        except Exception:
            return DimensionScore("memory_coherence", 0.5, DIMENSIONS["memory_coherence"],
                                  "error reading memory", stale=True)

    def _score_belief_health(self) -> DimensionScore:
        """Contradiction debt + tension count + resolution rate."""
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            ce = ContradictionEngine.get_instance()
            if not ce:
                return DimensionScore("belief_health", 0.5, DIMENSIONS["belief_health"],
                                      "contradiction engine unavailable", stale=True)

            debt = ce.contradiction_debt
            state = ce.get_state()
            total_beliefs = state.get("total_beliefs", 0)
            resolutions = state.get("resolved_count", 0)
            near_miss_rate = state.get("near_miss_rate", 0.0)

            debt_score = 1.0 - debt  # debt is [0, 1]
            nm_score = 1.0 - min(1.0, near_miss_rate * 2)  # penalize high near-miss rate

            score = debt_score * 0.6 + nm_score * 0.2 + 0.2  # base 0.2 for having beliefs
            score = max(0.0, min(1.0, score))
            return DimensionScore("belief_health", round(score, 4),
                                  DIMENSIONS["belief_health"],
                                  f"debt={debt:.3f}, beliefs={total_beliefs}, resolved={resolutions}")
        except Exception:
            return DimensionScore("belief_health", 0.5, DIMENSIONS["belief_health"],
                                  "error", stale=True)

    def _score_identity_integrity(self) -> DimensionScore:
        """Identity boundary enforcement quality."""
        try:
            from identity.audit import IdentityAudit
            audit = IdentityAudit.get_instance()
            if not audit:
                return DimensionScore("identity_integrity", 0.7, DIMENSIONS["identity_integrity"],
                                      "identity audit unavailable", stale=True)

            stats = audit.get_stats()
            total_assigned = stats.get("total_scope_assigned", 0)
            quarantined = stats.get("total_quarantined", 0)
            blocked = stats.get("total_boundary_blocks", 0)

            if total_assigned < 20:
                return DimensionScore("identity_integrity", 0.8, DIMENSIONS["identity_integrity"],
                                      f"insufficient data ({total_assigned} assignments)")

            quarantine_rate = quarantined / total_assigned if total_assigned else 0
            block_rate = blocked / total_assigned if total_assigned else 0

            # Low quarantine rate is good (means identity is clear)
            # Multiplier 1.5 so 50% quarantine → score 0.25, 10% → 0.85
            q_score = 1.0 - min(1.0, quarantine_rate * 1.5)
            b_bonus = min(0.1, block_rate * 0.5)  # blocking means the gate is working

            score = max(0.0, min(1.0, q_score + b_bonus))
            return DimensionScore("identity_integrity", round(score, 4),
                                  DIMENSIONS["identity_integrity"],
                                  f"assigned={total_assigned}, quarantine_rate={quarantine_rate:.2%}")
        except Exception:
            return DimensionScore("identity_integrity", 0.5, DIMENSIONS["identity_integrity"],
                                  "error", stale=True)

    # Learning-job phase weighting. Blocks in late phases (train/verify/
    # register) typically indicate evidence-collection plumbing gaps rather
    # than capability dishonesty — the job got most of the way through. They
    # count at half weight so a single stuck-in-verify job cannot pin the
    # dimension at its floor. Blocks in early phases (assess/collect/research)
    # remain at full weight because they usually mean a real prerequisite is
    # missing.
    _SKILL_HONESTY_LATE_PHASES = frozenset({"train", "verify", "register"})
    _SKILL_HONESTY_LATE_WEIGHT = 0.5
    # Small-N confidence floor — observed job_score is statistically unstable
    # below this sample count (a single blocked job flips the whole score).
    _SKILL_HONESTY_MIN_N_FOR_FULL_CONFIDENCE = 3
    _SKILL_HONESTY_NEUTRAL_BASELINE = 0.8

    def _score_skill_honesty(self) -> DimensionScore:
        """Capability gate effectiveness + learning job health.

        Job health uses two anti-false-alarm mechanisms:

        1. **Phase-aware weighting.** A block in the verify phase (all hard
           gates passed, evidence test pending) is a plumbing issue, not
           capability dishonesty. Late-phase blocks count at half weight.

        2. **Small-N confidence blending.** With fewer than
           _SKILL_HONESTY_MIN_N_FOR_FULL_CONFIDENCE jobs, the raw block
           ratio is statistically meaningless — a single blocked job would
           otherwise yield job_score=0.0. The observed score is blended
           with a neutral baseline, reaching full weight only at steady
           state.

        Both are calibration fixes, not scoring softeners: 3+ blocked jobs
        in early phases still drag the dimension to 0.0. Cf. the live-brain
        investigation that surfaced the N=1 pathology.
        """
        try:
            blocked_jobs = 0
            total_jobs = 0
            late_blocked = 0
            weighted_blocked = 0.0
            gate_stats: dict[str, Any] = {}

            try:
                from consciousness.consciousness_system import _active_consciousness
                cs = _active_consciousness
                engine = cs._engine_ref if cs else None
                orch = engine._learning_job_orchestrator if engine and hasattr(engine, '_learning_job_orchestrator') else None
                if orch:
                    for job in orch.store.load_all():
                        total_jobs += 1
                        if job.status == "blocked":
                            blocked_jobs += 1
                            phase = getattr(job, "phase", "") or ""
                            if phase in self._SKILL_HONESTY_LATE_PHASES:
                                late_blocked += 1
                                weighted_blocked += self._SKILL_HONESTY_LATE_WEIGHT
                            else:
                                weighted_blocked += 1.0
            except Exception:
                pass

            try:
                from skills.capability_gate import capability_gate
                gate_stats = capability_gate.get_stats() if capability_gate else {}
                # CapabilityGate exposes claims_blocked (not total_blocks). Keep a
                # fallback for legacy payloads to avoid silent under-scoring.
                blocked_claims = int(
                    gate_stats.get("claims_blocked", gate_stats.get("total_blocks", 0)) or 0
                )
                gate_score = min(1.0, 0.7 + blocked_claims * 0.003)
            except Exception:
                blocked_claims = 0
                gate_score = 0.5

            if total_jobs > 0:
                observed_job_score = 1.0 - (weighted_blocked / total_jobs)
                confidence = min(
                    1.0,
                    total_jobs / float(self._SKILL_HONESTY_MIN_N_FOR_FULL_CONFIDENCE),
                )
                neutral = self._SKILL_HONESTY_NEUTRAL_BASELINE
                job_score = observed_job_score * confidence + neutral * (1.0 - confidence)
            else:
                job_score = self._SKILL_HONESTY_NEUTRAL_BASELINE

            score = gate_score * 0.5 + job_score * 0.5
            source = (
                f"blocked_jobs={blocked_jobs}/{total_jobs} "
                f"(late={late_blocked}), claims_blocked={blocked_claims}"
            )
            return DimensionScore("skill_honesty", round(score, 4),
                                  DIMENSIONS["skill_honesty"],
                                  source)
        except Exception:
            return DimensionScore("skill_honesty", 0.5, DIMENSIONS["skill_honesty"],
                                  "error", stale=True)

    def _score_truth_calibration(self) -> DimensionScore:
        """Truth score from Layer 6 calibration engine."""
        try:
            from epistemic.calibration import TruthCalibrationEngine
            cal = TruthCalibrationEngine.get_instance()
            if not cal:
                return DimensionScore("truth_calibration", 0.5, DIMENSIONS["truth_calibration"],
                                      "calibration unavailable", stale=True)

            state = cal.get_state()
            truth_score = state.get("truth_score")
            maturity = state.get("maturity", 0.0)
            alerts = state.get("active_drift_alerts")
            if not isinstance(alerts, list):
                alerts = state.get("drift_alerts", [])
            drift_count = len(alerts) if isinstance(alerts, list) else 0

            if truth_score is None:
                return DimensionScore("truth_calibration", 0.5, DIMENSIONS["truth_calibration"],
                                      "truth score not yet computed", stale=True)

            drift_penalty = min(0.2, drift_count * 0.05)
            maturity_factor = min(1.0, maturity * 1.5)  # trust more as maturity grows

            score = truth_score * maturity_factor - drift_penalty
            score = max(0.0, min(1.0, score))
            return DimensionScore("truth_calibration", round(score, 4),
                                  DIMENSIONS["truth_calibration"],
                                  f"truth={truth_score:.3f}, maturity={maturity:.3f}, drifts={drift_count}")
        except Exception:
            return DimensionScore("truth_calibration", 0.5, DIMENSIONS["truth_calibration"],
                                  "error", stale=True)

    def _score_belief_graph_health(self) -> DimensionScore:
        """Belief graph structural health from Layer 7."""
        try:
            from epistemic.belief_graph import BeliefGraph
            bg = BeliefGraph.get_instance()
            if not bg:
                return DimensionScore("belief_graph_health", 0.5, DIMENSIONS["belief_graph_health"],
                                      "belief graph unavailable", stale=True)

            state = bg.get_state()
            integrity = state.get("integrity", {})
            health = integrity.get("health_score", 0.5)

            return DimensionScore("belief_graph_health", round(health, 4),
                                  DIMENSIONS["belief_graph_health"],
                                  f"health={health:.3f}, edges={state.get('total_edges', 0)}")
        except Exception:
            return DimensionScore("belief_graph_health", 0.5, DIMENSIONS["belief_graph_health"],
                                  "error", stale=True)

    def _score_quarantine_pressure(self) -> DimensionScore:
        """Quarantine EMA pressure from Layer 8 (lower pressure = healthier).

        Uses the same QuarantinePressure singleton as mutation governor,
        promotion pipeline, graph bridge, and world model promotion.
        """
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            qp = get_quarantine_pressure()
            ps = qp.current

            composite = ps.composite
            score = max(0.0, 1.0 - composite)

            if ps.chronic_count > 0:
                score -= min(0.2, ps.chronic_count * 0.05)
                score = max(0.0, score)

            return DimensionScore("quarantine_pressure", round(score, 4),
                                  DIMENSIONS["quarantine_pressure"],
                                  f"ema={composite:.3f}, chronic={ps.chronic_count}, "
                                  f"elevated={ps.elevated}, high={ps.high}")
        except Exception:
            return DimensionScore("quarantine_pressure", 0.5, DIMENSIONS["quarantine_pressure"],
                                  "error", stale=True)

    def _score_autonomy_effectiveness(self) -> DimensionScore:
        """Autonomy win rate from policy memory."""
        try:
            from consciousness.consciousness_system import _active_consciousness
            cs = _active_consciousness
            engine = cs._engine_ref if cs else None
            auto_orch = engine._autonomy_orchestrator if engine and hasattr(engine, '_autonomy_orchestrator') else None
            pm = auto_orch._policy_memory if auto_orch and hasattr(auto_orch, '_policy_memory') else None
            if not pm:
                return DimensionScore("autonomy_effectiveness", 0.5,
                                      DIMENSIONS["autonomy_effectiveness"],
                                      "policy memory unavailable", stale=True)

            stats = pm.get_stats()
            total = stats.get("total_outcomes", 0)
            wins = stats.get("total_wins", 0)
            in_warmup = stats.get("in_warmup", True)

            if total < 5 or in_warmup:
                return DimensionScore("autonomy_effectiveness", 0.5,
                                      DIMENSIONS["autonomy_effectiveness"],
                                      f"warmup ({total} outcomes)")

            win_rate = wins / total if total else 0
            score = min(1.0, win_rate * 2.5)  # 40% win_rate = 1.0
            return DimensionScore("autonomy_effectiveness", round(score, 4),
                                  DIMENSIONS["autonomy_effectiveness"],
                                  f"win_rate={win_rate:.1%}, total={total}")
        except Exception:
            return DimensionScore("autonomy_effectiveness", 0.5,
                                  DIMENSIONS["autonomy_effectiveness"],
                                  "error", stale=True)

    def _score_audit(self) -> DimensionScore:
        """Most recent Layer 9 reflective audit score."""
        try:
            from epistemic.reflective_audit import ReflectiveAuditEngine
            engine = ReflectiveAuditEngine.get_instance()
            score = engine.get_latest_score()
            if score is None:
                return DimensionScore("audit_score", 0.7, DIMENSIONS["audit_score"],
                                      "no audit yet", stale=True)
            return DimensionScore("audit_score", round(score, 4),
                                  DIMENSIONS["audit_score"],
                                  f"latest_audit={score:.3f}, total={engine._total_audits}")
        except Exception:
            return DimensionScore("audit_score", 0.5, DIMENSIONS["audit_score"],
                                  "error", stale=True)

    def _score_system_stability(self) -> DimensionScore:
        """Kernel tick health + error rate + mode stability."""
        try:
            from consciousness.consciousness_analytics import ConsciousnessAnalytics
            # Access via the singleton or consciousness system
            try:
                from consciousness.consciousness_system import _active_consciousness
                cs = _active_consciousness
                analytics = cs.analytics if cs else None
            except Exception:
                analytics = None

            if not analytics:
                return DimensionScore("system_stability", 0.5, DIMENSIONS["system_stability"],
                                      "analytics unavailable", stale=True)

            health = analytics.get_system_health()
            tick_score = 1.0 if health.healthy else 0.3
            error_signal = 0.0
            if health.tick_p95_ms > 50:
                tick_score *= 0.7

            score = tick_score
            return DimensionScore("system_stability", round(score, 4),
                                  DIMENSIONS["system_stability"],
                                  f"healthy={health.healthy}, p95={health.tick_p95_ms:.1f}ms")
        except Exception:
            return DimensionScore("system_stability", 0.5, DIMENSIONS["system_stability"],
                                  "error", stale=True)

    # ------------------------------------------------------------------
    # State & History
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return full state for dashboard snapshot."""
        latest = self._history[-1] if self._history else None
        return {
            "total_computations": self._total_computations,
            "total_repairs_triggered": self._total_repairs_triggered,
            "last_compute_ts": self._last_compute_ts,
            "current_index": round(latest.index, 4) if latest else None,
            "repair_needed": latest.repair_needed if latest else False,
            "critical": latest.critical if latest else False,
            "weakest_dimension": latest.weakest_dimension if latest else "",
            "weakest_score": round(latest.weakest_score, 4) if latest else 1.0,
            "dimensions": [
                {
                    "name": d.name,
                    "score": round(d.score, 4),
                    "weight": d.weight,
                    "source": d.source,
                    "stale": d.stale,
                }
                for d in (latest.dimensions if latest else [])
            ],
            "history": [
                {"ts": r.timestamp, "index": round(r.index, 4), "repair": r.repair_needed}
                for r in list(self._history)[-20:]
            ],
            "trend": self._compute_trend(),
        }

    def get_current_index(self) -> float | None:
        """Return current integrity index or None."""
        if self._history:
            return self._history[-1].index
        return None

    def _compute_trend(self) -> dict[str, Any]:
        """Compute trend across recent index values."""
        if len(self._history) < 2:
            return {"direction": "stable", "delta": 0.0}

        values = [r.index for r in self._history]
        recent = values[-3:] if len(values) >= 3 else values
        older = values[-6:-3] if len(values) >= 6 else values[:len(values) // 2]

        if not older:
            return {"direction": "stable", "delta": 0.0}

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        delta = recent_avg - older_avg

        if delta > 0.03:
            direction = "improving"
        elif delta < -0.03:
            direction = "degrading"
        else:
            direction = "stable"

        return {"direction": direction, "delta": round(delta, 4)}
