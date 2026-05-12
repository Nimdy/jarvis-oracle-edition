"""Reflective Audit Engine — Layer 9 epistemic introspection.

Scans subsystem state for 6 audit dimensions:
  1. incorrect_learning   — beliefs with high contradiction pressure or low calibration
  2. identity_breach      — cross-identity contamination, quarantined writes leaking
  3. source_trust         — over-reliance on low-provenance memories, stale externals
  4. autonomy_failure     — repeated failing patterns in autonomy policy memory
  5. skill_stagnation     — stuck/blocked learning jobs, unverified claims persisting
  6. memory_hygiene       — orphaned associations, weight distribution anomalies

Each scan produces AuditFindings. Findings are accumulated into an AuditReport
with a severity-weighted score. The engine maintains a ring buffer of recent
reports for trend detection.

Read-only: never mutates beliefs, memories, policy, or identity state.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

AuditCategory = Literal[
    "incorrect_learning",
    "identity_breach",
    "source_trust",
    "autonomy_failure",
    "skill_stagnation",
    "memory_hygiene",
    "ingestion_health",
    "spatial_integrity",
]

AuditSeverity = Literal["info", "warning", "critical"]

SEVERITY_WEIGHT: dict[AuditSeverity, float] = {
    "info": 0.1,
    "warning": 0.4,
    "critical": 1.0,
}

MAX_FINDINGS_PER_REPORT = 50
REPORT_HISTORY_SIZE = 30
MAX_RECOMMENDATIONS_PER_FINDING = 3


@dataclass(frozen=True)
class AuditFinding:
    """A single audit observation with optional recommendation."""
    category: AuditCategory
    severity: AuditSeverity
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    related_ids: tuple[str, ...] = ()


@dataclass
class AuditReport:
    """Aggregate result of a single audit cycle."""
    timestamp: float
    findings: list[AuditFinding] = field(default_factory=list)
    duration_ms: float = 0.0
    categories_scanned: int = 0
    score: float = 1.0  # 1.0 = perfect, 0.0 = critical issues

    def compute_score(self) -> float:
        if not self.findings:
            return 1.0
        total_severity = sum(SEVERITY_WEIGHT[f.severity] for f in self.findings)
        return max(0.0, 1.0 - min(1.0, total_severity / 5.0))


class ReflectiveAuditEngine:
    """Layer 9: introspective audit engine for sleep/dream cycles.

    Singleton pattern consistent with other epistemic layers. Lazy-initialized
    from consciousness_system._run_reflective_audit().
    """

    _instance: ReflectiveAuditEngine | None = None

    def __init__(self) -> None:
        self._reports: deque[AuditReport] = deque(maxlen=REPORT_HISTORY_SIZE)
        self._total_audits: int = 0
        self._total_findings: int = 0
        self._finding_counts: dict[AuditCategory, int] = {}
        self._last_audit_ts: float = 0.0

    @classmethod
    def get_instance(cls) -> ReflectiveAuditEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run_audit(self) -> AuditReport:
        """Execute a full audit cycle across all 6 dimensions.

        Returns an AuditReport with findings. This method is designed to be
        called from the consciousness tick loop during dream/sleep modes.
        """
        start = time.monotonic()
        report = AuditReport(timestamp=time.time())

        scanners = [
            self._scan_incorrect_learning,
            self._scan_identity_breach,
            self._scan_source_trust,
            self._scan_autonomy_failure,
            self._scan_skill_stagnation,
            self._scan_memory_hygiene,
            self._scan_ingestion_health,
            self._scan_spatial_integrity,
        ]

        for scanner in scanners:
            try:
                findings = scanner()
                for f in findings[:MAX_FINDINGS_PER_REPORT - len(report.findings)]:
                    report.findings.append(f)
                report.categories_scanned += 1
            except Exception:
                logger.debug("Audit scanner %s failed", scanner.__name__, exc_info=True)

        report.duration_ms = (time.monotonic() - start) * 1000
        report.score = report.compute_score()

        self._reports.append(report)
        self._total_audits += 1
        self._total_findings += len(report.findings)
        self._last_audit_ts = report.timestamp
        for f in report.findings:
            self._finding_counts[f.category] = self._finding_counts.get(f.category, 0) + 1

        return report

    # ------------------------------------------------------------------
    # Dimension 1: Incorrect Learning
    # ------------------------------------------------------------------

    def _scan_incorrect_learning(self) -> list[AuditFinding]:
        """Detect beliefs with high contradiction pressure or low-calibration backing."""
        findings: list[AuditFinding] = []
        try:
            from epistemic.belief_graph import BeliefGraph
            bg = BeliefGraph.get_instance()
            if not bg:
                return findings

            from epistemic.contradiction_engine import ContradictionEngine
            ce = ContradictionEngine.get_instance()
            beliefs = ce.belief_store.get_active_beliefs() if ce else []

            edge_store = bg._edge_store

            for belief in beliefs:
                if belief.epistemic_status in ("questioned", "provisional"):
                    continue
                # High contradiction pressure on stabilized belief = suspect
                pressure = 0.0
                support = 0.0
                for edge in edge_store.get_incoming(belief.belief_id) + edge_store.get_outgoing(belief.belief_id):
                    if edge.edge_type == "contradicts":
                        pressure += edge.strength
                    elif edge.edge_type == "supports":
                        support += edge.strength

                if pressure > 0.3 and pressure > support * 0.5:
                    findings.append(AuditFinding(
                        category="incorrect_learning",
                        severity="warning" if pressure > 0.5 else "info",
                        description=f"Belief '{belief.rendered_claim[:60]}' has high contradiction pressure ({pressure:.2f}) relative to support ({support:.2f})",
                        evidence={
                            "belief_id": belief.belief_id,
                            "contradiction_pressure": round(pressure, 3),
                            "support_strength": round(support, 3),
                            "epistemic_status": belief.epistemic_status,
                            "extraction_confidence": belief.extraction_confidence,
                        },
                        recommendation="Consider downgrading epistemic_status to 'questioned' or scheduling for re-evaluation",
                        related_ids=(belief.belief_id,),
                    ))

                # Low extraction confidence but stabilized = potentially wrong
                if belief.extraction_confidence < 0.3 and belief.epistemic_status == "stabilized":
                    findings.append(AuditFinding(
                        category="incorrect_learning",
                        severity="info",
                        description=f"Stabilized belief '{belief.rendered_claim[:60]}' has low extraction confidence ({belief.extraction_confidence:.2f})",
                        evidence={
                            "belief_id": belief.belief_id,
                            "extraction_confidence": belief.extraction_confidence,
                        },
                        recommendation="Re-extract from source memory with improved claim extractor",
                        related_ids=(belief.belief_id,),
                    ))

        except Exception:
            logger.debug("Incorrect learning scan failed", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # Dimension 2: Identity Boundary Breach
    # ------------------------------------------------------------------

    def _scan_identity_breach(self) -> list[AuditFinding]:
        """Check for identity boundary violations in recent memory and beliefs."""
        findings: list[AuditFinding] = []
        try:
            from identity.audit import IdentityAudit
            audit = IdentityAudit.get_instance()
            if not audit:
                return findings

            stats = audit.get_stats()

            quarantined = stats.get("total_quarantined", 0)
            blocked = stats.get("total_boundary_blocks", 0)

            if quarantined > 10:
                findings.append(AuditFinding(
                    category="identity_breach",
                    severity="warning",
                    description=f"High quarantine volume: {quarantined} memories quarantined due to low identity confidence",
                    evidence={"quarantined_count": quarantined, "blocked_count": blocked},
                    recommendation="Review identity enrollment; consider re-enrolling voice/face profiles",
                ))

            # Check for needs_resolution memories that may be leaking into preferences
            try:
                from memory.storage import MemoryStorage
                storage = MemoryStorage.get_instance()
                if storage:
                    unresolved = 0
                    for m in storage.get_recent(200):
                        if getattr(m, "identity_needs_resolution", False):
                            unresolved += 1
                    if unresolved > 20:
                        findings.append(AuditFinding(
                            category="identity_breach",
                            severity="warning",
                            description=f"{unresolved} recent memories have unresolved identity — may contaminate preference injection",
                            evidence={"unresolved_count": unresolved},
                            recommendation="Identity enrollment or speaker clarification needed",
                        ))
            except Exception:
                pass

            # Check identity flip count from Layer 3A
            try:
                from perception.identity_fusion import _active_instance as fusion
                if fusion:
                    status = fusion.get_status()
                    flips = status.get("flip_count", 0)
                    if flips > 10:
                        findings.append(AuditFinding(
                            category="identity_breach",
                            severity="warning" if flips > 20 else "info",
                            description=f"Identity flip count is {flips} — indicates unstable biometric matching",
                            evidence={"flip_count": flips},
                            recommendation="Review voice/face profile quality; consider re-enrollment",
                        ))
            except Exception:
                pass

        except Exception:
            logger.debug("Identity breach scan failed", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # Dimension 3: Source Trust
    # ------------------------------------------------------------------

    def _scan_source_trust(self) -> list[AuditFinding]:
        """Detect over-reliance on low-provenance or stale external sources."""
        findings: list[AuditFinding] = []
        try:
            from memory.storage import MemoryStorage
            storage = MemoryStorage.get_instance()
            if not storage:
                return findings

            stats = storage.get_stats()
            by_prov = stats.get("by_provenance", {})
            total = sum(by_prov.values()) if by_prov else 0

            if total < 10:
                return findings

            unknown_count = by_prov.get("unknown", 0)
            unknown_pct = unknown_count / total if total else 0

            if unknown_pct > 0.2:
                findings.append(AuditFinding(
                    category="source_trust",
                    severity="warning" if unknown_pct > 0.4 else "info",
                    description=f"{unknown_count}/{total} memories ({unknown_pct:.0%}) have unknown provenance — Layer 2 tagging incomplete",
                    evidence={"unknown_count": unknown_count, "total": total, "pct": round(unknown_pct, 3)},
                    recommendation="Investigate memory creation paths bypassing provenance tagging",
                ))

            # Check if model_inference dominates without enough observed/user_claim grounding
            mi_count = by_prov.get("model_inference", 0)
            observed = by_prov.get("observed", 0)
            user_claim = by_prov.get("user_claim", 0)
            grounded = observed + user_claim

            if mi_count > grounded * 3 and mi_count > 30:
                findings.append(AuditFinding(
                    category="source_trust",
                    severity="info",
                    description=f"Model inference memories ({mi_count}) outnumber grounded sources ({grounded}) by {mi_count/max(1,grounded):.1f}x",
                    evidence={
                        "model_inference": mi_count,
                        "observed": observed,
                        "user_claim": user_claim,
                        "ratio": round(mi_count / max(1, grounded), 2),
                    },
                    recommendation="Increase observational data collection; model inferences may be self-reinforcing",
                ))

            # Check truth calibration for per-provenance accuracy issues
            try:
                from epistemic.calibration import TruthCalibrationEngine
                cal = TruthCalibrationEngine.get_instance()
                if cal:
                    cal_state = cal.get_state()
                    prov_acc = cal_state.get("per_provenance_accuracy", {})
                    for prov, acc in prov_acc.items():
                        if isinstance(acc, (int, float)) and acc < 0.4 and prov not in ("unknown",):
                            findings.append(AuditFinding(
                                category="source_trust",
                                severity="warning",
                                description=f"Provenance '{prov}' has low accuracy: {acc:.2f}",
                                evidence={"provenance": prov, "accuracy": acc},
                                recommendation=f"Reduce trust weight for '{prov}' sources or improve extraction quality",
                            ))
            except Exception:
                pass

        except Exception:
            logger.debug("Source trust scan failed", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # Dimension 4: Autonomy Failure Patterns
    # ------------------------------------------------------------------

    def _scan_autonomy_failure(self) -> list[AuditFinding]:
        """Identify repeated failing patterns in the autonomy policy memory."""
        findings: list[AuditFinding] = []
        try:
            from consciousness.consciousness_system import _active_consciousness
            cs = _active_consciousness
            engine = cs._engine_ref if cs else None
            auto_orch = engine._autonomy_orchestrator if engine and hasattr(engine, '_autonomy_orchestrator') else None
            pm = auto_orch._policy_memory if auto_orch and hasattr(auto_orch, '_policy_memory') else None
            if not pm:
                return findings

            stats = pm.get_stats()
            total = stats.get("total_outcomes", 0)
            wins = stats.get("total_wins", 0)
            losses = stats.get("total_losses", 0)

            if total < 10:
                return findings

            win_rate = wins / total if total else 0

            if win_rate < 0.25:
                findings.append(AuditFinding(
                    category="autonomy_failure",
                    severity="critical" if win_rate < 0.15 else "warning",
                    description=f"Autonomy win rate is {win_rate:.0%} ({wins}/{total}) — system is exploring without effective learning",
                    evidence={"wins": wins, "losses": losses, "total": total, "win_rate": round(win_rate, 3)},
                    recommendation="Review metric triggers and drive strategy; consider raising MIN_MEANINGFUL_DELTA or adjusting drive urgency damping",
                ))

            # Identify specific avoid patterns
            avoid = stats.get("avoid_patterns", [])
            for pattern in avoid[:5]:
                tags = pattern.get("tags", [])
                p_total = pattern.get("total", 0)
                p_wr = pattern.get("win_rate", 0)
                avg_delta = pattern.get("avg_delta", 0)
                if p_total >= 3 and p_wr < 0.15:
                    findings.append(AuditFinding(
                        category="autonomy_failure",
                        severity="warning",
                        description=f"Autonomy pattern {tags} has {p_wr:.0%} win rate over {p_total} attempts (avg_delta={avg_delta:.3f})",
                        evidence={"tags": tags, "total": p_total, "win_rate": p_wr, "avg_delta": avg_delta},
                        recommendation=f"Veto pattern {tags} or rotate to alternative tool/action",
                        related_ids=tuple(str(t) for t in tags),
                    ))

            # Check calibration domain score for autonomy
            try:
                from epistemic.calibration import TruthCalibrationEngine
                cal = TruthCalibrationEngine.get_instance()
                if cal:
                    cal_state = cal.get_state()
                    auto_score = cal_state.get("domain_scores", {}).get("autonomy", 1.0)
                    if isinstance(auto_score, (int, float)) and auto_score < 0.4:
                        findings.append(AuditFinding(
                            category="autonomy_failure",
                            severity="critical",
                            description=f"Autonomy calibration domain score is critically low: {auto_score:.3f}",
                            evidence={"domain_score": auto_score},
                            recommendation="Pause autonomy research until win rate improves; focus on consolidating existing knowledge",
                        ))
            except Exception:
                pass

        except Exception:
            logger.debug("Autonomy failure scan failed", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # Dimension 5: Skill Stagnation
    # ------------------------------------------------------------------

    def _scan_skill_stagnation(self) -> list[AuditFinding]:
        """Detect stuck, blocked, or stale learning jobs and unverified claims."""
        findings: list[AuditFinding] = []
        try:
            from consciousness.consciousness_system import _active_consciousness
            cs = _active_consciousness
            engine = cs._engine_ref if cs else None
            orch = engine._learning_job_orchestrator if engine and hasattr(engine, '_learning_job_orchestrator') else None
            if not orch:
                return findings

            blocked_count = 0
            stale_count = 0
            for job in orch.store.load_all():
                if job.status == "blocked":
                    blocked_count += 1
                if job.status == "active":
                    try:
                        import datetime as _dt
                        updated_ts = _dt.datetime.fromisoformat(
                            job.updated_at.rstrip("Z")
                        ).replace(tzinfo=_dt.timezone.utc).timestamp()
                        age = time.time() - updated_ts
                        if age > 7200:  # 2 hours stale
                            stale_count += 1
                    except Exception:
                        pass

            if blocked_count > 5:
                findings.append(AuditFinding(
                    category="skill_stagnation",
                    severity="warning" if blocked_count > 10 else "info",
                    description=f"{blocked_count} learning jobs are blocked — accumulated stale state polluting skill registry",
                    evidence={"blocked_count": blocked_count},
                    recommendation="Clean up blocked jobs: mark as failed and propagate to skill registry, or archive",
                ))

            if stale_count > 0:
                findings.append(AuditFinding(
                    category="skill_stagnation",
                    severity="warning",
                    description=f"{stale_count} active learning jobs are stale (>2h without progress)",
                    evidence={"stale_count": stale_count},
                    recommendation="Check executor dispatch; stale jobs may have dead executor pipelines",
                ))

            try:
                from skills.capability_gate import capability_gate as gate
                if gate:
                    gate_stats = gate.get_stats()
                    blocks = gate_stats.get("total_blocks", 0)
                    if blocks > 50:
                        findings.append(AuditFinding(
                            category="skill_stagnation",
                            severity="info",
                            description=f"Capability gate has blocked {blocks} claims — LLM frequently generating unverified capability statements",
                            evidence={"total_blocks": blocks},
                            recommendation="Review LLM system prompt skill injection; consider stronger anti-hallucination directives",
                        ))
            except Exception:
                pass

        except Exception:
            logger.debug("Skill stagnation scan failed", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # Dimension 6: Memory Hygiene
    # ------------------------------------------------------------------

    def _scan_memory_hygiene(self) -> list[AuditFinding]:
        """Check memory corpus health: orphans, weight distribution, integrity."""
        findings: list[AuditFinding] = []
        try:
            from memory.storage import MemoryStorage
            storage = MemoryStorage.get_instance()
            if not storage:
                return findings

            stats = storage.get_stats()
            total = stats.get("total", 0)
            if total < 10:
                return findings

            # Weight distribution
            avg_weight = stats.get("avg_weight", 0.5)
            if avg_weight > 0.75:
                findings.append(AuditFinding(
                    category="memory_hygiene",
                    severity="info",
                    description=f"Average memory weight {avg_weight:.3f} is high — weight economy may need adjustment",
                    evidence={"avg_weight": avg_weight, "total": total},
                    recommendation="Increase decay rates or lower initial weight caps to create salience gradient",
                ))

            # Orphan associations
            orphan_count = stats.get("orphan_associations", 0)
            if orphan_count > 20:
                findings.append(AuditFinding(
                    category="memory_hygiene",
                    severity="info",
                    description=f"{orphan_count} orphaned associations in memory graph",
                    evidence={"orphan_count": orphan_count},
                    recommendation="Run association repair cycle in next dream phase",
                ))

            # Belief graph health
            try:
                from epistemic.belief_graph import BeliefGraph
                bg = BeliefGraph.get_instance()
                if bg:
                    bg_state = bg.get_state()
                    integrity = bg_state.get("integrity", {})
                    health = integrity.get("health_score", 1.0)
                    orphan_rate = integrity.get("orphan_rate", 0.0)
                    fragmentation = integrity.get("fragmentation", 0.0)

                    if health < 0.7:
                        findings.append(AuditFinding(
                            category="memory_hygiene",
                            severity="warning",
                            description=f"Belief graph health is {health:.3f} (orphan_rate={orphan_rate:.3f}, fragmentation={fragmentation:.3f})",
                            evidence={
                                "health_score": health,
                                "orphan_rate": orphan_rate,
                                "fragmentation": fragmentation,
                            },
                            recommendation="Run belief graph compaction and bridge re-scan in next dream cycle",
                        ))
            except Exception:
                pass

        except Exception:
            logger.debug("Memory hygiene scan failed", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # Dimension 7: Ingestion Health
    # ------------------------------------------------------------------

    def _scan_ingestion_health(self) -> list[AuditFinding]:
        """Check document library ingestion health: source count, quality, chunk coverage."""
        findings: list[AuditFinding] = []
        try:
            from library.db import get_connection
            conn = get_connection()
            row = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
            total_sources = row[0] if row else 0

            if total_sources == 0:
                findings.append(AuditFinding(
                    category="ingestion_health",
                    severity="info",
                    description="No documents in library — knowledge pipeline inactive",
                    evidence={"total_sources": 0},
                    recommendation="Ingest at least one source to activate knowledge pipeline",
                ))
                return findings

            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            total_chunks = row[0] if row else 0

            row = conn.execute(
                "SELECT AVG(quality_score) FROM sources WHERE quality_score > 0"
            ).fetchone()
            avg_quality = round(row[0], 3) if row and row[0] else 0.0

            if avg_quality < 0.5 and total_sources >= 3:
                findings.append(AuditFinding(
                    category="ingestion_health",
                    severity="warning",
                    description=f"Average ingestion quality score {avg_quality:.3f} is low across {total_sources} sources",
                    evidence={"avg_quality": avg_quality, "total_sources": total_sources},
                    recommendation="Review source selection criteria — low quality sources may degrade belief formation",
                ))

            if total_chunks == 0 and total_sources > 0:
                findings.append(AuditFinding(
                    category="ingestion_health",
                    severity="warning",
                    description=f"{total_sources} sources but 0 chunks — chunking pipeline may be failing",
                    evidence={"total_sources": total_sources, "total_chunks": 0},
                    recommendation="Check chunker output and embedding pipeline",
                ))

        except Exception:
            logger.debug("Ingestion health scan failed", exc_info=True)
        return findings

    def _scan_spatial_integrity(self) -> list[AuditFinding]:
        """Check spatial intelligence subsystem health."""
        findings: list[AuditFinding] = []
        try:
            engine = self._engine
            if not engine:
                return findings
            cs = getattr(engine, "_consciousness", None) or getattr(engine, "consciousness", None)
            if not cs:
                return findings
            perc_orch = getattr(cs, "_perc_orch", None)
            if not perc_orch:
                return findings
            spatial_est = getattr(perc_orch, "_spatial_estimator", None)
            if spatial_est is None:
                return findings

            state = spatial_est.get_state()
            cal_state = state.get("calibration", {}).get("state", "invalid")

            if cal_state == "invalid":
                findings.append(AuditFinding(
                    category="spatial_integrity",
                    severity="info",
                    description="Spatial calibration is invalid — no spatial estimates active",
                    evidence={"calibration_state": cal_state},
                    recommendation="Run calibration to enable spatial intelligence",
                ))
            elif cal_state == "stale":
                findings.append(AuditFinding(
                    category="spatial_integrity",
                    severity="warning",
                    description="Spatial calibration is stale — promoted claims blocked",
                    evidence={"calibration_state": cal_state,
                              "age_s": state.get("calibration", {}).get("age_s")},
                    recommendation="Reverify calibration to restore full spatial pipeline",
                ))

            spatial_val = getattr(perc_orch, "_spatial_validator", None)
            if spatial_val:
                val_state = spatial_val.get_state()
                rejections = val_state.get("total_rejections", 0)
                promoted = val_state.get("total_promoted", 0)
                total = rejections + promoted
                if total > 10 and promoted == 0:
                    findings.append(AuditFinding(
                        category="spatial_integrity",
                        severity="warning",
                        description=f"Spatial validator has {rejections} rejections and 0 promotions",
                        evidence={"rejections": rejections, "promoted": promoted},
                        recommendation="Check spatial thresholds and calibration quality",
                    ))
        except Exception:
            logger.debug("Spatial integrity scan failed", exc_info=True)
        return findings

    # ------------------------------------------------------------------
    # State & Reporting
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return audit engine state for dashboard snapshot."""
        recent = list(self._reports)[-5:] if self._reports else []
        return {
            "total_audits": self._total_audits,
            "total_findings": self._total_findings,
            "last_audit_ts": self._last_audit_ts,
            "finding_counts": dict(self._finding_counts),
            "recent_reports": [
                {
                    "timestamp": r.timestamp,
                    "score": round(r.score, 4),
                    "finding_count": len(r.findings),
                    "duration_ms": round(r.duration_ms, 1),
                    "categories_scanned": r.categories_scanned,
                    "findings": [
                        {
                            "category": f.category,
                            "severity": f.severity,
                            "description": f.description[:120],
                            "recommendation": f.recommendation[:120],
                        }
                        for f in r.findings[:10]
                    ],
                }
                for r in recent
            ],
            "trend": self._compute_trend(),
        }

    def get_latest_score(self) -> float | None:
        """Return the most recent audit score, or None if no audits yet."""
        if self._reports:
            return self._reports[-1].score
        return None

    def get_latest_report(self) -> AuditReport | None:
        """Return the most recent audit report."""
        if self._reports:
            return self._reports[-1]
        return None

    def _compute_trend(self) -> dict[str, Any]:
        """Compute trend across recent audit scores."""
        if len(self._reports) < 2:
            return {"direction": "stable", "delta": 0.0}

        scores = [r.score for r in self._reports]
        recent = scores[-3:] if len(scores) >= 3 else scores
        older = scores[-6:-3] if len(scores) >= 6 else scores[:len(scores) // 2]

        if not older:
            return {"direction": "stable", "delta": 0.0}

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        delta = recent_avg - older_avg

        if delta > 0.05:
            direction = "improving"
        elif delta < -0.05:
            direction = "degrading"
        else:
            direction = "stable"

        return {"direction": direction, "delta": round(delta, 4)}
