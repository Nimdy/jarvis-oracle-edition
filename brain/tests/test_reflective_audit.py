"""Tests for Layer 9: Reflective Audit Engine."""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from epistemic.reflective_audit.engine import (
    ReflectiveAuditEngine,
    AuditFinding,
    AuditReport,
    SEVERITY_WEIGHT,
    MAX_FINDINGS_PER_REPORT,
)


def _fresh_engine() -> ReflectiveAuditEngine:
    """Create a fresh engine instance (not singleton) for test isolation."""
    engine = ReflectiveAuditEngine.__new__(ReflectiveAuditEngine)
    engine.__init__()
    return engine


# ---------------------------------------------------------------------------
# AuditFinding
# ---------------------------------------------------------------------------

class TestAuditFinding:
    def test_create_finding(self):
        f = AuditFinding(
            category="incorrect_learning",
            severity="warning",
            description="test finding",
        )
        assert f.category == "incorrect_learning"
        assert f.severity == "warning"
        assert f.description == "test finding"
        assert f.evidence == {}
        assert f.recommendation == ""
        assert f.related_ids == ()

    def test_finding_with_evidence(self):
        f = AuditFinding(
            category="source_trust",
            severity="critical",
            description="bad source",
            evidence={"score": 0.1},
            recommendation="fix it",
            related_ids=("id1", "id2"),
        )
        assert f.evidence == {"score": 0.1}
        assert f.related_ids == ("id1", "id2")

    def test_finding_is_frozen(self):
        f = AuditFinding(category="memory_hygiene", severity="info", description="ok")
        try:
            f.category = "other"  # type: ignore
            assert False, "Should be frozen"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------

class TestAuditReport:
    def test_empty_report_score_is_perfect(self):
        r = AuditReport(timestamp=time.time())
        assert r.compute_score() == 1.0

    def test_single_info_finding(self):
        r = AuditReport(timestamp=time.time(), findings=[
            AuditFinding("memory_hygiene", "info", "minor issue"),
        ])
        score = r.compute_score()
        assert 0.95 < score < 1.0

    def test_single_critical_finding(self):
        r = AuditReport(timestamp=time.time(), findings=[
            AuditFinding("autonomy_failure", "critical", "critical issue"),
        ])
        score = r.compute_score()
        assert score < 0.85

    def test_multiple_critical_findings_low_score(self):
        findings = [
            AuditFinding("autonomy_failure", "critical", f"issue {i}")
            for i in range(5)
        ]
        r = AuditReport(timestamp=time.time(), findings=findings)
        score = r.compute_score()
        assert score == 0.0

    def test_mixed_severity(self):
        r = AuditReport(timestamp=time.time(), findings=[
            AuditFinding("memory_hygiene", "info", "minor"),
            AuditFinding("source_trust", "warning", "moderate"),
            AuditFinding("autonomy_failure", "critical", "severe"),
        ])
        score = r.compute_score()
        assert 0.5 < score < 0.85


# ---------------------------------------------------------------------------
# Engine Core
# ---------------------------------------------------------------------------

class TestReflectiveAuditEngine:
    def test_initial_state(self):
        e = _fresh_engine()
        state = e.get_state()
        assert state["total_audits"] == 0
        assert state["total_findings"] == 0
        assert state["last_audit_ts"] == 0.0
        assert state["recent_reports"] == []

    def test_run_audit_produces_report(self):
        e = _fresh_engine()
        report = e.run_audit()
        assert isinstance(report, AuditReport)
        assert report.timestamp > 0
        assert report.categories_scanned > 0
        assert 0.0 <= report.score <= 1.0

    def test_state_updates_after_audit(self):
        e = _fresh_engine()
        e.run_audit()
        state = e.get_state()
        assert state["total_audits"] == 1
        assert state["last_audit_ts"] > 0
        assert len(state["recent_reports"]) == 1

    def test_multiple_audits_accumulate(self):
        e = _fresh_engine()
        for _ in range(3):
            e.run_audit()
        state = e.get_state()
        assert state["total_audits"] == 3
        assert len(state["recent_reports"]) == 3

    def test_latest_score(self):
        e = _fresh_engine()
        assert e.get_latest_score() is None
        report = e.run_audit()
        assert e.get_latest_score() == report.score

    def test_latest_report(self):
        e = _fresh_engine()
        assert e.get_latest_report() is None
        report = e.run_audit()
        assert e.get_latest_report() is report


# ---------------------------------------------------------------------------
# Trend Computation
# ---------------------------------------------------------------------------

class TestAuditTrend:
    def test_no_data_stable(self):
        e = _fresh_engine()
        trend = e._compute_trend()
        assert trend["direction"] == "stable"
        assert trend["delta"] == 0.0

    def test_single_report_stable(self):
        e = _fresh_engine()
        e.run_audit()
        trend = e._compute_trend()
        assert trend["direction"] == "stable"


# ---------------------------------------------------------------------------
# Severity Weights
# ---------------------------------------------------------------------------

class TestSeverityWeights:
    def test_ordering(self):
        assert SEVERITY_WEIGHT["info"] < SEVERITY_WEIGHT["warning"]
        assert SEVERITY_WEIGHT["warning"] < SEVERITY_WEIGHT["critical"]

    def test_ranges(self):
        for w in SEVERITY_WEIGHT.values():
            assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# Report Serialization (via get_state)
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_report_in_state_has_required_fields(self):
        e = _fresh_engine()
        e.run_audit()
        state = e.get_state()
        report = state["recent_reports"][0]
        assert "timestamp" in report
        assert "score" in report
        assert "finding_count" in report
        assert "duration_ms" in report
        assert "categories_scanned" in report
        assert "findings" in report


# ---------------------------------------------------------------------------
# Sacred Invariants (Layer 9)
# ---------------------------------------------------------------------------

class TestSacredInvariants:
    def test_audit_never_mutates_beliefs(self):
        """Layer 9 is read-only: audit findings do not modify beliefs."""
        e = _fresh_engine()
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            ce = ContradictionEngine.get_instance()
            if ce:
                before_debt = ce.contradiction_debt
                before_count = len(ce.belief_store.get_active_beliefs())
                e.run_audit()
                assert ce.contradiction_debt == before_debt
                assert len(ce.belief_store.get_active_beliefs()) == before_count
        except ImportError:
            pass  # no contradiction engine available in test env

    def test_audit_never_modifies_memory(self):
        """Layer 9 findings never trigger memory writes."""
        e = _fresh_engine()
        try:
            from memory.storage import MemoryStorage
            storage = MemoryStorage.get_instance()
            if storage:
                before = storage.get_stats().get("total", 0)
                e.run_audit()
                after = storage.get_stats().get("total", 0)
                assert after == before
        except ImportError:
            pass

    def test_audit_score_in_range(self):
        """Audit score is always [0.0, 1.0]."""
        e = _fresh_engine()
        for _ in range(5):
            report = e.run_audit()
            assert 0.0 <= report.score <= 1.0

    def test_finding_categories_valid(self):
        """All findings use valid category strings."""
        valid = {"incorrect_learning", "identity_breach", "source_trust",
                 "autonomy_failure", "skill_stagnation", "memory_hygiene",
                 "ingestion_health"}
        e = _fresh_engine()
        report = e.run_audit()
        for f in report.findings:
            assert f.category in valid

    def test_finding_severities_valid(self):
        """All findings use valid severity strings."""
        valid = {"info", "warning", "critical"}
        e = _fresh_engine()
        report = e.run_audit()
        for f in report.findings:
            assert f.severity in valid
