"""V2 capability page consumes the same skill audit packet as the old Learning tab."""
from pathlib import Path

_STATIC = Path(__file__).resolve().parent.parent / "dashboard" / "static"
_CAP = _STATIC / "v2" / "capability.html"
_DRILL = _STATIC / "v2" / "v2-drill.js"
_SHARED = _STATIC / "v2" / "shared.js"
_CSS = _STATIC / "v2" / "v2.css"


def test_v2_capability_loads_skill_audit_module() -> None:
    html = _CAP.read_text(encoding="utf-8")
    assert "/static/v2/v2-drill.js" in html
    assert 'id="skill-stats"' in html
    assert 'id="skill-gate"' in html
    assert 'id="skill-jobs"' in html
    assert "openSkillAudit" in html
    assert "SKILL_PHASES=['assess','research','acquire','integrate','collect','train','verify','register']" in html.replace(" ", "") or (
        "'research'" in html and "'acquire'" in html and "'integrate'" in html
    )


def test_v2_drill_skill_detail_covers_old_modal_sections() -> None:
    js = _DRILL.read_text(encoding="utf-8")
    assert "function skillDetail" in js
    for needle in (
        "Request context",
        "Contract / resolver",
        "Proposed plugin design",
        "Evidence classes",
        "Operational handoff",
        "Acquisition proof chain",
        "Why this is not verified yet",
        "Evidence checks",
        "Artifacts",
        "Timeline",
        "Integrity notes",
        "Operator review",
        "sk-tab",
        "BASELINE",
        "cannot be deleted",
        "phase_glass",
        "Delete learned skill",
        "speech_output",
        "slice(-40)",
        "MATRIX",
        "/handoff/approve",
        "/handoff/reject",
        "/handoff/retry",
        "v2d-skill-open-acq",
        "openAcquisitionReview",
    ):
        assert needle in js, needle
    assert "skillDetail: skillDetail" in js


def test_v2_modal_supports_wide_audit() -> None:
    shared = _SHARED.read_text(encoding="utf-8")
    css = _CSS.read_text(encoding="utf-8")
    assert "opts.wide" in shared
    assert ".v2-modal.wide" in css


def test_v2_capability_plan_review_desk_ports_old_learning_tab() -> None:
    """Human validation desk: reject/revise, completeness lock, tests, view code."""
    html = _CAP.read_text(encoding="utf-8")
    for needle in (
        "Reject &amp; Revise",
        "approved_as_is",
        "reason_category",
        "incomplete_design",
        "missing_tests",
        "security_concern",
        "suggested_changes",
        "prior_rejection",
        "View full code",
        "Plan is incomplete",
        "Approve deploy",
        "Deny deploy",
        "pending_approvals",
        "stalled_jobs",
        "input validation",
        "#acquisition:",
        "verdict:'rejected'",
        "Approve plan",
        "openAcquisitionReview",
        "_mountAcqThenOpen",
    ):
        assert needle in html, needle
    assert html.count("approved_as_is") >= 2
    assert "{verdict:'approved'}" not in html
