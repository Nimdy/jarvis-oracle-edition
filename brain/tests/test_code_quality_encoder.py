"""Tests for the CodeQualityEncoder hemisphere feature encoder."""

import pytest
from hemisphere.code_quality_encoder import (
    CodeQualityEncoder,
    FEATURE_DIM,
    _VERDICT_CLASSES,
)


class _MockFileDiff:
    def __init__(self, path, original="", new=""):
        self.path = path
        self.original_content = original
        self.new_content = new


class _MockPatch:
    def __init__(self, files=None, confidence=0.7, provider="coder_server", requires_approval=False):
        self.files = files or []
        self.confidence = confidence
        self.provider = provider
        self.requires_approval = requires_approval

    def check_capability_escalation(self):
        return []


class _MockReport:
    def __init__(self, **kwargs):
        self.lint_passed = kwargs.get("lint_passed", True)
        self.lint_executed = kwargs.get("lint_executed", True)
        self.all_tests_passed = kwargs.get("all_tests_passed", True)
        self.tests_executed = kwargs.get("tests_executed", True)
        self.sim_passed = kwargs.get("sim_passed", True)
        self.sim_executed = kwargs.get("sim_executed", True)
        self.sim_p95_before = kwargs.get("sim_p95_before", 10.0)
        self.sim_p95_after = kwargs.get("sim_p95_after", 12.0)
        self.diagnostics = kwargs.get("diagnostics", [])
        self.recommendation = kwargs.get("recommendation", "promote")
        self._silent_stubs = kwargs.get("silent_stubs", False)

    def has_silent_stubs(self):
        return self._silent_stubs


class _MockRequest:
    def __init__(self, req_type="bug_fix", priority=0.6, requires_approval=False):
        self.type = req_type
        self.priority = priority
        self.requires_approval = requires_approval


class _MockPlan:
    def __init__(self, risk=0.4, category="consciousness"):
        self.estimated_risk = risk
        self.write_category = category


class _MockRecord:
    def __init__(self, **kwargs):
        self.request = kwargs.get("request", _MockRequest())
        self.plan = kwargs.get("plan", _MockPlan())
        self.patch = kwargs.get("patch", _MockPatch(
            files=[_MockFileDiff("brain/consciousness/engine.py",
                                 "old\nline\n", "old\nline\nnew\n")],
        ))
        self.report = kwargs.get("report", _MockReport())
        self.iterations = kwargs.get("iterations", 2)
        self.upgrade_id = kwargs.get("upgrade_id", "upg_test123")


class TestCodeQualityEncoderEncode:
    def test_output_dimension(self):
        vec = CodeQualityEncoder.encode(_MockRecord())
        assert len(vec) == FEATURE_DIM

    def test_all_values_in_range(self):
        vec = CodeQualityEncoder.encode(_MockRecord())
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"dim {i} out of range: {v}"

    def test_request_type_one_hot(self):
        record = _MockRecord(request=_MockRequest(req_type="bug_fix"))
        vec = CodeQualityEncoder.encode(record)
        assert vec[3] == 1.0  # bug_fix is index 3
        assert vec[0] == 0.0  # performance_optimization is index 0
        assert vec[1] == 0.0

    def test_priority(self):
        record = _MockRecord(request=_MockRequest(priority=0.8))
        vec = CodeQualityEncoder.encode(record)
        assert vec[5] == pytest.approx(0.8)

    def test_risk(self):
        record = _MockRecord(plan=_MockPlan(risk=0.7))
        vec = CodeQualityEncoder.encode(record)
        assert vec[6] == pytest.approx(0.7)

    def test_patch_confidence(self):
        record = _MockRecord(patch=_MockPatch(confidence=0.9))
        vec = CodeQualityEncoder.encode(record)
        assert vec[11] == pytest.approx(0.9)

    def test_iterations_norm(self):
        record = _MockRecord(iterations=3)
        vec = CodeQualityEncoder.encode(record)
        assert vec[13] == pytest.approx(2 / 4)  # (3-1)/4

    def test_sandbox_all_pass(self):
        record = _MockRecord(report=_MockReport())
        vec = CodeQualityEncoder.encode(record)
        assert vec[16] == 1.0  # lint_passed
        assert vec[17] == 1.0  # lint_executed
        assert vec[18] == 1.0  # tests_passed
        assert vec[20] == 1.0  # sim_passed
        assert vec[25] == 1.0  # recommendation=promote

    def test_sandbox_failed(self):
        report = _MockReport(
            lint_passed=False, lint_executed=True,
            all_tests_passed=False, tests_executed=False,
            sim_passed=False, sim_executed=False,
            recommendation="rollback",
        )
        record = _MockRecord(report=report)
        vec = CodeQualityEncoder.encode(record)
        assert vec[16] == 0.0  # lint_passed
        assert vec[17] == 1.0  # lint_executed
        assert vec[18] == 0.0  # tests_passed
        assert vec[19] == 0.0  # tests_executed
        assert vec[25] == 0.0  # recommendation=rollback

    def test_silent_stubs_detected(self):
        report = _MockReport(silent_stubs=True)
        record = _MockRecord(report=report)
        vec = CodeQualityEncoder.encode(record)
        assert vec[24] == 1.0

    def test_no_patch_graceful(self):
        record = _MockRecord(patch=None)
        record.patch = None
        vec = CodeQualityEncoder.encode(record)
        assert len(vec) == FEATURE_DIM
        for v in vec:
            assert 0.0 <= v <= 1.0

    def test_no_report_graceful(self):
        record = _MockRecord(report=None)
        record.report = None
        vec = CodeQualityEncoder.encode(record)
        assert len(vec) == FEATURE_DIM
        assert vec[16] == 0.0  # lint_passed defaults to 0
        assert vec[25] == 0.0  # recommendation defaults to 0


class TestCodeQualityEncoderDict:
    def test_from_proposal_dict(self):
        d = {
            "what": {"type": "performance_optimization", "description": "test"},
            "why": {"priority": 0.7, "evidence": [], "evidence_detail": {}},
            "where": {"target_module": "test.py", "files_modified": ["a.py", "b.py"]},
            "who": {"provider": "coder_server"},
            "iterations": 2,
            "sandbox": {
                "lint_passed": True, "lint_executed": True,
                "tests_passed": True, "tests_executed": True,
                "sim_passed": False, "sim_executed": True,
                "sim_p95_after": 15.0,
                "diagnostics_count": 3,
                "has_silent_stubs": False,
                "recommendation": "manual_review",
            },
        }
        vec = CodeQualityEncoder.encode(d)
        assert len(vec) == FEATURE_DIM
        assert vec[0] == 1.0  # performance_optimization
        assert vec[5] == pytest.approx(0.7)
        assert vec[8] == pytest.approx(2 / 3)
        assert vec[16] == 1.0
        assert vec[20] == 0.0  # sim_passed=False
        assert vec[25] == 0.5  # manual_review

    def test_empty_dict(self):
        vec = CodeQualityEncoder.encode({})
        assert len(vec) == FEATURE_DIM
        for v in vec:
            assert 0.0 <= v <= 1.0


class TestCodeQualityEncoderVerdict:
    @pytest.mark.parametrize("verdict,expected_idx", [
        ("verified_improved", 0),
        ("verified_stable", 1),
        ("verified_regressed", 2),
        ("rolled_back", 3),
    ])
    def test_verdict_one_hot(self, verdict, expected_idx):
        label = CodeQualityEncoder.encode_verdict_label(verdict)
        assert len(label) == len(_VERDICT_CLASSES)
        assert label[expected_idx] == 1.0
        assert sum(label) == 1.0

    def test_unknown_verdict_defaults_to_stable(self):
        label = CodeQualityEncoder.encode_verdict_label("pending_verification")
        assert label[1] == 1.0  # verified_stable
        assert sum(label) == 1.0

    def test_none_verdict(self):
        label = CodeQualityEncoder.encode_verdict_label("none")
        assert label[1] == 1.0


class TestCodeQualityEncoderModuleHistory:
    """Track 6: Per-module patch history dimensions (28-34)."""

    def _sample_history(self) -> dict:
        return {
            "total_patches": 5,
            "verdict_counts": {"improved": 3, "stable": 1, "regressed": 1, "rolled_back": 0},
            "last_patch_age_s": 3600.0,
            "avg_iterations": 2.5,
            "recidivism": True,
            "has_history": True,
        }

    def test_output_is_35_dims(self):
        vec = CodeQualityEncoder.encode(_MockRecord())
        assert len(vec) == FEATURE_DIM
        assert FEATURE_DIM == 35

    def test_history_dims_populated(self):
        import math
        hist = self._sample_history()
        vec = CodeQualityEncoder.encode(_MockRecord(), module_history=hist)
        assert vec[28] == pytest.approx(5 / 10)       # total_patches
        assert vec[29] == pytest.approx(4 / 5)         # success_rate (3+1)/5
        assert vec[30] == pytest.approx(1 / 5)         # regression_rate (1+0)/5
        assert vec[31] == pytest.approx(math.exp(-3600 / 86400))  # recency
        assert vec[32] == 1.0                           # recidivism
        assert vec[33] == pytest.approx(2.5 / 5)       # avg_iterations
        assert vec[34] == 1.0                           # has_patch_history

    def test_no_history_zeros(self):
        vec = CodeQualityEncoder.encode(_MockRecord(), module_history=None)
        for i in range(28, 35):
            assert vec[i] == 0.0

    def test_empty_history_dict(self):
        vec = CodeQualityEncoder.encode(_MockRecord(), module_history={})
        for i in range(28, 35):
            assert vec[i] == 0.0

    def test_has_history_false(self):
        hist = {"has_history": False, "total_patches": 0}
        vec = CodeQualityEncoder.encode(_MockRecord(), module_history=hist)
        assert vec[34] == 0.0

    def test_history_clamped(self):
        hist = self._sample_history()
        hist["total_patches"] = 100
        hist["avg_iterations"] = 50
        vec = CodeQualityEncoder.encode(_MockRecord(), module_history=hist)
        assert vec[28] == 1.0
        assert vec[33] == 1.0

    def test_no_recidivism(self):
        hist = self._sample_history()
        hist["recidivism"] = False
        vec = CodeQualityEncoder.encode(_MockRecord(), module_history=hist)
        assert vec[32] == 0.0

    def test_history_from_dict(self):
        d = {
            "what": {"type": "bug_fix"},
            "why": {"priority": 0.5},
            "where": {"target_module": "test.py", "files_modified": []},
            "who": {"provider": "ollama"},
            "sandbox": {},
        }
        hist = self._sample_history()
        vec = CodeQualityEncoder.encode(d, module_history=hist)
        assert len(vec) == FEATURE_DIM
        assert vec[34] == 1.0  # has_patch_history

    def test_all_new_dims_in_range(self):
        hist = self._sample_history()
        vec = CodeQualityEncoder.encode(_MockRecord(), module_history=hist)
        for i in range(28, 35):
            assert 0.0 <= vec[i] <= 1.0, f"dim {i} out of range: {vec[i]}"
