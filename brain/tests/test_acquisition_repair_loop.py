"""Tests for the acquisition codegen repair loop + web_scraping contract fixture.

The implementation lane runs the linked skill contract smoke against each
candidate bundle and feeds expected-vs-actual mismatches back to the coder
(think→code→validate), up to _MAX_CODEGEN_REPAIR_ATTEMPTS rounds, instead of
one-shot-and-fail. These tests pin the repair-decision helpers deterministically
(no network): the contract smoke runs the candidate handler in-process, so a
handler that inspects the input URL can satisfy both fixtures without a real
fetch.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from acquisition.orchestrator import AcquisitionOrchestrator
from acquisition.job import PluginCodeBundle


def _orch():
    # Methods under test use no instance state; bypass the heavy constructor.
    return AcquisitionOrchestrator.__new__(AcquisitionOrchestrator)


def _job(skill_id):
    return types.SimpleNamespace(requested_by={"skill_id": skill_id}, artifacts=[], skill_id=skill_id)


def _bundle(handler_src):
    return PluginCodeBundle(acquisition_id="t", code_files={"handler.py": handler_src})


# A handler that returns the contract-expected output by inspecting the input
# URL — passes both web_scraping fixtures with no real network call.
_GOOD = (
    "def run(args):\n"
    "    url = args.get('text', '')\n"
    "    if 'example.com' in url:\n"
    "        return {'title': 'Example Domain', 'status_code': 200, 'url': url}\n"
    "    return {'title': '', 'status_code': 500, 'url': url}\n"
)

_WRONG = "def run(args):\n    return {'title': 'WRONG', 'status_code': 999}\n"


class TestContractSmokeForRepair:
    def test_matching_plugin_passes(self):
        ok, mismatches = _orch()._run_contract_smoke_for_repair(_job("web_scraping_v1"), _bundle(_GOOD))
        assert ok is True
        assert mismatches == []

    def test_wrong_plugin_reports_mismatches(self):
        ok, mismatches = _orch()._run_contract_smoke_for_repair(_job("web_scraping_v1"), _bundle(_WRONG))
        assert ok is False
        assert len(mismatches) >= 1
        first = mismatches[0]
        assert first.get("expected", {}).get("title") == "Example Domain"
        assert first.get("actual", {}).get("title") == "WRONG"

    def test_no_contract_skips(self):
        # No registered contract for the skill -> never block the build.
        ok, mismatches = _orch()._run_contract_smoke_for_repair(_job("nonexistent_skill_xyz"), _bundle(_GOOD))
        assert ok is True
        assert mismatches == []

    def test_no_skill_id_skips(self):
        job = types.SimpleNamespace(requested_by={}, artifacts=[], skill_id="")
        ok, mismatches = _orch()._run_contract_smoke_for_repair(job, _bundle(_GOOD))
        assert ok is True
        assert mismatches == []


class TestRepairFeedbackFormatting:
    def test_contract_smoke_feedback_carries_expected_and_actual(self):
        fb = _orch()._format_repair_feedback({
            "stage": "contract_smoke",
            "errors": [{
                "fixture": "scrape_example_domain_title",
                "expected": {"title": "Example Domain", "status_code": 200},
                "actual": {"title": "WRONG", "status_code": 999},
            }],
        })
        assert "FAILED" in fb
        assert "Example Domain" in fb  # expected surfaced
        assert "WRONG" in fb           # actual surfaced
        assert "scrape_example_domain_title" in fb

    def test_code_validation_feedback_lists_errors(self):
        fb = _orch()._format_repair_feedback({
            "stage": "code_validation",
            "errors": ["SyntaxError: unexpected EOF", "denied pattern: subprocess"],
        })
        assert "SyntaxError: unexpected EOF" in fb
        assert "subprocess" in fb


class TestWebScrapingContractFixture:
    def test_contract_now_has_fixtures(self):
        from skills.execution_contracts import get_contract
        c = get_contract("web_scraping_v1")
        assert c is not None
        assert len(c.smoke_fixtures) >= 1
        names = {f.name for f in c.smoke_fixtures}
        assert "scrape_example_domain_title" in names
