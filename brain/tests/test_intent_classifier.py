"""Tests for acquisition/classifier.py — intent classification and lane routing.

The classifier is the entry point for the acquisition pipeline. It determines
which outcome class and lane set a user request maps to. Misclassification
means the wrong lanes activate (e.g., plugin_creation instead of knowledge_only),
which wastes resources or skips safety gates.

Covers:
  - All 6 pattern banks (knowledge, plugin, core_upgrade, hardware, skill, specialist)
  - _LANE_MAP completeness (all 7 outcome classes have lanes)
  - _RISK_MAP completeness and value ranges
  - Empty input handling
  - Default-to-knowledge fallback
  - Confidence thresholds
  - ClassificationResult dataclass
  - SkillResolver fallback (mocked)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock

from acquisition.classifier import (
    IntentClassifier,
    ClassificationResult,
    _LANE_MAP,
    _RISK_MAP,
    _KNOWLEDGE_PATTERNS,
    _PLUGIN_PATTERNS,
    _CORE_UPGRADE_PATTERNS,
    _HARDWARE_PATTERNS,
    _SKILL_PATTERNS,
    _SPECIALIST_PATTERNS,
)


# ---------------------------------------------------------------------------
# Lane and risk map completeness
# ---------------------------------------------------------------------------

class TestMaps:
    def test_lane_map_has_all_7_outcome_classes(self):
        expected = {
            "knowledge_only", "skill_creation", "plugin_creation",
            "core_upgrade", "specialist_nn", "hardware_integration", "mixed",
        }
        assert set(_LANE_MAP.keys()) == expected

    def test_risk_map_has_all_7_outcome_classes(self):
        assert set(_RISK_MAP.keys()) == set(_LANE_MAP.keys())

    def test_risk_tiers_in_range(self):
        for cls, tier in _RISK_MAP.items():
            assert 0 <= tier <= 3, f"{cls} has out-of-range risk tier {tier}"

    def test_knowledge_has_lowest_risk(self):
        assert _RISK_MAP["knowledge_only"] == 0

    def test_hardware_has_highest_risk(self):
        assert _RISK_MAP["hardware_integration"] == 3

    def test_all_lanes_start_with_evidence_grounding(self):
        for cls, lanes in _LANE_MAP.items():
            assert lanes[0] == "evidence_grounding", f"{cls} doesn't start with evidence_grounding"

    def test_all_lanes_end_with_truth(self):
        for cls, lanes in _LANE_MAP.items():
            assert lanes[-1] == "truth", f"{cls} doesn't end with truth"

    def test_plugin_creation_has_plan_review(self):
        assert "plan_review" in _LANE_MAP["plugin_creation"]

    def test_core_upgrade_has_self_improve(self):
        assert "self_improve" in _LANE_MAP["core_upgrade"]

    def test_specialist_nn_has_matrix_specialist(self):
        assert "matrix_specialist" in _LANE_MAP["specialist_nn"]

    def test_knowledge_only_is_fast_path(self):
        assert len(_LANE_MAP["knowledge_only"]) == 2


# ---------------------------------------------------------------------------
# Pattern bank matching
# ---------------------------------------------------------------------------

class TestPatternBanks:
    def _classify(self, text):
        classifier = IntentClassifier()
        with patch.dict("sys.modules", {"skills.resolver": MagicMock(SkillResolver=MagicMock(return_value=MagicMock(resolve=MagicMock(return_value=None))))}):
            return classifier.classify(text)

    def test_knowledge_pattern_what_is(self):
        r = self._classify("what does quantum computing mean")
        assert r.outcome_class == "knowledge_only"

    def test_knowledge_pattern_how_does(self):
        r = self._classify("how does the memory system work")
        assert r.outcome_class == "knowledge_only"

    def test_knowledge_pattern_learn_about(self):
        r = self._classify("learn about neural networks")
        assert r.outcome_class == "knowledge_only"

    def test_plugin_pattern_build_tool(self):
        r = self._classify("build a web scraper tool")
        assert r.outcome_class == "plugin_creation"

    def test_plugin_pattern_create_service(self):
        r = self._classify("create a data connector service")
        assert r.outcome_class == "plugin_creation"

    def test_plugin_pattern_scrape(self):
        r = self._classify("scrape data from this web page")
        assert r.outcome_class == "plugin_creation"

    def test_core_upgrade_improve_yourself(self):
        r = self._classify("improve yourself in memory management")
        assert r.outcome_class == "core_upgrade"

    def test_core_upgrade_self_improve(self):
        r = self._classify("self-improve your reasoning")
        assert r.outcome_class == "core_upgrade"

    def test_hardware_gpio(self):
        r = self._classify("control the GPIO pins on the Pi5")
        assert r.outcome_class == "hardware_integration"

    def test_hardware_motor(self):
        r = self._classify("connect a servo motor")
        assert r.outcome_class == "hardware_integration"

    def test_skill_learn_to(self):
        r = self._classify("learn how to sing a song")
        assert r.outcome_class == "skill_creation"

    def test_specialist_train_model(self):
        r = self._classify("train a specialist classifier for emotions")
        assert r.outcome_class == "specialist_nn"

    def test_specialist_matrix_protocol(self):
        r = self._classify("use the matrix protocol for a new model")
        assert r.outcome_class == "specialist_nn"

    def test_pattern_count(self):
        assert len(_KNOWLEDGE_PATTERNS) == 2
        assert len(_PLUGIN_PATTERNS) == 2
        assert len(_CORE_UPGRADE_PATTERNS) == 2
        assert len(_HARDWARE_PATTERNS) == 2
        assert len(_SKILL_PATTERNS) == 2
        assert len(_SPECIALIST_PATTERNS) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input_defaults_to_knowledge(self):
        classifier = IntentClassifier()
        r = classifier.classify("")
        assert r.outcome_class == "knowledge_only"
        assert r.confidence == 0.0

    def test_whitespace_only_defaults_to_knowledge(self):
        classifier = IntentClassifier()
        r = classifier.classify("   ")
        assert r.outcome_class == "knowledge_only"

    def test_no_match_defaults_to_knowledge(self):
        classifier = IntentClassifier()
        with patch.dict("sys.modules", {"skills.resolver": MagicMock(SkillResolver=MagicMock(return_value=MagicMock(resolve=MagicMock(return_value=None))))}):
            r = classifier.classify("asdfghjkl random noise")
        assert r.outcome_class == "knowledge_only"
        assert r.confidence <= 0.5

    def test_classification_result_has_lanes(self):
        classifier = IntentClassifier()
        with patch.dict("sys.modules", {"skills.resolver": MagicMock(SkillResolver=MagicMock(return_value=MagicMock(resolve=MagicMock(return_value=None))))}):
            r = classifier.classify("build a web scraper tool")
        assert isinstance(r.required_lanes, list)
        assert len(r.required_lanes) > 0

    def test_classification_result_has_risk_tier(self):
        classifier = IntentClassifier()
        r = classifier.classify("")
        assert isinstance(r.risk_tier, int)
        assert 0 <= r.risk_tier <= 3


# ---------------------------------------------------------------------------
# ClassificationResult dataclass
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_creation(self):
        r = ClassificationResult(
            outcome_class="plugin_creation",
            confidence=0.8,
            required_lanes=["evidence_grounding", "truth"],
            risk_tier=2,
            reasoning="test",
        )
        assert r.outcome_class == "plugin_creation"
        assert r.confidence == 0.8
        assert r.risk_tier == 2

    def test_timestamp_set(self):
        r = ClassificationResult(
            outcome_class="knowledge_only",
            confidence=0.5,
            required_lanes=[],
            risk_tier=0,
        )
        assert r.classified_at > 0


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_shape(self):
        classifier = IntentClassifier()
        status = classifier.get_status()
        assert status["available"] is True
        assert "pattern_banks" in status
        banks = status["pattern_banks"]
        assert "knowledge" in banks
        assert "plugin" in banks
        assert "core_upgrade" in banks
        assert "hardware" in banks
        assert "skill" in banks
        assert "specialist" in banks
