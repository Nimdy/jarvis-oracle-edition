"""Phase 5 intervention extraction regression tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from autonomy.orchestrator import AutonomyOrchestrator
from autonomy.research_intent import ResearchFinding, ResearchIntent, ResearchResult


class _FakeIntegrator:
    def __init__(self, source_ids: list[str]) -> None:
        self._source_ids = source_ids

    def get_source_ids_for_intent(self, _intent_id: str, limit: int = 5) -> list[str]:
        return self._source_ids[:limit]


def _mk_orch(source_ids: list[str]) -> AutonomyOrchestrator:
    orch = AutonomyOrchestrator.__new__(AutonomyOrchestrator)
    orch._integrator = _FakeIntegrator(source_ids)
    return orch


def test_extract_interventions_generates_routing_and_prompt_candidates():
    orch = _mk_orch(["src_a", "src_b"])
    intent = ResearchIntent(
        question="How can we reduce conversation friction from route misses?",
        source_event="metric:friction_rate",
    )
    result = ResearchResult(
        intent_id=intent.id,
        tool_used="academic",
        success=True,
        findings=[
            ResearchFinding(
                content=(
                    "A routing classifier with explicit follow-up disambiguation improves "
                    "intent accuracy. Prompt clarity and concise response format reduce "
                    "verbosity complaints."
                ),
                provenance="paper",
                confidence=0.9,
            )
        ],
    )

    interventions = orch._extract_interventions(intent, result, memories_created=2)
    change_types = {iv.change_type for iv in interventions}
    assert "routing_rule" in change_types
    assert "prompt_frame" in change_types
    assert len(interventions) <= 3
    for iv in interventions:
        assert iv.source_ids == ["src_a", "src_b"]


def test_extract_interventions_uses_threshold_and_metric_direction():
    orch = _mk_orch(["src_mem"])
    intent = ResearchIntent(
        question="Improve memory retrieval hit rate",
        source_event="metric:retrieval",
    )
    result = ResearchResult(
        intent_id=intent.id,
        tool_used="academic",
        success=True,
        findings=[
            ResearchFinding(
                content="Adjust threshold tuning and memory weight decay to improve retrieval ranker outcomes.",
                provenance="paper",
                confidence=0.8,
            )
        ],
    )

    interventions = orch._extract_interventions(intent, result, memories_created=1)
    by_type = {iv.change_type: iv for iv in interventions}
    assert "threshold_change" in by_type
    assert "memory_weighting_rule" in by_type
    assert by_type["memory_weighting_rule"].expected_direction == "up"


def test_extract_interventions_returns_empty_for_non_actionable_findings():
    orch = _mk_orch(["src_x"])
    intent = ResearchIntent(
        question="Reflect on broad systems meaning",
        source_event="existential:curiosity",
    )
    result = ResearchResult(
        intent_id=intent.id,
        tool_used="web",
        success=True,
        findings=[
            ResearchFinding(
                content="This article discusses philosophy in broad terms without implementation guidance.",
                provenance="web",
                confidence=0.6,
            )
        ],
    )

    interventions = orch._extract_interventions(intent, result, memories_created=1)
    assert interventions == []


def test_extract_interventions_fallback_for_metric_friction_intent():
    orch = _mk_orch(["src_friction"])
    intent = ResearchIntent(
        question="What conversation patterns cause repeated user corrections and rephrases?",
        source_event="metric:friction_rate",
        source_hint="codebase",
    )
    result = ResearchResult(
        intent_id=intent.id,
        tool_used="codebase",
        success=True,
        findings=[
            ResearchFinding(
                content="Module context found in codebase budgeted scan.",
                provenance="codebase_budgeted_context",
                confidence=0.7,
            )
        ],
    )

    interventions = orch._extract_interventions(intent, result, memories_created=1)
    assert len(interventions) >= 2
    by_type = {iv.change_type: iv for iv in interventions}
    assert "routing_rule" in by_type
    assert "prompt_frame" in by_type
    assert by_type["routing_rule"].expected_metric == "friction_rate"
    assert by_type["routing_rule"].expected_direction == "down"
    assert by_type["routing_rule"].source_ids == ["src_friction"]
