from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_memory_storage_default_capacity():
    from memory.storage import MemoryStorage

    storage = MemoryStorage()
    assert storage._max_capacity == 2000


def test_memory_tool_semantic_search_uses_top_k(monkeypatch):
    from tools import memory_tool
    import memory.search as memory_search

    captured: dict[str, int] = {}

    def fake_semantic_search(query: str, top_k: int = 0, **_: object):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(memory_search, "semantic_search", fake_semantic_search)
    memory_tool._semantic_search("remember this", limit=6)

    assert captured["top_k"] == 6


def test_tool_router_records_voice_intent_teacher_signal(monkeypatch):
    from reasoning.tool_router import ToolRouter, ToolType
    import hemisphere.distillation as distillation

    calls: list[dict[str, object]] = []

    class DummyCollector:
        def record(self, teacher, signal_type, data, metadata=None, origin="system", fidelity=1.0):
            calls.append(
                {
                    "teacher": teacher,
                    "signal_type": signal_type,
                    "data": data,
                    "metadata": metadata or {},
                    "origin": origin,
                    "fidelity": fidelity,
                }
            )

    monkeypatch.setattr(distillation, "distillation_collector", DummyCollector())

    result = ToolRouter().route("What time is it?")

    assert result.tool == ToolType.TIME
    assert calls, "tool routing should emit a teacher signal for voice-intent distillation"
    latest = calls[-1]
    assert latest["teacher"] == "tool_router"
    assert latest["signal_type"] == "route_label"
    assert len(latest["data"]) == 8
    assert sum(latest["data"]) == 1.0
    assert latest["metadata"]["tool"] == ToolType.TIME.value


def test_prepare_distillation_tensors_accepts_tool_router_teacher():
    from hemisphere.data_feed import prepare_distillation_tensors
    from hemisphere.distillation import TeacherSignal
    from hemisphere.types import DISTILLATION_CONFIGS

    now = time.time()
    config = DISTILLATION_CONFIGS["voice_intent"]

    class FakeCollector:
        def get_training_batch(self, teacher: str, limit: int = 200, min_fidelity: float = 0.0):
            feature_source = getattr(config, "feature_source", "audio_features")
            if teacher == feature_source:
                dim = config.input_dim
                return [
                    TeacherSignal(
                        teacher=feature_source,
                        signal_type="features",
                        data=[0.1] * dim,
                        timestamp=now + i,
                        metadata={},
                        fidelity=1.0,
                    )
                    for i in range(config.min_samples)
                ]
            if teacher == "tool_router":
                return [
                    TeacherSignal(
                        teacher="tool_router",
                        signal_type="route_label",
                        data=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        timestamp=now + i,
                        metadata={"tool": "TIME"},
                        fidelity=0.9,
                    )
                    for i in range(config.min_samples)
                ]
            return []

    tensors = prepare_distillation_tensors("voice_intent", FakeCollector(), config)
    assert tensors is not None

    features, labels, weights = tensors
    assert features.shape[0] >= config.min_samples
    assert features.shape[1] == config.input_dim
    assert labels.shape[1] == config.output_dim
    assert weights.shape[0] == features.shape[0]


def test_web_user_formatter_is_grounded():
    from tools.web_search_tool import SearchResult, format_results_for_user

    text = format_results_for_user(
        [
            SearchResult(
                title="Jarvis roadmap",
                url="https://example.com/jarvis",
                snippet="A concrete live roadmap for Jarvis and its memory system.",
            )
        ],
        query="jarvis roadmap",
    )

    assert "Live web results for 'jarvis roadmap':" in text
    assert "Jarvis roadmap" in text
    assert "https://example.com/jarvis" in text
    assert "Do NOT say" not in text


def test_academic_user_formatter_is_grounded():
    from tools.academic_search_tool import AcademicResult, format_academic_results_for_user

    text = format_academic_results_for_user(
        [
            AcademicResult(
                title="Distilling Intent Models",
                abstract="A study of teacher-student intent transfer for compact assistants.",
                venue="Journal of Applied AI",
                year=2025,
                doi="10.1000/test",
            )
        ],
        query="intent distillation",
    )

    assert "Scholarly results for 'intent distillation':" in text
    assert "Distilling Intent Models" in text
    assert "Journal of Applied AI (2025)" in text
    assert "DOI 10.1000/test" in text
