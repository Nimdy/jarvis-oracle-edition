from __future__ import annotations

from autonomy.policy_memory import AutonomyPolicyMemory, PolicyOutcome


def _new_memory(monkeypatch) -> AutonomyPolicyMemory:
    # Avoid writing to ~/.jarvis during tests.
    monkeypatch.setattr(AutonomyPolicyMemory, "_persist_append", lambda self, outcome: None)
    mem = AutonomyPolicyMemory()
    mem._outcomes.clear()
    mem._tag_index.clear()
    mem._tool_index.clear()
    mem._type_index.clear()
    monkeypatch.setattr(mem, "is_warmup", lambda: False)
    return mem


def _record(
    mem: AutonomyPolicyMemory,
    *,
    tags: tuple[str, ...],
    net_delta: float,
    worked: bool,
) -> None:
    mem.record_outcome(
        PolicyOutcome(
            intent_id="intent",
            intent_type="metric:test",
            tool_used="web",
            topic_tags=tags,
            question_summary="test",
            net_delta=net_delta,
            stable=True,
            worked=worked,
        )
    )


def test_get_avoid_patterns_includes_strict_low_win_cluster(monkeypatch) -> None:
    mem = _new_memory(monkeypatch)
    tags = ("topic", "strict")

    _record(mem, tags=tags, net_delta=-0.11, worked=False)
    _record(mem, tags=tags, net_delta=-0.08, worked=False)
    _record(mem, tags=tags, net_delta=-0.03, worked=False)

    avoid = mem.get_avoid_patterns()
    assert len(avoid) == 1
    assert avoid[0]["tags"] == ["topic", "strict"]
    assert avoid[0]["total"] == 3
    assert avoid[0]["win_rate"] == 0.0


def test_get_avoid_patterns_includes_regression_heavy_low_win_cluster(monkeypatch) -> None:
    mem = _new_memory(monkeypatch)
    tags = ("conversation_quality", "friction")

    # Mirrors observed real-world pattern: mixed outcomes, low win rate, negative mean.
    _record(mem, tags=tags, net_delta=-0.23, worked=False)
    _record(mem, tags=tags, net_delta=0.016, worked=False)  # near-neutral, below win threshold
    _record(mem, tags=tags, net_delta=-0.14, worked=False)
    _record(mem, tags=tags, net_delta=0.11, worked=True)

    avoid = mem.get_avoid_patterns()
    assert len(avoid) == 1
    assert avoid[0]["tags"] == ["conversation_quality", "friction"]
    assert avoid[0]["total"] == 4
    assert avoid[0]["win_rate"] == 0.25
    assert avoid[0]["loss_rate"] == 0.5
    assert avoid[0]["avg_delta"] < 0.0


def test_get_avoid_patterns_skips_low_win_without_regression_pressure(monkeypatch) -> None:
    mem = _new_memory(monkeypatch)
    tags = ("topic", "not_bad_enough")

    # Low win-rate alone should not trigger unless failure pattern is sustained.
    _record(mem, tags=tags, net_delta=-0.03, worked=False)   # one meaningful regression
    _record(mem, tags=tags, net_delta=0.015, worked=False)   # near-neutral
    _record(mem, tags=tags, net_delta=0.015, worked=False)   # near-neutral
    _record(mem, tags=tags, net_delta=0.40, worked=True)     # one strong success

    avoid = mem.get_avoid_patterns()
    assert avoid == []
