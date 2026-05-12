"""Regression tests for memory-neutral web-search behavior."""

from consciousness.reflection import ReflectionEngine
from memory.episodes import EpisodicMemory


def test_remove_last_user_turn_if_match_drops_empty_active_episode(tmp_path) -> None:
    episodes = EpisodicMemory(persist_path=str(tmp_path / "episodes.json"))
    episodes.add_user_turn("search web for jarvis updates", speaker="unknown")

    removed = episodes.remove_last_user_turn_if_match("search web for jarvis updates")
    assert removed is True
    assert episodes.get_active_episode() is None
    assert episodes.get_episode_count() == 0


def test_remove_last_user_turn_if_match_ignores_non_matching_text(tmp_path) -> None:
    episodes = EpisodicMemory(persist_path=str(tmp_path / "episodes.json"))
    episodes.add_user_turn("search web for robotics news", speaker="unknown")

    removed = episodes.remove_last_user_turn_if_match("different text")
    assert removed is False
    active = episodes.get_active_episode()
    assert active is not None
    assert active.turn_count() == 1


def test_reflection_engine_skips_web_search_responses() -> None:
    engine = ReflectionEngine()

    engine._on_conversation_response(text="Top live web results for 'x'...", tool="WEB_SEARCH")
    assert len(engine._pending_interactions) == 0

    engine._on_conversation_response(text="Normal answer", tool="NONE")
    assert len(engine._pending_interactions) == 1
