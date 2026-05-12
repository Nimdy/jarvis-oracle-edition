"""Regression tests for search-route capability-denial guardrails.

These tests ensure tool-backed search routes cannot emit contradictory
"no web/research access" claims after the route already executed.
"""

from reasoning.search_route_guard import (
    contains_search_access_denial,
    guard_search_tool_reply,
)


def test_detects_explicit_internet_access_denial() -> None:
    text = "I don't have access to the internet, so I can't browse live results."
    assert contains_search_access_denial(text) is True


def test_detects_local_only_capability_denial() -> None:
    text = "Everything runs locally only, so I cannot perform live web search."
    assert contains_search_access_denial(text) is True


def test_ignores_valid_tool_grounded_search_reply() -> None:
    text = "Live web results: 1) NOAA forecast says rain tomorrow."
    assert contains_search_access_denial(text) is False


def test_guard_replaces_contradictory_reply_with_fallback() -> None:
    guarded, replaced = guard_search_tool_reply(
        "I can't access the web from here.",
        "Live web results for 'weather': 1. NOAA ...",
        lane="realtime",
        search_query="weather tomorrow",
    )
    assert replaced is True
    assert guarded.startswith("Live web results")


def test_guard_keeps_non_contradictory_reply() -> None:
    original = "Scholarly results for 'sleep quality': 1. Paper A ..."
    guarded, replaced = guard_search_tool_reply(
        original,
        "fallback",
        lane="scholarly",
        search_query="sleep quality",
    )
    assert replaced is False
    assert guarded == original
