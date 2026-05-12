"""Tests for web-search query extraction normalization."""

import asyncio
import sys
import types

from tools.web_search_tool import extract_search_query, fetch_page_content_for_summary


def test_extract_search_query_strips_web_search_prefix() -> None:
    q = extract_search_query("web search, jarvis pi project")
    assert q == "jarvis pi project"


def test_extract_search_query_strips_duckduckgo_prefix() -> None:
    q = extract_search_query("DuckDuckGo search for current events please")
    assert q == "current events"


def test_fetch_page_content_for_summary_truncates(monkeypatch) -> None:
    fake_module = types.ModuleType("library.ingest")

    def _fake_fetch_url(url: str) -> tuple[str, str]:
        assert url == "https://example.com"
        return "x" * 200, ""

    fake_module._fetch_url = _fake_fetch_url
    monkeypatch.setitem(sys.modules, "library.ingest", fake_module)

    content, error = asyncio.run(
        fetch_page_content_for_summary("https://example.com", max_chars=80),
    )
    assert error == ""
    assert len(content) == 80


def test_fetch_page_content_for_summary_propagates_error(monkeypatch) -> None:
    fake_module = types.ModuleType("library.ingest")
    fake_module._fetch_url = lambda _url: ("", "HTTP 503: unavailable")
    monkeypatch.setitem(sys.modules, "library.ingest", fake_module)

    content, error = asyncio.run(fetch_page_content_for_summary("https://example.com"))
    assert content == ""
    assert error == "HTTP 503: unavailable"
