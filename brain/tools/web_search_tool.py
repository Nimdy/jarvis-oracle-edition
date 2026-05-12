"""Web Search Tool -- DuckDuckGo search with fencing and caching.

Uses the ``ddgs`` metasearch library with the DuckDuckGo backend explicitly.

Fencing rules:
  - Results can inform plan rationale and thinking LLM analysis
  - Raw web code NEVER enters patches without full lint + AST + test validation
  - All results cached with timestamps for reproducibility
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_FILE = Path("~/.jarvis/web_search_cache.json").expanduser()
CACHE_TTL_S = 3600.0       # 1 hour cache TTL
MAX_RESULTS = 5
SEARCH_BACKEND = "duckduckgo"
MAX_CACHE_ENTRIES = 200
MAX_PAGE_SUMMARY_CHARS = 12_000


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class SearchCache:
    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._load()

    def get(self, query: str) -> list[SearchResult] | None:
        key = self._key(query)
        entry = self._cache.get(key)
        if entry and time.time() - entry["timestamp"] < CACHE_TTL_S:
            return [SearchResult(**r) for r in entry["results"]]
        return None

    def put(self, query: str, results: list[SearchResult]) -> None:
        key = self._key(query)
        self._cache[key] = {
            "query": query,
            "timestamp": time.time(),
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet, "timestamp": r.timestamp}
                for r in results
            ],
        }
        # Trim cache
        if len(self._cache) > MAX_CACHE_ENTRIES:
            oldest = sorted(self._cache.items(), key=lambda x: x[1]["timestamp"])
            for k, _ in oldest[:len(self._cache) - MAX_CACHE_ENTRIES]:
                del self._cache[k]
        self._save()

    def _key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:16]

    def _save(self) -> None:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load(self) -> None:
        if CACHE_FILE.exists():
            try:
                self._cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}


_cache = SearchCache()


# ---------------------------------------------------------------------------
# Query extraction
# ---------------------------------------------------------------------------

_STRIP_PREFIXES = re.compile(
    r"^(?:hey\s+jarvis[,.]?\s*)?(?:can you\s+|could you\s+|please\s+)?"
    r"(?:(?:search\s+(?:the\s+)?(?:web|internet|online)\s+(?:for\s+)?|"
    r"web\s+search\s*(?:,|for\s+)?|"
    r"look\s+up\s+|find\s+(?:online\s+|on\s+the\s+web\s+)?|"
    r"(?:duck\s*duck\s*go|duckduckgo)\s*(?:search\s*)?(?:for\s+)?|"
    r"google\s+|search\s+for\s+|"
    r"what\s+is\s+the\s+latest\s+(?:on\s+|about\s+)?)"
    r"(?:and\s+(?:also\s+)?(?:look\s+up|find|search\s+for)\s+)?)",
    re.IGNORECASE,
)

_STRIP_SUFFIXES = re.compile(
    r"\s*(?:(?:for me\s*)?(?:please)?|and\s+see\s+(?:if|what).*|"
    r"and\s+(?:tell|let)\s+me.*|and\s+see\s+.*|"
    r"and\s+report\s+back.*|and\s+show\s+me.*)$",
    re.IGNORECASE,
)


def extract_search_query(user_text: str) -> str:
    """Strip conversational preamble/suffix to get a clean search query."""
    q = _STRIP_PREFIXES.sub("", user_text.strip())
    q = _STRIP_SUFFIXES.sub("", q).strip()
    q = q.strip("?.,! ")
    return q if q else user_text.strip()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def search_web(query: str, max_results: int = MAX_RESULTS) -> list[SearchResult]:
    """Search the web using DuckDuckGo via the ``ddgs`` library.

    Results are cached with timestamps for reproducibility.
    """

    cached = _cache.get(query)
    if cached is not None:
        logger.debug("Web search cache hit: %s (%d results)", query, len(cached))
        return cached

    results: list[SearchResult] = []

    try:
        from ddgs import DDGS  # noqa: E402 — lazy import

        raw_results = DDGS().text(query, max_results=max_results, backend=SEARCH_BACKEND)

        for r in raw_results:
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))

        _cache.put(query, results)
        logger.info("Web search: '%s' -> %d results", query[:50], len(results))

    except ImportError:
        logger.warning("ddgs package not installed — run: pip install ddgs")
    except Exception:
        logger.exception("Web search failed for: %s", query[:50])

    return results


async def fetch_page_content_for_summary(
    url: str,
    *,
    max_chars: int = MAX_PAGE_SUMMARY_CHARS,
) -> tuple[str, str]:
    """Fetch page text for user-selected result summarization.

    Reuses library ingest fetching to keep SSRF and content-type guards aligned
    with the ingestion pipeline. Returns ``(content, error)``.
    """
    clean_url = (url or "").strip()
    if not clean_url:
        return "", "empty_url"
    try:
        from library.ingest import _fetch_url  # Reuse guarded fetch path.
    except Exception as exc:
        return "", f"fetch_dependency_error:{exc.__class__.__name__}"

    try:
        content, err = await asyncio.to_thread(_fetch_url, clean_url)
    except Exception as exc:
        return "", f"fetch_failed:{exc.__class__.__name__}"

    if err:
        return "", err
    text = (content or "").strip()
    if not text:
        return "", "no_content"
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, ""


def format_results_for_llm(results: list[SearchResult], max_chars: int = 2000) -> str:
    """Format search results as context for LLM injection.

    Fencing: results are formatted as reference material, never as raw code
    to be pasted into patches.
    """
    if not results:
        return "(no web search results)"

    parts = ["## Web Search Results (reference only -- do NOT copy code directly)\n"]
    used = len(parts[0])

    for i, r in enumerate(results, 1):
        entry = f"{i}. **{r.title}**\n   URL: {r.url}\n   {r.snippet}\n"
        if used + len(entry) > max_chars:
            break
        parts.append(entry)
        used += len(entry)

    return "\n".join(parts)


def format_results_for_user(results: list[SearchResult], query: str = "", limit: int = 3) -> str:
    """Format live web results into a direct user-facing fallback reply."""
    if not results:
        return "I couldn't find any live web results right now."

    lines: list[str] = []
    if query:
        lines.append(f"Live web results for '{query}':")
    else:
        lines.append("Live web results:")

    for idx, result in enumerate(results[:limit], 1):
        title = (result.title or result.url or "Untitled result").strip()
        snippet = (result.snippet or "").strip()
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        entry = f"{idx}. {title}"
        if snippet:
            entry += f" — {snippet}"
        if result.url:
            entry += f" ({result.url})"
        lines.append(entry)

    return "\n".join(lines)
