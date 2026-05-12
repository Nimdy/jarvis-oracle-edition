"""Deterministic guardrails for search-route response consistency."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_SEARCH_ACCESS_DENIAL_RE = re.compile(
    r"\b(?:"
    r"(?:i|we)\s+(?:do\s*not|don't|cannot|can't|can not)\s+"
    r"(?:have\s+)?(?:access(?:\s+to)?|browse|search|reach)\s+(?:the\s+)?(?:internet|web|online|research(?:\s+databases?)?)"
    r"|(?:no|without)\s+(?:internet|web|online)\s+access"
    r"|(?:internet|web)\s+access\s+(?:is\s+)?(?:disabled|unavailable|restricted)"
    r"|cannot\s+perform\s+(?:live\s+)?(?:web|internet|online)\s+search"
    r"|(?:all|everything)\s+(?:runs|is)\s+locally(?:\s+only)?"
    r"|offline[-\s]?only"
    r")\b",
    re.IGNORECASE,
)


def contains_search_access_denial(text: str) -> bool:
    """Return True when a reply denies search/web capability."""
    return bool(text and _SEARCH_ACCESS_DENIAL_RE.search(text))


def guard_search_tool_reply(
    candidate_reply: str,
    fallback_reply: str,
    *,
    lane: str,
    search_query: str,
) -> tuple[str, bool]:
    """Fail closed when tool route and generated answer contradict each other.

    If model output denies web/research access after a successful search route,
    return deterministic tool-backed fallback text.
    """
    if not candidate_reply:
        return candidate_reply, False
    if contains_search_access_denial(candidate_reply):
        logger.warning(
            "Search route contradiction blocked (lane=%s query='%s'): %s",
            lane,
            search_query[:80],
            candidate_reply[:180].replace("\n", " "),
        )
        return fallback_reply or candidate_reply, True
    return candidate_reply, False
