"""Correction detector: semantic user correction -> Layer 5 debt.

Only fires on factual/inferential claims (not greetings, relational language,
reflective talk, or emotional mirroring). Three-gate system:
  1. Correction phrase match
  2. Content overlap with injected memory
  3. Previous response was a factual/inferential route
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("jarvis.calibration.correction")

_TOKEN_RE = re.compile(r"\b[\w']+\b")

_CORRECTION_PATTERNS = [
    re.compile(r"\bthat'?s?\s+(?:not\s+(?:right|correct|true|accurate)|wrong|incorrect)\b", re.I),
    re.compile(r"\bno[,.]?\s+(?:it'?s?\s+(?:actually|not)|that'?s?\s+not|i\s+(?:said|meant))\b", re.I),
    re.compile(r"\byou\s+(?:misunderstood|got\s+(?:it|that)\s+wrong|are\s+wrong)\b", re.I),
    re.compile(r"\bnot\s+\w+[,.]?\s+(?:it'?s?|that'?s?)\s+\w+", re.I),
    re.compile(r"\bi\s+(?:never\s+said|didn'?t\s+(?:say|mean))\b", re.I),
    re.compile(r"\b(?:actually|correction)[,:]?\s+", re.I),
]
_OWNERSHIP_CORRECTION_PATTERNS = [
    re.compile(r"\bthat'?s?\s+my\b", re.I),
    re.compile(r"\bnot\s+(?:yours?|you)\b", re.I),
    re.compile(r"\bmine,\s*not\s+yours\b", re.I),
    re.compile(r"\bmy\s+(?:birthday|name|wife|husband|partner|memory|preference|fact)\b", re.I),
]
_USER_PRIVATE_FACT_PATTERNS = [
    re.compile(r"\bmy\s+(?:birthday|name|wife|husband|partner|girlfriend|boyfriend|mom|mother|dad|father|brother|sister|son|daughter)\b", re.I),
    re.compile(r"\bmy\s+(?:favorite|favourite|preference|preferences)\b", re.I),
    re.compile(r"\bi\s+(?:like|love|enjoy|prefer|hate|dislike)\b", re.I),
]
_ARITHMETIC_PATTERNS = [
    re.compile(r"\b\d+\s*[\+\-\*/]\s*\d+\s*(?:=|equals?)\s*\d+\b", re.I),
    re.compile(r"\b\d+\s*[\+\-\*/]\s*\d+\b", re.I),
    re.compile(r"\b(?:plus|minus|times|multiplied|divided|equals?)\b", re.I),
]
_CAPABILITY_PATTERNS = [
    re.compile(r"\b(?:can|can't|cannot|able to|capable of|skill|skills|learned|verified)\b", re.I),
    re.compile(r"\b(?:sing|code|search|identify|recognize|classify|detect|analyze|perform)\b", re.I),
]
_SYSTEM_FACT_PATTERNS = [
    re.compile(r"\b(?:memory|memories|latency|gpu|cpu|uptime|port|dashboard|kernel|hemisphere|policy|model|vram)\b", re.I),
    re.compile(r"\b(?:truth score|belief graph|autonomy|stability|introspection|status)\b", re.I),
]
_SCENE_PERCEPTION_PATTERNS = [
    re.compile(r"\b(?:image|camera|snapshot|frame|scene|visual|vision)\b", re.I),
    re.compile(r"\b(?:world\s+prediction|world\s+model)\b", re.I),
    re.compile(r"\b(?:do\s+not\s+see|don't\s+see|not\s+there|isn't\s+there)\b", re.I),
    re.compile(r"\b(?:on|at)\s+the\s+(?:left|right|desk|table|monitor|screen|shelf)\b", re.I),
]

_FACTUAL_ROUTES = frozenset({
    "none", "memory", "introspection", "academic_search",
    "codebase", "web_search",
})

_MIN_RESPONSE_LENGTH = 20


def _has_correction_phrase(text: str) -> bool:
    return any(p.search(text) for p in _CORRECTION_PATTERNS)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _payload_indices_with_overlap(
    user_text: str,
    memory_payloads: list[str],
    threshold: int = 3,
) -> list[int]:
    """Return payload indices with meaningful lexical overlap to the correction."""
    if not memory_payloads:
        return []

    user_words = _tokenize(user_text)
    if len(user_words) < 3:
        return []

    trivial = {
        "the", "a", "an", "is", "was", "are", "it", "i", "you", "that", "this",
        "to", "in", "of", "and", "for", "not", "no", "my", "your", "me",
    }
    hits: list[int] = []
    for idx, payload in enumerate(memory_payloads):
        payload_words = _tokenize(str(payload))
        overlap = (user_words & payload_words) - trivial
        if len(overlap) >= threshold:
            hits.append(idx)
    return hits


def _content_overlap(user_text: str, memory_payloads: list[str], threshold: int = 3) -> bool:
    """Check if user's correction overlaps with injected memory content via n-gram comparison."""
    return bool(_payload_indices_with_overlap(user_text, memory_payloads, threshold))


def _text_overlap(user_text: str, reference_text: str, threshold: int = 2) -> bool:
    """Check whether a correction substantially overlaps the prior response text."""
    if not user_text or not reference_text:
        return False

    user_words = _tokenize(user_text)
    ref_words = _tokenize(reference_text)
    if len(user_words) < 3 or len(ref_words) < 3:
        return False

    if any(p.search(user_text or "") for p in _ARITHMETIC_PATTERNS) and any(p.search(reference_text or "") for p in _ARITHMETIC_PATTERNS):
        numeric_overlap = {
            tok for tok in (user_words & ref_words)
            if any(ch.isdigit() for ch in tok)
        }
        if numeric_overlap:
            return True

    trivial = {
        "the", "a", "an", "is", "was", "are", "it", "i", "you", "that", "this",
        "to", "in", "of", "and", "for", "not", "no", "my", "your", "me",
    }
    meaningful_overlap = (user_words & ref_words) - trivial
    return len(meaningful_overlap) >= threshold


def _classify_correction_kind(
    user_text: str,
    last_response_text: str,
    memory_payloads: list[str],
) -> str:
    """Classify whether the correction targets facts or self/user scoping."""
    user_lower = (user_text or "").lower()
    response_lower = (last_response_text or "").lower()
    payload_blob = " ".join(str(p).lower() for p in (memory_payloads or []))

    ownership_correction = any(p.search(user_text or "") for p in _OWNERSHIP_CORRECTION_PATTERNS)
    self_claim_response = bool(re.search(r"\b(?:my|mine|i|i'm|i am|me)\b", response_lower))
    user_owned_payload = "user's " in payload_blob or "the user" in payload_blob

    if ownership_correction and self_claim_response and (user_owned_payload or "not yours" in user_lower):
        return "identity_scope_leak"
    return "factual_mismatch"


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text or "") for p in patterns)


def _classify_authority_domain(
    user_text: str,
    last_response_text: str,
    memory_payloads: list[str],
    last_tool_route: str,
    correction_kind: str,
) -> tuple[str, str, bool]:
    """Return domain, adjudication policy, and whether user correction auto-applies."""
    joined = " ".join([user_text or "", last_response_text or "", *[str(p) for p in (memory_payloads or [])]])

    if correction_kind == "identity_scope_leak":
        return ("user_private_fact", "accept_user_scope", True)
    if _matches_any(joined, _USER_PRIVATE_FACT_PATTERNS):
        return ("user_private_fact", "accept_user_fact", True)
    if _matches_any(joined, _ARITHMETIC_PATTERNS):
        return ("objective_math_logic", "contest_or_contextualize", False)
    if correction_kind == "factual_mismatch" and _matches_any(joined, _SCENE_PERCEPTION_PATTERNS):
        return ("scene_or_perception_fact", "verify_with_perception", False)
    if last_tool_route.lower() in {"introspection", "status", "system_status", "codebase"} or _matches_any(joined, _SYSTEM_FACT_PATTERNS):
        return ("jarvis_system_fact", "verify_with_subsystems", False)
    if _matches_any(joined, _CAPABILITY_PATTERNS):
        return ("capability_or_skill", "verify_with_registry", False)
    return ("objective_or_external_fact", "require_evidence", False)


class CorrectionDetector:
    """Detects user corrections and routes them to Layer 5 debt + confidence calibrator."""

    def __init__(self) -> None:
        self._total_corrections: int = 0
        self._total_checks: int = 0

    def check(
        self,
        user_text: str,
        is_negative: bool,
        last_response_text: str,
        last_tool_route: str,
        injected_memory_payloads: list[str],
    ) -> dict[str, Any] | None:
        """Returns correction info dict if a correction is detected, else None.

        Only fires when all three gates pass:
          1. Correction phrase in user_text
          2. Content overlap with injected memory payloads
          3. Last response was a factual/inferential route
        """
        self._total_checks += 1

        if not is_negative:
            return None

        if last_tool_route.lower() not in _FACTUAL_ROUTES:
            return None

        if len(last_response_text) < _MIN_RESPONSE_LENGTH:
            return None

        if not _has_correction_phrase(user_text):
            return None

        overlap_indices = _payload_indices_with_overlap(user_text, injected_memory_payloads)
        memory_overlap = bool(overlap_indices)
        response_overlap = _text_overlap(user_text, last_response_text)
        if not memory_overlap and not response_overlap:
            return None

        self._total_corrections += 1
        correction_kind = _classify_correction_kind(
            user_text,
            last_response_text,
            injected_memory_payloads,
        )
        authority_domain, adjudication_policy, auto_accept = _classify_authority_domain(
            user_text,
            last_response_text,
            injected_memory_payloads,
            last_tool_route,
            correction_kind,
        )
        return {
            "user_text": user_text[:200],
            "route": last_tool_route,
            "overlap_detected": True,
            "overlap_basis": "memory" if memory_overlap else "response",
            "overlap_memory_indices": overlap_indices,
            "correction_kind": correction_kind,
            "authority_domain": authority_domain,
            "adjudication_policy": adjudication_policy,
            "auto_accept_user_correction": auto_accept,
        }

    def get_stats(self) -> dict[str, int]:
        return {
            "total_corrections": self._total_corrections,
            "total_checks": self._total_checks,
        }
