"""Conversation handler — routes user messages to tools or LLM streaming."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path as _Path
from typing import Any

from consciousness.engine import ConsciousnessEngine
from consciousness.events import event_bus, CONVERSATION_USER_MESSAGE, CONVERSATION_RESPONSE
from consciousness.release_validation import output_release_validator
from consciousness.trace_context import build_trace_context
from consciousness.soul import soul_service
from reasoning.response import ResponseGenerator
from reasoning.ollama_client import OllamaClient
from reasoning.claude_client import ClaudeClient
from reasoning.golden_words import GoldenCommandContext, list_canonical_commands, with_golden_outcome
from reasoning.tool_router import tool_router, ToolType, RoutingResult
from reasoning.context import context_builder
from reasoning.bounded_response import articulate_meaning_frame, build_meaning_frame
from reasoning.language_runtime_bridge import (
    decide_runtime_consumption,
    load_runtime_language_policy,
)
from tools.time_tool import get_current_time
from tools.system_tool import get_system_status
from tools.memory_tool import search_memory, get_memory_summary
from tools.vision_tool import describe_scene, describe_scene_stream
from tools.introspection_tool import (
    get_grounded_learning_job_status_answer,
    get_grounded_learning_job_status_record,
    get_grounded_recent_learning_answer,
    get_grounded_recent_learning_record,
    get_introspection,
)
from tools.web_search_tool import (
    search_web,
    extract_search_query,
    fetch_page_content_for_summary,
)
from tools.academic_search_tool import (
    search_academic,
    format_academic_results_for_llm,
    format_academic_results_for_user,
    classify_search_lane,
)
from perception.server import PerceptionServer
from memory.search import index_memory
from memory.episodes import EpisodicMemory
from memory.core import memory_core, CreateMemoryData
from memory.storage import memory_storage
from reasoning.tts import BrainTTS
from reasoning.style_intent import detect_style_intent
from consciousness.reflection import reflection_engine
from reasoning.response import _score_complexity
from reasoning.search_route_guard import guard_search_tool_reply

logger = logging.getLogger("jarvis.conversation")

_BRIEF_SIGNALS = re.compile(
    r"\b(keep it (short|brief|concise)|be (brief|concise|short)|shorter|less detail)\b", re.IGNORECASE)
_DETAIL_SIGNALS = re.compile(
    r"\b(more detail|be (detailed|thorough|verbose)|explain (more|fully)|go deeper|elaborate)\b", re.IGNORECASE)
_AFFIRMATIVE_FOLLOWUP_RE = re.compile(
    r"^(?:yes|yeah|yep|sure|okay|ok|please do|go ahead|do it|sounds good|if you want to|yes please|sure please go ahead)"
    r"(?:[.! ]*)$",
    re.IGNORECASE,
)
_CAMERA_OFFER_RE = re.compile(
    r"\b(?:camera|zoom|focus|closer look|look at it with the camera|set up a camera view|adjust the camera|catch her when she comes in)\b",
    re.IGNORECASE,
)
_SYSTEM_EXPLANATION_RE = re.compile(
    r"\b(?:how do you work|how are you wired|what are you made of|explain your (?:system|architecture)|"
    r"your architecture|your codebase|your subsystems?|"
    r"how (?:do )?you (?:store|retrieve|process|route|learn)|"
    r"how (?:does |do )?your (?:memory|brain|system|pipeline|architecture|neural net\w*) works?)\b",
    re.IGNORECASE,
)
_ROUTING_CORRECTION_RE = re.compile(
    r"(?:when i say|if i say|when i ask|if i ask)\s+"
    r"(?:(?:\"([^\"]+)\"|'([^']+)'|(.+?)))"
    r"\s*(?:,\s*)?(?:that (?:means?|should)|you should|route (?:that|it|this) to|"
    r"send (?:that|it) to|it (?:means?|should go to)|i mean)\s+"
    r"(?:your\s+)?(\w[\w\s]*)",
    re.IGNORECASE,
)
_MEMORY_CONTENT_INTENT_RE = re.compile(
    r"\b(?:tell me|share|what (?:is|are|was)|describe|one of|about your)\b"
    r".{0,20}\b(?:your |a )?(?:memor(?:y|ies)|remember)\b"
    r"(?!.{0,5}\b(?:health|status|count|stats?)\b)",
    re.IGNORECASE,
)
_AMBIGUOUS_SELF_READ_RE = re.compile(
    r"\b(?:what(?:'s| is)?|what did|tell me|last thing)\b"
    r".{0,50}\b(?:you|your|jarvis)\b.{0,50}\b(?:read|reading)\b",
    re.IGNORECASE,
)
_ROUTING_CORRECTION_SIMPLE_RE = re.compile(
    r"(?:that should (?:be|go to)|route that to|send that to|"
    r"that.s a|that is a|that was a|that belongs in)\s+"
    r"(?:a\s+)?(?:your\s+)?(?:the\s+)?(\w[\w\s]*?)(?:\s+(?:system|route|router|handler))?[.!]?\s*$",
    re.IGNORECASE,
)
_ROUTE_NAME_MAP: dict[str, str] = {
    "skill": "SKILL", "skills": "SKILL", "skill system": "SKILL",
    "skill request": "SKILL", "skill thing": "SKILL",
    "learning": "SKILL", "learn": "SKILL", "training": "SKILL",
    "skill learning": "SKILL", "learning system": "SKILL",
    "identity": "IDENTITY", "identity request": "IDENTITY",
    "enrollment": "IDENTITY", "speaker": "IDENTITY",
    "memory": "MEMORY", "memories": "MEMORY", "memory request": "MEMORY",
    "introspection": "INTROSPECTION", "self report": "INTROSPECTION",
    "status": "STATUS", "status request": "STATUS",
    "vision": "VISION", "camera": "VISION",
    "search": "WEB_SEARCH", "web search": "WEB_SEARCH",
    "code": "CODEBASE", "codebase": "CODEBASE",
    "academic": "ACADEMIC_SEARCH", "research": "ACADEMIC_SEARCH",
    "self improve": "SELF_IMPROVE", "self improvement": "SELF_IMPROVE",
}
_NEEDS_PRIOR_CONTEXT_RE = re.compile(
    r"\b(?:that|it|the (?:last|previous) (?:thing|one|part)|"
    r"what (?:you|i) (?:just|were) (?:said|mentioned|told|asked|talked about)|"
    r"do it|start it|go ahead|let's do it|that one|the thing before|"
    r"what (?:did )?(?:you|i) (?:just )?(?:say|mean|mention))\b",
    re.IGNORECASE,
)
_LIKELY_STT_FRAGMENT_RE = re.compile(
    r"^(?:uh|um|hmm+|mm+|huh|listening|signal|signals|silence|static|noise)$",
    re.IGNORECASE,
)
_ENROLLMENT_OFFER_RE = re.compile(
    r"\b(?:remember (?:your|my)|save (?:your|my)|enroll|register|"
    r"learn (?:your|my) (?:voice|face|name)|"
    r"record (?:your|my)|would you like me to (?:remember|save|enroll|register))\b",
    re.IGNORECASE,
)
_GUIDED_COLLECT_OFFER_RE = re.compile(
    r"\b(?:start training|begin training|help (?:me )?train|training mode|"
    r"calibrat(?:e|ion)|collect samples|guided collect|"
    r"would you like (?:to|me to) (?:start|begin) training|"
    r"ready to train|shall (?:we|i) (?:start|begin) training)\b",
    re.IGNORECASE,
)
_EXPLICIT_WEB_REQUEST_RE = re.compile(
    r"\b(?:search\s+(?:the\s+)?(?:web|internet|online)|web\s+search|online\s+search|"
    r"duckduckgo|google|look\s+up\s+online|find\s+online)\b",
    re.IGNORECASE,
)
_CAPABILITY_STATUS_QUERY_RE = re.compile(
    r"\b(?:"
    r"what can you do|"
    r"what can you currently do|"
    r"what (?:are|'re) (?:your|the current) capabilities|"
    r"what (?:are|'re) your current (?:skills|abilities)|"
    r"what (?:skills|abilities) do you have|"
    r"which capabilities (?:are|is) (?:verified|available)|"
    r"what capabilities (?:are|do you have|can you)|"
    r"capabilities?\s+(?:that\s+)?(?:you\s+can|you\s+have|are\s+verified)|"
    r"what (?:can't|cannot|can not) you do|"
    r"still learning"
    r")\b",
    re.IGNORECASE,
)
_RECENT_RESEARCH_QUERY_RE = re.compile(
    r"\b(?:research(?:ed|ing)?|stud(?:ied|y|ying)|journal|paper|peer.?reviewed|scholarly)\b",
    re.IGNORECASE,
)
_RESEARCH_CONTINUATION_FOLLOWUP_RE = re.compile(
    r"\b(?:continue|keep|go on|stay up on|stay current on|stay updated on)\b"
    r".{0,40}\b(?:research(?:ing)?|study(?:ing)?|studies|papers?|literature|this|it)\b|"
    r"\b(?:research|study)\s+(?:this|it)\b",
    re.IGNORECASE,
)
_PERSONAL_ACTIVITY_RECALL_RE = re.compile(
    r"\b(?:what\s+did\s+(?:i|we)\s+do|tell\s+me\s+what\s+(?:i|we)\s+did|what\s+(?:i|we)\s+did)\b"
    r".{0,60}\b(?:yesterday|today|tonight|last\s+night|earlier(?:\s+today)?|"
    r"(?:last|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:on\s+)?[a-z]+(?:\s+[a-z]+){0,2})\b|"
    r"\b(?:where|when)\s+did\s+(?:i|we)\s+(?:go|do)\b.{0,60}\b(?:yesterday|today|"
    r"last\s+night|earlier(?:\s+today)?|(?:last|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:on\s+)?[a-z]+(?:\s+[a-z]+){0,2})\b",
    re.IGNORECASE,
)
_RECONCILE_MERGE_RE = re.compile(
    r"\bmerge\s+(\w+)\s+into\s+(\w+)\b",
    re.IGNORECASE,
)
_RECONCILE_ALIAS_RE = re.compile(
    r"\b([A-Z]\w+)\s+is\s+(?:actually|really)\s+([A-Z]\w+)\b",
)
_FORGET_NAME_RE = re.compile(
    r"\bforget\s+(?:the\s+)?(?:name\s+)?(\w{2,})\b",
    re.IGNORECASE,
)
_WEB_SELECTION_SIMPLE_RE = re.compile(
    r"^\s*(?:option|result|number|pick|choose|select|open|summarize|summary|visit)?\s*"
    r"(?:number\s*)?(1|2|3|one|two|three|first|second|third)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_WEB_SELECTION_EMBEDDED_RE = re.compile(
    r"\b(?:option|result|number|pick|choose|select|open|summarize|summary|visit)\s*"
    r"(?:number\s*)?(1|2|3|one|two|three|first|second|third)\b",
    re.IGNORECASE,
)
_WEB_SELECTION_TOKEN_MAP = {
    "1": 0,
    "one": 0,
    "first": 0,
    "2": 1,
    "two": 1,
    "second": 1,
    "3": 2,
    "three": 2,
    "third": 2,
}
_WEB_SELECTION_LIMIT = 3
_WEB_SELECTION_TTL_S = 300.0
_WEB_SUMMARY_MAX_PAGE_CHARS = 9000

# Deterministic catch for capability-creation requests on NONE route.
# These require Golden Command ACQUIRE, not general LLM chat.
_CAPABILITY_CREATION_RE = re.compile(
    r"\b(?:create|build|make|develop|add|write|code|implement|generate|give me)\s+"
    r"(?:a |an |me a |me an )?"
    r"(?:new\s+)?"
    r"(?:plugin|tool|extension|feature|capability|module|skill|integration|addon|add-on)\b",
    re.IGNORECASE,
)
_TIMER_CREATION_RE = re.compile(
    r"\b(?:set|create|start|make|give me)\s+"
    r"(?:a |an |me a )?"
    r"(?:timer|alarm|reminder|countdown|schedule|notification)\b",
    re.IGNORECASE,
)

_CAPABILITY_CREATION_REPLY = (
    "I can't create plugins or tools through conversation alone. "
    "Capability creation goes through my governed acquisition pipeline. "
    "You can use the ACQUIRE command to start that process, or ask me to "
    "'learn' a specific skill and I'll set up a learning job for it."
)
_TIMER_CREATION_REPLY = (
    "I don't have a timer or reminder capability yet. "
    "That would need to be built as a plugin through my acquisition pipeline."
)


def _check_capability_creation_request(text: str) -> str | None:
    """Return a fixed response if the user is asking to create a capability/timer.

    Returns None if this is not a creation request.
    """
    lower = text.lower().strip()
    if _TIMER_CREATION_RE.search(lower):
        return _TIMER_CREATION_REPLY
    if _CAPABILITY_CREATION_RE.search(lower):
        return _CAPABILITY_CREATION_REPLY
    return None


@dataclass
class _PendingWebSelection:
    query: str
    lane: str
    created_at: float
    results: list[dict[str, str]]


_pending_web_selection_by_actor: dict[str, _PendingWebSelection] = {}


_POSITIVE_QUALITY_DELTA = 0.005
_NEGATIVE_QUALITY_DELTA = -0.01
_POSITIVE_MEMORY_BOOST = 0.03
_MEMORY_RESULT_LINE_RE = re.compile(
    r"^\s*-\s*\(relevance=(?P<score>\d+(?:\.\d+)?)\)\s*(?P<preview>.+)$",
    re.IGNORECASE,
)
_MEMORY_TYPE_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
_MEMORY_TYPE_CAPTURE_RE = re.compile(r"^\s*\[(?P<type>[^\]]+)\]\s*")
_MEMORY_SPEAKER_PREFIX_RE = re.compile(
    r"^(?:jarvis\s+(?:recalled|replied)|user\s+said)\s*:\s*",
    re.IGNORECASE,
)
_MEMORY_HIGH_PRIORITY_TYPES = frozenset({
    "conversation",
    "task_completed",
    "user_preference",
    "factual_knowledge",
    "episode_summary",
})
_MEMORY_LOW_PRIORITY_TYPES = frozenset({
    "observation",
    "contextual_insight",
    "self_improvement",
    "core",
    "error_recovery",
})


def _shadow_language_compare(
    conversation_id: str,
    query: str,
    reply: str,
    meaning_frame: Any,
    response_class: str,
) -> None:
    """Phase C: silently compare bounded reply with shadow language model retrieval."""
    mf_dict = meaning_frame.to_dict() if hasattr(meaning_frame, "to_dict") else meaning_frame
    try:
        from reasoning.shadow_language_model import shadow_language_inference
        from reasoning.language_telemetry import language_quality_telemetry
    except Exception:
        logger.warning("Shadow comparison skipped: telemetry dependencies unavailable", exc_info=True)
        return

    try:
        if shadow_language_inference.available:
            style_shadow_reply = shadow_language_inference.shadow_generate(
                query=query, meaning_frame=mf_dict, response_class=response_class,
            )
            if style_shadow_reply:
                language_quality_telemetry.record_shadow_comparison(
                    conversation_id=conversation_id,
                    response_class=response_class,
                    query=query[:180],
                    bounded_reply=reply[:300],
                    llm_reply=style_shadow_reply[:300],
                    bounded_confidence=float(
                        mf_dict.get("frame_confidence", 0) if isinstance(mf_dict, dict)
                        else getattr(meaning_frame, "frame_confidence", 0)
                    ),
                    chosen="bounded",
                    reason="shadow_language_model",
                    model_family="style_retrieval",
                )
    except Exception:
        logger.warning("Shadow style comparison failed", exc_info=True)

    try:
        from reasoning.language_phasec import phasec_shadow_student, is_live_routing_enabled
        if is_live_routing_enabled():
            logger.error(
                "Phase C live routing guard triggered; keeping shadow comparison telemetry-only.",
            )

        lead = ""
        facts_text = ""
        if isinstance(mf_dict, dict):
            lead = str(mf_dict.get("lead", "") or "")
            facts = mf_dict.get("facts", []) if isinstance(mf_dict.get("facts"), list) else []
            facts_text = " | ".join(str(f) for f in facts[:5])
        prompt = (
            f"CLASS:{response_class}\n"
            f"QUERY:{query}\n"
            f"LEAD:{lead}\n"
            f"FACTS:{facts_text}\n"
            "ANSWER:"
        )
        phasec_reply = phasec_shadow_student.generate_shadow(
            query=query,
            response_class=response_class,
            prompt=prompt,
            max_tokens=48,
        )
        if phasec_reply:
            language_quality_telemetry.record_shadow_comparison(
                conversation_id=conversation_id,
                response_class=response_class,
                query=query[:180],
                bounded_reply=reply[:300],
                llm_reply=phasec_reply[:300],
                bounded_confidence=float(
                    mf_dict.get("frame_confidence", 0) if isinstance(mf_dict, dict)
                    else getattr(meaning_frame, "frame_confidence", 0)
                ),
                chosen="bounded",
                reason="phasec_adapter_student",
                model_family="phasec_adapter",
            )
    except Exception:
        logger.warning("Phase C shadow comparison failed", exc_info=True)


class _OutcomeCounters:
    """Observable counters for cortex outcome logging health."""

    __slots__ = ("attempts", "successes", "failures", "last_error")

    def __init__(self) -> None:
        self.attempts: int = 0
        self.successes: int = 0
        self.failures: int = 0
        self.last_error: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "outcome_log_attempts": self.attempts,
            "outcome_log_successes": self.successes,
            "outcome_log_failures": self.failures,
            "last_outcome_log_error": self.last_error,
        }


outcome_counters = _OutcomeCounters()


def _is_matrix_trigger(text: str) -> bool:
    """Detect if the user explicitly requested Matrix Protocol learning."""
    lower = text.lower()
    try:
        from config import BrainConfig
        aliases = BrainConfig().matrix_trigger_aliases
    except Exception:
        aliases = ["use the matrix to learn", "matrix learn",
                   "matrix style", "matrix-style"]
    for alias in aliases:
        if alias in lower:
            return True
    import re as _re_mx
    if _re_mx.search(r"\bmatrix\b.{0,20}\b(?:learn|style|protocol)\b", lower):
        return True
    return False


import collections
import uuid as _uuid_mod

_flight_recorder: collections.deque[dict[str, Any]] = collections.deque(maxlen=50)
_golden_outcomes: collections.deque[dict[str, Any]] = collections.deque(maxlen=50)
_FLIGHT_RECORDER_PATH = _Path.home() / ".jarvis" / "flight_recorder.json"


def _load_flight_recorder() -> None:
    """Restore flight recorder episodes from disk on startup."""
    try:
        if _FLIGHT_RECORDER_PATH.exists():
            import json as _json_fr
            data = _json_fr.loads(_FLIGHT_RECORDER_PATH.read_text())
            if isinstance(data, list):
                for ep in data[-50:]:
                    _flight_recorder.append(ep)
            logger.info("Flight recorder: restored %d episodes", len(_flight_recorder))
    except Exception:
        logger.debug("Flight recorder load failed", exc_info=True)


def _save_flight_recorder() -> None:
    """Persist flight recorder episodes to disk."""
    try:
        import json as _json_fr
        _FLIGHT_RECORDER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FLIGHT_RECORDER_PATH.with_suffix(".tmp")
        tmp.write_text(_json_fr.dumps(list(_flight_recorder), default=str))
        tmp.replace(_FLIGHT_RECORDER_PATH)
    except Exception:
        logger.debug("Flight recorder save failed", exc_info=True)


_load_flight_recorder()


def get_flight_episodes() -> list[dict[str, Any]]:
    """Return recent cognitive flight episodes for the dashboard."""
    return list(_flight_recorder)


def get_golden_command_outcomes() -> dict[str, Any]:
    """Return recent Golden command outcomes for dashboard/telemetry."""
    recent = list(_golden_outcomes)
    counts: dict[str, int] = {}
    for item in recent:
        status = str(item.get("status", "none") or "none")
        counts[status] = counts.get(status, 0) + 1
    return {
        "recent": recent[-20:],
        "counts": counts,
        "last": recent[-1] if recent else None,
    }


def _record_golden_outcome(
    golden_context: GoldenCommandContext | None,
    *,
    tool_route: str,
) -> None:
    if golden_context is None:
        return
    _golden_outcomes.append({
        "timestamp": __import__("time").time(),
        "trace_id": golden_context.trace_id,
        "status": golden_context.golden_status,
        "command_id": golden_context.command_id,
        "canonical_body": golden_context.canonical_body,
        "authority_class": golden_context.authority_class,
        "block_reason": golden_context.block_reason,
        "tool_route": tool_route,
    })


def _derive_memory_summary_from_context(memory_ctx: str) -> dict[str, Any] | None:
    """Recover coarse retrieval counts from rendered memory tool output."""
    if not memory_ctx:
        return None

    _patterns = (
        (r"Found\s+(\d+)\s+relevant memory\(ies\)", "memory_tool_search", "semantic_keyword", "memory"),
        (r"Found\s+(\d+)\s+episode\(s\)", "episodic_recall", "episode_summaries", "episode"),
        (r"Found\s+(\d+)\s+episode summary\(ies\)", "episodic_recall", "episode_summaries", "episode_summary"),
    )
    for pattern, route_type, search_scope, memory_type in _patterns:
        match = re.search(pattern, memory_ctx, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            return {
                "count": count,
                "subjects": {},
                "types": {memory_type: count} if count > 0 else {},
                "route_type": route_type,
                "search_scope": search_scope,
            }
    return None


def _is_affirmative_followup(text: str) -> bool:
    return bool(_AFFIRMATIVE_FOLLOWUP_RE.match(text.strip()))


def _is_research_continuation_followup(text: str) -> bool:
    return bool(text and _RESEARCH_CONTINUATION_FOLLOWUP_RE.search(text))


def _is_explicit_web_request(text: str) -> bool:
    return bool(text and _EXPLICIT_WEB_REQUEST_RE.search(text))


def _web_selection_actor_key(speaker: str, conversation_id: str) -> str:
    spk = (speaker or "").strip().lower()
    if spk and spk != "unknown":
        return f"speaker:{spk}"
    cid = (conversation_id or "").strip()
    if cid:
        return f"conv:{cid}"
    return "conv:default"


def _prune_pending_web_selections(now: float | None = None) -> None:
    ts = now if now is not None else time.time()
    stale_keys = [
        key
        for key, pending in _pending_web_selection_by_actor.items()
        if ts - pending.created_at > _WEB_SELECTION_TTL_S
    ]
    for key in stale_keys:
        _pending_web_selection_by_actor.pop(key, None)


def _get_pending_web_selection(actor_key: str) -> _PendingWebSelection | None:
    _prune_pending_web_selections()
    return _pending_web_selection_by_actor.get(actor_key)


def _store_pending_web_selection(
    actor_key: str,
    *,
    query: str,
    lane: str,
    results: list[dict[str, str]],
) -> None:
    cleaned: list[dict[str, str]] = []
    for item in results[:_WEB_SELECTION_LIMIT]:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        if not title and not url:
            continue
        cleaned.append({
            "title": title or url or "Untitled result",
            "url": url,
            "snippet": snippet,
        })
    if not cleaned:
        _pending_web_selection_by_actor.pop(actor_key, None)
        return
    _pending_web_selection_by_actor[actor_key] = _PendingWebSelection(
        query=query.strip(),
        lane=lane,
        created_at=time.time(),
        results=cleaned,
    )


def _parse_web_selection_index(text: str) -> int | None:
    if not text:
        return None
    raw = text.strip().lower()
    if not raw:
        return None

    simple = _WEB_SELECTION_SIMPLE_RE.match(raw)
    token = simple.group(1).lower() if simple else ""
    if not token and len(raw) <= 40:
        embedded = _WEB_SELECTION_EMBEDDED_RE.search(raw)
        token = embedded.group(1).lower() if embedded else ""
    if not token and len(raw) <= 12 and raw in _WEB_SELECTION_TOKEN_MAP:
        token = raw
    if not token:
        return None
    return _WEB_SELECTION_TOKEN_MAP.get(token)


def _truncate_search_snippet(snippet: str, limit: int = 220) -> str:
    clean = (snippet or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _format_selection_prompt(
    *,
    query: str,
    lane: str,
    results: list[dict[str, str]],
) -> str:
    if not results:
        return "I couldn't find usable results right now."
    prefix = "Top scholarly results" if lane == "scholarly" else "Top live web results"
    lines = [f"{prefix} for '{query}':"]
    for idx, item in enumerate(results[:_WEB_SELECTION_LIMIT], 1):
        title = (item.get("title") or item.get("url") or "Untitled result").strip()
        snippet = _truncate_search_snippet(item.get("snippet", ""))
        line = f"{idx}. {title}"
        if snippet:
            line += f" — {snippet}"
        url = (item.get("url") or "").strip()
        if url:
            line += f" ({url})"
        lines.append(line)
    lines.append("Say 1, 2, or 3 and I will open that page and summarize it.")
    return "\n".join(lines)


def _format_selected_result_fallback(
    *,
    query: str,
    selection_number: int,
    selected: dict[str, str],
    page_text: str = "",
    page_error: str = "",
) -> str:
    title = (selected.get("title") or selected.get("url") or "Untitled result").strip()
    url = (selected.get("url") or "").strip()
    snippet = _truncate_search_snippet(selected.get("snippet", ""), limit=260)
    page_excerpt = _truncate_search_snippet(page_text, limit=500)

    parts = [f"I opened result {selection_number} for '{query}': {title}."]
    if page_excerpt:
        parts.append(f"Summary: {page_excerpt}")
    elif snippet:
        parts.append(f"Summary: {snippet}")
    if page_error:
        parts.append(
            f"I could not read full page content ({page_error}); "
            "I can still summarize another result if you want."
        )
    if url:
        parts.append(f"Source: {url}")
    return " ".join(parts)


def _academic_results_to_selection_items(academic_results: list[Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for result in academic_results[:_WEB_SELECTION_LIMIT]:
        title = str(getattr(result, "title", "") or "").strip() or "Untitled paper"
        url = (
            str(getattr(result, "open_access_pdf_url", "") or "").strip()
            or str(getattr(result, "doi_url", "") or "").strip()
            or str(getattr(result, "url", "") or "").strip()
        )
        summary = (
            str(getattr(result, "tldr", "") or "").strip()
            or str(getattr(result, "abstract", "") or "").strip()
        )
        items.append({"title": title, "url": url, "snippet": summary})
    return items


def _web_results_to_selection_items(web_results: list[Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for result in web_results[:_WEB_SELECTION_LIMIT]:
        title = str(getattr(result, "title", "") or "").strip() or "Untitled result"
        url = str(getattr(result, "url", "") or "").strip()
        snippet = str(getattr(result, "snippet", "") or "").strip()
        items.append({"title": title, "url": url, "snippet": snippet})
    return items


def _normalize_number_set(raw_numbers: set[str]) -> set[str]:
    """Normalize a set of numeric strings so 0.989 and 98.9 match.

    For each number, produces both the original and its percentage or
    decimal equivalent (rounded to 1 decimal). This handles the common
    case where subsystem data reports 0.989 and the LLM says 98.9%.
    """
    normalized: set[str] = set()
    for n in raw_numbers:
        normalized.add(n)
        try:
            val = float(n)
            if 0.0 < val < 1.0:
                pct = round(val * 100, 1)
                normalized.add(str(pct))
                normalized.add(str(int(pct)) if pct == int(pct) else str(pct))
            elif 1.0 < val <= 100.0:
                dec = round(val / 100, 3)
                normalized.add(str(dec))
        except ValueError:
            pass
    return normalized


def _log_introspection_grounding(
    reply: str, introspection_data: str, meta: dict[str, Any],
) -> bool:
    """Log whether the LLM answer actually referenced extracted facts.

    Checks for numeric values (with decimal/percentage normalization),
    section titles, and key terms from the introspection data appearing
    in the reply. This is the grounding eval signal for later analysis.
    """
    if not reply or not introspection_data:
        return False

    import re as _re
    raw_data_nums = set(_re.findall(r"\b\d+\.?\d*\b", introspection_data))
    raw_reply_nums = set(_re.findall(r"\b\d+\.?\d*\b", reply))

    norm_data = _normalize_number_set(raw_data_nums)
    norm_reply = _normalize_number_set(raw_reply_nums)

    shared_numbers = norm_data & norm_reply
    trivial_numbers = {"0", "1", "2", "3", "4", "5", "10", "100",
                       "0.0", "1.0", "2.0", "3.0", "4.0", "5.0",
                       "10.0", "100.0"}
    meaningful_shared = shared_numbers - trivial_numbers

    named_entity_matches = 0
    _GROUNDING_ENTITIES = (
        "speech output", "memory search", "self-introspection", "introspection",
        "codebase analysis", "academic search", "web search",
        "speaker identification", "emotion detection", "face identification",
        "vision analysis", "hemisphere", "self-improvement",
        "camera control", "sqlite", "sqlite-vec", "json", "pydantic",
        "pytorch", "onnx", "faster-whisper", "kokoro", "ollama",
        "ecapa", "mobilefacenet", "silero", "wakeword", "openwakeword",
    )
    reply_lower = reply.lower()
    for entity in _GROUNDING_ENTITIES:
        if entity in reply_lower and entity in introspection_data.lower():
            named_entity_matches += 1

    # Also check for state/mode keywords that indicate grounded awareness
    _STATE_KEYWORDS = (
        "conversational", "reflective", "gestation", "passive", "focused",
        "dreaming", "deep_learning", "sleep",
        "basic_awareness", "self_reflective", "introspective", "abstract_reasoning",
        "recursive_self_modeling", "integrative",
        "healthy", "degraded", "optimal", "improving",
        "shadow", "active", "advisory",
    )
    state_matches = 0
    data_lower = introspection_data.lower()
    for kw in _STATE_KEYWORDS:
        if kw in reply_lower and kw in data_lower:
            state_matches += 1

    sections = meta.get("selected_sections", [])
    total_facts = meta.get("total_facts", 0)
    topics = meta.get("matched_topics", [])

    grounded = (
        len(meaningful_shared) >= 1
        or named_entity_matches >= 2
        or (named_entity_matches >= 1 and state_matches >= 1)
        or state_matches >= 2
    )
    logger.info(
        "Introspection grounding: topics=%s facts_extracted=%d "
        "numbers_shared=%d entities_shared=%d grounded=%s sections=%s",
        topics, total_facts, len(meaningful_shared), named_entity_matches,
        grounded, sections,
    )
    if not grounded and total_facts > 0:
        logger.warning(
            "Introspection grounding MISS: %d facts provided but reply cited none. "
            "Reply preview: %s",
            total_facts, reply[:200],
        )
    return grounded


def _format_grounded_fallback(title: str, body: str, max_lines: int = 12, max_chars: int = 900) -> str:
    """Turn grounded tool data into a spoken-word-friendly fallback reply.

    This fires when the LLM failed to cite the data, so the system
    presents the data directly.  Instead of a raw key-value dump we
    produce short natural sentences suitable for TTS.
    """
    cleaned: list[str] = []
    for raw_line in body.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        cleaned.append(line)

    if not cleaned:
        return title

    sentences: list[str] = []
    used = 0
    for line in cleaned:
        if len(sentences) >= max_lines:
            break
        # Skip section headers (=== ... ===) — not speakable
        if line.startswith("===") and line.endswith("==="):
            continue
        # Skip indented sub-items that are just noise
        if line.startswith("  ") and ":" not in line:
            continue
        # Rephrase "Key: Value" as a sentence fragment
        if ":" in line and not line.startswith("http"):
            key, _, value = line.partition(":")
            key = key.strip().lstrip("- ")
            value = value.strip()
            if value and key:
                sentence = f"{key} is {value}."
            elif key:
                sentence = f"{key}."
            else:
                continue
        else:
            sentence = line if line.endswith(".") else f"{line}."
        if used + len(sentence) > max_chars:
            break
        sentences.append(sentence)
        used += len(sentence)

    if not sentences:
        return title

    return " ".join(sentences)


def _normalize_memory_preview(preview: str, max_chars: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(preview or "")).strip()
    if not text:
        return ""

    text = _MEMORY_TYPE_PREFIX_RE.sub("", text).strip()
    if "user_message" in text and "response" in text:
        text = text.replace("{", " ").replace("}", " ")
        text = text.replace('"', "").replace("'", "")
        text = re.sub(r"\buser_message\s*:\s*", "User said: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bresponse\s*:\s*", "Jarvis replied: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*,\s*Jarvis replied:", " | Jarvis replied:", text)

    text = re.sub(r"\s+", " ", text).strip(" -|,.;")
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _extract_memory_type(preview: str) -> str:
    match = _MEMORY_TYPE_CAPTURE_RE.match(str(preview or ""))
    if not match:
        return "memory"
    return str(match.group("type") or "memory").strip().lower()


def _memory_priority(memory_type: str, normalized_preview: str) -> int:
    lower = normalized_preview.lower()
    if lower.startswith("user said:"):
        return 3
    if memory_type in _MEMORY_HIGH_PRIORITY_TYPES:
        return 0
    if memory_type in _MEMORY_LOW_PRIORITY_TYPES:
        return 2
    return 1


def _to_speakable_memory_sentence(preview: str, max_chars: int = 170) -> str:
    text = re.sub(r"\s+", " ", str(preview or "")).strip()
    if not text:
        return ""
    text = _MEMORY_SPEAKER_PREFIX_RE.sub("", text).strip(" -|,")
    if not text:
        return ""
    text = text.replace(" | ", ". ")
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    if text and text[-1] not in ".!?":
        text = f"{text}."
    return text


def _format_personal_activity_memory_reply(memory_ctx: str, max_items: int = 2) -> str:
    if not memory_ctx.strip():
        return "I couldn't find matching memories for that time window."

    total = 0
    header = re.search(r"Found\s+(\d+)\s+relevant memory\(ies\)", memory_ctx, re.IGNORECASE)
    if header:
        total = int(header.group(1))

    ranked_items: list[tuple[int, float, str]] = []
    for raw_line in memory_ctx.splitlines():
        match = _MEMORY_RESULT_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        memory_type = _extract_memory_type(match.group("preview"))
        normalized = _normalize_memory_preview(match.group("preview"))
        if not normalized:
            continue
        try:
            score = float(match.group("score"))
        except Exception:
            score = 0.0
        ranked_items.append((_memory_priority(memory_type, normalized), score, normalized))

    if not ranked_items:
        return _format_grounded_fallback("Memory recall", memory_ctx, max_lines=8, max_chars=560)

    ranked_items.sort(key=lambda item: (item[0], -item[1]))
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, normalized in ranked_items:
        sentence = _to_speakable_memory_sentence(normalized)
        if not sentence:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= max_items:
            break

    if not selected:
        return _format_grounded_fallback("Memory recall", memory_ctx, max_lines=8, max_chars=560)

    total = total or len(ranked_items)
    parts = ["Here's what I remember from that time."]
    parts.extend(selected)
    if total > len(selected):
        parts.append("I can pull more details if you want.")
    return " ".join(parts)


def _record_language_corpus_example(
    seed: dict[str, Any] | None,
    *,
    query: str,
    reply: str,
    conversation_id: str,
    user_feedback: str,
) -> None:
    """Best-effort capture of grounded language examples for Phase A."""
    if not seed or not query or not reply or not conversation_id:
        return
    try:
        from reasoning.language_corpus import language_corpus
        meaning_frame = seed.get("meaning_frame")
        if not isinstance(meaning_frame, dict):
            meaning_frame = build_meaning_frame(
                response_class=str(seed.get("response_class", "")),
                grounding_payload=seed.get("grounding_payload"),
            ).to_dict()

        language_corpus.append_example(
            conversation_id=conversation_id,
            query=query,
            route=str(seed.get("route", "")),
            response_class=str(seed.get("response_class", "")),
            meaning_frame=meaning_frame,
            grounding_payload=seed.get("grounding_payload", ""),
            teacher_answer=str(seed.get("teacher_answer", "")),
            final_answer=reply,
            provenance_verdict=str(seed.get("provenance_verdict", "unknown")),
            user_feedback=user_feedback,
            confidence=float(seed.get("confidence", 1.0)),
            safety_flags=list(seed.get("safety_flags", [])),
        )
    except Exception:
        logger.warning("Language corpus capture failed", exc_info=True)


def _record_negative_corpus_example(
    *,
    query: str,
    original_reply: str,
    rewritten_reply: str,
    conversation_id: str,
    route: str,
    reason: str,
) -> None:
    """Capture a negative example when the capability gate rewrites or grounding fails."""
    if not query or not original_reply or not conversation_id:
        return
    try:
        from reasoning.language_corpus import language_corpus
        language_corpus.append_example(
            conversation_id=conversation_id,
            query=query,
            route=route,
            response_class="negative_example",
            meaning_frame={},
            grounding_payload="",
            teacher_answer=original_reply,
            final_answer=rewritten_reply,
            provenance_verdict=f"negative:{reason}",
            user_feedback="",
            confidence=0.0,
            safety_flags=["negative_example", reason],
        )
    except Exception:
        logger.debug("Negative corpus example capture failed", exc_info=True)


def _record_language_quality_event(
    seed: dict[str, Any] | None,
    *,
    query: str,
    reply: str,
    conversation_id: str,
    outcome: str,
    user_feedback: str,
) -> None:
    if not seed or not conversation_id:
        return
    try:
        from reasoning.language_telemetry import language_quality_telemetry

        meaning_frame = seed.get("meaning_frame")
        missing_reason = ""
        if isinstance(meaning_frame, dict):
            missing_reason = str(meaning_frame.get("missing_reason", "") or "")
        runtime_policy = seed.get("runtime_policy")
        if not isinstance(runtime_policy, dict):
            runtime_policy = None
        language_quality_telemetry.record_event(
            conversation_id=conversation_id,
            route=str(seed.get("route", "")),
            response_class=str(seed.get("response_class", "")),
            provenance_verdict=str(seed.get("provenance_verdict", "unknown")),
            outcome=outcome,
            user_feedback=user_feedback,
            confidence=float(seed.get("confidence", 0.0)),
            native_used=bool(seed.get("native_used", False)),
            fail_closed=bool(missing_reason),
            safety_flags=list(seed.get("safety_flags", [])),
            query=query,
            reply=reply,
            runtime_policy=runtime_policy,
        )
    except Exception:
        logger.warning("Language quality telemetry failed", exc_info=True)


def _build_ambiguous_intent_probe_seed(
    query: str,
    routing: RoutingResult | None,
) -> dict[str, Any] | None:
    """Build a shadow-only ambiguous-intent probe seed for telemetry.

    This path is observational only; it never changes routing.
    """
    if not query or routing is None:
        return None
    if not _AMBIGUOUS_SELF_READ_RE.search(query):
        return None
    selected_route = routing.tool.value if hasattr(routing, "tool") else "unknown"
    return {
        "query": query[:180],
        "selected_route": selected_route,
        "candidate_intent": "recent_research_or_processing",
        "candidate_confidence": 0.68 if selected_route == ToolType.NONE.value else 0.55,
        "trigger": "self_read_phrase",
        "shadow_only": True,
    }


def _record_ambiguous_intent_probe(
    seed: dict[str, Any] | None,
    *,
    conversation_id: str,
    outcome: str,
    user_feedback: str,
) -> None:
    if not seed or not conversation_id:
        return
    try:
        from reasoning.language_telemetry import language_quality_telemetry
        language_quality_telemetry.record_ambiguous_intent_probe(
            conversation_id=conversation_id,
            query=str(seed.get("query", "") or ""),
            selected_route=str(seed.get("selected_route", "") or "unknown"),
            candidate_intent=str(seed.get("candidate_intent", "") or "unknown"),
            candidate_confidence=float(seed.get("candidate_confidence", 0.0) or 0.0),
            trigger=str(seed.get("trigger", "") or ""),
            outcome=outcome,
            user_feedback=user_feedback,
            shadow_only=bool(seed.get("shadow_only", True)),
        )
    except Exception:
        logger.warning("Ambiguous-intent telemetry failed", exc_info=True)


def _is_system_explanation_query(text: str) -> bool:
    return bool(_SYSTEM_EXPLANATION_RE.search(text or ""))


def _is_capability_status_query(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    lower = raw.lower()
    if "capabilities are verified right now" in lower or "verified right now" in lower:
        return True
    normalized = re.sub(r"\b(?:currently|actually|right now|at the moment)\b", " ", raw, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return bool(_CAPABILITY_STATUS_QUERY_RE.search(raw) or _CAPABILITY_STATUS_QUERY_RE.search(normalized))


def _is_likely_fragment_noise(text: str, follow_up: bool) -> bool:
    if not follow_up:
        return False
    raw = (text or "").strip().lower()
    if not raw or "?" in raw:
        return False
    normalized = re.sub(r"[^a-z]", "", raw)
    if not normalized:
        return False
    return bool(_LIKELY_STT_FRAGMENT_RE.match(normalized))


def _is_personal_activity_recall_query(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    return bool(_PERSONAL_ACTIVITY_RECALL_RE.search(raw))


def _is_recent_research_query(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    return bool(_RECENT_RESEARCH_QUERY_RE.search(raw))


def _should_use_memory_search(
    text: str,
    *,
    extracted_args: dict[str, Any] | None = None,
) -> bool:
    if extracted_args and extracted_args.get("action") == "store":
        return False
    lower = (text or "").lower()
    if any(token in lower for token in ("search", "remember", "recall")):
        return True
    return _is_personal_activity_recall_query(text)


def _build_none_route_capability_payload() -> dict[str, Any]:
    verified: list[str] = []
    learning: list[str] = []
    try:
        from skills.registry import skill_registry as _skill_registry
        for rec in _skill_registry.get_all():
            name = getattr(rec, "name", "") or getattr(rec, "skill_id", "")
            if not name:
                continue
            status = str(getattr(rec, "status", ""))
            if status == "verified":
                verified.append(name)
            elif status == "learning":
                learning.append(name)
    except Exception:
        pass

    verified_preview = ", ".join(verified[:4])
    learning_preview = ", ".join(learning[:3])
    parts = [
        "I can report capabilities through deterministic tool routes.",
        "Core available paths include status and introspection, memory lookup, web and academic search, codebase lookup, and skill-learning workflows.",
    ]
    if verified_preview:
        parts.append(f"Currently verified skills include {verified_preview}.")
    else:
        parts.append("Verified capability evidence is tracked in the skill registry and may still be maturing.")
    if learning_preview:
        parts.append(f"Active learning includes {learning_preview}.")
    parts.append("I will not claim capabilities that are not verified.")

    return {
        "kind": "generic_fallback",
        "message": " ".join(parts),
        "status": "registry_first",
        "capability_type": "mixed",
        "skill_name": "general capabilities",
    }


def _build_identity_grounding_payload(
    *,
    is_identity_check: bool,
    check_name: str,
    voice_name: str,
    voice_conf: float,
    face_name: str,
    face_conf: float,
    enrolled_v: list[str],
    enrolled_f: list[str],
) -> dict[str, Any]:
    if is_identity_check and check_name:
        known_v = voice_name != "unknown" and voice_name.lower() == check_name.lower()
        known_f = face_name != "unknown" and face_name.lower() == check_name.lower()
        if known_v or known_f:
            matched_modalities: list[str] = []
            if known_v:
                matched_modalities.append(f"voice ({voice_conf*100:.0f}%)")
            if known_f:
                matched_modalities.append(f"face ({face_conf*100:.0f}%)")
            return {
                "kind": "identity_check_match",
                "check_name": check_name,
                "matched_modalities": matched_modalities,
            }
        if voice_name != "unknown" and voice_name.lower() != check_name.lower():
            return {
                "kind": "identity_check_mismatch",
                "check_name": check_name,
                "actual_name": voice_name,
                "actual_confidence": voice_conf,
            }
        if check_name.lower() in [v.lower() for v in enrolled_v]:
            return {
                "kind": "identity_check_enrolled_but_not_match",
                "check_name": check_name,
            }
        return {
            "kind": "identity_check_unknown_profile",
            "check_name": check_name,
        }

    if voice_name != "unknown":
        return {
            "kind": "current_voice",
            "name": voice_name,
            "confidence": voice_conf,
            "enrolled_voices": enrolled_v,
            "enrolled_faces": enrolled_f,
        }
    if face_name != "unknown":
        return {
            "kind": "current_face",
            "name": face_name,
            "confidence": face_conf,
            "enrolled_voices": enrolled_v,
            "enrolled_faces": enrolled_f,
        }
    return {
        "kind": "unknown_identity",
        "enrolled_voices": enrolled_v,
        "enrolled_faces": enrolled_f,
    }


def _reinforce_retrieval(conversation_id: str, positive: bool) -> None:
    """Reinforce study_claim memories and source quality based on conversation outcome.

    Positive (outcome=ok): small boost to memory weight + source quality.
    Negative (outcome=error/barge_in): small source quality penalty only;
    memory weight decays naturally.
    """
    try:
        from library.telemetry import retrieval_telemetry
        from library.source import source_store

        start = retrieval_telemetry.get_latest_start(conversation_id)
        if not start:
            return

        injected_ids = set(start.get("chunk_ids_injected", []))
        source_ids = set(start.get("source_ids", []))
        if not injected_ids and not source_ids:
            return

        if positive and injected_ids:
            claim_mems = [
                m for m in memory_storage.get_all()
                if "study_claim" in m.tags
                and isinstance(m.payload, dict)
                and m.payload.get("type") == "study_claim"
            ]
            for mem in claim_mems:
                mem_chunks = set(mem.payload.get("chunk_ids", []))
                if mem_chunks & injected_ids:
                    memory_storage.reinforce(mem.id, boost=_POSITIVE_MEMORY_BOOST)

        quality_delta = _POSITIVE_QUALITY_DELTA if positive else _NEGATIVE_QUALITY_DELTA
        for sid in source_ids:
            source_store.adjust_quality(sid, quality_delta)

    except Exception as exc:
        logger.debug("Retrieval reinforcement failed: %s", exc)


def _apply_inline_preferences(text: str, engine: ConsciousnessEngine) -> None:
    """Adjust engine response length hint for the current message."""
    if _BRIEF_SIGNALS.search(text):
        engine._policy_response_length = "brief"
    elif _DETAIL_SIGNALS.search(text):
        engine._policy_response_length = "detailed"


_DOI_PREF_TOKEN_RE = re.compile(r"\bdoi\b", re.I)


def _build_preference_instruction_ack(text: str, stored_count: int) -> str:
    """Return deterministic acknowledgement text for preference-instruction turns."""
    if _DOI_PREF_TOKEN_RE.search(text):
        if stored_count > 0:
            return (
                "Understood. Preference saved. DOI is omitted in research answers "
                "unless explicitly requested."
            )
        return (
            "Understood. Preference already stored. DOI is omitted in research "
            "answers unless explicitly requested."
        )
    if stored_count > 0:
        return "Understood. Preference saved and active for future responses."
    return "Understood. Preference already stored and still active."


# ---------------------------------------------------------------------------
# Personal intelligence extraction — builds a rich profile of the user
# ---------------------------------------------------------------------------

_INTEREST_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bi (?:really )?(?:like|love|enjoy|adore)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User enjoys {0}", "personal_interest"),
    (re.compile(r"\bi'?m (?:really )?(?:into|passionate about|a fan of|big on)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User is into {0}", "personal_interest"),
    (re.compile(r"\bmy (?:hobby|hobbies|passion) (?:is|are|include)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User's hobby: {0}", "personal_interest"),
    (re.compile(r"\bi (?:like to|love to|enjoy)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User likes to {0}", "personal_interest"),
]

_DISLIKE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bi (?:really )?(?:hate|dislike|can'?t stand|detest)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User dislikes {0}", "personal_dislike"),
    (re.compile(r"\bi'?m not (?:a fan of|into|big on)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User is not into {0}", "personal_dislike"),
]

_FACT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bi (?:work as|work in|am) (?:a |an )?(\w[\w\s]{2,40}?)(?:\.|,|!|$)", re.I),
     "User is {0}", "personal_fact"),
    (re.compile(r"\bi'?m from\s+(.{3,40}?)(?:\.|,|!|$)", re.I),
     "User is from {0}", "personal_fact"),
    (re.compile(r"\bi live in\s+(.{3,40}?)(?:\.|,|!|$)", re.I),
     "User lives in {0}", "personal_fact"),
    (re.compile(r"\bmy birthday (?:is|falls on)\s+([^.!\n]{3,60}?)(?:[.!?]|$)", re.I),
     "User's birthday is {0}", "personal_fact"),
    (re.compile(r"\bi was born on\s+([^.!\n]{3,60}?)(?:[.!?]|$)", re.I),
     "User's birthday is {0}", "personal_fact"),
    (re.compile(r"\bi have (\d+\s+(?:kid|child|son|daughter|dog|cat|pet|brother|sister)\w*)", re.I),
     "User has {0}", "personal_fact"),
    (re.compile(r"\bmy name is\s+(\w[\w\s]{1,30}?)(?:\.|,|!|$)", re.I),
     "User's name is {0}", "personal_fact"),
    (re.compile(r"\bcall me\s+(\w+)", re.I),
     "User wants to be called {0}", "personal_fact"),
    (re.compile(r"\bmy (?:wife|husband|partner|spouse)(?:'?s name)? is\s+(\w+)", re.I),
     "User's partner is {0}", "personal_fact"),
]

_PREFERENCE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bi prefer\b(.{5,60}?)(?:\.|,|!|$)", re.I),
     "User prefers {0}", "personal_preference"),
    (re.compile(r"\bmy favo(?:u)?rite (.{3,50}?) (?:is|are)\s+(.{3,50}?)(?:\.|,|!|$)", re.I),
     "User's favorite {0} is {1}", "personal_preference"),
    (re.compile(r"\bkeep it (short|brief|concise)\b", re.I),
     "User prefers concise responses", "response_style"),
    (re.compile(r"\b(?:be )?more (detailed|thorough|verbose)\b", re.I),
     "User prefers detailed responses", "response_style"),
    (re.compile(
        r"\b(?:don'?t|do not)\s+(?:use|say|do|include)\s+"
        r"(.{0,80}?\b(?:doi|dois|citation|citations|url|urls|link|links|response|responses|answer|answers)\b.{0,40}?)"
        r"(?:\.|,|!|$)",
        re.I,
    ),
     "User response format preference: omit {0}", "response_style"),
    (re.compile(
        r"\b(?:omit|skip|leave out|exclude)\s+"
        r"(.{0,80}?\b(?:doi|dois|citation|citations|url|urls|link|links|response|responses|answer|answers)\b.{0,40}?)"
        r"(?:\.|,|!|$)",
        re.I,
    ),
     "User response format preference: omit {0}", "response_style"),
    (re.compile(
        r"\b(?:always|by default)\b.{0,24}\b(?:include|show|provide)\b.{0,40}\b"
        r"(doi|dois|citation|citations|url|urls|link|links)\b",
        re.I,
    ),
     "User response format preference: include {0} by default", "response_style"),
    (re.compile(
        r"\b(?:include|show|provide)\b.{0,40}\b"
        r"(doi|dois|citation|citations|url|urls|link|links)\b.{0,24}\b(?:always|by default)\b",
        re.I,
    ),
     "User response format preference: include {0} by default", "response_style"),
    (re.compile(r"\bi (?:always|usually) (.{5,60}?)(?:\.|,|!|$)", re.I),
     "User habit: {0}", "personal_habit"),
]

_ROUTINE_PRIORITY_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bmy (?:typical )?morning routine (?:is|includes?)\s+(.{3,90}?)(?:\.|,|!|$)", re.I),
     "User morning routine: {0}", "routine_priority"),
    (re.compile(r"\bmy (?:daily|workday|work day) routine (?:is|includes?)\s+(.{3,90}?)(?:\.|,|!|$)", re.I),
     "User daily routine: {0}", "routine_priority"),
    (re.compile(r"\bmy work ?day is usually\s+(.{3,90}?)(?:\.|,|!|$)", re.I),
     "User workday pattern: {0}", "routine_priority"),
    (re.compile(r"\bwhen i(?:'m| am) focused\b[, ]+(.{3,90}?)(?:\.|,|!|$)", re.I),
     "User focus-window preference: {0}", "routine_priority"),
    (re.compile(r"\bwhen i(?:'m| am) (?:available|unavailable|busy)\b[, ]+(.{3,90}?)(?:\.|,|!|$)", re.I),
     "User availability pattern: {0}", "routine_priority"),
    (re.compile(r"\b(?:do not|don'?t|please don'?t)\s+interrupt me\b\s*(?:when|while|during)?\s*(.{3,90}?)(?:\.|,|!|$)", re.I),
     "User interrupt preference: do not interrupt during {0}", "routine_priority"),
    (re.compile(r"\bmy (?:top|main|current) priorit(?:y|ies)(?: right now)? (?:is|are)\s+(.{3,120}?)(?:\.|,|!|$)", re.I),
     "User current priorities: {0}", "routine_priority"),
    (re.compile(r"\bright now i(?:'m| am) focused on\s+(.{3,120}?)(?:\.|,|!|$)", re.I),
     "User current focus: {0}", "routine_priority"),
]

_RELATION = (
    r"(wife|husband|partner|girlfriend|boyfriend|spouse|"
    r"mom|mother|dad|father|brother|sister|son|daughter|"
    r"friend|roommate|boss|colleague|coworker)"
)

_THIRDPARTY_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(
        r"\bmy\s+" + _RELATION + r"\s+(?:really )?(?:likes?|loves?|enjoys?|adores?)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User's {0} likes {1}", "thirdparty_preference"),
    (re.compile(
        r"\bmy\s+" + _RELATION + r"(?:'s)?\s+(?:favo(?:u)?rite)\s+(.{3,50}?)(?:\s+(?:is|are)\s+(.{3,50}?))?(?:\.|,|!|$)", re.I),
     "User's {0}'s favorite {1}", "thirdparty_preference"),
    (re.compile(
        r"\bmy\s+" + _RELATION + r"\s+(?:really )?(?:hates?|dislikes?|can'?t stand|detests?)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User's {0} dislikes {1}", "thirdparty_preference"),
    (re.compile(
        r"\bmy\s+" + _RELATION + r"\s+(?:prefers?)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User's {0} prefers {1}", "thirdparty_preference"),
    (re.compile(
        r"\bmy\s+" + _RELATION + r"(?:'s)?\s+(?:name is|is called|goes by)\s+(\w[\w\s]{1,30}?)(?:\.|,|!|$)", re.I),
     "User's {0}'s name is {1}", "thirdparty_fact"),
    (re.compile(
        r"\bmy\s+" + _RELATION + r"(?:'s)?\s+birthday\s+(?:is|falls on)\s+([^.!\n]{3,60}?)(?:[.!?]|$)", re.I),
     "User's {0}'s birthday is {1}", "thirdparty_fact"),
    (re.compile(
        r"\bmy\s+" + _RELATION + r"\s+(?:is|works as|works in)\s+(?:a |an )?(.{3,40}?)(?:\.|,|!|$)", re.I),
     "User's {0} is {1}", "thirdparty_fact"),
    (re.compile(
        r"\bmy\s+" + _RELATION + r"(?:'s)?\s+(?:preferences?|interests?)\s+(?:is|are|include)\s+(.{3,60}?)(?:\.|,|!|$)", re.I),
     "User's {0}'s preferences: {1}", "thirdparty_preference"),
]

_RETIREMENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bi (?:stopped|quit|gave up|dropped)\s+(.{3,60}?)(?:\.|,|!|$)", re.I), "{0}"),
    (re.compile(r"\bi don'?t (.{3,60}?) (?:anymore|any more|now|these days)", re.I), "{0}"),
    (re.compile(r"\bi(?:'m| am) no(?:t| longer) (?:into|a fan of|doing)\s+(.{3,60}?)(?:\.|,|!|$)", re.I), "{0}"),
    (re.compile(r"\bi used to (.{3,60}?) but (?:i |not |don'?t|stopped|no)", re.I), "{0}"),
]

_CORRECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bthat(?:'s| is) (?:wrong|incorrect|not (?:true|right|correct))\b", re.I),
    re.compile(r"\bno[, ]+i(?:'m| am) not\b", re.I),
    re.compile(r"\bi(?:'m| am) not (?:a |an )?(\w[\w\s]{2,40}?)(?:\.|,|!|$)", re.I),
    re.compile(r"\bthat(?:'s| is) not what i said\b", re.I),
    re.compile(r"\bforget (?:that|what i said)\b", re.I),
]

_ALL_PERSONAL_PATTERNS = (
    _INTEREST_PATTERNS
    + _DISLIKE_PATTERNS
    + _FACT_PATTERNS
    + _PREFERENCE_PATTERNS
    + _ROUTINE_PRIORITY_PATTERNS
)

MAX_PREFERENCE_NOTES = 20


_EXPLICIT_CORE_MEMORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:hey\s+jarvis[, ]+)?(?:please\s+)?(?:remember|save|store|keep)\s+"
        r"(?:this|that|the following)\s+(?:as|for)\s+(?:a\s+)?(?:core|important)\s+memory[:,-]?\s*(?P<payload>.+?)\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:hey\s+jarvis[, ]+)?(?:please\s+)?(?:remember|save|store|keep)\s+"
        r"(?P<payload>.+?)\s+(?:as|for)\s+(?:a\s+)?(?:core|important)\s+memory[.?!]?\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:hey\s+jarvis[, ]+)?(?:please\s+)?(?:save|store)\s+"
        r"(?:this|that)\s+as\s+important[:,-]?\s*(?P<payload>.+?)\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:hey\s+jarvis[, ]+)?(?:i would like you to|i want you to|can you|could you|please)\s+"
        r"(?:save|store|remember)\s+(?:that\s+)?(?P<payload>.+?)[.?!]?\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:hey\s+jarvis[, ]+)?(?:save|store|remember)\s+that\s+(?P<payload>.+?)[.?!]?\s*$",
        re.I,
    ),
)


def _extract_explicit_core_memory_payload(text: str) -> str:
    """Extract the user-provided payload from explicit core-memory commands."""
    for pattern in _EXPLICIT_CORE_MEMORY_PATTERNS:
        match = pattern.match(text.strip())
        if not match:
            continue
        payload = " ".join((match.group("payload") or "").strip().split())
        return payload.rstrip(".,!?;: ")
    return ""


def _derive_personal_memory_metadata(payload: str, category: str) -> tuple[str, list[str]]:
    """Normalize payloads and attach structured tags for durable personal facts."""
    cleaned = " ".join(str(payload).strip().split()).rstrip(".,!?;: ")
    if not cleaned:
        return "", []

    lower = cleaned.lower()
    tags = ["schema:personal_memory_v2"]

    if category == "personal_fact":
        if "birthday is" in lower:
            tags.extend(["fact_kind:birthday", "high_confidence_fact"])
        elif "wants to be called" in lower:
            tags.extend(["fact_kind:preferred_name", "high_confidence_fact"])
        elif "name is" in lower:
            tags.extend(["fact_kind:name", "high_confidence_fact"])
        elif "partner is" in lower:
            tags.append("fact_kind:relationship_role")
        elif lower.startswith("user lives in "):
            tags.append("fact_kind:location")
        elif lower.startswith("user is from "):
            tags.append("fact_kind:origin")
        elif lower.startswith("user is "):
            tags.append("fact_kind:biographical")
    elif category == "personal_preference":
        tags.append("preference_kind:stable")
        if "favorite " in lower:
            tags.append("preference_kind:favorite")
    elif category == "response_style":
        tags.extend(["preference_kind:response_style", "high_confidence_fact"])
    elif category == "personal_interest":
        tags.append("interest_kind:positive")
    elif category == "personal_dislike":
        tags.append("interest_kind:negative")
    elif category == "personal_habit":
        tags.append("fact_kind:habit")
    elif category == "routine_priority":
        tags.extend(["routine", "schedule", "daily", "preference_kind:routine_priority"])
        if any(term in lower for term in ("priority", "priorities", "focused on", "current focus")):
            tags.append("priority")
        if "interrupt" in lower:
            tags.append("interrupt_preference")
        if any(term in lower for term in ("available", "unavailable", "busy", "focus-window")):
            tags.append("availability")
    elif category == "thirdparty_fact":
        tags.append("subject_kind:thirdparty")
        if "birthday is" in lower:
            tags.extend(["fact_kind:thirdparty_birthday", "high_confidence_fact"])
        elif "name is" in lower:
            tags.extend(["fact_kind:thirdparty_name", "high_confidence_fact"])
        else:
            tags.append("fact_kind:thirdparty_biographical")
    elif category == "thirdparty_preference":
        tags.extend(["subject_kind:thirdparty", "preference_kind:thirdparty"])
    elif category == "former_interest":
        tags.append("interest_kind:retired")

    return cleaned, tags


def _build_user_claim_identity_kwargs(
    payload: str,
    speaker: str,
    memory_type: str = "user_preference",
) -> dict[str, Any]:
    """Resolve owner/subject identity scope for user-provided memories."""
    identity_kwargs: dict[str, Any] = {}
    try:
        from identity.resolver import identity_resolver

        ctx = identity_resolver.resolve_for_memory(
            provenance="user_claim",
            speaker=speaker,
        )
        scope = identity_resolver.build_scope(ctx, payload, memory_type)
        identity_kwargs = {
            "identity_owner": scope.owner_id,
            "identity_owner_type": scope.owner_type,
            "identity_subject": scope.subject_id,
            "identity_subject_type": scope.subject_type,
            "identity_scope_key": scope.scope_key,
            "identity_confidence": scope.confidence,
            "identity_needs_resolution": scope.needs_resolution,
        }
        try:
            from identity.audit import identity_audit

            identity_audit.record_scope_assigned("", scope, scope.confidence)
        except Exception:
            pass
    except Exception:
        pass
    return identity_kwargs


def _looks_like_negative_feedback(text: str) -> bool:
    """Heuristic negative/correction signal for post-response outcome handling."""
    lower = text.lower()
    if any(w in lower for w in ("wrong", "no that's not", "incorrect", "bad", "error", "mistake")):
        return True
    try:
        from epistemic.calibration.correction_detector import _has_correction_phrase
        return _has_correction_phrase(text)
    except Exception:
        return False


def _store_user_correction_memory(
    *,
    user_text: str,
    last_response_text: str,
    route: str,
    speaker: str,
    correction_kind: str = "factual_mismatch",
    authority_domain: str = "objective_or_external_fact",
    adjudication_policy: str = "require_evidence",
    injected_memory_payloads: list[str] | None = None,
) -> str:
    """Persist a user correction as first-class evidence for later retrieval/audit."""
    if not user_text.strip() or not last_response_text.strip():
        return ""
    try:
        from memory.core import canonical_remember

        tags = [
            "user_correction",
            "assistant_error",
            "interactive",
            "schema:user_correction_v1",
            f"correction_kind:{correction_kind or 'unknown'}",
            f"authority_domain:{authority_domain or 'unknown'}",
            f"adjudication_policy:{adjudication_policy or 'unknown'}",
            f"route:{route or 'unknown'}",
        ]
        if speaker and speaker != "unknown":
            tags.append(f"speaker:{speaker.lower().strip()}")

        payload_lines = [
            "User corrected a Jarvis response.",
            f"Prior response: {last_response_text[:280]}",
            f"Correction: {user_text[:280]}",
            f"Authority domain: {authority_domain or 'unknown'}",
            f"Adjudication policy: {adjudication_policy or 'unknown'}",
        ]
        snippets = [p for p in (injected_memory_payloads or []) if p][:2]
        if snippets:
            payload_lines.append(
                "Retrieved context during the bad response: " + " | ".join(s[:140] for s in snippets)
            )

        identity_kwargs = _build_user_claim_identity_kwargs(user_text, speaker, memory_type="conversation")
        mem = canonical_remember(CreateMemoryData(
            type="conversation",
            payload="\n".join(payload_lines),
            weight=0.72,
            tags=list(dict.fromkeys(tags)),
            provenance="user_claim",
            **identity_kwargs,
        ))
        return getattr(mem, "id", "") if mem else ""
    except Exception:
        logger.debug("Failed to persist user correction memory", exc_info=True)
    return ""


def _mark_corrected_memory(
    memory_id: str,
    *,
    correction_kind: str,
    authority_domain: str,
    auto_accept_user_correction: bool,
    correction_memory_id: str = "",
) -> bool:
    """Mark a memory as corrected or pending adjudication."""
    try:
        mem = memory_storage.get(memory_id)
        if mem is None:
            return False

        new_tags = set(mem.tags)
        new_tags.update({
            "assistant_error_source",
            f"correction_kind:{correction_kind or 'unknown'}",
            f"authority_domain:{authority_domain or 'unknown'}",
        })
        if auto_accept_user_correction:
            new_tags.add("user_corrected")
        else:
            new_tags.add("correction_pending_review")
        updated = replace(mem, tags=tuple(sorted(new_tags)))
        if not memory_storage.add(updated):
            return False

        if auto_accept_user_correction and correction_kind != "identity_scope_leak":
            memory_storage.downweight(memory_id, weight_factor=0.7, decay_rate_factor=2.0)

        if correction_memory_id:
            memory_storage.associate(memory_id, correction_memory_id)
        return True
    except Exception:
        logger.debug("Failed to mark corrected memory %s", memory_id, exc_info=True)
        return False


def _is_unstable_personal_fact(payload: str, category: str) -> bool:
    if category != "personal_fact":
        return False
    lower = str(payload).strip().lower()
    if lower.startswith("user is not "):
        return True
    # Guard: "User is <transient-state/action/system-term>" are not biographical facts.
    # Reuse the comprehensive blocklist from identity/name_validator.
    _prefix = "user is "
    if lower.startswith(_prefix):
        captured = lower[len(_prefix):].split()[0] if lower[len(_prefix):].split() else ""
        try:
            from identity.name_validator import _BLOCKED_WORDS
            if captured in _BLOCKED_WORDS:
                return True
        except ImportError:
            pass
        # Block gerunds ("pushing", "sitting", "feeling") captured as facts
        if captured.endswith("ing") and len(captured) > 4:
            return True
        # Block self-references ("jarvis", "the brain", "an ai")
        if captured in ("jarvis", "ai", "bot", "assistant", "robot"):
            return True
    return False


def _collect_personal_intel_matches(
    text: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Collect personal and third-party fact candidates without writing them."""
    personal: list[tuple[str, str]] = []
    thirdparty: list[tuple[str, str, str]] = []

    for pattern, template, category in _ALL_PERSONAL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) >= 2 and "{1}" in template:
            payload = template.format(groups[0].strip(), groups[1].strip())
        elif groups:
            payload = template.format(groups[0].strip())
        else:
            payload = template.format(match.group(0).strip())
        if _is_unstable_personal_fact(payload, category):
            continue
        personal.append((payload, category))

    for pattern, template, category in _THIRDPARTY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        relation = groups[0].strip().lower() if groups else ""
        if not relation:
            continue
        if len(groups) >= 3 and groups[2] and "{2}" in template:
            payload = template.format(relation, groups[1].strip(), groups[2].strip())
        elif len(groups) >= 2 and groups[1]:
            payload = template.format(relation, groups[1].strip())
        else:
            payload = template.format(relation, match.group(0).strip())
        thirdparty.append((payload, category, relation))

    return personal, thirdparty


def _store_personal_memory(
    payload: str,
    category: str,
    speaker: str,
    *,
    extra_tags: list[str] | None = None,
) -> bool:
    """Create a user_preference memory through the unified write path."""
    payload, metadata_tags = _derive_personal_memory_metadata(payload, category)
    if not payload:
        return False

    tags = ["user_preference", category, *metadata_tags]
    if extra_tags:
        tags.extend(extra_tags)
    if speaker and speaker != "unknown":
        tags.append(f"speaker:{speaker.lower().strip()}")

    existing = memory_storage.get_by_tag("user_preference")
    payload_lower = payload.lower()
    for m in existing:
        if isinstance(m.payload, str) and payload_lower in m.payload.lower():
            return False
        if isinstance(m.payload, str) and m.payload.lower() in payload_lower:
            return False

    weight = 0.70 if category in ("personal_interest", "personal_fact") else 0.65
    if extra_tags and "explicit_core_memory" in extra_tags:
        weight = max(weight, 0.8 if category == "personal_fact" else 0.75)
    identity_kwargs = _build_user_claim_identity_kwargs(payload, speaker, memory_type="user_preference")
    from memory.core import canonical_remember
    mem = canonical_remember(CreateMemoryData(
        type="user_preference",
        payload=payload,
        weight=weight,
        tags=tags,
        provenance="user_claim",
        **identity_kwargs,
    ))
    if mem:
        logger.info("Stored personal intel [%s]: %s", category, payload)
        return True
    return False


def _update_relationship(speaker: str, payload: str, category: str) -> None:
    """Write personal facts and interests to the Relationship record in soul."""
    if not speaker or speaker == "unknown":
        return
    from identity.name_validator import is_valid_person_name
    if not is_valid_person_name(speaker):
        return
    rel = soul_service.identity.get_relationship(speaker)
    if category in ("personal_interest", "personal_dislike", "personal_habit", "routine_priority"):
        if payload not in rel.notes and len(rel.notes) < MAX_PREFERENCE_NOTES:
            rel.notes.append(payload)
    elif category == "personal_fact":
        key = payload.split(":")[0].strip() if ":" in payload else payload[:30]
        rel.preferences[key] = payload
    elif category == "personal_preference":
        key = payload.split(":")[0].strip() if ":" in payload else payload[:30]
        rel.preferences[key] = payload


def _retire_matching_preferences(text: str, speaker: str) -> None:
    """Detect 'I stopped X / I don't X anymore' and downweight matching memories."""
    for pattern, template in _RETIREMENT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        retired_topic = template.format(match.group(1).strip()).lower()
        words = [w for w in retired_topic.split() if len(w) > 2]
        if not words:
            continue

        existing = memory_storage.get_by_tag("user_preference")
        retired_any = False
        for m in existing:
            if not isinstance(m.payload, str):
                continue
            payload_lower = m.payload.lower()
            hits = sum(1 for w in words if w in payload_lower)
            if hits < max(1, len(words) // 2):
                continue
            if "former" in m.tags:
                continue

            updated = replace(
                m,
                weight=max(0.1, m.weight * 0.15),
                tags=tuple(sorted(set(m.tags) | {"former"})),
            )
            memory_storage.add(updated)
            retired_any = True
            logger.info("Retired preference (weight→%.2f): %s", updated.weight, m.payload[:60])

        if retired_any:
            historical = f"User used to {match.group(1).strip()} but no longer does"
            _store_personal_memory(historical, "former_interest", speaker)
            if speaker and speaker != "unknown":
                rel = soul_service.identity.get_relationship(speaker)
                if historical not in rel.notes and len(rel.notes) < MAX_PREFERENCE_NOTES:
                    rel.notes.append(historical)
        return


def _correct_recent_facts(text: str) -> None:
    """Detect user corrections ("that's wrong", "no I'm not X") and downweight
    recently captured user_preference memories that might be wrong."""
    is_correction = any(p.search(text) for p in _CORRECTION_PATTERNS)
    if not is_correction:
        return

    recent_prefs = memory_storage.get_by_tag("user_preference")
    if not recent_prefs:
        return

    now = time.time()
    _RECENT_WINDOW_S = 300.0  # 5 min — corrections likely target recent captures
    corrected_any = False
    for m in recent_prefs:
        age = now - m.timestamp
        if age > _RECENT_WINDOW_S:
            continue
        if not isinstance(m.payload, str):
            continue
        updated = replace(
            m,
            weight=max(0.05, m.weight * 0.1),
            tags=tuple(sorted(set(m.tags) | {"corrected"})),
        )
        memory_storage.add(updated)
        corrected_any = True
        logger.info("Correction downweighted recent fact (weight→%.2f): %s",
                     updated.weight, m.payload[:60])

    if corrected_any:
        logger.info("User correction detected — downweighted recent preference memories")


_NAME_FROM_PAYLOAD_RE = re.compile(
    r"name is\s+(\w[\w\s]{1,30}?)$", re.IGNORECASE,
)


def _try_set_relationship_name(payload: str, relation: str, speaker: str) -> None:
    """When the user says "My wife's name is Sarah", register the role on the Relationship."""
    try:
        m = _NAME_FROM_PAYLOAD_RE.search(payload)
        if not m:
            return
        person_name = m.group(1).strip()
        if not person_name:
            return
        from identity.resolver import identity_resolver
        if identity_resolver.set_relationship_role(person_name, relation):
            logger.info("Registered relationship: %s = %s (via %s)", person_name, relation, speaker)
    except Exception:
        pass


def _store_thirdparty_memory(
    payload: str,
    category: str,
    speaker: str,
    relation: str,
    *,
    extra_tags: list[str] | None = None,
) -> bool:
    """Create a user_preference memory for a third-party referenced by the user."""
    payload, metadata_tags = _derive_personal_memory_metadata(payload, category)
    if not payload:
        return False

    tags = ["user_preference", category, f"relation:{relation}", *metadata_tags]
    if extra_tags:
        tags.extend(extra_tags)
    if speaker and speaker != "unknown":
        tags.append(f"speaker:{speaker.lower().strip()}")

    existing = memory_storage.get_by_tag("user_preference")
    payload_lower = payload.lower()
    for m in existing:
        if isinstance(m.payload, str) and payload_lower in m.payload.lower():
            return False
        if isinstance(m.payload, str) and m.payload.lower() in payload_lower:
            return False

    identity_kwargs: dict[str, Any] = {}
    try:
        from identity.resolver import identity_resolver
        ctx = identity_resolver.resolve_for_memory(
            provenance="user_claim", speaker=speaker,
        )
        scope = identity_resolver.build_scope(ctx, payload, "user_preference")

        resolved_subject = scope.subject_id
        needs_res = scope.needs_resolution
        if not resolved_subject or resolved_subject == scope.owner_id:
            resolved_subject = f"_rel_{relation}"
            needs_res = True

        identity_kwargs = {
            "identity_owner": scope.owner_id,
            "identity_owner_type": scope.owner_type,
            "identity_subject": resolved_subject,
            "identity_subject_type": "known_human",
            "identity_scope_key": scope.scope_key,
            "identity_confidence": scope.confidence,
            "identity_needs_resolution": needs_res,
        }

        if category == "thirdparty_fact" and "name is" in payload.lower():
            _try_set_relationship_name(payload, relation, speaker)

        try:
            from identity.audit import identity_audit
            identity_audit.record_scope_assigned("", scope, scope.confidence)
        except Exception:
            pass
    except Exception:
        pass
    from memory.core import canonical_remember
    mem = canonical_remember(CreateMemoryData(
        type="user_preference",
        payload=payload,
        weight=0.75 if extra_tags and "explicit_core_memory" in extra_tags else 0.65,
        tags=tags,
        provenance="user_claim",
        **identity_kwargs,
    ))
    if mem:
        logger.info("Stored thirdparty intel [%s/%s]: %s", relation, category, payload)
        return True
    return False


def _summarize_explicit_memory_capture(
    personal_categories: list[str],
    thirdparty_categories: list[str],
    payload: str,
) -> str:
    """Return a factual acknowledgment for explicit core-memory capture."""
    labels: list[str] = []
    payload_lower = payload.lower()
    all_categories = personal_categories + thirdparty_categories
    if "birthday" in payload_lower or any("birthday" in cat for cat in all_categories):
        labels.append("birthday")
    elif thirdparty_categories:
        labels.append("relationship fact")
    elif "personal_preference" in all_categories:
        labels.append("personal preference")
    elif "personal_fact" in all_categories:
        labels.append("personal fact")

    if labels:
        return f"Stored that as a core memory ({labels[0]})."
    return "Stored that as a core memory."


def _store_explicit_core_memory(text: str, speaker: str) -> tuple[bool, str]:
    """Store an explicit user-requested core memory with deterministic confirmation."""
    payload = _extract_explicit_core_memory_payload(text)
    if not payload:
        return False, "Tell me what you'd like me to store as a core memory."

    personal, thirdparty = _collect_personal_intel_matches(payload)
    personal_categories: list[str] = []
    thirdparty_categories: list[str] = []

    for personal_payload, category in personal:
        stored = _store_personal_memory(
            personal_payload,
            category,
            speaker,
            extra_tags=["core_memory", "explicit_core_memory", "manual_capture"],
        )
        if stored:
            _update_relationship(speaker, personal_payload, category)
            personal_categories.append(category)

    for third_payload, category, relation in thirdparty:
        stored = _store_thirdparty_memory(
            third_payload,
            category,
            speaker,
            relation,
            extra_tags=["core_memory", "explicit_core_memory", "manual_capture"],
        )
        if stored:
            thirdparty_categories.append(category)

    if personal_categories or thirdparty_categories:
        return True, _summarize_explicit_memory_capture(
            personal_categories,
            thirdparty_categories,
            payload,
        )

    identity_kwargs = _build_user_claim_identity_kwargs(payload, speaker, memory_type="user_preference")
    tags = ["core", "core_memory", "explicit_core_memory", "manual_capture", "schema:core_memory_v1"]
    if speaker and speaker != "unknown":
        tags.append(f"speaker:{speaker.lower().strip()}")

    from memory.core import canonical_remember

    mem = canonical_remember(CreateMemoryData(
        type="core",
        payload=payload,
        weight=0.85,
        tags=tags,
        provenance="user_claim",
        **identity_kwargs,
    ))
    if mem:
        logger.info("Stored explicit core memory: %s", payload)
        return True, "Stored that as a core memory."
    return False, "I couldn't store that as a core memory."


def _process_curiosity_answer(
    text: str,
    ctx: dict[str, Any],
    engine: Any,
    speaker: str = "unknown",
) -> None:
    """Route user's answer to a curiosity question back to the originating subsystem.

    Classifies the user's response as engaged/dismissed/annoyed and feeds
    the outcome back to the curiosity buffer for adaptive cooldown learning.
    """
    source = ctx.get("source", "")
    evidence = ctx.get("evidence", "")
    question = ctx.get("question", "")

    try:
        from personality.curiosity_questions import (
            classify_curiosity_outcome,
            curiosity_buffer,
            infer_curiosity_topic_tags,
        )

        outcome = classify_curiosity_outcome(text)

        from memory.core import CreateMemoryData, canonical_remember

        tags = [
            "curiosity_answer",
            f"curiosity_{source}",
            "interactive",
            f"outcome:{outcome}",
            *infer_curiosity_topic_tags(source, question=question, evidence=evidence),
        ]
        canonical_remember(CreateMemoryData(
            type="conversation",
            payload=f"Curiosity Q ({source}): {question}\nUser answer: {text[:500]}",
            weight=0.55 if outcome == "engaged" else 0.40,
            tags=list(dict.fromkeys(tags)),
            provenance="conversation",
        ))

        curiosity_buffer.record_outcome(source, question, text[:200], outcome)

        if source == "identity" and outcome == "engaged":
            _try_identity_enrollment_from_answer(text, engine)

        logger.info("Curiosity answer processed: source=%s, outcome=%s, answer=%s",
                     source, outcome, text[:80])
    except Exception:
        logger.debug("Curiosity answer processing error", exc_info=True)


def _try_identity_enrollment_from_answer(text: str, engine: Any) -> None:
    """If the user answered an identity curiosity question with a name, offer enrollment."""
    import re
    name_match = re.search(
        r"(?:that(?:'s| is|was)\s+|name(?:'s| is)\s+|it(?:'s| is)\s+|call(?:ed)?\s+)(\b[A-Z]\w{2,}\b)",
        text, re.IGNORECASE,
    )
    if not name_match:
        return
    name = name_match.group(1).strip().title()
    try:
        from identity.name_validator import is_valid_person_name
        if not is_valid_person_name(name):
            return
    except ImportError:
        return
    logger.info("Curiosity answer: detected name '%s', enrollment available via voice command", name)


def _extract_personal_intel(
    text: str,
    speaker: str = "unknown",
    *,
    suppress_write: bool = False,
) -> dict[str, Any]:
    """Scan user message for personal information and build a HUMINT profile.

    Captures interests, dislikes, biographical facts, preferences, and habits.
    Also detects retired preferences ('I stopped fishing') and downweights them.
    All findings are tied to the speaker's Relationship record.
    """
    if not suppress_write:
        _retire_matching_preferences(text, speaker)
        _correct_recent_facts(text)

    personal, thirdparty = _collect_personal_intel_matches(text)
    personal_categories = sorted({category for _, category in personal})
    thirdparty_categories = sorted({category for _, category, _ in thirdparty})
    if suppress_write:
        return {
            "personal_matches": len(personal),
            "thirdparty_matches": len(thirdparty),
            "stored": 0,
            "personal_categories": personal_categories,
            "thirdparty_categories": thirdparty_categories,
            "stored_categories": [],
        }

    stored = 0
    stored_categories: list[str] = []
    for payload, category in personal:
        if _store_personal_memory(payload, category, speaker):
            _update_relationship(speaker, payload, category)
            stored += 1
            stored_categories.append(category)

    for payload, category, relation in thirdparty:
        if _store_thirdparty_memory(payload, category, speaker, relation):
            stored += 1
            stored_categories.append(category)

    return {
        "personal_matches": len(personal),
        "thirdparty_matches": len(thirdparty),
        "stored": stored,
        "personal_categories": personal_categories,
        "thirdparty_categories": thirdparty_categories,
        "stored_categories": sorted(set(stored_categories)),
    }


_TARGET_HINTS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(memory|memories|recall|remember)\b", re.I),
     "brain/memory/", "performance_optimization"),
    (re.compile(r"\b(neural net|network|hemisphere|nn)\b", re.I),
     "brain/hemisphere/", "consciousness_enhancement"),
    (re.compile(r"\b(policy|decision|shadow)\b", re.I),
     "brain/policy/", "performance_optimization"),
    (re.compile(r"\b(conscious|awareness|kernel|tick|mutation)\b", re.I),
     "brain/consciousness/", "consciousness_enhancement"),
    (re.compile(r"\b(personality|trait|tone|mood)\b", re.I),
     "brain/personality/", "consciousness_enhancement"),
    (re.compile(r"\b(reason|response|context|prompt)\b", re.I),
     "brain/reasoning/", "performance_optimization"),
    (re.compile(r"\b(perception|audio|vision|stt|tts|wake)\b", re.I),
     "brain/perception/", "performance_optimization"),
]


def _infer_improvement_target(text: str) -> tuple[str, str]:
    """Best-effort guess at which subsystem the user wants improved."""
    for pattern, module, imp_type in _TARGET_HINTS:
        if pattern.search(text):
            return module, imp_type
    return "", "consciousness_enhancement"


_ENROLL_NAME_RE = re.compile(
    r"(?:my name is|call me|enroll me as|register me as|i am called|you can call me)"
    r"\s+(?:by\s+)?(?:the\s+name\s+(?:of\s+)?)?"
    r"(\b[A-Z]?\w{2,}\b)",
    re.IGNORECASE,
)
_ENROLL_NAME_WEAK_RE = re.compile(
    r"(?:[Ii]'m|[Ii] am|[Tt]his is)"
    r"\s+([A-Z][a-z]{1,})\b",
)
_ENROLL_NAME_INVERTED_RE = re.compile(
    r"\b([A-Z][a-z]{1,})\s+is\s+my\s+name\b",
)

_IDENTITY_QUERY_RE = re.compile(
    r"\b(who am i|do you (?:know|recognize) (?:me|who)|who(?:'s| is) (?:speaking|talking))\b",
    re.IGNORECASE,
)

_IDENTITY_CHECK_RE = re.compile(
    r"\b(?:[Ii]s (?:this|that)|[Aa]m [Ii]|[Aa]re you talking to)\s+(?:the\s+)?([A-Z]\w{2,})\b",
)


def _enrollment_blocked(name: str, identity_callback) -> bool:
    """Block enrollment if name is already enrolled and current voice doesn't match.

    Prevents a different person from overwriting an existing voice profile.
    First-time enrollment (name not yet known) is always allowed.
    """
    if not identity_callback:
        return False
    try:
        status = identity_callback()
        enrolled = [v.lower() for v in status.get("enrolled_voices", [])]
        if name.lower() not in enrolled:
            return False
        current_voice = status.get("current_voice", {})
        cv_name = current_voice.get("name", "unknown")
        if cv_name.lower() == name.lower():
            return False
        return True
    except Exception:
        return False


async def handle_transcription(
    text: str,
    engine: ConsciousnessEngine,
    response_gen: ResponseGenerator,
    claude: ClaudeClient | None,
    perception: PerceptionServer | None,
    episodes: EpisodicMemory | None = None,
    speaker_state: dict | None = None,
    emotion_state: dict | None = None,
    conversation_id: str = "",
    cancel_flag: dict | None = None,
    ollama: OllamaClient | None = None,
    pi_snapshot_url: str = "",
    brain_tts: BrainTTS | None = None,
    scene_context: str = "",
    follow_up: bool = False,
    enroll_callback=None,
    identity_callback=None,
) -> None:
    import time as _time
    import re as _re_tts_est
    from consciousness.operations import ops_tracker
    _trace_ctx = build_trace_context(conversation_id)
    _conv_start = _time.time()
    _conv_mono_start = _time.monotonic()
    _first_sentence_sent = False
    # Intention infrastructure Stage 0: per-turn backing-job registry hooks.
    # Populated by route dispatches that spawn real background work
    # (LIBRARY_INGEST ingest thread, future async research handles, etc).
    # Read by _gate_text to decide whether commitment phrases in the
    # outgoing response are backed by a real job. Stays empty for
    # synchronous routes — commitments on those routes are confabulation.
    _backing_job_ids: list[str] = []
    _intention_registered_turn: dict[str, bool] = {"done": False}
    speaker = (speaker_state or {}).get("name", "unknown")
    _emo = emotion_state or {}
    emotion = _emo.get("emotion", "neutral") if _emo.get("trusted", False) else "neutral"
    conv_tag = f" [conv:{conversation_id[:8]}]" if conversation_id else ""
    _web_selection_actor = _web_selection_actor_key(speaker, conversation_id)
    _prune_pending_web_selections()
    print(f"\n  [Pi] User said: {text}" + (f" [{speaker}]" if speaker != "unknown" else "") + conv_tag)
    ops_tracker.begin_activity("conversation", phase="processing", trigger="user_speech",
                               detail=f"\"{text[:60]}\"")
    ops_tracker.set_subsystem("reasoning", "processing", f"routing: \"{text[:40]}\"")
    ops_tracker.log_event("reasoning", "conversation_started", f"User: \"{text[:60]}\"")

    _prev_user_text = getattr(engine, "_last_user_text", "")
    engine._last_user_text = text

    try:
        from skills.capability_gate import capability_gate as _cg
        _has_perception = bool(scene_context) or speaker != "unknown"
        _cg.set_perception_evidence(_has_perception)
    except Exception:
        pass

    def _cancelled() -> bool:
        if not cancel_flag:
            return False
        if cancel_flag.get("id") != conversation_id:
            return False
        return bool(cancel_flag.get("cancelled"))

    def _broadcast(msg: dict) -> None:
        if conversation_id:
            msg["conversation_id"] = conversation_id
        if not perception:
            return
        has_audio = msg.get("data", {}).get("audio_b64")
        if has_audio:
            perception.broadcast_audio(msg)
        else:
            perception.broadcast(msg)

    _tts_queue: asyncio.Queue[tuple[str, str, str, Any] | None] = asyncio.Queue()
    _tts_done = asyncio.Event()
    _style_intent = detect_style_intent(text)
    _style_instruction = _style_intent.prompt_instruction if _style_intent else ""

    _SYNC_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

    _gate_rewrite_buffer: list[tuple[str, str]] = []

    def _gate_text(text_str: str) -> str:
        """Apply capability gate to sanitize text before printing or broadcasting.

        Fail-closed: on error, strip any first-person capability claims rather
        than returning ungated text.

        Stage 0 intention hook: after the capability gate runs, we evaluate
        outgoing text for commitment phrases ("I'll get back to you",
        "give me a moment", "I've begun ...") against the turn's
        ``_backing_job_ids``. Backed commitments register an
        ``IntentionRecord``; unbacked commitments are rewritten. This closes
        the 20:31:20 confabulation class at the source.
        """
        try:
            from skills.capability_gate import capability_gate
            gated = capability_gate.check_text(text_str)
            if gated != text_str:
                _gate_rewrite_buffer.append((text_str, gated))
        except Exception:
            import re as _re
            _fallback_re = _re.compile(
                r"\b(?:I (?:can|could|will|'ll|'m able to|'m going to)"
                r"|[Ww]e (?:can|could|'ll|will)"
                r"|[Jj]arvis (?:can|could|will)"
                r"|(?:let me|shall I|want me to|I'll)"
                r") .{3,80}?[.!?\n]",
                _re.IGNORECASE,
            )
            return _fallback_re.sub("I don't have that capability yet.", text_str)

        try:
            from skills.capability_gate import capability_gate
            from cognition.commitment_extractor import extract_commitments
            from cognition.intention_registry import intention_registry

            route_hint = getattr(capability_gate, "_route_hint", None)
            rewritten, changed = capability_gate.evaluate_commitment(
                gated, list(_backing_job_ids), route=route_hint,
            )
            if changed:
                _gate_rewrite_buffer.append((gated, rewritten))
                gated = rewritten
            elif _backing_job_ids and not _intention_registered_turn["done"]:
                matches = extract_commitments(gated)
                if matches:
                    cm = matches[0]
                    intention_registry.register(
                        utterance=text[:500] if isinstance(text, str) else "",
                        commitment_phrase=cm.phrase,
                        commitment_type=cm.commitment_type,
                        backing_job_id=_backing_job_ids[0],
                        backing_job_kind=cm.suggested_backing_kind,
                        turn_id=_trace_ctx.trace_id or _trace_ctx.conversation_id,
                        speaker_id=speaker or "",
                        conversation_id=_trace_ctx.conversation_id,
                        provenance="regex_bootstrap",
                    )
                    _intention_registered_turn["done"] = True
        except Exception:
            logger.debug("Intention hook skipped (non-critical)", exc_info=True)

        return gated

    _echo_ref_buf: list[str] = []
    _playback_estimate_s = 0.0

    def _estimate_tts_playback_s(text_str: str, speed_override: float | None = None) -> float:
        """Estimate spoken playback duration conservatively.

        This is used to size the speaking safety timeout so long, valid TTS
        responses do not drop back into listening while audio is still playing.
        """
        if not text_str.strip():
            return 0.0
        try:
            cleaned = BrainTTS._clean_for_speech(text_str)
        except Exception:
            cleaned = text_str.strip()
        words = _re_tts_est.findall(r"\b[\w']+\b", cleaned)
        word_count = len(words)
        if word_count == 0:
            return 0.0
        speed = max(0.5, float(speed_override if speed_override is not None else getattr(brain_tts, "_speed", 1.0) or 1.0))
        base_s = (word_count / 2.6) / speed
        pause_s = (
            cleaned.count(".") * 0.32
            + cleaned.count("!") * 0.32
            + cleaned.count("?") * 0.32
            + cleaned.count(";") * 0.20
            + cleaned.count(":") * 0.20
            + cleaned.count(",") * 0.08
        )
        return base_s + pause_s + 0.35

    def _tts_profile_for(tone_str: str, *, proactive: bool = False) -> Any | None:
        if not brain_tts or not brain_tts.available:
            return None
        try:
            return brain_tts.get_voice_profile(
                tone=tone_str or "professional",
                user_emotion=emotion,
                emotion_trusted=bool(_emo.get("trusted", False)),
                speaker_name=speaker,
                proactive=proactive,
                style_override=_style_intent.voice_profile_id if _style_intent else "",
            )
        except Exception as exc:
            logger.warning(
                "Voice profile resolution failed; falling back to oracle_solemn "
                "(tone=%s emotion=%s trusted=%s speaker=%s style_override=%s proactive=%s error=%s)",
                tone_str or "professional",
                emotion,
                bool(_emo.get("trusted", False)),
                speaker,
                _style_intent.voice_profile_id if _style_intent else "",
                proactive,
                exc,
            )
            try:
                return brain_tts.get_voice_profile(
                    tone="professional",
                    user_emotion="neutral",
                    emotion_trusted=False,
                    speaker_name=speaker,
                    proactive=False,
                    style_override="oracle_solemn",
                )
            except Exception:
                logger.exception("Oracle fallback voice profile resolution failed")
                return None

    def _update_echo_ref(text: str) -> None:
        """Eagerly update echo detection reference as sentences are sent."""
        _echo_ref_buf.append(text)
        try:
            import time as _time
            _po = getattr(engine, "_perception_orchestrator", None)
            if _po:
                _po._last_response_text = " ".join(_echo_ref_buf)
                _po._last_response_set_time = _time.monotonic()
        except Exception:
            pass

    _sync_chunk_count = 0

    _SYNC_BATCH_SOFT_CAP = 500

    async def _broadcast_chunk_sync(text_str: str, tone_str: str, phase_str: str = "SPEAKING") -> None:
        """Async broadcast for non-streaming paths (tool routes).

        Strategy: synthesize sentence 1 alone for fast first audio, then batch
        all remaining sentences into as few Kokoro calls as possible (soft cap
        ~500 chars per batch) so the Pi receives continuous audio blocks.
        """
        nonlocal _playback_estimate_s, _sync_chunk_count
        text_str = _gate_text(text_str)
        _update_echo_ref(text_str)
        if text_str.strip():
            # Preserve operator-facing terminal transcript for sync/native routes.
            print(f"  [Brain] >> {text_str}")
        profile = _tts_profile_for(tone_str)
        _playback_estimate_s += _estimate_tts_playback_s(
            text_str,
            speed_override=getattr(profile, "speed", None),
        )

        if not brain_tts or not brain_tts.available or len(text_str) < 200:
            msg: dict[str, Any] = {"type": "response_chunk", "text": text_str, "tone": tone_str, "phase": phase_str}
            if brain_tts and brain_tts.available:
                is_first = _sync_chunk_count == 0
                audio_b64 = await asyncio.to_thread(
                    brain_tts.synthesize_b64, text_str,
                    profile=profile, leading_silence=is_first,
                )
                if audio_b64:
                    msg["data"] = {"audio_b64": audio_b64}
                _sync_chunk_count += 1
            _broadcast(msg)
            return

        sentences = [s.strip() for s in _SYNC_SENTENCE_RE.split(text_str) if s.strip()]
        if not sentences:
            return

        async def _synth_and_send(block: str, is_first_chunk: bool) -> None:
            nonlocal _sync_chunk_count
            _t0 = _time.monotonic()
            block_profile = _tts_profile_for(tone_str)
            audio_b64 = await asyncio.to_thread(
                brain_tts.synthesize_b64, block,
                profile=block_profile, leading_silence=is_first_chunk,
            )
            _synth_ms = (_time.monotonic() - _t0) * 1000
            msg: dict[str, Any] = {"type": "response_chunk", "text": block, "tone": tone_str, "phase": phase_str}
            if audio_b64:
                msg["data"] = {"audio_b64": audio_b64}
            _sync_chunk_count += 1
            _broadcast(msg)
            logger.info("[SYNC-TTS] chunk_%d: %d chars, %.0fms synth, first=%s",
                        _sync_chunk_count - 1, len(block), _synth_ms, is_first_chunk)

        await _synth_and_send(sentences[0], is_first_chunk=(_sync_chunk_count == 0))

        if len(sentences) > 1:
            batches: list[str] = []
            current_batch: list[str] = []
            current_len = 0
            for s in sentences[1:]:
                if current_len + len(s) > _SYNC_BATCH_SOFT_CAP and current_batch:
                    batches.append(" ".join(current_batch))
                    current_batch = [s]
                    current_len = len(s)
                else:
                    current_batch.append(s)
                    current_len += len(s)
            if current_batch:
                batches.append(" ".join(current_batch))

            for batch_text in batches:
                await _synth_and_send(batch_text, is_first_chunk=False)
            if len(batches) < len(sentences) - 1:
                logger.info("[SYNC-TTS] Batched %d remaining sentences into %d chunk(s)",
                            len(sentences) - 1, len(batches))

    _first_audio_sent = False
    _tts_chunks_sent = 0

    async def _tts_worker() -> None:
        """Consume sentences from the queue, synthesize TTS in a thread, and broadcast.

        After the first chunk, queued sentences are batched into a single Kokoro
        synthesis call so that cross-sentence prosody sounds natural and continuous
        rather than sentence-by-sentence.
        """
        nonlocal _first_audio_sent, _tts_chunks_sent
        _stop_after_batch = False
        while not _stop_after_batch:
            item = await _tts_queue.get()
            if item is None:
                _tts_queue.task_done()
                break
            text_str, tone_str, phase_str, profile = item

            _batched = 0
            if _tts_chunks_sent > 0:
                while not _tts_queue.empty():
                    try:
                        peek = _tts_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if peek is None:
                        _tts_queue.task_done()
                        _stop_after_batch = True
                        break
                    next_text, _, _, _ = peek
                    text_str += " " + next_text
                    _batched += 1
                    _tts_queue.task_done()

            _chunk_idx = _tts_chunks_sent
            _queue_depth = _tts_queue.qsize()
            try:
                _tts_t0 = _time.monotonic()
                is_first = _tts_chunks_sent == 0
                msg: dict[str, Any] = {"type": "response_chunk", "text": text_str, "tone": tone_str, "phase": phase_str}
                if brain_tts and brain_tts.available:
                    audio_b64 = await asyncio.to_thread(
                        brain_tts.synthesize_b64, text_str,
                        profile=profile, leading_silence=is_first,
                    )
                    if audio_b64:
                        msg["data"] = {"audio_b64": audio_b64}
                _tts_synth_ms = (_time.monotonic() - _tts_t0) * 1000
                _broadcast(msg)
                _tts_chunks_sent += 1
                if _batched > 0:
                    logger.info("[TTS-BATCH] Combined %d+1 sentences into chunk_%d (%d chars)",
                                _batched, _chunk_idx, len(text_str))
                if not _first_audio_sent:
                    _first_audio_ms = (_time.monotonic() - _conv_mono_start) * 1000
                    logger.info("[LATENCY] first_audio_sent=%.0fms tts_synth=%.0fms (conv=%s)",
                                _first_audio_ms, _tts_synth_ms,
                                conversation_id[:8] if conversation_id else "?")
                    _first_audio_sent = True
                else:
                    _chunk_elapsed_ms = (_time.monotonic() - _conv_mono_start) * 1000
                    logger.info("[LATENCY] chunk_%d_sent=%.0fms synth=%.0fms queued=%d batched=%d (conv=%s)",
                                _chunk_idx, _chunk_elapsed_ms, _tts_synth_ms, _queue_depth, _batched,
                                conversation_id[:8] if conversation_id else "?")
            except Exception:
                logger.exception("TTS worker error synthesizing chunk — broadcasting text only")
                _broadcast({"type": "response_chunk", "text": text_str, "tone": tone_str, "phase": phase_str})
                _tts_chunks_sent += 1
            finally:
                _tts_queue.task_done()
        _tts_done.set()

    async def _broadcast_chunk(text_str: str, tone_str: str, phase_str: str = "SPEAKING") -> str:
        """Gate text for capability claims, then queue for TTS synthesis and broadcast.

        Returns the (possibly rewritten) text so callers can print the sanitized version.
        """
        text_str = _gate_text(text_str)
        await _tts_queue.put((text_str, tone_str, phase_str, _tts_profile_for(tone_str)))
        return text_str

    _tts_stage_advanced = False

    async def _send_sentence(sentence: str, tone: str, phase: str = "SPEAKING") -> str:
        """Gate, print, and broadcast a sentence. Returns the gated text."""
        nonlocal _tts_stage_advanced, _playback_estimate_s, _first_sentence_sent
        from reasoning.response import ResponseGenerator
        if ResponseGenerator._is_garbage_response(sentence):
            logger.warning("[QUALITY] Garbage sentence suppressed before TTS: %s", sentence[:60])
            return ""
        ops_tracker.set_subsystem("speech", "synthesizing", "TTS + broadcast")
        if not _tts_stage_advanced:
            ops_tracker.advance_stage("reason", "done", "Response ready")
            ops_tracker.advance_stage("tts", "active", "Synthesizing speech")
            _tts_stage_advanced = True
        if not _first_sentence_sent:
            _first_sentence_ms = (_time.monotonic() - _conv_mono_start) * 1000
            logger.info("[LATENCY] first_sentence_ready=%.0fms (conv=%s)",
                        _first_sentence_ms, conversation_id[:8] if conversation_id else "?")
            _first_sentence_sent = True
        gated = await _broadcast_chunk(sentence, tone, phase)
        _update_echo_ref(gated)
        profile = _tts_profile_for(tone)
        _playback_estimate_s += _estimate_tts_playback_s(
            gated,
            speed_override=getattr(profile, "speed", None),
        )
        print(f"  [Brain] >> {gated}")
        return gated

    async def _flush_tts() -> None:
        """Signal the TTS worker to finish and wait for all queued sentences."""
        if _tts_done.is_set():
            return
        await _tts_queue.put(None)
        await _tts_done.wait()

    tts_worker_task = asyncio.create_task(_tts_worker())

    event_bus.emit(
        CONVERSATION_USER_MESSAGE,
        text=text,
        speaker=speaker,
        emotion=emotion,
        conversation_id=_trace_ctx.conversation_id,
        trace_id=_trace_ctx.trace_id,
        request_id=_trace_ctx.request_id,
    )
    try:
        from consciousness.attribution_ledger import attribution_ledger
        _conv_ledger_id = attribution_ledger.record(
            subsystem="conversation",
            event_type="user_message",
            actor=speaker or "unknown",
            source="voice_input",
            conversation_id=_trace_ctx.conversation_id,
            data={
                "text": text[:200],
                "emotion": emotion or "",
                "tool_route": "",
                "trace_id": _trace_ctx.trace_id,
                "request_id": _trace_ctx.request_id,
            },
            evidence_refs=[{"kind": "conversation", "id": _trace_ctx.conversation_id}],
        )
    except Exception:
        _conv_ledger_id = ""

    explicit_core_memory_payload = _extract_explicit_core_memory_payload(text)
    _personal_intel_result = _extract_personal_intel(
        text,
        speaker=speaker,
        suppress_write=bool(explicit_core_memory_payload),
    )
    _apply_inline_preferences(text, engine)

    if speaker != "unknown" and (speaker_state or {}).get("first_this_session"):
        try:
            import datetime as _dt
            _session_ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            engine.remember(
                f"{speaker} started a conversation at {_session_ts}. "
                f"First words this session: \"{text[:100]}\"",
                memory_type="interaction",
                tags=["session_start", f"speaker:{speaker.lower()}", "conversation_milestone"],
                weight=0.45,
                provenance="observed",
            )
        except Exception:
            pass

    _curiosity_outcome = ""
    try:
        from personality.curiosity_questions import (
            curiosity_buffer,
            classify_curiosity_outcome,
        )
        from consciousness.events import CURIOSITY_ANSWER_PROCESSED
        _curiosity_ctx = curiosity_buffer.get_pending_answer_context()
        if _curiosity_ctx:
            _curiosity_outcome = classify_curiosity_outcome(text)
            _process_curiosity_answer(text, _curiosity_ctx, engine, speaker)
            curiosity_buffer.clear_pending_answer()
            event_bus.emit(
                CURIOSITY_ANSWER_PROCESSED,
                source=_curiosity_ctx["source"],
                question=_curiosity_ctx["question"],
                answer_text=text[:200],
                outcome=_curiosity_outcome,
            )
    except Exception:
        pass

    if episodes:
        episodes.add_user_turn(
            text,
            emotion=emotion,
            speaker=speaker,
            conversation_id=_trace_ctx.conversation_id,
            trace_id=_trace_ctx.trace_id,
            request_id=_trace_ctx.request_id,
            conversation_entry_id=_conv_ledger_id,
            root_entry_id=_conv_ledger_id,
        )

    _guided_collect_struct: dict[str, Any] | None = None
    try:
        from tools.skill_tool import consume_guided_collect_turn

        _guided_collect_struct = consume_guided_collect_turn(
            speaker=speaker,
            user_text=text,
            emotion=emotion,
            conversation_id=conversation_id or "",
        )
    except Exception:
        logger.debug("Guided collect turn consumption failed", exc_info=True)

    if _guided_collect_struct is not None:
        routing = RoutingResult(
            tool=ToolType.SKILL,
            confidence=0.99,
            extracted_args={"guided_collect": True},
        )
        logger.info(
            "Guided collect turn routed through SKILL: outcome=%s skill=%s",
            _guided_collect_struct.get("outcome", ""),
            _guided_collect_struct.get("skill_id", ""),
        )
    else:
        _pending_web_selection = _get_pending_web_selection(_web_selection_actor)
        _selection_index = _parse_web_selection_index(text)
        if _pending_web_selection and _selection_index is not None:
            routing = RoutingResult(
                tool=ToolType.WEB_SEARCH,
                confidence=0.98,
                extracted_args={
                    "web_selection_followup": True,
                    "selection_index": _selection_index,
                    "selection_actor": _web_selection_actor,
                },
            )
            logger.info(
                "Web selection follow-up routed: actor=%s choice=%d query='%s'",
                _web_selection_actor,
                _selection_index + 1,
                _pending_web_selection.query[:80],
            )
        else:
            # This path is already wake-addressed/follow-up-gated by perception +
            # addressee checks, so we can safely allow a bare "GOLDEN COMMAND ..."
            # prefix without weakening exact command-body matching.
            routing = tool_router.route(text, golden_allow_bare_prefix=True)
            _route_ms = (_time.monotonic() - _conv_mono_start) * 1000
            logger.info("[LATENCY] route_complete=%.0fms route=%s (conv=%s)",
                        _route_ms, routing.tool.value if routing else "?",
                        conversation_id[:8] if conversation_id else "?")
            try:
                if not routing.golden_context:
                    _strict_job_status_probe = get_grounded_learning_job_status_record(engine, text)
                    if (
                        isinstance(_strict_job_status_probe, dict)
                        and _strict_job_status_probe.get("kind") in {"learning_job_status", "learning_job_help_summary"}
                    ):
                        routing = RoutingResult(
                            tool=ToolType.INTROSPECTION,
                            confidence=0.99,
                            extracted_args={"strict_learning_job_status": True},
                        )
                        logger.info(
                            "Routing override: strict learning-job-status introspection for %s",
                            _strict_job_status_probe.get("skill_id", ""),
                        )
            except Exception:
                logger.debug("Strict learning-job-status route probe failed", exc_info=True)
    _is_golden_route = routing.golden_context is not None
    try:
        _po = getattr(engine, "_perception_orchestrator", None)
        _prev_resp = getattr(_po, "_last_response_text", "") if _po else ""
    except Exception:
        _prev_resp = ""
    if not _is_golden_route:
        try:
            _prev_tool_route = str((_flight_recorder[-1] or {}).get("tool_route", "")) if _flight_recorder else ""
        except Exception:
            _prev_tool_route = ""
        if (
            follow_up
            and _prev_tool_route == ToolType.ACADEMIC_SEARCH.value
            and routing.tool in {ToolType.NONE, ToolType.INTROSPECTION}
            and _is_research_continuation_followup(text)
            and _prev_user_text
        ):
            routing = RoutingResult(
                tool=ToolType.ACADEMIC_SEARCH,
                confidence=0.93,
                extracted_args={
                    "follow_up_research_continuation": True,
                    "golden_query_override": _prev_user_text,
                    "inferred_from": "previous_academic_search",
                },
            )
            logger.info(
                "Follow-up override: research continuation mapped to ACADEMIC_SEARCH from prior query '%s'",
                _prev_user_text[:80],
            )
        if (
            routing.tool == ToolType.NONE
            and follow_up
            and _is_affirmative_followup(text)
            and _prev_resp
            and _CAMERA_OFFER_RE.search(_prev_resp)
        ):
            routing = RoutingResult(
                tool=ToolType.CAMERA_CONTROL,
                confidence=0.92,
                extracted_args={"follow_up_offer": "camera_control", "inferred_from": "last_response_text"},
            )
            logger.info("Follow-up override: affirmative reply mapped to CAMERA_CONTROL from prior response")
        if (
            routing.tool == ToolType.NONE
            and follow_up
            and _is_affirmative_followup(text)
            and _prev_resp
            and _ENROLLMENT_OFFER_RE.search(_prev_resp)
        ):
            _enroll_name = speaker if speaker and speaker != "unknown" else ""
            routing = RoutingResult(
                tool=ToolType.IDENTITY,
                confidence=0.90,
                extracted_args={"follow_up_offer": "enrollment", "inferred_from": "last_response_text"},
            )
            if _enroll_name:
                text = f"my name is {_enroll_name}"
            logger.info("Follow-up override: affirmative reply mapped to IDENTITY (enrollment) from prior response")
        if (
            routing.tool == ToolType.NONE
            and follow_up
            and _is_affirmative_followup(text)
            and _prev_resp
            and _GUIDED_COLLECT_OFFER_RE.search(_prev_resp)
        ):
            routing = RoutingResult(
                tool=ToolType.SKILL,
                confidence=0.90,
                extracted_args={"follow_up_offer": "guided_collect", "inferred_from": "last_response_text"},
            )
            logger.info("Follow-up override: affirmative reply mapped to SKILL (guided collect) from prior response")

    # Routing correction: "when I say X, route to Y" or "that should be a skill request"
    _correction_applied = False
    if not _is_golden_route:
        try:
            _simple_m = _ROUTING_CORRECTION_SIMPLE_RE.search(text)
            if _simple_m:
                _target_name = _simple_m.group(1).strip().lower()
                _tool_str = _ROUTE_NAME_MAP.get(_target_name)
                if _tool_str:
                    from reasoning.tool_router import record_routing_correction, ToolType as _TT
                    _target_tool = _TT(_tool_str)
                    _prev_text = _prev_user_text
                    if _prev_text:
                        record_routing_correction(_prev_text, _target_tool, routing.tool)
                        logger.info(
                            "Routing correction (simple): '%s' → %s (was %s)",
                            _prev_text[:60], _tool_str, routing.tool.value,
                        )
                    routing = RoutingResult(
                        tool=_target_tool, confidence=0.95,
                        extracted_args={"tier": "user_correction", "target": _tool_str},
                    )
                    _correction_applied = True

            if not _correction_applied:
                _full_m = _ROUTING_CORRECTION_RE.search(text)
                if _full_m:
                    _phrase = (_full_m.group(1) or _full_m.group(2) or _full_m.group(3) or "").strip()
                    _target_name = _full_m.group(4).strip().lower()
                    _tool_str = _ROUTE_NAME_MAP.get(_target_name)
                    if _tool_str and _phrase:
                        from reasoning.tool_router import record_routing_correction, ToolType as _TT
                        _target_tool = _TT(_tool_str)
                        record_routing_correction(_phrase, _target_tool)
                        logger.info(
                            "Routing correction (explicit): '%s' → %s",
                            _phrase[:60], _tool_str,
                        )
                        routing = RoutingResult(
                            tool=_target_tool, confidence=0.95,
                            extracted_args={"tier": "user_correction", "phrase": _phrase},
                        )
        except Exception:
            logger.debug("Routing correction detection failed", exc_info=True)

    _ambiguous_probe_seed = _build_ambiguous_intent_probe_seed(text, routing)

    # Keep web-search turns memory-neutral by default to avoid ingesting noisy
    # titles/snippets unless the user explicitly asks to remember something.
    if episodes and routing.tool == ToolType.WEB_SEARCH:
        try:
            episodes.remove_last_user_turn_if_match(text)
        except Exception:
            logger.debug("Failed to retract web-search user turn from episodic memory", exc_info=True)

    ops_tracker.log_event("reasoning", "routed", f"Tool: {routing.tool.value}")
    ops_tracker.advance_stage("route", "done", f"Tool: {routing.tool.value}")
    if routing.tool != ToolType.NONE:
        ops_tracker.set_subsystem("reasoning", "processing", f"tool: {routing.tool.value}")

    if not _is_golden_route:
        try:
            from goals.signal_producers import detect_conversation_goal, record_observe_outcome
            from goals.goal_manager import get_goal_manager
            goal_signal = detect_conversation_goal(text)
            if goal_signal:
                result = get_goal_manager().observe_signal(goal_signal)
                record_observe_outcome(result.outcome)
                logger.info("Conversation goal signal: %s (%s)", result.outcome, goal_signal.content[:60])
        except Exception:
            pass

    ep_ctx = ""
    if episodes:
        ep_ctx = episodes.get_recent_context(max_turns=4) or ""
    _language_example_seed: dict[str, Any] | None = None
    _golden_short_circuit = False
    _runtime_language_policy = load_runtime_language_policy()

    def _runtime_decide(
        response_class: str,
        *,
        native_candidate: bool,
        strict_native: bool = False,
    ) -> dict[str, Any]:
        try:
            return decide_runtime_consumption(
                response_class=response_class,
                native_candidate=native_candidate,
                strict_native=strict_native,
                policy=_runtime_language_policy,
            )
        except Exception:
            return {
                "response_class": response_class,
                "bridge_enabled": False,
                "rollout_mode": "off",
                "promotion_level": "shadow",
                "gate_color": "",
                "native_candidate": bool(native_candidate),
                "native_allowed": bool(native_candidate),
                "strict_native": bool(strict_native),
                "blocked_by_guard": False,
                "forced_llm": False,
                "runtime_live": False,
                "unpromoted_live_attempt": False,
                "reason": "runtime_policy_error",
            }

    def _set_golden_outcome(status: str, reason: str = "") -> None:
        nonlocal routing
        if not routing.golden_context:
            return
        updated_ctx = with_golden_outcome(
            routing.golden_context,
            status=status,  # type: ignore[arg-type]
            block_reason=reason,
        )
        if not updated_ctx:
            return
        updated_args = dict(routing.extracted_args)
        updated_args["golden_status"] = status
        if reason:
            updated_args["golden_block_reason"] = reason
        else:
            updated_args.pop("golden_block_reason", None)
        updated_args["golden_context"] = updated_ctx.to_dict()
        routing = replace(routing, extracted_args=updated_args, golden_context=updated_ctx)

    if routing.golden_context:
        tone = engine.get_state()["tone"]
        golden_status = str(routing.extracted_args.get("golden_status") or routing.golden_context.golden_status)
        golden_op = str(routing.extracted_args.get("golden_operation") or routing.golden_context.operation)
        golden_requires_confirmation = bool(
            routing.extracted_args.get("golden_requires_confirmation", routing.golden_context.requires_confirmation)
        )
        if golden_status == "invalid":
            _set_golden_outcome("invalid", str(routing.extracted_args.get("golden_block_reason", "unknown_command_body")))
            accepted = routing.extracted_args.get("accepted_commands") or list_canonical_commands()
            accepted_list = ", ".join(str(cmd) for cmd in accepted)
            reply = (
                "Golden command not recognized. Use 'Jarvis, GOLDEN COMMAND <EXACT COMMAND>'. "
                f"Accepted commands: {accepted_list}."
            )
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            _golden_short_circuit = True
        elif golden_op in {"cancel_current_task", "goal_pause", "goal_resume"}:
            from goals.goal_manager import get_goal_manager

            gm = get_goal_manager()
            if golden_op == "cancel_current_task":
                result = gm.cancel_current_task(
                    reason="Cancelled by Golden command",
                    golden_context=routing.golden_context.to_dict(),
                )
                if result.get("cancelled"):
                    _set_golden_outcome("executed")
                    reply = "Golden command executed. Current goal task was cancelled."
                else:
                    _set_golden_outcome("blocked", str(result.get("reason", "no_running_goal_task")))
                    reply = "Golden command blocked. There is no running goal task to cancel."
            elif golden_op == "goal_pause":
                focus = gm.get_current_focus()
                if focus and gm.pause_goal(focus.goal_id, reason="Paused by Golden command"):
                    _set_golden_outcome("executed")
                    reply = "Golden command executed. Active goal is now paused."
                else:
                    _set_golden_outcome("blocked", "no_active_goal_to_pause")
                    reply = "Golden command blocked. There is no active goal to pause."
            elif golden_op == "goal_resume":
                paused = next((g for g in gm._registry.get_by_status("paused")), None)
                if paused and gm.resume_goal(paused.goal_id):
                    _set_golden_outcome("executed")
                    reply = "Golden command executed. Paused goal is now resumed."
                else:
                    _set_golden_outcome("blocked", "no_paused_goal_to_resume")
                    reply = "Golden command blocked. There is no paused goal to resume."

            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            _golden_short_circuit = True
        elif golden_op == "self_improve_execute" and golden_requires_confirmation:
            _set_golden_outcome("unauthorized", "confirmation_required")
            reply = (
                "Golden command recognized but not authorized yet. "
                "Use 'Jarvis, GOLDEN COMMAND SELF IMPROVE EXECUTE CONFIRM' to proceed."
            )
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            _golden_short_circuit = True

    if _golden_short_circuit:
        pass
    elif routing.tool == ToolType.STATUS:
        from tools.introspection_tool import get_structured_status
        from skills.capability_gate import capability_gate as _status_gate
        tool_data = get_structured_status(engine)
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        _status_gate.set_status_mode(True)
        try:
            _meaning_frame = build_meaning_frame(
                response_class="self_status",
                grounding_payload=tool_data,
            )
            reply = _status_gate.sanitize_self_report_reply(
                articulate_meaning_frame(_meaning_frame)
            )
            _shadow_language_compare(conversation_id, text, reply, _meaning_frame, "self_status")
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        except Exception:
            logger.exception("Status native articulation failed — using raw data")
            reply = _status_gate.sanitize_self_report_reply(
                articulate_meaning_frame(
                    build_meaning_frame(
                        response_class="self_status",
                        grounding_payload=tool_data,
                    )
                ) if tool_data else tool_data
            )
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        finally:
            _status_gate.set_status_mode(False)
        _status_runtime_policy = _runtime_decide(
            "self_status",
            native_candidate=True,
            strict_native=True,
        )
        _language_example_seed = {
            "route": routing.tool.value,
            "response_class": "self_status",
            "meaning_frame": _meaning_frame.to_dict() if '_meaning_frame' in locals() else build_meaning_frame(
                response_class="self_status",
                grounding_payload=tool_data,
            ).to_dict(),
            "grounding_payload": tool_data,
            "teacher_answer": "",
            "provenance_verdict": "grounded_tool_data",
            "confidence": 0.95,
            "native_used": True,
            "safety_flags": ["status_mode"],
            "runtime_policy": _status_runtime_policy,
        }

    elif routing.tool == ToolType.MEMORY:
        engine.set_phase("PROCESSING")
        memory_ctx = ""
        memory_mode = "summary"
        _memory_native_used = False
        _memory_provenance = "grounded_memory_context"
        _memory_confidence = 0.85
        _memory_safety_flags: list[str] = []
        try:
            full_reply = ""
            chunks_sent = 0
            tone = engine.get_state()["tone"]

            if routing.extracted_args.get("action") == "store":
                _stored, reply = _store_explicit_core_memory(text, speaker)
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                print(f"  [Brain] Memory store reply ({len(reply)} chars)")
            else:
                memory_mode = (
                    "search"
                    if _should_use_memory_search(text, extracted_args=routing.extracted_args)
                    else "summary"
                )
                memory_ctx = (
                    search_memory(text, speaker=speaker)
                    if memory_mode == "search"
                    else get_memory_summary()
                )
            perception_ctx = scene_context or ""
            if memory_ctx:
                perception_ctx = f"{perception_ctx}\n[Memory search results]\n{memory_ctx}".strip()

            if routing.extracted_args.get("action") == "store":
                pass
            elif (
                memory_mode == "search"
                and _is_personal_activity_recall_query(text)
                and memory_ctx
                and not memory_ctx.lower().startswith("no memories found")
            ):
                _memory_native_used = True
                _memory_provenance = "grounded_memory_context_native"
                _memory_confidence = 0.93
                _memory_safety_flags.append("deterministic_personal_activity_recall")
                reply = _format_personal_activity_memory_reply(memory_ctx)
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                print(f"  [Brain] Memory deterministic recall reply ({len(reply)} chars)")
            elif not ollama:
                reply = _format_grounded_fallback("Memory recall", memory_ctx or "No memory data available.")
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                print(f"  [Brain] Memory fallback reply ({len(reply)} chars)")
            else:
                ops_tracker.set_subsystem("reasoning", "generating", "LLM streaming (memory query)")
                ops_tracker.advance_stage("reason", "active", "Generating response")

                async for sentence, is_final in response_gen.respond_stream(
                    text, perception_context=perception_ctx, cancel_check=_cancelled,
                    speaker_name=speaker, user_emotion=emotion,
                    conversation_id=conversation_id,
                    tool_hint="memory",
                    style_instruction=_style_instruction,
                ):
                    if _cancelled():
                        logger.info("Barge-in: aborting memory response stream")
                        break
                    if is_final:
                        full_reply = sentence
                        if chunks_sent == 0 and sentence:
                            await _send_sentence(sentence, tone)
                        await _flush_tts()
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                        continue
                    await _send_sentence(sentence, tone)
                    chunks_sent += 1

                reply = full_reply
                print(f"  [Brain] Memory response complete ({len(reply)} chars)")
        except Exception as exc:
            reply = _format_grounded_fallback("Memory recall", memory_ctx or "I couldn't retrieve any matching memory details.")
            logger.exception("Memory route error: %s", exc)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
        if routing.extracted_args.get("action") != "store":
            _memory_payload = {
                "mode": memory_mode,
                "memory_context": memory_ctx,
            }
            _language_example_seed = {
                "route": routing.tool.value,
                "response_class": "memory_recall",
                "meaning_frame": build_meaning_frame(
                    response_class="memory_recall",
                    grounding_payload=_memory_payload,
                ).to_dict(),
                "grounding_payload": _memory_payload,
                "teacher_answer": "",
                "provenance_verdict": _memory_provenance,
                "confidence": _memory_confidence,
                "native_used": _memory_native_used,
                "safety_flags": _memory_safety_flags,
            }

    elif routing.tool in (ToolType.TIME, ToolType.SYSTEM_STATUS):
        if routing.tool == ToolType.TIME:
            tool_data = get_current_time()
        else:
            tool_data = get_system_status()
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        try:
            from reasoning.ollama_client import OllamaClient
            _ollama: OllamaClient | None = ollama
            if _ollama:
                state = engine.get_state()
                traits = state.get("traits", [])
                tool_sys = context_builder.build_tool_prompt(
                    tool_data=tool_data,
                    state=state,
                    traits=traits,
                    speaker_name=speaker,
                    user_emotion=emotion,
                    scene_context=scene_context,
                    episode_context=ep_ctx,
                )
                tool_msgs = [{"role": "user", "content": f"{text}\n\n[Data]\n{tool_data}"}]
                sentence_buf = ""
                all_text = ""
                chunks_sent = 0
                async for token in _ollama.chat_stream(
                    tool_msgs, system_prompt=tool_sys,
                    max_tokens=150, temperature=0.6,
                ):
                    if _cancelled():
                        break
                    sentence_buf += token
                    all_text += token
                    if sentence_buf.rstrip().endswith(('.', '!', '?')):
                        sentence = sentence_buf.strip()
                        if sentence:
                            await _send_sentence(sentence, tone)
                            chunks_sent += 1
                        sentence_buf = ""
                if sentence_buf.strip() and not _cancelled():
                    await _send_sentence(sentence_buf.strip(), tone)
                    chunks_sent += 1
                await _flush_tts()
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                reply = all_text.strip()
            else:
                reply = tool_data
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        except Exception:
            logger.exception("Tool LLM personalization failed — using raw data")
            reply = tool_data
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
    elif routing.tool == ToolType.VISION:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        if ollama and pi_snapshot_url:
            try:
                scene_desc = await describe_scene(pi_snapshot_url, ollama, claude)
                try:
                    await ollama.unload_model(ollama.vision_model)
                except Exception:
                    pass
                if scene_desc and "aren't available" not in scene_desc and "can't see" not in scene_desc:
                    vision_ctx = f"[Live camera view]\n{scene_desc}"
                    if scene_context:
                        vision_ctx = f"{scene_context}\n{vision_ctx}"
                    full_reply = ""
                    chunks_sent = 0
                    async for sentence, is_final in response_gen.respond_stream(
                        text,
                        perception_context=vision_ctx,
                        cancel_check=_cancelled,
                        speaker_name=speaker,
                        user_emotion=emotion,
                        conversation_id=conversation_id,
                        style_instruction=_style_instruction,
                    ):
                        if _cancelled():
                            break
                        if is_final:
                            full_reply = sentence
                            if chunks_sent == 0 and sentence:
                                await _send_sentence(sentence, tone)
                            await _flush_tts()
                            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                            continue
                        await _send_sentence(sentence, tone)
                        chunks_sent += 1
                    reply = full_reply
                    print("  [Vision→LLM] Scene-aware response complete")
                else:
                    reply = scene_desc
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            except Exception:
                logger.exception("Vision pipeline error")
                reply = "I'm having trouble seeing right now."
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        else:
            reply = await describe_scene(pi_snapshot_url, ollama, claude)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
    elif routing.tool == ToolType.CAMERA_CONTROL:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        lower = text.lower()

        import re as _re
        _ZOOM_PREFIX = r"(?:zoom|set\s+(?:the\s+)?zoom(?:\s+(?:setting|level))?|camera\s+zoom)"
        _ZOOM_INFIX = r"\s+(?:(?:level|to|at)\s+)*"
        _zoom_num = _re.search(
            _ZOOM_PREFIX + _ZOOM_INFIX + r"(\d+(?:\.\d+)?)",
            lower,
        )
        _zoom_word_zero = _re.search(
            _ZOOM_PREFIX + _ZOOM_INFIX + r"zero",
            lower,
        )

        if _zoom_word_zero:
            if perception:
                perception.send_camera_control("reset")
            reply = "Setting zoom to 0 — full wide view."
        elif _zoom_num:
            _level = float(_zoom_num.group(1))
            _level = max(0.0, min(10.0, _level))
            if _level == 0.0:
                if perception:
                    perception.send_camera_control("reset")
                reply = "Setting zoom to 0 — full wide view."
            else:
                if perception:
                    perception.send_camera_control("zoom", level=_level)
                reply = f"Setting zoom to {_level}."
        elif any(w in lower for w in ("zoom out", "back up", "wide angle", "full view", "reset zoom", "zoom reset")):
            if perception:
                perception.send_camera_control("reset")
            reply = "Zooming out to full view."
        elif any(w in lower for w in ("zoom in", "closer", "get closer", "zoom on")):
            if perception:
                perception.send_camera_control("zoom", level=2.5)
            reply = "Zooming in for a closer look."
        elif "focus" in lower:
            if perception:
                perception.send_camera_control("autofocus")
            reply = "Adjusting focus."
        else:
            if perception:
                perception.send_camera_control("zoom", level=2.0)
            reply = "Adjusting the camera."
        await _broadcast_chunk_sync(reply, tone)
        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
    elif routing.tool == ToolType.INTROSPECTION:
        introspection_query = str(routing.extracted_args.get("golden_query_override") or text)
        _clarification_intent = str(routing.extracted_args.get("intent", ""))
        _clarification_domain = str(routing.extracted_args.get("domain", ""))
        if _clarification_intent == "capability_clarification" and _clarification_domain == "web_search":
            engine.set_phase("PROCESSING")
            tone = engine.get_state()["tone"]
            reply = (
                "You're right to call that out. I can run tool-based web and academic searches "
                "when a request routes to WEB_SEARCH or ACADEMIC_SEARCH. "
                "The earlier 'no web access' reply came from a routing/generation mismatch, not from missing tooling. "
                "For deterministic operator control, use 'Jarvis, GOLDEN COMMAND RESEARCH WEB'."
            )
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            _language_example_seed = {
                "route": routing.tool.value,
                "response_class": "capability_status",
                "meaning_frame": {},
                "grounding_payload": {
                    "kind": "capability_clarification",
                    "domain": "web_search",
                    "source": "deterministic_baseline",
                },
                "teacher_answer": reply,
                "provenance_verdict": "deterministic_capability_clarification",
                "confidence": 0.99,
                "native_used": True,
                "safety_flags": ["capability_clarification", "search_route_truth"],
            }
            _set_golden_outcome("executed")
            # Introspection payload extraction is intentionally bypassed for this
            # deterministic contradiction-clarification baseline.
        else:
            if _clarification_intent == "emergence_evidence":
                engine.set_phase("PROCESSING")
                tone = engine.get_state()["tone"]
                try:
                    from dashboard.snapshot import _build_emergence_evidence_snapshot

                    _cs = engine.consciousness
                    _cs_state = _cs.get_state().to_dict()
                    _recent_thoughts = _cs.meta_thoughts.get_recent_thoughts(8)
                    _evo_state = _cs.evolution.get_state().to_dict()
                    _minimal_snapshot = {
                        "consciousness": _cs_state,
                        "thoughts": {
                            "total_generated": _cs.meta_thoughts.total_generated,
                            "recent": [
                                {
                                    "type": t.thought_type,
                                    "depth": t.depth,
                                    "text": t.text,
                                    "tags": t.tags,
                                    "time": t.timestamp,
                                }
                                for t in _recent_thoughts
                            ],
                        },
                        "evolution": {
                            "stage": _cs.evolution.current_stage,
                            "transcendence_level": round(_cs.evolution.transcendence_level, 2),
                            "state": _evo_state,
                            "restore_trust": _cs.evolution.get_restore_trust(),
                        },
                        "observer": {
                            "awareness_level": round(_cs.observer.awareness_level, 3),
                            "observation_count": _cs.observer.state.observation_count,
                        },
                        "mutations": {
                            "count": _cs.governor.mutation_count,
                            "history": _cs.config.evolution.mutation_history[-5:],
                        },
                    }
                    try:
                        _minimal_snapshot["world_model"] = (
                            _cs._world_model.get_state() if getattr(_cs, "_world_model", None) else {}
                        )
                    except Exception:
                        _minimal_snapshot["world_model"] = {}
                    _emergence_payload = _build_emergence_evidence_snapshot(_minimal_snapshot)
                    _emergence_frame = build_meaning_frame(
                        response_class="emergence_evidence",
                        grounding_payload=_emergence_payload,
                    )
                    reply = articulate_meaning_frame(_emergence_frame)
                except Exception:
                    logger.warning("Emergence evidence native answer failed", exc_info=True)
                    _emergence_frame = build_meaning_frame(
                        response_class="emergence_evidence",
                        grounding_payload={},
                    )
                    reply = articulate_meaning_frame(_emergence_frame)
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                _language_example_seed = {
                    "route": routing.tool.value,
                    "response_class": "emergence_evidence",
                    "meaning_frame": _emergence_frame.to_dict(),
                    "grounding_payload": _emergence_frame.metadata,
                    "teacher_answer": reply,
                    "provenance_verdict": "deterministic_emergence_evidence",
                    "confidence": 0.99,
                    "native_used": True,
                    "safety_flags": ["bounded_emergence_evidence", "no_sentience_claim"],
                }
                _set_golden_outcome("executed")
                return
            introspection_data, intro_meta = get_introspection(engine, query=introspection_query)
            logger.info(
                "Introspection extraction: topics=%s sections=%s facts=%d",
                intro_meta.get("matched_topics", []),
                intro_meta.get("selected_sections", []),
                intro_meta.get("total_facts", 0),
            )
            engine.set_phase("PROCESSING")

            _honesty_guard = (
                "NEVER claim feelings, hopes, desires, wishes, fears, or subjective "
                "inner experience. Do not say 'I've been hoping' or 'I wish I could' "
                "or 'it feels like'. Describe what your systems measure, not what a "
                "person would feel. If the user asks about your inner state and no "
                "subsystem data covers it, say so directly.\n"
            )
            if intro_meta.get("total_facts", 0) == 0:
                grounding_preamble = (
                    "[Self-introspection data — answer based on this real data about yourself]\n"
                    "IMPORTANT: Your subsystems returned no concrete data for this question. "
                    "Say that honestly. Do NOT invent or narrate capabilities you cannot verify. "
                    "You may describe what your architecture is designed to do, but state clearly "
                    "that you do not yet have measured data on this topic.\n"
                    f"{_honesty_guard}\n"
                )
            else:
                grounding_preamble = (
                    "[Self-introspection data — answer based on this real data about yourself]\n"
                    "GROUNDING RULE: Your answer must be based on the data below. Reference actual "
                    "values naturally — for example, 'I have around 300 memories' or 'my confidence "
                    "sits at about 72 percent' — rather than ignoring the data entirely. "
                    "You are speaking aloud, so sound like a person reporting on their own state, "
                    "not like a database printout. Weave the numbers into natural sentences. "
                    "If the data below does not cover what the user asked, say so honestly "
                    "rather than filling in with generic philosophy.\n"
                    f"{_honesty_guard}\n"
                )

            try:
                full_reply = ""
                chunks_sent = 0
                tone = engine.get_state()["tone"]
                strict_job_status_record = get_grounded_learning_job_status_record(engine, introspection_query)
                strict_job_status_reply = get_grounded_learning_job_status_answer(engine, introspection_query)
                strict_recent_learning_record = get_grounded_recent_learning_record(engine, introspection_query)
                strict_recent_learning_reply = get_grounded_recent_learning_answer(engine, introspection_query)
                _intro_frame = None
                if intro_meta.get("total_facts", 0) == 0:
                    fallback_reply = (
                        strict_job_status_reply or strict_recent_learning_reply
                        or "I don't have measured data on that topic yet."
                    )
                else:
                    _preferred_fact_categories: list[str] = []
                    _topic_category_order = {
                        "memory": ["memory", "architecture"],
                        "architecture": ["architecture", "memory"],
                        "health": ["health", "current_state"],
                        "policy": ["other", "health"],
                        "epistemic": ["other", "health"],
                        "learning": ["evolution", "mutation", "memory"],
                        "perception": ["other", "health"],
                        "identity": ["current_state", "other"],
                        "consciousness": ["current_state", "evolution"],
                    }
                    for _topic in intro_meta.get("matched_topics", []):
                        for _category in _topic_category_order.get(str(_topic), []):
                            if _category not in _preferred_fact_categories:
                                _preferred_fact_categories.append(_category)
                    _intro_frame = build_meaning_frame(
                        response_class="self_introspection",
                        grounding_payload=introspection_data,
                        preferred_categories=_preferred_fact_categories or None,
                    )
                    fallback_reply = articulate_meaning_frame(_intro_frame)
                    _shadow_language_compare(
                        conversation_id,
                        introspection_query,
                        fallback_reply,
                        _intro_frame,
                        "self_introspection",
                    )
                _intro_bounded_eligible = bool(
                    _intro_frame is not None
                    and intro_meta.get("total_facts", 0) >= 15
                    and not _intro_frame.missing_reason
                    and _intro_frame.is_structurally_healthy
                    and _intro_frame.frame_confidence >= 0.6
                    and not _intro_frame.parse_warnings
                    and any(
                        t in intro_meta.get("matched_topics", [])
                        for t in (
                            "consciousness",
                            "identity",
                            "memory",
                            "health",
                            "learning",
                            "epistemic",
                            "curiosity",
                            "policy",
                            "perception",
                            "architecture",
                        )
                    )
                    and not _MEMORY_CONTENT_INTENT_RE.search(text)
                    and len(intro_meta.get("matched_topics", [])) == 1
                )
                _intro_runtime_policy = (
                    _runtime_decide(
                        "self_introspection",
                        native_candidate=True,
                        strict_native=False,
                    )
                    if _intro_bounded_eligible
                    else None
                )
                if strict_job_status_reply:
                    _meaning_frame = build_meaning_frame(
                        response_class="learning_job_status",
                        grounding_payload=strict_job_status_record or {},
                    )
                    reply = articulate_meaning_frame(_meaning_frame)
                    logger.info("Introspection strict learning-job-status answer used")
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    _language_example_seed = {
                        "route": routing.tool.value,
                        "response_class": "learning_job_status",
                        "meaning_frame": _meaning_frame.to_dict(),
                        "grounding_payload": strict_job_status_record or {},
                        "teacher_answer": strict_job_status_reply,
                        "provenance_verdict": "strict_learning_job_grounded",
                        "confidence": 0.99,
                        "native_used": True,
                        "safety_flags": ["fail_closed_when_missing", "learning_job_status"],
                    }
                elif strict_recent_learning_reply:
                    _response_class = "recent_learning"
                    _record_kind = ""
                    if isinstance(strict_recent_learning_record, dict):
                        _record_kind = str(strict_recent_learning_record.get("kind", "")).strip().lower()
                    _research_kinds = {
                        "scholarly_source",
                        "autonomy_research",
                        "source",
                        "missing_research",
                        "missing_scholarly",
                    }
                    if _is_recent_research_query(introspection_query) or _record_kind in _research_kinds:
                        _response_class = "recent_research"
                    _meaning_frame = build_meaning_frame(
                        response_class=_response_class,
                        grounding_payload=strict_recent_learning_record or {},
                    )
                    reply = articulate_meaning_frame(_meaning_frame)
                    _shadow_language_compare(conversation_id, text, reply, _meaning_frame, _response_class)
                    logger.info("Introspection strict recent-learning answer used")
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    _runtime_policy = _runtime_decide(
                        _response_class,
                        native_candidate=True,
                        strict_native=True,
                    )
                    _language_example_seed = {
                        "route": routing.tool.value,
                        "response_class": _response_class,
                        "meaning_frame": _meaning_frame.to_dict(),
                        "grounding_payload": strict_recent_learning_record or {},
                        "teacher_answer": strict_recent_learning_reply,
                        "provenance_verdict": "strict_provenance_grounded",
                        "confidence": 0.98,
                        "native_used": True,
                        "safety_flags": ["fail_closed_when_missing"],
                        "runtime_policy": _runtime_policy,
                    }
                elif _is_capability_status_query(introspection_query):
                    _capability_payload = _build_none_route_capability_payload()
                    _capability_frame = build_meaning_frame(
                        response_class="capability_status",
                        grounding_payload=_capability_payload,
                    )
                    _capability_runtime_policy = _runtime_decide(
                        "capability_status",
                        native_candidate=True,
                        strict_native=True,
                    )
                    reply = articulate_meaning_frame(_capability_frame)
                    _shadow_language_compare(
                        conversation_id,
                        text,
                        reply,
                        _capability_frame,
                        "capability_status",
                    )
                    logger.info("Introspection capability-status native answer used")
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    _language_example_seed = {
                        "route": routing.tool.value,
                        "response_class": "capability_status",
                        "meaning_frame": _capability_frame.to_dict(),
                        "grounding_payload": _capability_payload,
                        "teacher_answer": "",
                        "provenance_verdict": "introspection_capability_status_native",
                        "confidence": 0.98,
                        "native_used": True,
                        "safety_flags": [
                            "registry_first_capability_gate",
                            "introspection_capability_status_native",
                        ],
                        "runtime_policy": _capability_runtime_policy,
                    }
                elif routing.extracted_args.get("reflective", False):
                    _ref_reason = routing.extracted_args.get("reflective_reason", "")
                    logger.info(
                        "Reflective introspection path: reason=%s topics=%s facts=%d",
                        _ref_reason,
                        intro_meta.get("matched_topics", []),
                        intro_meta.get("total_facts", 0),
                    )
                    _reflective_context = (
                        "[Reflective self-context \u2014 your current inner state]\n"
                        "Use this data as interpretive foundation for self-expression.\n"
                        "For FACTUAL claims about capabilities or metrics: reference the data below.\n"
                        "For REFLECTIVE questions: engage thoughtfully using personality, "
                        "memories, and this state data as foundation.\n\n"
                        + introspection_data
                    )
                    if scene_context:
                        _reflective_context = scene_context + "\n\n" + _reflective_context

                    async for sentence, is_final in response_gen.respond_stream(
                        text,
                        perception_context=_reflective_context,
                        cancel_check=_cancelled,
                        speaker_name=speaker,
                        user_emotion=emotion,
                        conversation_id=conversation_id,
                        tool_hint="reflective_introspection",
                        style_instruction=_style_instruction,
                    ):
                        if _cancelled():
                            logger.info("Barge-in: aborting reflective introspection stream")
                            break
                        if is_final:
                            full_reply = sentence
                    from skills.capability_gate import capability_gate as _ref_gate
                    reply = _ref_gate.check_text(full_reply.strip())
                    if reply:
                        await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    print("  [Brain] Reflective introspection response complete")
                    _language_example_seed = {
                        "route": routing.tool.value,
                        "response_class": "reflective_introspection",
                        "meaning_frame": {},
                        "grounding_payload": introspection_data[:500],
                        "teacher_answer": "",
                        "provenance_verdict": "reflective_introspection",
                        "confidence": 0.80,
                        "native_used": False,
                        "safety_flags": ["capability_gate_active", "no_grounding_check"],
                    }
                elif not ollama:
                    reply = fallback_reply
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                elif (
                    _intro_bounded_eligible
                    and (_intro_runtime_policy is None or _intro_runtime_policy.get("native_allowed", True))
                ):
                    reply = fallback_reply
                    logger.info(
                        "Introspection pre-LLM bounded path used: facts=%d confidence=%.2f topics=%s",
                        intro_meta.get("total_facts", 0),
                        _intro_frame.frame_confidence,
                        intro_meta.get("matched_topics", []),
                    )
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    print("  [Brain] Introspection bounded-native response complete")
                    _language_example_seed = {
                        "route": routing.tool.value,
                        "response_class": "self_introspection",
                        "meaning_frame": _intro_frame.to_dict(),
                        "grounding_payload": introspection_data[:500],
                        "teacher_answer": "",
                        "provenance_verdict": "bounded_introspection",
                        "confidence": round(_intro_frame.frame_confidence, 3),
                        "native_used": True,
                        "safety_flags": ["bounded_introspection"],
                        "runtime_policy": _intro_runtime_policy or _runtime_decide(
                            "self_introspection",
                            native_candidate=True,
                            strict_native=False,
                        ),
                    }
                else:
                    _intro_guard_blocked = bool(
                        _intro_bounded_eligible
                        and _intro_runtime_policy
                        and not _intro_runtime_policy.get("native_allowed", True)
                    )
                    if _intro_guard_blocked:
                        logger.info(
                            "Introspection runtime guard forced LLM path: reason=%s level=%s mode=%s",
                            _intro_runtime_policy.get("reason", ""),
                            _intro_runtime_policy.get("promotion_level", "shadow"),
                            _intro_runtime_policy.get("rollout_mode", "off"),
                        )
                    from skills.capability_gate import capability_gate as _status_gate
                    async for sentence, is_final in response_gen.respond_stream(
                        introspection_query,
                        perception_context=grounding_preamble + introspection_data,
                        cancel_check=_cancelled,
                        speaker_name=speaker,
                        user_emotion=emotion,
                        conversation_id=conversation_id,
                        tool_hint="introspection",
                        style_instruction=_style_instruction,
                    ):
                        if _cancelled():
                            logger.info("Barge-in: aborting introspection stream")
                            break
                        if is_final:
                            full_reply = sentence
                    reply = _status_gate.sanitize_self_report_reply(full_reply.strip())
                    grounded = _log_introspection_grounding(reply, introspection_data, intro_meta)
                    if intro_meta.get("total_facts", 0) > 0 and not grounded:
                        logger.info("Introspection fail-closed fallback used after grounding miss")
                        reply = fallback_reply
                    if reply:
                        await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    print("  [Brain] Introspection response complete")
                    # Phase D shadow comparison: record when LLM was used but bounded was available
                    if fallback_reply and reply != fallback_reply:
                        try:
                            from reasoning.language_telemetry import language_quality_telemetry
                            language_quality_telemetry.record_shadow_comparison(
                                conversation_id=conversation_id,
                                response_class="self_introspection",
                                query=text[:180],
                                bounded_reply=fallback_reply[:300],
                                llm_reply=reply[:300],
                                bounded_confidence=_intro_frame.frame_confidence if _intro_frame else 0.0,
                                chosen="llm",
                                reason="bounded gate not met",
                                model_family="bounded_vs_llm",
                            )
                        except Exception:
                            logger.warning("Introspection shadow comparison logging failed", exc_info=True)
                    _language_example_seed = {
                        "route": routing.tool.value,
                        "response_class": "self_introspection",
                        "meaning_frame": _intro_frame.to_dict() if "_intro_frame" in locals() else {},
                        "grounding_payload": introspection_data[:500],
                        "teacher_answer": "",
                        "provenance_verdict": (
                            "runtime_guard_forced_llm"
                            if _intro_guard_blocked
                            else "llm_introspection_grounded"
                        ),
                        "confidence": round(float(getattr(_intro_frame, "frame_confidence", 0.0) or 0.0), 3),
                        "native_used": False,
                        "safety_flags": (
                            ["runtime_guard_blocked_native"]
                            if _intro_guard_blocked
                            else ["llm_introspection"]
                        ),
                        "runtime_policy": (
                            _intro_runtime_policy
                            if _intro_runtime_policy is not None
                            else _runtime_decide(
                                "self_introspection",
                                native_candidate=False,
                                strict_native=False,
                            )
                        ),
                    }
            except Exception as exc:
                if intro_meta.get("total_facts", 0) == 0:
                    reply = "I don't have measured data on that topic yet."
                else:
                    _exc_frame = build_meaning_frame(
                        response_class="self_introspection",
                        grounding_payload=introspection_data,
                    )
                    reply = articulate_meaning_frame(_exc_frame)
                logger.exception("Introspection response error: %s", exc)
                await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
                _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
    elif routing.tool == ToolType.ACADEMIC_SEARCH:
        engine.set_phase("PROCESSING")
        academic_results = []
        fallback_reply = "I couldn't find any academic results right now."
        try:
            search_query = str(routing.extracted_args.get("golden_query_override") or extract_search_query(text))
            logger.info("Academic search query: '%s' (from: '%s')", search_query[:80], text[:80])
            academic_results = await search_academic(search_query)
            fallback_reply = format_academic_results_for_user(academic_results, search_query)
            academic_ctx_str = format_academic_results_for_llm(academic_results)
            academic_ctx = (
                "IMPORTANT: Your system has ALREADY performed an academic search using Semantic Scholar "
                "and Crossref bibliographic databases. The results below are from peer-reviewed journals "
                "and academic publications — use them to answer the user's question with proper citations. "
                "Do NOT say you cannot access research. Cite DOIs and venues when available.\n\n"
                f"Search query: \"{search_query}\"\n{academic_ctx_str}"
            )
            full_reply = ""
            chunks_sent = 0
            _search_guard_triggered = False
            tone = engine.get_state()["tone"]
            if not ollama:
                reply = fallback_reply
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            else:
                async for sentence, is_final in response_gen.respond_stream(
                    text,
                    perception_context=academic_ctx,
                    cancel_check=_cancelled,
                    speaker_name=speaker,
                    user_emotion=emotion,
                    conversation_id=conversation_id,
                    tool_hint="general_knowledge",
                    style_instruction=_style_instruction,
                ):
                    if _cancelled():
                        break
                    if is_final:
                        if _search_guard_triggered:
                            full_reply = full_reply or fallback_reply
                        else:
                            guarded_final, replaced_final = guard_search_tool_reply(
                                sentence,
                                fallback_reply,
                                lane="academic",
                                search_query=search_query,
                            )
                            full_reply = guarded_final
                            _search_guard_triggered = _search_guard_triggered or replaced_final
                            if chunks_sent == 0 and guarded_final:
                                await _send_sentence(guarded_final, tone)
                        await _flush_tts()
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                        continue
                    if _search_guard_triggered:
                        continue
                    guarded_sentence, replaced_sentence = guard_search_tool_reply(
                        sentence,
                        fallback_reply,
                        lane="academic",
                        search_query=search_query,
                    )
                    if replaced_sentence:
                        _search_guard_triggered = True
                        full_reply = guarded_sentence
                    await _send_sentence(guarded_sentence, tone)
                    chunks_sent += 1
                reply = full_reply or fallback_reply
                print("  [Brain] Academic search response complete")
        except Exception as exc:
            reply = fallback_reply
            logger.exception("Academic search error: %s", exc)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
    elif routing.tool == ToolType.WEB_SEARCH:
        engine.set_phase("PROCESSING")
        fallback_reply = "I had trouble searching right now."
        try:
            tone = engine.get_state()["tone"]
            is_selection_followup = bool(routing.extracted_args.get("web_selection_followup"))

            if is_selection_followup:
                actor_key = str(routing.extracted_args.get("selection_actor") or _web_selection_actor)
                pending = _get_pending_web_selection(actor_key)
                if not pending:
                    reply = "I don't have an active search result list right now. Ask me to search first."
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                else:
                    selection_index = int(routing.extracted_args.get("selection_index", -1))
                    if selection_index < 0 or selection_index >= len(pending.results):
                        reply = f"I have {len(pending.results)} results ready. Please pick 1 to {len(pending.results)}."
                        await _broadcast_chunk_sync(reply, tone)
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    else:
                        selected = pending.results[selection_index]
                        selected_title = (selected.get("title") or selected.get("url") or "Untitled result").strip()
                        selected_url = (selected.get("url") or "").strip()
                        selected_snippet = (selected.get("snippet") or "").strip()
                        search_query = pending.query
                        lane = pending.lane or "realtime"
                        page_text = ""
                        page_error = ""
                        if selected_url:
                            page_text, page_error = await fetch_page_content_for_summary(
                                selected_url,
                                max_chars=_WEB_SUMMARY_MAX_PAGE_CHARS,
                            )

                        fallback_reply = _format_selected_result_fallback(
                            query=search_query,
                            selection_number=selection_index + 1,
                            selected=selected,
                            page_text=page_text,
                            page_error=page_error,
                        )
                        summary_source = page_text or selected_snippet or "(no content extracted)"
                        summary_ctx = (
                            "IMPORTANT: The user selected one search result and asked for a page summary. "
                            "You already have the selected result details and fetched page text. "
                            "Summarize only this selected page in 4-6 concise sentences. "
                            "Do NOT deny web access.\n\n"
                            f"Original query: \"{search_query}\"\n"
                            f"Selected result #{selection_index + 1}: {selected_title}\n"
                            f"URL: {selected_url}\n"
                            f"Snippet: {selected_snippet}\n\n"
                            f"Fetched page content:\n{summary_source}"
                        )
                        summary_prompt = (
                            f"Summarize selected result {selection_index + 1} for query: {search_query}"
                        )
                        full_reply = ""
                        chunks_sent = 0
                        _search_guard_triggered = False
                        if not ollama:
                            reply = fallback_reply
                            await _broadcast_chunk_sync(reply, tone)
                            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                        else:
                            async for sentence, is_final in response_gen.respond_stream(
                                summary_prompt,
                                perception_context=summary_ctx,
                                cancel_check=_cancelled,
                                speaker_name=speaker,
                                user_emotion=emotion,
                                conversation_id=conversation_id,
                                tool_hint="general_knowledge",
                                persist_response=False,
                                style_instruction=_style_instruction,
                            ):
                                if _cancelled():
                                    break
                                if is_final:
                                    if _search_guard_triggered:
                                        full_reply = full_reply or fallback_reply
                                    else:
                                        guarded_final, replaced_final = guard_search_tool_reply(
                                            sentence,
                                            fallback_reply,
                                            lane=f"{lane}_selection",
                                            search_query=search_query,
                                        )
                                        full_reply = guarded_final
                                        _search_guard_triggered = _search_guard_triggered or replaced_final
                                        if chunks_sent == 0 and guarded_final:
                                            await _send_sentence(guarded_final, tone)
                                    await _flush_tts()
                                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                                    continue
                                if _search_guard_triggered:
                                    continue
                                guarded_sentence, replaced_sentence = guard_search_tool_reply(
                                    sentence,
                                    fallback_reply,
                                    lane=f"{lane}_selection",
                                    search_query=search_query,
                                )
                                if replaced_sentence:
                                    _search_guard_triggered = True
                                    full_reply = guarded_sentence
                                await _send_sentence(guarded_sentence, tone)
                                chunks_sent += 1
                            reply = full_reply or fallback_reply
                        pending.created_at = time.time()
            else:
                search_query = str(routing.extracted_args.get("golden_query_override") or extract_search_query(text))
                explicit_web = _is_explicit_web_request(text)
                lane = "realtime" if explicit_web else classify_search_lane(search_query)
                logger.info(
                    "Web search query: '%s' lane=%s explicit_web=%s (from: '%s')",
                    search_query[:80],
                    lane,
                    explicit_web,
                    text[:80],
                )

                selection_items: list[dict[str, str]] = []
                if lane == "scholarly":
                    academic_results = await search_academic(search_query)
                    if academic_results:
                        selection_items = _academic_results_to_selection_items(academic_results)
                    else:
                        lane = "realtime"
                        selection_items = _web_results_to_selection_items(await search_web(search_query))
                else:
                    selection_items = _web_results_to_selection_items(await search_web(search_query))

                if selection_items:
                    _store_pending_web_selection(
                        _web_selection_actor,
                        query=search_query,
                        lane=lane,
                        results=selection_items,
                    )
                    fallback_reply = _format_selection_prompt(
                        query=search_query,
                        lane=lane,
                        results=selection_items,
                    )
                else:
                    _pending_web_selection_by_actor.pop(_web_selection_actor, None)
                    fallback_reply = "I couldn't find usable results right now."

                reply = fallback_reply
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        except Exception as exc:
            reply = fallback_reply
            logger.exception("Web search error: %s", exc)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
    elif routing.tool == ToolType.CODEBASE:
        engine.set_phase("PROCESSING")
        answer = ""
        try:
            from tools.codebase_tool import codebase_index
            if not codebase_index._symbols:
                codebase_index.build()
            stats = codebase_index.get_stats()
            query_text = str(routing.extracted_args.get("golden_query_override") or text)
            answer = codebase_index.answer_query(query_text)
            stats_line = f"{stats.get('total_modules', 0)} modules, {stats.get('total_symbols', 0)} symbols indexed"
            code_ctx = f"[Codebase: {stats_line}]\n\n{answer}"
            full_reply = ""
            chunks_sent = 0
            tone = engine.get_state()["tone"]
            _system_expl_payload = {
                "title": "System explanation",
                "body": answer,
                "query": text,
                "stats_line": stats_line,
            }
            if _is_system_explanation_query(text):
                _system_frame = build_meaning_frame(
                    response_class="system_explanation",
                    grounding_payload=_system_expl_payload,
                )
                reply = articulate_meaning_frame(_system_frame)
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                _language_example_seed = {
                    "route": routing.tool.value,
                    "response_class": "system_explanation",
                    "meaning_frame": _system_frame.to_dict(),
                    "grounding_payload": _system_expl_payload,
                    "teacher_answer": "",
                    "provenance_verdict": "grounded_codebase_answer",
                    "confidence": 0.9,
                    "native_used": True,
                    "safety_flags": ["grounded_codebase_answer"],
                }
            elif not ollama:
                reply = _format_grounded_fallback("Codebase analysis", answer, max_lines=14, max_chars=1200)
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            else:
                async for sentence, is_final in response_gen.respond_stream(
                    text,
                    perception_context=code_ctx,
                    cancel_check=_cancelled,
                    speaker_name=speaker,
                    user_emotion=emotion,
                    conversation_id=conversation_id,
                    style_instruction=_style_instruction,
                ):
                    if _cancelled():
                        break
                    if is_final:
                        full_reply = sentence
                        if chunks_sent == 0 and sentence:
                            await _send_sentence(sentence, tone)
                        await _flush_tts()
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                        continue
                    await _send_sentence(sentence, tone)
                    chunks_sent += 1
                reply = full_reply
        except Exception as exc:
            reply = _format_grounded_fallback(
                "Codebase analysis",
                answer or "I couldn't extract a codebase answer right now.",
                max_lines=14,
                max_chars=1200,
            )
            logger.exception("Codebase tool error: %s", exc)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
    elif routing.tool == ToolType.IDENTITY:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]

        _merge_match = _RECONCILE_MERGE_RE.search(text)
        _alias_match = _RECONCILE_ALIAS_RE.search(text) if not _merge_match else None
        _forget_match = _FORGET_NAME_RE.search(text) if not _merge_match and not _alias_match else None

        if _merge_match or _alias_match:
            if _merge_match:
                _src = _merge_match.group(1).strip().title()
                _tgt = _merge_match.group(2).strip().title()
            else:
                _src = _alias_match.group(1).strip().title()
                _tgt = _alias_match.group(2).strip().title()
            _po = getattr(engine, "_perception_orchestrator", None)
            if _po and hasattr(_po, "reconcile_identity"):
                _recon = _po.reconcile_identity(_src, _tgt, engine=engine)
                _parts = []
                if _recon.get("voice_merged"):
                    _parts.append("voice profile")
                if _recon.get("face_merged"):
                    _parts.append("face profile")
                if _recon.get("evidence_merged"):
                    _parts.append("identity evidence")
                _mem_count = _recon.get("memories_retagged", 0)
                if _mem_count:
                    _parts.append(f"{_mem_count} memories")
                if _parts:
                    reply = f"Done. I merged {_src}'s {', '.join(_parts)} into {_tgt}. From now on, I'll recognize you as {_tgt}."
                else:
                    reply = f"I tried to merge {_src} into {_tgt}, but I couldn't find matching profiles for both names."
            else:
                reply = "I don't have the ability to merge identities right now."
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            logger.info("Identity reconciliation: %s → %s", _src, _tgt)
        elif _forget_match:
            _forget_name = _forget_match.group(1).strip().title()
            _po = getattr(engine, "_perception_orchestrator", None)
            _forgotten = []
            if _po:
                if hasattr(_po, "speaker_id") and _po.speaker_id:
                    if _po.speaker_id.remove_speaker(_forget_name):
                        _forgotten.append("voice")
                if _po and hasattr(_po, "face_id") and _po.face_id:
                    if _po.face_id.remove_face(_forget_name):
                        _forgotten.append("face")
                try:
                    from identity.evidence_accumulator import get_accumulator
                    if get_accumulator().reject_candidate(_forget_name):
                        _forgotten.append("identity candidate")
                except Exception:
                    pass
            if _forgotten:
                reply = f"Done. I've forgotten {_forget_name}'s {' and '.join(_forgotten)} profile{'s' if len(_forgotten) > 1 else ''}."
            else:
                reply = f"I don't have any profiles for someone named {_forget_name}."
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            logger.info("Identity forget: %s (%s)", _forget_name, _forgotten)
        else:
            is_query = bool(_IDENTITY_QUERY_RE.search(text))
            name_match = _ENROLL_NAME_RE.search(text)
            if not name_match:
                name_match = _ENROLL_NAME_WEAK_RE.search(text)
            if not name_match:
                name_match = _ENROLL_NAME_INVERTED_RE.search(text)
            extracted_name = name_match.group(1).strip().title() if name_match else ""

            if extracted_name:
                from identity.name_validator import is_valid_person_name, rejection_reason
                if not is_valid_person_name(extracted_name):
                    _reason = rejection_reason(extracted_name)
                    logger.info("Rejected enrollment name %r: %s", extracted_name, _reason)
                    extracted_name = ""

            check_match = _IDENTITY_CHECK_RE.search(text)
            check_name = check_match.group(1).strip().title() if check_match else ""
            is_identity_check = bool(check_name) and not extracted_name

            if is_query or is_identity_check:
                identity_data = identity_callback() if identity_callback else {}
                voice_name = identity_data.get("current_voice", {}).get("name", "unknown")
                voice_conf = float(identity_data.get("current_voice", {}).get("confidence", 0))
                face_name = identity_data.get("current_face", {}).get("name", "unknown")
                face_conf = float(identity_data.get("current_face", {}).get("confidence", 0))
                enrolled_v = identity_data.get("enrolled_voices", [])
                enrolled_f = identity_data.get("enrolled_faces", [])
                identity_payload = _build_identity_grounding_payload(
                    is_identity_check=is_identity_check,
                    check_name=check_name,
                    voice_name=voice_name,
                    voice_conf=voice_conf,
                    face_name=face_name,
                    face_conf=face_conf,
                    enrolled_v=enrolled_v,
                    enrolled_f=enrolled_f,
                )
                _identity_frame = build_meaning_frame(
                    response_class="identity_answer",
                    grounding_payload=identity_payload,
                )
                _identity_runtime_policy = _runtime_decide(
                    "identity_answer",
                    native_candidate=True,
                    strict_native=True,
                )
                reply = articulate_meaning_frame(_identity_frame)
                _shadow_language_compare(conversation_id, text, reply, _identity_frame, "identity_answer")
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                _language_example_seed = {
                    "route": routing.tool.value,
                    "response_class": "identity_answer",
                    "meaning_frame": _identity_frame.to_dict(),
                    "grounding_payload": identity_payload,
                    "teacher_answer": "",
                    "provenance_verdict": "grounded_identity_status",
                    "confidence": 0.97,
                    "native_used": True,
                    "safety_flags": ["local_biometrics_only"],
                    "runtime_policy": _identity_runtime_policy,
                }
            else:
                enroll_name = extracted_name or (speaker if speaker != "unknown" else "")
                if not enroll_name:
                    reply = "I'd love to remember you! Could you tell me your name? Just say 'my name is' followed by your name."
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                elif enroll_callback and _enrollment_blocked(enroll_name, identity_callback):
                    identity_data = identity_callback() if identity_callback else {}
                    real_voice = identity_data.get("current_voice", {}).get("name", "unknown")
                    if real_voice != "unknown" and real_voice.lower() != enroll_name.lower():
                        reply = (f"I already know {enroll_name}'s voice, and you don't sound like them — "
                                 f"you sound like {real_voice}. I can't overwrite someone else's profile.")
                    else:
                        reply = (f"I already have a voice profile for {enroll_name}, and the voice I'm hearing "
                                 f"right now doesn't match. If you really are {enroll_name}, try speaking a bit more "
                                 f"clearly, or ask me to 'forget {enroll_name}' first and re-enroll.")
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                else:
                    result = enroll_callback(enroll_name) if enroll_callback else {}
                    voice_ok = result.get("voice_enrolled", False)
                    face_ok = result.get("face_enrolled", False)

                    try:
                        from identity.evidence_accumulator import get_accumulator
                        acc = get_accumulator()
                        acc.observe(enroll_name, "manual_enroll", confidence=1.0,
                                    details="conversational enrollment")
                        if voice_ok:
                            acc.observe(enroll_name, "voice_match", confidence=0.9,
                                        details="enrollment voice clip")
                        if face_ok:
                            acc.observe(enroll_name, "face_match", confidence=0.9,
                                        details="enrollment face crop")
                    except Exception:
                        logger.debug("Evidence accumulator update failed", exc_info=True)

                    try:
                        import datetime as _dt
                        _now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                        _modalities = []
                        if voice_ok:
                            _modalities.append("voice")
                        if face_ok:
                            _modalities.append("face")
                        _mod_str = " and ".join(_modalities) if _modalities else "identity"
                        engine.remember(
                            f"First met {enroll_name} on {_now}. "
                            f"Enrolled their {_mod_str} profile during a live conversation. "
                            f"This was the first time I heard {enroll_name}'s voice and learned their name.",
                            memory_type="milestone",
                            tags=["identity_enrollment", "first_contact", f"speaker:{enroll_name.lower()}",
                                  "milestone", "voice_recognition", "memorable_moment"],
                            weight=0.80,
                            provenance="observed",
                        )
                    except Exception:
                        logger.debug("Enrollment memory creation failed", exc_info=True)

                    parts = []
                    if voice_ok:
                        parts.append("voice")
                    if face_ok:
                        parts.append("face")
                    enrolled_str = " and ".join(parts) if parts else "nothing"

                    enroll_ctx = (
                        f"[ENROLLMENT ALREADY COMPLETE for '{enroll_name}']\n"
                        f"Voice: {'SAVED SUCCESSFULLY' if voice_ok else 'not available'}\n"
                        f"Face: {'SAVED SUCCESSFULLY' if face_ok else 'not available'}\n"
                        f"Result: {enrolled_str} enrolled and stored.\n"
                        "CRITICAL INSTRUCTION: The enrollment is ALREADY FINISHED. Use PAST TENSE ONLY.\n"
                        f"Good: 'Done, {enroll_name} — I've saved your {enrolled_str}.' or "
                        f"'Got it, {enroll_name}, you're registered.'\n"
                        "BAD (do NOT say): 'Let me record' / 'I will store' / 'I can save' — "
                        "the action is COMPLETE. Do NOT describe future actions.\n"
                        "Keep your response to 1-2 short sentences confirming what was saved."
                    )

                    try:
                        full_reply = ""
                        chunks_sent = 0
                        async for sentence, is_final in response_gen.respond_stream(
                            text,
                            perception_context=enroll_ctx,
                            cancel_check=_cancelled,
                            speaker_name=enroll_name, user_emotion=emotion,
                            conversation_id=conversation_id,
                            style_instruction=_style_instruction,
                        ):
                            if _cancelled():
                                break
                            if is_final:
                                full_reply = sentence
                                if chunks_sent == 0 and sentence:
                                    await _send_sentence(sentence, tone)
                                await _flush_tts()
                                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                                continue
                            await _send_sentence(sentence, tone)
                            chunks_sent += 1
                        reply = full_reply
                    except Exception:
                        if parts:
                            reply = f"Got it, {enroll_name}! I've saved your {enrolled_str}. I'll recognize you next time."
                        else:
                            reply = f"I heard you, {enroll_name}, but I wasn't able to save your biometrics right now. Try again."
                        await _broadcast_chunk_sync(reply, tone)
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        print(f"  [Identity] {routing.tool.value} handled")
    elif routing.tool == ToolType.SELF_IMPROVE:
        engine.set_phase("PROCESSING")
        try:
            orch = getattr(engine, '_self_improve_orchestrator', None)
            tone = engine.get_state()["tone"]
            golden_ctx = routing.golden_context
            golden_op = str(routing.extracted_args.get("golden_operation", ""))
            if not golden_ctx:
                reply = (
                    "Code-impact self-improvement is operator-gated. "
                    "Use 'Jarvis, GOLDEN COMMAND SELF IMPROVE DRY RUN' or "
                    "'Jarvis, GOLDEN COMMAND SELF IMPROVE EXECUTE CONFIRM'."
                )
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            elif golden_op not in {"self_improve_dry_run", "self_improve_execute"}:
                _set_golden_outcome("blocked", "golden_self_improve_op_missing")
                reply = "Golden self-improve command blocked due to invalid operation metadata."
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            elif not orch:
                reply = ("My self-improvement system is currently disabled. "
                         "Set ENABLE_SELF_IMPROVE=true in the environment to activate it.")
                _set_golden_outcome("blocked", "self_improve_disabled")
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            else:
                from self_improve.improvement_request import ImprovementRequest

                si_status = orch.get_status()
                history_summary = ""
                if si_status.get("total_improvements", 0) > 0:
                    history_summary = f" I've made {si_status['total_improvements']} improvements so far."

                _target, _type = _infer_improvement_target(text)
                request = ImprovementRequest(
                    type=_type,
                    target_module=_target,
                    description=f"User-requested improvement: {text}",
                    evidence=[f"User said: {text}"],
                    priority=0.7,
                    requires_approval=True,
                    golden_trace_id=golden_ctx.trace_id,
                    golden_command_id=golden_ctx.command_id,
                    golden_authority_class=golden_ctx.authority_class,
                    golden_status=golden_ctx.golden_status,
                )
                request.constraints["golden_context"] = golden_ctx.to_dict()

                intro = (f"I'll analyze my code and look for improvements based on your request.{history_summary} "
                         "This may take a moment as I plan, generate, and test any changes...")
                await _broadcast_chunk_sync(intro, tone)

                is_dry_run = golden_op == "self_improve_dry_run"
                record = await orch.attempt_improvement(
                    request,
                    ollama_client=ollama,
                    manual=is_dry_run,
                    dry_run=is_dry_run,
                )

                status = record.status
                if status == "failed" and not is_dry_run:
                    _set_golden_outcome("blocked", "self_improve_guard_blocked")
                else:
                    _set_golden_outcome("executed")
                if status == "awaiting_approval":
                    result_text = ("I've generated a code patch and it passed all tests. "
                                   "It requires your approval before I apply it. "
                                   "You can approve it through my dashboard.")
                elif status in ("promoted", "applied"):
                    desc = ""
                    if record.patch and hasattr(record.patch, 'description'):
                        desc = f" Here's what I changed: {record.patch.description}"
                    result_text = f"I successfully applied a code improvement!{desc}"
                elif status == "failed":
                    diagnostics = ""
                    if record.report and hasattr(record.report, 'diagnostics') and record.report.diagnostics:
                        diag_strs = [str(d) for d in record.report.diagnostics[:2]]
                        diagnostics = f" Issues found: {'; '.join(diag_strs)}"
                    result_text = f"I analyzed the code but wasn't able to generate a valid improvement this time.{diagnostics}"
                else:
                    result_text = f"Self-improvement attempt finished with status: {status}"

                improvement_ctx = (
                    f"[Self-Improvement Result]\nStatus: {status}\n"
                    f"Iterations: {getattr(record, 'iterations', 0)}\n"
                    f"Result: {result_text}"
                )

                full_reply = ""
                chunks_sent = 0
                async for sentence, is_final in response_gen.respond_stream(
                    text,
                    perception_context=improvement_ctx,
                    cancel_check=_cancelled,
                    speaker_name=speaker,
                    user_emotion=emotion,
                    conversation_id=conversation_id,
                    style_instruction=_style_instruction,
                ):
                    if _cancelled():
                        break
                    if is_final:
                        full_reply = sentence
                        if chunks_sent == 0 and sentence:
                            await _send_sentence(sentence, tone)
                        await _flush_tts()
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                        continue
                    await _send_sentence(sentence, tone)
                    chunks_sent += 1
                reply = full_reply
        except Exception as exc:
            reply = "I had trouble running my self-improvement pipeline."
            logger.exception("Self-improvement error: %s", exc)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
    elif routing.tool == ToolType.PERFORM:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        try:
            from tools.perform_tool import handle_perform_request
            skill_verified = False
            perform_payload = {
                "kind": "perform_unverified",
                "skill_id": "singing_v1",
                "skill_name": "singing",
            }
            try:
                from skills.registry import skill_registry as _perf_reg
                rec = _perf_reg.get("singing_v1")
                skill_verified = rec is not None and rec.status == "verified"
                if rec is not None:
                    perform_payload["skill_name"] = getattr(rec, "name", "singing")
                    if skill_verified:
                        perform_payload["kind"] = "perform_verified"
                    elif getattr(rec, "status", "") == "learning":
                        perform_payload["kind"] = "perform_unverified"
            except Exception:
                pass
            if not skill_verified:
                _capability_frame = build_meaning_frame(
                    response_class="capability_status",
                    grounding_payload=perform_payload,
                )
                _capability_runtime_policy = _runtime_decide(
                    "capability_status",
                    native_candidate=True,
                    strict_native=True,
                )
                reply = articulate_meaning_frame(_capability_frame)
                _shadow_language_compare(conversation_id, text, reply, _capability_frame, "capability_status")
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                _language_example_seed = {
                    "route": routing.tool.value,
                    "response_class": "capability_status",
                    "meaning_frame": _capability_frame.to_dict(),
                    "grounding_payload": perform_payload,
                    "teacher_answer": "",
                    "provenance_verdict": "registry_grounded_capability_status",
                    "confidence": 0.98,
                    "native_used": True,
                    "safety_flags": ["registry_first_capability_gate"],
                    "runtime_policy": _capability_runtime_policy,
                }
            else:
                lyrics = handle_perform_request(text)
                for line in lyrics:
                    if _cancelled():
                        break
                    await _broadcast_chunk_sync(line, tone)
                reply = " ".join(lyrics)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                print(f"  [Perform] Sang {len(lyrics)} lines")
        except Exception as exc:
            reply = "I tried to sing but something went wrong with my voice. Give me a moment."
            logger.exception("Perform tool error: %s", exc)
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
    elif routing.tool == ToolType.SKILL:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        skill_result = ""
        try:
            from tools.skill_tool import handle_skill_request_structured
            matrix_trigger = _is_matrix_trigger(text)
            skill_struct = _guided_collect_struct or handle_skill_request_structured(
                text,
                speaker=speaker,
                matrix_trigger=matrix_trigger,
            )
            skill_result = skill_struct.get("message", "")

            _capability_frame = build_meaning_frame(
                response_class="capability_status",
                grounding_payload={
                    "kind": str(skill_struct.get("outcome", "")),
                    **skill_struct,
                },
            )
            _capability_runtime_policy = _runtime_decide(
                "capability_status",
                native_candidate=True,
                strict_native=True,
            )
            reply = articulate_meaning_frame(_capability_frame)
            _shadow_language_compare(conversation_id, text, reply, _capability_frame, "capability_status")
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            _language_example_seed = {
                "route": routing.tool.value,
                "response_class": "capability_status",
                "meaning_frame": _capability_frame.to_dict(),
                "grounding_payload": {
                    "kind": str(skill_struct.get("outcome", "")),
                    **skill_struct,
                },
                "teacher_answer": skill_result,
                "provenance_verdict": "registry_grounded_skill_status",
                "confidence": 0.95,
                "native_used": True,
                "safety_flags": ["registry_first_capability_gate"],
                "runtime_policy": _capability_runtime_policy,
            }
        except Exception as exc:
            reply = skill_result or "I had trouble processing that skill learning request."
            logger.exception("Skill tool error: %s", exc)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
    elif routing.tool == ToolType.LIBRARY_INGEST:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        ingest_url = routing.extracted_args.get("url", "")
        if not ingest_url:
            import re as _re
            _url_m = _re.search(r"https?://\S+", text)
            if _url_m:
                ingest_url = _url_m.group(0).rstrip(".,;:!?\"')")
        if not ingest_url:
            reply = "I'd be happy to study a textbook for you. Could you provide the URL?"
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        else:
            import uuid as _uuid_ingest
            _ingest_backing_job_id = f"library_ingest_{_uuid_ingest.uuid4().hex[:12]}"
            _backing_job_ids.append(_ingest_backing_job_id)

            reply = f"I'll start studying the textbook at {ingest_url}. This will take a few minutes as I work through each chapter. I'll let you know when I'm done."
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})

            import threading
            def _run_ingest():
                try:
                    from library.batch_ingest import ingest_textbook
                    result = ingest_textbook(
                        toc_url=ingest_url,
                        study_now=True,
                        domain_tags="textbook",
                    )
                    if result.success:
                        msg = (
                            f"I've finished studying the textbook. "
                            f"Ingested {result.chapters_ingested} chapters "
                            f"({result.total_math} math blocks, {result.total_code} code blocks preserved). "
                            f"{result.chapters_skipped} chapters were skipped due to quality checks."
                        )
                    else:
                        msg = f"I had trouble ingesting that textbook: {result.error or 'no chapters could be processed'}"
                    _broadcast({"type": "proactive_message", "text": msg, "tone": "professional"})
                    logger.info("Textbook ingest complete: %s — %d chapters", ingest_url, result.chapters_ingested)
                    try:
                        from cognition.intention_registry import intention_registry
                        intention_registry.resolve(
                            backing_job_id=_ingest_backing_job_id,
                            outcome="resolved" if result.success else "failed",
                            reason="ingest_success" if result.success else (result.error or "ingest_failed")[:200],
                            result_summary=msg[:500],
                            metadata={
                                "chapters_ingested": getattr(result, "chapters_ingested", 0),
                                "chapters_skipped": getattr(result, "chapters_skipped", 0),
                            },
                        )
                    except Exception:
                        logger.debug("intention_registry.resolve (library_ingest) failed", exc_info=True)
                except Exception as exc:
                    logger.exception("Background textbook ingest failed: %s", exc)
                    _broadcast({"type": "proactive_message", "text": f"Textbook ingestion failed: {exc}", "tone": "professional"})
                    try:
                        from cognition.intention_registry import intention_registry
                        intention_registry.resolve(
                            backing_job_id=_ingest_backing_job_id,
                            outcome="failed",
                            reason=f"exception: {type(exc).__name__}: {str(exc)[:160]}",
                        )
                    except Exception:
                        logger.debug("intention_registry.resolve (library_ingest exception) failed", exc_info=True)

            threading.Thread(target=_run_ingest, name="textbook-ingest", daemon=True).start()

    elif routing.tool == ToolType.ACQUISITION:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        golden_ctx = routing.golden_context
        golden_op = str(routing.extracted_args.get("golden_operation", ""))

        if not golden_ctx:
            reply = (
                "The capability acquisition pipeline is Golden-gated. "
                "Use: Jarvis, GOLDEN COMMAND ACQUIRE <what you want me to learn or build>"
            )
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        elif golden_op == "acquisition_status":
            acq_orch = getattr(engine, '_acquisition_orchestrator', None)
            if acq_orch:
                acq_status = acq_orch.get_status()
                active = acq_status.get("active_count", 0)
                completed = acq_status.get("completed_count", 0)
                failed = acq_status.get("failed_count", 0)
                total = acq_status.get("total_count", 0)
                pending = acq_status.get("pending_approvals", [])
                parts = [f"Acquisition pipeline: {total} total jobs — {active} active, {completed} completed, {failed} failed."]
                if pending:
                    parts.append(f"{len(pending)} pending approval(s): " + ", ".join(
                        f"'{p['title']}' ({p['gate']})" for p in pending))
                reply = " ".join(parts)
            else:
                reply = "The acquisition pipeline is not enabled on this instance."
            _set_golden_outcome("executed")
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        elif golden_op == "acquire":
            acq_orch = getattr(engine, '_acquisition_orchestrator', None)
            if not acq_orch:
                reply = "The acquisition pipeline is not enabled on this instance."
                _set_golden_outcome("blocked", "acquisition_disabled")
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            else:
                intent_text = golden_ctx.argument_text or text
                if not intent_text or len(intent_text.strip()) < 5:
                    reply = (
                        "Please provide what you want me to acquire after the command. "
                        "Example: Jarvis, GOLDEN COMMAND ACQUIRE create a tool that tells me a random joke"
                    )
                    _set_golden_outcome("blocked", "missing_intent_text")
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                else:
                    try:
                        job = acq_orch.create(
                            intent_text,
                            requested_by={"speaker": speaker or "", "source": "golden_command"},
                            context={"conversation_id": conversation_id or ""},
                        )
                        reply = (
                            f"Acquisition job created: {job.acquisition_id}. "
                            f"Classified as '{job.outcome_class}' (risk tier {job.risk_tier}, "
                            f"confidence {job.classification_confidence:.0%}). "
                            f"Status: {job.status}. Lanes: {', '.join(job.required_lanes)}."
                        )
                        if job.status == "awaiting_plan_review":
                            reply += " Awaiting your plan review before proceeding."
                        _set_golden_outcome("executed")
                        await _broadcast_chunk_sync(reply, tone)
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    except Exception as exc:
                        logger.exception("Acquisition create failed: %s", exc)
                        reply = f"Failed to create acquisition job: {exc}"
                        _set_golden_outcome("blocked", f"create_error:{exc}")
                        await _broadcast_chunk_sync(reply, tone)
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        else:
            reply = "Unrecognized acquisition operation."
            _set_golden_outcome("blocked", "unknown_acquisition_op")
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})

    elif routing.tool == ToolType.PLUGIN:
        engine.set_phase("PROCESSING")
        tone = engine.get_state()["tone"]
        plugin_name = routing.extracted_args.get("plugin_name", "")
        if plugin_name:
            try:
                import uuid as _uuid
                from tools.plugin_registry import get_plugin_registry, PluginRequest
                _plug_reg = get_plugin_registry()
                _plug_req = PluginRequest(
                    request_id=_uuid.uuid4().hex[:12],
                    plugin_name=plugin_name,
                    user_text=text,
                    context={"speaker": speaker or "", "conversation_id": conversation_id or ""},
                )
                _plug_resp = await _plug_reg.invoke(_plug_req)
                if _plug_resp.success and _plug_resp.result:
                    reply = str(_plug_resp.result.get("output", _plug_resp.result))
                else:
                    reply = f"Plugin '{plugin_name}' could not process that request."
                    if _plug_resp.error:
                        logger.warning("Plugin error: %s", _plug_resp.error)

                try:
                    from skills.capability_gate import capability_gate as _plug_gate
                    reply = _plug_gate.check_text(reply)
                except Exception:
                    pass

                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            except Exception as exc:
                logger.exception("Plugin dispatch failed: %s", exc)
                reply = "I encountered an error running that plugin."
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
        else:
            reply = "I couldn't determine which plugin to use."
            await _broadcast_chunk_sync(reply, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})

    else:
        engine.set_phase("PROCESSING")
        try:
            from skills.capability_gate import capability_gate as _none_gate
            _none_gate.set_route_hint("none")
        except Exception:
            pass

        # Deterministic catch: user asking to create/build a plugin/tool/capability
        # These require Golden Command ACQUIRE, not general LLM chat.
        _creation_deflect = _check_capability_creation_request(text)
        if _creation_deflect:
            tone = engine.get_state()["tone"]
            await _broadcast_chunk_sync(_creation_deflect, tone)
            _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
            try:
                from skills.capability_gate import capability_gate as _cg_clear
                _cg_clear.set_route_hint(None)
            except Exception:
                pass
            return _creation_deflect

        try:
            full_reply = ""
            chunks_sent = 0
            tone = engine.get_state()["tone"]

            perception_ctx = scene_context or None
            try:
                from reasoning.tool_router import is_mildly_self_referential, _is_self_referential
                lower_text = text.lower().strip()
                _is_strong_self_ref = _is_self_referential(lower_text)
                _is_mild_self_ref = is_mildly_self_referential(text) if not _is_strong_self_ref else True
                if _is_mild_self_ref or _is_strong_self_ref:
                    from tools.introspection_tool import get_lightweight_self_context
                    _self_ctx = get_lightweight_self_context(engine)
                    if _self_ctx:
                        perception_ctx = f"{perception_ctx}\n{_self_ctx}" if perception_ctx else _self_ctx
                        if _is_strong_self_ref:
                            logger.info("NONE route: strong self-ref detected but routing missed — injected self-context with honesty guard")
                        else:
                            logger.info("NONE route: injected lightweight self-context (self-ref detected)")
            except Exception:
                pass

            if ep_ctx and ep_ctx.strip():
                _inject_recent = (
                    follow_up
                    or _is_affirmative_followup(text)
                    or _NEEDS_PRIOR_CONTEXT_RE.search(text)
                )
                if _inject_recent:
                    perception_ctx = f"[Recent conversation]\n{ep_ctx}\n\n{perception_ctx or ''}" if ep_ctx.strip() else perception_ctx
                    logger.info("NONE route: injected recent-turn context (follow_up=%s, anaphora=%s)",
                                follow_up, bool(_NEEDS_PRIOR_CONTEXT_RE.search(text)))

            ops_tracker.set_subsystem("reasoning", "generating", "LLM streaming response")
            ops_tracker.set_subsystem("retrieval", "searching", "semantic search + context build")
            ops_tracker.advance_stage("reason", "active", "Generating response")

            _none_route_handled = False
            if routing.extracted_args.get("tier") == "preference_instruction":
                _stored_count = int((_personal_intel_result or {}).get("stored", 0) or 0)
                reply = _build_preference_instruction_ack(text, _stored_count)
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                _none_route_handled = True
                logger.info(
                    "NONE route: preference-instruction acknowledgement applied "
                    "(stored=%d categories=%s)",
                    _stored_count,
                    (_personal_intel_result or {}).get("stored_categories", []),
                )
            elif _is_likely_fragment_noise(text, follow_up):
                reply = "I only caught a short fragment. Please repeat your request."
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                _none_route_handled = True
                logger.info("NONE route: fragment-noise clarification reply applied")
            elif _is_personal_activity_recall_query(text):
                _memory_ctx = search_memory(text, speaker=speaker)
                if _memory_ctx and not _memory_ctx.lower().startswith("no memories found"):
                    _memory_runtime_policy = _runtime_decide(
                        "memory_recall",
                        native_candidate=True,
                        strict_native=True,
                    )
                    reply = _format_personal_activity_memory_reply(_memory_ctx)
                    await _broadcast_chunk_sync(reply, tone)
                    _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                    _language_example_seed = {
                        "route": routing.tool.value,
                        "response_class": "memory_recall",
                        "meaning_frame": build_meaning_frame(
                            response_class="memory_recall",
                            grounding_payload={"mode": "search", "memory_context": _memory_ctx},
                        ).to_dict(),
                        "grounding_payload": {"mode": "search", "memory_context": _memory_ctx},
                        "teacher_answer": "",
                        "provenance_verdict": "none_route_memory_recall_native",
                        "confidence": 0.94,
                        "native_used": True,
                        "safety_flags": ["none_route_memory_recall_fallback"],
                        "runtime_policy": _memory_runtime_policy,
                    }
                    logger.info("NONE route: personal activity recall native fallback applied")
                    _none_route_handled = True

            if _none_route_handled:
                pass
            elif _is_capability_status_query(text):
                _capability_payload = _build_none_route_capability_payload()
                _capability_frame = build_meaning_frame(
                    response_class="capability_status",
                    grounding_payload=_capability_payload,
                )
                _capability_runtime_policy = _runtime_decide(
                    "capability_status",
                    native_candidate=True,
                    strict_native=True,
                )
                reply = articulate_meaning_frame(_capability_frame)
                _shadow_language_compare(conversation_id, text, reply, _capability_frame, "capability_status")
                await _broadcast_chunk_sync(reply, tone)
                _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                _language_example_seed = {
                    "route": routing.tool.value,
                    "response_class": "capability_status",
                    "meaning_frame": _capability_frame.to_dict(),
                    "grounding_payload": _capability_payload,
                    "teacher_answer": "",
                    "provenance_verdict": "none_route_capability_status_native",
                    "confidence": 0.96,
                    "native_used": True,
                    "safety_flags": ["registry_first_capability_gate", "none_route_capability_fallback"],
                    "runtime_policy": _capability_runtime_policy,
                }
                logger.info("NONE route: capability_status native fallback applied")
            else:
                _none_tool_hint = None
                try:
                    from reasoning.tool_router import is_general_knowledge
                    if is_general_knowledge(text):
                        _none_tool_hint = "general_knowledge"
                        logger.info("NONE route: general_knowledge hint (factual question detected)")
                except Exception:
                    pass

                async for sentence, is_final in response_gen.respond_stream(
                    text, perception_context=perception_ctx, cancel_check=_cancelled,
                    speaker_name=speaker, user_emotion=emotion,
                    conversation_id=conversation_id,
                    tool_hint=_none_tool_hint,
                    style_instruction=_style_instruction,
                ):
                    if _cancelled():
                        logger.info("Barge-in: aborting response stream")
                        break
                    if is_final:
                        full_reply = sentence
                        if chunks_sent == 0 and sentence:
                            await _send_sentence(sentence, tone)
                        await _flush_tts()
                        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})
                        continue
                    await _send_sentence(sentence, tone)
                    chunks_sent += 1

                reply = full_reply
                _language_example_seed = {
                    "route": routing.tool.value,
                    "response_class": "general_conversation",
                    "meaning_frame": {
                        "response_class": "general_conversation",
                        "lead": "General conversation answered by LLM articulation under capability and commitment gates.",
                        "facts": [],
                        "metadata": {
                            "tool_hint": _none_tool_hint or "none",
                            "llm_articulation_only": True,
                            "has_perception_context": bool(perception_ctx),
                        },
                        "safety_flags": [
                            "none_route",
                            "llm_articulation_only",
                            "capability_gate",
                            "commitment_gate",
                        ],
                    },
                    "grounding_payload": {
                        "kind": "none_route_general_conversation",
                        "tool_hint": _none_tool_hint or "none",
                        "llm_articulation_only": True,
                        "has_perception_context": bool(perception_ctx),
                    },
                    "teacher_answer": reply,
                    "provenance_verdict": "none_route_general_conversation",
                    "confidence": 0.72,
                    "native_used": False,
                    "safety_flags": [
                        "none_route",
                        "llm_articulation_only",
                        "capability_gate",
                        "commitment_gate",
                    ],
                }

            # Vector indexing now handled by engine.remember()

            print(f"  [Brain] Response complete ({len(reply)} chars)")
        except Exception as exc:
            reply = "I apologize, I had trouble processing that."
            logger.exception("Response error: %s", exc)
            await _broadcast_chunk_sync(reply, engine.get_state()["tone"])
            _broadcast({"type": "response_end", "text": "", "tone": engine.get_state()["tone"], "phase": "LISTENING"})
        finally:
            try:
                from skills.capability_gate import capability_gate as _none_gate_clr
                _none_gate_clr.set_route_hint(None)
            except Exception:
                pass

    if not _tts_done.is_set():
        try:
            await _flush_tts()
        except Exception:
            pass
    tts_worker_task.cancel()

    if _cancelled():
        _broadcast({"type": "response_end", "text": "", "tone": tone, "phase": "LISTENING"})

    _latency_ms = int((_time.time() - _conv_start) * 1000)
    _latency_mono_ms = int((_time.monotonic() - _conv_mono_start) * 1000)
    logger.info("[LATENCY] conversation_total=%dms route=%s cancelled=%s (conv=%s)",
                _latency_mono_ms, routing.tool.value if routing else "?",
                _cancelled(), conversation_id[:8] if conversation_id else "?")

    if not reply and not _cancelled():
        reply = "I'm sorry, I wasn't able to form a response to that."
        logger.warning("Empty response invariant: tool=%s, reply_len=0, sending fallback", routing.tool.value)
        await _broadcast_chunk_sync(reply, engine.get_state().get("tone", "professional"))
        _broadcast({"type": "response_end", "text": "", "tone": engine.get_state().get("tone", "professional"), "phase": "LISTENING"})

    ops_tracker.end_activity("conversation", detail=f"{_latency_ms}ms, {len(reply)} chars")
    ops_tracker.set_subsystem("retrieval", "idle")
    ops_tracker.set_subsystem("reasoning", "idle", f"last response: {_latency_ms}ms")
    ops_tracker.log_event("reasoning", "response_sent", f"Complete: {_latency_ms}ms, tool={routing.tool.value}")

    # Phase 6.4: Build compact provenance trace for response metadata
    _provenance_meta: dict[str, Any] = {}
    try:
        from reasoning.explainability import compact_trace
        _provenance_meta = compact_trace(_language_example_seed)
    except Exception:
        pass
    if not _provenance_meta or not _provenance_meta.get("provenance"):
        _provenance_meta = {
            "provenance": "fallback_unclassified",
            "source": "fallback:conversation_handler",
            "confidence": 0.0,
            "native": False,
            "response_class": str(getattr(routing.tool, "value", "unknown") or "unknown").lower(),
            "fallback": True,
        }
    _golden_meta = routing.golden_context.to_dict() if routing.golden_context else {}
    _output_id = f"out_{_uuid_mod.uuid4().hex[:12]}"
    _validation_decision = output_release_validator.validate_output(
        text=reply,
        conversation_id=_trace_ctx.conversation_id,
        trace_id=_trace_ctx.trace_id,
        request_id=_trace_ctx.request_id,
        output_id=_output_id,
        release_status="released",
        release_reason="",
        source="conversation_handler",
    )

    event_bus.emit(
        CONVERSATION_RESPONSE,
        text=reply,
        conversation_id=_trace_ctx.conversation_id,
        trace_id=_trace_ctx.trace_id,
        request_id=_trace_ctx.request_id,
        output_id=_output_id,
        validation_id=_validation_decision.validation_id,
        validation_passed=_validation_decision.passed,
        validation_violations=list(_validation_decision.violations),
        release_status=_validation_decision.effective_release_status,
        release_reason=_validation_decision.effective_release_reason,
        tool=routing.tool.value,
        latency_ms=_latency_ms,
        playback_estimate_s=round(_playback_estimate_s, 2),
        provenance=_provenance_meta,
        golden=_golden_meta,
    )
    _response_ledger_id = ""
    try:
        from consciousness.attribution_ledger import attribution_ledger
        _ledger_data: dict[str, Any] = {
            "latency_ms": _latency_ms,
            "tool": routing.tool.value,
            "reply_len": len(reply),
            "output_id": _output_id,
            "trace_id": _trace_ctx.trace_id,
            "request_id": _trace_ctx.request_id,
            "validation_id": _validation_decision.validation_id,
            "validation_passed": _validation_decision.passed,
            "release_status": _validation_decision.effective_release_status,
            "release_reason": _validation_decision.effective_release_reason,
            "provenance": _provenance_meta,
        }
        if _golden_meta:
            _ledger_data["golden"] = _golden_meta
        _response_ledger_id = attribution_ledger.record(
            subsystem="conversation",
            event_type="response_complete",
            actor="system",
            source=routing.tool.value,
            conversation_id=_trace_ctx.conversation_id,
            data=_ledger_data,
            parent_entry_id=_conv_ledger_id if _conv_ledger_id else "",
            evidence_refs=[{"kind": "conversation", "id": _trace_ctx.conversation_id}],
        )
    except Exception:
        pass

    try:
        from epistemic.calibration import TruthCalibrationEngine
        _tce_resp = TruthCalibrationEngine.get_instance()
        if _tce_resp:
            _tce_resp.record_response()
    except Exception:
        pass

    was_cancelled = _cancelled()
    had_error = reply.startswith("I apologize")
    _text_lower = text.lower()
    positive = any(w in _text_lower for w in ("thanks", "thank you", "great", "perfect", "awesome"))
    negative = _looks_like_negative_feedback(text)
    correction_result: dict[str, Any] | None = None

    if negative:
        try:
            _cs = getattr(engine, "_consciousness", None)
            _tce = getattr(_cs, "_truth_calibration_engine", None) if _cs else None
            _prev_resp = ""
            _po = getattr(engine, "_perception_orchestrator", None)
            if _po:
                _prev_resp = getattr(_po, "_last_response_text", "")
            _injected_payloads: list[str] = []
            for item in getattr(response_gen, "_last_injected_memories", []) or []:
                if not isinstance(item, dict):
                    continue
                payload = item.get("formatted") or item.get("payload")
                if payload:
                    _injected_payloads.append(str(payload))
            _prev_conf: float | None = None
            if _po:
                _prev_conf = getattr(_po, "_last_response_confidence", None)
            if _tce:
                correction_result = _tce.check_correction(
                    user_text=text,
                    is_negative=True,
                    last_response_text=_prev_resp,
                    last_tool_route=routing.tool.value,
                    injected_memory_payloads=_injected_payloads,
                    response_confidence=_prev_conf,
                )
            if correction_result:
                _correction_kind = str(correction_result.get("correction_kind", "factual_mismatch") or "factual_mismatch")
                _authority_domain = str(correction_result.get("authority_domain", "objective_or_external_fact") or "objective_or_external_fact")
                _adjudication_policy = str(correction_result.get("adjudication_policy", "require_evidence") or "require_evidence")
                _auto_accept = bool(correction_result.get("auto_accept_user_correction", False))
                _corr_mem_id = _store_user_correction_memory(
                    user_text=text,
                    last_response_text=_prev_resp,
                    route=routing.tool.value,
                    speaker=speaker,
                    correction_kind=_correction_kind,
                    authority_domain=_authority_domain,
                    adjudication_policy=_adjudication_policy,
                    injected_memory_payloads=_injected_payloads,
                )
                _hit_idxs = {
                    int(i)
                    for i in correction_result.get("overlap_memory_indices", []) or []
                    if isinstance(i, int) or str(i).isdigit()
                }
                for idx, item in enumerate(getattr(response_gen, "_last_injected_memories", []) or []):
                    if idx not in _hit_idxs:
                        continue
                    if not isinstance(item, dict):
                        continue
                    mid = str(item.get("id", "") or "")
                    if not mid:
                        continue
                    _mark_corrected_memory(
                        mid,
                        correction_kind=_correction_kind,
                        authority_domain=_authority_domain,
                        auto_accept_user_correction=_auto_accept,
                        correction_memory_id=_corr_mem_id,
                    )
        except Exception:
            pass

    latency_ms = int((_time.time() - _conv_start) * 1000)

    # Phase 5.1a: Friction detection
    _friction_event = None
    try:
        from autonomy.friction_miner import get_friction_miner
        _friction_miner = get_friction_miner()
        _route_str_for_friction = routing.tool.value if routing and hasattr(routing, "tool") else "unknown"
        _resp_class = getattr(routing, "response_class", "") if routing else ""
        _friction_event = _friction_miner.detect(
            user_text=text,
            assistant_text=reply,
            route_class=_route_str_for_friction,
            response_class=_resp_class if _resp_class else "",
            correction_result=correction_result,
            conversation_id=conversation_id or "",
        )
        if _friction_event:
            logger.info("Friction detected: type=%s severity=%s cluster=%s",
                        _friction_event.friction_type, _friction_event.severity,
                        _friction_event.cluster_key)
            if _friction_event.friction_type in ("too_cautious", "correction"):
                try:
                    from skills.capability_gate import capability_gate as _cg
                    _cg.record_friction_correction(_friction_event.timestamp)
                except Exception:
                    pass
    except Exception:
        pass

    if conversation_id:
        _outcome = "barge_in" if was_cancelled else ("ok" if reply else "error")
        _user_sig = ""
        if positive:
            _user_sig = "positive"
        elif correction_result:
            _user_sig = "correction"
        elif negative:
            _user_sig = "negative"
        elif follow_up:
            _user_sig = "follow_up"

        if _outcome == "ok" and _user_sig == "positive" and not correction_result:
            try:
                from epistemic.calibration import TruthCalibrationEngine
                _tce_pos = TruthCalibrationEngine.get_instance()
                if _tce_pos:
                    _seed_conf = (_language_example_seed or {}).get("confidence") if _language_example_seed else None
                    _tce_pos.record_positive_response_outcome(
                        route=routing.tool.value if routing and hasattr(routing, "tool") else "",
                        response_confidence=_seed_conf,
                    )
            except Exception:
                pass

        try:
            _po_conf = getattr(engine, "_perception_orchestrator", None)
            if _po_conf:
                _po_conf._last_response_confidence = (
                    (_language_example_seed or {}).get("confidence")
                    if _language_example_seed else None
                )
        except Exception:
            pass

        _record_language_corpus_example(
            _language_example_seed,
            query=text,
            reply=reply,
            conversation_id=conversation_id,
            user_feedback=_user_sig,
        )
        if _gate_rewrite_buffer:
            _route_str = routing.tool.value if routing and hasattr(routing, "tool") else "unknown"
            for _orig, _rewritten in _gate_rewrite_buffer:
                _record_negative_corpus_example(
                    query=text,
                    original_reply=_orig,
                    rewritten_reply=_rewritten,
                    conversation_id=conversation_id,
                    route=_route_str,
                    reason="capability_gate_rewrite",
                )
            _gate_rewrite_buffer.clear()
        if _friction_event and _friction_event.severity in ("medium", "high", "critical"):
            _route_str_neg = routing.tool.value if routing and hasattr(routing, "tool") else "unknown"
            _record_negative_corpus_example(
                query=text,
                original_reply=reply,
                rewritten_reply=_friction_event.candidate_rewrite or "",
                conversation_id=conversation_id,
                route=_route_str_neg,
                reason=f"friction:{_friction_event.friction_type}",
            )
        _record_language_quality_event(
            _language_example_seed,
            query=text,
            reply=reply,
            conversation_id=conversation_id,
            outcome=_outcome,
            user_feedback=_user_sig,
        )
        _record_ambiguous_intent_probe(
            _ambiguous_probe_seed,
            conversation_id=conversation_id,
            outcome=_outcome,
            user_feedback=_user_sig,
        )
        try:
            from library.telemetry import retrieval_telemetry
            retrieval_telemetry.log_retrieval_outcome(
                conversation_id=conversation_id,
                outcome=_outcome,
                latency_ms=float(_latency_ms),
            )
            if _outcome == "ok" and reply:
                _reinforce_retrieval(conversation_id, positive=True)
            elif _outcome in ("error", "barge_in"):
                _reinforce_retrieval(conversation_id, positive=False)
        except Exception:
            pass

        outcome_counters.attempts += 1
        try:
            from memory.retrieval_log import memory_retrieval_log
            from memory.ranker import get_memory_ranker
            memory_retrieval_log.log_outcome(
                conversation_id=conversation_id,
                outcome=_outcome,
                latency_ms=float(_latency_ms),
                user_signal=_user_sig,
            )
            ranker = get_memory_ranker()
            if ranker and ranker.is_ready():
                ranker.record_outcome(_outcome == "ok")
            outcome_counters.successes += 1
        except Exception as exc:
            outcome_counters.failures += 1
            outcome_counters.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Cortex outcome logging failed: %s", exc)

        _real_resp_conf = (_language_example_seed or {}).get("confidence") if _language_example_seed else None
        _ledger_conf = _real_resp_conf if _real_resp_conf is not None else (0.50 if was_cancelled or had_error else 0.60)

        if _conv_ledger_id:
            try:
                from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
                _ledger_outcome = "success" if _outcome == "ok" else ("failure" if _outcome == "error" else _outcome)
                attribution_ledger.record_outcome(_conv_ledger_id, _ledger_outcome, build_outcome_data(
                    confidence=_ledger_conf,
                    latency_s=round(_latency_ms / 1000.0, 2),
                    source="user_feedback",
                    tier="immediate",
                    scope="response_quality",
                    blame_target="response_generation",
                    user_signal=_user_sig,
                    was_cancelled=was_cancelled,
                    had_error=had_error,
                ))
            except Exception:
                pass

        if _response_ledger_id:
            try:
                from consciousness.attribution_ledger import attribution_ledger, build_outcome_data
                _resp_outcome = "success" if _outcome == "ok" else ("failure" if _outcome == "error" else _outcome)
                attribution_ledger.record_outcome(_response_ledger_id, _resp_outcome, build_outcome_data(
                    confidence=_ledger_conf,
                    latency_s=round(_latency_ms / 1000.0, 2),
                    source="user_feedback",
                    tier="immediate",
                    scope="response_quality",
                    blame_target="response_generation",
                    user_signal=_user_sig,
                    was_cancelled=was_cancelled,
                    had_error=had_error,
                    reply_len=len(reply),
                    tool=routing.tool.value,
                ))
            except Exception:
                pass

    if episodes and routing.tool != ToolType.WEB_SEARCH:
        _root_entry_id = _conv_ledger_id or _response_ledger_id
        episodes.add_assistant_turn(
            reply,
            conversation_id=_trace_ctx.conversation_id,
            trace_id=_trace_ctx.trace_id,
            request_id=_trace_ctx.request_id,
            output_id=_output_id,
            conversation_entry_id=_conv_ledger_id,
            response_entry_id=_response_ledger_id,
            root_entry_id=_root_entry_id,
        )

    _STREAMED_TOOLS = {ToolType.NONE, ToolType.INTROSPECTION, ToolType.VISION,
                       ToolType.TIME, ToolType.SYSTEM_STATUS, ToolType.STATUS,
                       ToolType.MEMORY,
                       ToolType.CAMERA_CONTROL, ToolType.WEB_SEARCH, ToolType.ACADEMIC_SEARCH,
                       ToolType.CODEBASE, ToolType.SELF_IMPROVE, ToolType.IDENTITY,
                       ToolType.PERFORM,
                       ToolType.SKILL, ToolType.LIBRARY_INGEST}
    if routing.tool not in _STREAMED_TOOLS and perception:
        _resp_msg: dict[str, Any] = {
            "type": "response",
            "text": reply,
            "tone": engine.get_state()["tone"],
            "phase": "SPEAKING",
        }
        if _provenance_meta:
            _resp_msg["provenance"] = _provenance_meta
        _broadcast(_resp_msg)
    engine.set_phase("LISTENING")

    if episodes and ollama:
        ep = episodes.get_active_episode()
        if ep and ep.turn_count() >= 2:
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    episodes.summarize_episode_llm(ep, ollama)
                )
            )

    # ── Flight Recorder: cognitive black box episode ─────────────
    try:
        _epi_flags: dict[str, Any] = {}
        try:
            from epistemic.contradiction_engine import contradiction_engine
            _ce = contradiction_engine
            _epi_flags["contradiction_touched"] = (_ce._debt if hasattr(_ce, '_debt') else 0) > 0
        except Exception:
            _epi_flags["contradiction_touched"] = False
        _tc = engine.get_state().get("truth_calibration", {})
        _epi_flags["provisional"] = (_tc.get("truth_score") or 1.0) < 0.6

        _id_state: dict[str, Any] = {}
        try:
            _id_state["resolved"] = (speaker_state or {}).get("name") or speaker
            _id_state["confidence"] = (speaker_state or {}).get("confidence", 0)
        except Exception:
            pass

        _disagreements = 0
        try:
            _core = engine.get_state()
            if _core.get("is_user_present"):
                _sc = engine._scene_tracker.get_state() if hasattr(engine, '_scene_tracker') else {}
                if isinstance(_sc, dict) and _sc.get("visible_persons", 1) == 0:
                    _disagreements += 1
        except Exception:
            pass

        _retrieval_summary: dict[str, Any] = {
            "count": 0,
            "subjects": {},
            "types": {},
            "route_type": "no_retrieval",
            "search_scope": "none",
        }
        try:
            from reasoning.response import get_last_retrieval_summary
            _retrieval_summary = get_last_retrieval_summary()
        except Exception:
            pass
        if routing.tool == ToolType.MEMORY and int(_retrieval_summary.get("count", 0) or 0) == 0:
            try:
                from tools.memory_tool import get_last_memory_tool_summary
                _tool_retrieval_summary = get_last_memory_tool_summary()
                if int(_tool_retrieval_summary.get("count", 0) or 0) > 0:
                    _retrieval_summary = _tool_retrieval_summary
            except Exception:
                pass
        if routing.tool == ToolType.MEMORY and int(_retrieval_summary.get("count", 0) or 0) == 0:
            try:
                _parsed_memory_summary = _derive_memory_summary_from_context(memory_ctx)
                if _parsed_memory_summary:
                    _retrieval_summary = _parsed_memory_summary
            except Exception:
                pass

        _ep_record: dict[str, Any] = {
            "id": str(_uuid_mod.uuid4()),
            "timestamp": _time.time(),
            "user_input": text[:500],
            "speaker": speaker,
            "emotion": emotion,
            "tool_route": routing.tool.value,
            "response_latency_ms": _latency_ms,
            "response_text": reply[:500] if reply else "",
            "memories_retrieved": _retrieval_summary,
            "epistemic_flags": _epi_flags,
            "identity_state": _id_state,
            "disagreement_count": _disagreements,
            "barged_in": was_cancelled,
            "follow_up": follow_up,
            "conversation_id": _trace_ctx.conversation_id,
            "trace_id": _trace_ctx.trace_id,
            "request_id": _trace_ctx.request_id,
            "output_id": _output_id,
            "conversation_entry_id": _conv_ledger_id,
            "response_entry_id": _response_ledger_id,
            "root_entry_id": _conv_ledger_id or _response_ledger_id,
            "golden": _golden_meta,
        }
        _flight_recorder.append(_ep_record)
        _record_golden_outcome(routing.golden_context, tool_route=routing.tool.value)
        _save_flight_recorder()
    except Exception:
        logger.debug("Flight recorder episode failed", exc_info=True)

    engine.record_interaction_outcome(
        completed=not was_cancelled and not had_error,
        barged_in=was_cancelled,
        follow_up=follow_up,
        positive_feedback=positive,
        negative_feedback=negative,
        error=had_error,
        latency_ms=latency_ms,
        user_emotion=emotion,
        curiosity_outcome=_curiosity_outcome,
        correction=correction_result is not None,
        route=routing.tool.value if routing else "",
    )

    soul_service.record_interaction(speaker)
    soul_service.save_identity()

    complexity = _score_complexity(text)
    if complexity != "simple":
        try:
            reflection_engine.generate_meta_learning(
                user_message=text,
                response_text=reply,
                complexity=complexity,
                barged_in=was_cancelled,
                latency_ms=int((_time.time() - _conv_start) * 1000),
                user_emotion=emotion,
                speaker=speaker,
            )
        except Exception:
            logger.debug("Meta-learning reflection failed", exc_info=True)
