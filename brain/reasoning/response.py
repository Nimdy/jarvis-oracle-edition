"""Response generator — full reasoning pipeline from user message to reply."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Literal

from consciousness.engine import ConsciousnessEngine
from memory.core import CreateMemoryData
from memory.search import semantic_search
from reasoning.ollama_client import OllamaClient
from reasoning.context import context_builder

logger = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

_REL_WORDS_RE = re.compile(
    r"\b(?:my\s+)?(wife|husband|partner|girlfriend|boyfriend|"
    r"mom|mother|dad|father|brother|sister|son|daughter|"
    r"friend|roommate|boss|coworker|colleague)\b",
    re.IGNORECASE,
)

# ── Conversation Memory Router ────────────────────────────────────────
RouteType = Literal[
    "self_preference",
    "referenced_person",
    "episodic",
    "belief_synthesis",
    "no_retrieval",
    "general",
]

SearchScope = Literal[
    "primary_user_only",
    "referenced_subject_only",
    "recent_conversation",
    "belief_graph",
    "full",
    "none",
]


@dataclass(frozen=True)
class MemoryRoute:
    route_type: RouteType
    referenced_entities: set[str] = field(default_factory=set)
    allow_preference_injection: bool = True
    allow_thirdparty_injection: bool = False
    allow_autonomy_recall: bool = False
    search_scope: SearchScope = "full"

    def __repr__(self) -> str:
        refs = ",".join(sorted(self.referenced_entities)) if self.referenced_entities else ""
        return f"MemoryRoute({self.route_type}, refs=[{refs}], scope={self.search_scope})"


_SELF_PREF_RE = re.compile(
    r"\b(?:what (?:do|did|does) I|my (?:favo[u]?rite|preference)|"
    r"what (?:foods?|things?|music|movies?|books?) (?:do )?I|"
    r"do I (?:like|love|enjoy|prefer|hate|dislike)|"
    r"what I (?:like|love|told you|said|mentioned))\b",
    re.IGNORECASE,
)

_REL_WORDS = (
    r"(?:wife|husband|partner|girlfriend|boyfriend|"
    r"mom|mother|dad|father|brother|sister|son|daughter|"
    r"friend|roommate|boss|coworker|colleague)"
)

_REFERENCED_PERSON_RE = re.compile(
    r"\b(?:what (?:does|did|do) (?:my\s+)?" + _REL_WORDS + r"|"
    r"what (?:my\s+)?" + _REL_WORDS + r"\s+(?:like|love|prefer|enjoy|hate)|"
    r"(?:tell me|can you tell me) (?:about|what) (?:my\s+)?" + _REL_WORDS + r")",
    re.IGNORECASE,
)

_EPISODIC_RE = re.compile(
    r"\b(?:what (?:did I (?:just|last)|were we|was I)|"
    r"(?:earlier|before|last time|previously) (?:you said|I said|we talked)|"
    r"remind me what|what did (?:you|I) (?:say|tell|mention))\b",
    re.IGNORECASE,
)

_BELIEF_RE = re.compile(
    r"\b(?:what (?:patterns?|trends?|themes?) do you|"
    r"what (?:have you )?notic|what kinds? of|"
    r"do you see (?:a |any )?pattern|"
    r"what do you (?:think|believe|observe) about)\b",
    re.IGNORECASE,
)

_NO_RETRIEVAL_RE = re.compile(
    r"^(?:jarvis[\s,]+)?(?:thanks?|thank you|ok(?:ay)?|sure|got it|no|yes|"
    r"good|great|cool|nice|welcome|hello|hi|hey|bye|goodbye|"
    r"that'?s (?:okay|fine|good|great|it|all)|never ?mind|"
    r"no,? (?:that'?s|it'?s) (?:okay|fine|good)|"
    r"you'?re welcome|sounds good|perfect|alright)(?:[\s,]+jarvis)?[.!? ]*$",
    re.IGNORECASE,
)
_LOW_INFORMATION_GENERAL_RE = re.compile(
    r"^(?:jarvis[\s,]+)?(?:day|date|today|tonight|morning|evening|afternoon|"
    r"birthday|bday|huh|hmm|hmm+|what|why|okay|ok|sure|right|check|checkpoint|help)"
    r"(?:[\s,]+jarvis)?[.!? ]*$",
    re.IGNORECASE,
)

_last_memory_route: MemoryRoute | None = None
_last_retrieval_summary: dict[str, object] = {
    "count": 0,
    "subjects": {},
    "types": {},
    "route_type": "no_retrieval",
    "search_scope": "none",
}


def get_last_memory_route() -> MemoryRoute | None:
    return _last_memory_route


def get_last_retrieval_summary() -> dict[str, object]:
    return dict(_last_retrieval_summary)


def route_memory_request(
    user_message: str,
    referenced_entities: set[str] | None = None,
) -> MemoryRoute:
    """Classify a user message into a memory retrieval route.

    Called before semantic search to determine scope, preference injection,
    and whether autonomy recall should be suppressed.
    """
    global _last_memory_route, _last_retrieval_summary
    refs = referenced_entities or set()
    text = user_message.strip()

    if _NO_RETRIEVAL_RE.match(text):
        route = MemoryRoute(
            route_type="no_retrieval",
            allow_preference_injection=False,
            allow_autonomy_recall=False,
            search_scope="none",
        )
        _last_memory_route = route
        _last_retrieval_summary = {
            "count": 0,
            "subjects": {},
            "types": {},
            "route_type": route.route_type,
            "search_scope": route.search_scope,
        }
        return route

    if (
        len(text.split()) <= 2
        and not refs
        and "?" not in text
        and _LOW_INFORMATION_GENERAL_RE.match(text)
    ):
        route = MemoryRoute(
            route_type="general",
            allow_preference_injection=False,
            allow_thirdparty_injection=False,
            allow_autonomy_recall=False,
            search_scope="none",
        )
        _last_memory_route = route
        _last_retrieval_summary = {
            "count": 0,
            "subjects": {},
            "types": {},
            "route_type": route.route_type,
            "search_scope": route.search_scope,
        }
        return route

    has_relation_ref = bool(refs) or bool(_REL_WORDS_RE.search(text))
    is_referenced_person_q = bool(_REFERENCED_PERSON_RE.search(text))

    if is_referenced_person_q or (has_relation_ref and "?" in text):
        actual_refs = refs.copy()
        for m in _REL_WORDS_RE.finditer(text):
            actual_refs.add(m.group(1).lower())
        route = MemoryRoute(
            route_type="referenced_person",
            referenced_entities=actual_refs,
            allow_preference_injection=False,
            allow_thirdparty_injection=True,
            allow_autonomy_recall=False,
            search_scope="referenced_subject_only",
        )
        _last_memory_route = route
        return route

    if _EPISODIC_RE.search(text):
        route = MemoryRoute(
            route_type="episodic",
            allow_preference_injection=False,
            allow_autonomy_recall=False,
            search_scope="recent_conversation",
        )
        _last_memory_route = route
        return route

    if _SELF_PREF_RE.search(text):
        route = MemoryRoute(
            route_type="self_preference",
            referenced_entities=refs,
            allow_preference_injection=True,
            allow_autonomy_recall=False,
            search_scope="primary_user_only",
        )
        _last_memory_route = route
        return route

    if _BELIEF_RE.search(text):
        route = MemoryRoute(
            route_type="belief_synthesis",
            allow_preference_injection=False,
            allow_autonomy_recall=False,
            search_scope="belief_graph",
        )
        _last_memory_route = route
        return route

    route = MemoryRoute(
        route_type="general",
        referenced_entities=refs,
        allow_preference_injection=True,
        allow_thirdparty_injection=has_relation_ref,
        allow_autonomy_recall=False,
        search_scope="full",
    )
    _last_memory_route = route
    return route


def _extract_referenced_entities(
    text: str, known_names: set[str],
) -> set[str]:
    """Extract entity names referenced in query text for boundary exceptions.

    Performs canonical expansion so that "wife" also matches "_rel_wife"
    and any resolved spouse name. This ensures boundary engine subject_id
    matching works regardless of how the memory was originally stored.
    """
    found: set[str] = set()
    text_lower = text.lower()
    for name in known_names:
        if name in text_lower:
            found.add(name)
    for m in _REL_WORDS_RE.finditer(text):
        found.add(m.group(1).lower())

    if found:
        try:
            from identity.resolver import identity_resolver
            found = identity_resolver.expand_reference_aliases(found)
        except Exception:
            for ref in list(found):
                found.add(f"_rel_{ref}")

    return found


def _supplement_subject_memories(
    existing: list | None,
    referenced_entities: set[str],
    identity_context: object | None = None,
    top_k: int = 5,
) -> list | None:
    """Directly scan memory storage for subject-matching memories.

    When search_scope is "referenced_subject_only", vector search alone may miss
    subject-specific memories that don't have strong embedding similarity to the
    query. This supplements by scanning user_preference memories whose
    identity_subject matches any of the expanded reference aliases.
    """
    try:
        from memory.storage import memory_storage

        existing_ids = {getattr(m, "id", "") for m in (existing or [])}
        subject_hits: list = []

        ref_lower = {r.lower() for r in referenced_entities}
        for m in memory_storage.get_by_tag("user_preference"):
            if m.id in existing_ids:
                continue
            subject = getattr(m, "identity_subject", "")
            if subject and subject.lower() in ref_lower:
                subject_hits.append(m)
            else:
                tags = set(getattr(m, "tags", []))
                for tag in tags:
                    if tag.startswith("relation:"):
                        role = tag.split(":", 1)[1]
                        if role in ref_lower or f"_rel_{role}" in ref_lower:
                            subject_hits.append(m)
                            break

        if subject_hits:
            logger.info(
                "Subject supplement: found %d subject-matching prefs for refs=%s",
                len(subject_hits), sorted(ref_lower),
            )
        else:
            logger.info(
                "Subject supplement: no subject-matching prefs for refs=%s",
                sorted(ref_lower),
            )
            return existing

        combined = list(existing or []) + subject_hits
        seen: set[str] = set()
        deduped: list = []
        for m in combined:
            mid = getattr(m, "id", id(m))
            if mid not in seen:
                seen.add(mid)
                deduped.append(m)

        return deduped[:top_k] if len(deduped) > top_k else deduped
    except Exception:
        return existing


def _fmt(mem: object) -> str:
    """Format a memory payload the same way the LLM prompt sees it."""
    try:
        from reasoning.context import _format_memory_payload
        return _format_memory_payload(mem)  # type: ignore[arg-type]
    except Exception:
        payload = getattr(mem, "payload", "")
        return str(payload)[:200] if payload else ""

_COMPLEX_SIGNALS = re.compile(
    r'\b(explain|how does|why|compare|difference|analyze|what if|elaborate|'
    r'pros and cons|trade.?off|algorithm|architecture|design|implement|debug|'
    r'step by step|in detail|thoroughly|deep dive)\b',
    re.IGNORECASE,
)
_SIMPLE_SIGNALS = re.compile(
    r'^(what time|what\'s the time|what day|thanks|thank you|ok|okay|'
    r'yes|no|sure|got it|hello|hi|hey|bye|good morning|good night|'
    r'how are you|how you doing|how ya doing|what\'s up)[\s?!.]*$',
    re.IGNORECASE,
)


def _score_complexity(text: str) -> str:
    """Score user message complexity. Returns 'simple', 'moderate', or 'complex'."""
    if _SIMPLE_SIGNALS.match(text.strip()):
        return "simple"
    complex_hits = len(_COMPLEX_SIGNALS.findall(text))
    word_count = len(text.split())
    if complex_hits >= 2 or word_count > 50:
        return "complex"
    if complex_hits >= 1 or word_count > 20:
        return "moderate"
    return "simple"


def _extract_surfaced_chunk_ids(memories: list) -> list[str]:
    """Extract chunk_ids from memory payloads that reference library chunks.

    These are the chunks that were *eligible* for injection based on retrieval,
    before prompt assembly decides which ones actually get resolved.
    Only study_claim payloads have real chunk_ids; library_pointer source_ids
    are already logged separately in the telemetry source_ids field.
    """
    ids: list[str] = []
    for m in memories:
        p = m.payload if isinstance(getattr(m, "payload", None), dict) else {}
        if p.get("type") == "study_claim":
            ids.extend(p.get("chunk_ids", []))
    return ids


@dataclass
class GeneratedResponse:
    text: str
    memory_tags: list[str]
    suggested_tone: str | None = None
    latency_ms: int = 0


class ResponseGenerator:
    def __init__(self, engine: ConsciousnessEngine, ollama_config: dict | None = None) -> None:
        self._engine = engine
        self._episodes = None
        cfg = ollama_config or {}
        self._ollama = OllamaClient(
            host=cfg.get("host", "http://localhost:11434"),
            model=cfg.get("model", "qwen3:8b"),
            vision_model=cfg.get("vision_model", "qwen2.5vl:7b"),
            temperature=cfg.get("temperature", 0.7),
            max_tokens=cfg.get("max_tokens", 1024),
            keep_alive=cfg.get("keep_alive", "5m"),
        )
        self._fast_model: str = cfg.get("fast_model", "")
        self._multimodal = None
        self._last_injected_memories: list[dict] = []

    def set_episodes(self, episodes) -> None:
        self._episodes = episodes

    def _build_context(
        self,
        user_message: str,
        perception_context: str | None = None,
        speaker_name: str | None = None,
        user_emotion: str | None = None,
        conversation_id: str = "",
        complexity: str = "",
        style_instruction: str = "",
        persist_conversation: bool = True,
        tool_hint: str | None = None,
    ) -> tuple[str, list[dict[str, str]], dict]:
        """Build system prompt, message history, and trait-adjusted LLM params."""
        state = self._engine.get_state()
        recent_memories = self._engine.get_recent_memories(10)
        modulation = self._engine.get_trait_modulation()

        from reasoning.context import reset_resolved_chunks
        reset_resolved_chunks(conversation_id or "")

        _identity_ctx = None
        _ref_entities: set[str] | None = None
        try:
            from identity.resolver import identity_resolver
            _identity_ctx = identity_resolver.resolve_for_memory(
                provenance="conversation", speaker=speaker_name or "",
            )
            _ref_entities = _extract_referenced_entities(
                user_message, identity_resolver.get_known_names(),
            )
        except Exception:
            pass

        route = route_memory_request(user_message, _ref_entities)
        logger.info(
            "Memory route: type=%s scope=%s refs=%s inject_pref=%s inject_third=%s",
            route.route_type, route.search_scope,
            sorted(route.referenced_entities) if route.referenced_entities else "[]",
            route.allow_preference_injection, route.allow_thirdparty_injection,
        )

        sem_memories = None
        if route.search_scope != "none":
            refs_for_search = route.referenced_entities or _ref_entities
            sem_memories = semantic_search(
                user_message, top_k=5, speaker=speaker_name or "",
                conversation_id=conversation_id,
                identity_context=_identity_ctx,
                referenced_entities=refs_for_search,
            ) or None

            if route.search_scope == "referenced_subject_only" and refs_for_search:
                sem_memories = _supplement_subject_memories(
                    sem_memories, refs_for_search, _identity_ctx, top_k=5,
                )

        _sem_count = len(sem_memories) if sem_memories else 0
        _sem_subjects: dict[str, int] = {}
        _sem_types: dict[str, int] = {}
        if sem_memories:
            for _m in sem_memories:
                subj = getattr(_m, "identity_subject", "") or "none"
                _sem_subjects[subj] = _sem_subjects.get(subj, 0) + 1
                _sem_types[_m.type] = _sem_types.get(_m.type, 0) + 1
        logger.info(
            "Retrieval: %d memories, subjects=%s types=%s",
            _sem_count, dict(_sem_subjects), dict(_sem_types),
        )
        _last_retrieval_summary = {
            "count": _sem_count,
            "subjects": dict(_sem_subjects),
            "types": dict(_sem_types),
            "route_type": route.route_type,
            "search_scope": route.search_scope,
        }

        surfaced_chunk_ids = _extract_surfaced_chunk_ids(sem_memories or [])

        episodic_context = ""
        if self._episodes:
            episodic_context = self._episodes.get_conversation_context(max_episodes=3)
            related = self._episodes.find_episodes_semantic(user_message, limit=3)
            seen_summaries = set()
            if episodic_context:
                for line in episodic_context.splitlines():
                    seen_summaries.add(line.strip())
            for ep in related:
                summary_line = f"- Past conversation about '{ep.topic}': {ep.summary[:120]}"
                if ep.summary and summary_line.strip() not in seen_summaries:
                    episodic_context += f"\n{summary_line}"
                    seen_summaries.add(summary_line.strip())

        combined_perception = perception_context or ""
        if episodic_context:
            combined_perception = (combined_perception + "\n" + episodic_context).strip()

        length_hint = getattr(self._engine, "policy_response_length", "")

        system_prompt = context_builder.build_system_prompt(
            state, list(modulation.applied_traits), recent_memories,
            combined_perception or None,
            semantic_memories=sem_memories,
            speaker_name=speaker_name,
            user_emotion=user_emotion,
            response_length_hint=length_hint,
            dominant_trait=modulation.dominant_trait,
            modulation_strength=modulation.modulation_strength,
            complexity_hint=complexity,
            conversation_id=conversation_id or "",
            memory_route=route,
            style_instruction=style_instruction,
            tool_hint=tool_hint,
        )
        from reasoning.context import get_resolved_chunks
        resolved = get_resolved_chunks(conversation_id or "")
        if (resolved or surfaced_chunk_ids) and conversation_id:
            try:
                from library.telemetry import retrieval_telemetry
                injected_chunk_ids = [r["chunk_id"] for r in resolved]
                retrieval_telemetry.log_retrieval_started(
                    conversation_id=conversation_id,
                    query=user_message[:200],
                    source_ids=list({r["source_id"] for r in resolved}),
                    chunk_ids_surfaced=surfaced_chunk_ids,
                    chunk_ids_injected=injected_chunk_ids,
                )
            except Exception:
                pass

        self._last_injected_memories = []
        if conversation_id:
            try:
                from memory.search import get_last_retrieval_event_id
                from memory.retrieval_log import memory_retrieval_log
                from memory.lifecycle_log import memory_lifecycle_log
                ret_event_id = get_last_retrieval_event_id()
                if ret_event_id and sem_memories:
                    injected_mem_ids = [m.id for m in sem_memories if hasattr(m, "id")]
                    memory_retrieval_log.mark_injected(ret_event_id, injected_mem_ids)
                    for mid in injected_mem_ids:
                        memory_lifecycle_log.log_injected(mid)
                    self._last_injected_memories = [
                        {"id": m.id, "payload": m.payload, "formatted": _fmt(m)}
                        for m in sem_memories if hasattr(m, "id")
                    ]
            except Exception:
                pass

        if conversation_id:
            context_builder.set_conversation_id(conversation_id)
        if persist_conversation:
            context_builder.add_user_message(user_message, conversation_id=conversation_id)
        if conversation_id:
            messages = context_builder.get_conversation_context(conversation_id)
            if len(messages) <= 1:
                messages = context_builder.get_recent_context()
        else:
            messages = context_builder.get_recent_context()
        if not persist_conversation:
            messages = [*messages, {"role": "user", "content": user_message}]

        from personality.traits import trait_modulator
        base_max_tokens = self._ollama._max_tokens
        base_temp = self._ollama._temperature
        adjusted_tokens = trait_modulator.apply_modulation(
            "responseLength", float(base_max_tokens), modulation)
        adjusted_tokens = trait_modulator.apply_modulation(
            "responseDepth", adjusted_tokens, modulation)
        adjusted_tokens = int(max(256, min(2048, adjusted_tokens)))
        creativity = modulation.total_modifiers.get("creativityBoost", 1.0)
        adjusted_temp = base_temp + (creativity - 1.0) * 0.15
        adjusted_temp = round(max(0.3, min(1.2, adjusted_temp)), 2)

        llm_params = {
            "max_tokens": adjusted_tokens,
            "temperature": adjusted_temp,
        }

        return system_prompt, messages, llm_params

    def _finalize_response(self, user_message: str, response_text: str, start: float,
                           conversation_id: str = "",
                           complexity: str = "",
                           speaker_name: str | None = None,
                           user_emotion: str | None = None,
                           outcome: str = "completed",
                           persist_response: bool = True) -> GeneratedResponse:
        """Store memory and return structured response after LLM completes."""
        latency_ms = int((time.time() - start) * 1000)
        if persist_response:
            context_builder.add_assistant_message(response_text, conversation_id=conversation_id)
            context_builder.save()
        tags = self._extract_tags(user_message, response_text)
        payload: dict = {
            "user_message": user_message,
            "response": response_text,
        }
        if complexity:
            payload["complexity"] = complexity
        if user_emotion and user_emotion != "neutral":
            payload["user_emotion"] = user_emotion
        payload["latency_ms"] = latency_ms
        payload["outcome"] = outcome
        if speaker_name and speaker_name != "unknown":
            payload["speaker"] = speaker_name
            tags.append(f"speaker:{speaker_name}")

        if persist_response:
            try:
                from memory.retrieval_log import detect_memory_references, memory_retrieval_log
                from memory.search import get_last_retrieval_event_id
                if self._last_injected_memories:
                    ref_ids = detect_memory_references(response_text, self._last_injected_memories)
                    if ref_ids:
                        ret_event_id = get_last_retrieval_event_id()
                        if ret_event_id:
                            memory_retrieval_log.record_references(ret_event_id, ref_ids)
                            memory_retrieval_log._append({
                                "type": "reference",
                                "event_id": ret_event_id,
                                "referenced_memory_ids": ref_ids,
                                "t": round(time.time(), 3),
                            })
            except Exception:
                pass

            if self._is_low_value_response(response_text):
                logger.info("Skipping memory for low-value response: %s", response_text[:60])
            else:
                self._engine.remember(CreateMemoryData(
                    type="conversation",
                    payload=payload,
                    weight=0.55,
                    tags=tags,
                    provenance="conversation",
                ))
        else:
            logger.debug("Skipping conversation persistence for this response")
        self._engine.set_phase("LISTENING")
        return GeneratedResponse(
            text=response_text,
            memory_tags=tags,
            latency_ms=latency_ms,
        )

    async def respond(
        self,
        user_message: str,
        perception_context: str | None = None,
        speaker_name: str | None = None,
        user_emotion: str | None = None,
        conversation_id: str = "",
        style_instruction: str = "",
    ) -> GeneratedResponse:
        start = time.time()
        self._engine.set_phase("PROCESSING")

        system_prompt, messages, llm_params = self._build_context(
            user_message, perception_context,
            speaker_name=speaker_name, user_emotion=user_emotion,
            conversation_id=conversation_id,
            style_instruction=style_instruction,
        )

        try:
            response_text = await self._ollama.chat(messages, system_prompt)
            return self._finalize_response(user_message, response_text, start,
                                           conversation_id=conversation_id,
                                           speaker_name=speaker_name,
                                           user_emotion=user_emotion)
        except Exception as exc:
            logger.exception("Response generation failed")
            self._engine.set_phase("LISTENING")
            self._engine.remember(CreateMemoryData(
                type="error_recovery",
                payload={"error": str(exc), "user_message": user_message},
                weight=0.5,
                tags=["error", "llm_failure"],
                provenance="model_inference",
            ))
            return GeneratedResponse(
                text="I apologize, but I am having difficulty processing that right now. Could you try again?",
                memory_tags=["error"],
                latency_ms=int((time.time() - start) * 1000),
            )

    async def respond_stream(
        self,
        user_message: str,
        perception_context: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        speaker_name: str | None = None,
        user_emotion: str | None = None,
        conversation_id: str = "",
        tool_hint: str | None = None,
        persist_response: bool = True,
        style_instruction: str = "",
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """Stream response sentence-by-sentence as the LLM generates tokens.

        Yields (sentence_text, is_final) tuples. The last yield has is_final=True
        and the full response text so the caller can use it for memory/logging.

        If cancel_check is provided it is polled every ~30 tokens; on True the
        stream stops immediately and no further chunks are yielded.  The caller
        is responsible for sending response_end and resetting the phase.

        tool_hint: when set (e.g. "introspection"), prevents the fast-model
        override from capping grounded tool-routed responses at 150 tokens.
        """
        start = time.time()
        self._engine.set_phase("PROCESSING")

        complexity = _score_complexity(user_message)
        if tool_hint and complexity == "simple":
            complexity = "moderate"

        system_prompt, messages, llm_params = self._build_context(
            user_message, perception_context,
            speaker_name=speaker_name, user_emotion=user_emotion,
            conversation_id=conversation_id,
            complexity=complexity,
            persist_conversation=persist_response,
            style_instruction=style_instruction,
            tool_hint=tool_hint,
        )
        model_override = None
        max_tokens = llm_params.get("max_tokens")
        temperature = llm_params.get("temperature")
        if tool_hint == "introspection":
            temperature = 0.35
            max_tokens = max(max_tokens, 1024)
            logger.info("Introspection override: temp=%.2f, max_tokens=%d (grounding mode)",
                        temperature, max_tokens)
        elif tool_hint == "reflective_introspection":
            max_tokens = max(max_tokens, 1024)
            logger.info("Reflective introspection: temp=%.2f (trait-adjusted), max_tokens=%d (personality mode)",
                        temperature, max_tokens)
        elif tool_hint == "memory":
            temperature = 0.55
            max_tokens = max(max_tokens, 512)
            logger.info("Memory override: temp=%.2f, max_tokens=%d (relational recall)",
                        temperature, max_tokens)
        elif tool_hint == "general_knowledge":
            temperature = 0.6
            max_tokens = max(max_tokens, 512)
            logger.info("General knowledge override: temp=%.2f, max_tokens=%d (factual, main model)",
                        temperature, max_tokens)
        elif self._fast_model and complexity == "simple":
            model_override = self._fast_model
            max_tokens = 150
            temperature = 0.8
            logger.info("Routing to fast model (%s, max_tokens=%d) — simple query",
                        self._fast_model, max_tokens)
        elif complexity == "complex":
            max_tokens = int(max_tokens * 1.5)
            temperature = max(0.3, temperature - 0.1)
            logger.info("Complex query: max_tokens=%d, temp=%.2f (boosted depth, lower temp)",
                        max_tokens, temperature)
        else:
            logger.info("Moderate query: max_tokens=%d, temp=%.2f (trait-adjusted)",
                        max_tokens, temperature)

        buffer = ""
        full_text = ""
        sentences_sent = 0
        _CANCEL_CHECK_INTERVAL = 30
        token_count = 0
        cancelled = False
        _llm_t0 = time.monotonic()
        _first_token_logged = False

        try:
            async for token in self._ollama.chat_stream(messages, system_prompt,
                                                         model_override=model_override,
                                                         max_tokens=max_tokens,
                                                         temperature=temperature):
                if not _first_token_logged:
                    _ttft_ms = (time.monotonic() - _llm_t0) * 1000
                    logger.info("[LATENCY] llm_first_token=%.0fms model=%s prompt_chars=%d",
                                _ttft_ms, model_override or self._ollama._model,
                                len(system_prompt))
                    _first_token_logged = True
                buffer += token
                full_text += token
                token_count += 1

                if cancel_check and token_count % _CANCEL_CHECK_INTERVAL == 0 and cancel_check():
                    cancelled = True
                    logger.info("respond_stream cancelled after %d tokens", token_count)
                    break

                while True:
                    match = _SENTENCE_END_RE.search(buffer)
                    if not match:
                        break
                    sentence = buffer[:match.start() + 1].strip()
                    buffer = buffer[match.end():]
                    if sentence:
                        sentences_sent += 1
                        yield (sentence, False)

            if cancelled:
                return

            remaining = buffer.strip()
            if remaining:
                sentences_sent += 1
                yield (remaining, False)

            self._finalize_response(user_message, full_text, start,
                                   conversation_id=conversation_id,
                                   complexity=complexity,
                                   speaker_name=speaker_name,
                                   user_emotion=user_emotion,
                                   persist_response=persist_response)
            self._engine.consciousness.record_response_latency(
                int((time.time() - start) * 1000),
            )
            yield (full_text, True)

        except Exception as exc:
            logger.exception("Streaming response failed")
            self._engine.set_phase("LISTENING")
            fallback = "I apologize, I had trouble processing that."
            yield (fallback, True)

    async def is_ready(self) -> bool:
        return await self._ollama.is_available()

    async def get_available_models(self) -> list[str]:
        return await self._ollama.list_models()

    def set_model(self, model: str) -> None:
        self._ollama.set_model(model)

    @property
    def ollama(self) -> OllamaClient:
        return self._ollama

    _LOW_VALUE_PHRASES = (
        "I don't have data on that yet",
        "I don't have that information",
        "I'm sorry, I wasn't able to form a response",
        "I apologize, I had trouble processing that",
        "I apologize, but I am having difficulty processing",
    )

    @classmethod
    def _is_low_value_response(cls, response: str) -> bool:
        """Returns True if the response is a generic non-answer that shouldn't be stored."""
        stripped = response.strip()
        if len(stripped) < 15:
            return True
        for phrase in cls._LOW_VALUE_PHRASES:
            if stripped.startswith(phrase):
                return True
        return False

    @staticmethod
    def _is_garbage_response(response: str) -> bool:
        """Detect nonsensical LLM output (wrong language, too short, gibberish).

        Returns True if the response should be suppressed rather than spoken.
        """
        stripped = response.strip().rstrip(".")
        if not stripped or len(stripped) < 2:
            return True
        ascii_chars = sum(1 for c in stripped if c.isascii())
        if ascii_chars / max(len(stripped), 1) < 0.5:
            return True
        return False

    @staticmethod
    def _extract_tags(user_message: str, response: str) -> list[str]:
        tags = ["conversation"]
        combined = f"{user_message} {response}".lower()
        patterns: list[tuple[str, str]] = [
            (r"\b(code|programming|function|bug|error)\b", "technical"),
            (r"\b(feel|emotion|sad|happy|stress|anxious)\b", "emotion"),
            (r"\b(remind|schedule|calendar|meeting|deadline)\b", "scheduling"),
            (r"\b(prefer|like|dislike|always|never|usually)\b", "preference"),
            (r"\b(help|please|thanks|thank you)\b", "assistance"),
            (r"\b(joke|funny|laugh|humor)\b", "humor"),
            (r"\b(private|secret|sensitive|personal)\b", "sensitive"),
            (r"\b(quick|fast|brief|hurry)\b", "quick"),
            (r"\b(explain|detail|how|why|what)\b", "detail"),
        ]
        for pattern, tag in patterns:
            if re.search(pattern, combined):
                tags.append(tag)
        return tags
