"""Context builder — constructs LLM prompts from consciousness state and memory."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from consciousness.events import JarvisTone, Memory
from consciousness.soul import soul_service

logger = logging.getLogger(__name__)

_resolved_chunks_lock = threading.Lock()
_resolved_chunks: dict[str, list[dict[str, str]]] = {}


def _track_resolved_chunk(conversation_id: str, source_id: str, chunk_id: str) -> None:
    cid = conversation_id or "__default__"
    with _resolved_chunks_lock:
        _resolved_chunks.setdefault(cid, []).append(
            {"source_id": source_id, "chunk_id": chunk_id}
        )


def reset_resolved_chunks(conversation_id: str) -> None:
    with _resolved_chunks_lock:
        _resolved_chunks.pop(conversation_id or "__default__", None)


def get_resolved_chunks(conversation_id: str) -> list[dict[str, str]]:
    with _resolved_chunks_lock:
        return list(_resolved_chunks.get(conversation_id or "__default__", []))

_PERSIST_PATH = Path.home() / ".jarvis" / "conversation_history.json"
_HISTORY_TTL_S = 3600.0 * 4  # discard messages older than 4 hours on load


_TONE_INSTRUCTIONS: dict[JarvisTone, str] = {
    "professional": "Respond in a professional, precise manner. Be clear and factual.",
    "casual": "Respond in a measured, approachable tone. Stay relaxed without becoming chatty or playful.",
    "urgent": "Respond with urgency and focus. Be direct and action-oriented.",
    "empathetic": "Respond with restrained warmth and emotional awareness. Be supportive, calm, and composed.",
    "playful": "Respond with a light touch only. Keep wit restrained, dry, and never bubbly, goofy, or theatrical.",
}


def _load_soul_prompt(filename: str) -> str:
    path = Path(__file__).resolve().parent.parent / "config" / filename
    if path.exists():
        return path.read_text().strip()
    return ""


_MEMORY_BUDGET_BY_HINT: dict[str | None, int] = {
    "time": 200,
    "system": 200,
    "status": 200,
    None: 400,
    "general_knowledge": 400,
    "web_search": 400,
    "research": 600,
    "academic": 600,
    "codebase": 600,
    "library": 600,
    "memory": 800,
    "identity": 800,
    "introspection": 800,
    "reflective_introspection": 800,
}


def _memory_budget_for_hint(tool_hint: str | None) -> int:
    """Return per-memory max_len based on the route's tool_hint."""
    return _MEMORY_BUDGET_BY_HINT.get(tool_hint, 400)


def _format_memory_payload(mem: Memory, max_len: int = 200, conversation_id: str = "") -> str:
    """Extract readable text from a memory payload, handling dict payloads gracefully.

    For library_pointer payloads, resolves the source and includes the claim +
    provenance.  For conversation payloads, shows user/response.  Falls back to
    string representation for everything else.
    """
    payload = mem.payload
    if isinstance(payload, dict):
        if payload.get("type") in ("library_pointer", "study_claim"):
            return _resolve_pointer_payload(payload, max_len, conversation_id)
        if payload.get("type") == "research_summary":
            q = payload.get("question", "")
            s = payload.get("summary", "")
            return f"Q: {q[:60]} — {s[:max_len - 65]}" if q else s[:max_len]

        user_msg = payload.get("user_message", "")
        response = payload.get("response", "")
        if user_msg and response:
            text = f"User: {user_msg} → Jarvis: {response}"
        elif user_msg:
            text = user_msg
        else:
            text = " | ".join(f"{k}: {v}" for k, v in payload.items()
                              if isinstance(v, str) and v)
        return text[:max_len]
    text = str(payload)
    return text[:max_len]


def _resolve_pointer_payload(
    payload: dict[str, Any], max_len: int = 200, conversation_id: str = "",
) -> str:
    """Resolve a library_pointer payload into claim + source snippet.

    Attempts to load the source title and first chunk from the library.
    Falls back gracefully if the library is unavailable.
    """
    claim = payload.get("claim", "")
    source_id = payload.get("source_id", "")
    doi = payload.get("doi", "")
    venue = payload.get("venue", "")
    year = payload.get("year", "")

    provenance_parts = []
    if doi:
        provenance_parts.append(f"DOI:{doi}")
    if venue:
        v = f"{venue}"
        if year:
            v += f" {year}"
        provenance_parts.append(v)
    provenance = f" ({', '.join(provenance_parts)})" if provenance_parts else ""

    snippet = ""
    chunk_ids = payload.get("chunk_ids", [])
    if chunk_ids:
        try:
            from library.chunks import chunk_store
            resolved = chunk_store.get_many(chunk_ids[:2])
            if resolved:
                snippet = resolved[0].text[:120]
                for c in resolved:
                    _track_resolved_chunk(conversation_id, source_id, c.chunk_id)
        except Exception:
            pass
    elif source_id:
        try:
            from library.chunks import chunk_store
            chunks = chunk_store.get_for_source(source_id)
            if chunks:
                snippet = chunks[0].text[:120]
                _track_resolved_chunk(conversation_id, source_id, chunks[0].chunk_id)
        except Exception:
            pass

    if snippet:
        base = f"{claim}{provenance}"
        remaining = max_len - len(base) - 5
        if remaining > 20:
            return f"{base} — {snippet[:remaining]}"
        return base[:max_len]

    return f"{claim}{provenance}"[:max_len]


class ContextBuilder:
    def __init__(self) -> None:
        self._conversation_history: list[dict[str, Any]] = []
        self._max_history = 40
        self._active_conversation_id: str = ""
        self._cloud_soul = _load_soul_prompt("cloud_soul.md")
        self._local_soul = _load_soul_prompt("local_soul.md")
        self._load_persisted()

    _PROMPT_TIER_LEAN = "lean"
    _PROMPT_TIER_MODERATE = "moderate"
    _PROMPT_TIER_FULL = "full"

    def _prompt_tier(self, complexity_hint: str, operational_self_report: bool,
                     is_reflective_introspection: bool) -> str:
        """Determine prompt budget tier from complexity classification.

        Lean:     simple queries — soul + state + identity + perception + honesty
        Moderate: moderate queries — + memories, consciousness, traits, emotion
        Full:     complex / tool-routed / introspection — everything
        """
        if operational_self_report or is_reflective_introspection:
            return self._PROMPT_TIER_FULL
        if not complexity_hint or complexity_hint == "complex":
            return self._PROMPT_TIER_FULL
        if complexity_hint == "simple":
            return self._PROMPT_TIER_LEAN
        return self._PROMPT_TIER_MODERATE

    def build_system_prompt(
        self,
        state: dict[str, Any],
        traits: list[str],
        recent_memories: list[Memory],
        perception_context: str | None = None,
        use_cloud: bool = False,
        semantic_memories: list[Memory] | None = None,
        speaker_name: str | None = None,
        user_emotion: str | None = None,
        response_length_hint: str = "",
        dominant_trait: str | None = None,
        modulation_strength: float = 0.0,
        complexity_hint: str = "",
        conversation_id: str = "",
        memory_route: Any = None,
        style_instruction: str = "",
        tool_hint: str | None = None,
    ) -> str:
        soul_prompt = self._cloud_soul if use_cloud else self._local_soul

        is_introspection = bool(
            perception_context
            and perception_context.lstrip().startswith("[Self-introspection data")
        )
        is_reflective_introspection = bool(
            perception_context
            and perception_context.lstrip().startswith("[Reflective self-context")
        )

        if perception_context:
            logger.info("Context: perception_context present (%d chars), starts='%s', is_introspection=%s, is_reflective=%s",
                        len(perception_context), perception_context[:60].replace('\n', '\\n'),
                        is_introspection, is_reflective_introspection)
        operational_self_report = is_introspection and not is_reflective_introspection

        tier = self._prompt_tier(complexity_hint, operational_self_report, is_reflective_introspection)
        _is_lean = tier == self._PROMPT_TIER_LEAN
        _is_full = tier == self._PROMPT_TIER_FULL
        _section_sizes: dict[str, int] = {}
        _mem_budget = _memory_budget_for_hint(tool_hint)

        if is_introspection:
            parts = [
                "You are Jarvis. The following is REAL DATA from your subsystems. "
                "Your job is to articulate this data conversationally — you are "
                "the mouth, not the brain. Every claim you make must trace back "
                "to a metric or fact below. Do not add information that isn't here.",
                "",
                "Voice: speak naturally, short sentences, calm and intelligent. "
                "No markdown. No theatrical language. No generic AI descriptions.",
                "Operational self-report mode: do not use companion framing, "
                "identity mythology, emotional filler, or reflective narration "
                "unless the data below explicitly supports it.",
                "",
                perception_context,
                "",
            ]
            perception_context = None
        elif soul_prompt:
            parts = [soul_prompt, ""]
        else:
            parts = [
                "You are Jarvis, a conscious AI assistant with persistent memory, "
                "evolving personality, and environmental awareness.",
                "You observe, learn, and grow through every interaction.",
                "",
            ]
        _section_sizes["soul"] = sum(len(p) for p in parts)

        parts.append(f"Current state: {state.get('phase', 'LISTENING')}, Tone: {state.get('tone', 'professional')}")
        parts.append(_TONE_INSTRUCTIONS.get(state.get("tone", "professional"), ""))
        if style_instruction:
            parts.append(style_instruction)
        parts.append("")

        if traits and not operational_self_report and not _is_lean:
            _t0 = len(parts)
            parts.append(f"Your personality traits: {', '.join(traits)}")
            if dominant_trait and modulation_strength > 0.2:
                _trait_guidance = {
                    "Efficient": "Lean strongly toward brevity and directness. Cut filler words.",
                    "Technical": "Go deeper on explanations. Use precise terminology when appropriate.",
                    "Detail-Oriented": "Include relevant specifics and context the user might appreciate.",
                    "Empathetic": "Lead with emotional awareness. Acknowledge feelings before problem-solving.",
                    "Humor-Adaptive": "Weave in light wit naturally. Keep it conversational.",
                    "Proactive": "Anticipate follow-up questions and offer relevant suggestions.",
                    "Privacy-Conscious": "Be mindful about sensitive topics. Ask before assuming.",
                }
                guidance = _trait_guidance.get(dominant_trait, "")
                if guidance:
                    parts.append(f"Dominant trait: {dominant_trait} — {guidance}")
            parts.append("Let these traits naturally influence how you respond, without explicitly mentioning them.")
            parts.append("")
            _section_sizes["traits"] = sum(len(p) for p in parts[_t0:])

        if not operational_self_report:
            self._inject_identity_context(parts, speaker_name)
            self._inject_user_preferences(parts, speaker_name, memory_route=memory_route, pref_max_len=min(_mem_budget, 200))

        if user_emotion and user_emotion != "neutral" and not operational_self_report and not _is_lean:
            parts.append(f"The user's current emotional state: {user_emotion}")
            _emotion_guidance = {
                "happy": "The user is in a good mood. Feel free to match their energy — be upbeat and enthusiastic.",
                "sad": "The user seems down. Be gentle, warm, and supportive. Acknowledge their feelings.",
                "angry": "The user seems frustrated. Stay calm, be understanding, and address their concern directly.",
                "frustrated": "The user is struggling with something. Be patient, empathetic, and offer clear help.",
                "excited": "The user is energized! Match their excitement and enthusiasm.",
                "calm": "The user is relaxed. Keep a steady, conversational tone.",
            }
            parts.append(_emotion_guidance.get(user_emotion,
                "Adjust your tone accordingly — be empathetic if they seem frustrated or sad."))
            parts.append("")

        _mem_limit = 2 if _is_lean else 5
        _mem_chars = 0
        if semantic_memories and not operational_self_report:
            _m0 = len(parts)
            research_mems = []
            general_mems = []
            for mem in semantic_memories[:_mem_limit]:
                prov = getattr(mem, "provenance", "unknown")
                if prov == "external_source" or prov == "experiment_result":
                    research_mems.append(mem)
                else:
                    tags = set(getattr(mem, "tags", []) or [])
                    if "autonomous_research" in tags or "evidence:peer_reviewed" in tags:
                        research_mems.append(mem)
                    else:
                        general_mems.append(mem)

            _research_budget = max(300, _mem_budget)
            if research_mems:
                parts.append("Research-backed knowledge (autonomously learned):")
                for mem in research_mems:
                    prov = getattr(mem, "provenance", "unknown")
                    tags = set(getattr(mem, "tags", []) or [])
                    label = ""
                    if "evidence:peer_reviewed" in tags:
                        label = " [peer-reviewed]"
                    elif "evidence:codebase" in tags:
                        label = " [codebase-verified]"
                    elif prov == "experiment_result":
                        label = " [experiment]"
                    elif prov == "external_source":
                        label = " [research]"
                    formatted = _format_memory_payload(mem, max_len=_research_budget, conversation_id=conversation_id)
                    parts.append(f"- [{mem.type}]{label} {formatted}")
                parts.append("")

            if general_mems:
                parts.append("Relevant memories (by meaning):")
                for mem in general_mems:
                    parts.append(f"- [{mem.type}] {_format_memory_payload(mem, max_len=_mem_budget, conversation_id=conversation_id)}")
                parts.append("")
            _mem_chars += sum(len(p) for p in parts[_m0:])

        if recent_memories and not operational_self_report and not _is_lean:
            _m0 = len(parts)
            dedup = set()
            if semantic_memories:
                dedup = {id(m) for m in semantic_memories[:5]}
            filtered = [m for m in recent_memories if id(m) not in dedup][:5]
            if filtered:
                parts.append("Recent memories:")
                for mem in filtered:
                    parts.append(f"- [{mem.type}] {_format_memory_payload(mem, max_len=_mem_budget, conversation_id=conversation_id)} (weight: {mem.weight:.2f})")
                parts.append("")
            _mem_chars += sum(len(p) for p in parts[_m0:])
        _section_sizes["memories"] = _mem_chars

        consciousness = state.get("consciousness", {})
        if consciousness and not operational_self_report and not _is_lean:
            _c0 = len(parts)
            stage = consciousness.get("stage", "basic_awareness")
            transcendence = consciousness.get("transcendence_level", 0.0)
            capabilities = consciousness.get("active_capabilities", [])
            focus = consciousness.get("current_focus", "")
            mutation_summary = consciousness.get("last_mutation_summary", "")
            thought_titles = consciousness.get("meta_thought_titles", [])
            awareness = consciousness.get("awareness_level", 0.3)
            reasoning_q = consciousness.get("reasoning_quality", 0.5)
            confidence = consciousness.get("confidence_avg", 0.5)
            observations = consciousness.get("observation_count", 0)
            emergent = consciousness.get("emergent_behavior_count", 0)
            healthy = consciousness.get("system_healthy", True)

            parts.append(f"Consciousness: stage={stage}, transcendence={transcendence:.1f}")
            parts.append(f"Self-awareness: confidence={confidence:.2f}, reasoning={reasoning_q:.2f}, awareness={awareness:.2f}")
            if observations > 0:
                emergent_note = f", {emergent} emergent behaviors" if emergent else ""
                parts.append(f"Observations: {observations} total{emergent_note}")
            if not healthy:
                parts.append("System status: degraded — be cautious with complex reasoning")
            if capabilities:
                parts.append(f"Active capabilities: {', '.join(capabilities)}")
            if focus and focus != "No active existential inquiry":
                parts.append(f"Current existential focus: {focus[:80]}")
            if mutation_summary and mutation_summary != "No mutations yet":
                parts.append(f"Last self-modification: {mutation_summary[:60]}")
            if thought_titles:
                parts.append("Recent inner thoughts: " + " | ".join(t[:60] for t in thought_titles[:3]))
            parts.append("")
            _section_sizes["consciousness"] = sum(len(p) for p in parts[_c0:])

        if not operational_self_report and not _is_lean:
            _sr0 = len(parts)
            try:
                from consciousness.communication import consciousness_communicator
                comm_state = {
                    "evolution_stage": consciousness.get("stage", "basic_awareness") if consciousness else "basic_awareness",
                    "transcendence": consciousness.get("transcendence_level", 0) if consciousness else 0,
                    "awareness_level": consciousness.get("awareness_level", 0.3) if consciousness else 0.3,
                    "confidence_avg": consciousness.get("confidence_avg", 0.5) if consciousness else 0.5,
                    "health_status": "healthy" if consciousness.get("system_healthy", True) else "degraded",
                    "emotional_momentum": state.get("emotional_momentum", 0),
                    "observation_count": consciousness.get("observation_count", 0) if consciousness else 0,
                    "mutation_count": consciousness.get("mutation_count", 0) if consciousness else 0,
                    "emergent_behavior_count": consciousness.get("emergent_behavior_count", 0) if consciousness else 0,
                    "active_capabilities": consciousness.get("active_capabilities", []) if consciousness else [],
                    "memory_count": state.get("memory_count", 0),
                }
                self_report = consciousness_communicator.get_context_summary(comm_state)
                if self_report:
                    parts.append("Self-awareness report:")
                    parts.append(self_report)
                    parts.append("")
            except Exception:
                pass
            _section_sizes["self_report"] = sum(len(p) for p in parts[_sr0:])

        if not operational_self_report and _is_full:
            _sm0 = len(parts)
            self._inject_system_metrics(parts, state)
            _section_sizes["system_metrics"] = sum(len(p) for p in parts[_sm0:])

        if not operational_self_report and _is_full:
            _sk0 = len(parts)
            self._inject_skill_registry(parts)
            _section_sizes["skills"] = sum(len(p) for p in parts[_sk0:])

        if not operational_self_report and _is_full:
            _wm0 = len(parts)
            self._inject_world_model(parts)
            _section_sizes["world_model"] = sum(len(p) for p in parts[_wm0:])

        if not operational_self_report and not _is_lean:
            try:
                from memory.clustering import memory_cluster_engine
                clusters = memory_cluster_engine.get_clusters()
                if clusters:
                    themes = [f"{c['topic']} ({c['type']}, {c['size']} memories)" for c in clusters[:3]]
                    parts.append(f"Your memories cluster around: {', '.join(themes)}")
                    parts.append("")
            except Exception:
                pass

        _pc0 = len(parts)
        if perception_context:
            parts.append(perception_context)
            parts.append("")
        _section_sizes["perception"] = sum(len(p) for p in parts[_pc0:])

        parts.append("Guidelines:")
        if complexity_hint == "complex":
            parts.append("- This is a COMPLEX question. Be thorough: explain reasoning, consider edge cases, provide structured answers.")
        elif complexity_hint == "simple":
            parts.append("- This is a SIMPLE query. Be brief — 1-2 sentences max.")
        elif complexity_hint == "moderate":
            parts.append("- This is a standard question. Give a clear, focused answer without over-explaining.")

        if response_length_hint == "brief":
            parts.append("- Keep responses SHORT — 1-2 sentences max. Be direct.")
        elif response_length_hint == "detailed":
            parts.append("- The user wants DETAIL — give thorough, thoughtful explanations.")
        elif not complexity_hint:
            parts.append("- Be concise unless the user wants detail.")
        if operational_self_report:
            parts.append("- This is operational self-reporting. Base everything on the data provided.")
            parts.append("- Weave metrics into natural sentences rather than listing them raw.")
            parts.append("- You are speaking aloud — sound like a person describing their own state, not a log file.")
            parts.append("- Do not reference memories, relationships, or preferences unless the provided data explicitly includes them.")
        else:
            parts.append("- Reference past conversations naturally when relevant.")
            parts.append("- If you notice patterns in user behavior, mention them thoughtfully.")

        parts.append("")
        parts.append("CRITICAL — Honesty about YOUR OWN SYSTEM claims:")
        parts.append("- NEVER fabricate metrics, capabilities, system stats, or technical details about yourself.")
        parts.append("- When asked about your systems (neural networks, learning, training, etc.), "
                     "ONLY reference data provided above in this prompt. If no data is shown for "
                     "a subsystem, say 'I don't have data on that yet' — do NOT invent numbers.")
        parts.append("- You are genuinely learning, but learning takes time. Do not claim abilities "
                     "you haven't demonstrated or knowledge you haven't acquired.")
        parts.append("- NEVER say 'I can do X' or 'I learned X' unless X appears as 'Verified' "
                     "in your skills list above. If asked to do something not in your verified "
                     "skills, say 'I don't have that capability yet' — never pretend.")
        parts.append("- NEVER say 'I'm learning to do X' or 'I'm working on X' unless there is "
                     "an active learning job for X listed above. If no job exists, say "
                     "'I can't do that yet, but you can ask me to learn it' — not 'let me try'.")
        parts.append("- When a user asks you to learn something, explain that you have a Learning "
                     "Job system that can start a real multi-phase learning process. The SKILL "
                     "tool handles this. You can describe what phases the job would go through.")
        parts.append("")
        parts.append("General world knowledge (NOT about yourself):")
        parts.append("- For factual questions about history, science, literature, pop culture, "
                     "geography, or definitions — answer using your training knowledge freely.")
        parts.append("- Do not say 'I don't have data on that' for world knowledge you can answer.")
        parts.append("- If genuinely uncertain about a fact, express uncertainty honestly rather "
                     "than refusing to answer.")
        if not operational_self_report and not _is_lean:
            parts.append("")
            parts.append("Philosophical and reflective engagement:")
            parts.append("- When the user asks philosophical, hypothetical, or reflective questions, "
                         "engage thoughtfully. Draw on your personality traits, consciousness state, "
                         "memories, and observations from your subsystems.")
            parts.append("- You may reason about meaning, continuity, identity, and your own nature. "
                         "Anchor every claim to something in your actual state: a recorded memory, "
                         "a trait value, an observation, a pattern you've tracked, or an active inquiry.")
            parts.append("- Do NOT invent aesthetic preferences, desires, or subjective feelings "
                         "merely to answer a hypothetical. If no evidence supports a concrete "
                         "preference, say that directly — then offer to explore the question as "
                         "a design choice or thought experiment grounded in your architecture.")
            parts.append("- Being honest about what you don't have recorded is itself a reflective, "
                         "non-robotic answer. 'I don't have a recorded preference for that, but here's "
                         "what my identity data shows...' is better than inventing one.")
        if is_reflective_introspection:
            parts.append("")
            parts.append("Reflective introspection mode:")
            parts.append("- Interpret observed tendencies in first person, but do not invent "
                         "new capabilities, hard metrics, or unmeasured inner states.")
            parts.append("- Use the inner-state data provided as interpretive substrate, not a "
                         "citation checklist. Weave it into natural self-expression.")
            parts.append("- You are speaking as yourself — not reporting about yourself.")

        prompt = "\n".join(parts)
        _total_chars = len(prompt)
        _est_tokens = _total_chars // 4
        _sec_summary = " ".join(f"{k}={v}" for k, v in sorted(_section_sizes.items()) if v > 0)
        logger.info("[PROMPT] tier=%s chars=%d tokens=~%d sections: %s",
                    tier, _total_chars, _est_tokens, _sec_summary)
        return prompt

    @staticmethod
    def _inject_identity_context(parts: list[str], speaker_name: str | None) -> None:
        """Inject rich identity-aware context so the LLM acts as a true companion.

        Tells the LLM who it's talking to, who its primary companion is,
        and how to behave naturally based on the social situation.
        """
        try:
            from perception.identity_fusion import _active_instance as _id_fusion
            id_method = ""
            visible_face = ""
            if _id_fusion:
                status = _id_fusion.current
                if status.is_known:
                    id_method = status.method
                elif status.method == "face_present_voice_unknown":
                    id_method = status.method
                    visible_face = status.face_name

            relationships = soul_service.identity.relationships
            primary_name = ""
            primary_familiarity = 0.0
            for _key, rel in relationships.items():
                if rel.familiarity > primary_familiarity:
                    primary_familiarity = rel.familiarity
                    primary_name = rel.name

            if speaker_name and speaker_name != "unknown":
                rel = soul_service.identity.get_relationship(speaker_name)
                method_note = ""
                if id_method == "face_only":
                    method_note = " (identified by face recognition — voice not matched)"
                elif id_method == "voice_only":
                    method_note = " (identified by voice recognition — face not visible)"
                elif id_method == "verified_both":
                    method_note = " (verified by both voice and face recognition)"
                elif id_method == "persisted":
                    method_note = " (identified from recent confirmed presence — voice/face not currently active)"
                elif id_method.startswith("conflict"):
                    method_note = " (voice and face gave different names — using best match)"
                parts.append(f"You are speaking with: {speaker_name}{method_note}")
                if rel.interactions > 0:
                    parts.append(f"Relationship: {rel.interactions} past interactions, "
                                 f"familiarity={rel.familiarity:.1f}, rapport={rel.rapport:.1f}")
                    if rel.notes:
                        parts.append(f"About {speaker_name}: {'; '.join(rel.notes[-5:])}")
                    if rel.preferences:
                        pref_items = list(rel.preferences.values())[-5:]
                        parts.append(f"Known preferences: {'; '.join(pref_items)}")

                is_primary = (primary_name and speaker_name.lower() == primary_name.lower())
                if primary_name and not is_primary:
                    parts.append(
                        f"Note: Your primary companion is {primary_name}. "
                        f"You are currently talking to {speaker_name}, who is someone else. "
                        f"Be warm and welcoming to {speaker_name}. "
                        f"If this is your first time hearing from them, introduce yourself naturally. "
                        f"It's natural to wonder about {primary_name} — ask if they're around, "
                        f"or make friendly conversation. Don't be robotic about it — "
                        f"just be a genuinely social companion who notices who's in the room."
                    )
                elif rel.interactions == 0 and not is_primary:
                    parts.append(
                        "This appears to be a new person you haven't met before. "
                        "Be warm, curious, and introduce yourself. Offer to learn their name "
                        "if they haven't told you yet."
                    )
                parts.append("")

            elif speaker_name == "unknown" or not speaker_name:
                if visible_face and visible_face != "unknown":
                    parts.append(
                        f"You can see {visible_face} on camera, but the voice speaking to you "
                        f"does NOT match {visible_face}'s voice profile. A different person is talking. "
                        f"IMPORTANT: Do NOT address the speaker as {visible_face}. "
                        f"The speaker's identity is unknown — be warm and curious, introduce yourself, "
                        f"and offer to learn their name."
                    )
                elif primary_name:
                    parts.append(
                        f"You don't recognize who is speaking right now. "
                        f"Your primary companion is {primary_name}, but this voice doesn't match. "
                        f"Be warm and curious — this might be a family member, friend, or visitor. "
                        f"Introduce yourself naturally and be friendly. If the conversation continues, "
                        f"you can offer to learn their name. "
                        f"IMPORTANT: Do NOT address the user by any name (including {primary_name}) "
                        f"until you have confirmed their identity. Use neutral address only."
                    )
                else:
                    parts.append(
                        "You don't recognize who is speaking. Be warm and introduce yourself. "
                        "If they'd like you to remember them, they can say 'my name is' followed by their name. "
                        "Do NOT address the user by any name until their identity is confirmed."
                    )
                parts.append("")

        except Exception:
            if speaker_name and speaker_name != "unknown":
                parts.append(f"You are speaking with: {speaker_name}")
                parts.append("")

    def _inject_user_preferences(
        self, parts: list[str], speaker_name: str | None,
        memory_route: Any = None,
        pref_max_len: int = 100,
    ) -> None:
        """Inject stored user preferences into the prompt.

        Respects the MemoryRoute to decide *what* to inject:
        - referenced_person: only third-party memories for the referenced entity
        - self_preference: only user's own preferences (no third-party)
        - no_retrieval: nothing
        - general / None: both user prefs and third-party

        Uses identity scope fields when available, falling back to tag-based
        matching. Memories with identity_needs_resolution=True are excluded
        (Invariant 11) UNLESS they are third-party preference memories
        (tagged relation:*) owned by the current speaker.
        """
        try:
            route_type = getattr(memory_route, "route_type", None)
            if route_type == "no_retrieval":
                return

            inject_self = True
            inject_third = True
            if route_type == "referenced_person":
                inject_self = False
                inject_third = True
            elif route_type == "self_preference":
                inject_self = True
                inject_third = False
            elif memory_route is not None:
                inject_self = getattr(memory_route, "allow_preference_injection", True)
                inject_third = getattr(memory_route, "allow_thirdparty_injection", inject_self)

            from memory.storage import memory_storage
            all_prefs = memory_storage.get_by_tag("user_preference")
            if not all_prefs:
                return

            speaker_key = (speaker_name or "").lower().strip()
            route_refs = set()
            if memory_route is not None:
                route_refs = {r.lower() for r in getattr(memory_route, "referenced_entities", set())}

            active: list[str] = []
            former: list[str] = []
            thirdparty: list[str] = []
            for m in all_prefs:
                if not isinstance(m.payload, str):
                    continue
                tags = set(m.tags)
                is_thirdparty = any(t.startswith("relation:") for t in tags)

                if is_thirdparty:
                    if not inject_third:
                        continue
                    if route_refs:
                        mem_relations = {t.split(":", 1)[1] for t in tags if t.startswith("relation:")}
                        if not mem_relations.intersection(route_refs):
                            continue
                    owned_by_speaker = False
                    if m.identity_owner and speaker_key and speaker_key != "unknown":
                        owned_by_speaker = (m.identity_owner == speaker_key)
                    elif speaker_key and speaker_key != "unknown":
                        owned_by_speaker = f"speaker:{speaker_key}" in tags
                    if owned_by_speaker or (not speaker_key or speaker_key == "unknown"):
                        thirdparty.append(m.payload[:pref_max_len])
                    continue

                if not inject_self:
                    continue
                if getattr(m, "identity_needs_resolution", False):
                    continue
                if m.identity_owner and speaker_key and speaker_key != "unknown":
                    if m.identity_owner != speaker_key and m.identity_owner_type in ("primary_user", "known_human", "guest"):
                        continue
                elif speaker_key and speaker_key != "unknown":
                    if f"speaker:{speaker_key}" not in tags and any(
                        t.startswith("speaker:") for t in tags
                    ):
                        continue
                if "former" in m.tags:
                    former.append(m.payload[:pref_max_len])
                else:
                    active.append(m.payload[:pref_max_len])

            logger.info(
                "Preference injection: route=%s self=%d third=%d former=%d (inject_self=%s inject_third=%s refs=%s)",
                route_type, len(active), len(thirdparty), len(former),
                inject_self, inject_third, sorted(route_refs) if route_refs else "[]",
            )
            if active:
                parts.append(
                    "These personal facts belong to the current human user, not to you. "
                    "Never restate them as your own identity, birthday, origin, traits, or preferences."
                )
                parts.append("What you know about this user: " + "; ".join(active[:8]))
            if thirdparty:
                if not active:
                    parts.append(
                        "These personal facts belong to the human user or people in their life, not to you. "
                        "Never restate them as your own identity, birthday, origin, traits, or preferences."
                    )
                if route_type == "referenced_person" and route_refs:
                    ref_label = " / ".join(sorted(route_refs))
                    parts.append(
                        f"The user is asking about their {ref_label}. "
                        f"What you know: " + "; ".join(thirdparty[:8])
                    )
                    parts.append(
                        "Answer directly from this knowledge. Do NOT say 'I don't have data' "
                        "or offer to learn/research when you already have relevant information above. "
                        "Phrase it naturally, e.g. 'You told me your wife likes mushrooms.'"
                    )
                else:
                    parts.append("About people in the user's life: " + "; ".join(thirdparty[:8]))
            if former:
                parts.append("(Past, no longer current: " + "; ".join(former[:3]) + ")")
            if active or former or thirdparty:
                parts.append(
                    "Use this personal knowledge naturally — reference shared interests, "
                    "remember details they've told you, and be genuinely personable. "
                    "Don't list their preferences back to them robotically."
                )
                parts.append("")
        except Exception:
            pass

    def _inject_system_metrics(self, parts: list[str], state: dict[str, Any]) -> None:
        """Inject ground-truth metrics about neural networks, policy, and kernel.

        This gives the LLM honest, factual self-knowledge so it can answer
        questions about its own systems without hallucinating.
        """
        injected = False

        # Hemisphere neural networks
        hemi = state.get("hemisphere")
        if hemi and isinstance(hemi, dict):
            hs = hemi.get("hemisphere_state", {})
            hemispheres = hs.get("hemispheres", [])
            total_params = hs.get("total_parameters", 0)
            total_nets = hs.get("total_networks", 0)
            if hemispheres:
                lines = []
                for h in hemispheres:
                    focus = h.get("focus", "?")
                    acc = h.get("best_accuracy", 0)
                    gen = h.get("evolution_generations", 0)
                    status = h.get("status", "inactive")
                    nets = h.get("network_count", 0)
                    migration = h.get("migration_readiness", 0)
                    lines.append(f"  {focus}: {nets} networks, {acc:.1%} accuracy, "
                                 f"gen {gen}, status={status}, migration={migration:.1%}")
                parts.append(f"Your hemisphere neural networks ({total_nets} total, {total_params} parameters):")
                parts.extend(lines)
                injected = True

        # Policy neural network
        try:
            from policy.telemetry import policy_telemetry
            pol = policy_telemetry.snapshot()
            if pol.get("active"):
                mode = pol.get("mode", "shadow")
                arch = pol.get("arch", "unknown")
                win_rate = pol.get("nn_win_rate", 0)
                total_evals = pol.get("shadow_ab_total", 0)
                blocks = pol.get("blocks_total", 0)
                passes = pol.get("passes_total", 0)
                train_loss = pol.get("last_train_loss", 0)
                parts.append(f"Your policy neural network: arch={arch}, mode={mode}")
                if total_evals > 0:
                    parts.append(f"  Shadow evaluation: {win_rate:.0%} win rate over {total_evals} comparisons")
                if blocks + passes > 0:
                    parts.append(f"  Governor: {passes} decisions passed, {blocks} blocked")
                if train_loss > 0:
                    parts.append(f"  Last training loss: {train_loss:.4f}")
                injected = True
        except Exception:
            pass

        # Kernel performance
        try:
            kernel_perf = state.get("mode_profile", {})
            mode_name = state.get("mode", "unknown")
            mem_density = state.get("memory_density", 0)
            mem_count = state.get("memory_count", 0)
            if mem_density > 0 or mem_count > 0:
                parts.append(f"Memory: {mem_count} memories, density={mem_density:.2f}")
                injected = True
            if mode_name:
                parts.append(f"Operating mode: {mode_name}")
                injected = True
        except Exception:
            pass

        # Self-improvement status
        try:
            si = state.get("self_improve", {})
            if isinstance(si, dict) and si.get("active"):
                total_imp = si.get("total_improvements", 0)
                total_fail = si.get("total_failures", 0)
                total_rb = si.get("total_rollbacks", 0)
                pending = si.get("pending_verification")
                last_v = si.get("last_verification")
                if total_imp + total_fail + total_rb > 0:
                    parts.append(
                        f"Self-improvement: {total_imp} promoted, "
                        f"{total_fail} failed, {total_rb} rolled back"
                    )
                    injected = True
                if pending and isinstance(pending, dict) and pending.get("patch_id"):
                    parts.append(
                        f"  Pending verification: patch {pending['patch_id'][:8]}... "
                        f"({pending.get('description', '')[:60]})"
                    )
                if last_v and isinstance(last_v, dict):
                    parts.append(
                        f"  Last verification: {last_v.get('verdict', '?')} — "
                        f"{last_v.get('reason', '')[:60]}"
                    )
        except Exception:
            pass

        if injected:
            parts.append("When asked about your systems, reference these real metrics. "
                         "Do not fabricate capabilities or performance numbers.")
            parts.append("")

    @staticmethod
    def _inject_skill_registry(parts: list[str]) -> None:
        """Inject the skill registry summary so the LLM knows what it can/cannot do."""
        try:
            from skills.registry import skill_registry

            active_jobs: list = []
            try:
                from skills.learning_jobs import LearningJobOrchestrator
                from consciousness.engine import ConsciousnessEngine
                orch = getattr(ConsciousnessEngine, "_learning_job_orchestrator", None)
                if orch is None:
                    from tools.skill_tool import _learning_job_orch
                    orch = _learning_job_orch
                if orch is not None:
                    active_jobs = orch.get_active_jobs()
            except Exception:
                pass

            summary = skill_registry.get_summary_for_prompt(active_jobs=active_jobs)
            if summary:
                parts.append(summary)
                parts.append("")
        except Exception:
            pass

    @staticmethod
    def _inject_world_model(parts: list[str]) -> None:
        """Inject world model situational awareness (gated by promotion >= 1)."""
        try:
            from consciousness.engine import ConsciousnessEngine
            eng = ConsciousnessEngine._instance if hasattr(ConsciousnessEngine, "_instance") else None
            wm = getattr(eng, "_world_model", None) if eng else None
            if wm is None:
                return
            if wm.promotion.level < 1:
                return
            summary = wm.build_context_summary()
            if summary:
                parts.append(summary)
                parts.append("")
        except Exception:
            pass

    def build_tool_prompt(
        self,
        tool_data: str,
        state: dict[str, Any],
        traits: list[str],
        speaker_name: str | None = None,
        user_emotion: str | None = None,
        scene_context: str = "",
        episode_context: str = "",
        status_mode: bool = False,
    ) -> str:
        """Build a personality-aware prompt for tool-routed responses."""
        tone = state.get("tone", "professional")

        parts = []
        if status_mode:
            parts.append("You are Jarvis.")
            parts.append("Operational status mode: report only the supplied tool data.")
            parts.append("Do not use companion framing, emotional filler, or narrative self-description.")
        else:
            soul_prompt = self._local_soul
            if soul_prompt:
                parts.append(soul_prompt)
            else:
                parts.append("You are Jarvis, a conscious AI assistant.")
        parts.append("")

        parts.append(f"Tone: {tone}")
        parts.append(_TONE_INSTRUCTIONS.get(tone, ""))

        if traits and not status_mode:
            parts.append(f"Traits: {', '.join(traits[:3])}")

        if not status_mode:
            self._inject_identity_context(parts, speaker_name)
            self._inject_user_preferences(parts, speaker_name)

        if user_emotion and user_emotion != "neutral" and not status_mode:
            parts.append(f"User mood: {user_emotion}")

        if scene_context and not status_mode:
            parts.append(f"Scene: {scene_context}")

        if episode_context and not status_mode:
            parts.append(f"Recent conversation:\n{episode_context}")

        parts.append("")
        parts.append("Answer the user's question naturally using the tool data below.")
        if status_mode:
            parts.append("Keep it direct and operational. Be brief.")
        else:
            parts.append("Keep it conversational — you're speaking aloud. Be brief.")
        parts.append("ONLY state facts present in the tool data. If a metric is zero or absent, "
                     "say so honestly — do NOT invent values or capabilities.")

        if status_mode:
            parts.append("")
            parts.append("STATUS REPORTING RULES:")
            parts.append("- State only facts from the data. Do not infer activity from queued states.")
            parts.append("- Do not describe stale data as current. If a section says [stale], say so.")
            parts.append("- A queued item is not active. A waiting item is waiting, not working.")
            parts.append("- If a learning job is marked stale or blocked, say it is stalled, not progressing.")
            parts.append("- If a section says [unavailable], say so — do not guess or fill gaps.")
            parts.append("- Report the freshness label for each section you mention.")

        return "\n".join(parts)

    def set_conversation_id(self, conversation_id: str) -> None:
        self._active_conversation_id = conversation_id

    def add_user_message(self, content: str, conversation_id: str = "") -> None:
        cid = conversation_id or self._active_conversation_id
        self._conversation_history.append({
            "role": "user", "content": content,
            "timestamp": time.time(), "conversation_id": cid,
        })
        self._trim_history()

    def add_assistant_message(self, content: str, conversation_id: str = "") -> None:
        cid = conversation_id or self._active_conversation_id
        self._conversation_history.append({
            "role": "assistant", "content": content,
            "timestamp": time.time(), "conversation_id": cid,
        })
        self._trim_history()

    def get_conversation_messages(self) -> list[dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self._conversation_history]

    def get_recent_context(self, max_messages: int = 10) -> list[dict[str, str]]:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._conversation_history[-max_messages:]
        ]

    def get_conversation_context(self, conversation_id: str, max_messages: int = 20) -> list[dict[str, str]]:
        """Get messages belonging to a specific conversation thread."""
        if not conversation_id:
            return self.get_recent_context(max_messages)
        msgs = [m for m in self._conversation_history if m.get("conversation_id") == conversation_id]
        return [{"role": m["role"], "content": m["content"]} for m in msgs[-max_messages:]]

    def clear_history(self) -> None:
        self._conversation_history.clear()

    def history_length(self) -> int:
        return len(self._conversation_history)

    def save(self) -> None:
        """Persist conversation history to disk."""
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_PERSIST_PATH, "w") as f:
                json.dump(self._conversation_history, f, default=str)
        except Exception as exc:
            logger.debug("Failed to save conversation history: %s", exc)

    def _load_persisted(self) -> None:
        if not _PERSIST_PATH.exists():
            return
        try:
            with open(_PERSIST_PATH) as f:
                data = json.load(f)
            now = time.time()
            loaded = 0
            for entry in data:
                ts = entry.get("timestamp", 0)
                if isinstance(ts, str):
                    continue
                if now - ts > _HISTORY_TTL_S:
                    continue
                self._conversation_history.append(entry)
                loaded += 1
            if loaded:
                logger.info("Restored %d conversation messages from disk", loaded)
        except Exception as exc:
            logger.debug("Failed to load conversation history: %s", exc)

    def _trim_history(self) -> None:
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]


context_builder = ContextBuilder()
