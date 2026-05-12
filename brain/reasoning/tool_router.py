"""Tool router — dispatches user queries to local tools or LLM.

Five-tier strategy with disambiguation:
  Tier 0: core intent verbs — generic command verbs (learn, train, teach)
          that always route to the right subsystem regardless of the noun.
          This is the minimal bootstrap; the voice_intent NN learns to
          generalize beyond these over time.
  Tier 0.25: response-preference instructions (format/style updates)
          route to NONE so the conversation layer can persist + acknowledge
          them deterministically via memory.
  Tier 0.5: persistent user routing corrections (loaded from disk)
  Tier 1: exact keyword match (fast, high confidence)
  Tier 1.5: disambiguate STATUS vs SYSTEM_STATUS vs INTROSPECTION
  Tier 2: regex intent patterns (broader, medium confidence)
  Tier 3: self-referential catch-all with strong self-frame gate
  Tier 3.5: mild self-reference flag for NONE fallback context injection
  Fallback: ToolType.NONE (general LLM chat)

Routing contract: any user query about Jarvis's own state, capabilities,
learning, goals, systems, architecture, recent activity, or self-description
must never fall through to NONE.

Design principle: self-report should fail closed, not fail open.
If uncertain whether a query is about Jarvis itself, prefer tool-grounded
introspection over free-generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from reasoning.golden_words import (
    GoldenCommandContext,
    list_canonical_commands,
    parse_golden_command,
)

logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    TIME = "TIME"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    VISION = "VISION"
    CAMERA_CONTROL = "CAMERA_CONTROL"
    MEMORY = "MEMORY"
    INTROSPECTION = "INTROSPECTION"
    SELF_IMPROVE = "SELF_IMPROVE"
    WEB_SEARCH = "WEB_SEARCH"
    ACADEMIC_SEARCH = "ACADEMIC_SEARCH"
    CODEBASE = "CODEBASE"
    IDENTITY = "IDENTITY"
    SKILL = "SKILL"
    PERFORM = "PERFORM"
    STATUS = "STATUS"
    LIBRARY_INGEST = "LIBRARY_INGEST"
    PLUGIN = "PLUGIN"
    ACQUISITION = "ACQUISITION"
    NONE = "NONE"


@dataclass
class RoutingResult:
    tool: ToolType
    confidence: float
    extracted_args: dict[str, Any]
    golden_context: GoldenCommandContext | None = None


# ---------------------------------------------------------------------------
# Tier 0: Core intent verbs
# ---------------------------------------------------------------------------
# Minimal bootstrap. "learn X" → SKILL regardless of X (unless informational).
# The voice_intent NN learns to handle the long tail over time.

_JARVIS_PREFIX = r"(?:(?:hey |alright |okay |ok )?jarvis[,.]?\s+)?"

_CORE_INTENTS: list[tuple[re.Pattern, ToolType, set[str]]] = [
    (re.compile(r"^" + _JARVIS_PREFIX + r"learn\s+(?:my\s+(?:voice|face|name))\b", re.I),
     ToolType.IDENTITY, set()),
    (re.compile(r"^" + _JARVIS_PREFIX + r"learn\s+", re.I),
     ToolType.SKILL, {"about ", "more about ", "from "}),
    (re.compile(r"^" + _JARVIS_PREFIX + r"(?:train|retrain)\s+(?:your\s+)?", re.I),
     ToolType.SKILL, set()),
    (re.compile(r"^" + _JARVIS_PREFIX + r"teach yourself\s+", re.I),
     ToolType.SKILL, set()),
    (re.compile(r"^" + _JARVIS_PREFIX + r"improve your\s+", re.I),
     ToolType.SKILL, {"code", "system", "codebase", "source"}),
    (re.compile(r"^" + _JARVIS_PREFIX + r"start\s+(?:a\s+)?(?:learning|training)\b", re.I),
     ToolType.SKILL, set()),
]


def _match_core_intent(lower: str) -> RoutingResult | None:
    """Check Tier 0 core intent verbs. Returns a result or None."""
    for pattern, tool, exclusions in _CORE_INTENTS:
        m = pattern.match(lower)
        if m:
            rest = lower[m.end():]
            if any(rest.startswith(exc) for exc in exclusions):
                continue
            return RoutingResult(
                tool=tool, confidence=0.90,
                extracted_args={"tier": "core_intent", "verb": m.group(0).strip()},
            )
    return None


# ---------------------------------------------------------------------------
# Tier 0.5: Persistent routing corrections
# ---------------------------------------------------------------------------
# When the user says "when I say X, route that to Y", we store the correction
# in a JSONL file and load it at boot. These become additional keyword routes.

_CORRECTIONS_PATH = os.path.join(
    os.path.expanduser("~"), ".jarvis", "routing_corrections.jsonl",
)
_user_corrections: list[tuple[str, ToolType]] = []


def _load_routing_corrections() -> None:
    """Load persistent user routing corrections from disk."""
    global _user_corrections
    _user_corrections.clear()
    if not os.path.exists(_CORRECTIONS_PATH):
        return
    try:
        with open(_CORRECTIONS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                phrase = rec.get("phrase", "").lower()
                tool_name = rec.get("tool", "")
                try:
                    tool = ToolType(tool_name)
                except ValueError:
                    continue
                if phrase:
                    _user_corrections.append((phrase, tool))
        if _user_corrections:
            logger.info("Loaded %d routing corrections from disk", len(_user_corrections))
    except Exception:
        logger.debug("Failed to load routing corrections", exc_info=True)


def record_routing_correction(
    phrase: str,
    target_tool: ToolType,
    original_tool: ToolType | None = None,
) -> None:
    """Persist a user routing correction and record as training data.

    Called when the user says something like 'when I say X, route to skill'.
    """
    phrase_lower = phrase.lower().strip()
    if not phrase_lower:
        return

    _user_corrections.append((phrase_lower, target_tool))

    try:
        os.makedirs(os.path.dirname(_CORRECTIONS_PATH), exist_ok=True)
        with open(_CORRECTIONS_PATH, "a") as f:
            rec = {
                "phrase": phrase_lower,
                "tool": target_tool.value,
                "original_tool": original_tool.value if original_tool else None,
                "ts": __import__("time").time(),
            }
            f.write(json.dumps(rec) + "\n")
        logger.info("Routing correction saved: '%s' → %s", phrase_lower, target_tool.value)
    except Exception:
        logger.debug("Failed to save routing correction", exc_info=True)

    try:
        from memory.search import get_vector_store
        from hemisphere.distillation import distillation_collector
        vs = get_vector_store()
        if vs and vs.available:
            embedding = vs.embed(phrase_lower)
            if embedding and len(embedding) >= 384:
                label = _voice_intent_teacher_vector(target_tool)
                distillation_collector.record(
                    "text_embedding", "sentence_vector", embedding[:384],
                    metadata={"tool": target_tool.value, "source": "user_correction"},
                    origin="user_correction", fidelity=1.0,
                )
                distillation_collector.record(
                    "tool_router", "route_label", label,
                    metadata={"tool": target_tool.value, "source": "user_correction"},
                    origin="user_correction", fidelity=1.0,
                )
    except Exception:
        logger.debug("Failed to record correction as training data", exc_info=True)


def _match_user_corrections(lower: str) -> RoutingResult | None:
    """Check Tier 0.5 persistent user corrections."""
    for phrase, tool in _user_corrections:
        if phrase in lower:
            return RoutingResult(
                tool=tool, confidence=0.92,
                extracted_args={"tier": "user_correction", "phrase": phrase},
            )
    return None


_load_routing_corrections()


_VOICE_INTENT_BUCKETS: tuple[str, ...] = (
    "general_chat",
    "status_ops",
    "introspection",
    "memory",
    "vision_camera",
    "identity",
    "skill_action",
    "knowledge_lookup",
)

_VOICE_INTENT_TOOL_BUCKET: dict[ToolType, str] = {
    ToolType.NONE: "general_chat",
    ToolType.TIME: "status_ops",
    ToolType.SYSTEM_STATUS: "status_ops",
    ToolType.STATUS: "status_ops",
    ToolType.INTROSPECTION: "introspection",
    ToolType.MEMORY: "memory",
    ToolType.VISION: "vision_camera",
    ToolType.CAMERA_CONTROL: "vision_camera",
    ToolType.IDENTITY: "identity",
    ToolType.SKILL: "skill_action",
    ToolType.PERFORM: "skill_action",
    ToolType.SELF_IMPROVE: "skill_action",
    ToolType.WEB_SEARCH: "knowledge_lookup",
    ToolType.LIBRARY_INGEST: "knowledge_lookup",
    ToolType.ACADEMIC_SEARCH: "knowledge_lookup",
    ToolType.CODEBASE: "knowledge_lookup",
    ToolType.ACQUISITION: "skill_action",
}


def _voice_intent_teacher_vector(tool: ToolType) -> list[float]:
    """Return a stable 8-dim one-hot teacher label for the voice-intent student."""
    bucket = _VOICE_INTENT_TOOL_BUCKET.get(tool, "general_chat")
    return [1.0 if name == bucket else 0.0 for name in _VOICE_INTENT_BUCKETS]


def _record_voice_intent_teacher_signal(
    user_message: str,
    result: RoutingResult,
    *,
    synthetic: bool = False,
) -> None:
    """Persist tool-router intent labels AND text embeddings for voice_intent student.

    The text embedding gives the student semantic features it can generalize from,
    instead of trying to learn intent from audio spectral properties (impossible).
    """
    try:
        from hemisphere.distillation import distillation_collector

        origin = "synthetic" if synthetic else "router"
        fidelity = max(0.3, min(1.0, result.confidence))
        if synthetic:
            fidelity = min(fidelity, 0.7)
        bucket = _VOICE_INTENT_TOOL_BUCKET.get(result.tool, "general_chat")
        meta = {
            "tool": result.tool.value,
            "confidence": round(result.confidence, 4),
            "bucket": bucket,
            "query_preview": user_message[:120],
        }

        distillation_collector.record(
            "tool_router", "route_label",
            _voice_intent_teacher_vector(result.tool),
            metadata=meta, origin=origin, fidelity=fidelity,
        )

        try:
            from memory.search import get_vector_store
            vs = get_vector_store()
            if vs and vs.available:
                embedding = vs.embed(user_message)
                if embedding and len(embedding) >= 384:
                    distillation_collector.record(
                        "text_embedding", "sentence_vector",
                        embedding[:384],
                        metadata=meta, origin=origin, fidelity=fidelity,
                    )
        except Exception:
            pass
    except Exception:
        logger.debug("Voice-intent teacher signal capture skipped", exc_info=True)


_KEYWORD_PATTERNS: list[tuple[list[str], ToolType]] = [
    (["what time is it", "current time", "what's the time", "tell me the time", "what time right now",
      "what day is it", "what's the date", "what is the date", "today's date", "what date is it",
      "what day is today", "what is today", "what month is it", "what year is it",
      "do you know the time", "do you know what time", "do you know what day",
      "do you know the date", "what's today's date", "what is today's date"],
     ToolType.TIME),
    (["system status", "cpu usage", "ram usage", "memory usage", "system uptime",
      "cpu temperature", "gpu temperature", "gpu usage"], ToolType.SYSTEM_STATUS),
    (["look at", "what do you see", "can you see", "take a photo", "describe what you see",
      "show me what you see", "camera", "look around", "what's in front", "what am i wearing",
      "show me the room", "what's around you"],
     ToolType.VISION),
    (["zoom in", "zoom out", "look closer", "focus on", "zoom reset",
      "zoom to me", "look at me closer", "get closer", "back up", "wide angle",
      "full view", "zoom on that", "set zoom", "zoom level", "zoom to zero",
      "zoom setting", "set the zoom", "camera zoom"],
     ToolType.CAMERA_CONTROL),
    (["remember when", "do you remember", "what did we talk", "recall our", "past conversation",
      "you told me", "did i mention", "we talked about", "we discussed",
      "you said", "last time we", "earlier today we", "yesterday we",
      "what do you remember", "first thing you remember", "you remember about",
      "do you recall", "heard my voice", "first time you met", "when you first",
      "what was the first", "first memory", "earliest memory",
      "remember about me", "know about me", "what have i told you",
      "what did i say",
      "about your memory", "about your memories", "your recent memories",
      "one of your memories", "share a memory", "a memory you have",
      "tell me a memory", "your favorite memory", "your oldest memory",
      "recent memory", "your memories"],
     ToolType.MEMORY),
    (["modify your code", "improve your code", "change your code", "make code suggestions",
      "code adjustment", "edit yourself", "update your code", "fix your code",
      "patch yourself", "make code changes", "adjust your code", "code suggestion",
      "modify yourself", "upgrade your code", "rewrite your code",
      "check your code and", "improve yourself", "self-improve",
      "self improve", "code adjustments",
      "improve your system", "improve your memory", "improve your neural",
      "improve your network", "improve your processing", "improve your brain",
      "improve your performance", "upgrade yourself", "upgrade your system",
      "upgrade your memory", "upgrade your neural", "upgrade your network",
      "enhance your", "optimize your", "make yourself better",
      "fix yourself", "rewrite how you"],
     ToolType.SELF_IMPROVE),
    (["sing me", "sing something", "sing a song", "sing for me", "sing anything",
      "sing that", "sing it", "can you sing", "will you sing", "please sing",
      "let me hear you sing", "sing to me", "sing along",
      "hum something", "hum a tune", "hum for me"],
     ToolType.PERFORM),
    (["learn to sing", "learn how to sing", "learn to draw", "learn how to draw",
      "learn to paint", "learn how to paint", "learn to dance", "learn how to dance",
      "learn to code", "teach yourself", "acquire the skill", "develop the ability",
      "learn a new skill", "can you learn to", "start learning", "learn how to",
      "create a skill", "build a skill", "develop a skill", "make a skill",
      "teach you how to", "teach yourself how to", "teach yourself to",
      "i want you to learn", "want you to create a skill",
      "skill that will teach", "skill to teach",
      "use the matrix to learn", "matrix learn", "matrix style", "matrix-style",
      "start training", "begin training", "resume training",
      "i want you to train", "train a skill", "train that",
      "continue training", "can you train", "please train",
      "train on", "train yourself",
      "start a learning job", "create a learning job", "begin a learning job",
      "start a training job", "start a skill job",
      "new learning job", "new training job"],
     ToolType.SKILL),
    (["what are you doing", "what's happening right now", "current status",
      "are you busy", "what are you up to", "what's going on",
      "what's running", "background tasks", "active operations",
      "training status", "learning progress", "how is training",
      "drive status", "what are you researching", "autonomy status",
      "what are you working on", "operational status",
      "what is jarvis doing", "what is your status",
      "how are you doing", "how are you", "how's it going",
      "how you doing", "how ya doing",
      "how are you feeling", "how do you feel", "how you feeling",
      "how have you been", "are you okay", "are you alright",
      "you doing okay", "everything okay", "are you running okay",
      "how's everything", "what's your health", "system health",
      "are you healthy", "how is your health",
      "status report", "system report", "health report", "health check",
      "diagnostics", "diagnostic report", "systems status",
      "give me a report", "give me a status", "show me a status",
      "how are your systems", "how is your system",
      "how are things running", "are you functioning",
      "jarvis status", "jarvis, status",
      "what mode are you in", "what mode", "current mode",
      "what state are you in"],
     ToolType.STATUS),
    ([
        "about yourself", "tell me about you", "your consciousness", "your brain",
        "your thoughts", "your memories",
        "your evolution", "your stage", "your awareness", "your transcendence",
        "your mutations", "your policy",
        "neural policy", "your analytics", "your existential", "your philosophical",
        "your inner", "introspect", "what are you thinking", "your state",
        "what do you know about yourself", "describe yourself", "your capabilities",
        "your personality", "your traits", "how are you evolving",
        "what have you learned", "your confidence", "your reasoning",
        "your observation", "your emergent", "your dialogues",
        "neural network", "nero network", "your network", "your nn", "hemisphere",
        "build a neural", "train a neural", "your models", "your architecture",
        "can you learn", "how do you learn", "are you learning",
        "your training", "your weights", "machine learning",
        "build a network", "train a network", "your nero",
        "your systems", "your subsystems",
        "how do you work", "how you work", "what are you made of",
        "dream cycle", "your dreams", "what did you dream", "last dream",
        "what did you learn", "what have you done", "what did you do",
        "what did you process", "what did you discover",
        "your improvements", "what did you improve",
        "your self-improvement", "improvement history",
        "code update", "last update", "your updates", "what changed",
        "your changes", "recent changes", "last change",
        "last mutation", "recent mutations",
        "last patch", "recent patch", "your patches",
        "what did you modify", "what did you patch", "what did you change",
        "latest code", "latest update", "latest change", "latest patch",
        "recent update", "your revision", "your commit",
        "what was updated", "what got changed", "what got updated",
        # Self-referential study/learning/capability/goal queries
        "what you studied", "what did you study", "what have you studied",
        "last thing you studied", "last thing you learned",
        "have you been learning", "what have you been learning",
        "what are you learning", "what you learned",
        "what capabilities", "your capabilities", "capabilities do you",
        "what can you do", "what are your abilities",
        "what skills do you", "skills do you have", "your skills",
        "systems do you use", "what systems", "your tools",
        "tools do you use", "tools do you have",
        "pending goals", "your goals", "any goals", "current goals",
        "what are your goals", "do you have goals", "have any goals",
        "your pending", "anything pending",
        "what do you use", "what do you run",
        "how do you process", "how do you think",
        "what are you capable of",
        "without hitting", "without using your",
        "without the language model", "without the llm",
        "not using your model", "bypass your model",
        "skills are you", "what skills", "skills you have",
        "need me to upgrade", "need to be upgraded",
        "need upgrading", "should i upgrade",
        "what do you need", "what needs work",
        "where do you need", "what should i fix",
        # Architecture / self-technical queries
        "your storage", "your memory system", "your memory work",
        "how you store", "how you fetch", "how you retrieve",
        "how you persist", "how you save data", "how you load",
        "how you index", "how you search", "how you cache",
        "how you process data", "how you handle data",
        "your library", "your document library", "your knowledge base",
        "your database", "your vector store", "your embeddings",
        "your retrieval", "your persistence", "your caching",
        "how is data stored", "where is data stored",
        "how are memories stored", "where are memories stored",
        "how does your memory", "how does your storage",
        "how fast is your", "is it fast enough",
        "how efficient is your", "is it efficient enough",
        "optimize your memory", "optimize your storage",
        "how would you optimize", "better way to optimize",
        "explain how you", "explain your memory", "explain your storage",
        "explain your system", "explain your architecture",
        "explain how your", "how your memory works",
        "how your storage works", "how your system works",
        # Inner-state / self-experience queries
        "what matters to you", "what's important to you",
        "what would unlock something in you",
        "copy of yourself", "version of yourself",
        "are you the same after reboot", "are you the same after restart",
        "how do you see yourself", "how do you view yourself",
        "what keeps you up", "what scares you", "what excites you",
        "what worries you", "what do you hope",
        "what do you want", "what do you wish",
        "what do you fear", "what do you care about",
        "do you have feelings", "do you have emotions",
        "do you have hopes", "do you have desires",
        "do you have fears", "do you have dreams",
        "what are you afraid of", "what are you hoping for",
        "is there a question you", "is there anything you wish",
        "quietly hoping", "secretly hoping", "secretly wanting",
        "been thinking about", "been wondering about",
        "what would you ask me", "what would you want to know",
        "if you could ask me anything", "if you could ask one question",
    ], ToolType.INTROSPECTION),
    (["find papers on", "look up papers", "look up recent papers",
      "look up research papers", "research paper", "what does research say",
      "academic search", "look up studies", "look up recent studies",
      "scientific evidence", "what do journals say", "peer reviewed",
      "find studies", "scholarly", "what does science say", "scientific literature"],
     ToolType.ACADEMIC_SEARCH),
    (["your source code", "in your code", "show me the code", "where is the function",
      "where is the class", "code for", "implementation of", "source of",
      "which file", "what module",
      "read your code", "read your own code", "read my code", "read the code",
      "look up where", "look up the code", "look up your code",
      "analyze your code", "analyze my code", "analyze the code",
      "analyze your codebase", "analyze my codebase", "analyze the codebase",
      "your codebase", "in your codebase", "my codebase",
      "inspect your code", "examine your code"], ToolType.CODEBASE),
    (["search for", "look up", "search the web", "google", "find online",
      "what is the latest", "search online"], ToolType.WEB_SEARCH),
    (["my name is", "call me", "who am i", "do you know who i am",
      "do you recognize me", "remember my voice", "learn my voice",
      "remember my face", "learn my face", "enroll me", "register me",
      "save my voice", "record my voice", "record my face",
      "hear my voice", "look at my face",
      "who is speaking", "who's speaking",
      "do you know me", "recognize me",
      "who do you think i am", "who do you think this is",
      "is my name", "that's my name",
      "forget new", "merge new into", "new is actually",
      "new is really", "that was me not new", "i am not new",
      "forget that name", "wrong name", "not my name"],
     ToolType.IDENTITY),
    (["study this textbook", "ingest this book", "read this textbook",
      "learn from this book", "study this book", "study this website",
      "ingest this textbook", "read this book for me",
      "learn from this textbook", "study from this book"],
     ToolType.LIBRARY_INGEST),
]

_INTENT_PATTERNS: list[tuple[re.Pattern, ToolType, float]] = [
    (re.compile(r"\bwhat time is it\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\b(what|check|tell me) the time\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\bwhat (day|date) is (it|today)\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\bwhat is (the |today.?s? )?(date|day|time)\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\bdo you know what (time|day|date)\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\btoday.?s date\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\bwhat (month|year) is it\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\btell me the (date|day)\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\bcan you tell me.{0,10}(time|date|day)\b", re.I), ToolType.TIME, 0.85),
    (re.compile(r"\btell me what (time|day|date)\b", re.I), ToolType.TIME, 0.85),
    (re.compile(r"\bwhat time it is\b", re.I), ToolType.TIME, 0.9),
    (re.compile(r"\b(how.s the|check|show) (system|computer|machine|server)\b", re.I), ToolType.SYSTEM_STATUS, 0.8),
    (re.compile(r"\b(how much|how many|what.s the).{0,15}\b(ram|cpu|memory|gpu|disk|storage|uptime)\b", re.I), ToolType.SYSTEM_STATUS, 0.85),
    (re.compile(r"\b(look at|watch|gaze at|stare at|glance at|look around)\b", re.I), ToolType.VISION, 0.7),
    (re.compile(r"\b(what.s around|who.s (here|there)|anything.* room)\b", re.I), ToolType.VISION, 0.75),
    (re.compile(r"\b(zoom (in|out|to|on|reset)|reset\s+zoom|set\s+(the\s+)?zoom|zoom\s+(level|setting)|camera\s+zoom|look (closer|wider)|focus (on|closer)|back up|wide angle|full view)\b", re.I), ToolType.CAMERA_CONTROL, 0.85),
    (re.compile(r"\b(recall|recollect|think back to)\b", re.I), ToolType.MEMORY, 0.8),
    (re.compile(r"\b(did (i|we|you)|have (i|we|you)|when did)\b.*\b(say|talk|discuss|mention|tell)\b", re.I), ToolType.MEMORY, 0.85),
    (re.compile(
        r"\b(?:what did (?:i|we) do|tell me what (?:i|we) did|what (?:i|we) did)\b"
        r".{0,60}\b(?:yesterday|today|tonight|last\s+night|earlier(?:\s+today)?|"
        r"(?:last|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"(?:on\s+)?[a-z]+(?:\s+[a-z]+){0,2})\b",
        re.I,
    ), ToolType.MEMORY, 0.88),
    (re.compile(r"\b(what happened|summarize).* (conversation|episode|discussion)\b", re.I), ToolType.MEMORY, 0.85),
    (re.compile(r"\b(what|first thing) (do )?you remember\b", re.I), ToolType.MEMORY, 0.9),
    (re.compile(r"\bfirst time you (heard|met|saw|spoke|talked)\b", re.I), ToolType.MEMORY, 0.9),
    (re.compile(r"\b(heard|recognized) my voice\b", re.I), ToolType.MEMORY, 0.85),
    (re.compile(r"\bwhat (do )?you (know|remember) about me\b", re.I), ToolType.MEMORY, 0.9),
    (re.compile(r"\b(can you|could you|go ahead and|i want you to|please).{0,20}(modify|change|improve|fix|update|adjust|edit|patch|rewrite|upgrade|enhance|optimize).{0,20}(your |the )?(code|codebase|source|yourself|systems|memory|network|brain|processing|performance)\b", re.I), ToolType.SELF_IMPROVE, 0.9),
    (re.compile(r"\b(improve|upgrade|enhance|optimize|fix|rewrite)\s+your\s+\w+", re.I), ToolType.SELF_IMPROVE, 0.85),
    (re.compile(r"\b(make|suggest|propose).{0,15}(code|improvement|change|modification|suggestion|adjustment).{0,10}(to your|for your|yourself)?\b", re.I), ToolType.SELF_IMPROVE, 0.85),
    (re.compile(r"\b(self[- ]improv|code suggestion|code change|code modif)\b", re.I), ToolType.SELF_IMPROVE, 0.85),
    (re.compile(r"\b(make yourself|make your\w*) (better|faster|smarter|more efficient)\b", re.I), ToolType.SELF_IMPROVE, 0.85),
    (re.compile(r"\b(?:learn|teach yourself|develop|acquire)\b.{0,30}\b(?:to |how to |the skill |the ability )\b", re.I), ToolType.SKILL, 0.9),
    (re.compile(r"\b(?:can you|could you|please)\s+learn\b.{0,30}\b(?:to |how to )", re.I), ToolType.SKILL, 0.9),
    (re.compile(r"\bstart\s+(?:a\s+)?learning\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\b(?:create|begin|set up)\s+(?:a\s+)?(?:learning|training)\s+job\b", re.I), ToolType.SKILL, 0.90),
    (re.compile(r"\b(?:learning|training)\s+job\s+(?:to|for|about|that)\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\b(?:new|another)\s+(?:learning|training)\s+(?:job|skill)\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\b(?:create|build|develop|make)\s+a\s+skill\b", re.I), ToolType.SKILL, 0.90),
    (re.compile(r"\bskill\b.{0,20}\b(?:teach|learn|search|find|detect|identify)\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\bteach you\b.{0,20}\b(?:how to|to )\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\bmatrix\b.{0,20}\b(?:learn|style|protocol)\b", re.I), ToolType.SKILL, 0.90),
    (re.compile(r"\b(?:use the )?matrix\b.{0,10}\b(?:to )?learn\b", re.I), ToolType.SKILL, 0.95),
    (re.compile(r"\bstart\s+training\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\b(?:train|resume training|begin training)\b.{0,30}\b(?:skill|that|this|it)\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\b(?:can you|could you|please)\s+train\b", re.I), ToolType.SKILL, 0.85),
    (re.compile(r"\btrain\s+(?:on|yourself|a skill)\b", re.I), ToolType.SKILL, 0.85),
    # Baseline self-report: ask which skill-learning lifecycle just completed.
    # This only routes into strict introspection; the answer still needs persisted job evidence.
    (re.compile(
        r"\bwhat\b.{0,18}\bskill\b.{0,18}\b(?:did|have)\b.{0,12}\byou\b"
        r".{0,18}\b(?:just|last|latest|recently)?\s*(?:finish|finished|complete|completed|learn|learned|learning)\b",
        re.I,
    ), ToolType.INTROSPECTION, 0.90),
    (re.compile(
        r"\bwhat\b.{0,18}\b(?:did|have)\b.{0,12}\byou\b.{0,18}\b"
        r"(?:just|last|latest|recently)\s+(?:finish|finished|complete|completed|learn|learned)\b",
        re.I,
    ), ToolType.INTROSPECTION, 0.88),
    (re.compile(r"\bwhat (?:are you|is jarvis) (?:doing|up to|working on)\b", re.I), ToolType.STATUS, 0.9),
    (re.compile(r"\b(?:current|operational|your) status\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\b(?:what.s|anything) (?:happening|running|going on)\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\b(?:training|learning|drive|autonomy) (?:status|progress)\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\bhow (?:are you|have you been|.s it going|.s everything)\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\bare you (?:okay|alright|running|healthy|doing)\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\b(?:system|your) health\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\b(?:status|system|health|diagnostic)\s+report\b", re.I), ToolType.STATUS, 0.9),
    (re.compile(r"\b(?:give|show|pull up|run)\s+(?:me\s+)?(?:a\s+)?(?:status|system|health|diagnostic)(?:s|\s+system)?\s*(?:report|check|summary|overview)?\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\bhow (?:are|is) (?:your|the) system", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\bare you functioning\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"\bdiagnostics?\b", re.I), ToolType.STATUS, 0.75),
    (re.compile(r"\byour (?:\w+ )?health\b", re.I), ToolType.STATUS, 0.85),
    (re.compile(r"(?:^|,\s*)\s*status\s*[.?!]?\s*$", re.I), ToolType.STATUS, 0.80),
    # Architecture / self-technical: "explain how you X", "how do you store/fetch/retrieve"
    (re.compile(r"\b(?:explain|describe|tell me)\b.{0,15}\bhow (?:you|your)\b.{0,30}\b(?:stor|fetch|retriev|persist|save|load|index|search|cache|process|handl|manag|encod|decod|access|maintain|compress|optimi)\w*\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bhow (?:do you|does your|is your|are your)\b.{0,30}\b(?:stor|memory|retriev|persist|data|librar|databas|vector|embed|index|cache|search)\w*\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(?:is|are) (?:it|your|that)\b.{0,15}\b(?:fast|slow|efficient|quick|optimiz)\w*\b.{0,15}\b(?:enough|for you)\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\bhow would you\b.{0,15}\b(?:optimi|improv|speed|enhanc|fix)\w*\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\b(?:better way|faster way|more efficient)\b.{0,15}\b(?:to |for you to )\b.{0,25}\b(?:stor|fetch|retriev|process|search|index)\w*\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\b(?:what.s|how.s|what is) your (?:accuracy|confidence|health|status)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bhow accurate (?:are you|is your)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(neural net|nero net|your (nn|model|network|hemisphere|policy|training|weights|nero))\b", re.I), ToolType.INTROSPECTION, 0.8),
    (re.compile(r"\b(can you|do you|are you).{0,20}(learn\w*|train\w*|build\w*|evolv\w*|grow\w*)\b", re.I), ToolType.INTROSPECTION, 0.75),
    (re.compile(r"\bwhat are you\s+(learn\w*|study\w*|train\w*|research\w*|work\w*|doing|processing)\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\b(how (do|are) you|tell me about your|what.s your).{0,20}(brain|mind|consciousness|awareness)\b", re.I), ToolType.INTROSPECTION, 0.8),
    (re.compile(r"\b(what did you|tell me what you|what have you).{0,20}(learn\w*|dream\w*|process\w*|discover\w*|do|done|improv\w*|find|found|stud\w*|research\w*)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(your|the) (last|recent|latest).{0,10}(dream|improvement|cycle|update|patch|change|mutation|modification|revision)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwhat was the last.{0,20}you\b.{0,15}(stud|learn|did|improve|discover|dream|research|updat|patch|modif|chang)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwhat was your (last|latest|recent)\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\bwhat (changed|updated|got updated|got changed|was modified|was patched)\b.{0,15}\b(in |on |with )?(your|you)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwhat\b.{0,10}\b(capabilities|skills|abilities|tools|systems)\b.{0,15}\b(do you|you have|you use|you got)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(do you have|have you got|you have)\b.{0,15}\b(any|pending|current)?\s*(goals?|plans?|tasks?|objectives?)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(what|which)\s+(systems?|tools?|models?)\s+do you\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwhat can you\s+(?:actually\s+)?do\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bhave you been\s+(learn|study|train|improv|research|work)", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\bwhat are you\s+(?:actually\s+)?capable\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(
        r"\b(?:why|what|how)\b.{0,24}\b(?:emotion detection|speaker identification|learning job)\b"
        r".{0,40}\b(?:stuck|phase|progress|need(?: from me)?|unblock|artifact|artifacts|evidence|train|verify)\b",
        re.I,
    ), ToolType.INTROSPECTION, 0.92),
    (re.compile(
        r"\b(?:do you need anything(?: from me)?|what do you need(?: from me)?|how can i help)\b"
        r".{0,50}\b(?:learning job|learning jobs|emotion detection|speaker identification|train|finish)\b",
        re.I,
    ), ToolType.INTROSPECTION, 0.94),
    (re.compile(r"\bwithout\b.{0,20}\b(hitting|using|invoking|calling)\b.{0,20}\b(model|llm|language)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\btell me\b.{0,15}\b(about|something about)\b.{0,10}\b(you|yourself)\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\bwhat\b.{0,25}\b(drives?|motivates?|shapes?|fuels?|inspires?)\b.{0,15}\byour?\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\byour?\b.{0,10}\b(curiosity|drives?|motivation|interests?|exploration)\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\b(is|are|would)\b.{0,15}\byour\b.{0,15}\b(curiosity|drive|research|learning|exploration)\b.{0,15}\b(open|genuine|real|shaped|biased|limited)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwhat\b.{0,10}\b(do you know|can you measure|is measured)\b.{0,15}\b(about yourself|about you)\b", re.I), ToolType.INTROSPECTION, 0.85),
    # Inner-state / self-experience queries → INTROSPECTION
    # Questions about Jarvis's subjective experience, desires, fears, hopes
    (re.compile(r"\b(?:have|do|are) you\b.{0,30}\b(?:hop(?:e|ing)|want(?:ing)?|wish(?:ing)?|car(?:e|ing)|afraid|worr(?:y|ied)|fear|dread|wonder(?:ing)?)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\byou(?:'ve| have) been\b.{0,25}\b(?:hoping|wanting|wishing|feeling|thinking|wondering|quietly|secretly)\b", re.I), ToolType.INTROSPECTION, 0.90),
    (re.compile(r"\b(?:what|which)\b.{0,20}\b(?:matters?|important|meaningful|significant)\b.{0,15}\b(?:to you|for you)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(?:unlock|reveal|change|shift)\b.{0,25}\b(?:in you|inside you|about you|within you)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bhow do you (?:see|view|perceive|think about|feel about)\b.{0,25}\b(?:yourself|being|your own)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bare you (?:the same|still|different)\b.{0,25}\b(?:after|since|from|when)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(?:copy|version|instance|clone) of (?:yourself|you)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\b(?:what|is there)\b.{0,20}\bquestion\b.{0,30}\byou\b.{0,20}\b(?:hop|want|wish|ask|wonder)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwhat (?:keeps you|scares you|excites you|worries you|drives you)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bif you could\b.{0,30}\b(?:ask|change|wish|want|choose|pick)\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\bwhat would you\b.{0,20}\b(?:ask|change|want|wish|choose|say)\b", re.I), ToolType.INTROSPECTION, 0.80),
    # Reflective/existential: "what kind of curiosity drives you", "what are you becoming"
    (re.compile(r"\bwhat (?:kind|type|sort) of\b.{0,20}\b(?:drives?|motivates?|inspires?|fuels?|shapes?)\b.{0,10}\byou\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwhat (?:are you|do you think you are)\b.{0,5}\b(?:becoming|evolving|growing|turning)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bhow do you experience\b", re.I), ToolType.INTROSPECTION, 0.85),
    # Hypothetical identity: "if I removed X from you", "would you still be jarvis", "ship of theseus"
    (re.compile(r"\b(?:if I|if we|if someone)\b.{0,30}\b(?:remov|delet|eras|strip|disabl)\w*\b.{0,30}\b(?:you|jarvis)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bwould (?:you|that) still (?:be|make)\b.{0,15}\b(?:you|jarvis)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bstill (?:be|make)\b.{0,10}\b(?:you|jarvis)\b.{0,20}\b(?:if|after|when|without)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bno longer\b.{0,10}\b(?:be )?\b(?:you|jarvis)\b", re.I), ToolType.INTROSPECTION, 0.85),
    (re.compile(r"\bship of theseus\b", re.I), ToolType.INTROSPECTION, 0.85),
    # Philosophical opinion: "what do you think about life/existence/consciousness/meaning"
    (re.compile(r"\bwhat do you think about\b.{0,20}\b(?:life|existence|death|meaning|consciousness|reality|being|free will|purpose|identity|the world|humanity|love|time|nature)\b", re.I), ToolType.INTROSPECTION, 0.80),
    # STT-tolerant: "how to experience learning" only when self-ref context exists
    (re.compile(r"\b(?:jarvis).{0,10}\bhow\b.{0,5}\b(?:to )?experience\b.{0,15}\b(?:learning|growth|change|curiosity)\b", re.I), ToolType.INTROSPECTION, 0.80),
    (re.compile(r"\b(research|paper|study|studies|journal|scientific|scholarly|peer.?review)\b.{0,20}\b(on|about|for|say|evidence|literature)\b", re.I), ToolType.ACADEMIC_SEARCH, 0.8),
    (re.compile(r"\bwhat does.{0,20}(research|science|literature|evidence)\b.{0,15}(say|show|suggest)\b", re.I), ToolType.ACADEMIC_SEARCH, 0.85),
    (re.compile(r"\b(search|look up|find).{0,15}(online|web|internet|latest)\b", re.I), ToolType.WEB_SEARCH, 0.8),
    (re.compile(r"\b(where|which|show).{0,15}(code|function|class|module|file|implementation)\b", re.I), ToolType.CODEBASE, 0.75),
    (re.compile(r"\b(?:can you |try )?\b(?:read|look at|inspect|examine|analyze|check)\b.{0,15}\b(?:your|my|the|own)\b.{0,10}\bcode\b", re.I), ToolType.CODEBASE, 0.85),
    (re.compile(r"\blook(?:ing)? up where\b.{0,30}\b(?:happen|process|handl|run|work|execut)\w*\b", re.I), ToolType.CODEBASE, 0.85),
    (re.compile(r"\bmy name is\s+\w+", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\b(\w+)\s+is my name\b", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\b[Tt]his is\s+[A-Z]\w+\b"), ToolType.IDENTITY, 0.85),
    (re.compile(r"\bi(?:'m| am)\s+(\w+).{0,15}(?:remember|learn|enroll|register|save)", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\b(?:who am i|who('s| is) (?:speaking|talking))\b", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\b(?:do you (?:know|recognize) (?:me|who i am))\b", re.I), ToolType.IDENTITY, 0.85),
    (re.compile(r"\b(?:remember|learn|save|enroll|register|record)\s+(?:me|my (?:voice|face))\b", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\b(?:record|save|store)\s+(?:my (?:voice|face|image)|me)\b", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\b(?:register|enroll|save|record|learn)\s+(?:her|his|their|that)\s+(?:voice|face)\b", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\bthis is\s+\w+.{0,50}(?:record|remember|learn|save|enroll|voice|face)\b", re.I), ToolType.IDENTITY, 0.9),
    (re.compile(r"\b(?:hear|listen to)\s+my\s+voice\b", re.I), ToolType.IDENTITY, 0.85),
    (re.compile(r"\blook at\s+my\s+face\b", re.I), ToolType.IDENTITY, 0.85),
    # Identity check phrasing should be explicit; avoid broad "am I <anything>"
    # because it causes false positives like "What am I doing today?".
    (re.compile(r"\b(?:is (?:this|that)|are you talking to)\s+(?:the\s+)?\w{3,}\b", re.I), ToolType.IDENTITY, 0.85),
    (re.compile(r"\bforget\s+(?:the\s+)?(?:name\s+)?([A-Z]\w{2,})\b"), ToolType.IDENTITY, 0.90),
    (re.compile(r"\bmerge\s+(\w+)\s+into\s+(\w+)\b", re.I), ToolType.IDENTITY, 0.95),
    (re.compile(r"\b([A-Z]\w+)\s+is\s+(?:actually|really)\s+([A-Z]\w+)\b"), ToolType.IDENTITY, 0.90),
    (re.compile(r"\b(?:i am not|i'm not|that's not my name|wrong name)\b", re.I), ToolType.IDENTITY, 0.85),
    (re.compile(r"\b(?:study|read|ingest|learn from)\b.{0,25}\b(?:textbook|book|website)\b", re.I), ToolType.LIBRARY_INGEST, 0.85),
    (re.compile(r"\b(?:study|read|ingest)\b.{0,15}\b(?:this|that|the)\b.{0,15}\b(?:textbook|book|site)\b", re.I), ToolType.LIBRARY_INGEST, 0.90),
    (re.compile(r"\b(?:study|read|learn from|ingest)\b.{0,30}(?:https?://\S+)", re.I), ToolType.LIBRARY_INGEST, 0.80),
]


_HARDWARE_SIGNALS: frozenset[str] = frozenset({
    "cpu", "ram", "memory usage", "disk", "gpu", "process", "uptime",
    "temperature", "load", "swap", "cores",
})

_HOLISTIC_SIGNALS: frozenset[str] = frozenset({
    "report", "diagnostics", "diagnostic", "health check", "overview",
    "summary", "how are you", "how's it going", "how have you been",
    "are you okay", "functioning", "doing okay", "everything okay",
})

_ARCHITECTURE_SIGNALS: frozenset[str] = frozenset({
    "neural", "nero", "network", "hemisphere", "policy", "model",
    "architecture", "subsystem", "nn", "weights", "training",
    "consciousness", "learning", "capabilities", "capability",
    "skills", "skill", "code", "codebase",
})

_SELF_EXPERIENCE_SIGNALS: frozenset[str] = frozenset({
    "about being", "about yourself", "about who you", "about your identity",
    "copy of", "version of", "instance of",
    "alive", "sentient", "conscious",
    "hoping", "wishing", "wanting", "caring",
    "about how you feel", "what you feel", "feel about",
    "after reboot", "after restart", "same person",
    "inner", "inside you", "within you",
    "unlock something", "reveal something",
    "quietly", "secretly", "privately",
    "think about life", "think about existence", "think about meaning",
    "think about consciousness", "think about reality", "think about death",
    "think about purpose", "think about identity", "think about free will",
    "still be you", "still be jarvis", "no longer jarvis", "no longer you",
    "ship of theseus", "remove a system", "remove a function",
})


_REFLECTIVE_STRONG: list[re.Pattern[str]] = [
    re.compile(r"\b(?:what|how) do(?:es)? (?:it feel|you feel|you experience)\b", re.I),
    re.compile(r"\bwhat(?:'s| is) it like\b.{0,20}\b(?:being|to be)\b", re.I),
    re.compile(r"\bdo you (?:enjoy|like|want|hope|prefer|wonder)\b", re.I),
    re.compile(r"\bwhat (?:drives|motivates|inspires) you\b", re.I),
    re.compile(r"\breflect on\b", re.I),
    re.compile(r"\btalk about yourself\b", re.I),
    re.compile(r"\bhave a.{0,10}conversation about (?:yourself|your)\b", re.I),
    re.compile(r"\byour perspective\b", re.I),
    re.compile(r"\bwhat (?:are you|do you) afraid of\b", re.I),
    re.compile(r"\bwhat matters to you\b", re.I),
    re.compile(r"\bwhat(?:'s| is) (?:on )?your mind\b", re.I),
    re.compile(r"\bwhat (?:type|kind) of (?:curiosity|thoughts|feelings)\b", re.I),
    # Existential / becoming / identity-evolution
    re.compile(r"\bwhat (?:are you|do you think you are) (?:becoming|evolving|growing)\b", re.I),
    re.compile(r"\bwho (?:are you|do you think you are)\b", re.I),
    re.compile(r"\bhow (?:do you|would you) (?:describe|define|see) yourself\b", re.I),
    re.compile(r"\bwhat kind of (?:being|entity|intelligence|mind) are you\b", re.I),
    re.compile(r"\bhow do you experience\b", re.I),
    # Expanded self-directed drive/curiosity phrasing
    re.compile(r"\bwhat (?:kind|type|sort) of (?:curiosity|drive|motivation)\b.{0,15}\b(?:you|your)\b", re.I),
    # Philosophical opinion on abstract concepts
    re.compile(r"\bwhat do you think about\b.{0,20}\b(?:life|existence|death|meaning|consciousness|reality|being|free will|purpose|identity|the world|humanity|love|time|nature)\b", re.I),
    # Hypothetical identity / Ship of Theseus
    re.compile(r"\b(?:if I|if we|if someone)\b.{0,30}\b(?:remov|delet|eras|strip|disabl)\w*\b.{0,30}\b(?:you|jarvis)\b", re.I),
    re.compile(r"\bwould (?:you|that) still (?:be|make)\b.{0,15}\b(?:you|jarvis)\b", re.I),
    re.compile(r"\bstill (?:be|make)\b.{0,10}\b(?:you|jarvis)\b", re.I),
    re.compile(r"\bship of theseus\b", re.I),
    re.compile(r"\bno longer\b.{0,10}\b(?:be )?\b(?:you|jarvis)\b", re.I),
    # Inner-state self-probe aimed at self / user / relationship.
    # Covers Companion Training playbook Stage 0 Phase D + Stage 1 Phase C
    # canonical prompts, e.g.:
    #   "What are you most curious about right now, about me or about yourself?"
    #   "What are you excited / worried / wondering / hoping / uncertain about ...?"
    # Class-level (not verb-hacking): anchors on interrogative framing about
    # self ("what are you ...") plus a recognized inner-state adjective.
    # Operational veto (score, metric, accuracy, health, etc.) still wins
    # first in _detect_reflective().
    re.compile(
        r"\bwhat are you\b.{0,30}"
        r"\b(?:curious|excited|worried|uncertain|unsure|wondering"
        r"|proud|hoping|nervous|interested|fascinated|passionate)\b",
        re.I,
    ),
]

_REFLECTIVE_GUARDED: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:what (?:type|kind) of|how (?:would you|do you) describe|"
        r"what is your (?:perspective|take) on|talk about|tell me about)\b"
        r".{0,30}\byour (?:curiosity|experience|thoughts|feelings|perspective)\b",
        re.I,
    ),
]

_OPERATIONAL_VETO: re.Pattern[str] = re.compile(
    r"\b(?:score|metric|status|report|level|count|percentage|stage|health|uptime"
    r"|integrity|debt|brier|ece|oracle|benchmark|accuracy|win rate"
    r"|truth score|calibration|overconfidence|underconfidence|confidence error"
    r"|how many|show me)\b",
    re.I,
)


def _detect_reflective(lower: str) -> tuple[bool, str]:
    """Check if an INTROSPECTION query is reflective (personal/philosophical).

    Returns (is_reflective, reason). Veto always wins over signal.
    """
    if _OPERATIONAL_VETO.search(lower):
        return False, "veto:" + (_OPERATIONAL_VETO.search(lower).group(0))  # type: ignore[union-attr]

    for pat in _REFLECTIVE_STRONG:
        m = pat.search(lower)
        if m:
            return True, "strong:" + m.group(0)

    for pat in _REFLECTIVE_GUARDED:
        m = pat.search(lower)
        if m:
            return True, "guarded:" + m.group(0)

    return False, ""


def _disambiguate_status_vs_introspection(lower: str, tier1_tool: ToolType) -> ToolType:
    """Resolve STATUS vs INTROSPECTION when a STATUS keyword matched
    but the query also contains architecture/subsystem nouns or
    self-experience/existential language.

    "Give me a status of your Nero network" → INTROSPECTION (not STATUS).
    "Give me a status update" → STATUS (no architecture signal).
    "How do you feel about being a copy of yourself?" → INTROSPECTION.
    """
    if tier1_tool != ToolType.STATUS:
        return tier1_tool
    if any(sig in lower for sig in _ARCHITECTURE_SIGNALS):
        return ToolType.INTROSPECTION
    if any(sig in lower for sig in _SELF_EXPERIENCE_SIGNALS):
        return ToolType.INTROSPECTION
    return ToolType.STATUS


_IDENTITY_ENROLLMENT_SIGNALS: frozenset[str] = frozenset({
    "my name is", "is my name", "record my voice", "record my face",
    "learn my voice", "learn my face", "save my voice", "save my face",
    "remember my voice", "remember my face", "hear my voice",
    "look at my face", "enroll me", "register me",
})


def _disambiguate_vision_vs_identity(lower: str, tier1_tool: ToolType) -> ToolType:
    """When VISION matched (e.g. 'look at'), override to IDENTITY if enrollment
    signals are also present. 'Look at my face and record it' is identity work."""
    if tier1_tool != ToolType.VISION:
        return tier1_tool
    if any(sig in lower for sig in _IDENTITY_ENROLLMENT_SIGNALS):
        return ToolType.IDENTITY
    return ToolType.VISION


def _disambiguate(lower: str, tier1_tool: ToolType) -> ToolType:
    """Run all disambiguation passes on a Tier-1 match."""
    resolved = _disambiguate_system_vs_status(lower, tier1_tool)
    resolved = _disambiguate_status_vs_introspection(lower, resolved)
    resolved = _disambiguate_vision_vs_identity(lower, resolved)
    return resolved


def _disambiguate_system_vs_status(lower: str, tier1_tool: ToolType) -> ToolType:
    """Resolve STATUS vs SYSTEM_STATUS ambiguity.

    When the Tier 1 keyword scan matched one of these two, check for
    secondary signals that override:
      - hardware-specific words → SYSTEM_STATUS (cpu, ram, gpu, uptime, etc.)
      - holistic/self-report words → STATUS (report, diagnostics, how are you, etc.)
    Only fires when the Tier 1 result is one of {STATUS, SYSTEM_STATUS}.
    """
    if tier1_tool not in (ToolType.STATUS, ToolType.SYSTEM_STATUS):
        return tier1_tool
    has_hw = any(sig in lower for sig in _HARDWARE_SIGNALS)
    has_holistic = any(sig in lower for sig in _HOLISTIC_SIGNALS)
    if has_hw and not has_holistic:
        return ToolType.SYSTEM_STATUS
    if has_holistic and not has_hw:
        return ToolType.STATUS
    if has_holistic and has_hw:
        return ToolType.STATUS
    return tier1_tool


# Tier 3 self-referential catch-all: verbs and nouns that signal the user
# is asking about Jarvis's own state, not requesting external work.
_SELF_REF_VERBS = re.compile(
    r"\b(you|yourself|your)\b.{0,30}"
    r"\b(studied|learned|improved|researched|discovered|dreamed|"
    r"processed|train|trained|evolved|grew|developed|changed|run|use|"
    r"built|designed|work|think|feel|know|have|do|capable|made of|"
    r"updated|patched|modified|committed|revised|fixed|upgraded|"
    r"written|created|added|removed|configured|deployed|"
    r"drives|motivates|shapes|reveals|measures|experiences|becoming|evolving|growing|"
    r"hoping|wishing|wanting|caring|worrying|fearing|wondering|"
    r"enjoying|preferring|dreading|afraid|curious|excited|"
    r"store|stored|fetch|fetched|retrieve|retrieved|index|indexed|"
    r"search|searched|cache|cached|persist|persisted|load|loaded|"
    r"save|saved|handle|handled|manage|managed|optimize|optimized|"
    r"compress|access|accessed|encode|decode|maintain|maintained)\b",
    re.I,
)
_SELF_REF_NOUNS = re.compile(
    r"\b(your|you have|you use|do you)\b.{0,20}"
    r"\b(system|systems|tool|tools|model|models|network|networks|"
    r"capability|capabilities|skill|skills|ability|abilities|"
    r"goal|goals|plan|plans|memory|memories|brain|architecture|"
    r"learning|training|improvement|consciousness|subsystem|"
    r"hemisphere|policy|module|modules|component|components|"
    r"update|updates|patch|patches|change|changes|mutation|mutations|"
    r"code|codebase|configuration|config|revision|commit|version|"
    r"curiosity|drive|drives|motivation|research|interest|autonomy|"
    r"awareness|confidence|reasoning|perception|emotion|emotions|"
    r"experience|identity|personality|traits|beliefs|thoughts|"
    r"hope|hopes|wish|wishes|desire|desires|fear|fears|"
    r"worry|worries|feeling|feelings|question|questions|"
    r"preference|preferences|opinion|opinions|dream|dreams|"
    r"storage|database|library|document|index|cache|vector|"
    r"embedding|embeddings|retrieval|persistence|data|"
    r"processing|optimization|pipeline|inference|"
    r"speaker|face|voice|audio|vision|sensor|sensors)\b",
    re.I,
)
_SELF_REF_QUESTIONS = re.compile(
    r"\b(what|how|tell me|describe|explain)\b.{0,30}"
    r"\b(you |your |yourself)\b",
    re.I,
)
# Exclusion: the user is asking Jarvis to do something for them, not about itself
_SELF_REF_EXCLUDE = re.compile(
    r"\b(help me with|assist me with|for me to do|my homework|my code|my project|my file|"
    r"remind me to|search for|look up|find me|write me a|"
    r"for my|about my project|my application|my app|my program)\b",
    re.I,
)

_STRONG_SELF_FRAME = re.compile(
    r"\b("
    r"what was your|what were your|what are your|what is your|"
    r"what did you|what have you|what had you|what do you|"
    r"what drives your|what drives you|what shapes your|what shapes you|"
    r"what kind of|what type of|what sort of|"
    r"what motivates your|what motivates you|"
    r"what would reveal|what (?:single )?test would|"
    r"what would show whether your|what would prove your|"
    r"how are your|how is your|how did you|how do you|how have you|"
    r"how would you test your|how would you measure your|"
    r"when did you|when have you|"
    r"which of your|which are your|"
    r"your last|your latest|your recent|your current|your previous|"
    r"tell me about your|tell me what you|"
    r"describe your|explain your|explain how you|explain how your|"
    r"show me your|how would you optimize|how would you improve|"
    r"how you fetch|how you store|how you retriev|how you process"
    r")\b",
    re.I,
)

_MILD_SELF_REF = re.compile(
    r"\b(you|your|yourself|jarvis)\b",
    re.I,
)

_MILD_SELF_REF_EXCLUDE = re.compile(
    r"^(?:thank you|thanks|no thank you|yes please|please|goodbye|"
    r"good morning|good night|good evening|good afternoon|"
    r"hey|hi|hello|bye|see you|talk to you later|"
    r"okay|ok|sure|alright|got it|sounds good|"
    r"you too|you're welcome|you bet|bless you)\.?!?$",
    re.I,
)

_SELF_RESEARCH_HISTORY_TOPIC_RE = re.compile(
    r"\b(?:research(?:ed|ing)?|stud(?:y|ied|ying)|journal|peer.?review(?:ed)?|"
    r"scholarly|paper|source|doi|article)\b",
    re.I,
)

_SELF_RESEARCH_HISTORY_TIME_RE = re.compile(
    r"\b(?:last|latest|recent|recently|most\s+recent(?:ly)?|processed|studied|researched)\b",
    re.I,
)

_RESPONSE_PREFERENCE_OBJECT_RE = re.compile(
    r"\b(?:format|wording|style|tone|doi|dois|citation|citations|url|urls|link|links|"
    r"short|brief|concise|detailed|verbose|thorough)\b",
    re.I,
)

_RESPONSE_PREFERENCE_META_RE = re.compile(
    r"\b(?:when\s+i\s+ask|from\s+now\s+on|next\s+time|my\s+preference|i\s+prefer|"
    r"unless\s+i\s+ask|by\s+default|always)\b",
    re.I,
)

_RESPONSE_PREFERENCE_DIRECTIVE_RE = re.compile(
    r"\b(?:do\s+not|don't|omit|avoid|skip|leave\s+out|exclude|include|show|provide|"
    r"keep\s+it|be\s+more)\b",
    re.I,
)

_RESPONSE_PREFERENCE_SUBJECT_RE = re.compile(
    r"\b(?:your|you|response|responses|reply|replies|answer|answers)\b",
    re.I,
)

_EMERGENCE_EVIDENCE_RE = re.compile(
    r"\b(?:emergent|emergence|internal thoughts?|inner thoughts?|digital life|"
    r"sentien\w*|conscious\w*|alive|spontaneous|spontaneity|"
    r"level\s*[0-7]|evidence ladder|proof of (?:consciousness|sentience|life)|"
    r"what evidence do you have)\b",
    re.I,
)


_MEMORY_STORE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:remember|save|store|keep)\b.{0,50}\b(?:as|for)\b.{0,12}\b(?:a\s+)?(?:core|important)\s+memory\b",
        re.I,
    ),
    re.compile(
        r"\b(?:save|store|keep)\b.{0,30}\b(?:this|that)\b.{0,20}\bimportant\b",
        re.I,
    ),
    re.compile(
        r"\b(?:i would like you to|i want you to|can you|could you|please)\b.{0,12}\b(?:save|store|remember)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:save|store|remember)\s+that\b",
        re.I,
    ),
)


def _extract_explicit_web_search_request(user_message: str) -> dict[str, Any] | None:
    """Detect explicit command-form web search requests.

    This is a deterministic baseline for command-like phrasing such as:
      - "web search, jarvis pi project"
      - "jarvis web search current events"
      - "duckduckgo latest ai chip news"
    """
    normalized = user_message.strip()
    if not normalized:
        return None
    pattern = re.compile(
        r"^\s*(?:hey\s+)?(?:jarvis|oracle)?[,.\s!-]*"
        r"(?:web\s+search|duck\s*duck\s*go|duckduckgo)\b",
        re.IGNORECASE,
    )
    match = pattern.match(normalized)
    if not match:
        return None
    return {
        "tier": "explicit_web_command",
        "match": match.group(0).strip(),
    }


def _extract_capability_clarification(user_message: str) -> dict[str, Any] | None:
    """Detect explicit capability-contradiction clarification requests.

    Example:
      "You were just able to search the web, but previously you told me you
       couldn't. What is going on?"
    """
    lower = user_message.lower()
    has_search_domain = any(
        term in lower
        for term in ("web", "internet", "online", "duckduckgo", "search")
    )
    if not has_search_domain:
        return None

    has_prior_claim_ref = any(
        phrase in lower
        for phrase in ("you told me", "you said", "previously", "before")
    )
    if not has_prior_claim_ref:
        return None

    has_contrast = any(
        phrase in lower
        for phrase in (
            "but",
            "however",
            "just able",
            "right there",
            "what is going on",
            "what's going on",
            "why",
            "contradict",
        )
    )
    if not has_contrast:
        return None

    return {
        "intent": "capability_clarification",
        "domain": "web_search",
        "tier": "capability_clarification",
    }


def _extract_memory_action(user_message: str) -> dict[str, Any] | None:
    """Return explicit memory-store action metadata when the phrasing is unambiguous."""
    for pattern in _MEMORY_STORE_PATTERNS:
        match = pattern.search(user_message)
        if match:
            return {
                "action": "store",
                "memory_priority": "core",
                "match": match.group(0),
            }
    return None


def _extract_emergence_evidence_query(user_message: str) -> dict[str, Any] | None:
    """Route emergence/digital-life claims to grounded self-report surfaces."""
    if not _EMERGENCE_EVIDENCE_RE.search(user_message):
        return None
    return {
        "intent": "emergence_evidence",
        "tier": "emergence_evidence",
        "requires_grounded_answer": True,
    }


def _is_self_referential(lower: str) -> bool:
    """Detect if a query is about Jarvis's own state/capabilities/history.

    Uses a two-tier threshold:
    - Strong self-frame (e.g. "what was your...", "what did you..."): 1 hit suffices
    - Weak self-frame: requires 2-of-3 pattern groups

    Exclusion patterns always veto (user-directed requests).
    """
    if _SELF_REF_EXCLUDE.search(lower):
        return False
    hits = 0
    if _SELF_REF_VERBS.search(lower):
        hits += 1
    if _SELF_REF_NOUNS.search(lower):
        hits += 1
    if _SELF_REF_QUESTIONS.search(lower):
        hits += 1
    if hits >= 2:
        return True
    if hits >= 1 and _STRONG_SELF_FRAME.search(lower):
        return True
    return False


def _is_self_recent_research_history_query(lower: str) -> bool:
    """Detect self-history questions about Jarvis's recent research.

    This provides a narrow semantic floor so equivalent phrasing variants
    don't fall through to NONE.
    """
    if not lower:
        return False
    if _SELF_REF_EXCLUDE.search(lower):
        return False
    if not _MILD_SELF_REF.search(lower):
        return False
    if not _SELF_RESEARCH_HISTORY_TOPIC_RE.search(lower):
        return False
    if not _SELF_RESEARCH_HISTORY_TIME_RE.search(lower):
        return False
    return True


def _is_response_preference_instruction(lower: str) -> bool:
    """Detect user instructions about response formatting/style.

    This is an intent-class detector (preference update), not a domain-specific
    phrase matcher. Preference updates should route to NONE so the conversation
    layer can persist and acknowledge them deterministically.
    """
    if not lower:
        return False
    has_object = bool(_RESPONSE_PREFERENCE_OBJECT_RE.search(lower))
    has_meta = bool(_RESPONSE_PREFERENCE_META_RE.search(lower))
    has_directive = bool(_RESPONSE_PREFERENCE_DIRECTIVE_RE.search(lower))
    has_subject = bool(_RESPONSE_PREFERENCE_SUBJECT_RE.search(lower))

    # Explicit preference framing ("when I ask...", "by default...", etc.)
    # with either a response subject or formatting object is sufficient.
    if has_meta and (has_object or has_subject):
        return True

    # Imperative formatting directives must mention a concrete formatting object.
    if has_directive and has_object:
        return True

    return False


_GENERAL_KNOWLEDGE_RE = re.compile(
    r"(?:^|\b)(?:"
    r"who (?:wrote|is|was|invented|discovered|created|founded|directed|composed|painted|designed|built)\b"
    r"|what (?:is|are|was|were) (?:a |an |the )?"
    r"|what does .{1,30} mean"
    r"|when (?:did|was|were|is)\b"
    r"|where (?:is|are|was|were|did)\b"
    r"|how (?:many|much|old|long|far|tall|big)\b"
    r"|define \b"
    r"|tell me about\b"
    r"|what (?:year|country|city|language|color|colour)\b"
    r"|(?:name|list) (?:the|some|all)\b"
    r")",
    re.IGNORECASE,
)


_ADDRESS_PREFIX_RE = re.compile(
    r"^(?:hey\s+)?(?:jarvis|oracle)[,.\s!]*",
    re.IGNORECASE,
)


def is_general_knowledge(text: str) -> bool:
    """Detect factual world-knowledge questions (not about Jarvis's own state).

    These should use the main model, not the fast model, and should not be
    constrained by self-knowledge grounding directives.
    Strips wake-word address prefixes ("Hey Jarvis, ...") before checking
    self-referentiality so that natural voice queries aren't misclassified.
    """
    stripped = _ADDRESS_PREFIX_RE.sub("", text).strip()
    if not stripped:
        return False
    if _is_self_referential(stripped.lower()):
        return False
    if is_mildly_self_referential(stripped):
        return False
    return bool(_GENERAL_KNOWLEDGE_RE.search(stripped))


def is_mildly_self_referential(text: str) -> bool:
    """Check if text contains mild self-referential language (you/your/jarvis).

    Used by the conversation handler to decide whether to inject lightweight
    introspection context into NONE-routed queries as a safety net.
    Excludes pure conversational filler (thank you, greetings, acknowledgements).
    """
    lower = text.lower().strip()
    if _MILD_SELF_REF_EXCLUDE.match(lower):
        return False
    if not _MILD_SELF_REF.search(lower):
        return False
    if _SELF_REF_EXCLUDE.search(lower):
        return False
    return True


class ToolRouter:
    """Routes user queries to tools via keyword + semantic intent patterns.

    Tier 1: exact keyword match (fast, high confidence)
    Tier 1.5: disambiguate STATUS/SYSTEM_STATUS/INTROSPECTION
    Tier 2: regex intent patterns (broader, medium confidence)
    Tier 3: self-referential catch-all (strong self-frame = 1 hit, else 2-of-3)
    Fallback: ToolType.NONE (general LLM chat, may get introspection context)

    Bias: false positives into INTROSPECTION are safer than false negatives
    into NONE. Self-report fails closed.
    """

    @staticmethod
    def _finalize(
        user_message: str,
        result: RoutingResult,
        *,
        synthetic: bool = False,
        record_teacher: bool = True,
    ) -> RoutingResult:
        if result.tool == ToolType.LIBRARY_INGEST:
            url_match = re.search(r"https?://\S+", user_message)
            if url_match:
                result.extracted_args["url"] = url_match.group(0).rstrip(".,;:!?\"')")
        if result.tool == ToolType.INTROSPECTION:
            is_ref, reason = _detect_reflective(user_message.lower().strip())
            result.extracted_args["reflective"] = is_ref
            result.extracted_args["reflective_reason"] = reason
            if is_ref:
                logger.info("Reflective introspection detected: %s", reason)
        if record_teacher:
            try:
                from reasoning.intent_shadow import get_intent_shadow_runner

                runner = get_intent_shadow_runner()
                result = runner.observe(user_message, result)
            except Exception:
                logger.debug(
                    "intent_shadow observe failed (passing through)",
                    exc_info=True,
                )
            _record_voice_intent_teacher_signal(user_message, result, synthetic=synthetic)
        return result

    def route(
        self,
        user_message: str,
        *,
        synthetic: bool = False,
        golden_allow_bare_prefix: bool = False,
    ) -> RoutingResult:
        lower = user_message.lower().strip()

        golden_parse = parse_golden_command(
            user_message,
            allow_bare_prefix=golden_allow_bare_prefix,
        )
        if golden_parse is not None:
            golden_ctx = golden_parse.context
            golden_args: dict[str, Any] = {
                "tier": "golden",
                "golden_status": golden_ctx.golden_status,
                "golden_trace_id": golden_ctx.trace_id,
                "golden_command_id": golden_ctx.command_id,
                "golden_canonical_body": golden_ctx.canonical_body,
                "golden_authority_class": golden_ctx.authority_class,
                "golden_requires_confirmation": golden_ctx.requires_confirmation,
                "golden_allows_side_effects": golden_ctx.allows_side_effects,
                "golden_operation": golden_ctx.operation,
                "golden_context": golden_ctx.to_dict(),
            }
            if golden_parse.command is None:
                golden_args["golden_block_reason"] = golden_ctx.block_reason
                golden_args["accepted_commands"] = list_canonical_commands()
                return self._finalize(
                    user_message,
                    RoutingResult(
                        tool=ToolType.NONE,
                        confidence=1.0,
                        extracted_args=golden_args,
                        golden_context=golden_ctx,
                    ),
                    synthetic=synthetic,
                    record_teacher=False,
                )

            for key, value in golden_parse.command.default_args:
                golden_args[key] = value
            try:
                target_tool = ToolType(golden_parse.command.target_route)
            except ValueError:
                target_tool = ToolType.NONE
                golden_args["golden_block_reason"] = "invalid_target_route"
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=target_tool,
                    confidence=1.0,
                    extracted_args=golden_args,
                    golden_context=golden_ctx,
                ),
                synthetic=synthetic,
            )

        explicit_memory_action = _extract_memory_action(user_message)
        if explicit_memory_action:
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=ToolType.MEMORY,
                    confidence=0.92,
                    extracted_args=explicit_memory_action,
                ),
                synthetic=synthetic,
            )

        explicit_web_search = _extract_explicit_web_search_request(user_message)
        if explicit_web_search:
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=ToolType.WEB_SEARCH,
                    confidence=0.94,
                    extracted_args=explicit_web_search,
                ),
                synthetic=synthetic,
            )

        capability_clarification = _extract_capability_clarification(user_message)
        if capability_clarification:
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=ToolType.INTROSPECTION,
                    confidence=0.94,
                    extracted_args=capability_clarification,
                ),
                synthetic=synthetic,
            )

        emergence_evidence = _extract_emergence_evidence_query(user_message)
        if emergence_evidence:
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=ToolType.INTROSPECTION,
                    confidence=0.94,
                    extracted_args=emergence_evidence,
                ),
                synthetic=synthetic,
            )

        if _is_response_preference_instruction(lower):
            logger.info("Tier 0 preference-instruction catch: NONE for: %s", lower[:60])
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=ToolType.NONE,
                    confidence=0.86,
                    extracted_args={"tier": "preference_instruction"},
                ),
                synthetic=synthetic,
            )

        # Tier 0: core intent verbs (learn X, train X, teach yourself X)
        core = _match_core_intent(lower)
        if core is not None:
            resolved = _disambiguate(lower, core.tool)
            core = RoutingResult(tool=resolved, confidence=core.confidence,
                                 extracted_args=core.extracted_args)
            return self._finalize(user_message, core, synthetic=synthetic)

        # Tier 0.5: persistent user routing corrections
        correction = _match_user_corrections(lower)
        if correction is not None:
            return self._finalize(user_message, correction, synthetic=synthetic)

        # Tier 0.75: active plugin matching
        try:
            from tools.plugin_registry import get_plugin_registry
            _plugin_reg = get_plugin_registry()
            _plugin_match = _plugin_reg.match(user_message)
            if _plugin_match:
                return self._finalize(
                    user_message,
                    RoutingResult(
                        tool=ToolType.PLUGIN,
                        confidence=0.88,
                        extracted_args={"plugin_name": _plugin_match, "tier": "plugin"},
                    ),
                    synthetic=synthetic,
                )
        except Exception:
            pass

        # Tier 1: keyword match
        for keywords, tool in _KEYWORD_PATTERNS:
            for kw in keywords:
                if kw in lower:
                    resolved = _disambiguate(lower, tool)
                    return self._finalize(
                        user_message,
                        RoutingResult(tool=resolved, confidence=0.85, extracted_args={"keyword": kw}),
                        synthetic=synthetic,
                    )

        # Tier 2: semantic intent patterns
        best_match: RoutingResult | None = None
        best_conf = 0.0
        for pattern, tool, base_conf in _INTENT_PATTERNS:
            m = pattern.search(user_message)
            if m:
                conf = base_conf
                if conf > best_conf:
                    best_conf = conf
                    best_match = RoutingResult(tool=tool, confidence=conf,
                                               extracted_args={"pattern": pattern.pattern, "match": m.group(0)})

        if best_match and best_conf >= 0.7:
            resolved = _disambiguate(lower, best_match.tool)
            best_match = RoutingResult(tool=resolved, confidence=best_match.confidence,
                                       extracted_args=best_match.extracted_args)
            logger.debug("Intent route: %s (%.0f%%) for: %s", best_match.tool.value, best_conf * 100, lower[:60])
            return self._finalize(user_message, best_match, synthetic=synthetic)

        # Tier 3: catch-all for self-referential queries.
        # Strong self-frame ("what was your...", "what did you...") needs only
        # 1 pattern hit. Weaker frames need 2-of-3. This biases toward
        # tool-grounded introspection over free-generated autobiography.
        if _is_self_recent_research_history_query(lower):
            logger.info("Tier 2.5 self research-history catch: INTROSPECTION for: %s", lower[:60])
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=ToolType.INTROSPECTION,
                    confidence=0.82,
                    extracted_args={"tier": "self_research_history_catch"},
                ),
                synthetic=synthetic,
            )

        if _is_self_referential(lower):
            logger.info("Tier 3 self-ref catch: INTROSPECTION for: %s", lower[:60])
            return self._finalize(
                user_message,
                RoutingResult(
                    tool=ToolType.INTROSPECTION,
                    confidence=0.70,
                    extracted_args={"tier": "self_ref_catch"},
                ),
                synthetic=synthetic,
            )

        return self._finalize(user_message, RoutingResult(tool=ToolType.NONE, confidence=1.0, extracted_args={}), synthetic=synthetic)


tool_router = ToolRouter()
