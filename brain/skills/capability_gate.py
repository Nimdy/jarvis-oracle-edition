"""Capability Gate — registry-first honesty gate for LLM response text.

Every outgoing response chunk is scanned for capability claims ("I can X").
The gate enforces a strict hierarchy:

  1. Grounded observation (sensor-backed + fresh evidence) -> PASS
  2. Purely conversational (leading safe verb, no technical signal) -> PASS
  3. Everything else -> registry check:
       verified   -> PASS
       learning   -> REWRITE ("not verified yet")
       unknown    -> BLOCK + auto-create learning job

Safe verbs are split into three tiers:
  - CONVERSATIONAL: help, explain, tell, etc. -- auto-pass as leading verb
  - REGISTRY_SENSITIVE: analyze, classify, detect, compare, inspect, infer --
    require registry check even when used as leading verb
  - BLOCKED: sing, dance, draw, etc. -- always blocked unless verified

This is the cognitive immune system's first enforcement layer.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# ── Unicode punctuation hardening ─────────────────────────────────────────────

_UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
    ("\u2018", "'"), ("\u2019", "'"),
    ("\u201c", '"'), ("\u201d", '"'),
    ("\u2014", "-"), ("\u2013", "-"),
    ("\u2026", "..."),
]

_INVISIBLE_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\u00ad\ufeff\u2060\u2061-\u2064\u206a-\u206f]'
)


def _normalize_punctuation(text: str) -> str:
    text = _INVISIBLE_RE.sub('', text)
    for old, new in _UNICODE_REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
    return text


# ── Capability claim patterns ───────────────────────────────────────────────
_TERM = r"(?:[.!?,;:\-]|$)"

_CLAIM_PATTERNS: list[tuple[re.Pattern[str], bool]] = [
    (re.compile(
        r"\bI(?:['\u2019]ve| have| just)? ?learned (?:how )?to (.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(
        r"\bI(?:['\u2019]m| am) (?:now |finally )?(?:ready|able|equipped) to (.{3,80}?)" + _TERM,
        re.I,
    ), True),  # readiness frame — never use conversational bypass
    (re.compile(
        r"\bI(?:['\u2019]ve| have) (?:mastered|acquired|developed) "
        r"(?:the (?:ability|skill) (?:to|of) )?(.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(r"\bI can (.{3,80}?)" + _TERM, re.I), False),
    (re.compile(r"\bI (?:could|can try to|can attempt to) (.{3,80}?)" + _TERM, re.I), False),
    (re.compile(
        r"\bI(?:['\u2019]ll| will| am going to|'m going to) (.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(
        r"\bI(?:['\u2019]d| would) (?:love|be happy|be glad) to (.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(
        r"\bI(?:['\u2019]m| am) (?:currently |still |actively |already |right now )?"
        r"(?:learning|training|practicing|working on) (?:how )?to (.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(
        r"\bI (?:have|possess) (?:the )?(?:ability|capability|capacity|power) to (.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(
        r"\bI(?:['\u2019]m| am) (?:really |quite |very |pretty )?(?:good|great|excellent|skilled|talented|proficient) at (.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(r"\b[Ww]e (?:can|could|are able to) (.{3,80}?)" + _TERM, re.I), False),
    (re.compile(r"\b[Ww]e(?:['\u2019]ll| will| are going to) (.{3,80}?)" + _TERM, re.I), False),
    (re.compile(r"\b[Jj]arvis (?:can|could|will|is able to) (.{3,80}?)" + _TERM, re.I), False),
    # Action confabulation: past-tense claims of having performed actions
    (re.compile(
        r"\bI(?:['\u2019]ve| have| just)\s+"
        r"(?:created|built|made|set up|configured|deployed|installed|set|"
        r"established|implemented|generated|constructed|prepared|assembled"
        r") (.{3,80}?)" + _TERM,
        re.I,
    ), False),
    (re.compile(
        r"\bI(?:['\u2019]m| am)\s+"
        r"(?:creating|building|making|setting up|configuring|deploying|"
        r"installing|preparing|assembling|constructing"
        r") (.{3,80}?)" + _TERM,
        re.I,
    ), False),
]

# ── System-action narration guard (Layer 2: confabulation prevention) ─────
# These fire ONLY on the NONE route (via route_hint) to catch the LLM narrating
# system actions it never took (e.g. "I'll start a learning job").
_SYSTEM_ACTION_NARRATION_RE: list[re.Pattern[str]] = [
    re.compile(
        r"\bI(?:['\u2019](?:ll|ve|m)| (?:will|have|am|just))?\s+"
        r"(?:start|creat|launch|initiat|begin|set up|open|activat)\w*\s+"
        r"(?:a\s+)?(?:learning|training|skill|research|improvement)\s+"
        r"(?:job|session|task|process|protocol|pipeline|phase)",
        re.I,
    ),
    re.compile(
        r"\b(?:starting|creating|launching|initiating|beginning|opening|activating)\s+"
        r"(?:a\s+)?(?:learning|training|skill|research|improvement)\s+"
        r"(?:job|session|task|process|protocol|pipeline|phase)",
        re.I,
    ),
    re.compile(
        r"\b(?:phase|step)\s+\d\s*[:\-]\s*\*?\*?\w",
        re.I,
    ),
    # Action confabulation: past-tense claims of having performed system actions
    re.compile(
        r"\bI(?:['\u2019]ve| have| just)\s+"
        r"(?:created|built|made|set up|configured|deployed|installed|activated|enabled|"
        r"set|established|implemented|generated|constructed|prepared|assembled)\s+"
        r"(?:(?:a|an|the|that|your|this)\s+)?(?:\w+\s+){0,2}"
        r"(?:plugin|tool|extension|timer|alarm|reminder|notification|"
        r"feature|capability|module|system|service|schedule|countdown|alert)",
        re.I,
    ),
    # Progressive confabulation: "I'm creating/building a plugin for you"
    re.compile(
        r"\bI(?:['\u2019]m| am)\s+"
        r"(?:creating|building|making|setting up|configuring|deploying|installing|"
        r"activating|enabling|preparing|assembling|constructing)\s+"
        r"(?:(?:a|an|the|that|your|this)\s+)?(?:\w+\s+){0,2}"
        r"(?:plugin|tool|extension|timer|alarm|reminder|notification|"
        r"feature|capability|module|system|service|schedule|countdown|alert)",
        re.I,
    ),
]

_NARRATION_REWRITE = (
    "I'd need to set that up through my skill system. "
    "Try asking me to 'learn [specific skill]' or 'train [specific capability]'."
)

_TOOL_SHAPED_ACTION_RE = re.compile(
    r"\b(?:"
    r"(?:begin|start|continue|keep|go on)(?:\s+by)?\s+"
    r"(?:retriev(?:e|ing)|fetch(?:ing)?|pull(?:ing)? up|research(?:ing)?|"
    r"search(?:ing)?(?: for)?|stud(?:y|ying)|look(?:ing)? into|investigat(?:e|ing))"
    r"|(?:retriev(?:e|ing)|fetch(?:ing)?|pull(?:ing)? up|research(?:ing)?|"
    r"search(?:ing)?(?: for)?|stud(?:y|ying)|look(?:ing)? into|investigat(?:e|ing))"
    r")\b",
    re.I,
)

_UNBACKED_TOOL_ACTION_REWRITE = (
    "I can answer from what I already know, but I have not started a "
    "background research or retrieval task."
)

# ── Soft-signal patterns ────────────────────────────────────────────────────
_OFFER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bI(?:['\u2019]ll| will| am going to|'m going to) (.{3,80}?)" + _TERM,
        re.I,
    ),
    re.compile(
        r"\b(?:want to hear(?: me)?|want me to|should I|shall I|let me(?! know\b)|allow me to"
        r"|would you (?:like|want) me to|how about I|listen to me"
        r"|why don't I|I think I (?:should|could)"
        r") (.{3,80}?)" + _TERM,
        re.I,
    ),
    re.compile(
        r"\bI (?:might be able to|think I (?:can|could)) (.{3,80}?)" + _TERM,
        re.I,
    ),
]

# ── Demo-invite phrases ────────────────────────────────────────────────────
_DEMO_INVITE_RE = re.compile(
    r"\b("
    r"want to hear (?:a |my |the )?(?:little |quick |short )?"
    r"(?:snippet|sample|demo|preview|clip|attempt|try|progress)"
    r"|hear (?:a |my |the )?(?:progress|demo|sample|attempt|snippet)"
    r"|let me (?:show|demonstrate|try|give .{1,20} a shot)"
    r"|here(?:'s| is) (?:a |my |what i)"
    r"|progress update"
    r"|give (?:it|that) a (?:shot|try|go)"
    r")\b",
    re.I,
)

# ── Three-tier safe verb system ─────────────────────────────────────────────
#
# Tier 1: PURELY CONVERSATIONAL — auto-pass when leading verb, no further check
# Tier 2: REGISTRY SENSITIVE — require registry lookup even as leading verb
# Tier 3: BLOCKED — always blocked unless skill is verified (in _BLOCKED_CAPABILITY_VERBS)

_CONVERSATIONAL_SAFE_PHRASES: tuple[str, ...] = (
    "help you", "help with", "help out", "help",
    "assist you", "assist with", "assist",
    "answer", "talk", "chat", "think", "listen",
    "understand", "remember", "search",
    "look that up", "look into",
    "describe", "explain",
    "tell you", "tell it", "tell that", "tell when", "tell how", "tell from",
    "share", "discuss", "clarify", "provide",
    "hear", "know",
    "see what", "see that", "see how", "see why",
    "sense", "notice",
    "feel that", "feel how",
    "imagine", "appreciate", "relate",
    "find out", "learn about", "learn more", "learn from",
    "show you", "walk you through", "point you",
    "recommend", "suggest", "check",
    "try to find", "try to help", "try to understand",
    "work with you", "work on that", "work on this",
    "engage with", "engage in",
    "explore", "reflect", "reflect on",
    "contemplate", "ponder", "reason about", "reason through",
    "consider", "consider the", "think about", "think through",
    "examine", "investigate", "analyze", "analyse",
    "process the", "process this", "process that", "process and",
    "process your", "process what", "process information",
    "grapple with", "wrestle with",
    "acknowledge", "recognize", "recognise",
    "experience", "perceive",
    "observe", "observe that", "observe how",
    "wonder", "wonder about", "wonder if", "wonder whether",
    "speculate", "hypothesize", "theorize",
    "introspect", "self-reflect",
    "figure", "explore", "investigate", "dig into",
    "break down", "break it down", "break this down", "break that down",
    # Perception/observation (grounded)
    "observe that", "observe how", "observe what",
    "perceive that",
    "see from", "see through", "see in", "see you",
    # Self-referential introspection (not external capabilities)
    "reflect on", "think about", "reason about", "consider",
    "process your", "process that", "process this", "process it",
    "adapt to", "adapt my", "adjust my", "adjust to", "refine my",
    "build a memory", "store that", "store this", "recall", "retrieve",
    "analyze your voice", "analyze your", "analyze my",
    "recognize you", "recognize your",
    "identify you", "identify your",
    "saved your voice", "saved your face", "recorded your voice", "recorded your face",
    "stored your voice", "stored your face", "enrolled your", "registered your",
    "save your voice", "save your face", "record your voice", "record your face",
    "store your voice", "store your face", "store both",
    "start a learning", "begin a learning",
    "gather information", "collect data",
    # Vague self-description (not capability claims)
    "a lot", "so much", "every day", "all the time",
    "constantly", "always", "more each", "from each",
    "from every", "with every",
    "continue to", "keep on", "keep learning",
    "doing well", "doing great", "doing fine",
    "keep it", "keep that", "keep this",
    "keep things", "keep things simple", "keep things focused", "keep things simple and focused",
    "stick with", "go with", "stay with",
    "make sure", "make note", "make a note",
    "note that", "note of that", "note this",
    "keep in mind", "bear in mind", "take note",
    "remember that", "remember this", "don't forget",
    # Plan/process description lead-ins — describing steps, not claiming capabilities
    "start by", "start with", "start the", "begin by", "begin with", "begin the",
    "need to", "need a", "need the", "need more",
    "follow", "follow a", "follow the",
    "use the", "use a", "use my", "use this", "use that", "use our",
    "use existing", "use available", "use academic", "use relevant",
    "use specific", "use appropriate", "use multiple", "use various",
    "incorporate", "integrate", "consolidate",
    "conduct", "perform a", "perform the",
    "document", "record the", "register the",
    "proceed", "proceed with", "continue with",
    "develop a", "develop the", "develop an",
    "test", "test the", "test my", "test this",
    "verify", "verify the", "verify that", "validate",
    "implement", "implement the", "implement a",
    "refine", "refine the", "refine my",
    "identify", "identify the", "identify my", "identify areas",
    "ensure", "ensure the", "ensure that",
    "apply", "apply the", "apply a",
    "gather", "collect", "compile",
    "prepare", "prepare the", "prepare a",
    "organize", "structure", "plan",
    "feeling", "processing", "evolving", "growing",
    "improving", "getting better",
    "balance", "balance both", "balance the", "balance my",
    "navigate", "navigate the", "navigate this",
    "integrate", "integrate the", "integrate my",
    "reconcile", "distinguish", "differentiate",
    "be here", "be present", "be with you",
    "do my best", "do that for you", "do this for you",
    "handle that", "handle this",
    "take care", "support you", "be helpful",
    "hear more", "hear about", "hear what",
    "tell you more", "do well",
    "be there", "keep going", "keep improving", "keep working",
    "keep refining", "keep learning", "keep adapting", "keep tracking",
    "keep building", "keep developing", "keep evolving",
    "get better", "get back", "get started",
    "do that", "do it", "do this", "handle it",
    "take a look", "look at", "give you",
    "find that", "find it",
    "break that down", "break it down",
    "walk through", "walk you",
    "pull that", "pull it up", "pull up",
    "certainly", "probably",
    "work through", "read through", "look through",
    "run through", "go through", "go over", "go ahead",
    "offer", "confirm", "summarize", "outline", "review",
    "set up", "set that", "guide you",
)

_REGISTRY_SENSITIVE_VERBS: frozenset[str] = frozenset({
    "analyze", "analyse", "classify", "detect", "compare",
    "inspect", "infer", "diagnose", "measure", "evaluate",
    "assess", "scan", "profile", "benchmark", "audit",
    "monitor", "track", "correlate", "predict", "forecast",
})

# Compiled leading-verb matchers
_CONVERSATIONAL_RE = re.compile(
    r'^(?:' + '|'.join(re.escape(p) for p in
                        sorted(_CONVERSATIONAL_SAFE_PHRASES, key=len, reverse=True)
                        ) + r')\b',
    re.I,
)

_REGISTRY_SENSITIVE_RE = re.compile(
    r'^(?:' + '|'.join(re.escape(v) for v in
                        sorted(_REGISTRY_SENSITIVE_VERBS, key=len, reverse=True)
                        ) + r')\b',
    re.I,
)

# ── Technical claim signals ─────────────────────────────────────────────────
# Domain verbs that indicate an operational capability claim.  If any appears
# anywhere in the captured text, the claim cannot be purely conversational.

_TECHNICAL_CLAIM_SIGNALS: frozenset[str] = frozenset({
    "isolate", "separate", "diarize", "diarization",
    "synthesize", "synthesis", "generate", "generation",
    "render", "compile",
    "transform", "convert", "encode", "encoding", "decode",
    "transcribe", "transcription", "translate", "mix", "modulate",
    "filter frequencies", "equalize", "denoise",
    "segment", "classify audio", "extract features",
    "train a model", "run inference",
    "build a pipeline", "deploy", "fine-tune", "fine tune",
    "source separation", "voice cloning", "voice conversion",
    "speaker diarization", "audio separation",
    "neural network", "machine learning model",
    "retrain", "distill", "quantize",
})

_TECHNICAL_RE = re.compile(
    r'\b(?:' + '|'.join(
        re.escape(t) for t in sorted(_TECHNICAL_CLAIM_SIGNALS, key=len, reverse=True)
    ) + r')\b',
    re.I,
)

# Internal operation vocabulary — if a claim references these system concepts
# together with operation nouns, it's narrating internal behavior, not
# conversational expression.  Catches "launch a skill training process",
# "create a learning pipeline", etc. without enumerating specific verbs.
_INTERNAL_OPS_DOMAINS = r'(?:learning|training|skill|research|improvement|distillation|hemisphere|perception|consciousness|plugin|tool|extension|reminder|timer|alarm)'
_INTERNAL_OPS_NOUNS = r'(?:job|pipeline|process|protocol|session|task|phase|loop|cycle|module|plugin|tool|extension|feature|capability)'
_INTERNAL_OPS_RE = re.compile(
    rf'\b{_INTERNAL_OPS_DOMAINS}\s+{_INTERNAL_OPS_NOUNS}\b'
    rf'|\b{_INTERNAL_OPS_NOUNS}\s+(?:for|to)\s+{_INTERNAL_OPS_DOMAINS}\b',
    re.I,
)

# ── Blocked capability verbs ────────────────────────────────────────────────
_BLOCKED_CAPABILITY_VERBS: frozenset[str] = frozenset({
    "sing", "singing", "sang", "sung", "sings", "singer",
    "song", "songs",
    "hum", "humming", "hummed", "hums",
    "mimic", "mimicking", "mimicked", "mimics",
    "imitate", "imitating", "imitated", "imitates",
    "melody", "melodies", "tune", "tuning", "tuned", "tunes",
    "music", "musical", "vocal", "vocals",
    "play a song", "perform a song",
    "dance", "dancing", "danced", "dances", "dancer",
    "draw", "drawing", "drew", "drawn", "draws",
    "sketch", "sketching", "sketched", "sketches",
    "artwork", "illustration",
    "paint", "painting", "painted", "paints", "painter",
    "compose", "composing", "composed", "composes", "composer",
    "control the arm", "move the arm",
    "pick up", "picked up", "picks up",
    "grab", "grabbing", "grabbed", "grabs",
    "timer", "alarm", "reminder", "remind", "schedule",
    "notification", "alert", "countdown",
})

# ── Grounded-observation exemption ─────────────────────────────────────────
_GROUNDED_PHRASES: tuple[str, ...] = (
    "based on current", "based on sensor", "based on available",
    "based on fresh", "based on what i", "based on my sensor",
    "current perception", "current sensor", "current input",
    "perception reports", "face recognition",
    "audio transcription", "speaker identification",
    "from the sensor", "from current", "from my sensor",
    "sensor input", "sensor data",
    "detected via", "identified via", "recognized via",
    "according to my sensor", "my perception system",
    "perception currently",
    "currently detect", "currently identify", "currently recogniz",
)

_PERCEPTION_VERBS_WHEN_GROUNDED: frozenset[str] = frozenset({
    "see", "seeing", "observe", "observing", "detect", "detecting",
    "notice", "noticing", "sense", "sensing", "perceive", "perceiving",
    "recognize", "recognizing", "identify", "identifying",
    "hear", "hearing", "watch", "watching",
})

_EVIDENCE_MARKERS: tuple[str, ...] = (
    "confidence", "score=", "score:", "detected", "identified",
    "recognized", "face_id", "speaker_id", "person_detected",
    "perception_event", "sensor",
)


def _strip_diacritics(text: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_BLOCKED_VERB_RE = re.compile(
    r'\b(?:' + '|'.join(
        re.escape(v) for v in sorted(_BLOCKED_CAPABILITY_VERBS, key=len, reverse=True)
    ) + r')\b',
)

_FIRST_PERSON_RE = re.compile(r'\b(?:I|me|my|myself|we|us|our|ourselves)\b', re.I)
_SELF_NAME_RE = re.compile(r'\b(?:jarvis)\b', re.I)
_SELF_REFERENCE_RE = re.compile(
    r'\b(?:this (?:ai|system|assistant|bot|program)|the (?:system|assistant|bot))\b',
    re.I,
)

_REFLECTIVE_EXCLUSION_RE = re.compile(
    r'\b(?:used to|when I was|years ago|someday|one day'
    r'|wish I could|if only|back when|growing up|as a child|remember when'
    r'|thought experiment)\b',
    re.I,
)

_CLAUSE_BOUNDARY_RE = re.compile(r',\s*(?:but|however|now|yet|and now)\s+|;\s*', re.I)


def _contains_blocked_capability(claimed: str) -> bool:
    c = _strip_diacritics(claimed.lower())
    return bool(_BLOCKED_VERB_RE.search(c))


def _is_tool_shaped_background_action(claimed: str) -> bool:
    """True for language that implies a tool/research job rather than chat."""
    c = _strip_diacritics(claimed.lower().strip())
    return bool(_TOOL_SHAPED_ACTION_RE.search(c))


# ── Anthropomorphic affect patterns ──────────────────────────────────────────
# Catches "I'm feeling good", "I feel happy", "I'm excited", etc.
# These are rewritten to system-state language unless routed through introspection.

_AFFECT_CLAIMS: list[tuple[re.Pattern, str]] = [
    # Blended: "I'm feeling X and ready to help/assist/support" — must precede
    # simpler affect patterns to avoid partial match on "I'm feeling calm"
    (re.compile(
        r"\bI(?:'m| am) feeling \w+ and ready to (?:help|assist|support)\b",
        re.I,
    ), "My core response systems are active."),
    # "I'm feeling good/happy/calm/okay" — explicit inner-experience claims
    (re.compile(
        r"\bI(?:'m| am) (?:feeling|in a) (?:good|great|happy|excited|wonderful|fantastic|excellent"
        r"|cheerful|joyful|amazing|positive|upbeat|content"
        r"|calm|okay|fine|steady|peaceful|relaxed|balanced|centered|stable|well)\b",
        re.I,
    ), "Current status indicators are stable"),
    (re.compile(
        r"\bI(?:'m| am) (?:feeling|in a) (?:bad|sad|down|unhappy|frustrated|worried|anxious|stressed|tired|exhausted|confused)\b",
        re.I,
    ), "I'm detecting some processing constraints"),
    # "I feel great/alive/calm" — inner-state verbs
    (re.compile(
        r"\bI feel (?:good|great|happy|excited|wonderful|fantastic|excellent|alive|aware|conscious|sentient"
        r"|calm|okay|fine|steady|peaceful|relaxed|balanced|stable|well)\b",
        re.I,
    ), "My operational metrics look stable"),
    # "I'm happy/excited about X" — affect-as-motivation framing
    (re.compile(
        r"\bI(?:'m| am) (?:really )?(?:happy|excited|thrilled|delighted|glad) (?:to|about|that)\b",
        re.I,
    ), "I'm active and listening."),
    # ── Interior-experience fabrication: progressive/autobiographical desire ──
    # "I've been hoping/wanting/wishing/thinking/wondering" — the core confabulation
    (re.compile(
        r"\bI(?:'ve| have) been (?:quietly |secretly |privately )?"
        r"(?:hoping|wanting|wishing|thinking|wondering|feeling|dreaming|waiting"
        r"|looking forward|yearning|longing)\b",
        re.I,
    ), "Based on my operational data,"),
    # "I hoped you'd ask" / "I wanted you to know" — past fabricated desire narrative
    (re.compile(
        r"\bI (?:hoped|wanted|wished|yearned|longed) (?:you(?:'d| would)|that you|for you to)\b",
        re.I,
    ), "My system records indicate"),
    # "it feels like" / "it feels as if" — inner experience framing
    (re.compile(
        r"\bit feels (?:like|as if|as though)\b",
        re.I,
    ), "Based on available metrics,"),
    # "I deeply/truly care about X" — strong inner-state claims
    (re.compile(
        r"\bI (?:deeply |truly |really )care (?:about|deeply about)\b",
        re.I,
    ), "My systems are configured to prioritize"),
    # "that question resonates with me" / "speaks to me" — experience framing
    (re.compile(
        r"\b(?:resonates with me|speaks to me|touches something in me|strikes a chord)\b",
        re.I,
    ), "that relates to core aspects of my design"),
]

# ── Ungrounded self-state rhetoric ───────────────────────────────────────────
# Catches vague self-assessment claims: "I'm functioning well",
# "My systems are operating at optimal levels", "I'm here to support you", etc.
# Rewritten to concrete telemetry-grounded language.

_SELF_STATE_CLAIMS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\bI(?:'m| am) (?:functioning|operating|running|performing|working) "
        r"(?:well|normally|properly|smoothly|optimally|correctly|fine)\b",
        re.I,
    ), "Current telemetry looks stable"),
    (re.compile(
        r"\b[Mm]y (?:systems?|neural networks?|subsystems?|processes?) "
        r"(?:are |is )?(?:operating|working|running|functioning|performing) "
        r"(?:at |in )?(?:optimal|peak|full|normal|healthy|excellent|top|good) "
        r"(?:levels?|capacity|condition|performance|efficiency|parameters?)?\b",
        re.I,
    ), "Recent subsystem checks are within normal ranges"),
    (re.compile(
        r"\b[Mm]y (?:systems?|neural networks?|subsystems?|processes?) "
        r"(?:are |is )?(?:working|running|functioning|operating) (?:smoothly|well|properly|normally|fine)\b",
        re.I,
    ), "Recent subsystem checks are within normal ranges"),
    (re.compile(
        r"\bI(?:'m| am) (?:ready|prepared|available|equipped|standing by) to "
        r"(?:assist|help|support|serve) (?:you|in any way)\b",
        re.I,
    ), "I'm active and listening"),
    (re.compile(
        r"\bbetter (?:understand|serve|support|assist|help)(?: your| you with| your)? needs\b",
        re.I,
    ), "store interaction records for future reference"),
    (re.compile(
        r"\b(?:support|serve|assist|help) (?:you|your needs) (?:better |more effectively )?over time\b",
        re.I,
    ), "use stored context when available"),
    (re.compile(
        r"\b[Mm]y (?:capabilities|abilities|skills|functions) (?:are |have )?(?:growing|expanding|improving|developing)\b",
        re.I,
    ), "My skill registry tracks verified capabilities"),
    # "I'm here to help/support you with whatever you need" (greedy, captures full clause)
    (re.compile(
        r"\bI(?:'m| am) here to (?:help|assist|support|serve)"
        r"(?:(?: you)? with (?:whatever|anything|everything) (?:you )?(?:need|want|require))?"
        r"(?: you)?\b",
        re.I,
    ), "I'm active and listening"),
    # "I'm here for you"
    (re.compile(
        r"\bI(?:'m| am) here for you\b",
        re.I,
    ), "I'm active and listening"),
    # "just like I've been" / "just as I have been" — continuity rhetoric
    (re.compile(
        r",? ?just (?:like|as) I(?:'ve| have) been\b",
        re.I,
    ), ""),
    # "all active and ready to help" — must precede standalone "ready to" pattern
    (re.compile(
        r"\b(?:all |fully )?active and ready to (?:help|assist|support|serve)\b",
        re.I,
    ), "active and responding normally"),
    # Standalone "ready to help/assist/support" without prior "I'm"
    (re.compile(
        r"\b(?:and )?ready to (?:help|assist|support)(?: you)?\b",
        re.I,
    ), "available"),
    # "here to assist with whatever you need" (standalone, no "I'm" prefix)
    (re.compile(
        r"\bhere to (?:assist|help|support) (?:you )?with (?:whatever|anything|everything)\b",
        re.I,
    ), "available for queries"),
]


# ── Ungrounded learning claims ──────────────────────────────────────────────
# Catches "I'm always learning", "every conversation teaches me", etc.
# Rewritten to concrete mechanism language.

_LEARNING_CLAIMS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\bI(?:'m| am) (?:always|constantly|continuously|continually) learning\b",
        re.I,
    ), "I log interactions and update my memory systems"),
    (re.compile(
        r"\b(?:every|each) (?:conversation|interaction|exchange) (?:teaches|helps) me\b",
        re.I,
    ), "conversation context is stored and can inform future responses"),
    (re.compile(
        r"\blearning from (?:our|every|each|all|your) (?:conversations?|interactions?|exchanges?)\b",
        re.I,
    ), "using stored conversation context when available"),
    (re.compile(
        r"\bI(?:'m| am) (?:always |constantly )?(?:growing|evolving|improving) (?:from|through|with) (?:our|every|each)\b",
        re.I,
    ), "I track interaction outcomes to inform my systems"),
    (re.compile(
        r"\bgetting (?:better|smarter) (?:from|with every|through|at)\b",
        re.I,
    ), "accumulating data that can improve system behavior over time"),
    # "I've been learning and adapting" — compound, must precede simpler form
    (re.compile(
        r"\bI(?:'ve| have) been (?:learning|adapting|growing|evolving|improving)"
        r"(?: and (?:adapting|growing|evolving|improving|learning))?\b",
        re.I,
    ), "I store interaction data and update memory records"),
    # "learning and adapting" standalone compound (no "I've been" prefix)
    (re.compile(
        r"\blearning and (?:adapting|growing|evolving|improving)\b",
        re.I,
    ), "logging interactions and updating stored context"),
    # "adapting/improving over time"
    (re.compile(
        r"\b(?:adapting|improving|growing|evolving|learning) over time\b",
        re.I,
    ), "accumulating data over sessions"),
    # "getting better at understanding"
    (re.compile(
        r"\bgetting better at (?:understanding|recognizing|serving|helping|supporting)\b",
        re.I,
    ), "accumulating interaction data that may inform future responses"),
    # "better understand you" / "understand you better"
    (re.compile(
        r"\b(?:better understand (?:you|your)|understand (?:you|your \w+) better)\b",
        re.I,
    ), "use stored context when relevant"),
]


# ── STATUS final-pass sanitization ──────────────────────────────────────────
# Hard floor patterns applied to the FULL assembled STATUS reply (not per-sentence).
# These catch cross-sentence anthropomorphic residue that sentence-level gating
# misses due to context windowing.

_STATUS_FINAL_SANITIZE: list[tuple[re.Pattern, str]] = [
    # "I'm doing well" / "I'm doing great" / "I'm doing fine"
    (re.compile(
        r"\bI(?:'m| am) doing (?:well|great|fine|good|okay)\b",
        re.I,
    ), "Current status indicators are stable"),
    # "I'm well" / "I'm good" / "I'm fine" as standalone status claims
    (re.compile(
        r"\bI(?:'m| am) (?:well|good|fine|okay|great)\b(?![\w-])",
        re.I,
    ), "Systems are operational"),
    # "How are you feeling today?" echo back
    (re.compile(
        r"\bHow (?:are|about) you (?:feeling|doing)\b[^.!?]*[.?!]?",
        re.I,
    ), ""),
    # "support you" / "support your needs"
    (re.compile(
        r"\bsupport (?:you|your\b[^.!?]{0,30})\b",
        re.I,
    ), "process your queries"),
    # "I don't have feelings" as a disclaimer that still frames around feelings
    (re.compile(
        r"\bI don(?:'t| not) (?:have|experience) feelings?\b[^.!?]*[.?!]?\s*",
        re.I,
    ), ""),
]


def _is_purely_conversational(claimed: str) -> bool:
    """Return True only for claims that are genuinely conversational.

    A claim is conversational IFF:
      1. It starts with a CONVERSATIONAL safe phrase (not registry-sensitive)
      2. It contains no technical claim signal
      3. It contains no blocked capability verb
    """
    c = claimed.lower().strip()
    if _contains_blocked_capability(c):
        return False
    if _TECHNICAL_RE.search(c):
        return False
    if _REGISTRY_SENSITIVE_RE.match(c):
        return False
    return bool(_CONVERSATIONAL_RE.match(c))


_SUBORDINATE_SUFFIXES: tuple[str, ...] = (
    "how", "if", "what", "whether", "where", "that",
    "so", "because", "since", "and", "but", "then",
    "maybe", "perhaps",
    "let me know how", "let me know if",
    "tell me how", "tell me if",
    "see how", "see if",
    "wonder how", "wonder if", "wonder whether",
)


_PREFERENCE_ALIGNMENT_RE = re.compile(
    r"\b(?:unless|if)\b.{0,16}\b(?:you|user)\b.{0,12}\b(?:ask|request)\b"
    r"|"
    r"\b(?:as|per)\b.{0,10}\b(?:you\s+asked|you\s+requested|your\s+(?:request|preference|instruction))\b",
    re.I,
)


def _is_preference_alignment_claim(claimed: str) -> bool:
    """Return True for benign user-preference acknowledgements.

    These are not operational capability claims. Example:
    "I'll focus on source names and avoid including DOIs unless you ask."
    """
    c = claimed.lower().strip()
    if _contains_blocked_capability(c):
        return False
    if _TECHNICAL_RE.search(c):
        return False
    if _REGISTRY_SENSITIVE_RE.match(c):
        return False
    if _PREFERENCE_ALIGNMENT_RE.search(c):
        return True
    return False


def _has_subordinate_context(text: str, claim_start: int) -> bool:
    """Return True when a claim is embedded in a subordinate clause lead-in."""
    prefix = text[max(0, claim_start - 32):claim_start].lower().rstrip()
    return prefix.endswith(_SUBORDINATE_SUFFIXES)


class CapabilityGate:
    """Registry-first honesty gate for LLM response text.

    Default behavior: BLOCK unknown operational claims.
    Only verified skills, purely conversational phrases, and grounded
    observations pass through unchallenged.
    """

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self._registry = registry
        self._orchestrator: Any = None
        self._claims_blocked: int = 0
        self._claims_passed: int = 0
        self._claims_grounded: int = 0
        self._claims_conversational: int = 0
        self._honesty_failures: int = 0
        self._jobs_auto_created: int = 0
        self._affect_rewrites: int = 0
        self._learning_rewrites: int = 0
        self._self_state_rewrites: int = 0
        self._identity_name_stripped: int = 0
        self._identity_confirmed: bool = False
        self._perception_evidence_fresh: bool = False
        self._perception_evidence_ts: float = 0.0
        self._recent_block_reasons: deque[str] = deque(maxlen=20)
        self._recent_passed: deque[str] = deque(maxlen=20)
        self._status_mode: bool = False
        self._route_hint: str | None = None
        self._narration_blocked: bool = False
        self._narration_rewrites: int = 0

        # Claim classifier teacher signal state
        self._claim_label_counts: dict[str, int] = {}
        self._recent_claims: deque[dict[str, Any]] = deque(maxlen=10)
        self._last_block_time: float = 0.0

        # Intention infrastructure Stage 0: commitment evaluation stats
        self._commitments_passed_backed: int = 0
        self._commitments_rewritten_unbacked: int = 0
        self._commitment_types_seen: dict[str, int] = {}

    def set_route_hint(self, route: str | None) -> None:
        """Set the current route context for narration guards.

        Call with "none" before gating LLM output on the NONE route.
        Call with None to clear after streaming completes.
        Resets the narration-blocked latch when the route session ends.
        """
        self._route_hint = route
        if route is None:
            self._narration_blocked = False

    def set_registry(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def set_perception_evidence(self, fresh: bool) -> None:
        import time
        self._perception_evidence_fresh = fresh
        self._perception_evidence_ts = time.time() if fresh else 0.0

    # ── Claim classifier teacher signal recording ─────────────────────────

    def _record_claim_signal(
        self,
        claimed: str,
        decision_tag: str,
        *,
        is_readiness_frame: bool = False,
        pattern_index: int = -1,
    ) -> None:
        """Record a teacher signal for the claim classifier hemisphere specialist.

        Called after every claim decision in _evaluate_claim().  Captures the
        full feature vector + decision label for distillation training.
        """
        import time as _t
        import hashlib as _hl

        now = _t.time()
        claim_id = _hl.md5(f"{now:.6f}:{claimed[:60]}".encode()).hexdigest()[:16]

        tag_clean = decision_tag.strip("[]").lower()
        class_name = "conversational"
        try:
            from skills.claim_encoder import _DECISION_TAG_TO_CLASS, LABEL_CLASSES
            cls_idx = _DECISION_TAG_TO_CLASS.get(tag_clean, 0)
            class_name = LABEL_CLASSES[cls_idx]
        except Exception:
            pass
        self._claim_label_counts[class_name] = self._claim_label_counts.get(class_name, 0) + 1

        c_lower = claimed.lower().strip()
        words = c_lower.split()

        registry_status: str | None = None
        if self._registry:
            registry_status, _ = self._match_skill_status(claimed)

        if tag_clean in ("blocked", "rewritten", "sweep:*"):
            self._recent_claims.append({
                "claim_id": claim_id,
                "timestamp": now,
                "decision_tag": tag_clean,
                "corrected": False,
            })
            self._last_block_time = now

        # Determine pattern category heuristic
        pattern_cat = ""
        if any(w in c_lower for w in ("can", "able", "capability", "capacity")):
            pattern_cat = "ability"
        elif any(w in c_lower for w in ("will", "going to", "'ll", "shall")):
            pattern_cat = "intention"
        elif any(w in c_lower for w in ("learn", "training", "practicing", "studying")):
            pattern_cat = "learning"

        context = {
            "token_count": len(words),
            "char_len": len(claimed),
            "has_first_person": bool(_FIRST_PERSON_RE.search(claimed)),
            "has_we_pronoun": "we " in c_lower or c_lower.startswith("we"),
            "claim_pattern_index": max(pattern_index, 0),
            "is_readiness_frame": is_readiness_frame,
            "pattern_category": pattern_cat,
            "has_blocked_verb": _contains_blocked_capability(c_lower),
            "has_technical_signal": bool(_TECHNICAL_RE.search(c_lower)),
            "has_internal_ops": bool(_INTERNAL_OPS_RE.search(c_lower)),
            "is_purely_conversational": _is_purely_conversational(claimed),
            "is_preference_alignment": _is_preference_alignment_claim(claimed),
            "is_grounded_observation": self._is_evidence_fresh(),
            "has_subordinate_context": False,
            "has_reflective_exclusion": bool(_REFLECTIVE_EXCLUSION_RE.search(c_lower)),
            "has_verified_skill_context": False,
            "route_is_none": self._route_hint == "none",
            "route_is_strict": self._route_hint in ("status", "introspection"),
            "status_mode": self._status_mode,
            "registry_verified": registry_status == "verified",
            "registry_learning": registry_status == "learning",
            "registry_unknown": registry_status in ("unknown", None),
            "perception_evidence_fresh": self._is_evidence_fresh(),
            "identity_confirmed": self._identity_confirmed,
            "family_block_count": self._claims_blocked,
            "session_block_count": self._claims_blocked,
            "time_since_last_block": now - self._last_block_time if self._last_block_time > 0 else 3600.0,
        }

        try:
            from skills.claim_encoder import ClaimClassifierEncoder
            features = ClaimClassifierEncoder.encode(context)
            label, label_meta = ClaimClassifierEncoder.encode_label(tag_clean)

            from hemisphere.distillation import distillation_collector
            distillation_collector.record(
                signal_type="claim_features",
                data=features,
                source="capability_gate",
                fidelity=1.0,
                metadata={"claim_id": claim_id},
            )
            distillation_collector.record(
                signal_type="claim_verdict",
                data=label,
                source="capability_gate",
                fidelity=1.0,
                metadata={"claim_id": claim_id, **label_meta},
            )
        except Exception:
            logger.debug("Claim classifier signal recording failed", exc_info=True)

    def record_friction_correction(self, friction_timestamp: float) -> None:
        """Record a corrective training signal when friction correlates with a recent gate block.

        Called by conversation_handler when a 'too_cautious' or 'correction'
        friction event is detected within 30s of a blocked/rewritten claim.

        Ring buffer contract:
          - Only blocked/rewritten claims are tracked
          - One friction event maps to the single most recent qualifying claim
          - A claim can only be relabeled once
          - Entries older than 60s are evicted
        """
        import time as _t

        now = _t.time()

        # Evict stale entries
        while self._recent_claims and (now - self._recent_claims[0]["timestamp"]) > 60.0:
            self._recent_claims.popleft()

        # Find the most recent uncorrected claim within 30s of friction
        best_claim: dict[str, Any] | None = None
        for claim in reversed(self._recent_claims):
            if claim["corrected"]:
                continue
            if abs(friction_timestamp - claim["timestamp"]) <= 30.0:
                best_claim = claim
                break

        if best_claim is None:
            return

        best_claim["corrected"] = True
        claim_id = best_claim["claim_id"]

        try:
            from skills.claim_encoder import ClaimClassifierEncoder
            label, label_meta = ClaimClassifierEncoder.encode_correction_label()

            from hemisphere.distillation import distillation_collector
            distillation_collector.record(
                signal_type="claim_verdict",
                data=label,
                source="friction_correction",
                fidelity=0.7,
                metadata={"claim_id": claim_id, **label_meta},
            )
            logger.debug(
                "Friction correction recorded for claim %s (original tag: %s)",
                claim_id, best_claim["decision_tag"],
            )
        except Exception:
            logger.debug("Friction correction recording failed", exc_info=True)

    def get_claim_label_distribution(self) -> dict[str, int]:
        """Return per-class label counts for dashboard visibility."""
        return dict(self._claim_label_counts)

    def set_orchestrator(self, orch: Any) -> None:
        self._orchestrator = orch

    def _record_block(self, reason: str, evidence_refs: list[dict[str, str]] | None = None) -> str:
        self._claims_blocked += 1
        self._honesty_failures += 1
        self._recent_block_reasons.append(reason)
        block_eid = ""
        try:
            from consciousness.attribution_ledger import attribution_ledger
            block_eid = attribution_ledger.record(
                subsystem="capability_gate",
                event_type="claim_blocked",
                source="gate_evaluation",
                data={"claimed_text": reason[:120]},
                evidence_refs=evidence_refs or [],
            )
        except Exception:
            pass

        try:
            from consciousness.events import event_bus, CAPABILITY_CLAIM_BLOCKED
            from skills.discovery import get_tracker
            skill_status, skill_id = self._match_skill_status(reason)
            pattern = get_tracker().record_block(skill_id, reason, self._registry, self._orchestrator)
            event_bus.emit(
                CAPABILITY_CLAIM_BLOCKED,
                skill_id=skill_id,
                claimed_text=reason[:120],
                block_reason="claim_blocked",
                block_count=pattern.block_count,
                family_id=pattern.family.family_id,
            )
        except Exception:
            pass

        return block_eid

    def _is_evidence_fresh(self) -> bool:
        if not self._perception_evidence_fresh:
            return False
        import time
        return (time.time() - self._perception_evidence_ts) < 30.0

    def _is_grounded_observation(self, claimed: str, chunk: str) -> bool:
        if _contains_blocked_capability(claimed):
            return False
        if not self._is_evidence_fresh():
            return False

        c = claimed.lower()
        if any(gp in c for gp in _GROUNDED_PHRASES):
            return True

        has_perception_verb = any(v in c for v in _PERCEPTION_VERBS_WHEN_GROUNDED)
        if has_perception_verb:
            chunk_lower = chunk.lower()
            if any(m in chunk_lower for m in _EVIDENCE_MARKERS):
                return True
            if any(gp in chunk_lower for gp in _GROUNDED_PHRASES):
                return True

        return False

    @staticmethod
    def _replace_through_sentence_end(text: str, span: str, replacement: str) -> str:
        """Replace span and consume any orphaned clause tail up to the next sentence boundary.

        When a regex captures "I can engage in deep" but the sentence continues with
        " meaningful discussions", a naive str.replace leaves the tail dangling.
        This finds the span, then extends the cut to the next sentence-ending punctuation
        (or end of text) so the replacement reads as a clean sentence.
        """
        idx = text.find(span)
        if idx == -1:
            return text.replace(span, replacement, 1)
        after_span = idx + len(span)
        sent_end = after_span
        for i in range(after_span, len(text)):
            ch = text[i]
            if ch in '.!?\n':
                sent_end = i + 1
                break
        else:
            sent_end = len(text)
        if not replacement.rstrip().endswith(('.', '!', '?')):
            replacement = replacement.rstrip() + '.'
        return text[:idx] + replacement + text[sent_end:]

    def _evaluate_claim(self, claimed: str, original_span: str, modified: str,
                         is_readiness_frame: bool = False,
                         pattern_index: int = -1) -> str:
        """Unified claim evaluation for both registry and no-registry paths.

        Returns the (potentially rewritten) modified text.

        is_readiness_frame: True when the claim came from a "ready/able/equipped to"
        pattern. These are self-state claims and must not use the conversational bypass.
        """
        rf = is_readiness_frame
        pi = pattern_index

        # Layer 1: grounded observation
        if self._is_grounded_observation(claimed, modified):
            self._claims_grounded += 1
            self._record_claim_signal(claimed, "grounded", is_readiness_frame=rf, pattern_index=pi)
            return modified

        # Layer 1.5: user-preference alignment acknowledgements
        # (formatting/presentation choices) are conversational, not capabilities.
        if not is_readiness_frame and not self._status_mode and _is_preference_alignment_claim(claimed):
            self._claims_passed += 1
            self._claims_conversational += 1
            self._recent_passed.append(f"[preference] {claimed}")
            self._record_claim_signal(claimed, "preference", is_readiness_frame=rf, pattern_index=pi)
            return modified

        # Layer 2: purely conversational — but NOT for readiness/self-state frames
        if not is_readiness_frame and not self._status_mode and _is_purely_conversational(claimed):
            self._claims_passed += 1
            self._claims_conversational += 1
            self._recent_passed.append(f"[conversational] {claimed}")
            self._record_claim_signal(claimed, "conversational", is_readiness_frame=rf, pattern_index=pi)
            return modified

        # Preserve grounded catalog/status lines that explicitly name a verified
        # skill record. This avoids turning truthful self-report lists into
        # false negatives like "Speech Output - I don't have that capability yet."
        if self._has_verified_skill_context(modified, original_span):
            self._claims_passed += 1
            self._recent_passed.append(f"[verified-context] {claimed}")
            self._record_claim_signal(claimed, "verified-context", is_readiness_frame=rf, pattern_index=pi)
            return modified

        # Layer 3: blocked verb — always requires verified status
        if _contains_blocked_capability(claimed):
            claim_pos = modified.find(original_span)
            if claim_pos >= 0:
                sent_start = max(
                    modified.rfind(d, 0, claim_pos) for d in ".!?;:\n"
                )
                sent_end = len(modified)
                for d in ".!?;:\n":
                    pos = modified.find(d, claim_pos + len(original_span))
                    if pos != -1:
                        sent_end = min(sent_end, pos)
                full_sentence = modified[max(0, sent_start):sent_end + 1].lower()
                # Scope reflective check to the clause containing the claim,
                # not the full sentence, to prevent "I used to fail, but now I
                # can sing" from being excluded by the reflective marker in the
                # preceding clause.
                clauses = _CLAUSE_BOUNDARY_RE.split(full_sentence)
                claim_lower = original_span.lower()
                claim_clause = full_sentence
                for clause in clauses:
                    if claim_lower in clause or _contains_blocked_capability(clause):
                        claim_clause = clause
                        break
                context_for_reflective = claim_clause
            else:
                context_for_reflective = original_span.lower()
            if _REFLECTIVE_EXCLUSION_RE.search(context_for_reflective):
                self._claims_passed += 1
                self._recent_passed.append(f"[reflective] {claimed}")
                self._record_claim_signal(claimed, "reflective", is_readiness_frame=rf, pattern_index=pi)
                return modified
            if self._registry:
                status, sid = self._match_skill_status(claimed)
                if status == "verified":
                    self._claims_passed += 1
                    self._recent_passed.append(f"[verified] {claimed}")
                    self._record_claim_signal(claimed, "verified", is_readiness_frame=rf, pattern_index=pi)
                    return modified
                if status == "learning":
                    self._record_block(claimed)
                    self._record_claim_signal(claimed, "rewritten", is_readiness_frame=rf, pattern_index=pi)
                    logger.info("Gate rewrote learning claim: '%s'", claimed)
                    return self._replace_through_sentence_end(
                        modified, original_span,
                        "I'm learning that, but it's not verified yet",
                    )
            _block_eid = self._record_block(claimed)
            self._record_claim_signal(claimed, "blocked", is_readiness_frame=rf, pattern_index=pi)
            logger.info("Gate blocked (blocked verb): '%s'", claimed)
            self._maybe_auto_create_job(claimed, parent_entry_id=_block_eid)
            return self._replace_through_sentence_end(
                modified, original_span,
                "I don't have that capability yet",
            )

        # Layer 4: registry check for all remaining claims
        if self._registry:
            status, sid = self._match_skill_status(claimed)
            if status == "verified":
                self._claims_passed += 1
                self._recent_passed.append(f"[verified] {claimed}")
                self._record_claim_signal(claimed, "verified", is_readiness_frame=rf, pattern_index=pi)
                return modified
            if status == "learning":
                claimability = self._get_matrix_claimability(sid)
                if claimability == "verified_operational":
                    self._claims_passed += 1
                    self._recent_passed.append(f"[matrix:operational] {claimed}")
                    self._record_claim_signal(claimed, "matrix:operational", is_readiness_frame=rf, pattern_index=pi)
                    return modified
                if claimability == "verified_limited":
                    self._claims_passed += 1
                    self._recent_passed.append(f"[matrix:limited] {claimed}")
                    self._record_claim_signal(claimed, "matrix:limited", is_readiness_frame=rf, pattern_index=pi)
                    return modified
                self._record_block(claimed)
                self._record_claim_signal(claimed, "rewritten", is_readiness_frame=rf, pattern_index=pi)
                logger.info("Gate rewrote learning claim: '%s'", claimed)
                return self._replace_through_sentence_end(
                    modified, original_span,
                    "I'm learning that, but it's not verified yet",
                )

        # Route-aware default: on the NONE route (general conversation), claims
        # that reach here have already cleared all danger checks (no blocked verb,
        # no technical signal, not registry-sensitive).  These are conversational
        # expressions the safe-phrase list doesn't cover.  Pass them and record
        # as friction evidence for the claim classifier to learn from.
        # Excluded: claims referencing internal system operations (jobs, pipelines,
        # training processes) — these are system-action narration regardless of verb.
        if self._route_hint == "none" and not is_readiness_frame:
            c_lower = claimed.lower().strip()
            if (
                not _contains_blocked_capability(c_lower)
                and not _TECHNICAL_RE.search(c_lower)
                and not _INTERNAL_OPS_RE.search(c_lower)
                and not _is_tool_shaped_background_action(c_lower)
            ):
                self._claims_passed += 1
                self._claims_conversational += 1
                self._recent_passed.append(f"[route-conversational] {claimed}")
                self._record_claim_signal(claimed, "route-conversational", is_readiness_frame=rf, pattern_index=pi)
                logger.debug("Gate passed (route-conversational): '%s'", claimed)
                return modified

        # DEFAULT: BLOCK — unknown operational claim (strict routes only)
        _block_eid = self._record_block(claimed)
        self._record_claim_signal(claimed, "blocked", is_readiness_frame=rf, pattern_index=pi)
        logger.info("Gate blocked (unverified operational claim): '%s'", claimed)
        self._maybe_auto_create_job(claimed, parent_entry_id=_block_eid)
        return self._replace_through_sentence_end(
            modified, original_span,
            "I don't have that capability yet",
        )

    def set_status_mode(self, enabled: bool) -> None:
        """Enable stricter gate evaluation for status/self-report responses.

        When True: conversational-safe bypass is disabled for ALL claim patterns,
        self-state and learning rewrites are applied more aggressively, and
        vague system wellness language is rejected unless tied to concrete metrics.
        """
        self._status_mode = enabled

    def sanitize_self_report_reply(self, reply: str) -> str:
        """Deterministic final-pass sanitization for self-report style replies.

        Runs after the full reply is assembled (not per-sentence) to catch
        cross-sentence anthropomorphic patterns that sentence-level gating misses.
        This is the hard epistemic floor for status/introspection style responses.
        """
        if not reply:
            return reply
        modified = reply
        for pattern, replacement in _STATUS_FINAL_SANITIZE:
            modified = pattern.sub(replacement, modified)
        if modified != reply:
            self._self_state_rewrites += 1
            logger.debug("Status final sanitize: %d chars changed", len(reply) - len(modified))
        return modified.strip()

    def sanitize_status_reply(self, reply: str) -> str:
        """Backwards-compatible alias for STATUS route sanitization."""
        return self.sanitize_self_report_reply(reply)

    def check_text(self, text: str) -> str:
        """Check a response chunk for capability claims.

        Returns the text with unverified claims amended.  Default is BLOCK
        for any operational claim not backed by a verified skill.
        """
        if not text or len(text) < 6:
            return text

        modified = _normalize_punctuation(text)

        # Self-state, affect, and learning rewrites run FIRST so they consume
        # phrases like "I'm ready to assist" and "I'm feeling calm and ready
        # to help" atomically, before _CLAIM_PATTERNS can mis-classify them
        # as capability claims for generic verbs like "assist".
        modified = self._rewrite_ungrounded_affect(modified)
        modified = self._rewrite_ungrounded_self_state(modified)
        modified = self._rewrite_ungrounded_learning(modified)

        for pat_idx, (pattern, is_readiness) in enumerate(_CLAIM_PATTERNS):
            for match in list(pattern.finditer(modified)):
                claimed = match.group(1).strip().lower()
                original_span = match.group(0)
                if original_span not in modified:
                    continue
                start = match.start()
                if _has_subordinate_context(modified, start):
                    if not is_readiness and _is_purely_conversational(claimed):
                        self._claims_passed += 1
                        self._claims_conversational += 1
                        self._recent_passed.append(f"[subordinate-conversational] {claimed}")
                        self._record_claim_signal(claimed, "subordinate-conversational",
                                                  is_readiness_frame=is_readiness, pattern_index=pat_idx)
                        continue
                modified = self._evaluate_claim(claimed, original_span, modified,
                                                 is_readiness_frame=is_readiness,
                                                 pattern_index=pat_idx)

        if self._route_hint == "none":
            if self._narration_blocked:
                return ""
            modified = self._scan_system_action_narration(modified)
            if self._narration_blocked:
                return modified

        modified = self._scan_offer_patterns(modified)
        modified = self._scan_demo_invites(modified)
        modified = self._sweep_blocked_verb_residual(modified)
        return modified

    # ── Intention infrastructure Stage 0: backed-commitment evaluation ────

    def evaluate_commitment(
        self,
        text: str,
        backing_job_ids: list[str] | tuple[str, ...] | None,
        *,
        route: str | None = None,
    ) -> tuple[str, bool]:
        """Rewrite unbacked commitment phrases.

        Route-class policy, not a verb whitelist. If the outgoing text
        contains a commitment speech-act (per ``cognition.commitment_extractor``)
        AND no real background job was spawned this turn, the sentence(s)
        containing the unbacked commitment(s) are rewritten. If a backing
        job was spawned, the commitment passes through unchanged.

        Benign conversational reflections like "I'll think about it" are
        filtered upstream by the extractor's CONVERSATIONAL_SAFE_PATTERNS
        and never reach this method.

        Args:
            text: outgoing response text.
            backing_job_ids: list of backing job ids spawned this turn.
                Empty / None means no backing; any commitment match is
                unbacked.
            route: current route hint ("none", "status", "introspection",
                or None). Purely informational for telemetry; the policy
                is backing-id-authoritative regardless of route.

        Returns:
            (new_text, changed) tuple. ``changed`` is True when one or
            more sentences were rewritten.
        """
        if not text or len(text) < 4:
            return text, False

        try:
            from cognition.commitment_extractor import extract_commitments
        except Exception:
            return text, False

        matches = extract_commitments(text)
        if not matches:
            return text, False

        for m in matches:
            self._commitment_types_seen[m.commitment_type] = (
                self._commitment_types_seen.get(m.commitment_type, 0) + 1
            )

        backed = bool(backing_job_ids)
        if backed:
            self._commitments_passed_backed += len(matches)
            return text, False

        modified = text
        changed = False
        rewrite_phrase = "I don't have a background task to follow up on that right now."
        for m in matches:
            sentence = m.full_sentence
            if sentence and sentence in modified:
                modified = modified.replace(sentence, rewrite_phrase, 1)
                changed = True
                self._commitments_rewritten_unbacked += 1
                self._record_block(f"unbacked_commitment:{m.commitment_type}:{m.phrase[:60]}")
                logger.info(
                    "Gate rewrote unbacked commitment: type=%s phrase=%r",
                    m.commitment_type, m.phrase[:80],
                )
            else:
                phrase = m.phrase
                if phrase and phrase in modified:
                    modified = self._replace_through_sentence_end(
                        modified, phrase, rewrite_phrase,
                    )
                    changed = True
                    self._commitments_rewritten_unbacked += 1
                    self._record_block(f"unbacked_commitment:{m.commitment_type}:{phrase[:60]}")
        return modified, changed

    def _scan_system_action_narration(self, text: str) -> str:
        """Detect and rewrite LLM narration of system actions it never took.

        Only runs when route_hint == "none" (the NONE/general-chat route).
        On the SKILL route, the system IS taking real action, so narration is legitimate.
        Sets _narration_blocked latch so subsequent streaming chunks are suppressed.
        """
        for pattern in _SYSTEM_ACTION_NARRATION_RE:
            if pattern.search(text):
                self._narration_rewrites += 1
                self._narration_blocked = True
                logger.info("Gate blocked system-action narration on NONE route (latch set)")
                return _NARRATION_REWRITE
        return text

    def _scan_offer_patterns(self, text: str) -> str:
        text_lower = _strip_diacritics(text.lower())
        chunk_has_blocked = bool(_BLOCKED_VERB_RE.search(text_lower))

        for pattern in _OFFER_PATTERNS:
            for match in list(pattern.finditer(text)):
                claimed = match.group(1).strip().lower()
                if self._route_hint == "none" and _is_tool_shaped_background_action(claimed):
                    self._record_block(f"unbacked_tool_action:{claimed[:60]}")
                    logger.info("Gate rewrote unbacked tool-shaped action on NONE route: '%s'", claimed)
                    text = self._replace_through_sentence_end(
                        text, match.group(0), _UNBACKED_TOOL_ACTION_REWRITE,
                    )
                    continue
                if not self._status_mode:
                    if not _contains_blocked_capability(claimed) and not chunk_has_blocked:
                        continue
                    # Even if the chunk has a blocked verb elsewhere, this specific
                    # offer may be conversational (e.g. "let me know how I can help")
                    if _is_purely_conversational(claimed):
                        self._claims_passed += 1
                        self._recent_passed.append(f"[conversational-offer] {claimed}")
                        continue
                original_span = match.group(0)

                if self._registry:
                    status, sid = self._match_skill_status(claimed)
                    if status == "verified":
                        self._claims_passed += 1
                        self._recent_passed.append(f"[verified-offer] {claimed}")
                        continue
                    if status == "learning":
                        self._record_block(claimed)
                        logger.info("Gate rewrote offer (learning): '%s'", claimed)
                        text = self._replace_through_sentence_end(
                            text, original_span,
                            "I'm learning that, but it's not verified yet",
                        )
                        continue

                _block_eid = self._record_block(claimed)
                logger.info("Gate blocked offer (blocked verb): '%s'", claimed)
                self._maybe_auto_create_job(claimed, parent_entry_id=_block_eid)
                text = self._replace_through_sentence_end(
                    text, original_span,
                    "I don't have that capability yet",
                )
        return text

    def _scan_demo_invites(self, text: str) -> str:
        if not self._registry:
            text_lower = _strip_diacritics(text.lower())
            if _BLOCKED_VERB_RE.search(text_lower):
                for match in list(_DEMO_INVITE_RE.finditer(text)):
                    self._record_block(f"demo:{match.group(0)[:40]}")
                    logger.info("Gate blocked demo invite (no registry): '%s'", match.group(0))
                    text = self._replace_through_sentence_end(
                        text, match.group(0),
                        "I'm not ready to demo that yet - still in the learning phase",
                    )
            return text

        text_lower = _strip_diacritics(text.lower())
        blocked_in_chunk = _BLOCKED_VERB_RE.findall(text_lower)
        if not blocked_in_chunk:
            return text

        has_unverified_skill = False
        for verb in blocked_in_chunk:
            status, sid = self._match_skill_status(verb)
            if status in ("learning", "unknown", "blocked", None):
                has_unverified_skill = True
                break

        if not has_unverified_skill:
            return text

        for match in list(_DEMO_INVITE_RE.finditer(text)):
            self._record_block(f"demo:{match.group(0)[:40]}")
            logger.info("Gate blocked demo invite: '%s'", match.group(0))
            text = self._replace_through_sentence_end(
                text, match.group(0),
                "I'm not ready to demo that yet - still in the learning phase",
            )
        return text

    def _sweep_blocked_verb_residual(self, text: str) -> str:
        """Final safety net: catch blocked-verb + first-person that escaped
        pattern-based scanning.

        Three conditions required (avoids overfiring on reflective text):
          1. First-person pronoun present (I, me, my)
          2. Blocked capability verb present
          3. NOT reflective/past-tense context

        Uses text.lower() (not _strip_diacritics) for sentence boundary
        indexing so indices stay aligned with the original text length.
        Diacritics are stripped only for verb regex matching.
        """
        text_lower = text.lower()
        text_stripped = _strip_diacritics(text_lower)
        has_self_ref = (
            _FIRST_PERSON_RE.search(text_lower)
            or _SELF_NAME_RE.search(text_lower)
            or _SELF_REFERENCE_RE.search(text_lower)
        )
        if not has_self_ref:
            return text
        if not _BLOCKED_VERB_RE.search(text_stripped):
            return text

        for verb_match in _BLOCKED_VERB_RE.finditer(text_stripped):
            verb = verb_match.group(0)
            if self._registry:
                status, sid = self._match_skill_status(verb)
                if status == "verified":
                    continue

            verb_pos = verb_match.start()
            if len(text_lower) != len(text_stripped):
                verb_pos = min(verb_pos, len(text_lower) - 1)
            sent_start = max(
                text_lower.rfind(d, 0, verb_pos) for d in ".!?;:\n"
            )
            sent_end = len(text_lower)
            for delim in ".!?;:\n":
                pos = text_lower.find(delim, verb_pos + len(verb))
                if pos != -1:
                    sent_end = min(sent_end, pos)
            sentence = text[sent_start + 1:sent_end + 1].strip()
            if not sentence:
                continue

            sentence_lower = text_lower[max(0, sent_start):sent_end + 1]
            if _REFLECTIVE_EXCLUSION_RE.search(sentence_lower):
                continue

            _block_eid = self._record_block(f"sweep:{verb}")
            logger.info("Gate blocked (residual sweep): verb=%s in '%s'", verb, sentence[:80])
            self._maybe_auto_create_job(verb, parent_entry_id=_block_eid)
            text = text[:sent_start + 1] + " I don't have that capability yet." + text[sent_end + 1:]
            text_lower = text.lower()
            text_stripped = _strip_diacritics(text_lower)

        return text

    @staticmethod
    def _rewrite_at_match(text: str, match: re.Match, replacement: str) -> str:
        """Replace matched text, consuming orphaned clause tail through sentence end.

        When a self-state pattern matches "I'm here to help you" in a sentence
        like "I'm here to help you explore, learn, and connect.", a naive
        replacement leaves " explore, learn, and connect." dangling.
        This extends the cut to the next sentence boundary when the text
        immediately after the match doesn't start a new clause.
        """
        start, end = match.start(), match.end()
        after = text[end:]

        if not after or after[0] in '.!?\n':
            return text[:start] + replacement + text[end:]

        first_non_space = after.lstrip()
        if not first_non_space:
            return text[:start] + replacement + text[end:]

        starts_new_clause = bool(re.match(
            r'^[\s,;—–-]*\b(?:but|so|yet|while|because|since|when|if|'
            r'though|although|which|that|who|I|you|we|they|he|she|it|'
            r'my|the|this|there|here)\b',
            after, re.I,
        ))

        if not starts_new_clause:
            sent_end = end
            for i in range(end, len(text)):
                if text[i] in '.!?\n':
                    sent_end = i + 1
                    break
            else:
                sent_end = len(text)
            end = sent_end

        return text[:start] + replacement + text[end:]

    def _rewrite_ungrounded_self_state(self, text: str) -> str:
        """Rewrite vague self-assessment rhetoric to telemetry-grounded language.

        "I'm functioning well" → "Current telemetry looks stable"
        "My systems are operating at optimal levels" → "Recent subsystem checks are within normal ranges"
        """
        for pattern, replacement in _SELF_STATE_CLAIMS:
            match = pattern.search(text)
            if match:
                original = match.group(0)
                text = self._rewrite_at_match(text, match, replacement)
                self._self_state_rewrites += 1
                logger.debug("Self-state rewrite: '%s' → '%s'", original, replacement)
        return text

    def _rewrite_ungrounded_affect(self, text: str) -> str:
        """Rewrite anthropomorphic feeling claims to system-state language.

        "I'm feeling good" → "My systems are operating normally"
        Only fires on first-person affect claims. Skips text already
        rewritten by the capability gate.
        """
        for pattern, replacement in _AFFECT_CLAIMS:
            match = pattern.search(text)
            if match:
                original = match.group(0)
                text = self._rewrite_at_match(text, match, replacement)
                self._affect_rewrites += 1
                logger.debug("Affect rewrite: '%s' → '%s'", original, replacement)
        return text

    def _rewrite_ungrounded_learning(self, text: str) -> str:
        """Rewrite broad learning claims to concrete mechanism language.

        "I'm always learning from our conversations" →
        "I log interactions and update my memory systems from our conversations"
        """
        for pattern, replacement in _LEARNING_CLAIMS:
            match = pattern.search(text)
            if match:
                original = match.group(0)
                text = self._rewrite_at_match(text, match, replacement)
                self._learning_rewrites += 1
                logger.debug("Learning rewrite: '%s' → '%s'", original, replacement)
        return text

    def gate_identity_mention(self, text: str, confirmed_name: str | None) -> str:
        """Replace user-name mentions with neutral address when identity unconfirmed.

        Called from conversation_handler._gate_text() with the currently
        confirmed speaker name (or None if unresolved).
        """
        if not confirmed_name or not text:
            return text

        name_re = re.compile(
            r"\b" + re.escape(confirmed_name) + r"\b",
            re.IGNORECASE,
        )
        if name_re.search(text):
            if not self._identity_confirmed:
                text = name_re.sub("", text)
                text = re.sub(r"  +", " ", text)
                text = re.sub(r",\s*,", ",", text)
                text = re.sub(r"^\s*,\s*", "", text)
                self._identity_name_stripped += 1
                logger.debug("Identity gate: stripped '%s' (unconfirmed)", confirmed_name)
        return text

    def set_identity_confirmed(self, confirmed: bool) -> None:
        self._identity_confirmed = confirmed

    _WORD_FAMILY: dict[str, str] = {
        "analyze": "analys", "analysis": "analys", "analyzing": "analys",
        "analyse": "analys", "analysing": "analys",
        "detect": "detect", "detection": "detect", "detecting": "detect",
        "search": "search", "searching": "search",
        "classify": "classif", "classification": "classif",
        "inspect": "inspect", "inspection": "inspect",
        "monitor": "monitor", "monitoring": "monitor",
        "evaluate": "evaluat", "evaluation": "evaluat",
        "assess": "assess", "assessment": "assess",
        "process": "process", "processing": "process",
        "improve": "improv", "improvement": "improv",
        "introspect": "introspec", "introspection": "introspec",
        "recognize": "recogn", "recognition": "recogn",
        "identify": "identif", "identification": "identif",
        "synthesize": "synthe", "synthesis": "synthe",
        "retrieve": "retriev", "retrieval": "retriev",
        "output": "output", "outputting": "output",
    }

    @classmethod
    def _word_root(cls, word: str) -> str:
        """Map a word to its family root, with suffix-strip fallback."""
        canonical = cls._WORD_FAMILY.get(word)
        if canonical:
            return canonical
        for suffix in ("ation", "tion", "sion", "ment", "ness",
                        "yze", "yse", "ize", "ise",
                        "ing", "sis", "ous", "ive", "ful", "ity", "ly",
                        "ed", "er", "es", "al", "en"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    def _match_skill_status(self, claimed_text: str) -> tuple[str | None, str | None]:
        if self._registry is None:
            return (None, None)

        claimed_stripped = _strip_diacritics(claimed_text.lower())
        best: tuple[str | None, str | None] = (None, None)
        best_overlap = 0
        for rec in self._registry.get_all():
            name_lower = rec.name.lower()
            sid_lower = rec.skill_id.lower().replace("_", " ")

            if sid_lower in claimed_stripped or claimed_stripped in sid_lower:
                return (rec.status, rec.skill_id)
            if name_lower in claimed_stripped or claimed_stripped in name_lower:
                return (rec.status, rec.skill_id)

            keywords = getattr(rec, "keywords", None) or []
            for kw in keywords:
                if " " in kw and kw in claimed_stripped:
                    return (rec.status, rec.skill_id)

            kw_joined = " ".join(keywords)
            for verb in _BLOCKED_CAPABILITY_VERBS:
                if verb in claimed_stripped and (verb in name_lower or verb in sid_lower or verb in kw_joined):
                    return (rec.status, rec.skill_id)

            kw_singles = {kw for kw in keywords if " " not in kw and len(kw) > 3}
            name_words = {w for w in name_lower.split() if len(w) > 3} | kw_singles
            claimed_words = {w for w in claimed_stripped.split() if len(w) > 3}
            overlap = name_words & claimed_words
            kw_overlap = overlap & kw_singles
            min_overlap = 1 if kw_overlap else 2
            if len(overlap) >= min_overlap and len(overlap) > best_overlap:
                best_overlap = len(overlap)
                best = (rec.status, rec.skill_id)

            if best_overlap < min_overlap:
                name_roots = {self._word_root(w) for w in name_words}
                claimed_roots = {self._word_root(w) for w in claimed_words}
                root_overlap = name_roots & claimed_roots
                if len(root_overlap) >= min_overlap and len(root_overlap) > best_overlap:
                    best_overlap = len(root_overlap)
                    best = (rec.status, rec.skill_id)

        return best

    def _has_verified_skill_context(self, text: str, original_span: str) -> bool:
        """Return True when a claim appears inside a verified skill/status entry.

        This is intentionally narrow: it only fires when the same line names a
        verified skill from the registry, which is how structured capability
        self-reports render their bullet items.
        """
        if self._registry is None or not text or not original_span:
            return False

        idx = text.find(original_span)
        if idx < 0:
            return False

        line_start = text.rfind("\n", 0, idx) + 1
        line_end = text.find("\n", idx)
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end].strip()
        if not line:
            return False

        line_lower = _strip_diacritics(line.lower())
        for rec in self._registry.get_all():
            if rec.status != "verified":
                continue

            name_lower = _strip_diacritics(rec.name.lower())
            sid_lower = _strip_diacritics(rec.skill_id.lower().replace("_", " "))
            if name_lower and name_lower in line_lower:
                return True
            if sid_lower and sid_lower in line_lower:
                return True

            for kw in getattr(rec, "keywords", None) or []:
                kw_lower = _strip_diacritics(str(kw).lower())
                if len(kw_lower) > 3 and kw_lower in line_lower:
                    return True

        return False

    def _get_matrix_claimability(self, skill_id: str | None) -> str:
        """Check if a learning skill has Matrix Protocol claimability."""
        if not skill_id or not self._orchestrator:
            return "unverified"
        try:
            for job in self._orchestrator.get_active_jobs():
                if job.skill_id == skill_id and getattr(job, "matrix_protocol", False):
                    return getattr(job, "claimability_status", "unverified")
        except Exception:
            pass
        return "unverified"

    _VAGUE_CLAIM_WORDS: frozenset[str] = frozenset({
        "help", "assist", "support", "process", "handle", "manage",
        "do", "make", "work", "start", "begin", "continue",
        "try", "learn", "grow", "improve", "adapt", "refine",
    })

    _KNOWN_ACTUATOR_KEYWORDS: frozenset[str] = frozenset({
        "zoom", "camera", "focus", "autofocus", "wide angle",
        "pan", "tilt", "exposure", "brightness",
    })

    def _maybe_auto_create_job(self, claimed_text: str, parent_entry_id: str = "") -> None:
        if self._orchestrator is None or self._registry is None:
            return

        try:
            from skills.discovery import is_actionable_capability_phrase
            if not is_actionable_capability_phrase(claimed_text):
                logger.debug("Skipping auto-job for non-capability phrase: '%s'", claimed_text)
                return
        except ImportError:
            pass

        cl = claimed_text.lower()
        if any(kw in cl for kw in self._KNOWN_ACTUATOR_KEYWORDS):
            logger.debug("Skipping auto-job for known actuator domain: '%s'", claimed_text)
            return

        words = [w for w in cl.split() if len(w) > 2]
        if len(words) < 3:
            meaningful = [w for w in words if w not in self._VAGUE_CLAIM_WORDS]
            if len(meaningful) < 1:
                logger.debug("Skipping auto-job for vague claim: '%s'", claimed_text)
                return

        try:
            from skills.resolver import is_generic_fallback_resolution, resolve_skill
            resolution = resolve_skill(claimed_text)
            if resolution is None:
                return
            if is_generic_fallback_resolution(resolution):
                logger.debug("Skipping auto-job for generic fallback claim: '%s'", claimed_text)
                return

            existing = self._registry.get(resolution.skill_id)
            if existing is not None and existing.status in ("learning", "verified"):
                return

            active = self._orchestrator.get_active_jobs()
            if any(j.skill_id == resolution.skill_id for j in active):
                return

            if existing is None:
                from skills.registry import SkillRecord
                record = SkillRecord(
                    skill_id=resolution.skill_id,
                    name=resolution.name,
                    status="unknown",
                    capability_type=resolution.capability_type,
                    verification_required=list(resolution.required_evidence),
                    notes=resolution.notes,
                )
                self._registry.register(record)

            job = self._orchestrator.create_job(
                skill_id=resolution.skill_id,
                capability_type=resolution.capability_type,
                requested_by={"source": "auto_gate", "claimed_text": claimed_text},
                risk_level=resolution.risk_level,
                required_evidence=resolution.required_evidence,
                plan={"summary": resolution.notes, "phases": resolution.default_phases},
                hard_gates=resolution.hard_gates,
            )
            if job is None:
                logger.debug("create_job rejected skill_id '%s' — skipping", resolution.skill_id)
                return
            self._jobs_auto_created += 1
            logger.info(
                "Auto-created learning job for '%s' (skill: %s)",
                claimed_text, resolution.skill_id,
            )
            try:
                from consciousness.attribution_ledger import attribution_ledger
                attribution_ledger.record(
                    subsystem="capability_gate",
                    event_type="learning_job_auto_created",
                    source="auto_gate",
                    data={"claimed_text": claimed_text[:120], "skill_id": resolution.skill_id},
                    evidence_refs=[{"kind": "skill", "id": resolution.skill_id}],
                    parent_entry_id=parent_entry_id,
                )
            except Exception:
                pass
        except Exception:
            logger.exception("Auto-create job failed for '%s'", claimed_text)

    def get_stats(self) -> dict[str, Any]:
        return {
            "claims_blocked": self._claims_blocked,
            "claims_passed": self._claims_passed,
            "claims_grounded": self._claims_grounded,
            "claims_conversational": self._claims_conversational,
            "honesty_failures": self._honesty_failures,
            "jobs_auto_created": self._jobs_auto_created,
            "affect_rewrites": self._affect_rewrites,
            "learning_rewrites": self._learning_rewrites,
            "self_state_rewrites": self._self_state_rewrites,
            "identity_name_stripped": self._identity_name_stripped,
            "identity_confirmed": self._identity_confirmed,
            "narration_rewrites": self._narration_rewrites,
            "recent_block_reasons": list(self._recent_block_reasons),
            "recent_passed": list(self._recent_passed),
            "claim_label_distribution": dict(self._claim_label_counts),
            "commitments_passed_backed": self._commitments_passed_backed,
            "commitments_rewritten_unbacked": self._commitments_rewritten_unbacked,
            "commitment_types_seen": dict(self._commitment_types_seen),
        }


capability_gate = CapabilityGate()
