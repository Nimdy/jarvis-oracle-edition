"""Identity Name Validator — rejects non-name tokens before they contaminate registries.

Guards against phrase fragments, behavior states, common English words, and other
non-person tokens being promoted into speaker profiles, face profiles, or rapport
relationships. Used at every identity creation boundary.
"""

from __future__ import annotations

import re

_BLOCKED_WORDS: frozenset[str] = frozenset({
    # Behavior / posture / state tokens (the "Staring" class of bug)
    "staring", "sitting", "standing", "walking", "running", "sleeping",
    "eating", "drinking", "typing", "coding", "reading", "talking",
    "thinking", "looking", "watching", "waiting", "leaning", "working",
    "listening", "playing", "smiling", "laughing", "crying", "yelling",
    "pointing", "waving", "nodding", "shaking", "moving", "leaving",
    "entering", "speaking", "singing", "dancing", "writing", "driving",
    "breathing", "blinking", "stretching", "reaching", "holding",
    "pushing", "pulling", "lifting", "dropping", "falling", "jumping",
    "kneeling", "crouching", "bending", "turning", "spinning",

    # Common objects / environment
    "monitor", "screen", "camera", "computer", "keyboard", "mouse",
    "chair", "desk", "table", "phone", "window", "door", "wall",
    "light", "speaker", "microphone", "laptop", "display", "remote",

    # System / technical tokens
    "unknown", "none", "null", "undefined", "anonymous", "default",
    "system", "admin", "user", "guest", "test", "debug", "error",
    "warning", "info", "true", "false", "yes", "no",

    # Common English filler
    "the", "this", "that", "here", "there", "what", "when", "where",
    "which", "while", "just", "really", "actually", "basically",
    "something", "nothing", "everything", "everyone", "someone",
    "anyone", "maybe", "probably", "already", "still", "very",
    "hello", "hey", "okay", "sure", "fine", "good", "bad",
    "like", "love", "hate", "want", "need", "know", "think",
    "feel", "said", "told", "asked", "going", "doing", "being",
    "having", "getting", "making", "taking", "coming",

    # Common "I'm ____" completions that are adjectives/states, not names
    "new", "back", "home", "ready", "busy", "sorry", "done",
    "late", "early", "lost", "tired", "confused", "happy", "sad",
    "sick", "great", "alright", "excited", "bored", "hungry",
    "cold", "hot", "warm", "scared", "nervous", "alone",
    "awake", "asleep", "free", "stuck", "safe", "right",
    "wrong", "certain", "curious", "available", "unavailable",

    # Jarvis-specific tokens that should never be person names
    "jarvis", "brain", "consciousness", "autonomy", "calibration",
    "quarantine", "hemisphere", "cortex", "kernel", "policy",
})

_MIN_NAME_LENGTH = 2
_MAX_NAME_LENGTH = 40

_ONLY_LETTERS_RE = re.compile(r"^[A-Za-z][A-Za-z' \-]{0,38}[A-Za-z]?$")


def is_valid_person_name(name: str) -> bool:
    """Return True if name is plausibly a person name, not a phrase fragment or object."""
    if not name or not isinstance(name, str):
        return False

    clean = name.strip()
    if len(clean) < _MIN_NAME_LENGTH or len(clean) > _MAX_NAME_LENGTH:
        return False

    if not _ONLY_LETTERS_RE.match(clean):
        return False

    if clean.lower() in _BLOCKED_WORDS:
        return False

    parts = clean.split()
    if len(parts) == 1 and parts[0].lower().endswith("ing") and len(parts[0]) > 4:
        return False

    return True


def rejection_reason(name: str) -> str | None:
    """Return a human-readable reason if the name is invalid, or None if valid."""
    if not name or not isinstance(name, str):
        return "empty name"

    clean = name.strip()
    if len(clean) < _MIN_NAME_LENGTH:
        return f"too short ({len(clean)} chars)"
    if len(clean) > _MAX_NAME_LENGTH:
        return f"too long ({len(clean)} chars)"
    if not _ONLY_LETTERS_RE.match(clean):
        return "contains non-letter characters"
    if clean.lower() in _BLOCKED_WORDS:
        return f"blocked word: '{clean.lower()}'"

    parts = clean.split()
    if len(parts) == 1 and parts[0].lower().endswith("ing") and len(parts[0]) > 4:
        return f"looks like a gerund/behavior state: '{clean}'"

    return None
