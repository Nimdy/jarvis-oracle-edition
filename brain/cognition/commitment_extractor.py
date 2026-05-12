"""Commitment extractor — regex bootstrap for outgoing commitment speech acts.

Stage 0 infrastructure. Detects when JARVIS's outgoing text makes a
commitment to the user (e.g. "I'll get back to you", "give me a moment",
"I will follow up"). The `IntentionRegistry` then binds the match to a
real backing job id captured this turn; the `CapabilityGate` rewrites
matches that have no backing job.

Scope:
  - Pure regex bootstrap. Lives in the "hardcoded rules are a bootstrap"
    band of the NN Maturity Flywheel. A future `claim_classifier` label
    class will shadow these patterns once we have ground-truth pairs.
  - No LLM. No side effects. Pure function.

Classification schema (commitment_type):
  - follow_up          : "I'll get back to you", "I'll let you know when done"
  - deferred_action    : "give me a moment", "let me process", "one moment"
  - future_work        : "I'll analyze", "I will research", "I'll look into"
  - task_started       : "I've begun", "I've initiated", "I've started" (past-tense
                         action claim; this is the class the 20:31:20 regression hit)
  - generic            : anything else matching the commitment shape

The extractor intentionally does NOT match benign conversational reflections
like "I'll think about it" — those are safe on the NONE route (see
capability_gate.py route-aware policy).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern

__all__ = [
    "CommitmentMatch",
    "extract_commitments",
    "COMMITMENT_PATTERNS",
    "CONVERSATIONAL_SAFE_PATTERNS",
]


# ---------------------------------------------------------------------------
# Conversational safe list: phrases that look like commitments but are
# reflective/polite and should NOT register an intention. These drop out of
# the extractor before any commitment pattern runs.
# ---------------------------------------------------------------------------

CONVERSATIONAL_SAFE_PATTERNS: list[Pattern[str]] = [
    re.compile(r"\bI(?:['\u2019]ll| will)\s+think about (?:it|that|this)\b", re.I),
    re.compile(r"\bI(?:['\u2019]ll| will)\s+consider (?:it|that|this)\b", re.I),
    re.compile(r"\bI(?:['\u2019]ll| will)\s+keep that in mind\b", re.I),
    re.compile(r"\bI(?:['\u2019]ll| will)\s+remember (?:that|this)\b", re.I),
    re.compile(r"\bI(?:['\u2019]ll| will)\s+take (?:that|this) (?:into account|under advisement)\b", re.I),
    re.compile(r"\bI(?:['\u2019]ll| will)\s+bear that in mind\b", re.I),
]


# ---------------------------------------------------------------------------
# Commitment patterns. Tuple of (compiled regex, commitment_type,
# suggested_backing_kind).
#
# suggested_backing_kind is a HINT only — the actual backing_job_kind is
# determined by the turn's route dispatch (LIBRARY_INGEST, ACADEMIC_SEARCH,
# WEB_SEARCH, etc). If no job was dispatched, CapabilityGate rewrites
# regardless of the suggested kind.
# ---------------------------------------------------------------------------

_COMMITMENT_SPEC: list[tuple[str, str, str]] = [
    # --- follow_up: explicit "I'll get back to you" style -----------------
    (
        r"\bI(?:['\u2019]ll| will| am going to)\s+(?:get|come|circle|follow)\s+back\s+(?:to you|with you|on this|on that)\b",
        "follow_up",
        "follow_up_any",
    ),
    (
        r"\bI(?:['\u2019]ll| will)\s+let you know\s+(?:when|once|as soon as)\b",
        "follow_up",
        "follow_up_any",
    ),
    (
        r"\bI(?:['\u2019]ll| will)\s+(?:tell|inform|update|notify)\s+you\s+(?:when|once|as soon as)\b",
        "follow_up",
        "follow_up_any",
    ),
    (
        r"\bI(?:['\u2019]ll| will)\s+report back\b",
        "follow_up",
        "follow_up_any",
    ),

    # --- deferred_action: "give me a moment", "one moment" ----------------
    (
        r"\b(?:give|allow)\s+me\s+a\s+(?:moment|minute|second|sec|few (?:moments|minutes|seconds))\b",
        "deferred_action",
        "any_background",
    ),
    (
        r"\b(?:please\s+)?allow me a moment to\s+(?:process|organize|analyze|think|work|study|research|review|retrieve|fetch|pull up|search)",
        "deferred_action",
        "any_background",
    ),
    (
        r"\b(?:just\s+)?(?:a|one)\s+(?:moment|minute|second|sec)\s+(?:please|while I)\b",
        "deferred_action",
        "any_background",
    ),
    (
        r"\blet me\s+(?:process|organize|work on|think on|look into|research|study|analyze|check|retrieve|fetch|pull up|search(?: for)?)\b",
        "deferred_action",
        "any_background",
    ),
    (
        r"\bhold\s+on\s+(?:while|as|a moment|one moment)\b",
        "deferred_action",
        "any_background",
    ),

    # --- future_work: "I will analyze/research/look into" -----------------
    (
        r"\bI(?:['\u2019]ll| will|\s+am going to)\s+(?:(?:begin|start|continue|keep|go on)(?:\s+by)?\s+)?(?:analyze|research(?:ing)?|look(?:ing)? into|investigat(?:e|ing)|stud(?:y|ying)|explore|search(?:ing)?(?: for)?|retriev(?:e|ing)|fetch(?:ing)?|pull(?:ing)? up|compile|summarize|integrate)\b",
        "future_work",
        "any_background",
    ),
    (
        r"\bI(?:['\u2019]m| am)\s+going to\s+(?:(?:begin|start|continue|keep|go on)(?:\s+by)?\s+)?(?:analyze|research(?:ing)?|look(?:ing)? into|investigat(?:e|ing)|stud(?:y|ying)|explore|search(?:ing)?(?: for)?|retriev(?:e|ing)|fetch(?:ing)?|pull(?:ing)? up)\b",
        "future_work",
        "any_background",
    ),

    # --- task_started: "I have begun / I've initiated / I've started" -----
    # This is the class the 20:31:20 regression hit.
    (
        r"\bI(?:['\u2019]ve| have|\s+just)\s+(?:initiated|begun|started|kicked off|launched|commenced|opened|fired off|dispatched|queued)\s+(?:a|an|the|my|some)?\s*(?:study|search|analysis|research|investigation|review|task|job|process|pipeline|query|lookup)\b",
        "task_started",
        "any_background",
    ),
    (
        r"\bI(?:['\u2019]m| am)\s+(?:currently\s+)?(?:initiating|beginning|starting|kicking off|launching|commencing|processing|analyzing|researching|studying|investigating|reviewing|searching for)\b",
        "task_started",
        "any_background",
    ),

    # --- generic promise-to-follow-up fallback ----------------------------
    (
        r"\b(?:once|when)\s+(?:I(?:['\u2019]m| am))?\s*(?:done|finished|complete|ready)\s*,?\s+I(?:['\u2019]ll| will)\b",
        "follow_up",
        "follow_up_any",
    ),
]


COMMITMENT_PATTERNS: list[tuple[Pattern[str], str, str]] = [
    (re.compile(pat, re.I), ctype, kind)
    for (pat, ctype, kind) in _COMMITMENT_SPEC
]


@dataclass
class CommitmentMatch:
    """A single commitment phrase found in outgoing text."""

    phrase: str
    commitment_type: str
    suggested_backing_kind: str
    span: tuple[int, int]
    full_sentence: str

    def __iter__(self):
        # Allow tuple-style unpacking for test ergonomics.
        yield self.phrase
        yield self.commitment_type
        yield self.suggested_backing_kind


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into (start, end, sentence) tuples. Simple, safe for speech.

    We do not need a linguistic sentence tokenizer here — JARVIS's outgoing
    text is pre-gated and typically short. This keeps the dependency graph
    minimal and avoids NLTK-style imports.
    """
    parts: list[tuple[int, int, str]] = []
    if not text:
        return parts
    buf_start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?":
            # Consume trailing punctuation and whitespace
            j = i + 1
            while j < n and text[j] in ".!?)]\"'\u2019 \t\n":
                j += 1
            sentence = text[buf_start:j].strip()
            if sentence:
                parts.append((buf_start, j, sentence))
            buf_start = j
            i = j
        else:
            i += 1
    if buf_start < n:
        tail = text[buf_start:].strip()
        if tail:
            parts.append((buf_start, n, tail))
    return parts


def _is_conversational_safe(text: str) -> bool:
    for pat in CONVERSATIONAL_SAFE_PATTERNS:
        if pat.search(text):
            return True
    return False


def extract_commitments(text: str) -> list[CommitmentMatch]:
    """Scan text for commitment speech acts.

    Returns a list of CommitmentMatch. May return multiple matches when
    the text contains several commitments (e.g. "I'll analyze X. I've also
    begun searching for Y."). Matches are ordered by their position in the
    original string.

    This function is pure: no side effects, no side reads.
    """
    if not text:
        return []

    results: list[CommitmentMatch] = []
    sentences = _split_sentences(text)
    for s_start, s_end, sentence in sentences:
        if _is_conversational_safe(sentence):
            continue
        seen_types: set[str] = set()
        for pat, ctype, kind in COMMITMENT_PATTERNS:
            m = pat.search(sentence)
            if not m:
                continue
            if ctype in seen_types:
                continue
            seen_types.add(ctype)
            phrase = m.group(0).strip()
            abs_start = s_start + m.start()
            abs_end = s_start + m.end()
            results.append(
                CommitmentMatch(
                    phrase=phrase,
                    commitment_type=ctype,
                    suggested_backing_kind=kind,
                    span=(abs_start, abs_end),
                    full_sentence=sentence,
                )
            )

    results.sort(key=lambda cm: cm.span[0])
    return results
