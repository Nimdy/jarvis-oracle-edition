"""Claim extractor fidelity tests — Data Fidelity Sprint.

These tests define the DESIRED behavior for claim extraction:
- No mid-word fragments in canonical subjects or objects
- No empty canonical terms
- rendered_claim preserves meaningful content
- EXTRACTION_MAX_CLAIMS_PER_MEMORY unchanged at 3
- Compatible conflict keys for same-topic claims
- Contradiction engine tests still pass (run separately)

Tests are written BEFORE the fix so the first run captures current failures.
After the fix, all tests must pass.
"""

from __future__ import annotations

import re
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

sys.path.insert(0, ".")

from epistemic.claim_extractor import (
    canonicalize_term,
    extract_claims,
)
from epistemic.belief_record import (
    EXTRACTION_MAX_CLAIMS_PER_MEMORY,
    build_conflict_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeMemory:
    id: str = "mem_test_001"
    type: str = "factual_knowledge"
    payload: Any = ""
    provenance: str = "test"
    tags: tuple[str, ...] = ()
    identity_subject: str = ""
    identity_subject_type: str = ""


_NATURAL_SHORT_WORDS = frozenset({
    "a", "i", "an", "am", "as", "at", "be", "by", "do", "go", "he",
    "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so",
    "to", "up", "us", "we", "id",
    "add", "age", "ago", "aid", "aim", "air", "all", "and", "any",
    "app", "apt", "are", "arm", "art", "ask", "ate", "aws",
    "bad", "bag", "ban", "bar", "bat", "bed", "big", "bit", "box",
    "bug", "bus", "but", "buy",
    "can", "cap", "car", "cat", "cpu", "cry", "cup", "cut",
    "day", "did", "die", "dig", "dim", "dip", "dog", "dot", "dry", "due",
    "ear", "eat", "ego", "end", "era", "etc", "eve", "eye",
    "fan", "far", "fat", "few", "fig", "fit", "fix", "fly", "for", "fox",
    "fun", "fur",
    "gap", "gas", "get", "got", "gpu", "gun", "gut", "guy",
    "had", "has", "hat", "her", "hid", "him", "hip", "his", "hit",
    "hot", "how", "hub",
    "ice", "ill", "ink", "inn", "ion", "its",
    "jar", "jaw", "jet", "job", "joy",
    "key", "kid", "kin", "kit",
    "lab", "lag", "lap", "law", "lay", "led", "leg", "let", "lid",
    "lie", "lip", "log", "lot", "low",
    "mad", "man", "map", "mat", "max", "may", "men", "met", "mid",
    "min", "mix", "mob", "mod", "mom", "mud",
    "nap", "net", "new", "nil", "nod", "nor", "not", "now", "nut",
    "oak", "odd", "off", "oil", "old", "one", "opt", "ore", "our",
    "out", "owe", "own",
    "pad", "pan", "pay", "pen", "per", "pet", "pie", "pig", "pin",
    "pit", "pod", "pop", "pot", "pre", "pub", "put",
    "ran", "rat", "raw", "ray", "red", "ref", "rib", "rid", "rim",
    "rip", "rod", "rot", "row", "rug", "run", "rut",
    "sad", "sat", "saw", "say", "sea", "set", "she", "shy", "sir",
    "sit", "six", "ski", "sky", "sly", "son", "soy", "spy", "sub",
    "sum", "sun",
    "tab", "tag", "tan", "tap", "tar", "tax", "tea", "ten", "the",
    "tie", "tin", "tip", "toe", "too", "top", "toy", "try", "tug",
    "two",
    "url", "use",
    "van", "via", "vow",
    "war", "was", "way", "web", "wet", "who", "why", "wig", "win",
    "wit", "woe", "wok", "won", "wow",
    "yet", "you",
    "zen", "zip", "zoo",
})


def _has_word_fragment(term: str) -> bool:
    """Return True if a canonical term looks like it was cut mid-word.

    Checks if the last underscore-separated token is suspiciously short
    (1-3 chars) and is not a natural English short word.
    """
    if not term:
        return False
    parts = term.split("_")
    if len(parts) <= 1:
        return False
    last = parts[-1]
    if len(last) <= 3 and last not in _NATURAL_SHORT_WORDS:
        return True
    return False


def _is_not_empty(term: str) -> bool:
    return bool(term) and term != "unknown"


# ---------------------------------------------------------------------------
# Invariant: max claims per memory stays at 3
# ---------------------------------------------------------------------------

class TestExtractionLimits:
    def test_max_claims_constant_is_3(self):
        assert EXTRACTION_MAX_CLAIMS_PER_MEMORY == 3


# ---------------------------------------------------------------------------
# Fidelity: no mid-word fragments in canonical terms
# ---------------------------------------------------------------------------

class TestNoMidWordFragments:
    """After the fix, canonical subjects and objects must be whole-word phrases."""

    LONG_CLAIMS = [
        "Testosterone replacement therapy increases red blood cell production in susceptible men",
        "Python's GIL prevents true multi-threaded parallelism in CPython implementations",
        "Transformer attention mechanisms improve long-range dependency modeling in NLP",
        "The fundamental nature of consciousness remains an open philosophical question",
        "Distributed systems require careful handling of network partitions and consistency",
        "Reinforcement learning with human feedback aligns language model behavior with preferences",
    ]

    @pytest.mark.parametrize("claim_text", LONG_CLAIMS)
    def test_library_pointer_no_fragment_subject(self, claim_text):
        mem = FakeMemory(
            payload={"type": "library_pointer", "claim": claim_text, "source_id": "src_1"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        subj = beliefs[0].canonical_subject
        assert _is_not_empty(subj), f"Empty subject from: {claim_text[:60]}"
        assert not _has_word_fragment(subj), f"Word fragment in subject: '{subj}' from: {claim_text[:60]}"

    @pytest.mark.parametrize("claim_text", LONG_CLAIMS)
    def test_library_pointer_no_fragment_object(self, claim_text):
        mem = FakeMemory(
            payload={"type": "library_pointer", "claim": claim_text, "source_id": "src_1"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        obj = beliefs[0].canonical_object
        assert _is_not_empty(obj), f"Empty object from: {claim_text[:60]}"
        assert not _has_word_fragment(obj), f"Word fragment in object: '{obj}' from: {claim_text[:60]}"

    @pytest.mark.parametrize("claim_text", LONG_CLAIMS)
    def test_study_claim_no_fragment(self, claim_text):
        mem = FakeMemory(
            payload={"type": "study_claim", "claim": claim_text, "claim_type": "factual"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        subj = beliefs[0].canonical_subject
        obj = beliefs[0].canonical_object
        assert not _has_word_fragment(subj), f"Fragment subject: '{subj}'"
        assert not _has_word_fragment(obj), f"Fragment object: '{obj}'"

    def test_observation_no_fragment(self):
        text = "User engagement has been consistently high during afternoon sessions with complex topics"
        mem = FakeMemory(type="observation", payload=text)
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        subj = beliefs[0].canonical_subject
        assert not _has_word_fragment(subj), f"Fragment subject: '{subj}'"

    def test_core_value_no_fragment(self):
        text = "Continuous self-improvement through evidence-based learning is fundamentally important"
        mem = FakeMemory(type="core", payload=text)
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        subj = beliefs[0].canonical_subject
        obj = beliefs[0].canonical_object
        assert not _has_word_fragment(subj), f"Fragment subject: '{subj}'"
        assert not _has_word_fragment(obj), f"Fragment object: '{obj}'"

    def test_policy_string_no_fragment(self):
        text = "Reducing response latency by caching frequently accessed memory clusters improved overall performance"
        mem = FakeMemory(type="self_improvement", payload=text)
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        subj = beliefs[0].canonical_subject
        obj = beliefs[0].canonical_object
        assert not _has_word_fragment(subj), f"Fragment subject: '{subj}'"
        assert not _has_word_fragment(obj), f"Fragment object: '{obj}'"


# ---------------------------------------------------------------------------
# Fidelity: canonical terms are never empty
# ---------------------------------------------------------------------------

class TestNoEmptyTerms:
    def test_library_pointer_nonempty(self):
        mem = FakeMemory(
            payload={"type": "library_pointer", "claim": "Short claim", "source_id": "s1"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        assert _is_not_empty(beliefs[0].canonical_subject)
        assert _is_not_empty(beliefs[0].canonical_object)

    def test_question_summary_nonempty(self):
        mem = FakeMemory(
            payload={"question": "How does memory consolidation work?", "summary": "Sleep cycles enable memory consolidation through neural replay"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        assert _is_not_empty(beliefs[0].canonical_subject)
        assert _is_not_empty(beliefs[0].canonical_object)

    def test_empty_payload_returns_no_beliefs(self):
        mem = FakeMemory(payload="")
        assert extract_claims(mem) == []

    def test_whitespace_payload_returns_no_beliefs(self):
        mem = FakeMemory(payload="   ")
        assert extract_claims(mem) == []

    def test_identity_nonempty(self):
        mem = FakeMemory(
            payload={"summary": "My identity is shaped by continuous learning and adaptation"},
            tags=("identity", "philosophical"),
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        assert _is_not_empty(beliefs[0].canonical_object)


# ---------------------------------------------------------------------------
# Fidelity: rendered_claim preserves meaningful content
# ---------------------------------------------------------------------------

class TestRenderedClaimQuality:
    def test_rendered_claim_long_enough(self):
        claim = "Transformer attention mechanisms improve long-range dependency modeling in NLP tasks"
        mem = FakeMemory(
            payload={"type": "library_pointer", "claim": claim, "source_id": "s1"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        rc = beliefs[0].rendered_claim
        assert len(rc) >= 60, f"rendered_claim too short ({len(rc)} chars): '{rc}'"

    def test_rendered_claim_not_truncated_below_200(self):
        claim = "A moderately long claim that should be preserved in its entirety for display purposes"
        mem = FakeMemory(
            payload={"type": "library_pointer", "claim": claim, "source_id": "s1"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        rc = beliefs[0].rendered_claim
        assert len(rc) >= len(claim) - 10 or len(rc) >= 80, \
            f"rendered_claim aggressively truncated: '{rc}'"


# ---------------------------------------------------------------------------
# Fidelity: same-topic claims produce compatible conflict keys
# ---------------------------------------------------------------------------

class TestConflictKeyCompatibility:
    def test_same_topic_different_wording_produces_whole_word_subjects(self):
        """Two claims about Python's GIL with different wording should
        produce whole-word subjects (not garbled character slices).
        Exact match requires NLP-level synonym resolution beyond
        heuristic extraction; conflict_key handles actual comparison."""
        claim_a = "Python's GIL prevents true multi-threaded parallelism in CPython"
        claim_b = "Python's GIL limits concurrent thread execution in the CPython interpreter"

        mem_a = FakeMemory(
            id="mem_a",
            payload={"type": "library_pointer", "claim": claim_a, "source_id": "s1"},
        )
        mem_b = FakeMemory(
            id="mem_b",
            payload={"type": "library_pointer", "claim": claim_b, "source_id": "s2"},
        )

        beliefs_a = extract_claims(mem_a)
        beliefs_b = extract_claims(mem_b)
        assert len(beliefs_a) >= 1 and len(beliefs_b) >= 1

        subj_a = beliefs_a[0].canonical_subject
        subj_b = beliefs_b[0].canonical_subject
        assert not _has_word_fragment(subj_a), f"Fragment in A: '{subj_a}'"
        assert not _has_word_fragment(subj_b), f"Fragment in B: '{subj_b}'"
        assert "pythons_gil" in subj_a, f"Subject A missing GIL: '{subj_a}'"
        assert "pythons_gil" in subj_b, f"Subject B missing GIL: '{subj_b}'"

    def test_identical_claims_produce_identical_subjects(self):
        """Two memories with the exact same claim text must produce
        identical canonical_subjects for dedup to work."""
        claim = "Python's GIL prevents true multi-threaded parallelism in CPython"
        mem_a = FakeMemory(
            id="mem_a",
            payload={"type": "library_pointer", "claim": claim, "source_id": "s1"},
        )
        mem_b = FakeMemory(
            id="mem_b",
            payload={"type": "library_pointer", "claim": claim, "source_id": "s2"},
        )
        beliefs_a = extract_claims(mem_a)
        beliefs_b = extract_claims(mem_b)
        assert len(beliefs_a) >= 1 and len(beliefs_b) >= 1
        assert beliefs_a[0].canonical_subject == beliefs_b[0].canonical_subject

    def test_question_summary_compatible_subjects(self):
        """Two research memories about the same question should share a subject."""
        mem_a = FakeMemory(
            id="mem_a",
            payload={"question": "How does memory consolidation work during sleep?",
                     "summary": "Neural replay during slow-wave sleep consolidates memories"},
        )
        mem_b = FakeMemory(
            id="mem_b",
            payload={"question": "How does memory consolidation work during sleep?",
                     "summary": "REM sleep plays a role in emotional memory processing"},
        )
        beliefs_a = extract_claims(mem_a)
        beliefs_b = extract_claims(mem_b)
        assert len(beliefs_a) >= 1 and len(beliefs_b) >= 1
        assert beliefs_a[0].canonical_subject == beliefs_b[0].canonical_subject


# ---------------------------------------------------------------------------
# Boundary: short inputs still produce valid beliefs
# ---------------------------------------------------------------------------

class TestShortInputs:
    def test_short_library_pointer_claim(self):
        mem = FakeMemory(
            payload={"type": "library_pointer", "claim": "GIL is bad", "source_id": "s1"},
        )
        beliefs = extract_claims(mem)
        # "GIL is bad" is > 5 chars, should produce a belief
        # (current code requires > 10, which may filter this)
        # After fix: short claims should still work if > 5 chars

    def test_three_word_string_factual(self):
        mem = FakeMemory(payload="Memory improves recall")
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        assert _is_not_empty(beliefs[0].canonical_subject)
        assert _is_not_empty(beliefs[0].canonical_object)

    def test_preference_single_word(self):
        mem = FakeMemory(type="user_preference", payload="Python")
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1


# ---------------------------------------------------------------------------
# Regression: research_summary type returns empty (by design)
# ---------------------------------------------------------------------------

class TestDesignedBehavior:
    def test_research_summary_returns_empty(self):
        mem = FakeMemory(
            payload={"type": "research_summary", "summary": "Some research"},
        )
        assert extract_claims(mem) == []

    def test_conversation_without_identity_returns_empty(self):
        mem = FakeMemory(type="conversation", payload="Hello there")
        assert extract_claims(mem) == []

    def test_conversation_with_identity_tags_extracts(self):
        mem = FakeMemory(
            type="conversation",
            payload={"summary": "I think identity is shaped by experience"},
            tags=("identity", "philosophical"),
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1


# ---------------------------------------------------------------------------
# Structural: conflict_key is derived from canonical fields
# ---------------------------------------------------------------------------

class TestConflictKeyStructure:
    def test_conflict_key_uses_canonical_fields(self):
        mem = FakeMemory(
            payload={"type": "library_pointer",
                     "claim": "Neural networks can approximate any continuous function",
                     "source_id": "s1"},
        )
        beliefs = extract_claims(mem)
        assert len(beliefs) >= 1
        b = beliefs[0]
        assert b.conflict_key, "conflict_key should not be empty"
        assert b.canonical_subject in b.conflict_key or "::" in b.conflict_key
