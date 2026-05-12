"""Tests for identity name validation — prevents phrase fragments, behavior states,
and common English words from being promoted into persistent person records.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from identity.name_validator import is_valid_person_name, rejection_reason


# ── Valid names ───────────────────────────────────────────────────────────────

def test_valid_simple_names():
    assert is_valid_person_name("David")
    assert is_valid_person_name("Sarah")
    assert is_valid_person_name("Chris")
    assert is_valid_person_name("Tonya")
    assert is_valid_person_name("Alex")


def test_valid_compound_names():
    assert is_valid_person_name("Mary Jane")
    assert is_valid_person_name("Jean-Pierre")
    assert is_valid_person_name("O'Brien")
    assert is_valid_person_name("Anna Marie")


def test_valid_short_names():
    assert is_valid_person_name("Li")
    assert is_valid_person_name("Jo")
    assert is_valid_person_name("Al")


# ── Blocked behavior/state tokens ("Staring" class of bug) ──────────────────

def test_rejects_staring():
    assert not is_valid_person_name("Staring")
    assert not is_valid_person_name("staring")
    assert not is_valid_person_name("STARING")


def test_rejects_gerunds():
    assert not is_valid_person_name("Walking")
    assert not is_valid_person_name("Sitting")
    assert not is_valid_person_name("Typing")
    assert not is_valid_person_name("Coding")
    assert not is_valid_person_name("Running")
    assert not is_valid_person_name("Sleeping")
    assert not is_valid_person_name("Watching")
    assert not is_valid_person_name("Leaning")


def test_rejects_common_objects():
    assert not is_valid_person_name("Monitor")
    assert not is_valid_person_name("Camera")
    assert not is_valid_person_name("Keyboard")
    assert not is_valid_person_name("Computer")


def test_rejects_system_tokens():
    assert not is_valid_person_name("Unknown")
    assert not is_valid_person_name("None")
    assert not is_valid_person_name("Default")
    assert not is_valid_person_name("System")
    assert not is_valid_person_name("Admin")
    assert not is_valid_person_name("Test")
    assert not is_valid_person_name("Guest")


def test_rejects_filler_words():
    assert not is_valid_person_name("Something")
    assert not is_valid_person_name("Nothing")
    assert not is_valid_person_name("Really")
    assert not is_valid_person_name("Hello")
    assert not is_valid_person_name("Maybe")


def test_rejects_jarvis_internal_tokens():
    assert not is_valid_person_name("Jarvis")
    assert not is_valid_person_name("Consciousness")
    assert not is_valid_person_name("Quarantine")


# ── Gerund rule catches unlisted -ing words ─────────────────────────────────

def test_gerund_catch_all():
    """Single-word tokens ending in -ing (>3 chars) are rejected even if not in blocklist."""
    assert not is_valid_person_name("Pondering")
    assert not is_valid_person_name("Snoring")
    assert not is_valid_person_name("Gesturing")
    assert not is_valid_person_name("Squinting")


def test_gerund_rule_does_not_block_real_names():
    """Short -ing words or multi-word names with -ing should not be blocked."""
    assert is_valid_person_name("Ming")
    assert is_valid_person_name("Bing")
    assert is_valid_person_name("King Sterling")


# ── Edge cases ──────────────────────────────────────────────────────────────

def test_rejects_empty_and_none():
    assert not is_valid_person_name("")
    assert not is_valid_person_name("   ")
    assert not is_valid_person_name(None)  # type: ignore


def test_rejects_single_char():
    assert not is_valid_person_name("A")
    assert not is_valid_person_name("X")


def test_rejects_numbers():
    assert not is_valid_person_name("User123")
    assert not is_valid_person_name("42")


def test_rejects_special_chars():
    assert not is_valid_person_name("@david")
    assert not is_valid_person_name("david!")
    assert not is_valid_person_name("test.user")


def test_rejects_very_long_names():
    assert not is_valid_person_name("A" * 50)


# ── Rejection reasons ──────────────────────────────────────────────────────

def test_rejection_reason_blocked_word():
    reason = rejection_reason("Staring")
    assert reason is not None
    assert "blocked" in reason or "gerund" in reason


def test_rejection_reason_valid_name():
    assert rejection_reason("David") is None
    assert rejection_reason("Sarah") is None


def test_rejection_reason_empty():
    assert rejection_reason("") is not None
    assert "empty" in rejection_reason("")


def test_rejection_reason_gerund():
    reason = rejection_reason("Pondering")
    assert reason is not None
    assert "gerund" in reason


# ── Soul integration: get_relationship rejects bad names ─────────────────────

def test_get_relationship_rejects_bad_name():
    from consciousness.soul import IdentityState
    state = IdentityState()
    rel = state.get_relationship("Staring")
    assert rel.name == "unknown"
    assert "staring" not in state.relationships


def test_get_relationship_accepts_good_name():
    """Valid name returns a relationship (transient if not promoted, persistent if promoted)."""
    from consciousness.soul import IdentityState
    state = IdentityState()
    rel = state.get_relationship("David")
    assert rel.name == "David"
    # Without promotion, the relationship is transient (not persisted)
    # This is the correct behavior of the evidence accumulator gating


def test_get_relationship_persists_when_already_known():
    """Pre-existing relationships are always returned."""
    from consciousness.soul import IdentityState, Relationship
    state = IdentityState()
    state.relationships["david"] = Relationship(name="David")
    rel = state.get_relationship("David")
    assert rel.name == "David"
    assert "david" in state.relationships


def test_get_relationship_returns_existing_without_revalidation():
    """Once a relationship exists, get_relationship returns it without re-checking."""
    from consciousness.soul import IdentityState, Relationship
    state = IdentityState()
    state.relationships["david"] = Relationship(name="David")
    rel = state.get_relationship("David")
    assert rel.name == "David"
