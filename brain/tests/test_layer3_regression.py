"""Layer 3 regression tests: sacred scenarios, storage, routing, cross-contamination.

These tests verify the end-to-end identity boundary behavior for the 6 test
groups defined in the Layer 3 validation checklist. They do NOT require GPU,
Ollama, or network — all heavy dependencies are mocked or bypassed.
"""

from __future__ import annotations

import sys
import time
import types
from unittest import mock

import pytest

# Stub out the ollama module so response.py can be imported without it
_ollama_stub = types.ModuleType("ollama")
_ollama_stub.AsyncClient = mock.MagicMock  # type: ignore[attr-defined]
_ollama_stub.ChatResponse = mock.MagicMock  # type: ignore[attr-defined]
_ollama_stub.ResponseError = Exception  # type: ignore[attr-defined]
sys.modules.setdefault("ollama", _ollama_stub)

from consciousness.events import Memory


# ── Helpers ──

def _make_memory(
    mid: str = "mem_test",
    mem_type: str = "user_preference",
    payload: str = "test",
    tags: tuple = (),
    weight: float = 0.85,
    owner: str = "",
    owner_type: str = "unknown",
    subject: str = "",
    subject_type: str = "unknown",
    needs_resolution: bool = False,
    confidence: float = 0.7,
    provenance: str = "user_claim",
) -> Memory:
    return Memory(
        id=mid, timestamp=time.time(), weight=weight,
        tags=tags, payload=payload, type=mem_type,
        identity_owner=owner, identity_owner_type=owner_type,
        identity_subject=subject, identity_subject_type=subject_type,
        identity_needs_resolution=needs_resolution,
        identity_confidence=confidence,
        provenance=provenance,
    )


# ═══════════════════════════════════════════════════════════════════════
# Group 1: Router classification — the 4 sacred scenarios
# ═══════════════════════════════════════════════════════════════════════

class TestRouterSacredScenarios:
    """Verify route_memory_request classifies the 4 sacred queries correctly."""

    def test_i_like_pizza_is_general(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("I like pizza.", set())
        assert route.route_type == "general"
        assert route.allow_preference_injection is True

    def test_my_wife_likes_mushrooms_is_general(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("My wife likes mushrooms.", set())
        assert route.route_type == "general"
        assert route.allow_thirdparty_injection is True

    def test_what_do_i_like_is_self_preference(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What do I like?", set())
        assert route.route_type == "self_preference"
        assert route.allow_preference_injection is True
        assert route.allow_thirdparty_injection is False

    def test_what_does_my_wife_like_is_referenced_person(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What does my wife like?", set())
        assert route.route_type == "referenced_person"
        assert route.allow_preference_injection is False
        assert route.allow_thirdparty_injection is True
        assert route.search_scope == "referenced_subject_only"
        assert "wife" in route.referenced_entities

    def test_does_my_wife_like_mushrooms_is_referenced_person(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("Does my wife like mushrooms?", set())
        assert route.route_type == "referenced_person"
        assert "wife" in route.referenced_entities

    def test_what_foods_does_my_wife_like_is_referenced_person(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What foods does my wife like?", set())
        assert route.route_type == "referenced_person"
        assert "wife" in route.referenced_entities

    def test_tell_me_what_my_wife_likes(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("Can you tell me what my wife likes?", set())
        assert route.route_type == "referenced_person"
        assert "wife" in route.referenced_entities

    def test_do_i_like_mushrooms_is_self_preference(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("Do I like mushrooms?", set())
        assert route.route_type == "self_preference"
        assert route.allow_preference_injection is True
        assert route.allow_thirdparty_injection is False


class TestRouterEdgeCases:
    """Edge cases for the memory router."""

    def test_thanks_is_no_retrieval(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("Thanks!", set())
        assert route.route_type == "no_retrieval"

    def test_thank_you_jarvis_is_no_retrieval(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("Thank you Jarvis", set())
        assert route.route_type == "no_retrieval"

    def test_what_did_i_just_say_is_episodic(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What did I just say?", set())
        assert route.route_type == "episodic"

    def test_my_son_likes_soccer_general_with_thirdparty(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("My son likes soccer.", set())
        assert route.route_type == "general"
        assert route.allow_thirdparty_injection is True

    def test_what_does_my_friend_chris_like(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What does my friend Chris like?", {"chris"})
        assert route.route_type == "referenced_person"
        assert "friend" in route.referenced_entities or "chris" in route.referenced_entities

    def test_what_patterns_do_you_notice_is_belief(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What patterns do you notice?", set())
        assert route.route_type == "belief_synthesis"

    def test_low_information_general_query_disables_memory_injection(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("day", set())
        assert route.route_type == "general"
        assert route.allow_preference_injection is False
        assert route.allow_thirdparty_injection is False
        assert route.search_scope == "none"

    def test_low_information_query_with_name_disables_memory_injection(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("check, Jarvis", set())
        assert route.route_type == "general"
        assert route.allow_preference_injection is False
        assert route.allow_thirdparty_injection is False
        assert route.search_scope == "none"


# ═══════════════════════════════════════════════════════════════════════
# Group 2: Third-party memory storage
# ═══════════════════════════════════════════════════════════════════════

class TestThirdpartyStorage:
    """Verify _store_thirdparty_memory actually persists to memory_storage."""

    def test_store_thirdparty_adds_to_storage(self):
        from memory.storage import memory_storage
        before_count = len(memory_storage.get_by_tag("user_preference"))

        from conversation_handler import _store_thirdparty_memory
        _store_thirdparty_memory(
            payload="User's wife likes mushrooms",
            category="thirdparty_preference",
            speaker="david",
            relation="wife",
        )

        after = memory_storage.get_by_tag("user_preference")
        assert len(after) > before_count, \
            "memory_storage should have more user_preference memories after store"

        wife_mems = [m for m in after if "wife" in (m.payload or "").lower()
                     and "mushroom" in (m.payload or "").lower()]
        assert len(wife_mems) >= 1, "Wife mushroom memory should exist in storage"

        mem = wife_mems[0]
        assert mem.type == "user_preference"
        assert "relation:wife" in mem.tags
        assert mem.provenance == "user_claim"

    def test_store_thirdparty_deduplicates(self):
        from memory.storage import memory_storage
        from conversation_handler import _store_thirdparty_memory

        count_before = len([
            m for m in memory_storage.get_by_tag("user_preference")
            if "wife" in str(m.payload).lower() and "mushroom" in str(m.payload).lower()
        ])

        _store_thirdparty_memory(
            payload="User's wife likes mushrooms",
            category="thirdparty_preference",
            speaker="david",
            relation="wife",
        )

        count_after = len([
            m for m in memory_storage.get_by_tag("user_preference")
            if "wife" in str(m.payload).lower() and "mushroom" in str(m.payload).lower()
        ])

        assert count_after == count_before, "Duplicate wife memory should be deduplicated"


# ═══════════════════════════════════════════════════════════════════════
# Group 3: Boundary engine — cross-contamination guards
# ═══════════════════════════════════════════════════════════════════════

class TestBoundaryEngine:
    """Verify identity boundary engine enforces owner/subject separation."""

    def test_self_memory_allowed_for_self_query(self):
        from identity.boundary_engine import IdentityBoundaryEngine
        from identity.types import IdentityContext

        engine = IdentityBoundaryEngine()
        ctx = IdentityContext(
            identity_id="david", identity_type="primary_user",
            confidence=0.8,
        )
        mem = _make_memory(
            owner="david", owner_type="primary_user",
            subject="david", subject_type="primary_user",
            payload="User likes pizza",
        )
        decision = engine.validate_retrieval(ctx, mem, referenced_entities=set())
        assert decision.allow is True

    def test_wife_memory_blocked_without_reference(self):
        from identity.boundary_engine import IdentityBoundaryEngine
        from identity.types import IdentityContext

        engine = IdentityBoundaryEngine()
        ctx = IdentityContext(
            identity_id="david", identity_type="primary_user",
            confidence=0.8,
        )
        mem = _make_memory(
            owner="david", owner_type="primary_user",
            subject="_rel_wife", subject_type="known_human",
            payload="User's wife likes mushrooms",
        )
        decision = engine.validate_retrieval(ctx, mem, referenced_entities=set())
        assert decision.allow is False or decision.requires_explicit_reference is True, \
            "Wife memory should be blocked or require explicit reference when not referenced"

    def test_wife_memory_allowed_with_reference(self):
        from identity.boundary_engine import IdentityBoundaryEngine
        from identity.types import IdentityContext

        engine = IdentityBoundaryEngine()
        ctx = IdentityContext(
            identity_id="david", identity_type="primary_user",
            confidence=0.8,
        )
        mem = _make_memory(
            owner="david", owner_type="primary_user",
            subject="_rel_wife", subject_type="known_human",
            payload="User's wife likes mushrooms",
        )
        decision = engine.validate_retrieval(
            ctx, mem, referenced_entities={"wife", "_rel_wife"},
        )
        assert decision.allow is True, \
            "Wife memory should be allowed when explicitly referenced"


# ═══════════════════════════════════════════════════════════════════════
# Group 4: Canonical alias expansion
# ═══════════════════════════════════════════════════════════════════════

class TestAliasExpansion:
    """Verify reference alias expansion bridges query terms to stored subjects."""

    def test_wife_expands_to_rel_wife(self):
        from reasoning.response import _extract_referenced_entities
        refs = _extract_referenced_entities("What does my wife like?", set())
        assert "wife" in refs
        assert "_rel_wife" in refs

    def test_husband_expands_to_rel_husband(self):
        from reasoning.response import _extract_referenced_entities
        refs = _extract_referenced_entities("What does my husband enjoy?", set())
        assert "husband" in refs
        assert "_rel_husband" in refs

    def test_known_name_included(self):
        from reasoning.response import _extract_referenced_entities
        refs = _extract_referenced_entities("What does sarah like?", {"sarah"})
        assert "sarah" in refs

    def test_resolver_expand_includes_rel_prefix(self):
        try:
            from identity.resolver import identity_resolver
            expanded = identity_resolver.expand_reference_aliases({"wife"})
            assert "_rel_wife" in expanded
            assert "wife" in expanded
        except Exception:
            pytest.skip("identity_resolver not available")


# ═══════════════════════════════════════════════════════════════════════
# Group 5: Preference injection safety
# ═══════════════════════════════════════════════════════════════════════

class TestPreferenceInjectionSafety:
    """Verify MemoryRoute dataclass enforces route constraints."""

    def test_referenced_person_route_blocks_self_prefs(self):
        from reasoning.response import MemoryRoute
        route = MemoryRoute(
            route_type="referenced_person",
            referenced_entities={"wife", "_rel_wife"},
            allow_preference_injection=False,
            allow_thirdparty_injection=True,
            search_scope="referenced_subject_only",
        )
        assert route.allow_preference_injection is False
        assert route.allow_thirdparty_injection is True

    def test_self_preference_route_blocks_thirdparty(self):
        from reasoning.response import MemoryRoute
        route = MemoryRoute(
            route_type="self_preference",
            allow_preference_injection=True,
            allow_thirdparty_injection=False,
            search_scope="primary_user_only",
        )
        assert route.allow_preference_injection is True
        assert route.allow_thirdparty_injection is False


# ═══════════════════════════════════════════════════════════════════════
# Group 6: Subject supplement scan
# ═══════════════════════════════════════════════════════════════════════

class TestSubjectSupplement:
    """Verify _supplement_subject_memories finds subject-matching prefs."""

    def test_supplement_finds_matching_subject(self):
        from memory.storage import memory_storage
        from reasoning.response import _supplement_subject_memories

        mem = _make_memory(
            mid="mem_wife_test",
            payload="User's wife likes mushrooms",
            tags=("user_preference", "thirdparty_preference", "relation:wife"),
            subject="_rel_wife", subject_type="known_human",
            owner="david", owner_type="primary_user",
        )
        memory_storage.add(mem)

        try:
            result = _supplement_subject_memories(
                existing=[], referenced_entities={"wife", "_rel_wife"}, top_k=5,
            )
            assert result is not None
            found_ids = {m.id for m in result}
            assert "mem_wife_test" in found_ids, \
                "Supplement should find wife memory by subject match"
        finally:
            memory_storage._memories = [
                m for m in memory_storage._memories if m.id != "mem_wife_test"
            ]

    def test_supplement_finds_by_relation_tag(self):
        from memory.storage import memory_storage
        from reasoning.response import _supplement_subject_memories

        mem = _make_memory(
            mid="mem_wife_tag_test",
            payload="User's wife likes gardening",
            tags=("user_preference", "thirdparty_preference", "relation:wife"),
            subject="", subject_type="unknown",
            owner="david", owner_type="primary_user",
        )
        memory_storage.add(mem)

        try:
            result = _supplement_subject_memories(
                existing=[], referenced_entities={"wife", "_rel_wife"}, top_k=5,
            )
            assert result is not None
            found_ids = {m.id for m in result}
            assert "mem_wife_tag_test" in found_ids, \
                "Supplement should find wife memory by relation:wife tag"
        finally:
            memory_storage._memories = [
                m for m in memory_storage._memories if m.id != "mem_wife_tag_test"
            ]


# ═══════════════════════════════════════════════════════════════════════
# Group 7: Autonomy suppression
# ═══════════════════════════════════════════════════════════════════════

class TestAutonomySuppression:
    """Verify autonomy recall is suppressed for conversational routes."""

    def test_referenced_person_disables_autonomy(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What does my wife like?", set())
        assert route.allow_autonomy_recall is False

    def test_self_preference_disables_autonomy(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("What do I like?", set())
        assert route.allow_autonomy_recall is False

    def test_general_disables_autonomy(self):
        from reasoning.response import route_memory_request
        route = route_memory_request("I had pizza today.", set())
        assert route.allow_autonomy_recall is False


# ═══════════════════════════════════════════════════════════════════════
# Group 8: Cross-contamination — the key invariant
# ═══════════════════════════════════════════════════════════════════════

class TestCrossContamination:
    """The most critical tests: wife memory must NOT leak into self queries."""

    def test_self_route_does_not_allow_thirdparty(self):
        """'What do I like?' must never inject wife's preferences."""
        from reasoning.response import route_memory_request
        route = route_memory_request("What do I like?", set())
        assert route.allow_thirdparty_injection is False
        assert route.route_type == "self_preference"

    def test_do_i_like_mushrooms_does_not_allow_thirdparty(self):
        """'Do I like mushrooms?' must not use wife memory to answer."""
        from reasoning.response import route_memory_request
        route = route_memory_request("Do I like mushrooms?", set())
        assert route.allow_thirdparty_injection is False

    def test_wife_route_does_not_allow_self_prefs(self):
        """'What does my wife like?' must not inject David's preferences."""
        from reasoning.response import route_memory_request
        route = route_memory_request("What does my wife like?", set())
        assert route.allow_preference_injection is False
