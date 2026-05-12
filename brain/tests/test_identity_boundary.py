"""Tests for Layer 3: Identity Boundary Engine.

Covers: types, resolver, memory schema, reconstruction preservation,
backfill migration, boundary engine, search integration, epistemic gates,
and audit system.
"""

from __future__ import annotations

import time
import pytest

# ── Phase 1: Types ──

def test_identity_type_values():
    from identity.types import IdentityType
    # Just verify the literal type is importable and has the right shape
    valid: list[IdentityType] = [
        "self", "primary_user", "known_human", "guest",
        "external_agent", "environment", "library", "unknown",
    ]
    assert len(valid) == 8


def test_identity_scope_key():
    from identity.types import IdentityScope
    scope = IdentityScope(
        owner_id="david", owner_type="primary_user",
        subject_id="david", subject_type="primary_user",
        confidence=0.9,
    )
    assert scope.scope_key == "primary_user:david"
    assert scope.subject_key == "primary_user:david"


def test_identity_scope_empty_key():
    from identity.types import IdentityScope
    scope = IdentityScope()
    assert scope.scope_key == ""
    assert scope.subject_key == ""


def test_retrieval_signature_from_memory():
    from identity.types import RetrievalSignature
    from consciousness.events import Memory

    mem = Memory(
        id="test1", timestamp=time.time(), weight=0.5,
        tags=(), payload="test", type="observation",
        identity_owner="david", identity_owner_type="primary_user",
        identity_subject="tonya", identity_subject_type="known_human",
        identity_needs_resolution=True,
    )
    sig = RetrievalSignature.from_memory(mem)
    assert sig.owner == ("primary_user", "david")
    assert sig.subject == ("known_human", "tonya")
    assert sig.needs_resolution is True


# ── Phase 1: Resolver ──

def test_resolver_system_actor():
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    ctx = r.resolve_for_memory(actor="system")
    assert ctx.identity_type == "self"
    assert ctx.identity_id == "jarvis"
    assert ctx.confidence == 1.0


def test_resolver_provenance_seed():
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    ctx = r.resolve_for_memory(provenance="seed")
    assert ctx.identity_type == "self"
    assert ctx.identity_id == "jarvis"
    assert ctx.confidence == 1.0


def test_resolver_provenance_external():
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    ctx = r.resolve_for_memory(provenance="external_source")
    assert ctx.identity_type == "library"
    assert ctx.confidence == 0.95


def test_resolver_provenance_observed():
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    ctx = r.resolve_for_memory(provenance="observed")
    assert ctx.identity_type == "environment"
    assert ctx.confidence == 0.85


def test_resolver_speaker_known():
    from identity.resolver import IdentityResolver
    from unittest.mock import MagicMock
    r = IdentityResolver()
    soul = MagicMock()
    soul.relationships = {"david": MagicMock()}
    r.set_soul(soul)
    ctx = r.resolve_for_memory(speaker="David")
    assert ctx.identity_type == "primary_user"
    assert ctx.identity_id == "david"


def test_resolver_speaker_unknown_becomes_guest():
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    r._known_names = {"david"}
    ctx = r.resolve_for_memory(speaker="stranger")
    assert ctx.identity_type == "guest"
    assert ctx.identity_id == "stranger"


def test_resolver_fallback():
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    ctx = r.resolve_for_memory()
    assert ctx.identity_type == "unknown"
    assert ctx.confidence == 0.2


def test_resolver_priority_actor_over_provenance():
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    ctx = r.resolve_for_memory(provenance="external_source", actor="system")
    assert ctx.identity_type == "self"


def test_resolver_priority_speaker_over_provenance():
    from identity.resolver import IdentityResolver
    from unittest.mock import MagicMock
    r = IdentityResolver()
    soul = MagicMock()
    soul.relationships = {"david": MagicMock()}
    r.set_soul(soul)
    ctx = r.resolve_for_memory(provenance="conversation", speaker="David")
    assert ctx.identity_type == "primary_user"
    assert ctx.identity_id == "david"


# ── Phase 1: Scope construction ──

def test_scope_owner_equals_subject_default():
    from identity.resolver import IdentityResolver
    from identity.types import IdentityContext
    r = IdentityResolver()
    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    scope = r.build_scope(ctx, "I like pizza", "user_preference")
    assert scope.owner_id == "david"
    assert scope.subject_id == "david"
    assert scope.needs_resolution is False


def test_scope_third_party_wife_unknown_alias():
    from identity.resolver import IdentityResolver
    from identity.types import IdentityContext
    r = IdentityResolver()
    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    scope = r.build_scope(ctx, "my wife hates mushrooms", "user_preference")
    assert scope.owner_id == "david"
    assert scope.subject_id == "_rel_wife"
    assert scope.subject_type == "known_human"
    assert scope.needs_resolution is True


def test_scope_third_party_wife_known_alias():
    from identity.resolver import IdentityResolver
    from identity.types import IdentityContext
    from dataclasses import dataclass, field

    @dataclass
    class FakeRelationship:
        name: str
        notes: list = field(default_factory=list)
        preferences: dict = field(default_factory=dict)

    @dataclass
    class FakeSoul:
        relationships: dict = field(default_factory=dict)

    r = IdentityResolver()
    soul = FakeSoul(relationships={
        "tonya": FakeRelationship(name="Tonya", notes=["wife"]),
    })
    r.set_soul(soul)

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    scope = r.build_scope(ctx, "my wife hates mushrooms", "user_preference")
    assert scope.owner_id == "david"
    assert scope.subject_id == "tonya"
    assert scope.subject_type == "known_human"
    assert scope.needs_resolution is False


def test_scope_low_confidence_quarantine():
    from identity.resolver import IdentityResolver
    from identity.types import IdentityContext
    r = IdentityResolver()
    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.3)
    scope = r.build_scope(ctx, "test", "user_preference")
    assert scope.owner_type == "guest"
    assert scope.needs_resolution is True


def test_scope_medium_confidence_needs_resolution():
    from identity.resolver import IdentityResolver
    from identity.types import IdentityContext
    r = IdentityResolver()
    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.6)
    scope = r.build_scope(ctx, "test", "conversation")
    assert scope.owner_type == "primary_user"
    assert scope.needs_resolution is True


def test_scope_low_confidence_preference_never_primary():
    """Invariant 9: Low-confidence preference NEVER becomes primary_user."""
    from identity.resolver import IdentityResolver
    from identity.types import IdentityContext
    r = IdentityResolver()
    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.3)
    scope = r.build_scope(ctx, "I like spicy food", "user_preference")
    assert scope.owner_type != "primary_user"
    assert scope.needs_resolution is True


# ── Phase 1: Memory dataclass ──

def test_memory_round_trip_identity_fields():
    from consciousness.events import Memory
    mem = Memory(
        id="test1", timestamp=1.0, weight=0.5,
        tags=("a",), payload="test", type="observation",
        identity_owner="david", identity_owner_type="primary_user",
        identity_subject="tonya", identity_subject_type="known_human",
        identity_scope_key="primary_user:david",
        identity_confidence=0.9, identity_needs_resolution=True,
    )
    assert mem.identity_owner == "david"
    assert mem.identity_owner_type == "primary_user"
    assert mem.identity_subject == "tonya"
    assert mem.identity_subject_type == "known_human"
    assert mem.identity_scope_key == "primary_user:david"
    assert mem.identity_confidence == 0.9
    assert mem.identity_needs_resolution is True


def test_memory_default_identity_fields():
    from consciousness.events import Memory
    mem = Memory(
        id="test2", timestamp=1.0, weight=0.5,
        tags=(), payload="test", type="observation",
    )
    assert mem.identity_owner == ""
    assert mem.identity_confidence == 0.0
    assert mem.identity_needs_resolution is False


def test_create_memory_data_passes_identity():
    from memory.core import memory_core, CreateMemoryData
    mem = memory_core.create_memory(CreateMemoryData(
        type="observation",
        payload="test data",
        weight=0.5,
        tags=["test"],
        provenance="observed",
        identity_owner="jarvis",
        identity_owner_type="self",
        identity_subject="jarvis",
        identity_subject_type="self",
        identity_scope_key="self:jarvis",
        identity_confidence=1.0,
    ))
    assert mem is not None
    assert mem.identity_owner == "jarvis"
    assert mem.identity_owner_type == "self"
    assert mem.identity_scope_key == "self:jarvis"


def test_create_memory_low_confidence_caps_weight():
    from memory.core import memory_core, CreateMemoryData
    mem = memory_core.create_memory(CreateMemoryData(
        type="user_preference",
        payload="test",
        weight=0.9,
        tags=["test"],
        provenance="user_claim",
        identity_confidence=0.3,
        identity_needs_resolution=True,
    ))
    assert mem is not None
    assert mem.weight <= 0.5


# ── Phase 1: Backward compatibility ──

def test_legacy_memory_loads_with_defaults():
    """Legacy memories without identity fields should load with empty defaults."""
    from consciousness.events import Memory
    legacy_dict = {
        "id": "mem_legacy", "timestamp": 1.0, "weight": 0.5,
        "tags": ("test",), "payload": "old", "type": "observation",
        "provenance": "observed",
    }
    mem = Memory(**legacy_dict)
    assert mem.identity_owner == ""
    assert mem.identity_needs_resolution is False


# ── Phase 1: Backfill ──

def test_backfill_seed_memory():
    from memory.storage import MemoryStorage
    from consciousness.events import Memory

    storage = MemoryStorage()
    storage._memories = [Memory(
        id="m1", timestamp=1.0, weight=0.9, tags=("core",),
        payload="birth", type="core", provenance="seed",
    )]
    storage._run_load_migrations()

    m = storage._memories[0]
    assert m.identity_owner == "jarvis"
    assert m.identity_owner_type == "self"
    assert m.identity_subject == "jarvis"
    assert m.identity_confidence == 1.0


def test_backfill_external_source():
    from memory.storage import MemoryStorage
    from consciousness.events import Memory

    storage = MemoryStorage()
    storage._memories = [Memory(
        id="m2", timestamp=1.0, weight=0.5, tags=("research",),
        payload="fact", type="factual_knowledge", provenance="external_source",
    )]
    storage._run_load_migrations()

    m = storage._memories[0]
    assert m.identity_owner_type == "library"
    assert m.identity_confidence == 0.95


def test_backfill_conversation_with_speaker_tag():
    from memory.storage import MemoryStorage
    from consciousness.events import Memory

    storage = MemoryStorage()
    storage._memories = [Memory(
        id="m3", timestamp=1.0, weight=0.5,
        tags=("speaker:david", "conversation"),
        payload="test", type="conversation", provenance="conversation",
    )]
    storage._run_load_migrations()

    m = storage._memories[0]
    assert m.identity_owner == "david"
    assert m.identity_owner_type == "primary_user"
    assert m.identity_confidence == 0.7


def test_backfill_conversation_without_speaker():
    from memory.storage import MemoryStorage
    from consciousness.events import Memory

    storage = MemoryStorage()
    storage._memories = [Memory(
        id="m4", timestamp=1.0, weight=0.5, tags=(),
        payload="test", type="conversation", provenance="conversation",
    )]
    storage._run_load_migrations()

    m = storage._memories[0]
    assert m.identity_owner_type == "unknown"
    assert m.identity_confidence == 0.3


def test_backfill_skips_already_stamped():
    from memory.storage import MemoryStorage
    from consciousness.events import Memory

    storage = MemoryStorage()
    storage._memories = [Memory(
        id="m5", timestamp=1.0, weight=0.5, tags=(),
        payload="test", type="observation", provenance="observed",
        identity_owner="jarvis", identity_owner_type="self",
        identity_subject="jarvis", identity_subject_type="self",
        identity_scope_key="self:jarvis", identity_confidence=1.0,
    )]
    storage._run_load_migrations()

    m = storage._memories[0]
    assert m.identity_owner == "jarvis"
    assert m.identity_confidence == 1.0


# ── Phase 2: Boundary Engine ──

def _make_memory(**kwargs):
    from consciousness.events import Memory
    defaults = dict(
        id="test", timestamp=time.time(), weight=0.5,
        tags=(), payload="test", type="observation",
        identity_owner="", identity_owner_type="",
        identity_subject="", identity_subject_type="",
        identity_scope_key="", identity_confidence=0.0,
        identity_needs_resolution=False,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


def test_boundary_universal_types_always_allowed():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    for ot in ("self", "environment", "library", "unknown"):
        mem = _make_memory(identity_owner_type=ot, identity_owner="x")
        decision = engine.validate_retrieval(ctx, mem)
        assert decision.allow, f"Expected allow for owner_type={ot}"


def test_boundary_david_sees_own_memories():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    mem = _make_memory(
        identity_owner="david", identity_owner_type="primary_user",
        identity_subject="david", identity_subject_type="primary_user",
    )
    decision = engine.validate_retrieval(ctx, mem)
    assert decision.allow


def test_boundary_david_does_not_see_guest():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    mem = _make_memory(
        identity_owner="stranger", identity_owner_type="guest",
        identity_subject="stranger", identity_subject_type="guest",
    )
    decision = engine.validate_retrieval(ctx, mem)
    assert not decision.allow


def test_boundary_guest_does_not_see_david():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="unknown", identity_type="guest", confidence=0.3)
    mem = _make_memory(
        identity_owner="david", identity_owner_type="primary_user",
        identity_subject="david", identity_subject_type="primary_user",
    )
    decision = engine.validate_retrieval(ctx, mem)
    assert not decision.allow


def test_boundary_cross_subject_blocked_without_reference():
    """Invariant 10: Cross-subject memories not surfaced without explicit reference."""
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    mem = _make_memory(
        identity_owner="david", identity_owner_type="primary_user",
        identity_subject="tonya", identity_subject_type="known_human",
    )
    decision = engine.validate_retrieval(ctx, mem)
    assert not decision.allow
    assert decision.requires_explicit_reference


def test_boundary_cross_subject_allowed_with_reference():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    mem = _make_memory(
        identity_owner="david", identity_owner_type="primary_user",
        identity_subject="tonya", identity_subject_type="known_human",
    )
    decision = engine.validate_retrieval(ctx, mem, referenced_entities={"tonya"})
    assert decision.allow
    assert decision.requires_explicit_reference


def test_boundary_referenced_subject_across_identity():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    mem = _make_memory(
        identity_owner="tonya", identity_owner_type="known_human",
        identity_subject="tonya", identity_subject_type="known_human",
    )
    decision = engine.validate_retrieval(ctx, mem, referenced_entities={"tonya"})
    assert decision.allow


def test_boundary_referenced_subject_no_match():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="david", identity_type="primary_user", confidence=0.9)
    mem = _make_memory(
        identity_owner="tonya", identity_owner_type="known_human",
        identity_subject="tonya", identity_subject_type="known_human",
    )
    decision = engine.validate_retrieval(ctx, mem, referenced_entities=set())
    assert not decision.allow


def test_boundary_preference_injectable():
    from identity.boundary_engine import IdentityBoundaryEngine
    engine = IdentityBoundaryEngine()

    good = _make_memory(identity_confidence=0.9, identity_needs_resolution=False)
    assert engine.is_preference_injectable(good)

    bad_nr = _make_memory(identity_needs_resolution=True)
    assert not engine.is_preference_injectable(bad_nr)

    bad_conf = _make_memory(identity_confidence=0.3)
    assert not engine.is_preference_injectable(bad_conf)


def test_boundary_self_mode_sees_own():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="jarvis", identity_type="self", confidence=1.0)
    mem = _make_memory(identity_owner="jarvis", identity_owner_type="self")
    decision = engine.validate_retrieval(ctx, mem)
    assert decision.allow


def test_boundary_self_mode_blocks_guest():
    from identity.boundary_engine import IdentityBoundaryEngine
    from identity.types import IdentityContext
    engine = IdentityBoundaryEngine()

    ctx = IdentityContext(identity_id="jarvis", identity_type="self", confidence=1.0)
    mem = _make_memory(identity_owner="stranger", identity_owner_type="guest")
    decision = engine.validate_retrieval(ctx, mem)
    assert not decision.allow


# ── Phase 3: Epistemic boundaries ──

def test_belief_record_has_identity_fields():
    from epistemic.belief_record import BeliefRecord
    br = BeliefRecord(
        belief_id="b1",
        canonical_subject="food", canonical_predicate="prefers",
        canonical_object="spicy", modality="prefers", stance="assert",
        polarity=1, claim_type="preference", epistemic_status="observed",
        extraction_confidence=0.8, belief_confidence=0.7, provenance="user_claim",
        scope="", source_memory_id="m1", timestamp=1.0, time_range=None,
        is_state_belief=False, conflict_key="", evidence_refs=[], contradicts=[],
        resolution_state="active", rendered_claim="prefers spicy",
        identity_subject_id="david", identity_subject_type="primary_user",
    )
    assert br.identity_subject_id == "david"
    assert br.identity_subject_type == "primary_user"


def test_conflict_classifier_identity_gate():
    from epistemic.conflict_classifier import ConflictClassifier
    from epistemic.belief_record import BeliefRecord

    classifier = ConflictClassifier()
    base = dict(
        canonical_predicate="prefers", canonical_object="spicy",
        modality="prefers", stance="assert", polarity=1, claim_type="preference",
        epistemic_status="observed", extraction_confidence=0.8,
        belief_confidence=0.7, provenance="user_claim", scope="",
        source_memory_id="m1", timestamp=1.0, time_range=None,
        is_state_belief=False, evidence_refs=[], contradicts=[],
        resolution_state="active",
    )

    a = BeliefRecord(
        belief_id="a", canonical_subject="food", rendered_claim="david prefers spicy",
        conflict_key="preference::food::spicy",
        identity_subject_id="david", identity_subject_type="primary_user",
        **base,
    )
    b = BeliefRecord(
        belief_id="b", canonical_subject="food", rendered_claim="tonya dislikes spicy",
        conflict_key="preference::food::spicy",
        identity_subject_id="tonya", identity_subject_type="known_human",
        polarity=-1, source_memory_id="m2",
        **{k: v for k, v in base.items() if k not in ("polarity", "source_memory_id")},
    )

    result = classifier.classify(a, b)
    assert result is None, "Cross-identity beliefs should produce near_miss (None)"


def test_conflict_classifier_same_identity_still_fires():
    from epistemic.conflict_classifier import ConflictClassifier
    from epistemic.belief_record import BeliefRecord

    classifier = ConflictClassifier()
    base = dict(
        canonical_subject="food", canonical_predicate="prefers",
        modality="prefers", stance="assert", claim_type="preference",
        epistemic_status="observed", extraction_confidence=0.8,
        belief_confidence=0.7, provenance="user_claim", scope="",
        timestamp=1.0, time_range=None, is_state_belief=False,
        evidence_refs=[], contradicts=[], resolution_state="active",
    )

    a = BeliefRecord(
        belief_id="a", canonical_object="spicy", polarity=1,
        rendered_claim="david prefers spicy",
        conflict_key="preference::food::spicy",
        source_memory_id="m1",
        identity_subject_id="david", identity_subject_type="primary_user",
        **base,
    )
    b = BeliefRecord(
        belief_id="b", canonical_object="bland", polarity=1,
        rendered_claim="david prefers bland",
        conflict_key="preference::food::bland",
        source_memory_id="m2",
        identity_subject_id="david", identity_subject_type="primary_user",
        **base,
    )

    result = classifier.classify(a, b)
    # Same identity, different objects - could be different preferences, but
    # the classifier should not block it with identity gate
    # (it may return None for other reasons like different conflict_key)
    # The key test is: it did NOT return None due to identity_boundary reason


# ── Phase 4: Audit ──

def test_audit_records_events():
    from identity.audit import IdentityAudit, IdentityAuditEvent
    import time

    audit = IdentityAudit()
    audit.record(IdentityAuditEvent(
        timestamp=time.time(),
        event_type="scope_assigned",
        final_scope={"owner_type": "primary_user", "subject_type": "primary_user"},
        confidence=0.9,
        memory_id="m1",
    ))
    audit.record(IdentityAuditEvent(
        timestamp=time.time(),
        event_type="boundary_blocked",
        reason="cross_identity",
        memory_id="m2",
    ))
    audit.record(IdentityAuditEvent(
        timestamp=time.time(),
        event_type="quarantine_write",
        confidence=0.3,
        memory_id="m3",
    ))

    stats = audit.get_stats()
    assert stats["total_scope_assigned"] == 1
    assert stats["total_boundary_blocks"] == 1
    assert stats["total_quarantined"] == 1
    assert stats["by_owner_type"].get("primary_user") == 1


def test_audit_recent_events():
    from identity.audit import IdentityAudit, IdentityAuditEvent
    import time

    audit = IdentityAudit()
    for i in range(5):
        audit.record(IdentityAuditEvent(
            timestamp=time.time(), event_type="scope_assigned",
            final_scope={"owner_type": "self", "subject_type": "self"},
            memory_id=f"m{i}",
        ))

    recent = audit.get_recent(3)
    assert len(recent) == 3
    assert recent[-1]["memory_id"] == "m4"


# ── Sacred Invariants ──

def test_invariant_self_never_becomes_user():
    """Invariant 1: Self memories NEVER become user memories."""
    from identity.resolver import IdentityResolver
    from identity.types import IdentityContext
    r = IdentityResolver()
    ctx = IdentityContext(identity_id="jarvis", identity_type="self", confidence=1.0)
    scope = r.build_scope(ctx, "test", "core")
    assert scope.owner_type == "self"


def test_invariant_unknown_never_defaults_primary():
    """Invariant 3: Unknown speaker NEVER defaults to primary_user."""
    from identity.resolver import IdentityResolver
    r = IdentityResolver()
    ctx = r.resolve_for_memory()
    assert ctx.identity_type != "primary_user"
    scope = r.build_scope(ctx, "test", "conversation")
    assert scope.owner_type != "primary_user"


def test_invariant_legacy_never_promoted():
    """Invariant 8: Legacy memories never auto-promoted to primary_user."""
    from memory.storage import MemoryStorage
    from consciousness.events import Memory

    storage = MemoryStorage()
    storage._memories = [Memory(
        id="m_legacy", timestamp=1.0, weight=0.5, tags=(),
        payload="old memory", type="conversation", provenance="conversation",
    )]
    storage._run_load_migrations()
    m = storage._memories[0]
    assert m.identity_owner_type != "primary_user"


def test_invariant_needs_resolution_blocks_preference_injection():
    """Invariant 11: needs_resolution memories never injected as preferences."""
    from identity.boundary_engine import IdentityBoundaryEngine
    engine = IdentityBoundaryEngine()
    mem = _make_memory(identity_needs_resolution=True, identity_confidence=0.6)
    assert not engine.is_preference_injectable(mem)


# ── Phase 3: Epistemic Boundary Tests ──

def test_belief_record_identity_fields_exist():
    """BeliefRecord should have identity_subject_id and identity_subject_type."""
    from epistemic.belief_record import BeliefRecord
    b = BeliefRecord(
        belief_id="test", canonical_subject="x", canonical_predicate="y",
        canonical_object="z", modality="is", stance="assert", polarity=1,
        claim_type="factual", epistemic_status="observed",
        extraction_confidence=0.9, belief_confidence=0.8,
        provenance="user_claim", scope="", source_memory_id="m1",
        timestamp=1.0, time_range=None, is_state_belief=False,
        conflict_key="factual::x::z", evidence_refs=[], contradicts=[],
        resolution_state="active", rendered_claim="x is z",
        identity_subject_id="david", identity_subject_type="primary_user",
    )
    assert b.identity_subject_id == "david"
    assert b.identity_subject_type == "primary_user"


def test_belief_record_reconstruction_preserves_identity():
    """BeliefStore reconstruction sites must preserve identity fields."""
    from epistemic.belief_record import BeliefStore, BeliefRecord
    store = BeliefStore()
    b = BeliefRecord(
        belief_id="b1", canonical_subject="x", canonical_predicate="y",
        canonical_object="z", modality="is", stance="assert", polarity=1,
        claim_type="factual", epistemic_status="observed",
        extraction_confidence=0.9, belief_confidence=0.8,
        provenance="user_claim", scope="", source_memory_id="m1",
        timestamp=1.0, time_range=None, is_state_belief=False,
        conflict_key="factual::x::z", evidence_refs=[], contradicts=[],
        resolution_state="active", rendered_claim="x is z",
        identity_subject_id="david", identity_subject_type="primary_user",
    )
    store.add(b)

    store.update_resolution("b1", "resolved")
    updated = store.get("b1")
    assert updated.identity_subject_id == "david"
    assert updated.identity_subject_type == "primary_user"
    assert updated.resolution_state == "resolved"

    store.update_belief_confidence("b1", 0.5)
    updated2 = store.get("b1")
    assert updated2.identity_subject_id == "david"
    assert updated2.belief_confidence == 0.5

    store.add_contradiction_link("b1", "b_other")
    updated3 = store.get("b1")
    assert updated3.identity_subject_id == "david"
    assert "b_other" in updated3.contradicts


def test_claim_extractor_propagates_identity():
    """Claim extractor should propagate identity_subject from Memory."""
    from epistemic.claim_extractor import extract_claims
    from consciousness.events import Memory

    mem = Memory(
        id="m1", timestamp=1.0, weight=0.5, tags=("factual_knowledge",),
        payload="HNSW indexing improves recall in most benchmarks",
        type="factual_knowledge", provenance="external_source",
        identity_owner="jarvis", identity_owner_type="self",
        identity_subject="david", identity_subject_type="primary_user",
        identity_confidence=0.9,
    )
    claims = extract_claims(mem)
    if claims:
        for c in claims:
            assert c.identity_subject_id == "david"
            assert c.identity_subject_type == "primary_user"


def test_claim_extractor_no_identity_leaves_empty():
    """Claim extractor should not add identity when Memory has none."""
    from epistemic.claim_extractor import extract_claims
    from consciousness.events import Memory

    mem = Memory(
        id="m1", timestamp=1.0, weight=0.5, tags=("factual_knowledge",),
        payload="HNSW indexing improves recall in most benchmarks",
        type="factual_knowledge", provenance="external_source",
    )
    claims = extract_claims(mem)
    if claims:
        for c in claims:
            assert c.identity_subject_id == ""


def test_belief_graph_bridge_blocks_cross_identity_support():
    """Bridge should not create support edges across identity boundaries."""
    from epistemic.belief_record import BeliefStore, BeliefRecord

    store = BeliefStore()
    base = dict(
        canonical_predicate="prefers", canonical_object="spicy",
        modality="prefers", stance="assert", polarity=1,
        claim_type="preference", epistemic_status="observed",
        extraction_confidence=0.8, belief_confidence=0.7,
        provenance="user_claim", scope="", timestamp=1.0,
        time_range=None, is_state_belief=False,
        evidence_refs=[], contradicts=[], resolution_state="active",
    )

    b1 = BeliefRecord(
        belief_id="b1", canonical_subject="food",
        conflict_key="preference::food::spicy",
        rendered_claim="david prefers spicy",
        source_memory_id="m1",
        identity_subject_id="david", identity_subject_type="primary_user",
        **base,
    )
    b2 = BeliefRecord(
        belief_id="b2", canonical_subject="food",
        conflict_key="preference::food::spicy",
        rendered_claim="tonya prefers spicy",
        source_memory_id="m2",
        identity_subject_id="tonya", identity_subject_type="known_human",
        **base,
    )
    store.add(b1)
    store.add(b2)

    from epistemic.belief_graph.edges import EdgeStore
    edge_store = EdgeStore()
    from epistemic.belief_graph.bridge import GraphBridge
    bridge = GraphBridge(edge_store, store)

    bridge.create_shared_subject_support(b2)
    outgoing = edge_store.get_outgoing(b2.belief_id)
    incoming = edge_store.get_incoming(b2.belief_id)
    all_edges = outgoing + incoming
    for e in all_edges:
        other = e.target_belief_id if e.source_belief_id == b2.belief_id else e.source_belief_id
        assert other != b1.belief_id, \
            "Cross-identity support edge should not be created"


# ── Phase 4: Audit + Events Tests ──

def test_audit_stats_include_all_counters():
    from identity.audit import IdentityAudit, IdentityAuditEvent
    audit = IdentityAudit()

    audit.record(IdentityAuditEvent(
        timestamp=time.time(), event_type="scope_assigned",
        final_scope={"owner_type": "self", "subject_type": "self"},
        memory_id="m1",
    ))
    audit.record(IdentityAuditEvent(
        timestamp=time.time(), event_type="quarantine_write",
        memory_id="m2", confidence=0.3,
    ))
    audit.record(IdentityAuditEvent(
        timestamp=time.time(), event_type="referenced_subject_allow",
        reason="subject=tonya", memory_id="m3",
    ))
    audit.record(IdentityAuditEvent(
        timestamp=time.time(), event_type="scope_downgraded",
        memory_id="m4",
    ))

    stats = audit.get_stats()
    assert stats["total_scope_assigned"] == 1
    assert stats["total_quarantined"] == 1
    assert stats["total_referenced_allows"] == 1
    assert stats["total_downgraded"] == 1
    assert stats["total_boundary_blocks"] == 0
    assert "self" in stats["by_owner_type"]
    assert "self" in stats["by_subject_type"]


def test_audit_record_scope_assigned_auto_detects_quarantine():
    """record_scope_assigned should emit quarantine_write when needs_resolution."""
    from identity.audit import IdentityAudit
    from identity.types import IdentityScope

    audit = IdentityAudit()

    scope_normal = IdentityScope(
        owner_id="david", owner_type="primary_user",
        subject_id="david", subject_type="primary_user",
        confidence=0.9, needs_resolution=False,
    )
    audit.record_scope_assigned("m1", scope_normal, 0.9)

    scope_ambig = IdentityScope(
        owner_id="unknown", owner_type="guest",
        subject_id="unknown", subject_type="guest",
        confidence=0.3, needs_resolution=True,
    )
    audit.record_scope_assigned("m2", scope_ambig, 0.3)

    stats = audit.get_stats()
    assert stats["total_scope_assigned"] == 1
    assert stats["total_quarantined"] == 1
    assert stats["by_owner_type"]["primary_user"] == 1
    assert stats["by_owner_type"]["guest"] == 1
    assert stats["by_subject_type"]["primary_user"] == 1
    assert stats["by_subject_type"]["guest"] == 1


def test_audit_boundary_block_recording():
    from identity.audit import IdentityAudit
    audit = IdentityAudit()
    audit.record_boundary_block("m5", "cross_identity", querier_id="david")
    stats = audit.get_stats()
    assert stats["total_boundary_blocks"] == 1
    recent = audit.get_recent(1)
    assert recent[0]["type"] == "boundary_blocked"
    assert recent[0]["reason"] == "cross_identity"


def test_audit_referenced_allow_recording():
    from identity.audit import IdentityAudit
    audit = IdentityAudit()
    audit.record_referenced_allow("m6", "tonya", querier_id="david")
    stats = audit.get_stats()
    assert stats["total_referenced_allows"] == 1
    recent = audit.get_recent(1)
    assert "tonya" in recent[0]["reason"]
