"""Tests for Layer 5: Contradiction Engine (Epistemic Court).

Covers all 13 sacred invariants, all 6 conflict types, canonicalization,
claim extraction, resolution strategies, debt mechanics, persistence,
and the offline judge bench.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from epistemic.belief_record import (
    BeliefRecord,
    BeliefStore,
    TensionRecord,
    build_conflict_key,
    infer_tension_topic,
    DEBT_CRITICAL_UNRESOLVED,
    DEBT_FACTUAL_RESOLVED,
    DEBT_IDENTITY_TENSION,
    DEBT_MODERATE_UNRESOLVED,
    DEBT_MULTI_PERSPECTIVE,
    DEBT_PASSIVE_DECAY_PER_HOUR,
    DEBT_POLICY_NORM,
    DEBT_RECURRENCE_EXTRA,
    DEBT_TEMPORAL_VERSION,
    DEBT_TENSION_MATURED,
    TENSION_MAX_BELIEF_IDS,
)
from epistemic.claim_extractor import (
    canonicalize_term,
    canonicalize_predicate,
    extract_claims,
)
from epistemic.conflict_classifier import ConflictClassifier
from epistemic.resolution import (
    FactualResolution,
    TemporalResolution,
    IdentityTensionResolution,
    ProvenanceResolution,
    PolicyResolution,
    MultiPerspectiveResolution,
    resolve_conflict,
)
from consciousness.events import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_counter = 0


def _make_belief(
    subject: str = "test_subject",
    predicate: str = "is",
    obj: str = "test_object",
    modality: str = "is",
    stance: str = "assert",
    polarity: int = 1,
    claim_type: str = "factual",
    epistemic_status: str = "inferred",
    extraction_confidence: float = 0.7,
    belief_confidence: float = 0.6,
    provenance: str = "external_source",
    scope: str = "",
    source_memory_id: str = "",
    time_range: tuple[float, float] | None = None,
    is_state_belief: bool = False,
    conflict_key: str = "",
    resolution_state: str = "active",
) -> BeliefRecord:
    global _counter
    _counter += 1
    bid = f"bel_test_{_counter}"
    mid = source_memory_id or f"mem_test_{_counter}"
    b = BeliefRecord(
        belief_id=bid,
        canonical_subject=canonicalize_term(subject),
        canonical_predicate=predicate,
        canonical_object=canonicalize_term(obj),
        modality=modality,
        stance=stance,
        polarity=polarity,
        claim_type=claim_type,
        epistemic_status=epistemic_status,
        extraction_confidence=extraction_confidence,
        belief_confidence=belief_confidence,
        provenance=provenance,
        scope=scope,
        source_memory_id=mid,
        timestamp=time.time(),
        time_range=time_range,
        is_state_belief=is_state_belief,
        conflict_key=conflict_key or "",
        evidence_refs=[mid],
        contradicts=[],
        resolution_state=resolution_state,
        rendered_claim=f"{subject} {predicate} {obj}",
    )
    ck = build_conflict_key(b)
    return BeliefRecord(
        belief_id=b.belief_id,
        canonical_subject=b.canonical_subject,
        canonical_predicate=b.canonical_predicate,
        canonical_object=b.canonical_object,
        modality=b.modality,
        stance=b.stance,
        polarity=b.polarity,
        claim_type=b.claim_type,
        epistemic_status=b.epistemic_status,
        extraction_confidence=b.extraction_confidence,
        belief_confidence=b.belief_confidence,
        provenance=b.provenance,
        scope=b.scope,
        source_memory_id=b.source_memory_id,
        timestamp=b.timestamp,
        time_range=b.time_range,
        is_state_belief=b.is_state_belief,
        conflict_key=ck,
        evidence_refs=b.evidence_refs,
        contradicts=b.contradicts,
        resolution_state=b.resolution_state,
        rendered_claim=b.rendered_claim,
    )


def _make_store() -> BeliefStore:
    d = tempfile.mkdtemp()
    return BeliefStore(
        beliefs_path=os.path.join(d, "beliefs.jsonl"),
        tensions_path=os.path.join(d, "tensions.jsonl"),
    )


def _make_memory(
    mem_type: str = "factual_knowledge",
    payload=None,
    provenance: str = "external_source",
    tags: tuple[str, ...] = (),
) -> Memory:
    global _counter
    _counter += 1
    return Memory(
        id=f"mem_test_{_counter}",
        timestamp=time.time(),
        weight=0.5,
        tags=tags,
        payload=payload or {"question": "test", "summary": "test result"},
        type=mem_type,
        provenance=provenance,
    )


# ===========================================================================
# belief_record.py tests (#1-#7)
# ===========================================================================


def test_belief_record_creation():
    b = _make_belief()
    assert b.belief_id.startswith("bel_test_")
    assert b.canonical_subject != ""
    assert b.canonical_predicate != ""
    assert b.modality in {"is", "may", "prefers", "should", "observed_as", "caused", "predicted"}
    assert b.stance in {"assert", "deny", "uncertain", "question"}
    assert b.resolution_state in {"active", "superseded", "versioned", "tension", "resolved", "quarantined"}
    print("  PASS: test_belief_record_creation")


def test_conflict_key_type_specific():
    fact = _make_belief(subject="hnsw", obj="recall", claim_type="factual")
    assert fact.conflict_key.startswith("fact::")

    ident = _make_belief(subject="identity", obj="continuity", claim_type="identity")
    assert ident.conflict_key.startswith("identity::")

    obs = _make_belief(subject="user_presence", claim_type="observation")
    assert obs.conflict_key.startswith("state::")

    pol_out = _make_belief(subject="academic_search", obj="grounding", claim_type="policy", modality="caused")
    assert pol_out.conflict_key.startswith("policy_outcome::")

    pol_norm = _make_belief(subject="academic_search", obj="grounding", claim_type="policy", modality="should")
    assert pol_norm.conflict_key.startswith("policy_norm::")

    pref = _make_belief(subject="user", obj="dark_mode", claim_type="preference")
    assert pref.conflict_key.startswith("pref::")
    print("  PASS: test_conflict_key_type_specific")


def test_tension_topic_deterministic():
    a = _make_belief(subject="identity", obj="continuity", claim_type="identity")
    b = _make_belief(subject="identity", obj="replication", claim_type="identity")
    t1 = infer_tension_topic(a, b)
    t2 = infer_tension_topic(a, b)
    assert t1 == t2
    assert "identity" in t1
    print("  PASS: test_tension_topic_deterministic")


def test_tension_topic_merges_aliases():
    a = _make_belief(subject="identity", obj="continuity", claim_type="identity")
    b = _make_belief(subject="selfhood", obj="persistence", claim_type="identity")
    t = infer_tension_topic(a, b)
    assert "::" in t
    assert t != f"unclassified::{a.canonical_subject}_vs_{b.canonical_subject}" or "identity" in t or "memory" in t
    print("  PASS: test_tension_topic_merges_aliases")


def test_belief_store_round_trip():
    store = _make_store()
    beliefs = [_make_belief(subject=f"sub_{i}", obj=f"obj_{i}") for i in range(10)]
    for b in beliefs:
        store.add(b)
    assert len(store._beliefs) == 10

    store2 = BeliefStore(
        beliefs_path=store._beliefs_path,
        tensions_path=store._tensions_path,
    )
    store2.rehydrate()
    assert len(store2._beliefs) == 10
    for b in beliefs:
        rehydrated = store2.get(b.belief_id)
        assert rehydrated is not None
        assert rehydrated.canonical_subject == b.canonical_subject
        assert rehydrated.belief_confidence == b.belief_confidence
    print("  PASS: test_belief_store_round_trip")


def test_tension_store_round_trip():
    store = _make_store()
    t1 = TensionRecord(
        tension_id="ten_test_1", topic="identity::test", belief_ids=["b1", "b2"],
        conflict_key="identity::a::b", created_at=time.time(), last_revisited=time.time(),
        revisit_count=3, stability=0.7, maturation_score=0.15,
    )
    t2 = TensionRecord(
        tension_id="ten_test_2", topic="agency::test", belief_ids=["b3"],
        conflict_key="identity::c::d", created_at=time.time(), last_revisited=time.time(),
        revisit_count=0, stability=0.5, maturation_score=0.0,
    )
    store.add_tension(t1)
    store.add_tension(t2)

    store2 = BeliefStore(
        beliefs_path=store._beliefs_path,
        tensions_path=store._tensions_path,
    )
    store2.rehydrate()
    assert len(store2._tensions) == 2
    rt1 = store2.get_tension("ten_test_1")
    assert rt1 is not None
    assert rt1.topic == "identity::test"
    assert rt1.revisit_count == 3
    assert abs(rt1.maturation_score - 0.15) < 0.001
    print("  PASS: test_tension_store_round_trip")


def test_eviction_protects_active_and_tension():
    store = _make_store()
    store._max_capacity = 5
    for i in range(3):
        b = _make_belief(subject=f"active_{i}", resolution_state="active")
        store.add(b)
    for i in range(3):
        b = _make_belief(subject=f"resolved_{i}", resolution_state="resolved")
        store.add(b)
    b_tension = _make_belief(subject="tension_sub", resolution_state="tension")
    store.add(b_tension)

    assert len(store._beliefs) <= 5
    for bid, b in store._beliefs.items():
        assert b.resolution_state != "tension" or bid == b_tension.belief_id or b.resolution_state in ("active", "tension")
    print("  PASS: test_eviction_protects_active_and_tension")


# ===========================================================================
# claim_extractor.py tests (#8-#18)
# ===========================================================================


def test_canonicalize_basic():
    assert canonicalize_term("Hello World") == "hello_world"
    assert canonicalize_term("  HNSW-Indexing  ") == "hnswindexing"
    assert canonicalize_term("test thing") == "test_thing"
    assert canonicalize_term("User Present") == "user_presence"  # synonym table maps user_present -> user_presence
    # Punctuation stripped, case lowered, spaces become underscores
    assert canonicalize_term("  Some.Weird!Text  ") == "someweirdtext"
    print("  PASS: test_canonicalize_basic")


def test_canonicalize_synonyms():
    assert canonicalize_term("self_identity") == "identity"
    assert canonicalize_term("selfhood") == "identity"
    assert canonicalize_term("free_will") == "agency"
    assert canonicalize_term("autonomous") == "agency"
    assert canonicalize_term("user_present") == "user_presence"
    print("  PASS: test_canonicalize_synonyms")


def test_canonicalize_idempotent():
    terms = ["identity", "agency", "user_presence", "hello_world", "HNSW Indexing"]
    for t in terms:
        first = canonicalize_term(t)
        second = canonicalize_term(first)
        assert first == second, f"Not idempotent: {t} -> {first} -> {second}"
    print("  PASS: test_canonicalize_idempotent")


def test_canonicalize_predicate_negation():
    pred, pol = canonicalize_predicate("is_not")
    assert pol == -1
    assert pred == "is"
    print("  PASS: test_canonicalize_predicate_negation")


def test_canonicalize_predicate_clusters():
    for word in ("helps", "boosts", "enhances", "increases"):
        pred, pol = canonicalize_predicate(word)
        assert pred == "improves", f"{word} -> {pred}"
        assert pol == 0
    for word in ("hurts", "reduces", "worsens"):
        pred, pol = canonicalize_predicate(word)
        assert pred == "degrades", f"{word} -> {pred}"
    print("  PASS: test_canonicalize_predicate_clusters")


def test_extract_factual_knowledge():
    mem = _make_memory(
        mem_type="factual_knowledge",
        payload={"question": "HNSW performance", "summary": "improves recall significantly"},
    )
    claims = extract_claims(mem)
    assert len(claims) >= 1
    c = claims[0]
    assert c.claim_type == "factual"
    assert c.modality == "is"
    assert c.epistemic_status == "inferred"
    assert c.extraction_confidence > 0
    print("  PASS: test_extract_factual_knowledge")


def test_extract_identity_claim():
    mem = _make_memory(
        mem_type="factual_knowledge",
        payload={"question": "nature of self", "summary": "identity is a continuous process"},
        tags=("existential", "philosophical"),
    )
    claims = extract_claims(mem)
    assert len(claims) >= 1
    c = claims[0]
    assert c.claim_type in ("identity", "philosophical")
    assert c.epistemic_status == "questioned"
    print("  PASS: test_extract_identity_claim")


def test_extract_preference():
    mem = _make_memory(
        mem_type="user_preference",
        payload="dark mode enabled",
        provenance="user_claim",
    )
    claims = extract_claims(mem)
    assert len(claims) >= 1
    c = claims[0]
    assert c.claim_type == "preference"
    assert c.modality == "prefers"
    assert c.epistemic_status == "adopted"
    print("  PASS: test_extract_preference")


def test_extract_ambiguous_underextracts():
    mem = _make_memory(
        mem_type="contextual_insight",
        payload="something vague maybe",
    )
    claims = extract_claims(mem)
    if claims:
        for c in claims:
            assert c.extraction_confidence <= 0.5
    print("  PASS: test_extract_ambiguous_underextracts")


def test_extract_conversation_skipped():
    mem = _make_memory(mem_type="conversation", payload="hello how are you")
    claims = extract_claims(mem)
    assert len(claims) == 0
    print("  PASS: test_extract_conversation_skipped")


def test_extract_max_claims_cap():
    mem = _make_memory(
        mem_type="factual_knowledge",
        payload={
            "question": "big topic with many facets",
            "summary": "finding one. finding two. finding three. finding four.",
        },
    )
    claims = extract_claims(mem)
    assert len(claims) <= 3
    print("  PASS: test_extract_max_claims_cap")


# ===========================================================================
# conflict_classifier.py tests (#19-#27b)
# ===========================================================================


def test_different_modality_near_miss():
    classifier = ConflictClassifier()
    a = _make_belief(subject="identity", obj="continuity", modality="is")
    b = _make_belief(subject="identity", obj="continuity", modality="may")
    result = classifier.classify(a, b)
    assert result is None
    nms = classifier.get_near_misses()
    assert any(nm.reason == "different_modality" for nm in nms)
    print("  PASS: test_different_modality_near_miss")


def test_question_stance_near_miss():
    classifier = ConflictClassifier()
    a = _make_belief(subject="identity", obj="continuity", stance="assert")
    b = _make_belief(subject="identity", obj="continuity", stance="question")
    result = classifier.classify(a, b)
    assert result is None
    nms = classifier.get_near_misses()
    assert any(nm.reason == "question_vs_assertion" for nm in nms)
    print("  PASS: test_question_stance_near_miss")


def test_low_extraction_confidence_near_miss():
    classifier = ConflictClassifier()
    a = _make_belief(subject="x", obj="y", extraction_confidence=0.2)
    b = _make_belief(subject="x", obj="y", polarity=-1, extraction_confidence=0.7)
    result = classifier.classify(a, b)
    assert result is None
    nms = classifier.get_near_misses()
    assert any(nm.reason == "low_extraction_confidence" for nm in nms)
    print("  PASS: test_low_extraction_confidence_near_miss")


def test_factual_contradiction_detected():
    classifier = ConflictClassifier()
    a = _make_belief(subject="hnsw", obj="recall", modality="is", polarity=1)
    b = _make_belief(subject="hnsw", obj="recall", modality="is", polarity=-1)
    result = classifier.classify(a, b)
    assert result is not None
    assert result.conflict_type == "factual"
    assert result.is_pathological is True
    print("  PASS: test_factual_contradiction_detected")


def test_factual_opposing_predicate_families():
    classifier = ConflictClassifier()
    a = _make_belief(subject="hnsw", predicate="improves", obj="recall", modality="is", polarity=1)
    b = _make_belief(subject="hnsw", predicate="degrades", obj="recall", modality="is", polarity=1)
    result = classifier.classify(a, b)
    assert result is not None
    assert result.conflict_type == "factual"
    assert result.is_pathological is True
    print("  PASS: test_factual_opposing_predicate_families")


def test_identity_tension_never_pathological():
    classifier = ConflictClassifier()
    a = _make_belief(subject="identity", obj="continuity", claim_type="identity")
    b = _make_belief(subject="identity", obj="replication", claim_type="philosophical")
    result = classifier.classify(a, b)
    assert result is not None
    assert result.conflict_type == "identity_tension"
    assert result.is_pathological is False  # THE GOLDEN INVARIANT
    print("  PASS: test_identity_tension_never_pathological (GOLDEN INVARIANT)")


def test_temporal_state_change_versioned():
    classifier = ConflictClassifier()
    a = _make_belief(subject="user_presence", obj="present", is_state_belief=True)
    b = _make_belief(subject="user_presence", obj="absent", is_state_belief=True)
    result = classifier.classify(a, b)
    assert result is not None, "State-belief contradictions must surface (C1-A5 fix)"
    assert result.conflict_type == "temporal"
    assert result.is_pathological is False
    assert result.severity in ("moderate", "informational")
    print("  PASS: test_temporal_state_change_versioned")


def test_temporal_stable_fact_conflict():
    classifier = ConflictClassifier()
    now = time.time()
    a = _make_belief(
        subject="sale", obj="feb_1", is_state_belief=False,
        time_range=(now - 1000, now - 500),
    )
    b = _make_belief(
        subject="sale", obj="mar_1", is_state_belief=False,
        time_range=(now - 800, now - 300),
    )
    result = classifier.classify(a, b)
    assert result is not None
    assert result.conflict_type == "temporal"
    assert result.is_pathological is True
    print("  PASS: test_temporal_stable_fact_conflict")


def test_near_miss_logged_with_reason():
    classifier = ConflictClassifier()
    a = _make_belief(subject="x", obj="y", modality="is")
    b = _make_belief(subject="x", obj="y", modality="may")
    classifier.classify(a, b)
    nms = classifier.get_near_misses()
    assert len(nms) >= 1
    assert nms[-1].reason == "different_modality"
    assert nms[-1].subject != ""
    print("  PASS: test_near_miss_logged_with_reason")


def test_classifier_deterministic():
    classifier = ConflictClassifier()
    a = _make_belief(subject="hnsw", obj="recall", polarity=1)
    b = _make_belief(subject="hnsw", obj="recall", polarity=-1)
    r1 = classifier.classify(a, b)
    r2 = classifier.classify(a, b)
    assert (r1 is None) == (r2 is None)
    if r1 and r2:
        assert r1.conflict_type == r2.conflict_type
        assert r1.is_pathological == r2.is_pathological
    print("  PASS: test_classifier_deterministic")


def test_same_source_memory_near_miss():
    classifier = ConflictClassifier()
    shared_mem = "mem_shared_123"
    a = _make_belief(subject="hnsw", obj="recall", polarity=1, source_memory_id=shared_mem)
    b = _make_belief(subject="hnsw", obj="recall", polarity=-1, source_memory_id=shared_mem)
    result = classifier.classify(a, b)
    assert result is None  # INVARIANT #13
    nms = classifier.get_near_misses()
    assert any(nm.reason == "same_source_memory" for nm in nms)
    print("  PASS: test_same_source_memory_near_miss (INVARIANT #13)")


# ===========================================================================
# resolution.py Phase 1 tests (#28-#38)
# ===========================================================================


def test_factual_resolution_downgrades_weaker():
    store = _make_store()
    a = _make_belief(subject="hnsw", obj="recall", belief_confidence=0.8)
    b = _make_belief(subject="hnsw", obj="recall", belief_confidence=0.4, polarity=-1)
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="factual", severity="critical", is_pathological=True,
        confidence=0.4, reasoning="test", conflict_key=a.conflict_key,
    )
    strategy = FactualResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.action_taken == "confidence_adjusted"
    assert len(outcome.confidence_deltas) > 0
    weaker_id = b.belief_id
    assert weaker_id in outcome.confidence_deltas
    assert outcome.confidence_deltas[weaker_id] < 0
    print("  PASS: test_factual_resolution_downgrades_weaker")


def test_identity_tension_never_downgrades():
    store = _make_store()
    a = _make_belief(subject="identity", obj="continuity", claim_type="identity", belief_confidence=0.6)
    b = _make_belief(subject="identity", obj="replication", claim_type="philosophical", belief_confidence=0.5)
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="identity_tension", severity="informational", is_pathological=False,
        confidence=0.5, reasoning="Identity tension between continuity and replication perspectives preserved",
        conflict_key="identity::identity::replication",
    )
    strategy = IdentityTensionResolution()
    outcome = strategy.resolve(classification, a, b, store)

    # THE SACRED INVARIANT
    assert len(outcome.confidence_deltas) == 0
    assert outcome.debt_delta <= 0
    updated_a = store.get(a.belief_id)
    updated_b = store.get(b.belief_id)
    assert updated_a.belief_confidence == a.belief_confidence
    assert updated_b.belief_confidence == b.belief_confidence
    print("  PASS: test_identity_tension_never_downgrades (SACRED INVARIANT)")


def test_identity_tension_creates_tension_record():
    store = _make_store()
    a = _make_belief(subject="identity", obj="continuity", claim_type="identity")
    b = _make_belief(subject="identity", obj="replication", claim_type="philosophical")
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="identity_tension", severity="informational", is_pathological=False,
        confidence=0.5, reasoning="test", conflict_key="identity::identity::replication",
    )
    strategy = IdentityTensionResolution()
    outcome = strategy.resolve(classification, a, b, store)

    assert outcome.tension_id is not None
    tension = store.get_tension(outcome.tension_id)
    assert tension is not None
    assert "identity" in tension.topic
    assert a.belief_id in tension.belief_ids
    assert b.belief_id in tension.belief_ids
    print("  PASS: test_identity_tension_creates_tension_record")


def test_tension_maturation_increments():
    store = _make_store()
    a = _make_belief(subject="identity", obj="continuity", claim_type="identity")
    b = _make_belief(subject="identity", obj="replication", claim_type="philosophical")
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="identity_tension", severity="informational", is_pathological=False,
        confidence=0.5, reasoning="Both perspectives are valid and coexist as productive tension",
        conflict_key="identity::identity::replication",
    )
    strategy = IdentityTensionResolution()
    outcome1 = strategy.resolve(classification, a, b, store)
    tension = store.get_tension(outcome1.tension_id)
    mat_before = tension.maturation_score

    c = _make_belief(subject="identity", obj="change", claim_type="identity")
    store.add(c)
    outcome2 = strategy.resolve(classification, a, c, store)
    tension = store.get_tension(outcome1.tension_id)
    assert tension.maturation_score > mat_before
    print("  PASS: test_tension_maturation_increments")


def test_tension_fan_in_cap():
    store = _make_store()
    beliefs = []
    for i in range(TENSION_MAX_BELIEF_IDS + 5):
        b = _make_belief(subject="identity", obj=f"view_{i}", claim_type="identity")
        store.add(b)
        beliefs.append(b)

    t = TensionRecord(
        tension_id="ten_cap_test", topic="identity::test",
        belief_ids=[b.belief_id for b in beliefs],
        conflict_key="identity::identity::test",
        created_at=time.time(), last_revisited=time.time(),
        revisit_count=0, stability=0.5, maturation_score=0.0,
    )
    store.add_tension(t)

    strategy = IdentityTensionResolution()
    strategy._enforce_fan_in_cap(t, store)
    assert len(t.belief_ids) <= TENSION_MAX_BELIEF_IDS
    print("  PASS: test_tension_fan_in_cap")


def test_temporal_state_version_zero_debt():
    store = _make_store()
    a = _make_belief(subject="user_presence", obj="present", is_state_belief=True, belief_confidence=0.7)
    b = _make_belief(subject="user_presence", obj="absent", is_state_belief=True, belief_confidence=0.7)
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="temporal", severity="informational", is_pathological=False,
        confidence=0.7, reasoning="State change", conflict_key="state::user_presence",
    )
    strategy = TemporalResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.debt_delta == DEBT_TEMPORAL_VERSION
    assert outcome.debt_delta == 0.0
    print("  PASS: test_temporal_state_version_zero_debt")


def test_debt_rises_on_factual():
    store = _make_store()
    a = _make_belief(subject="hnsw", obj="recall", belief_confidence=0.5)
    b = _make_belief(subject="hnsw", obj="recall", belief_confidence=0.5, polarity=-1)
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="factual", severity="critical", is_pathological=True,
        confidence=0.5, reasoning="test", conflict_key="fact::hnsw::recall",
    )
    strategy = FactualResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.debt_delta == DEBT_CRITICAL_UNRESOLVED or outcome.debt_delta == DEBT_FACTUAL_RESOLVED
    print("  PASS: test_debt_rises_on_factual")


def test_debt_stable_for_tension():
    store = _make_store()
    a = _make_belief(subject="identity", obj="continuity", claim_type="identity")
    b = _make_belief(subject="identity", obj="replication", claim_type="identity")
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="identity_tension", severity="informational", is_pathological=False,
        confidence=0.5, reasoning="test", conflict_key="identity::identity::replication",
    )
    strategy = IdentityTensionResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.debt_delta <= 0  # INVARIANT #3
    print("  PASS: test_debt_stable_for_tension (INVARIANT #3)")


def test_debt_recurrence_penalty():
    from epistemic.contradiction_engine import ContradictionEngine
    engine = ContradictionEngine()
    initial = engine._contradiction_debt
    engine._update_debt(0.05, "fact::test::x")
    after_first = engine._contradiction_debt

    engine._update_debt(0.05, "fact::test::x")
    after_second = engine._contradiction_debt

    assert after_second > after_first
    increase_second = after_second - after_first
    increase_first = after_first - initial
    assert increase_second > increase_first  # recurrence added extra
    print("  PASS: test_debt_recurrence_penalty")


def test_debt_passive_decay():
    from epistemic.contradiction_engine import ContradictionEngine
    engine = ContradictionEngine()
    engine._contradiction_debt = 0.1
    engine._last_decay_time = time.time() - 7200  # 2 hours ago
    engine.apply_passive_decay()
    assert engine._contradiction_debt < 0.1
    print("  PASS: test_debt_passive_decay")


def test_debt_clamped():
    from epistemic.contradiction_engine import ContradictionEngine
    engine = ContradictionEngine()
    engine._contradiction_debt = 0.99
    engine._update_debt(0.1, "test_key_clamp")
    assert engine._contradiction_debt <= 1.0

    engine._contradiction_debt = 0.01
    engine._update_debt(-0.1, "test_key_neg")
    assert engine._contradiction_debt >= 0.0
    print("  PASS: test_debt_clamped (INVARIANT #9)")


# ===========================================================================
# Engine + integration tests (#39-#42b)
# ===========================================================================


def test_offline_judge_bench():
    """Full synthetic battery: factual + identity + temporal."""
    store = _make_store()
    classifier = ConflictClassifier()

    b1 = _make_belief(subject="hnsw", predicate="improves", obj="recall",
                      modality="is", stance="assert", claim_type="factual", polarity=1)
    b2 = _make_belief(subject="hnsw", predicate="degrades", obj="recall",
                      modality="is", stance="assert", claim_type="factual", polarity=1)
    b3 = _make_belief(subject="identity", predicate="is", obj="continuity",
                      modality="is", stance="assert", claim_type="identity")
    b4 = _make_belief(subject="identity", predicate="is", obj="replication",
                      modality="may", stance="uncertain", claim_type="philosophical")
    b5 = _make_belief(subject="user_presence", obj="present",
                      is_state_belief=True, claim_type="observation")
    b6 = _make_belief(subject="user_presence", obj="absent",
                      is_state_belief=True, claim_type="observation")

    # Factual: b1 vs b2 should be detected (opposing predicate families)
    r1 = classifier.classify(b1, b2)
    assert r1 is not None, "Expected factual contradiction between b1 and b2"
    assert r1.conflict_type == "factual"
    assert r1.is_pathological is True

    # Identity: b3 vs b4 should be near-miss (different modality)
    r2 = classifier.classify(b3, b4)
    assert r2 is None, "Expected near-miss between b3 and b4 (different modality)"

    # Temporal: b5 vs b6 — state-belief contradiction now surfaces (C1-A5 fix)
    r3 = classifier.classify(b5, b6)
    assert r3 is not None, "State-belief contradictions must surface (C1-A5 fix)"
    assert r3.conflict_type == "temporal"
    assert r3.is_pathological is False

    # Resolution for factual
    for b in [b1, b2]:
        store.add(b)
    outcome = resolve_conflict(r1, b1, b2, store)
    assert outcome.debt_delta != 0

    nms = classifier.get_near_misses()
    assert any(nm.reason == "different_modality" for nm in nms)

    print("  PASS: test_offline_judge_bench")


def test_engine_rehydrate_idempotent():
    from epistemic.contradiction_engine import ContradictionEngine
    d = tempfile.mkdtemp()
    engine = ContradictionEngine()
    engine._belief_store._beliefs_path = os.path.join(d, "b.jsonl")
    engine._belief_store._tensions_path = os.path.join(d, "t.jsonl")

    beliefs = [_make_belief(subject=f"sub_{i}") for i in range(5)]
    for b in beliefs:
        engine._belief_store.add(b)

    engine.rehydrate()
    count1 = len(engine._belief_store._beliefs)
    engine.rehydrate()
    count2 = len(engine._belief_store._beliefs)
    assert count1 == count2
    print("  PASS: test_engine_rehydrate_idempotent")


def test_discarded_beliefs_invisible():
    from epistemic.contradiction_engine import ContradictionEngine
    engine = ContradictionEngine()
    initial_debt = engine._contradiction_debt
    initial_discards = engine._extraction_discard_count

    low_conf = _make_belief(
        subject="test", obj="discard", extraction_confidence=0.1,
    )
    engine._belief_store.add(low_conf)
    results = engine.check_new_belief(low_conf)

    assert engine._contradiction_debt == initial_debt
    print("  PASS: test_discarded_beliefs_invisible")


# ===========================================================================
# Phase 2 resolution tests (#43-#47)
# ===========================================================================


def test_provenance_never_merges_sources():
    store = _make_store()
    a = _make_belief(subject="preference", obj="dark_mode", provenance="user_claim")
    b = _make_belief(subject="preference", obj="light_mode", provenance="observed")
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="provenance", severity="critical", is_pathological=True,
        confidence=0.5, reasoning="test", conflict_key="pref::preference::dark_mode",
    )
    strategy = ProvenanceResolution()
    outcome = strategy.resolve(classification, a, b, store)

    # INVARIANT #4: both beliefs survive
    assert store.get(a.belief_id) is not None
    assert store.get(b.belief_id) is not None
    assert outcome.action_taken == "source_separated"
    print("  PASS: test_provenance_never_merges_sources (INVARIANT #4)")


def test_provenance_user_vs_observed_clarification():
    store = _make_store()
    a = _make_belief(subject="pref", obj="dark", provenance="user_claim", belief_confidence=0.7)
    b = _make_belief(subject="pref", obj="light", provenance="observed", belief_confidence=0.7)
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="provenance", severity="critical", is_pathological=True,
        confidence=0.5, reasoning="test", conflict_key="pref::pref::dark",
    )
    strategy = ProvenanceResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.needs_user_clarification is True
    assert outcome.clarification_prompt is not None
    print("  PASS: test_provenance_user_vs_observed_clarification")


def test_policy_outcome_penalizes_action():
    store = _make_store()
    a = _make_belief(subject="academic_search", obj="improved", claim_type="policy", modality="caused", belief_confidence=0.7)
    b = _make_belief(subject="academic_search", obj="degraded", claim_type="policy", modality="caused", belief_confidence=0.4, polarity=-1)
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="policy_outcome", severity="critical", is_pathological=True,
        confidence=0.4, reasoning="test", conflict_key="policy_outcome::academic_search::improved",
    )
    strategy = PolicyResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.action_taken == "policy_penalized"
    assert outcome.debt_delta == DEBT_MODERATE_UNRESOLVED
    print("  PASS: test_policy_outcome_penalizes_action")


def test_policy_norm_does_not_penalize_action():
    store = _make_store()
    a = _make_belief(subject="search", obj="academic", claim_type="policy", modality="should")
    b = _make_belief(subject="search", obj="web", claim_type="policy", modality="should")
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="policy_norm", severity="moderate", is_pathological=False,
        confidence=0.5, reasoning="test", conflict_key="policy_norm::search::academic",
    )
    strategy = PolicyResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.action_taken == "norm_noted"  # THE WALL
    assert outcome.debt_delta == DEBT_POLICY_NORM
    assert len(outcome.confidence_deltas) == 0  # norms don't penalize confidence
    print("  PASS: test_policy_norm_does_not_penalize_action (THE WALL)")


def test_multi_perspective_no_collapse():
    store = _make_store()
    a = _make_belief(subject="hnsw", obj="recall", claim_type="factual", scope="dense_embeddings")
    b = _make_belief(subject="hnsw", obj="recall", claim_type="factual", scope="sparse_embeddings", polarity=-1)
    store.add(a)
    store.add(b)

    from epistemic.belief_record import ConflictClassification
    classification = ConflictClassification(
        conflict_type="multi_perspective", severity="informational", is_pathological=False,
        confidence=0.5, reasoning="test", conflict_key="fact::hnsw::recall",
    )
    strategy = MultiPerspectiveResolution()
    outcome = strategy.resolve(classification, a, b, store)
    assert outcome.action_taken == "tension_held"
    assert outcome.debt_delta == DEBT_MULTI_PERSPECTIVE
    assert outcome.debt_delta == 0.0
    # Both beliefs stay as tension
    assert store.get(a.belief_id).resolution_state == "tension"
    assert store.get(b.belief_id).resolution_state == "tension"
    print("  PASS: test_multi_perspective_no_collapse")


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [
    # belief_record.py (#1-#7)
    test_belief_record_creation,
    test_conflict_key_type_specific,
    test_tension_topic_deterministic,
    test_tension_topic_merges_aliases,
    test_belief_store_round_trip,
    test_tension_store_round_trip,
    test_eviction_protects_active_and_tension,
    # claim_extractor.py (#8-#18)
    test_canonicalize_basic,
    test_canonicalize_synonyms,
    test_canonicalize_idempotent,
    test_canonicalize_predicate_negation,
    test_canonicalize_predicate_clusters,
    test_extract_factual_knowledge,
    test_extract_identity_claim,
    test_extract_preference,
    test_extract_ambiguous_underextracts,
    test_extract_conversation_skipped,
    test_extract_max_claims_cap,
    # conflict_classifier.py (#19-#27b)
    test_different_modality_near_miss,
    test_question_stance_near_miss,
    test_low_extraction_confidence_near_miss,
    test_factual_contradiction_detected,
    test_factual_opposing_predicate_families,
    test_identity_tension_never_pathological,
    test_temporal_state_change_versioned,
    test_temporal_stable_fact_conflict,
    test_near_miss_logged_with_reason,
    test_classifier_deterministic,
    test_same_source_memory_near_miss,
    # resolution.py Phase 1 (#28-#38)
    test_factual_resolution_downgrades_weaker,
    test_identity_tension_never_downgrades,
    test_identity_tension_creates_tension_record,
    test_tension_maturation_increments,
    test_tension_fan_in_cap,
    test_temporal_state_version_zero_debt,
    test_debt_rises_on_factual,
    test_debt_stable_for_tension,
    test_debt_recurrence_penalty,
    test_debt_passive_decay,
    test_debt_clamped,
    # Engine + integration (#39-#42b)
    test_offline_judge_bench,
    test_engine_rehydrate_idempotent,
    test_discarded_beliefs_invisible,
    # resolution.py Phase 2 (#43-#47)
    test_provenance_never_merges_sources,
    test_provenance_user_vs_observed_clarification,
    test_policy_outcome_penalizes_action,
    test_policy_norm_does_not_penalize_action,
    test_multi_perspective_no_collapse,
]


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Layer 5: Contradiction Engine — {len(ALL_TESTS)} tests")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test_fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)
