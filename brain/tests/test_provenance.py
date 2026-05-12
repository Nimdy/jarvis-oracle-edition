"""Tests for provenance-aware memory (Layer 2 epistemic immune system).

Covers:
  - ProvenanceType enum values
  - Memory dataclass accepts provenance field with backward compat
  - CreateMemoryData threads provenance through create_memory()
  - resolve_provenance_boost() helper: field-based + tag fallback
  - Salience feature vector dimension (11)
  - Serialization round-trip preserves provenance
  - Legacy memories load with provenance="unknown"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import (
    Memory, ProvenanceType, PROVENANCE_BOOST, PROVENANCE_ORDINAL,
    resolve_provenance_boost,
)
from memory.core import CreateMemoryData, memory_core


def test_provenance_type_values():
    expected = {
        "observed", "user_claim", "conversation", "model_inference",
        "external_source", "experiment_result", "derived_pattern",
        "seed", "unknown",
    }
    assert set(PROVENANCE_BOOST.keys()) == expected
    assert set(PROVENANCE_ORDINAL.keys()) == expected
    print("  PASS: provenance type values")


def test_memory_default_provenance():
    """Legacy/default memories get provenance='unknown'."""
    mem = Memory(
        id="mem_test", timestamp=1.0, weight=0.5,
        tags=("test",), payload="hello", type="conversation",
    )
    assert mem.provenance == "unknown"
    print("  PASS: memory default provenance")


def test_memory_explicit_provenance():
    mem = Memory(
        id="mem_test2", timestamp=1.0, weight=0.5,
        tags=("test",), payload="hello", type="observation",
        provenance="observed",
    )
    assert mem.provenance == "observed"
    print("  PASS: memory explicit provenance")


def test_create_memory_data_threads_provenance():
    data = CreateMemoryData(
        type="observation",
        payload="User waved",
        weight=0.3,
        tags=["gesture"],
        provenance="observed",
    )
    mem = memory_core.create_memory(data)
    assert mem is not None
    assert mem.provenance == "observed"
    print("  PASS: CreateMemoryData threads provenance")


def test_create_memory_data_default_provenance():
    data = CreateMemoryData(
        type="conversation",
        payload="Hello there",
        weight=0.5,
        tags=["chat"],
    )
    mem = memory_core.create_memory(data)
    assert mem is not None
    assert mem.provenance == "unknown"
    print("  PASS: CreateMemoryData default provenance")


def test_resolve_provenance_boost_field_based():
    """When provenance is set, use the field-based lookup."""
    mem = Memory(
        id="mem_ext", timestamp=1.0, weight=0.5,
        tags=(), payload="research", type="factual_knowledge",
        provenance="external_source",
    )
    boost = resolve_provenance_boost(mem)
    assert boost == 0.10
    print("  PASS: resolve_provenance_boost field-based")


def test_resolve_provenance_boost_tag_fallback():
    """Legacy memories with provenance='unknown' use tag-based fallback."""
    mem = Memory(
        id="mem_legacy", timestamp=1.0, weight=0.5,
        tags=("evidence:peer_reviewed", "autonomous_research"),
        payload="old research", type="factual_knowledge",
        provenance="unknown",
    )
    boost = resolve_provenance_boost(mem)
    assert boost == 0.12
    print("  PASS: resolve_provenance_boost tag fallback")


def test_resolve_provenance_boost_no_provenance():
    """Memory with provenance='unknown' and no tags gets 0."""
    mem = Memory(
        id="mem_plain", timestamp=1.0, weight=0.5,
        tags=("random",), payload="plain", type="conversation",
        provenance="unknown",
    )
    boost = resolve_provenance_boost(mem)
    assert boost == 0.0
    print("  PASS: resolve_provenance_boost no provenance")


def test_resolve_provenance_boost_all_types():
    """All provenance types return valid floats."""
    for prov, expected in PROVENANCE_BOOST.items():
        mem = Memory(
            id=f"mem_{prov}", timestamp=1.0, weight=0.5,
            tags=(), payload="test", type="observation",
            provenance=prov,
        )
        boost = resolve_provenance_boost(mem)
        assert boost == expected, f"Provenance {prov}: got {boost}, expected {expected}"
    print("  PASS: resolve_provenance_boost all types")


def test_serialization_roundtrip():
    """Memory -> dict -> Memory preserves provenance."""
    from dataclasses import asdict
    mem = Memory(
        id="mem_rt", timestamp=1.0, weight=0.5,
        tags=("test",), payload="hello", type="conversation",
        provenance="user_claim",
    )
    d = asdict(mem)
    assert d["provenance"] == "user_claim"

    d["tags"] = tuple(d["tags"])
    d["associations"] = tuple(d["associations"])
    restored = Memory(**d)
    assert restored.provenance == "user_claim"
    print("  PASS: serialization roundtrip")


def test_legacy_dict_loads_as_unknown():
    """A dict without 'provenance' key loads as Memory with default 'unknown'."""
    legacy = {
        "id": "mem_old", "timestamp": 1.0, "weight": 0.5,
        "tags": ("chat",), "payload": "old memory", "type": "conversation",
        "associations": (), "decay_rate": 0.01, "is_core": False,
        "last_validated": 1.0, "association_count": 0, "priority": 500,
    }
    mem = Memory(**legacy)
    assert mem.provenance == "unknown"
    print("  PASS: legacy dict loads as unknown")


def test_salience_input_dim():
    from memory.salience import INPUT_DIM
    assert INPUT_DIM == 11, f"Expected INPUT_DIM=11, got {INPUT_DIM}"
    print("  PASS: salience INPUT_DIM == 11")


def test_salience_feature_vector_length():
    """Verify that lifecycle_log produces 11-dim feature vectors."""
    from memory.lifecycle_log import MemoryLifecycleLog
    log = MemoryLifecycleLog()
    log.log_created(
        memory_id="mem_fv_test",
        memory_type="observation",
        weight=0.5,
        decay_rate=0.02,
        tags=("test",),
        payload="test payload",
        source="observation",
        provenance="observed",
        memory_count=10,
    )

    import time
    log._creations["mem_fv_test"].created_at = time.time() - 7200
    log._creations["mem_fv_test"].retrieved = True
    log._creations["mem_fv_test"].retrieval_count = 3

    pairs = log.get_salience_training_pairs()
    if pairs:
        assert len(pairs[0].features) == 11, f"Expected 11 features, got {len(pairs[0].features)}"
        prov_val = pairs[0].features[10]
        assert 0.0 <= prov_val <= 1.0, f"Provenance feature out of range: {prov_val}"
    print("  PASS: salience feature vector length == 11")


def test_provenance_survives_decay():
    """Regression: decay_all() must preserve provenance on frozen-dataclass rebuild."""
    from memory.storage import MemoryStorage
    import time

    storage = MemoryStorage()
    storage._last_decay_time = time.time() - 86400  # 1 day ago

    storage.add(Memory(
        id="m_decay_test", timestamp=time.time() - 3600, weight=0.7,
        tags=("test",), payload="decay provenance test", type="conversation",
        provenance="external_source",
    ))

    decayed = storage.decay_all()
    assert decayed >= 1

    m = storage.get("m_decay_test")
    assert m is not None
    assert m.provenance == "external_source", f"Expected external_source, got {m.provenance}"
    print("  PASS: provenance survives decay_all()")


def test_provenance_survives_reinforce():
    """Regression: reinforce() must preserve provenance on frozen-dataclass rebuild."""
    from memory.storage import MemoryStorage

    storage = MemoryStorage()
    storage.add(Memory(
        id="m_reinforce_test", timestamp=1.0, weight=0.5,
        tags=("test",), payload="reinforce test", type="conversation",
        provenance="user_claim",
    ))

    storage.reinforce("m_reinforce_test", 0.1)
    m = storage.get("m_reinforce_test")
    assert m is not None
    assert m.provenance == "user_claim", f"Expected user_claim, got {m.provenance}"
    print("  PASS: provenance survives reinforce()")


def test_provenance_survives_associate():
    """Regression: associate() must preserve provenance on both memories."""
    from memory.storage import MemoryStorage

    storage = MemoryStorage()
    storage.add(Memory(
        id="m_a", timestamp=1.0, weight=0.5,
        tags=("test",), payload="a", type="conversation",
        provenance="observed",
    ))
    storage.add(Memory(
        id="m_b", timestamp=1.0, weight=0.5,
        tags=("test",), payload="b", type="factual_knowledge",
        provenance="external_source",
    ))

    storage.associate("m_a", "m_b")
    a = storage.get("m_a")
    b = storage.get("m_b")
    assert a.provenance == "observed", f"Expected observed, got {a.provenance}"
    assert b.provenance == "external_source", f"Expected external_source, got {b.provenance}"
    print("  PASS: provenance survives associate()")


def test_backfill_infers_provenance():
    """Load migration should backfill unknown provenance from type+tags."""
    from memory.storage import MemoryStorage

    storage = MemoryStorage()

    legacy_data = [
        {"id": "m_obs", "timestamp": 1.0, "weight": 0.4, "tags": ["test"],
         "payload": "saw something", "type": "observation", "associations": [],
         "provenance": "unknown"},
        {"id": "m_conv", "timestamp": 1.0, "weight": 0.5, "tags": ["test"],
         "payload": "chat", "type": "conversation", "associations": [],
         "provenance": "unknown"},
        {"id": "m_pref", "timestamp": 1.0, "weight": 0.6, "tags": ["test"],
         "payload": "likes jazz", "type": "user_preference", "associations": [],
         "provenance": "unknown"},
        {"id": "m_fk", "timestamp": 1.0, "weight": 0.5, "tags": ["source:academic"],
         "payload": "paper result", "type": "factual_knowledge", "associations": [],
         "provenance": "unknown"},
    ]
    storage.load_from_json(legacy_data)

    stats = storage.get_stats()
    bp = stats["by_provenance"]
    assert bp.get("observed", 0) >= 1
    assert bp.get("conversation", 0) >= 1
    assert bp.get("user_claim", 0) >= 1
    assert bp.get("external_source", 0) >= 1
    assert bp.get("unknown", 0) == 0, f"Expected 0 unknown, got {bp}"
    print("  PASS: backfill infers provenance from type+tags")


def test_lifecycle_jsonl_persists_provenance():
    """Regression: log_created() must write provenance to JSONL on disk."""
    import json
    import tempfile
    from memory.lifecycle_log import MemoryLifecycleLog

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        log = MemoryLifecycleLog(path=tmp_path)
        log.init()
        log.log_created(
            memory_id="m_jsonl_prov",
            memory_type="observation",
            weight=0.45,
            decay_rate=0.02,
            tags=("test", "vision"),
            payload="saw a person",
            source="observation",
            provenance="observed",
            salience_advised=True,
        )

        with open(tmp_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        creation = [l for l in lines if l.get("mid") == "m_jsonl_prov"]
        assert len(creation) == 1, f"Expected 1 creation event, got {len(creation)}"
        rec = creation[0]
        assert rec.get("provenance") == "observed", f"JSONL missing provenance: {rec}"
        assert rec.get("salience_advised") is True, f"JSONL missing salience_advised: {rec}"
        print("  PASS: lifecycle JSONL persists provenance + salience_advised")
    finally:
        os.unlink(tmp_path)


def test_lifecycle_rehydrate_restores_provenance():
    """Regression: rehydrate() must restore provenance from JSONL, not default to 'unknown'."""
    import json
    import tempfile
    from memory.lifecycle_log import MemoryLifecycleLog

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        log1 = MemoryLifecycleLog(path=tmp_path)
        log1.init()

        test_cases = [
            ("m_rh_obs", "observation", "observed", True),
            ("m_rh_conv", "conversation", "conversation", False),
            ("m_rh_ext", "factual_knowledge", "external_source", False),
            ("m_rh_user", "user_preference", "user_claim", True),
        ]
        for mid, mtype, prov, sal in test_cases:
            log1.log_created(
                memory_id=mid,
                memory_type=mtype,
                weight=0.5,
                decay_rate=0.01,
                tags=("test",),
                payload="test",
                source="test",
                provenance=prov,
                salience_advised=sal,
            )

        log2 = MemoryLifecycleLog(path=tmp_path)
        log2.init()
        count = log2.rehydrate()
        assert count == 4, f"Expected 4 rehydrated, got {count}"

        for mid, mtype, prov, sal in test_cases:
            rec = log2._creations.get(mid)
            assert rec is not None, f"Missing rehydrated record: {mid}"
            assert rec.provenance == prov, (
                f"Rehydrated {mid}: expected provenance={prov}, got {rec.provenance}"
            )
            assert rec.salience_advised == sal, (
                f"Rehydrated {mid}: expected salience_advised={sal}, got {rec.salience_advised}"
            )

        print("  PASS: rehydrate() restores provenance + salience_advised from JSONL")
    finally:
        os.unlink(tmp_path)


def test_lifecycle_rehydrate_legacy_defaults_unknown():
    """Pre-fix JSONL lines without provenance field should rehydrate as 'unknown'."""
    import json
    import tempfile
    from memory.lifecycle_log import MemoryLifecycleLog

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        legacy_line = json.dumps({
            "type": "created", "mid": "m_legacy_noprov", "t": 1000.0,
            "mem_type": "conversation", "weight": 0.5, "decay_rate": 0.01,
            "tags": 1, "payload_len": 20, "source": "conversation",
            "user_present": False, "mode": "passive",
        })
        f.write(legacy_line + "\n")
        tmp_path = f.name

    try:
        log = MemoryLifecycleLog(path=tmp_path)
        log.init()
        count = log.rehydrate()
        assert count == 1, f"Expected 1 rehydrated, got {count}"

        rec = log._creations.get("m_legacy_noprov")
        assert rec is not None
        assert rec.provenance == "unknown", (
            f"Legacy record should default to unknown, got {rec.provenance}"
        )
        assert rec.salience_advised is False
        print("  PASS: legacy JSONL without provenance rehydrates as 'unknown'")
    finally:
        os.unlink(tmp_path)


def test_lifecycle_provenance_feature_survives_rehydration():
    """End-to-end: provenance feature in salience training pair is correct after rehydrate."""
    import json
    import tempfile
    import time
    from memory.lifecycle_log import MemoryLifecycleLog, _PROVENANCE_MAP

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        log1 = MemoryLifecycleLog(path=tmp_path)
        log1.init()
        log1.log_created(
            memory_id="m_feat_prov",
            memory_type="observation",
            weight=0.5,
            decay_rate=0.02,
            tags=("test",),
            payload="test",
            source="observation",
            provenance="observed",
        )

        log2 = MemoryLifecycleLog(path=tmp_path)
        log2.init()
        log2.rehydrate()

        rec = log2._creations["m_feat_prov"]
        rec.created_at = time.time() - 7200
        rec.retrieved = True
        rec.retrieval_count = 3

        pairs = log2.get_salience_training_pairs()
        assert len(pairs) >= 1, "Expected at least 1 training pair"

        prov_feature = pairs[0].features[10]
        expected = _PROVENANCE_MAP["observed"] / 8.0
        assert abs(prov_feature - expected) < 0.001, (
            f"Provenance feature: expected {expected} (observed), got {prov_feature}"
        )
        print("  PASS: provenance feature survives rehydration into salience training")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("\n=== Provenance-Aware Memory Tests ===\n")
    test_provenance_type_values()
    test_memory_default_provenance()
    test_memory_explicit_provenance()
    test_create_memory_data_threads_provenance()
    test_create_memory_data_default_provenance()
    test_resolve_provenance_boost_field_based()
    test_resolve_provenance_boost_tag_fallback()
    test_resolve_provenance_boost_no_provenance()
    test_resolve_provenance_boost_all_types()
    test_serialization_roundtrip()
    test_legacy_dict_loads_as_unknown()
    test_salience_input_dim()
    test_salience_feature_vector_length()
    test_provenance_survives_decay()
    test_provenance_survives_reinforce()
    test_provenance_survives_associate()
    test_backfill_infers_provenance()
    test_lifecycle_jsonl_persists_provenance()
    test_lifecycle_rehydrate_restores_provenance()
    test_lifecycle_rehydrate_legacy_defaults_unknown()
    test_lifecycle_provenance_feature_survives_rehydration()
    print("\n  All tests passed!\n")
