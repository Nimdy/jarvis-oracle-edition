"""Memory-integrity telemetry — provenance distribution for the banter-firewall
dashboard. Real counts from the store; no fabricated gauges.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

try:
    from memory.storage import MemoryStorage
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("memory.storage import unavailable", allow_module_level=True)


def _stub(provs):
    return SimpleNamespace(
        _memories=[SimpleNamespace(provenance=p) for p in provs],
        _lock=threading.Lock(),
    )


def test_distribution_counts_and_firewall_metrics():
    d = MemoryStorage.get_provenance_distribution(_stub([
        "user_claim", "casual_conversation", "casual_conversation",
        "external_source", "observed", "conversation", None,
    ]))
    assert d["total"] == 7
    assert d["banter_protected"] == 2          # casual_conversation count = firewall catches
    assert d["user_claims"] == 1
    assert d["validated_external"] == 1
    assert d["observed"] == 1


def test_classes_sorted_with_trust_boosts():
    d = MemoryStorage.get_provenance_distribution(_stub(
        ["casual_conversation"] * 3 + ["user_claim"]))
    classes = {c["provenance"]: c for c in d["classes"]}
    # banter is the biggest bucket here -> first, and 0.0 trust
    assert d["classes"][0]["provenance"] == "casual_conversation"
    assert classes["casual_conversation"]["trust_boost"] == 0.0
    assert classes["casual_conversation"]["count"] == 3
    assert classes["user_claim"]["trust_boost"] == 0.04


def test_empty_store():
    d = MemoryStorage.get_provenance_distribution(_stub([]))
    assert d["total"] == 0 and d["banter_protected"] == 0 and d["classes"] == []


def test_none_provenance_bucketed_as_unknown():
    d = MemoryStorage.get_provenance_distribution(_stub([None, None]))
    cls = {c["provenance"]: c["count"] for c in d["classes"]}
    assert cls.get("unknown") == 2
