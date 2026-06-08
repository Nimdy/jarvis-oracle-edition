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


# --- regression: "recent" must be by TIMESTAMP, not list position -------------
import time as _time


def _tmem(ts, mid="m", payload="p", prov="user_claim"):
    return SimpleNamespace(timestamp=ts, id=mid * 14, type="x",
                           provenance=prov, weight=0.5, payload=payload)


def _store(mems):
    return SimpleNamespace(_memories=mems, _lock=threading.Lock())


def test_get_recent_sorts_by_timestamp_not_list_order():
    # the list is OUT of time order — a newer write sits mid-list, old ones at the tail
    # (exactly the dream/consolidation-artifact bug: [-count:] would return the OLD tail)
    s = _store([_tmem(100, "a"), _tmem(500, "new"), _tmem(50, "b"), _tmem(10, "old")])
    out = MemoryStorage.get_recent(s, 2)
    # the 2 newest by timestamp are 500 and 100 (NOT the list tail [50, 10])
    assert sorted(m.timestamp for m in out) == [100, 500]


def test_recent_with_provenance_is_newest_first():
    now = _time.time()
    s = _store([
        _tmem(now - 1000, "o", "old", "conversation"),
        _tmem(now - 10, "n", "new", "casual_conversation"),
        _tmem(now - 5000, "x", "older", "seed"),
    ])
    out = MemoryStorage.get_recent_with_provenance(s, 3)
    assert out[0]["provenance"] == "casual_conversation"      # the newest
    assert out[0]["age_s"] <= out[1]["age_s"] <= out[2]["age_s"]


def test_recent_with_provenance_preview_widened():
    long = "x" * 300
    s = _store([_tmem(1.0, "a", long)])
    out = MemoryStorage.get_recent_with_provenance(s, 1)
    assert len(out[0]["payload_preview"]) == 140  # was 60


def test_recent_episodes_by_time_not_list_order():
    from memory.episodes import EpisodicMemory

    def ep(ended):
        e = SimpleNamespace(is_active=False, ended_at=ended, started_at=ended - 10)
        e.turn_count = lambda: 100  # comfortably above MIN_TURNS_FOR_EPISODE
        return e

    # list order is jumbled; newest (300) sits mid-list
    stub = SimpleNamespace(_episodes=[ep(100), ep(300), ep(50), ep(200)])
    out = EpisodicMemory.get_recent_episodes(stub, 2)
    assert [e.ended_at for e in out] == [200, 300]  # 2 newest by time, ascending
