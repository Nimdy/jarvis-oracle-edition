"""OSV voice distillation seed — invariants after re-plumbing onto the canonical DistillationCollector:
only faithful teacher pairs are recorded (as teacher='native_voice'), the firewall keeps bad pairs out,
and nothing is written to the memory store.
"""

import asyncio

import cognition.self_view.voice_seed as vs


class _Client:
    def __init__(self, resp):
        self.resp = resp

    async def chat(self, messages, system_prompt=None, model_override=None):
        return self.resp


class _FakeCollector:
    def __init__(self):
        self.records = []

    def record(self, teacher, signal_type, data, metadata=None, origin="system", fidelity=1.0):
        self.records.append({"teacher": teacher, "signal_type": signal_type, "data": data,
                             "origin": origin, "fidelity": fidelity})

    def get_recent_signals(self, teacher, n=8):
        return [{**r["data"], "ts": 0.0} for r in self.records if r["teacher"] == teacher][-n:][::-1]

    def get_stats(self):
        n = len(self.records)
        return {"teachers": {"native_voice": {"total": n, "lived": n, "synthetic": 0, "quarantined": 0}}}


GROUNDED = ("My code-grounded architecture covers 98 subsystems across 45 domains, behind a 15-layer "
            "integrity stack. By designed status: 58 shipped/live, 14 shadow.")
FAITHFUL = ("Honestly? I'm a layered thing — about 98 subsystems over 45 domains, behind a 15-layer "
            "integrity stack. 58 are live, 14 run quietly in shadow.")


def _reset(monkeypatch):
    col = _FakeCollector()
    monkeypatch.setattr(vs, "_collector", lambda: col)
    vs._fidelity["attempts"] = 0
    vs._fidelity["logged"] = 0
    vs._fidelity["rejected"] = {}
    return col


def test_faithful_pair_recorded_to_collector(monkeypatch):
    col = _reset(monkeypatch)
    s = asyncio.run(vs.capture_teacher_pair(GROUNDED, "capabilities", _Client(FAITHFUL)))
    assert s["logged"] is True
    assert len(col.records) == 1
    rec = col.records[0]
    assert rec["teacher"] == "native_voice" and rec["origin"] == "conversation"
    assert rec["data"]["grounded"] == GROUNDED and rec["data"]["voiced"] == FAITHFUL
    assert vs.seed_stats()["entries"] == 1


def test_unfaithful_teacher_not_recorded(monkeypatch):
    col = _reset(monkeypatch)
    bad = _Client("I'm made of around 200 subsystems and I feel truly conscious.")
    s = asyncio.run(vs.capture_teacher_pair(GROUNDED, "capabilities", bad))
    assert s["logged"] is False and s["reason"].startswith("teacher_rejected")
    assert col.records == []  # corpus unpolluted
    assert vs.seed_stats()["fidelity"]["rejected"]  # rejection tallied


def test_none_client_noop(monkeypatch):
    _reset(monkeypatch)
    s = asyncio.run(vs.capture_teacher_pair(GROUNDED, "identity", None))
    assert s["logged"] is False and "no_input" in s["reason"]


def test_recent_pairs_and_by_kind(monkeypatch):
    _reset(monkeypatch)
    asyncio.run(vs.capture_teacher_pair(GROUNDED, "capabilities", _Client(FAITHFUL)))
    assert vs.recent_pairs(8)[0]["kind"] == "capabilities"
    assert vs.by_kind().get("capabilities") == 1


def test_writes_nothing_to_memory_store():
    src = open(vs.__file__).read()
    for forbidden in ("canonical_remember", "memory.add", "add_memory", "store_memory"):
        assert forbidden not in src
