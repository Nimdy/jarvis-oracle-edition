"""OSV voice distillation seed — invariants: only faithful teacher pairs logged, deduped, bounded,
and the corpus is the ONLY thing written (never the memory store)."""

import asyncio
import json

import cognition.self_view.voice_seed as vs


def _reset(tmp_path):
    vs.SEED_PATH = tmp_path / "seed.jsonl"
    vs._seen.clear()
    vs._seen_set.clear()
    vs._count_cache = None


class _Client:
    def __init__(self, resp):
        self.resp = resp

    async def chat(self, messages, system_prompt=None, model_override=None):
        return self.resp


GROUNDED = ("My code-grounded architecture covers 98 subsystems across 45 domains, behind a 15-layer "
            "integrity stack. By designed status: 58 shipped/live, 14 shadow.")
FAITHFUL = ("Honestly? I'm a layered thing — about 98 subsystems over 45 domains, behind a 15-layer "
            "integrity stack. 58 are live, 14 run quietly in shadow.")


def test_faithful_pair_logged_once_then_deduped(tmp_path):
    _reset(tmp_path)
    c = _Client(FAITHFUL)
    s1 = asyncio.run(vs.capture_teacher_pair(GROUNDED, "capabilities", c))
    assert s1["logged"] is True
    s2 = asyncio.run(vs.capture_teacher_pair(GROUNDED, "capabilities", c))  # same -> dedup
    assert s2["logged"] is False and s2["reason"] == "dedup"
    lines = vs.SEED_PATH.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["grounded"] == GROUNDED and rec["voiced"] == FAITHFUL and rec["kind"] == "capabilities"


def test_unfaithful_teacher_not_logged(tmp_path):
    _reset(tmp_path)
    bad = _Client("I'm made of around 200 subsystems and I feel truly conscious.")
    s = asyncio.run(vs.capture_teacher_pair(GROUNDED, "capabilities", bad))
    assert s["logged"] is False and s["reason"].startswith("teacher_rejected")
    assert not vs.SEED_PATH.exists() or vs.SEED_PATH.read_text() == ""  # corpus unpolluted


def test_none_client_noop(tmp_path):
    _reset(tmp_path)
    s = asyncio.run(vs.capture_teacher_pair(GROUNDED, "identity", None))
    assert s["logged"] is False and s["reason"] == "no_input_or_client"


def test_cap_is_bounded(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr(vs, "MAX_SEED_ENTRIES", 2)
    c = _Client(FAITHFUL)
    for i in range(5):
        asyncio.run(vs.capture_teacher_pair(GROUNDED + f" variant {i}", "capabilities", c))
    lines = vs.SEED_PATH.read_text().strip().splitlines()
    assert len(lines) <= 2  # hard cap honored


def test_writes_nothing_but_the_corpus(tmp_path):
    # memory-safety: the module imports no memory-store writers
    src = open(vs.__file__).read()
    for forbidden in ("canonical_remember", "memory.add", "add_memory", "store_memory"):
        assert forbidden not in src
