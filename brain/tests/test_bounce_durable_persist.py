"""Bounce-durable persist of tags, downweight, and spark/grounding clocks.

Isolated HOME only. Does not touch the live ~/.jarvis on the brain host.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.events import Memory
from memory.persistence import MemoryPersistence
from memory.storage import MemoryStorage


def _mem(mid: str, *, weight: float = 0.5, tags: tuple[str, ...] = (), payload: str = "kitchen stove") -> Memory:
    return Memory(
        id=mid,
        timestamp=time.time(),
        weight=weight,
        tags=tags,
        payload=payload,
        type="conversation",
        provenance="conversation",
    )


def test_merge_disk_corrections_keeps_tags_and_lower_weight():
    ram = [{"id": "mem_91hZ", "weight": 0.5, "tags": [], "decay_rate": 0.01, "payload": "kitchen stove"}]
    disk = [{"id": "mem_91hZ", "weight": 0.2, "tags": ["not_fact", "ungrounded_vision"], "decay_rate": 0.05, "payload": "kitchen stove"}]
    out = MemoryPersistence._merge_disk_corrections(ram, disk)
    assert out[0]["id"] == "mem_91hZ"
    assert "not_fact" in out[0]["tags"]
    assert "ungrounded_vision" in out[0]["tags"]
    assert out[0]["weight"] == 0.2
    assert out[0]["decay_rate"] == 0.05
    assert out[0]["payload"] == "kitchen stove"


def test_merge_restatement_does_not_restore_corrected_or_min_weight():
    """Lived: Tanya restatement 0.70 no-corrected lost to disk 0.07+corrected."""
    ram = [{
        "id": "mem_n7M6",
        "weight": 0.70,
        "tags": ["personal_fact", "user_preference"],
        "decay_rate": 0.005,
        "payload": "User's wife is Tanya",
        "last_validated": 200.0,
        "timestamp": 100.0,
    }]
    disk = [{
        "id": "mem_n7M6",
        "weight": 0.07,
        "tags": ["corrected", "personal_fact", "user_preference"],
        "decay_rate": 0.005,
        "payload": "User's wife is Tanya",
        "last_validated": 100.0,
        "timestamp": 100.0,
    }]
    out = MemoryPersistence._merge_disk_corrections(ram, disk)
    assert out[0]["weight"] == 0.70
    assert "corrected" not in out[0]["tags"]
    assert "personal_fact" in out[0]["tags"]


def test_merge_does_not_drop_payload_or_other_memories():
    ram = [
        {"id": "mem_91hZ", "weight": 0.5, "tags": ["conversation"], "decay_rate": 0.01, "payload": "kitchen stove"},
        {"id": "mem_other", "weight": 0.4, "tags": ["ok"], "decay_rate": 0.01, "payload": "desk"},
    ]
    disk = [{"id": "mem_91hZ", "weight": 0.1, "tags": ["not_fact"], "decay_rate": 0.08, "payload": "should not replace"}]
    out = MemoryPersistence._merge_disk_corrections(ram, disk)
    by_id = {m["id"]: m for m in out}
    assert by_id["mem_91hZ"]["payload"] == "kitchen stove"
    assert "not_fact" in by_id["mem_91hZ"]["tags"]
    assert "conversation" in by_id["mem_91hZ"]["tags"]
    assert by_id["mem_other"]["payload"] == "desk"


def test_load_merges_disk_tags_onto_existing_ram():
    store = MemoryStorage(max_capacity=50)
    store.add(_mem("mem_91hZ", weight=0.5, tags=("conversation",), payload="kitchen stove"))
    loaded = store.load_from_json([
        {
            "id": "mem_91hZ",
            "timestamp": time.time(),
            "weight": 0.15,
            "tags": ["not_fact"],
            "payload": "kitchen stove",
            "type": "conversation",
            "decay_rate": 0.09,
            "provenance": "conversation",
        }
    ])
    assert loaded >= 1
    got = next(m for m in store.get_all() if m.id == "mem_91hZ")
    assert "not_fact" in got.tags
    assert "conversation" in got.tags
    assert got.weight <= 0.15 + 1e-9
    assert got.payload == "kitchen stove"


def test_downweight_flush_and_reload():
    home = Path(tempfile.mkdtemp(prefix="jarvis-persist-"))
    old_home = os.environ.get("HOME")
    from memory.storage import memory_storage
    from memory import persistence as persist_mod
    old_path = persist_mod.memory_persistence._path
    old_memories = list(memory_storage._memories)
    try:
        os.environ["HOME"] = str(home)
        jarvis = home / ".jarvis"
        jarvis.mkdir(parents=True, exist_ok=True)

        persist_mod.memory_persistence._path = str(jarvis / "memories.json")
        memory_storage._memories = []

        mem = _mem("mem_91hZ", weight=0.5, tags=("conversation",))
        memory_storage.add(mem)
        persist_mod.memory_persistence.save()
        assert memory_storage.downweight("mem_91hZ", weight_factor=0.4, decay_rate_factor=2.0)

        on_disk = json.loads((jarvis / "memories.json").read_text())
        row = next(x for x in on_disk if x["id"] == "mem_91hZ")
        assert row["weight"] < 0.5
        assert row["payload"]

        bounced = MemoryStorage(max_capacity=50)
        bounced.load_from_json(on_disk)
        got = next(m for m in bounced.get_all() if m.id == "mem_91hZ")
        assert got.weight < 0.5
        assert got.payload
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        persist_mod.memory_persistence._path = old_path
        memory_storage._memories = old_memories


def test_grounding_clocks_survive_reset():
    home = Path(tempfile.mkdtemp(prefix="jarvis-spark-"))
    old_home = os.environ.get("HOME")
    from autonomy import drives as drives_mod
    old_path = drives_mod.GROUNDING_PROMOTION_PATH
    try:
        os.environ["HOME"] = str(home)
        (home / ".jarvis").mkdir(parents=True, exist_ok=True)

        drives_mod.GROUNDING_PROMOTION_PATH = (home / ".jarvis" / "grounding_drive_promotion.json")
        drives_mod.GroundingDrivePromotion.reset_instance()
        gate = drives_mod.GroundingDrivePromotion.get_instance()
        start_level = gate.level
        gate.note_shadow_selection(
            question="was the kitchen scene a lie?",
            belief_id="mem_91hZ",
            facet="scene",
            channel="operator",
            urgency=0.7,
            verb="ask",
        )
        count = gate.get_status()["selections_shadowed"]
        recent = gate.get_recent_selections()
        assert count >= 1
        assert any("kitchen" in (r.get("question") or "") for r in recent)
        assert gate.level == start_level

        drives_mod.GroundingDrivePromotion.reset_instance()
        bounced = drives_mod.GroundingDrivePromotion.get_instance()
        assert bounced.get_status()["selections_shadowed"] == count
        assert any("kitchen" in (r.get("question") or "") for r in bounced.get_recent_selections())
        assert bounced.level == start_level
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        drives_mod.GROUNDING_PROMOTION_PATH = old_path
        drives_mod.GroundingDrivePromotion.reset_instance()


if __name__ == "__main__":
    test_merge_disk_corrections_keeps_tags_and_lower_weight()
    test_merge_does_not_drop_payload_or_other_memories()
    test_load_merges_disk_tags_onto_existing_ram()
    test_downweight_flush_and_reload()
    test_grounding_clocks_survive_reset()
    print("PASS bounce-durable persist")
