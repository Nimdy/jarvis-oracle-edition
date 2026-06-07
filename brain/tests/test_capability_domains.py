"""Matrix v2 Phase 1 — Capability Domain isolation + clean deletion.

The brain-injury analogy as an automated guarantee: deleting a domain removes ONLY
that domain's data (its isolated dir), with zero residue in sibling domains or the
registry root. Isolated stores live under each domain's own root_dir.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cognition.capability_domains import CapabilityDomainRegistry


def _reg():
    return CapabilityDomainRegistry(root=Path(tempfile.mkdtemp()) / "domains")


def test_create_makes_isolated_dir_and_paths():
    r = _reg()
    d = r.create("Robot Arm xArm6", kind="physical")
    assert d.domain_id.startswith("dom_robot_arm_xarm6_")
    assert d.kind == "physical" and d.status == "created"
    assert Path(d.root_dir).is_dir()
    # isolated store paths live UNDER the domain's own dir
    assert Path(d.knowledge_db).parent == Path(d.root_dir)
    assert Path(d.memory_path).parent == Path(d.root_dir)
    assert r.get(d.domain_id) is d


def test_list_and_persistence_across_instances():
    root = Path(tempfile.mkdtemp()) / "domains"
    r1 = CapabilityDomainRegistry(root=root)
    a = r1.create("Alpha")
    b = r1.create("Beta")
    assert {d.domain_id for d in r1.list()} == {a.domain_id, b.domain_id}
    # a fresh registry on the same root restores them
    r2 = CapabilityDomainRegistry(root=root)
    assert {d.domain_id for d in r2.list()} == {a.domain_id, b.domain_id}


def test_clean_deletion_zero_residue_siblings_intact():
    r = _reg()
    a = r.create("Snowboarding")
    b = r.create("Cooking")
    # write isolated data into each domain
    (Path(a.root_dir) / "knowledge.db").write_text("snowboard facts")
    (Path(b.root_dir) / "knowledge.db").write_text("cooking facts")

    assert r.delete(a.domain_id) is True
    # A is gone with zero residue...
    assert r.get(a.domain_id) is None
    assert not Path(a.root_dir).exists()
    # ...and B (sibling) is completely intact — "forget snowboarding, keep cooking"
    assert r.get(b.domain_id) is not None
    assert Path(b.root_dir).is_dir()
    assert (Path(b.root_dir) / "knowledge.db").read_text() == "cooking facts"


def test_deletion_persists_and_unknown_is_false():
    root = Path(tempfile.mkdtemp()) / "domains"
    r1 = CapabilityDomainRegistry(root=root)
    a = r1.create("Temp")
    assert r1.delete(a.domain_id) is True
    assert r1.delete("dom_does_not_exist") is False
    # deletion survives reload (registry.json updated)
    r2 = CapabilityDomainRegistry(root=root)
    assert r2.get(a.domain_id) is None
    assert r2.list() == []


def test_registry_root_untouched_by_deletion():
    r = _reg()
    a = r.create("X")
    r.delete(a.domain_id)
    # the registry root + index file remain (only the domain dir was removed)
    assert r._root.is_dir()
    assert (r._root / "registry.json").exists()
    idx = json.loads((r._root / "registry.json").read_text())
    assert idx["domains"] == []


def test_status_shape():
    r = _reg()
    r.create("One")
    r.create("Two")
    s = r.status()
    assert s["count"] == 2
    assert s["by_status"].get("created") == 2
    assert len(s["domains"]) == 2
    # public view leaks no filesystem paths
    assert "root_dir" not in s["domains"][0]
    assert "knowledge_db" not in s["domains"][0]
