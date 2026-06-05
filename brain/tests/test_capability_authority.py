"""Capability authority (shadow-first, reversible) — general, skill-agnostic.

Verifies the doctrine in docs/CAPABILITY_AUTHORITY_DESIGN.md against the registry
mechanism, fully isolated from the live registry (temp paths).
"""
import tempfile
from pathlib import Path


def _fresh_registry():
    import tools.plugin_registry as pr
    tmp = Path(tempfile.mkdtemp())
    pr._REGISTRY_PATH = tmp / "plugin_registry.json"
    pr._AUDIT_DIR = tmp / "plugin_audit"
    pr._AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    reg = pr.PluginRegistry(plugins_dir=tmp / "plugins")
    return pr, reg


def _add(reg, pr, name, skill_id, state="shadow"):
    reg._records[name] = pr.PluginRecord(name=name, skill_id=skill_id, state=state)
    return reg._records[name]


def test_one_active_per_skill_and_floor_recorded():
    pr, reg = _fresh_registry()
    _add(reg, pr, "v1", "skill_x")
    _add(reg, pr, "v2", "skill_x")

    assert reg.make_authoritative("v1", approved_by="owner")
    assert reg._records["v1"].state == "active"
    assert reg.active_for_skill("skill_x").name == "v1"

    # promoting v2 atomically displaces v1 to shadow and records it as the floor
    assert reg.make_authoritative("v2", approved_by="owner")
    assert reg._records["v2"].state == "active"
    assert reg._records["v1"].state == "shadow"            # only ONE active per skill
    assert reg._records["v2"].prior_authoritative == "v1"
    assert reg._records["v1"].last_authoritative_at > 0
    print("  PASS: one active per skill + floor recorded")


def test_demote_falls_back_to_known_good():
    pr, reg = _fresh_registry()
    _add(reg, pr, "v1", "skill_x")
    _add(reg, pr, "v2", "skill_x")
    reg.make_authoritative("v1", approved_by="owner")
    reg.make_authoritative("v2", approved_by="owner")      # v1 is now the floor

    rep = reg.demote("v2", reason="test", actor="owner")
    assert rep["ok"] and rep["fell_back_to"] == "v1" and not rep["dormant"]
    assert reg._records["v2"].state == "shadow"
    assert reg.active_for_skill("skill_x").name == "v1"    # skill keeps serving on v1
    print("  PASS: demote restores the known-good floor")


def test_demote_single_version_goes_dormant():
    pr, reg = _fresh_registry()
    _add(reg, pr, "solo", "skill_solo")
    reg.make_authoritative("solo", approved_by="owner")

    rep = reg.demote("solo", reason="test", actor="owner")
    assert rep["ok"] and rep["dormant"] and rep["fell_back_to"] is None
    assert reg.active_for_skill("skill_solo") is None       # dormant = safe, not wrong
    assert reg._records["solo"].state == "shadow"
    print("  PASS: lone version demotes to dormant (safe)")


def test_demote_is_not_owner_gated_actor_recorded():
    pr, reg = _fresh_registry()
    _add(reg, pr, "v1", "skill_x")
    reg.make_authoritative("v1", approved_by="owner")
    # an autonomous actor can demote (lowering authority is always safe)
    rep = reg.demote("v1", reason="self-protect", actor="auto:immune")
    assert rep["ok"] and rep["actor"] == "auto:immune"
    print("  PASS: autonomous demote allowed")


def test_circuit_breaker_auto_demotes_live_capability():
    pr, reg = _fresh_registry()
    _add(reg, pr, "good", "skill_y")
    _add(reg, pr, "bad", "skill_y")
    reg.make_authoritative("good", approved_by="owner")
    reg.make_authoritative("bad", approved_by="owner")      # bad live, good is floor
    assert reg.active_for_skill("skill_y").name == "bad"

    # trip the breaker on the LIVE plugin -> auto-demote to shadow + restore floor
    rec = reg._records["bad"]
    for _ in range(pr.CIRCUIT_BREAKER_FAILURES):
        reg._record_failure(rec)
    assert reg._records["bad"].state == "shadow"            # not "disabled" anymore
    assert reg.active_for_skill("skill_y").name == "good"   # floor restored autonomously
    print("  PASS: circuit breaker auto-demotes live capability + falls back")


def test_shadow_crasher_still_disabled():
    pr, reg = _fresh_registry()
    rec = _add(reg, pr, "sh", "skill_z", state="shadow")    # never authoritative
    for _ in range(pr.CIRCUIT_BREAKER_FAILURES):
        reg._record_failure(rec)
    assert reg._records["sh"].state == "disabled"           # non-authoritative crasher -> disabled
    print("  PASS: shadow repeat-crasher still disabled")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("ALL CAPABILITY-AUTHORITY TESTS PASS")
