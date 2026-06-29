"""EventBus.on_any global tap — the trusted-routing-source primitive for the cognitive-flow viz.

on_any() must see EVERY fired event (including types with no per-type listener), must be
bulletproof (a throwing observer cannot break emit, siblings, or real handlers), and must
unregister cleanly. These invariants are why the dashboard flow stream can be the trusted
source for ALL routing instead of a curated allowlist.
"""

import consciousness.events as ev


def _fresh_bus():
    bus = ev.EventBus()
    bus.open_barrier()  # otherwise events buffer instead of firing
    return bus


def test_on_any_sees_unlistened_events():
    bus = _fresh_bus()
    seen = []
    bus.on_any(lambda et, kw: seen.append((et, kw.get("x"))))
    bus.on("foo:bar", lambda **k: None)
    bus.emit("foo:bar", x=1)          # has a per-type listener
    bus.emit("never:listened", x=2)   # NO per-type listener
    assert ("foo:bar", 1) in seen
    assert ("never:listened", 2) in seen, "unlistened event missed -> not a trusted source"


def test_on_any_is_bulletproof():
    bus = _fresh_bus()
    survived = []
    handler_ran = []
    bus.on("e:x", lambda **k: handler_ran.append(True))
    bus.on_any(lambda et, kw: (_ for _ in ()).throw(RuntimeError("boom")))  # always raises
    bus.on_any(lambda et, kw: survived.append(et))
    bus.emit("e:x", x=1)
    assert handler_ran == [True], "a throwing observer broke a real handler"
    assert "e:x" in survived, "a throwing observer broke a sibling observer"


def test_on_any_unregisters():
    bus = _fresh_bus()
    seen = []
    unreg = bus.on_any(lambda et, kw: seen.append(et))
    bus.emit("a:b")
    unreg()
    bus.emit("c:d")
    assert seen == ["a:b"], "observer still fired after unregister"


if __name__ == "__main__":
    test_on_any_sees_unlistened_events()
    test_on_any_is_bulletproof()
    test_on_any_unregisters()
    print("ALL on_any tests PASS")
