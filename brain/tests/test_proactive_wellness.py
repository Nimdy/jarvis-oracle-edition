"""#83 small multiplier — cap TTS wellness nagging to 4h."""
from __future__ import annotations

import time
from types import SimpleNamespace

from personality.proactive import ProactiveBehavior


def _mem(typ: str, tags=(), ts: float | None = None):
    return SimpleNamespace(type=typ, tags=tags, timestamp=ts if ts is not None else time.time())


def test_wellness_volume_nags_at_most_every_4h(monkeypatch):
    pb = ProactiveBehavior()
    monkeypatch.setattr(
        "personality.proactive.memory_storage.get_all",
        lambda: [_mem("conversation") for _ in range(16)],
    )
    first = pb._check_wellness(None, 1.0)
    assert first is not None
    assert first.trigger == "high_conversation_volume"
    second = pb._check_wellness(None, 1.0)
    assert second is None
    pb._last_wellness_ts = time.time() - (4 * 3600) - 1
    third = pb._check_wellness(None, 1.0)
    assert third is not None


def test_screen_wellness_also_sets_the_4h_cap(monkeypatch):
    pb = ProactiveBehavior()
    monkeypatch.setattr(
        "personality.proactive.memory_storage.get_all",
        lambda: [_mem("observation", ("screen",)) for _ in range(21)],
    )
    first = pb._check_wellness({"app": "editor"}, 1.0)
    assert first is not None
    assert first.trigger == "extended_screen_time"
    second = pb._check_wellness({"app": "editor"}, 1.0)
    assert second is None
