"""Tests for `/api/hrr/samples` + `get_hrr_samples()` (P4 dashboard surface).

Covers:
* Empty-state shape (no readers registered → all three lists empty, enabled=False).
* Registered readers: `get_hrr_samples()` invokes them with the requested n.
* n-parameter clamping (negative → 0, above-capacity → capacity).
* Authority flags reiterated as False on the samples endpoint too.
* Reader that raises → empty list (safe, no 500).
* Vectors are never emitted by the three shipping owners' `recent()` methods.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _reset_registries():
    import library.vsa.status as S
    S._WORLD_SHADOW_READER = None
    S._SIMULATION_SHADOW_READER = None
    S._RECALL_ADVISORY_READER = None
    S._WORLD_SHADOW_RECENT = None
    S._SIMULATION_SHADOW_RECENT = None
    S._RECALL_ADVISORY_RECENT = None


def test_samples_default_empty_when_disabled():
    _reset_registries()
    from library.vsa.runtime_config import HRRRuntimeConfig
    from library.vsa.status import get_hrr_samples

    payload = get_hrr_samples(config=HRRRuntimeConfig.disabled())
    assert payload["enabled"] is False
    assert payload["world_shadow"] == []
    assert payload["simulation_shadow"] == []
    assert payload["recall_advisory"] == []
    for flag in (
        "policy_influence", "belief_write_enabled", "canonical_memory",
        "autonomy_influence", "llm_raw_vector_exposure", "soul_integrity_influence",
    ):
        assert payload[flag] is False


def test_samples_registered_readers_are_called_with_n():
    _reset_registries()
    from library.vsa.status import (
        get_hrr_samples,
        register_recall_advisory_recent,
        register_simulation_shadow_recent,
        register_world_shadow_recent,
    )

    calls = {"world": None, "sim": None, "rec": None}

    def w(n):
        calls["world"] = n
        return [{"tick": i, "binding_cleanliness": 0.9 + i * 0.01} for i in range(n)]

    def s(n):
        calls["sim"] = n
        return [{"trace_id": i, "side_effects": 0} for i in range(n)]

    def r(n):
        calls["rec"] = n
        return [{"ts": 1000 + i, "overlap_at_k": 0.3} for i in range(n)]

    register_world_shadow_recent(w)
    register_simulation_shadow_recent(s)
    register_recall_advisory_recent(r)

    payload = get_hrr_samples(n_world=5, n_simulation=3, n_recall=2)
    assert calls == {"world": 5, "sim": 3, "rec": 2}
    assert len(payload["world_shadow"]) == 5
    assert len(payload["simulation_shadow"]) == 3
    assert len(payload["recall_advisory"]) == 2

    _reset_registries()


def test_samples_n_is_clamped():
    _reset_registries()
    from library.vsa.status import (
        get_hrr_samples,
        register_world_shadow_recent,
    )

    captured = []
    register_world_shadow_recent(lambda n: captured.append(n) or [])

    get_hrr_samples(n_world=-10, n_simulation=0, n_recall=0)
    assert captured[-1] == 0

    captured.clear()
    get_hrr_samples(n_world=10_000, n_simulation=0, n_recall=0)
    # World capacity is 500; clamp caps at 500.
    assert captured[-1] == 500

    _reset_registries()


def test_samples_reader_exception_yields_empty_list():
    _reset_registries()
    from library.vsa.status import (
        get_hrr_samples,
        register_world_shadow_recent,
    )

    def boom(n):
        raise RuntimeError("explode")

    register_world_shadow_recent(boom)

    payload = get_hrr_samples(n_world=5, n_simulation=0, n_recall=0)
    assert payload["world_shadow"] == []  # no crash, no propagation

    _reset_registries()


def test_samples_reader_non_list_yields_empty():
    _reset_registries()
    from library.vsa.status import (
        get_hrr_samples,
        register_world_shadow_recent,
    )

    register_world_shadow_recent(lambda n: {"not": "a list"})

    payload = get_hrr_samples(n_world=5, n_simulation=0, n_recall=0)
    assert payload["world_shadow"] == []

    _reset_registries()


def test_world_shadow_recent_does_not_emit_raw_vectors():
    """Live wiring check: the world encoder's `recent()` must not leak vectors."""
    _reset_registries()
    import numpy as np

    from cognition.hrr_world_encoder import HRRWorldShadow
    from library.vsa.runtime_config import HRRRuntimeConfig

    cfg = HRRRuntimeConfig(enabled=True, dim=64, sample_every_ticks=1)
    shadow = HRRWorldShadow(cfg)

    class FakeWS:
        facts = [("a", "b", "c"), ("x", "y", "z")]
        def to_dict(self):
            return {"facts": self.facts}

    for _ in range(3):
        shadow.maybe_sample(FakeWS())

    items = shadow.recent(5)
    assert isinstance(items, list)
    for m in items:
        assert isinstance(m, dict)
        for k, v in m.items():
            assert not isinstance(v, np.ndarray), (
                f"world_shadow.recent leaked a vector in key {k!r}"
            )

    _reset_registries()


def test_samples_payload_full_shape_with_registered_readers():
    """End-to-end shape check of `get_hrr_samples` without a live FastAPI stack."""
    _reset_registries()
    from library.vsa.runtime_config import HRRRuntimeConfig
    from library.vsa.status import (
        get_hrr_samples,
        register_recall_advisory_recent,
        register_simulation_shadow_recent,
        register_world_shadow_recent,
    )

    register_world_shadow_recent(
        lambda n: [{"tick": 50, "binding_cleanliness": 0.9, "side_effects": 0}]
    )
    register_simulation_shadow_recent(
        lambda n: [{"trace_id": 1, "binding_cleanliness": 0.85, "side_effects": 0}]
    )
    register_recall_advisory_recent(
        lambda n: [{"ts": 1000, "overlap_at_k": 0.4, "fs_writes": 0}]
    )

    cfg = HRRRuntimeConfig(enabled=True, dim=64, sample_every_ticks=50)
    payload = get_hrr_samples(n_world=5, n_simulation=2, n_recall=1, config=cfg)

    for key in (
        "enabled", "stage", "world_shadow", "simulation_shadow",
        "recall_advisory", "policy_influence", "canonical_memory",
        "belief_write_enabled", "autonomy_influence",
        "llm_raw_vector_exposure", "soul_integrity_influence",
    ):
        assert key in payload

    assert payload["enabled"] is True
    assert payload["stage"] == "shadow_substrate_operational"
    assert payload["canonical_memory"] is False
    assert payload["policy_influence"] is False
    assert len(payload["world_shadow"]) == 1
    assert payload["world_shadow"][0]["side_effects"] == 0
    _reset_registries()


def test_hrr_dashboard_page_file_exists_and_wires_apis():
    """The `/hrr` page is shipped and references both HRR endpoints."""
    from pathlib import Path

    import dashboard  # noqa: F401

    static_dir = Path(__file__).resolve().parents[1] / "dashboard" / "static"
    page = static_dir / "hrr.html"
    assert page.exists(), f"missing dashboard page: {page}"
    body = page.read_text(encoding="utf-8")
    assert "HRR / VSA Shadow Substrate" in body
    assert "/api/hrr/status" in body
    assert "/api/hrr/samples" in body
    assert "holographic_cognition_hrr" in body
    # Public marker must be visibly pinned in the UI text itself.
    assert "PRE-MATURE" in body


def test_hrr_dashboard_route_is_registered_in_app_module():
    """Structural check that `/hrr` and `/api/hrr/samples` routes are wired."""
    from pathlib import Path

    app_src = (
        Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    ).read_text(encoding="utf-8")
    assert '@app.get("/hrr"' in app_src
    assert '@app.get("/api/hrr/samples")' in app_src
    assert '@app.get("/api/hrr/status")' in app_src
