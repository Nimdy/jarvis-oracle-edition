"""Tests for ``GET /api/self-test``.

The endpoint consolidates several truth signals into one response:

  - cache readiness (503 when the snapshot loop has not produced a cache)
  - engine liveness
  - serializer shape invariants (specifically the ``specialists`` block)
  - attestation ledger vs ``prior_attested_ok`` claim consistency
  - validation pack delegation

These tests lock the contract without spinning up the real engine. They
operate on ``dashboard.app._create_app()`` + a FastAPI TestClient and
patch the module-level ``_cache`` / ``_engine`` / ``_cache_time``
globals directly — the endpoint is intentionally a pure read of those
globals so this is sufficient coverage.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

# Heavy import: pulls in ConsciousnessEngine, ResponseGenerator, etc.
# That is expected on the desktop brain where the full runtime deps live.
# On leaner environments the importorskip below keeps collection sane.
try:
    from dashboard import app as dashboard_app
except Exception as exc:  # pragma: no cover - import gating only
    pytest.skip(f"dashboard.app import unavailable: {exc}", allow_module_level=True)


@pytest.fixture
def client(monkeypatch):
    """Fresh TestClient with all module globals reset."""
    monkeypatch.setattr(dashboard_app, "_cache", {}, raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_hash", "", raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", 0.0, raising=True)
    monkeypatch.setattr(dashboard_app, "_engine", None, raising=True)
    app = dashboard_app._create_app()
    return TestClient(app)


def _healthy_cache() -> dict:
    return {
        "_ts": time.time(),
        "self_improve": {
            "active": True,
            "specialists": {
                "specialists": [],
                "distillation": {"total_signals": 0, "total_quarantined": 0},
            },
        },
        "autonomy": {
            "attestation": {
                "prior_attested_ok": False,
                "attestation_strength": "none",
                "records": [],
            }
        },
        "eval": {
            "pvl": {},
            "maturity_tracker": {},
            "language": {},
        },
    }


def test_returns_503_when_cache_empty(client, monkeypatch):
    resp = client.get("/api/self-test")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["status"] == "not_ready"
    assert "cache_ready" in body["checks"]
    assert body["checks"]["cache_ready"]["ok"] is False


def test_returns_200_with_healthy_cache_shape(client, monkeypatch):
    monkeypatch.setattr(dashboard_app, "_cache", _healthy_cache(), raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", time.time(), raising=True)

    resp = client.get("/api/self-test")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "checks" in body
    checks = body["checks"]
    assert set(checks.keys()) >= {
        "cache_ready",
        "engine_alive",
        "serializer_shape",
        "attestation_ledger",
        "validation_pack",
    }
    assert checks["cache_ready"]["ok"] is True
    assert checks["serializer_shape"]["ok"] is True


def test_serializer_shape_fails_when_specialists_raw_empty(client, monkeypatch):
    bad_cache = _healthy_cache()
    bad_cache["self_improve"]["specialists"] = {}
    monkeypatch.setattr(dashboard_app, "_cache", bad_cache, raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", time.time(), raising=True)

    resp = client.get("/api/self-test")
    assert resp.status_code == 200
    body = resp.json()
    shape = body["checks"]["serializer_shape"]
    assert shape["ok"] is False
    assert "list" in shape["detail"] or "dict" in shape["detail"]


def test_serializer_shape_fails_on_degraded_marker(client, monkeypatch):
    degraded = _healthy_cache()
    degraded["self_improve"]["specialists"] = {
        "specialists": [],
        "distillation": {},
        "_error": "RuntimeError",
    }
    monkeypatch.setattr(dashboard_app, "_cache", degraded, raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", time.time(), raising=True)

    resp = client.get("/api/self-test")
    body = resp.json()
    shape = body["checks"]["serializer_shape"]
    assert shape["ok"] is False
    assert "RuntimeError" in shape["detail"]


def test_attestation_check_claim_without_ledger(client, monkeypatch, tmp_path):
    """If prior_attested_ok is True but no ledger exists, check fails."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = _healthy_cache()
    cache["autonomy"]["attestation"]["prior_attested_ok"] = True
    monkeypatch.setattr(dashboard_app, "_cache", cache, raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", time.time(), raising=True)

    resp = client.get("/api/self-test")
    body = resp.json()
    att = body["checks"]["attestation_ledger"]
    assert att["ok"] is False
    assert "missing" in att["detail"].lower()


def test_attestation_check_passes_when_claim_and_ledger_agree(
    client, monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    eval_dir = tmp_path / ".jarvis" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "ever_proven_attestation.json").write_text(
        '[{"capability_id": "autonomy.l3"}]', encoding="utf-8"
    )

    cache = _healthy_cache()
    cache["autonomy"]["attestation"]["prior_attested_ok"] = True
    monkeypatch.setattr(dashboard_app, "_cache", cache, raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", time.time(), raising=True)

    resp = client.get("/api/self-test")
    body = resp.json()
    att = body["checks"]["attestation_ledger"]
    assert att["ok"] is True


def test_response_includes_snapshot_ts(client, monkeypatch):
    cache = _healthy_cache()
    stamp = 1_700_000_000.25
    cache["_ts"] = stamp
    monkeypatch.setattr(dashboard_app, "_cache", cache, raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", time.time(), raising=True)

    resp = client.get("/api/self-test")
    body = resp.json()
    assert body["snapshot_ts"] == stamp


def test_is_readonly_cache_not_mutated(client, monkeypatch):
    cache = _healthy_cache()
    before_keys = set(cache.keys())
    monkeypatch.setattr(dashboard_app, "_cache", cache, raising=True)
    monkeypatch.setattr(dashboard_app, "_cache_time", time.time(), raising=True)

    client.get("/api/self-test")
    client.get("/api/self-test")

    assert set(cache.keys()) == before_keys, (
        "/api/self-test must be side-effect-free; cache was mutated"
    )
