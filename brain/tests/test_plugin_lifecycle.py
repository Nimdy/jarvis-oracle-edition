"""Plugin lifecycle integration tests — the last runtime boundary.

Proves the full quarantine → shadow → supervised → active → disabled lifecycle,
including routing, invocation behavior per state, circuit breaker, rollback,
import validation, and audit trail.

This is an integration test: it uses a real PluginRegistry with a temp directory,
real plugin files on disk, and real invocation. Only persistence (atomic_write_json)
and event_bus are mocked.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.plugin_registry import (
    PluginRegistry,
    PluginManifest,
    PluginRecord,
    PluginRequest,
    PluginResponse,
    CIRCUIT_BREAKER_FAILURES,
    CIRCUIT_BREAKER_WINDOW_S,
    ALWAYS_ALLOWED_IMPORTS,
    NEVER_ALLOWED_IMPORTS,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_clean_plugin_code(name: str = "test_plugin") -> dict[str, str]:
    """Minimal valid plugin code with a sync handler."""
    return {
        "__init__.py": (
            f'PLUGIN_MANIFEST = {{"name": "{name}", "intent_patterns": ["test invoke {name}"]}}\n'
            f'\ndef handle(user_text, context):\n'
            f'    return {{"answer": "hello from {name}"}}\n'
        ),
    }


def _make_failing_plugin_code(name: str = "fail_plugin") -> dict[str, str]:
    """Plugin that always raises."""
    return {
        "__init__.py": (
            f'PLUGIN_MANIFEST = {{"name": "{name}", "intent_patterns": []}}\n'
            f'\ndef handle(user_text, context):\n'
            f'    raise RuntimeError("intentional failure")\n'
        ),
    }


def _make_manifest(name: str, **kwargs) -> PluginManifest:
    kwargs.setdefault("version", "1.0.0")
    return PluginManifest(name=name, **kwargs)


class _RegistryFixture:
    """Creates a PluginRegistry backed by a temp directory."""

    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="jarvis_plugin_test_")
        self.plugins_dir = Path(self.tmpdir) / "plugins"
        self.plugins_dir.mkdir()
        self.audit_dir = Path(self.tmpdir) / "audit"
        self.audit_dir.mkdir()
        self.registry_path = Path(self.tmpdir) / "plugin_registry.json"

    def make_registry(self) -> PluginRegistry:
        self._patches = [
            patch("tools.plugin_registry._REGISTRY_PATH", self.registry_path),
            patch("tools.plugin_registry._AUDIT_DIR", self.audit_dir),
            patch("tools.plugin_registry._PLUGINS_DIR", self.plugins_dir),
        ]
        for p in self._patches:
            p.start()
        reg = PluginRegistry(plugins_dir=self.plugins_dir)
        return reg

    def cleanup(self):
        for p in getattr(self, "_patches", []):
            try:
                p.stop()
            except RuntimeError:
                pass
        shutil.rmtree(self.tmpdir)


def _patch_persistence(fixture):
    """Patch the save/load paths for the registry."""
    return [
        patch("tools.plugin_registry._REGISTRY_PATH", fixture.registry_path),
        patch("tools.plugin_registry._AUDIT_DIR", fixture.audit_dir),
        patch("tools.plugin_registry._PLUGINS_DIR", fixture.plugins_dir),
    ]


# ---------------------------------------------------------------------------
# 1. Quarantine: lands quarantined, not routable
# ---------------------------------------------------------------------------

class TestQuarantineState:
    def test_plugin_lands_quarantined(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("greeter")
            manifest = _make_manifest("greeter")
            ok, errors = reg.quarantine("greeter", code, manifest)
            assert ok is True
            assert errors == []
            rec = reg.get_record("greeter")
            assert rec is not None
            assert rec.state == "quarantined"
        finally:
            fix.cleanup()

    def test_quarantined_not_routable(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("greeter")
            manifest = _make_manifest("greeter", intent_patterns=["greet me"])
            reg.quarantine("greeter", code, manifest)
            assert reg.match("greet me") is None
        finally:
            fix.cleanup()

    def test_quarantined_invoke_fails(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("greeter")
            manifest = _make_manifest("greeter")
            reg.quarantine("greeter", code, manifest)

            req = PluginRequest(plugin_name="greeter", user_text="hello")
            resp = _run(reg.invoke(req))
            assert resp.success is False
            assert "not available" in resp.error.lower()
        finally:
            fix.cleanup()

    def test_invalid_code_rejected(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            bad_code = {"__init__.py": "def foo(:\n    pass\n"}
            manifest = _make_manifest("bad_plugin")
            ok, errors = reg.quarantine("bad_plugin", bad_code, manifest)
            assert ok is False
            assert len(errors) >= 1
            assert any("syntax" in e.lower() for e in errors)
        finally:
            fix.cleanup()

    def test_forbidden_import_rejected(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            evil_code = {"__init__.py": "import subprocess\ndef handle(t, c): pass\n"}
            manifest = _make_manifest("evil_plugin")
            ok, errors = reg.quarantine("evil_plugin", evil_code, manifest)
            assert ok is False
            assert any("forbidden" in e.lower() or "subprocess" in e.lower() for e in errors)
        finally:
            fix.cleanup()

    def test_no_name_in_manifest_rejected(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("unnamed")
            manifest = PluginManifest()  # name=""
            ok, errors = reg.quarantine("unnamed", code, manifest)
            assert ok is False
            assert any("name" in e.lower() for e in errors)
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 2. Activation and promotion: quarantined → shadow → supervised → active
# ---------------------------------------------------------------------------

class TestPromotionLifecycle:
    def _setup(self):
        fix = _RegistryFixture()
        reg = fix.make_registry()
        code = _make_clean_plugin_code("lifecycle_plugin")
        manifest = _make_manifest("lifecycle_plugin", intent_patterns=["test invoke lifecycle_plugin"])
        reg.quarantine("lifecycle_plugin", code, manifest)
        return fix, reg

    def test_activate_to_shadow(self):
        fix, reg = self._setup()
        try:
            assert reg.activate("lifecycle_plugin", "shadow") is True
            rec = reg.get_record("lifecycle_plugin")
            assert rec.state == "shadow"
        finally:
            fix.cleanup()

    def test_promote_shadow_to_supervised(self):
        fix, reg = self._setup()
        try:
            reg.activate("lifecycle_plugin", "shadow")
            assert reg.promote("lifecycle_plugin") is True
            rec = reg.get_record("lifecycle_plugin")
            assert rec.state == "supervised"
        finally:
            fix.cleanup()

    def test_promote_supervised_to_active(self):
        fix, reg = self._setup()
        try:
            reg.activate("lifecycle_plugin", "shadow")
            reg.promote("lifecycle_plugin")
            assert reg.promote("lifecycle_plugin") is True
            rec = reg.get_record("lifecycle_plugin")
            assert rec.state == "active"
        finally:
            fix.cleanup()

    def test_promote_active_fails(self):
        fix, reg = self._setup()
        try:
            reg.activate("lifecycle_plugin", "active")
            assert reg.promote("lifecycle_plugin") is False
        finally:
            fix.cleanup()

    def test_promote_quarantined_fails(self):
        fix, reg = self._setup()
        try:
            assert reg.promote("lifecycle_plugin") is False
        finally:
            fix.cleanup()

    def test_promote_nonexistent_fails(self):
        fix, reg = self._setup()
        try:
            assert reg.promote("ghost") is False
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 3. Shadow invocation: runs handler but nullifies output
# ---------------------------------------------------------------------------

class TestShadowInvocation:
    def test_shadow_invoke_succeeds_but_result_is_none(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("shadow_test")
            manifest = _make_manifest("shadow_test")
            reg.quarantine("shadow_test", code, manifest)
            reg.activate("shadow_test", "shadow")

            req = PluginRequest(plugin_name="shadow_test", user_text="hello")
            resp = _run(reg.invoke(req))
            assert resp.success is True
            assert resp.result is None  # shadow discards output
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 4. Supervised: serves output and logs
# ---------------------------------------------------------------------------

class TestSupervisedInvocation:
    def test_supervised_returns_result(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("sup_test")
            manifest = _make_manifest("sup_test")
            reg.quarantine("sup_test", code, manifest)
            reg.activate("sup_test", "shadow")
            reg.promote("sup_test")  # → supervised

            req = PluginRequest(plugin_name="sup_test", user_text="hello")
            resp = _run(reg.invoke(req))
            assert resp.success is True
            assert resp.result is not None
            assert "answer" in resp.result
        finally:
            fix.cleanup()

    def test_supervised_logs_audit(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("audit_test")
            manifest = _make_manifest("audit_test")
            reg.quarantine("audit_test", code, manifest)
            reg.activate("audit_test", "shadow")
            reg.promote("audit_test")

            req = PluginRequest(plugin_name="audit_test", user_text="hello", request_id="req_001")
            _run(reg.invoke(req))

            audit_file = fix.audit_dir / "audit_test.jsonl"
            assert audit_file.exists()
            lines = audit_file.read_text().strip().split("\n")
            assert len(lines) >= 1
            entry = json.loads(lines[-1])
            assert entry["success"] is True
            assert entry["request_id"] == "req_001"
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 5. Active routing: only active/supervised plugins routable via match()
# ---------------------------------------------------------------------------

class TestRouting:
    def test_active_plugin_matches(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("router_test")
            manifest = _make_manifest("router_test", intent_patterns=["test invoke router_test"])
            reg.quarantine("router_test", code, manifest)
            reg.activate("router_test", "shadow")
            reg.promote("router_test")  # supervised
            reg.promote("router_test")  # active

            result = reg.match("test invoke router_test")
            assert result == "router_test"
        finally:
            fix.cleanup()

    def test_shadow_plugin_not_routable(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("shadow_route")
            manifest = _make_manifest("shadow_route", intent_patterns=["test invoke shadow_route"])
            reg.quarantine("shadow_route", code, manifest)
            reg.activate("shadow_route", "shadow")

            assert reg.match("test invoke shadow_route") is None
        finally:
            fix.cleanup()

    def test_disabled_plugin_not_routable(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("disabled_route")
            manifest = _make_manifest("disabled_route", intent_patterns=["test invoke disabled_route"])
            reg.quarantine("disabled_route", code, manifest)
            reg.activate("disabled_route", "active")
            reg.disable("disabled_route", reason="test")

            assert reg.match("test invoke disabled_route") is None
        finally:
            fix.cleanup()

    def test_no_match_returns_none(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            assert reg.match("something completely unrelated") is None
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 6. Circuit breaker: N failures in window → auto-disable
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_trips_after_consecutive_failures(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_failing_plugin_code("breaker_test")
            manifest = _make_manifest("breaker_test")
            reg.quarantine("breaker_test", code, manifest)
            reg.activate("breaker_test", "active")

            mock_events = MagicMock()
            mock_events.event_bus = MagicMock()
            with patch.dict("sys.modules", {"consciousness.events": mock_events}):
                for i in range(CIRCUIT_BREAKER_FAILURES):
                    req = PluginRequest(plugin_name="breaker_test", user_text="fail")
                    resp = _run(reg.invoke(req))
                    assert resp.success is False

            rec = reg.get_record("breaker_test")
            assert rec.state == "disabled"
        finally:
            fix.cleanup()

    def test_circuit_breaker_window_respected(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_failing_plugin_code("window_test")
            manifest = _make_manifest("window_test")
            reg.quarantine("window_test", code, manifest)
            reg.activate("window_test", "active")

            rec = reg.get_record("window_test")
            old_time = time.time() - CIRCUIT_BREAKER_WINDOW_S - 100
            rec.recent_failures = [old_time, old_time + 1]

            mock_events = MagicMock()
            mock_events.event_bus = MagicMock()
            with patch.dict("sys.modules", {"consciousness.events": mock_events}):
                req = PluginRequest(plugin_name="window_test", user_text="fail")
                _run(reg.invoke(req))

            rec = reg.get_record("window_test")
            assert rec.state == "active", "Old failures outside window should not count"
        finally:
            fix.cleanup()

    def test_successful_invocations_dont_trigger_breaker(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("healthy_test")
            manifest = _make_manifest("healthy_test")
            reg.quarantine("healthy_test", code, manifest)
            reg.activate("healthy_test", "active")

            for _ in range(10):
                req = PluginRequest(plugin_name="healthy_test", user_text="hello")
                resp = _run(reg.invoke(req))
                assert resp.success is True

            rec = reg.get_record("healthy_test")
            assert rec.state == "active"
            assert rec.success_count == 10
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 7. Rollback: restores prior version/state
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_restores_to_quarantined(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("rollback_test")
            manifest = _make_manifest("rollback_test", version="2.0.0")
            reg.quarantine("rollback_test", code, manifest)
            reg.activate("rollback_test", "active")

            rec = reg.get_record("rollback_test")
            rec.fallback_version = "1.0.0"

            assert reg.rollback("rollback_test") is True
            rec = reg.get_record("rollback_test")
            assert rec.state == "quarantined"
            assert rec.version == "1.0.0"
        finally:
            fix.cleanup()

    def test_rollback_without_fallback_fails(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("no_fallback")
            manifest = _make_manifest("no_fallback")
            reg.quarantine("no_fallback", code, manifest)
            assert reg.rollback("no_fallback") is False
        finally:
            fix.cleanup()

    def test_rollback_nonexistent_fails(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            assert reg.rollback("ghost_plugin") is False
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 8. Disable: manual or circuit breaker
# ---------------------------------------------------------------------------

class TestDisable:
    def test_disable_removes_handler(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("disable_test")
            manifest = _make_manifest("disable_test")
            reg.quarantine("disable_test", code, manifest)
            reg.activate("disable_test", "active")

            assert reg.get_handler("disable_test") is not None
            reg.disable("disable_test", reason="test")
            assert reg.get_handler("disable_test") is None

            rec = reg.get_record("disable_test")
            assert rec.state == "disabled"
        finally:
            fix.cleanup()

    def test_disabled_invoke_fails(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("dis_invoke")
            manifest = _make_manifest("dis_invoke")
            reg.quarantine("dis_invoke", code, manifest)
            reg.activate("dis_invoke", "active")
            reg.disable("dis_invoke")

            req = PluginRequest(plugin_name="dis_invoke", user_text="hello")
            resp = _run(reg.invoke(req))
            assert resp.success is False
            assert "not available" in resp.error.lower()
        finally:
            fix.cleanup()

    def test_disable_logs_audit(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = _make_clean_plugin_code("audit_dis")
            manifest = _make_manifest("audit_dis")
            reg.quarantine("audit_dis", code, manifest)
            reg.activate("audit_dis", "active")
            reg.disable("audit_dis", reason="manual_test")

            audit_file = fix.audit_dir / "audit_dis.jsonl"
            assert audit_file.exists()
            lines = audit_file.read_text().strip().split("\n")
            last = json.loads(lines[-1])
            assert last["action"] == "disabled"
            assert last["reason"] == "manual_test"
        finally:
            fix.cleanup()


# ---------------------------------------------------------------------------
# 9. Import allowlists
# ---------------------------------------------------------------------------

class TestImportAllowlists:
    def test_always_allowed_pass(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            code = {"__init__.py": "import json\nimport re\ndef handle(t, c): return {'ok': True}\n"}
            manifest = _make_manifest("safe_imports")
            ok, errors = reg.quarantine("safe_imports", code, manifest)
            assert ok is True
        finally:
            fix.cleanup()

    def test_never_allowed_blocked(self):
        for mod in ["subprocess", "ctypes", "socket"]:
            fix = _RegistryFixture()
            try:
                reg = fix.make_registry()
                code = {"__init__.py": f"import {mod}\ndef handle(t, c): pass\n"}
                manifest = _make_manifest(f"bad_{mod}")
                ok, errors = reg.quarantine(f"bad_{mod}", code, manifest)
                assert ok is False, f"Expected {mod} to be blocked"
                assert any(mod in e.lower() or "forbidden" in e.lower() for e in errors)
            finally:
                fix.cleanup()


# ---------------------------------------------------------------------------
# 10. get_status() and PluginRecord serialization
# ---------------------------------------------------------------------------

class TestStatusAndSerialization:
    def test_get_status_shape(self):
        fix = _RegistryFixture()
        try:
            reg = fix.make_registry()
            status = reg.get_status()
            assert "total_plugins" in status
            assert "by_state" in status
            assert "plugins" in status
            assert "routes" in status
        finally:
            fix.cleanup()

    def test_record_roundtrip(self):
        rec = PluginRecord(
            name="test", state="active", version="2.0",
            invocation_count=5, success_count=4, failure_count=1,
        )
        d = rec.to_dict()
        restored = PluginRecord.from_dict(d)
        assert restored.name == "test"
        assert restored.state == "active"
        assert restored.invocation_count == 5

    def test_manifest_roundtrip(self):
        m = PluginManifest(
            name="test", version="1.0", description="desc",
            intent_patterns=["hello"], risk_tier=2,
        )
        d = m.to_dict()
        restored = PluginManifest.from_dict(d)
        assert restored.name == "test"
        assert restored.intent_patterns == ["hello"]

    def test_constants_match_docs(self):
        assert CIRCUIT_BREAKER_FAILURES == 3
        assert CIRCUIT_BREAKER_WINDOW_S == 600
        assert "subprocess" in NEVER_ALLOWED_IMPORTS
        assert "json" in ALWAYS_ALLOWED_IMPORTS
