"""Tests for Phase 7: Tiered Plugin Isolation.

Covers:
  - Backward compatibility (from_dict defaults for new fields)
  - Validation rules (execution_mode, pinned_dependencies, setup_commands)
  - Subprocess invoke path routing
  - Environment hardening (env var stripping)
  - PluginProcessManager lifecycle
  - Environment setup lane
  - Child wrapper (plugin_runner_child.py)
  - Singleton accessor
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tools.plugin_registry import (
    PluginManifest,
    PluginRecord,
    PluginRegistry,
    PluginRequest,
    PluginResponse,
    get_plugin_registry,
)

_loop = asyncio.new_event_loop()

def _run(coro):
    return _loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tmpdir() -> str:
    return tempfile.mkdtemp(prefix="jarvis_test_iso_")


def _make_registry(tmpdir: str) -> PluginRegistry:
    """Create an isolated PluginRegistry."""
    plugins_dir = Path(tmpdir) / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    reg_path = Path(tmpdir) / "plugin_registry.json"
    audit_dir = Path(tmpdir) / "plugin_audit"
    audit_dir.mkdir(exist_ok=True)

    with patch("tools.plugin_registry._REGISTRY_PATH", reg_path), \
         patch("tools.plugin_registry._AUDIT_DIR", audit_dir):
        reg = PluginRegistry(plugins_dir=plugins_dir)
    reg._reg_path = reg_path
    reg._audit_dir = audit_dir
    return reg


def _make_manifest(**overrides) -> PluginManifest:
    defaults = dict(
        name="test_plugin",
        version="1.0.0",
        description="Test plugin",
        keywords=["test"],
        supervision_mode="shadow",
        risk_tier=0,
    )
    defaults.update(overrides)
    return PluginManifest(**defaults)


def _stub_code_in_process(handler_body: str = 'return {"ok": True}') -> dict[str, str]:
    return {
        "__init__.py": (
            'PLUGIN_MANIFEST = {"name": "test_plugin", "intent_patterns": []}\n\n'
            f'async def handle(text: str, context: dict) -> dict:\n'
            f'    {handler_body}\n'
        ),
    }


def _stub_code_isolated() -> dict[str, str]:
    return {
        "__init__.py": (
            'PLUGIN_MANIFEST = {\n'
            '    "name": "iso_plugin",\n'
            '    "intent_patterns": ["\\\\biso\\\\b"],\n'
            '    "execution_mode": "isolated_subprocess",\n'
            '    "pinned_dependencies": ["requests==2.31.0"],\n'
            '}\n\n'
            'def handle(text: str, context: dict) -> dict:\n'
            '    return {"output": f"hello from isolated: {text}"}\n'
        ),
    }


# ===========================================================================
# 1. BACKWARD COMPATIBILITY
# ===========================================================================

class TestBackwardCompatibility:
    """New fields have safe defaults; old serialized data still loads."""

    def test_manifest_defaults(self):
        m = PluginManifest(name="old_plugin")
        assert m.execution_mode == "in_process"
        assert m.pinned_dependencies == []
        assert m.invocation_schema_version == "1"

    def test_record_defaults(self):
        r = PluginRecord(name="old_plugin")
        assert r.execution_mode == "in_process"
        assert r.venv_ready is False

    def test_manifest_from_dict_missing_new_fields(self):
        old_data = {"name": "legacy", "version": "0.9.0"}
        m = PluginManifest(**{k: v for k, v in old_data.items()
                              if k in {f.name for f in PluginManifest.__dataclass_fields__.values()}})
        assert m.execution_mode == "in_process"
        assert m.pinned_dependencies == []

    def test_record_from_dict_missing_new_fields(self):
        old_data = {"name": "legacy", "state": "active", "version": "0.5.0"}
        known = {f.name for f in PluginRecord.__dataclass_fields__.values()}
        r = PluginRecord(**{k: v for k, v in old_data.items() if k in known})
        assert r.execution_mode == "in_process"
        assert r.venv_ready is False

    def test_in_process_plugin_still_works(self):
        """Existing in-process plugins are unaffected by Phase 7 additions."""
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest()
            ok, errors = reg.quarantine("test_plugin", code, manifest)
            assert ok
            assert not errors
            rec = reg.get_record("test_plugin")
            assert rec.execution_mode == "in_process"
            assert rec.venv_ready is False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 2. VALIDATION RULES
# ===========================================================================

class TestValidationRules:

    def test_valid_in_process(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest(execution_mode="in_process")
            ok, errors = reg.quarantine("test_plugin", code, manifest)
            assert ok
            assert not errors
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_execution_mode(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest(execution_mode="docker")
            ok, errors = reg.quarantine("bad_mode", code, manifest)
            assert not ok
            assert any("execution_mode" in e for e in errors)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_in_process_with_pinned_deps_rejected(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest(
                execution_mode="in_process",
                pinned_dependencies=["requests==2.31.0"],
            )
            ok, errors = reg.quarantine("bad_deps", code, manifest)
            assert not ok
            assert any("pinned_dependencies" in e for e in errors)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_floating_version_rejected(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest(
                name="float_dep",
                execution_mode="isolated_subprocess",
                pinned_dependencies=["requests>=2.0"],
            )
            ok, errors = reg.quarantine("float_dep", code, manifest)
            assert not ok
            assert any("exact pin" in e for e in errors)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_exact_pin_accepted(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = {
                "__init__.py": (
                    'PLUGIN_MANIFEST = {"name": "pinned_ok", "intent_patterns": []}\n'
                    'def handle(text, context): return {"ok": True}\n'
                ),
            }
            manifest = _make_manifest(
                name="pinned_ok",
                execution_mode="isolated_subprocess",
                pinned_dependencies=["requests==2.31.0", "beautifulsoup4==4.12.2"],
            )
            ok, errors = reg.quarantine("pinned_ok", code, manifest)
            assert ok, f"Unexpected errors: {errors}"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 3. INVOKE ROUTING
# ===========================================================================

class TestInvokeRouting:

    def test_in_process_invoke_uses_handler(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest()
            ok, _ = reg.quarantine("test_plugin", code, manifest)
            assert ok

            rec = reg.get_record("test_plugin")
            rec.state = "active"
            reg._try_load_handler("test_plugin")

            req = PluginRequest(
                request_id="r1",
                plugin_name="test_plugin",
                user_text="hello",
                context={},
            )
            resp = _run(reg.invoke(req))
            assert resp.success
            assert resp.result == {"ok": True}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_isolated_plugin_not_in_handlers(self):
        """isolated_subprocess plugins must never populate _handlers."""
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_isolated()
            manifest = _make_manifest(
                name="iso_plugin",
                execution_mode="isolated_subprocess",
                pinned_dependencies=["requests==2.31.0"],
            )
            ok, _ = reg.quarantine("iso_plugin", code, manifest)
            assert ok
            rec = reg.get_record("iso_plugin")
            rec.state = "active"
            reg._try_load_handler("iso_plugin")
            assert "iso_plugin" not in reg._handlers
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_isolated_invoke_calls_subprocess_path(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_isolated()
            manifest = _make_manifest(
                name="iso_plugin",
                execution_mode="isolated_subprocess",
                pinned_dependencies=["requests==2.31.0"],
            )
            ok, _ = reg.quarantine("iso_plugin", code, manifest)
            assert ok
            rec = reg.get_record("iso_plugin")
            rec.state = "active"

            mock_mgr = MagicMock()
            mock_mgr.invoke = AsyncMock(return_value={
                "request_id": "r1",
                "success": True,
                "result": {"output": "subprocess output"},
                "error": None,
            })
            mock_mgr.venv_ready = True
            mock_mgr.is_running = True
            reg._process_managers["iso_plugin"] = mock_mgr

            req = PluginRequest(
                request_id="r1",
                plugin_name="iso_plugin",
                user_text="test",
                context={},
            )
            resp = _run(reg.invoke(req))
            assert resp.success
            assert resp.result == {"output": "subprocess output"}
            mock_mgr.invoke.assert_called_once()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 4. ENVIRONMENT HARDENING
# ===========================================================================

class TestEnvironmentHardening:

    def test_stripped_env_vars(self):
        from tools.plugin_process import _build_clean_env

        test_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "JARVIS_GPU_TIER": "premium",
            "OLLAMA_HOST": "http://localhost",
            "ANTHROPIC_API_KEY": "sk-secret",
            "OPENAI_API_KEY": "sk-other",
            "MY_SECRET": "hidden",
            "MY_TOKEN": "tok123",
            "DASHBOARD_API_KEY": "abc",
            "S2_API_KEY": "s2key",
            "SAFE_VAR": "visible",
        }

        with patch.dict(os.environ, test_env, clear=True):
            clean = _build_clean_env()

        assert "PATH" in clean
        assert "HOME" in clean
        assert "SAFE_VAR" in clean
        assert "JARVIS_GPU_TIER" not in clean
        assert "OLLAMA_HOST" not in clean
        assert "ANTHROPIC_API_KEY" not in clean
        assert "OPENAI_API_KEY" not in clean
        assert "MY_SECRET" not in clean
        assert "MY_TOKEN" not in clean
        assert "DASHBOARD_API_KEY" not in clean
        assert "S2_API_KEY" not in clean

    def test_no_brain_imports_in_child_wrapper(self):
        """plugin_runner_child.py must not import any brain modules."""
        child_path = Path(__file__).parent.parent / "tools" / "plugin_runner_child.py"
        content = child_path.read_text()
        assert "from brain" not in content
        assert "import brain" not in content
        assert "from consciousness" not in content
        assert "from tools.plugin_registry" not in content
        assert "from reasoning" not in content


# ===========================================================================
# 5. PLUGIN PROCESS MANAGER
# ===========================================================================

class TestPluginProcessManager:

    def test_init_state(self):
        from tools.plugin_process import PluginProcessManager
        tmpdir = _make_tmpdir()
        try:
            mgr = PluginProcessManager(
                plugin_name="test",
                plugin_dir=Path(tmpdir),
                pinned_dependencies=["requests==2.31.0"],
            )
            assert not mgr.venv_ready
            assert not mgr.is_running
            status = mgr.get_status()
            assert status["execution_mode"] == "isolated_subprocess"
            assert status["venv_ready"] is False
            assert status["child_running"] is False
            assert status["pinned_dependencies"] == ["requests==2.31.0"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_status_shape(self):
        from tools.plugin_process import PluginProcessManager
        tmpdir = _make_tmpdir()
        try:
            mgr = PluginProcessManager(
                plugin_name="shape_test",
                plugin_dir=Path(tmpdir),
            )
            status = mgr.get_status()
            required_keys = {
                "plugin_name", "execution_mode", "venv_ready", "venv_path",
                "child_running", "child_pid", "invocation_count",
                "pinned_dependencies", "idle_timeout_s",
            }
            assert required_keys.issubset(set(status.keys()))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_verify_imports_override_accepted(self):
        """Manifest may declare an explicit verify_imports list for the
        dist-name != import-name case (e.g. python-dateutil -> dateutil,
        PyYAML -> yaml). The constructor accepts and stores it."""
        from tools.plugin_process import PluginProcessManager
        tmpdir = _make_tmpdir()
        try:
            mgr = PluginProcessManager(
                plugin_name="dist_vs_import",
                plugin_dir=Path(tmpdir),
                pinned_dependencies=["python-dateutil==2.9.0"],
                verify_imports=["dateutil"],
            )
            assert mgr._verify_imports_override == ["dateutil"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_verify_imports_override_empty_defaults_to_heuristic(self):
        """Omitted/empty verify_imports means use the dist-name heuristic."""
        from tools.plugin_process import PluginProcessManager
        tmpdir = _make_tmpdir()
        try:
            mgr_default = PluginProcessManager(
                plugin_name="heur_default",
                plugin_dir=Path(tmpdir),
                pinned_dependencies=["requests==2.31.0"],
            )
            mgr_empty = PluginProcessManager(
                plugin_name="heur_empty",
                plugin_dir=Path(tmpdir),
                pinned_dependencies=["requests==2.31.0"],
                verify_imports=[],
            )
            assert mgr_default._verify_imports_override == []
            assert mgr_empty._verify_imports_override == []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 6. CHILD WRAPPER (plugin_runner_child.py)
# ===========================================================================

class TestChildWrapper:

    def test_child_happy_path(self):
        """Child loads plugin, handles request, returns JSON."""
        tmpdir = _make_tmpdir()
        try:
            plugin_dir = Path(tmpdir) / "test_child_plugin"
            plugin_dir.mkdir()
            init_py = plugin_dir / "__init__.py"
            init_py.write_text(
                'def handle(text, context):\n'
                '    return {"output": f"echo: {text}"}\n'
            )

            child_script = Path(__file__).parent.parent / "tools" / "plugin_runner_child.py"
            req = json.dumps({"request_id": "c1", "user_text": "ping", "context": {}})
            shutdown = json.dumps({"action": "shutdown"})
            stdin_data = f"{req}\n{shutdown}\n"

            result = subprocess.run(
                [sys.executable, str(child_script), str(plugin_dir)],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"

            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            assert len(lines) >= 1
            resp = json.loads(lines[0])
            assert resp["success"] is True
            assert resp["result"]["output"] == "echo: ping"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_child_bad_json_request(self):
        tmpdir = _make_tmpdir()
        try:
            plugin_dir = Path(tmpdir) / "bad_json_plugin"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text(
                'def handle(text, context): return {"ok": True}\n'
            )
            child_script = Path(__file__).parent.parent / "tools" / "plugin_runner_child.py"

            shutdown = json.dumps({"action": "shutdown"})
            stdin_data = f"not-valid-json\n{shutdown}\n"

            result = subprocess.run(
                [sys.executable, str(child_script), str(plugin_dir)],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            assert len(lines) >= 1
            resp = json.loads(lines[0])
            assert resp["success"] is False
            assert "Invalid JSON" in resp["error"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_child_handler_exception(self):
        tmpdir = _make_tmpdir()
        try:
            plugin_dir = Path(tmpdir) / "err_plugin"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text(
                'def handle(text, context): raise ValueError("boom")\n'
            )
            child_script = Path(__file__).parent.parent / "tools" / "plugin_runner_child.py"

            req = json.dumps({"request_id": "e1", "user_text": "x", "context": {}})
            shutdown = json.dumps({"action": "shutdown"})
            stdin_data = f"{req}\n{shutdown}\n"

            result = subprocess.run(
                [sys.executable, str(child_script), str(plugin_dir)],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            assert len(lines) >= 1
            resp = json.loads(lines[0])
            assert resp["success"] is False
            assert "boom" in resp["error"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_child_missing_handler(self):
        tmpdir = _make_tmpdir()
        try:
            plugin_dir = Path(tmpdir) / "no_handler"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text('x = 1\n')
            child_script = Path(__file__).parent.parent / "tools" / "plugin_runner_child.py"

            result = subprocess.run(
                [sys.executable, str(child_script), str(plugin_dir)],
                input="",
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 1
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            assert len(lines) >= 1
            resp = json.loads(lines[0])
            assert resp["success"] is False
            assert "no handle()" in resp["error"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_child_shutdown_command(self):
        tmpdir = _make_tmpdir()
        try:
            plugin_dir = Path(tmpdir) / "shutdown_test"
            plugin_dir.mkdir()
            (plugin_dir / "__init__.py").write_text(
                'def handle(text, context): return {"ok": True}\n'
            )
            child_script = Path(__file__).parent.parent / "tools" / "plugin_runner_child.py"

            shutdown = json.dumps({"action": "shutdown"})
            result = subprocess.run(
                [sys.executable, str(child_script), str(plugin_dir)],
                input=f"{shutdown}\n",
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 7. ENVIRONMENT SETUP LANE
# ===========================================================================

class TestEnvironmentSetupLane:

    def test_artifact_skipped_for_in_process(self):
        from acquisition.job import EnvironmentSetupArtifact
        art = EnvironmentSetupArtifact(
            plugin_name="stdlib_only",
            execution_mode="in_process",
            skipped=True,
            skip_reason="in_process needs no venv",
        )
        d = art.to_dict()
        assert d["skipped"] is True
        assert d["execution_mode"] == "in_process"

        restored = EnvironmentSetupArtifact.from_dict(d)
        assert restored.skipped is True
        assert restored.plugin_name == "stdlib_only"

    def test_artifact_round_trip(self):
        from acquisition.job import EnvironmentSetupArtifact
        art = EnvironmentSetupArtifact(
            plugin_name="fancy",
            execution_mode="isolated_subprocess",
            venv_path="/tmp/test_venv",
            pinned_dependencies=["requests==2.31.0"],
            installed_packages=["requests==2.31.0"],
            import_verification_passed=True,
        )
        d = art.to_dict()
        restored = EnvironmentSetupArtifact.from_dict(d)
        assert restored.execution_mode == "isolated_subprocess"
        assert restored.pinned_dependencies == ["requests==2.31.0"]
        assert restored.import_verification_passed is True

    def test_environment_setup_in_plugin_creation_lanes(self):
        from acquisition.classifier import _LANE_MAP
        lanes = _LANE_MAP.get("plugin_creation", [])
        assert "environment_setup" in lanes
        impl_idx = lanes.index("implementation")
        env_idx = lanes.index("environment_setup")
        quarantine_idx = lanes.index("plugin_quarantine")
        assert impl_idx < env_idx < quarantine_idx


# ===========================================================================
# 8. OBSERVABILITY
# ===========================================================================

class TestObservability:

    def test_status_includes_execution_mode(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest()
            ok, _ = reg.quarantine("test_plugin", code, manifest)
            assert ok

            status = reg.get_status()
            assert "subprocess_count" in status
            assert "subprocess_running" in status
            assert status["subprocess_count"] == 0

            for p in status["plugins"]:
                assert "execution_mode" in p
                assert "venv_ready" in p
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_isolated_plugin_shows_in_subprocess_count(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_isolated()
            manifest = _make_manifest(
                name="iso_plugin",
                execution_mode="isolated_subprocess",
                pinned_dependencies=["requests==2.31.0"],
            )
            ok, _ = reg.quarantine("iso_plugin", code, manifest)
            assert ok

            status = reg.get_status()
            assert status["subprocess_count"] == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# 9. SINGLETON ACCESSOR
# ===========================================================================

class TestSingleton:

    def test_get_plugin_registry_returns_same_instance(self):
        with patch("tools.plugin_registry._registry_singleton", None):
            r1 = get_plugin_registry()
            r2 = get_plugin_registry()
            assert r1 is r2


# ===========================================================================
# 10. AUDIT TRAIL INCLUDES EXECUTION MODE
# ===========================================================================

class TestAuditTrail:

    def test_audit_entry_contains_execution_mode(self):
        tmpdir = _make_tmpdir()
        try:
            reg = _make_registry(tmpdir)
            code = _stub_code_in_process()
            manifest = _make_manifest()
            ok, _ = reg.quarantine("test_plugin", code, manifest)
            assert ok

            rec = reg.get_record("test_plugin")
            rec.state = "active"
            reg._try_load_handler("test_plugin")

            req = PluginRequest(
                request_id="audit1",
                plugin_name="test_plugin",
                user_text="hello",
                context={},
            )
            resp = _run(reg.invoke(req))
            assert resp.audit_entry is not None
            assert resp.audit_entry.get("execution_mode") == "in_process"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
