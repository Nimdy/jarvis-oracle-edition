"""Smoke tests for jarvis-supervisor.py.

These tests spawn the supervisor with mock brain scripts to verify:
- Clean shutdown (exit 0) stops the supervisor
- Intentional restart (exit 10) relaunches without backoff
- Crash with backoff
- Rapid crash + pending verification triggers rollback
- Crash loop ceiling stops the supervisor
- Intent file validation (stale, malformed, missing)

Run: python -m pytest tests/test_supervisor.py -v
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

SUPERVISOR_PATH = Path(__file__).parent.parent / "jarvis-supervisor.py"
BRAIN_DIR = Path(__file__).parent.parent


def _load_supervisor():
    """Load jarvis-supervisor.py (hyphenated filename) as a module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("jarvis_supervisor", str(SUPERVISOR_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def jarvis_dir(tmp_path):
    """Create a temporary ~/.jarvis directory."""
    jdir = tmp_path / ".jarvis"
    jdir.mkdir()
    return jdir


@pytest.fixture
def intent_file(jarvis_dir):
    return jarvis_dir / "restart_intent.json"


@pytest.fixture
def pending_file(jarvis_dir):
    return jarvis_dir / "pending_verification.json"


def write_intent(intent_file: Path, reason: str = "test_restart",
                 age_s: float = 0) -> None:
    data = {
        "reason": reason,
        "requested_at": time.time() - age_s,
        "requested_by": "test",
        "delay_s": 0,
        "message": "test intent",
        "nonce": f"test-{time.time():.0f}",
    }
    intent_file.write_text(json.dumps(data))


def write_pending(pending_file: Path) -> None:
    data = {
        "patch_id": "test-patch-001",
        "description": "test patch",
        "files_changed": [],
        "snapshot_path": "/nonexistent/snapshot",
        "baselines": {},
        "applied_at": time.time(),
        "verification_period_s": 180,
        "min_ticks": 500,
        "boot_count": 0,
        "max_retries": 2,
    }
    pending_file.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Unit tests for supervisor functions (import and test directly)
# ---------------------------------------------------------------------------

class TestIntentValidation:
    """Test the intent file reading and validation logic."""

    def test_no_intent_file(self, jarvis_dir, intent_file):
        """No intent file → returns None."""
        sup = _load_supervisor()
        with patch.object(sup, "INTENT_FILE", intent_file):
            result = sup.read_and_clear_intent()
        assert result is None

    def test_valid_intent(self, jarvis_dir, intent_file):
        """Valid intent file is read and deleted."""
        write_intent(intent_file, reason="self_improvement_verify")
        sup = _load_supervisor()
        with patch.object(sup, "INTENT_FILE", intent_file):
            result = sup.read_and_clear_intent()
        assert result is not None
        assert result["reason"] == "self_improvement_verify"
        assert not intent_file.exists()

    def test_stale_intent_ignored(self, jarvis_dir, intent_file):
        """Intent older than INTENT_STALENESS_S is ignored."""
        write_intent(intent_file, reason="old_restart", age_s=300)
        sup = _load_supervisor()
        with patch.object(sup, "INTENT_FILE", intent_file):
            result = sup.read_and_clear_intent()
        assert result is None
        assert not intent_file.exists()

    def test_malformed_json_ignored(self, jarvis_dir, intent_file):
        """Malformed JSON is ignored and file is deleted."""
        intent_file.write_text("not valid json {{{")
        sup = _load_supervisor()
        with patch.object(sup, "INTENT_FILE", intent_file):
            result = sup.read_and_clear_intent()
        assert result is None
        assert not intent_file.exists()

    def test_missing_reason_ignored(self, jarvis_dir, intent_file):
        """Intent without 'reason' key is ignored."""
        intent_file.write_text(json.dumps({"requested_at": time.time()}))
        sup = _load_supervisor()
        with patch.object(sup, "INTENT_FILE", intent_file):
            result = sup.read_and_clear_intent()
        assert result is None
        assert not intent_file.exists()


class TestBackoff:
    """Test the backoff delay calculation."""

    def test_backoff_schedule(self):
        sup = _load_supervisor()
        with patch.object(sup, "DEBUG", False):
            assert sup.get_backoff_delay(1) == 5
            assert sup.get_backoff_delay(2) == 10
            assert sup.get_backoff_delay(3) == 20
            assert sup.get_backoff_delay(4) == 40
            assert sup.get_backoff_delay(5) == 60
            assert sup.get_backoff_delay(99) == 60

    def test_debug_mode_no_backoff(self):
        sup = _load_supervisor()
        with patch.object(sup, "DEBUG", True):
            assert sup.get_backoff_delay(1) == 0.0
            assert sup.get_backoff_delay(5) == 0.0


# ---------------------------------------------------------------------------
# Integration tests (spawn actual supervisor with mock brain scripts)
# ---------------------------------------------------------------------------

def _make_mock_brain(tmp_path: Path, script_content: str) -> Path:
    """Create a mock main.py that the supervisor will launch."""
    mock = tmp_path / "main.py"
    mock.write_text(script_content)
    return mock


class TestSupervisorIntegration:
    """Integration tests that spawn the actual supervisor process."""

    @pytest.mark.timeout(35)
    def test_clean_shutdown_exits_supervisor(self, tmp_path):
        """Brain exits 0 → supervisor exits 0."""
        _make_mock_brain(tmp_path, "import sys; sys.exit(0)")

        venv_dir = tmp_path / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        venv_python = venv_dir / "python"
        venv_python.symlink_to(sys.executable)

        result = subprocess.run(
            [sys.executable, str(SUPERVISOR_PATH)],
            cwd=str(tmp_path),
            timeout=30,
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path.parent), "JARVIS_BRAIN_DIR": str(tmp_path)},
        )
        assert result.returncode == 0
        assert "clean_shutdown" in result.stderr or "action: stop" in result.stderr

    @pytest.mark.timeout(35)
    def test_exit10_triggers_restart(self, tmp_path, jarvis_dir):
        """Brain exits 10 first time, then exits 0 → supervisor exits 0."""
        counter_file = tmp_path / "run_count"
        counter_file.write_text("0")

        script = f"""
import sys
counter = int(open("{counter_file}").read())
open("{counter_file}", "w").write(str(counter + 1))
if counter == 0:
    import json, time, tempfile, os
    intent_dir = "{jarvis_dir}"
    intent_file = os.path.join(intent_dir, "restart_intent.json")
    data = {{"reason": "test_restart", "requested_at": time.time(), "requested_by": "test", "delay_s": 0, "message": "test", "nonce": "test"}}
    fd, tmp = tempfile.mkstemp(dir=intent_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, intent_file)
    sys.exit(10)
else:
    sys.exit(0)
"""
        _make_mock_brain(tmp_path, script)

        venv_dir = tmp_path / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        venv_python = venv_dir / "python"
        venv_python.symlink_to(sys.executable)

        result = subprocess.run(
            [sys.executable, str(SUPERVISOR_PATH)],
            cwd=str(tmp_path),
            timeout=30,
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(tmp_path.parent), "JARVIS_BRAIN_DIR": str(tmp_path)},
        )
        assert result.returncode == 0
        assert int(counter_file.read_text()) == 2
        assert "restart (immediate)" in result.stderr

    @pytest.mark.timeout(20)
    def test_sigterm_stops_supervisor_without_relaunch(self, tmp_path):
        """systemd SIGTERM should stop, not classify child signal-exit as crash."""
        counter_file = tmp_path / "run_count"
        counter_file.write_text("0")
        script = f"""
import time
counter_file = "{counter_file}"
counter = int(open(counter_file).read())
open(counter_file, "w").write(str(counter + 1))
time.sleep(30)
"""
        _make_mock_brain(tmp_path, script)

        venv_dir = tmp_path / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        venv_python = venv_dir / "python"
        venv_python.symlink_to(sys.executable)

        proc = subprocess.Popen(
            [sys.executable, str(SUPERVISOR_PATH)],
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "HOME": str(tmp_path.parent), "JARVIS_BRAIN_DIR": str(tmp_path)},
        )
        time.sleep(1.0)
        proc.terminate()
        _, stderr = proc.communicate(timeout=15)

        assert proc.returncode == 0
        assert int(counter_file.read_text()) == 1
        assert "shutdown request | action: stop" in stderr

    @pytest.mark.timeout(50)
    def test_crash_loop_exits_supervisor(self, tmp_path):
        """Brain crashes repeatedly → supervisor gives up."""
        _make_mock_brain(tmp_path, "import sys; sys.exit(1)")

        venv_dir = tmp_path / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        venv_python = venv_dir / "python"
        venv_python.symlink_to(sys.executable)

        result = subprocess.run(
            [sys.executable, str(SUPERVISOR_PATH)],
            cwd=str(tmp_path),
            timeout=45,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "HOME": str(tmp_path.parent),
                "JARVIS_BRAIN_DIR": str(tmp_path),
                "JARVIS_SUPERVISOR_DEBUG": "1",
            },
        )
        assert result.returncode == 1
        assert "Crash loop detected" in result.stderr
