"""Tests for codegen/coder_server.py — lifecycle, command building, status.

Tests CoderServer without spawning llama-server processes.
All subprocess and network I/O is mocked.

SAFETY: Every test removes its CoderServer from the global _active_servers
list and nullifies _process to prevent the atexit handler from trying to
kill mock process groups on interpreter shutdown.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from codegen.coder_server import (
    CoderServer,
    _HEALTH_POLL_INTERVAL_S,
    _HEALTH_TIMEOUT_S,
    _SHUTDOWN_GRACE_S,
    MAX_PARSE_RETRIES,
    JSON_REPAIR_PROMPT,
    _active_servers,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _safe_server(**kwargs) -> CoderServer:
    """Create a CoderServer and ensure it is cleaned up on test exit."""
    srv = CoderServer(**kwargs)
    return srv


def _teardown(srv: CoderServer) -> None:
    """Remove server from global list and clear process to prevent atexit kills."""
    srv._process = None
    if srv in _active_servers:
        _active_servers.remove(srv)


# ---------------------------------------------------------------------------
# Construction & Availability
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        srv = _safe_server()
        try:
            assert srv._port == 8081
            assert srv._gpu_layers == 0
            assert srv._ctx_size == 16384
            assert srv._process is None
            assert srv._total_generations == 0
        finally:
            _teardown(srv)

    def test_custom_params(self):
        srv = _safe_server(
            model_path="/tmp/test.gguf",
            server_port=9999,
            ctx_size=4096,
            gpu_layers=10,
            max_tokens=8192,
            temperature=0.5,
        )
        try:
            assert srv._port == 9999
            assert srv._gpu_layers == 10
            assert srv._ctx_size == 4096
            assert srv._max_tokens == 8192
            assert srv._temperature == 0.5
        finally:
            _teardown(srv)

    def test_registered_in_active_servers(self):
        initial = len(_active_servers)
        srv = _safe_server()
        try:
            assert len(_active_servers) == initial + 1
        finally:
            _teardown(srv)

    def test_is_available_no_model(self):
        srv = _safe_server(model_path="")
        try:
            assert srv.is_available() is False
        finally:
            _teardown(srv)

    def test_is_available_model_missing(self):
        srv = _safe_server(model_path="/nonexistent/model.gguf")
        try:
            assert srv.is_available() is False
        finally:
            _teardown(srv)

    def test_is_available_with_model_and_binary(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            with patch("shutil.which", return_value="/usr/bin/llama-server"):
                srv = _safe_server(model_path=f.name)
                try:
                    assert srv.is_available() is True
                finally:
                    _teardown(srv)


# ---------------------------------------------------------------------------
# Command Building
# ---------------------------------------------------------------------------

class TestBuildCommand:
    def test_cpu_only_no_gpu_flags(self):
        srv = _safe_server(model_path="/tmp/model.gguf", gpu_layers=0)
        try:
            cmd = srv._build_command()
            assert "-ngl" not in cmd
            assert "-m" in cmd
            assert "--ctx-size" in cmd
            assert "--port" in cmd
        finally:
            _teardown(srv)

    def test_gpu_layers_present(self):
        srv = _safe_server(model_path="/tmp/model.gguf", gpu_layers=33)
        try:
            cmd = srv._build_command()
            assert "-ngl" in cmd
            ngl_idx = cmd.index("-ngl")
            assert cmd[ngl_idx + 1] == "33"
        finally:
            _teardown(srv)

    def test_port_in_command(self):
        srv = _safe_server(model_path="/tmp/model.gguf", server_port=7777)
        try:
            cmd = srv._build_command()
            port_idx = cmd.index("--port")
            assert cmd[port_idx + 1] == "7777"
        finally:
            _teardown(srv)


# ---------------------------------------------------------------------------
# is_running
# ---------------------------------------------------------------------------

class TestIsRunning:
    def test_no_process(self):
        srv = _safe_server()
        try:
            assert srv.is_running() is False
        finally:
            _teardown(srv)

    def test_process_alive(self):
        srv = _safe_server()
        try:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            srv._process = mock_proc
            assert srv.is_running() is True
        finally:
            _teardown(srv)

    def test_process_exited(self):
        srv = _safe_server()
        try:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            srv._process = mock_proc
            assert srv.is_running() is False
        finally:
            _teardown(srv)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown:
    def test_shutdown_no_process(self):
        srv = _safe_server()
        try:
            _run(srv.shutdown())
            assert srv._process is None
        finally:
            _teardown(srv)

    def test_shutdown_sends_sigterm(self):
        srv = _safe_server()
        try:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = 0
            srv._process = mock_proc

            with patch("os.killpg") as mock_kill, \
                 patch("os.getpgid", return_value=12345):
                _run(srv.shutdown())

            mock_kill.assert_called_once()
            assert srv._process is None
        finally:
            _teardown(srv)

    def test_shutdown_handles_process_not_found(self):
        srv = _safe_server()
        try:
            mock_proc = MagicMock()
            mock_proc.pid = 99999
            srv._process = mock_proc

            with patch("os.killpg", side_effect=ProcessLookupError), \
                 patch("os.getpgid", return_value=99999):
                _run(srv.shutdown())

            assert srv._process is None
        finally:
            _teardown(srv)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_shape_idle(self):
        srv = _safe_server(model_path="/nonexistent.gguf")
        try:
            status = srv.get_status()
            assert status["available"] is False
            assert status["running"] is False
            assert status["pid"] is None
            assert status["total_generations"] == 0
            assert "port" in status
            assert "gpu_layers" in status
            assert "ctx_size" in status
        finally:
            _teardown(srv)

    def test_status_with_running_process(self):
        srv = _safe_server(model_path="/nonexistent.gguf")
        try:
            mock_proc = MagicMock()
            mock_proc.pid = 42
            mock_proc.poll.return_value = None
            srv._process = mock_proc

            status = srv.get_status()
            assert status["running"] is True
            assert status["pid"] == 42
        finally:
            _teardown(srv)


# ---------------------------------------------------------------------------
# Constants and JSON repair prompt
# ---------------------------------------------------------------------------

class TestConstants:
    def test_health_poll_reasonable(self):
        assert 0.5 <= _HEALTH_POLL_INTERVAL_S <= 10.0

    def test_health_timeout_reasonable(self):
        assert 30 <= _HEALTH_TIMEOUT_S <= 600

    def test_shutdown_grace_reasonable(self):
        assert 5 <= _SHUTDOWN_GRACE_S <= 60

    def test_json_repair_prompt_has_schema(self):
        assert "files" in JSON_REPAIR_PROMPT
        assert "edits" in JSON_REPAIR_PROMPT
        assert "search" in JSON_REPAIR_PROMPT
        assert "replace" in JSON_REPAIR_PROMPT

    def test_max_parse_retries(self):
        assert MAX_PARSE_RETRIES >= 1
