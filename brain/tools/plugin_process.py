"""Plugin Process Manager — per-plugin subprocess lifecycle for isolated execution.

Manages child processes that run plugins in their own venvs, communicating via
JSON-over-stdin/stdout.  Modeled on the CoderServer pattern (codegen/coder_server.py).

Process isolation provides:
  - Separate Python interpreter per plugin (own venv with pinned deps)
  - Stripped environment variables (no JARVIS_*, OLLAMA_*, API keys)
  - Plugin directory as cwd (no brain path access)
  - Idle shutdown after configurable timeout

Process isolation does NOT provide:
  - Filesystem sandboxing (child can read/write the filesystem)
  - Network sandboxing (child can make network calls)
  - Resource limits (no cgroup/rlimit enforcement)

The correct claim is "process-isolated", not "sandboxed".
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_JARVIS_DIR = Path.home() / ".jarvis"
_VENV_BASE_DIR = _JARVIS_DIR / "plugin_venvs"

IDLE_TIMEOUT_S = 300.0  # 5 minutes
REQUEST_TIMEOUT_S = 60.0
VENV_CREATE_TIMEOUT_S = 120.0
PIP_INSTALL_TIMEOUT_S = 300.0

_STRIPPED_ENV_PREFIXES = (
    "JARVIS_",
    "OLLAMA_",
    "ANTHROPIC_",
    "OPENAI_",
    "S2_",
    "CROSSREF_",
    "DASHBOARD_",
)

_STRIPPED_ENV_SUBSTRINGS = (
    "API_KEY",
    "SECRET",
    "PASSWORD",
    "TOKEN",
    "CREDENTIAL",
)

_active_managers: list[PluginProcessManager] = []


def _cleanup_on_exit() -> None:
    """Kill any child processes left running on interpreter shutdown."""
    for mgr in _active_managers:
        mgr._force_kill()


atexit.register(_cleanup_on_exit)


def _build_clean_env() -> dict[str, str]:
    """Build environment dict with sensitive variables stripped."""
    clean: dict[str, str] = {}
    for key, val in os.environ.items():
        if any(key.startswith(p) for p in _STRIPPED_ENV_PREFIXES):
            continue
        if any(s in key.upper() for s in _STRIPPED_ENV_SUBSTRINGS):
            continue
        clean[key] = val
    return clean


class PluginProcessManager:
    """Manages one child process per isolated plugin.

    Lifecycle:
      1. ensure_venv() — create venv + install pinned deps (once)
      2. invoke(request_dict) — lazy-start child, send JSON, read JSON response
      3. idle timeout — child auto-killed after IDLE_TIMEOUT_S with no requests
      4. shutdown() — explicit clean shutdown
    """

    def __init__(
        self,
        plugin_name: str,
        plugin_dir: Path,
        pinned_dependencies: list[str] | None = None,
        verify_imports: list[str] | None = None,
    ) -> None:
        self._plugin_name = plugin_name
        self._plugin_dir = plugin_dir
        self._pinned_deps = pinned_dependencies or []
        self._verify_imports_override = [s for s in (verify_imports or []) if s]

        self._venv_dir = _VENV_BASE_DIR / plugin_name
        self._venv_python: Path | None = None
        self._venv_ready = False
        self._install_log: str = ""

        self._process: asyncio.subprocess.Process | None = None
        self._last_request_at: float = 0.0
        self._idle_check_task: asyncio.Task | None = None
        self._invocation_count: int = 0
        self._shutting_down = False

        _active_managers.append(self)

    @property
    def venv_ready(self) -> bool:
        return self._venv_ready

    @property
    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
        )

    @property
    def install_log(self) -> str:
        return self._install_log

    # ── Venv Lifecycle ────────────────────────────────────────────────

    async def ensure_venv(self) -> tuple[bool, str]:
        """Create venv and install pinned dependencies. Returns (ok, log)."""
        _VENV_BASE_DIR.mkdir(parents=True, exist_ok=True)

        venv_python = self._venv_dir / "bin" / "python"
        if self._venv_ready and venv_python.exists():
            return True, "venv already ready"

        log_parts: list[str] = []

        if not self._venv_dir.exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "venv", str(self._venv_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=VENV_CREATE_TIMEOUT_S
                )
                if proc.returncode != 0:
                    err = stderr.decode(errors="replace")[:500]
                    log_parts.append(f"venv creation failed (rc={proc.returncode}): {err}")
                    self._install_log = "\n".join(log_parts)
                    return False, self._install_log
                log_parts.append(f"venv created at {self._venv_dir}")
            except asyncio.TimeoutError:
                log_parts.append("venv creation timed out")
                self._install_log = "\n".join(log_parts)
                return False, self._install_log
            except Exception as exc:
                log_parts.append(f"venv creation error: {exc}")
                self._install_log = "\n".join(log_parts)
                return False, self._install_log

        if not venv_python.exists():
            log_parts.append(f"venv python not found at {venv_python}")
            self._install_log = "\n".join(log_parts)
            return False, self._install_log

        self._venv_python = venv_python

        if self._pinned_deps:
            pip_cmd = [str(venv_python), "-m", "pip", "install", "--no-input"]
            pip_cmd.extend(self._pinned_deps)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *pip_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=_build_clean_env(),
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=PIP_INSTALL_TIMEOUT_S
                )
                out_text = stdout.decode(errors="replace")
                err_text = stderr.decode(errors="replace")
                log_parts.append(f"pip install rc={proc.returncode}")
                if out_text.strip():
                    log_parts.append(f"pip stdout: {out_text[:1000]}")
                if err_text.strip():
                    log_parts.append(f"pip stderr: {err_text[:500]}")
                if proc.returncode != 0:
                    self._install_log = "\n".join(log_parts)
                    return False, self._install_log
            except asyncio.TimeoutError:
                log_parts.append("pip install timed out")
                self._install_log = "\n".join(log_parts)
                return False, self._install_log
            except Exception as exc:
                log_parts.append(f"pip install error: {exc}")
                self._install_log = "\n".join(log_parts)
                return False, self._install_log

            ok = await self._verify_imports()
            if not ok:
                log_parts.append("import verification failed")
                self._install_log = "\n".join(log_parts)
                return False, self._install_log
            log_parts.append("import verification passed")

        self._venv_ready = True
        log_parts.append("venv ready")
        self._install_log = "\n".join(log_parts)
        logger.info("Plugin venv ready: %s (%d deps)", self._plugin_name, len(self._pinned_deps))
        return True, self._install_log

    async def _verify_imports(self) -> bool:
        """Verify pinned packages are importable in the venv.

        When the manifest declares ``verify_imports``, those module names
        are used verbatim. Otherwise falls back to a distribution-name
        heuristic (``pkg-name`` -> ``pkg_name``), which is correct for
        most packages but fails for dist-name != import-name cases
        (e.g. ``python-dateutil`` -> ``dateutil``, ``PyYAML`` -> ``yaml``).
        """
        if not self._venv_python:
            return False
        if self._verify_imports_override:
            pkg_names = list(self._verify_imports_override)
        else:
            pkg_names = []
            for dep in self._pinned_deps:
                name = re.split(r"[=<>!~]", dep)[0].strip().replace("-", "_")
                if name:
                    pkg_names.append(name)
        if not pkg_names:
            return True
        import_stmts = "; ".join(f"import {n}" for n in pkg_names)
        try:
            proc = await asyncio.create_subprocess_exec(
                str(self._venv_python), "-c", import_stmts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_build_clean_env(),
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                logger.warning(
                    "Import verification failed for %s: %s",
                    self._plugin_name, stderr.decode(errors="replace")[:300],
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Import verification error for %s: %s", self._plugin_name, exc)
            return False

    # ── Child Process Lifecycle ───────────────────────────────────────

    async def _ensure_child(self) -> bool:
        """Start the child process if not running. Returns True when ready."""
        if self.is_running:
            return True

        if not self._venv_ready:
            ok, _ = await self.ensure_venv()
            if not ok:
                return False

        python = self._venv_python or (self._venv_dir / "bin" / "python")
        if not python.exists():
            logger.error("Venv python missing for %s: %s", self._plugin_name, python)
            return False

        child_script = Path(__file__).parent / "plugin_runner_child.py"
        if not child_script.exists():
            logger.error("Child wrapper not found: %s", child_script)
            return False

        clean_env = _build_clean_env()
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            self._process = await asyncio.create_subprocess_exec(
                str(python), str(child_script), str(self._plugin_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._plugin_dir),
                env=clean_env,
            )
            logger.info(
                "Plugin child started: %s (pid=%d)",
                self._plugin_name, self._process.pid,
            )
            self._start_idle_monitor()
            return True
        except Exception as exc:
            logger.exception("Failed to start plugin child: %s", self._plugin_name)
            self._process = None
            return False

    async def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the child and return the response dict."""
        if not await self._ensure_child():
            return {
                "request_id": request.get("request_id", ""),
                "success": False,
                "result": None,
                "error": f"Failed to start subprocess for plugin '{self._plugin_name}'",
            }

        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        self._last_request_at = time.monotonic()
        self._invocation_count += 1
        req_line = json.dumps(request) + "\n"

        try:
            self._process.stdin.write(req_line.encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self._process = None
            return {
                "request_id": request.get("request_id", ""),
                "success": False,
                "result": None,
                "error": f"Child process pipe broken: {exc}",
            }

        timeout = request.get("timeout_s", REQUEST_TIMEOUT_S)
        try:
            raw_line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Plugin child timeout: %s (%.0fs)", self._plugin_name, timeout)
            await self._kill_child()
            return {
                "request_id": request.get("request_id", ""),
                "success": False,
                "result": None,
                "error": f"Plugin subprocess timed out after {timeout}s",
            }

        if not raw_line:
            await self._kill_child()
            return {
                "request_id": request.get("request_id", ""),
                "success": False,
                "result": None,
                "error": "Child process exited unexpectedly (empty response)",
            }

        try:
            resp = json.loads(raw_line.decode(errors="replace"))
        except json.JSONDecodeError as exc:
            logger.warning(
                "Malformed JSON from plugin child %s: %s",
                self._plugin_name, raw_line[:200],
            )
            return {
                "request_id": request.get("request_id", ""),
                "success": False,
                "result": None,
                "error": f"Malformed JSON response from child: {exc}",
            }

        return resp

    # ── Shutdown / Cleanup ────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully shut down the child process."""
        self._shutting_down = True
        if self._idle_check_task and not self._idle_check_task.done():
            self._idle_check_task.cancel()

        if not self.is_running:
            self._process = None
            return

        assert self._process is not None
        assert self._process.stdin is not None

        try:
            shutdown_msg = json.dumps({"action": "shutdown"}) + "\n"
            self._process.stdin.write(shutdown_msg.encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
            logger.info("Plugin child stopped gracefully: %s", self._plugin_name)
        except asyncio.TimeoutError:
            logger.warning("Plugin child did not stop gracefully: %s, sending SIGKILL", self._plugin_name)
            await self._kill_child()

        self._process = None
        self._shutting_down = False

    async def _kill_child(self) -> None:
        """Force-kill the child process."""
        if self._process is None:
            return
        try:
            self._process.kill()
            await self._process.wait()
        except (ProcessLookupError, OSError):
            pass
        self._process = None

    def _force_kill(self) -> None:
        """Synchronous force-kill for atexit handler."""
        if self._process is None or self._process.returncode is not None:
            return
        try:
            self._process.kill()
        except (ProcessLookupError, OSError):
            pass

    # ── Idle Monitor ──────────────────────────────────────────────────

    def _start_idle_monitor(self) -> None:
        """Start background task that kills the child after idle timeout."""
        if self._idle_check_task and not self._idle_check_task.done():
            self._idle_check_task.cancel()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._idle_check_task = loop.create_task(self._idle_monitor_loop())
        except RuntimeError:
            pass

    async def _idle_monitor_loop(self) -> None:
        """Periodically check for idle timeout and shut down the child."""
        while not self._shutting_down:
            await asyncio.sleep(30)
            if self._shutting_down:
                break
            if not self.is_running:
                break
            idle = time.monotonic() - self._last_request_at
            if idle >= IDLE_TIMEOUT_S:
                logger.info(
                    "Plugin child idle shutdown: %s (idle %.0fs)",
                    self._plugin_name, idle,
                )
                await self.shutdown()
                break

    # ── Venv Cleanup ──────────────────────────────────────────────────

    def remove_venv(self) -> bool:
        """Remove the plugin's venv directory. Returns True on success."""
        if self._venv_dir.exists():
            try:
                shutil.rmtree(self._venv_dir)
                self._venv_ready = False
                logger.info("Removed venv for plugin: %s", self._plugin_name)
                return True
            except Exception as exc:
                logger.warning("Failed to remove venv for %s: %s", self._plugin_name, exc)
                return False
        return True

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Status dict for dashboard/API exposure."""
        return {
            "plugin_name": self._plugin_name,
            "execution_mode": "isolated_subprocess",
            "venv_ready": self._venv_ready,
            "venv_path": str(self._venv_dir) if self._venv_dir.exists() else None,
            "child_running": self.is_running,
            "child_pid": self._process.pid if self.is_running else None,
            "invocation_count": self._invocation_count,
            "pinned_dependencies": self._pinned_deps,
            "idle_timeout_s": IDLE_TIMEOUT_S,
        }
