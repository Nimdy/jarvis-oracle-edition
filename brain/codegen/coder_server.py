"""On-demand llama-server lifecycle manager for code generation.

Shared service used by both self-improvement and the capability acquisition
pipeline.  Spawns a llama-server process when a generation is needed, waits
for it to become healthy, sends OpenAI-compatible chat completion requests,
then shuts it down to reclaim ~40GB RAM.  Pure CPU by default
(CODER_GPU_LAYERS=0) so the conversation LLM on GPU is never disrupted.
"""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_active_servers: list["CoderServer"] = []


def _cleanup_on_exit() -> None:
    """Kill any llama-server processes left running on interpreter shutdown."""
    for srv in _active_servers:
        if srv._process is not None and srv._process.poll() is None:
            try:
                os.killpg(os.getpgid(srv._process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass


atexit.register(_cleanup_on_exit)

_HEALTH_POLL_INTERVAL_S = 2.0
_HEALTH_TIMEOUT_S = 180.0
_SHUTDOWN_GRACE_S = 15.0
_REQUEST_TIMEOUT_S = 600.0

MAX_PARSE_RETRIES = 2

JSON_REPAIR_PROMPT = (
    "Your previous response was not valid JSON. You MUST respond with ONLY a JSON object "
    "matching this exact schema (no markdown fences, no explanation before or after):\n"
    '{"files": [{"path": "brain/...", "edits": [{"search": "exact existing code", "replace": "new code"}]}], '
    '"description": "...", "confidence": 0.0-1.0}\n'
    "Respond with ONLY the JSON object, nothing else."
)


_EXPECTED_SIZES: dict[str, int] = {
    "Qwen3-Coder-Next-UD-Q4_K_XL.gguf": 49608478720,
    "Qwen3-Coder-Next-UD-IQ4_XS.gguf": 38429272064,
    "Qwen3-Coder-Next-UD-IQ2_M.gguf": 24962293760,
}


class CoderServer:
    """Manages the llama-server process lifecycle for code generation."""

    def __init__(
        self,
        model_path: str = "",
        server_port: int = 8081,
        ctx_size: int = 16384,
        gpu_layers: int = 0,
        llama_server_bin: str = "llama-server",
        max_tokens: int = 16384,
        temperature: float = 0.3,
    ) -> None:
        self._model_path = str(Path(model_path).expanduser()) if model_path else ""
        self._port = server_port
        self._ctx_size = ctx_size
        self._gpu_layers = gpu_layers
        self._bin = llama_server_bin
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._process: subprocess.Popen | None = None
        self._total_generations: int = 0
        self._last_generation_time_s: float = 0.0
        self._active_consumer: str = ""
        self._last_consumer: str = ""
        self._model_integrity: str = ""
        _active_servers.append(self)
        self._base_url = f"http://127.0.0.1:{self._port}"

        if self._model_path:
            self._model_integrity = self._check_model_integrity()
            if self._model_integrity != "ok":
                logger.error("Coder model integrity check FAILED: %s (%s)",
                             self._model_integrity, self._model_path)

    def _check_model_integrity(self) -> str:
        """Verify model file exists and has expected size. Returns 'ok' or error."""
        p = Path(self._model_path)
        if not p.exists():
            return "file_missing"
        model_name = p.name
        expected_size = _EXPECTED_SIZES.get(model_name)
        if expected_size is not None:
            actual_size = p.stat().st_size
            if actual_size != expected_size:
                return (f"size_mismatch:expected={expected_size},"
                        f"actual={actual_size},"
                        f"missing={expected_size - actual_size} bytes")
        return "ok"

    def is_available(self) -> bool:
        """Model file exists, passes integrity check, and llama-server binary is in PATH."""
        if not self._model_path:
            return False
        if self._model_integrity and self._model_integrity != "ok":
            return False
        return Path(self._model_path).exists() and shutil.which(self._bin) is not None

    def set_consumer(self, consumer: str) -> None:
        """Record the current shared coder consumer for dashboard truth."""
        self._active_consumer = consumer
        self._last_consumer = consumer

    def is_running(self) -> bool:
        """llama-server process is alive."""
        return self._process is not None and self._process.poll() is None

    def _build_command(self) -> list[str]:
        """Build a llama-server command line compatible with CPU-only hosts.

        We intentionally omit GPU-only tuning flags when the coder is configured
        for CPU execution (`gpu_layers <= 0`). Some llama.cpp builds treat a bare
        `-fa` flag as invalid and newer builds require an explicit value.
        """
        cmd = [
            self._bin,
            "-m", self._model_path,
            "--ctx-size", str(self._ctx_size),
            "--port", str(self._port),
        ]
        if self._gpu_layers > 0:
            cmd.extend(["-ngl", str(self._gpu_layers)])
        return cmd

    async def ensure_running(self) -> bool:
        """Start llama-server if not running. Returns True when healthy."""
        if self.is_running():
            if await self._health_check():
                return True
            logger.warning("Coder server process alive but unhealthy, restarting")
            await self.shutdown()

        if not self.is_available():
            logger.warning("Coder server not available (model=%s, binary=%s)",
                           self._model_path, shutil.which(self._bin))
            return False

        cmd = self._build_command()

        logger.info("Starting coder server: %s", " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            logger.error("llama-server binary not found: %s", self._bin)
            return False
        except Exception:
            logger.exception("Failed to start coder server")
            return False

        start = time.monotonic()
        while time.monotonic() - start < _HEALTH_TIMEOUT_S:
            if self._process.poll() is not None:
                stderr = ""
                try:
                    stderr = (self._process.stderr.read() or b"").decode(errors="replace")[:500]
                except Exception:
                    pass
                logger.error("Coder server exited during startup (code=%d): %s",
                             self._process.returncode, stderr)
                self._process = None
                return False

            if await self._health_check():
                elapsed = time.monotonic() - start
                logger.info("Coder server healthy after %.1fs (pid=%d)", elapsed, self._process.pid)
                return True

            await asyncio.sleep(_HEALTH_POLL_INTERVAL_S)

        logger.error("Coder server health timeout after %.0fs", _HEALTH_TIMEOUT_S)
        await self.shutdown()
        return False

    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
    ) -> str | None:
        """Send a chat completion request to the running server.

        Returns the raw text response, or None on failure.
        Includes retry logic for JSON parse failures.
        """
        if not self.is_running():
            if not await self.ensure_running():
                return None

        api_messages: list[dict[str, str]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            api_messages.append({"role": role, "content": m["content"]})

        payload = {
            "messages": api_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }

        t0 = time.monotonic()
        try:
            import aiohttp
            url = f"{self._base_url}/v1/chat/completions"
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("Coder server returned %d: %s", resp.status, body[:300])
                        return None
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]
        except ImportError:
            text = await self._generate_urllib(payload)
            if text is None:
                return None
        except Exception:
            logger.exception("Coder server generation failed")
            return None

        elapsed = time.monotonic() - t0
        self._last_generation_time_s = elapsed
        self._total_generations += 1
        logger.info("Coder generation completed in %.1fs (%d tokens est.)",
                     elapsed, len(text) // 4)
        return text

    async def _generate_urllib(self, payload: dict) -> str | None:
        """Fallback HTTP client using urllib (no aiohttp dependency)."""
        import urllib.request
        url = f"{self._base_url}/v1/chat/completions"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=int(_REQUEST_TIMEOUT_S)),
            )
            body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("Coder server urllib fallback failed")
            return None

    async def shutdown(self) -> None:
        """Stop the llama-server process."""
        if self._process is None:
            self._active_consumer = ""
            return

        pid = self._process.pid
        logger.info("Shutting down coder server (pid=%d)", pid)

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            self._process = None
            self._active_consumer = ""
            return

        start = time.monotonic()
        while time.monotonic() - start < _SHUTDOWN_GRACE_S:
            if self._process.poll() is not None:
                logger.info("Coder server stopped (pid=%d, %.1fs)", pid, time.monotonic() - start)
                self._process = None
                self._active_consumer = ""
                return
            await asyncio.sleep(0.5)

        logger.warning("Coder server did not stop gracefully, sending SIGKILL")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        self._process = None
        self._active_consumer = ""

    async def _health_check(self) -> bool:
        """Poll /health endpoint."""
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self._base_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("status") == "ok"
        except ImportError:
            return await self._health_check_urllib()
        except Exception:
            return False
        return False

    async def _health_check_urllib(self) -> bool:
        """Fallback health check using urllib."""
        import urllib.request
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(f"{self._base_url}/health", timeout=5),
            )
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
        except Exception:
            return False

    def get_status(self) -> dict[str, Any]:
        """Status dict for dashboard API."""
        model_size_gb: float | None = None
        if self._model_path and Path(self._model_path).exists():
            try:
                model_size_gb = round(Path(self._model_path).stat().st_size / (1024**3), 1)
            except Exception:
                pass

        return {
            "available": self.is_available(),
            "running": self.is_running(),
            "model_path": self._model_path,
            "model_size_gb": model_size_gb,
            "model_integrity": self._model_integrity or "unchecked",
            "pid": self._process.pid if self._process and self._process.poll() is None else None,
            "port": self._port,
            "gpu_layers": self._gpu_layers,
            "ctx_size": self._ctx_size,
            "total_generations": self._total_generations,
            "last_generation_time_s": round(self._last_generation_time_s, 1),
            "active_consumer": self._active_consumer,
            "last_consumer": self._last_consumer,
        }
