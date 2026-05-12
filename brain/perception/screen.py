"""Screen processor — active window awareness on the laptop."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import Callable

from consciousness.events import event_bus, PERCEPTION_SCREEN_CONTEXT
from consciousness.engine import ConsciousnessEngine
from memory.core import CreateMemoryData

logger = logging.getLogger(__name__)

_APP_CATEGORIES: dict[str, str] = {
    "firefox": "browsing", "chrome": "browsing", "chromium": "browsing", "brave": "browsing",
    "code": "coding", "cursor": "coding", "vim": "coding", "nvim": "coding", "emacs": "coding",
    "terminal": "terminal", "kitty": "terminal", "alacritty": "terminal", "gnome-terminal": "terminal", "konsole": "terminal",
    "slack": "communication", "discord": "communication", "teams": "communication", "zoom": "meeting",
    "thunderbird": "email", "evolution": "email",
    "spotify": "media", "vlc": "media",
    "nautilus": "files", "dolphin": "files", "thunar": "files",
    "gimp": "design", "inkscape": "design", "blender": "design",
    "libreoffice": "documents", "evince": "documents",
}


class ScreenProcessor:
    def __init__(self, poll_s: float = 5.0) -> None:
        self._poll_s = poll_s
        self._task: asyncio.Task | None = None
        self._last_context: dict | None = None
        self._context_history: list[dict] = []
        self._max_history = 50
        self._engine: ConsciousnessEngine | None = None

    def set_engine(self, engine: ConsciousnessEngine) -> None:
        self._engine = engine

    def start(self) -> None:
        if self._task:
            return
        self._task = asyncio.get_event_loop().create_task(self._poll_loop())
        logger.info("Screen awareness polling every %.1fs", self._poll_s)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def _poll_loop(self) -> None:
        try:
            while True:
                await self._poll()
                await asyncio.sleep(self._poll_s)
        except asyncio.CancelledError:
            pass

    async def _poll(self) -> None:
        try:
            active = await self._get_active_window()
            idle_ms = await self._get_idle_time()
            category = self._categorize_app(active["app"].lower())

            context = {
                "app": active["app"],
                "title": self._sanitize_title(active["title"]),
                "idle": idle_ms > 300_000,
                "idle_time_ms": idle_ms,
                "timestamp": time.time(),
                "category": category,
            }

            if self._has_changed(context):
                self._last_context = context
                self._context_history.append(context)
                if len(self._context_history) > self._max_history:
                    self._context_history = self._context_history[-self._max_history:]

                event_bus.emit(PERCEPTION_SCREEN_CONTEXT,
                               app=context["app"], title=context["title"], idle=context["idle"])

                if self._engine and category != "unknown":
                    self._engine.remember(CreateMemoryData(
                        type="observation",
                        payload=f"User switched to {context['app']} ({category})",
                        weight=0.15,
                        tags=["screen", "app_switch", category],
                        provenance="observed",
                    ))
        except Exception:
            pass

    def get_context_summary(self) -> str:
        if not self._last_context:
            return "Screen context unavailable"
        parts = [f"Using {self._last_context['app']} ({self._last_context['category']})"]
        if self._last_context["idle"]:
            mins = int(self._last_context["idle_time_ms"] / 60_000)
            parts.append(f"idle for {mins}min")
        return ". ".join(parts)

    def get_last_context(self) -> dict | None:
        return self._last_context

    async def _get_active_window(self) -> dict[str, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "xdotool", "getactivewindow", "getwindowname",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
            title = stdout.decode().strip()
            try:
                proc2 = await asyncio.create_subprocess_exec(
                    "xdotool", "getactivewindow", "getwindowclassname",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=2)
                app = stdout2.decode().strip()
            except Exception:
                app = title.split(" - ")[-1].strip() if " - " in title else "Unknown"
            return {"app": app, "title": title}
        except Exception:
            return await self._get_active_window_wayland()

    async def _get_active_window_wayland(self) -> dict[str, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c",
                'gdbus call --session --dest org.gnome.Shell --object-path /org/gnome/Shell '
                '--method org.gnome.Shell.Eval "global.display.focus_window?.get_title()"',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
            import re
            match = re.search(r"'([^']*)'", stdout.decode())
            return {"app": "Unknown", "title": match.group(1) if match else "Unknown"}
        except Exception:
            return {"app": "Unknown", "title": "Unknown"}

    async def _get_idle_time(self) -> int:
        try:
            proc = await asyncio.create_subprocess_exec(
                "xprintidle",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
            return int(stdout.decode().strip()) if stdout else 0
        except Exception:
            return 0

    @staticmethod
    def _categorize_app(app_name: str) -> str:
        for key, category in _APP_CATEGORIES.items():
            if key in app_name:
                return category
        return "unknown"

    @staticmethod
    def _sanitize_title(title: str) -> str:
        import re
        title = re.sub(r"https?://[^\s]+", "[url]", title)
        title = re.sub(r"/home/[^\s/]+", "~", title)
        title = re.sub(r"(?i)(password|secret|token|key|credential)", "[redacted]", title)
        return title[:100]

    def _has_changed(self, new_ctx: dict) -> bool:
        if not self._last_context:
            return True
        return (
            self._last_context["app"] != new_ctx["app"]
            or self._last_context["title"] != new_ctx["title"]
            or self._last_context["idle"] != new_ctx["idle"]
        )
