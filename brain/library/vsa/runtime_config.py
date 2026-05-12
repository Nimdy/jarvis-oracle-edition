"""Boot-time HRR runtime flag reader (P4 + P5 + P5.1 persistence).

Precedence (highest wins):

1. **Environment variables** — exact backwards-compatible names from P4/P5:

   * ``ENABLE_HRR_SHADOW`` — master gate for the P4 HRR shadow substrate.
   * ``HRR_SHADOW_DIM`` — vector dimensionality (default 1024).
   * ``HRR_SHADOW_SAMPLE_EVERY_TICKS`` — world-shadow cadence.
   * ``ENABLE_HRR_SPATIAL_SCENE`` — twin gate for the P5 mental-world
     spatial scene graph. Both ``ENABLE_HRR_SHADOW`` **and**
     ``ENABLE_HRR_SPATIAL_SCENE`` must be true for the P5 lane to activate.
   * ``HRR_SPATIAL_SCENE_SAMPLE_EVERY_TICKS`` — P5 spatial-shadow cadence.

2. **Persistent runtime flag file** — JSON, default location
   ``~/.jarvis/runtime_flags.json``. Override the path with
   ``JARVIS_RUNTIME_FLAGS`` (used by tests). Recognized keys:

   * ``enable_hrr_shadow`` (bool)
   * ``enable_hrr_spatial_scene`` (bool)
   * ``hrr_sample_every_ticks`` (int)
   * ``hrr_spatial_scene_sample_every_ticks`` (int)
   * ``hrr_shadow_dim`` (int)

3. **Hardcoded safe defaults** — every flag ``False`` / off. Fresh clones
   never auto-enable HRR shadow runtime.

The persistence layer means an operator can opt their own brain into HRR
shadow once and have it survive restarts, without making HRR globally
default-on for fresh installs. The runtime config snapshot exposes
``flag_sources`` so the dashboard can show *why* HRR is enabled.

Live toggling still requires a process restart (send ``SIGTERM`` to
``main.py`` — the supervisor respawns and re-reads file + env). An
already-running brain never observes a shell ``export`` or a file
rewrite mid-tick, so there is no ambiguity about whether HRR was on or
off during a given tick.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precedence label vocabulary
# ---------------------------------------------------------------------------

SOURCE_DEFAULT: str = "default"
SOURCE_RUNTIME_FLAGS: str = "runtime_flags"
SOURCE_ENVIRONMENT: str = "environment"

_VALID_SOURCES = {SOURCE_DEFAULT, SOURCE_RUNTIME_FLAGS, SOURCE_ENVIRONMENT}


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

DEFAULT_RUNTIME_FLAGS_PATH: Path = Path.home() / ".jarvis" / "runtime_flags.json"
RUNTIME_FLAGS_PATH_ENV: str = "JARVIS_RUNTIME_FLAGS"

_FILE_KEY_BOOL = {
    "enable_hrr_shadow": "enabled",
    "enable_hrr_spatial_scene": "spatial_scene_enabled",
}
_FILE_KEY_INT = {
    "hrr_shadow_dim": ("dim", 16),
    "hrr_sample_every_ticks": ("sample_every_ticks", 1),
    "hrr_spatial_scene_sample_every_ticks": ("spatial_scene_sample_every_ticks", 1),
}


def _resolve_runtime_flags_path(override: Optional[Path] = None) -> Path:
    """Resolve the runtime-flags file location.

    Precedence: explicit ``override`` arg > ``JARVIS_RUNTIME_FLAGS`` env >
    ``~/.jarvis/runtime_flags.json``.
    """
    if override is not None:
        return Path(override)
    env_path = os.environ.get(RUNTIME_FLAGS_PATH_ENV)
    if env_path:
        return Path(env_path)
    return DEFAULT_RUNTIME_FLAGS_PATH


def _load_runtime_flags(
    path: Optional[Path] = None,
) -> Tuple[Mapping[str, Any], Optional[str]]:
    """Read and validate the runtime-flags JSON file.

    Returns ``(flags_dict, error_str_or_None)``. Missing file is **not**
    an error — returns ``({}, None)``. Malformed JSON or unexpected
    payload shape is reported as an error string and the dict returned
    is empty (safe default).
    """
    target = _resolve_runtime_flags_path(path)
    if not target.exists():
        return {}, None
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"runtime_flags read failed ({type(exc).__name__}: {exc})"
        logger.warning("HRR %s — falling back to safe defaults", msg)
        return {}, msg
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"runtime_flags JSON malformed ({exc.msg} at line {exc.lineno})"
        logger.warning("HRR %s — falling back to safe defaults", msg)
        return {}, msg
    if not isinstance(data, dict):
        msg = "runtime_flags root must be a JSON object"
        logger.warning("HRR %s — falling back to safe defaults", msg)
        return {}, msg
    return data, None


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _bool_from_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_from_env(name: str, minimum: int = 1) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return max(minimum, v)


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    return None


def _coerce_int(value: Any, minimum: int) -> Optional[int]:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return max(minimum, v)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HRRRuntimeConfig:
    """Immutable snapshot of HRR runtime flags taken at boot.

    Fields mirror the env-only contract for backwards compatibility. The
    new ``flag_sources`` field carries provenance (which precedence layer
    set each flag) so the dashboard / API can show operators *why* HRR is
    enabled. ``runtime_flags_error`` carries any malformed-file warning
    (None when the file is absent or parsed cleanly).
    """

    enabled: bool = False
    dim: int = 1024
    sample_every_ticks: int = 50
    tick_interval_s: float = 0.1  # brain tick cadence

    # P5 mental-world spatial scene graph — twin gate, opt-in, default OFF.
    spatial_scene_enabled: bool = False
    spatial_scene_sample_every_ticks: int = 50

    # P5.1 provenance. Tuple-of-pairs because dataclass(frozen=True) needs hashable defaults.
    flag_sources: Tuple[Tuple[str, str], ...] = field(
        default_factory=lambda: (
            ("enabled", SOURCE_DEFAULT),
            ("dim", SOURCE_DEFAULT),
            ("sample_every_ticks", SOURCE_DEFAULT),
            ("spatial_scene_enabled", SOURCE_DEFAULT),
            ("spatial_scene_sample_every_ticks", SOURCE_DEFAULT),
        )
    )
    runtime_flags_path: Optional[str] = None
    runtime_flags_error: Optional[str] = None

    @property
    def sample_interval_s(self) -> float:
        """Approximate wall-clock seconds between HRR world-shadow samples."""
        return float(self.sample_every_ticks) * float(self.tick_interval_s)

    @property
    def spatial_scene_active(self) -> bool:
        """True iff the P5 lane should sample on the current boot.

        Requires the P4 master gate AND the P5 twin gate. Either being off
        suppresses sampling. Both default off.
        """
        return bool(self.enabled and self.spatial_scene_enabled)

    @property
    def spatial_scene_sample_interval_s(self) -> float:
        """Approximate wall-clock seconds between P5 spatial-shadow samples."""
        return float(self.spatial_scene_sample_every_ticks) * float(self.tick_interval_s)

    @property
    def flag_sources_dict(self) -> dict[str, str]:
        """Dict view of :attr:`flag_sources` for JSON serialization."""
        return {k: v for k, v in self.flag_sources}

    @property
    def enabled_source(self) -> str:
        """Where the master ``enabled`` flag came from (for the dashboard summary)."""
        return self.flag_sources_dict.get("enabled", SOURCE_DEFAULT)

    @property
    def spatial_scene_enabled_source(self) -> str:
        """Where the P5 ``spatial_scene_enabled`` flag came from."""
        return self.flag_sources_dict.get("spatial_scene_enabled", SOURCE_DEFAULT)

    @classmethod
    def from_env(
        cls,
        runtime_flags_path: Optional[Path] = None,
    ) -> "HRRRuntimeConfig":
        """Build a config snapshot using default → file → env precedence.

        ``runtime_flags_path`` is mostly for tests; production callers
        should pass ``None`` and let the resolver pick up
        ``~/.jarvis/runtime_flags.json`` (or ``JARVIS_RUNTIME_FLAGS`` env
        if exported).
        """
        # Layer 0: hardcoded safe defaults
        enabled: bool = False
        dim: int = 1024
        sample_every_ticks: int = 50
        spatial_scene_enabled: bool = False
        spatial_scene_sample_every_ticks: int = 50

        sources: dict[str, str] = {
            "enabled": SOURCE_DEFAULT,
            "dim": SOURCE_DEFAULT,
            "sample_every_ticks": SOURCE_DEFAULT,
            "spatial_scene_enabled": SOURCE_DEFAULT,
            "spatial_scene_sample_every_ticks": SOURCE_DEFAULT,
        }

        # Layer 1: persistent runtime-flags file
        resolved_path = _resolve_runtime_flags_path(runtime_flags_path)
        flags, file_error = _load_runtime_flags(resolved_path)

        for file_key, attr_name in _FILE_KEY_BOOL.items():
            if file_key in flags:
                v = _coerce_bool(flags[file_key])
                if v is not None:
                    if attr_name == "enabled":
                        enabled = v
                    elif attr_name == "spatial_scene_enabled":
                        spatial_scene_enabled = v
                    sources[attr_name] = SOURCE_RUNTIME_FLAGS

        for file_key, (attr_name, minimum) in _FILE_KEY_INT.items():
            if file_key in flags:
                v = _coerce_int(flags[file_key], minimum)
                if v is not None:
                    if attr_name == "dim":
                        dim = v
                    elif attr_name == "sample_every_ticks":
                        sample_every_ticks = v
                    elif attr_name == "spatial_scene_sample_every_ticks":
                        spatial_scene_sample_every_ticks = v
                    sources[attr_name] = SOURCE_RUNTIME_FLAGS

        # Layer 2: environment variables (highest precedence)
        env_master = _bool_from_env("ENABLE_HRR_SHADOW")
        if env_master is not None:
            enabled = env_master
            sources["enabled"] = SOURCE_ENVIRONMENT

        env_dim = _int_from_env("HRR_SHADOW_DIM", minimum=16)
        if env_dim is not None:
            dim = env_dim
            sources["dim"] = SOURCE_ENVIRONMENT

        env_world_every = _int_from_env("HRR_SHADOW_SAMPLE_EVERY_TICKS", minimum=1)
        if env_world_every is not None:
            sample_every_ticks = env_world_every
            sources["sample_every_ticks"] = SOURCE_ENVIRONMENT

        env_p5 = _bool_from_env("ENABLE_HRR_SPATIAL_SCENE")
        if env_p5 is not None:
            spatial_scene_enabled = env_p5
            sources["spatial_scene_enabled"] = SOURCE_ENVIRONMENT

        env_p5_every = _int_from_env(
            "HRR_SPATIAL_SCENE_SAMPLE_EVERY_TICKS", minimum=1
        )
        if env_p5_every is not None:
            spatial_scene_sample_every_ticks = env_p5_every
            sources["spatial_scene_sample_every_ticks"] = SOURCE_ENVIRONMENT

        flag_sources_tuple = tuple(sorted(sources.items()))

        return cls(
            enabled=enabled,
            dim=dim,
            sample_every_ticks=sample_every_ticks,
            spatial_scene_enabled=spatial_scene_enabled,
            spatial_scene_sample_every_ticks=spatial_scene_sample_every_ticks,
            flag_sources=flag_sources_tuple,
            runtime_flags_path=str(resolved_path),
            runtime_flags_error=file_error,
        )

    @classmethod
    def disabled(cls) -> "HRRRuntimeConfig":
        """Explicit OFF snapshot — useful for tests that need a known-off config."""
        return cls(enabled=False)


__all__ = [
    "HRRRuntimeConfig",
    "SOURCE_DEFAULT",
    "SOURCE_RUNTIME_FLAGS",
    "SOURCE_ENVIRONMENT",
    "DEFAULT_RUNTIME_FLAGS_PATH",
    "RUNTIME_FLAGS_PATH_ENV",
]
