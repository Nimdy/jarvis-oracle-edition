"""Tests for P5.1 HRR runtime-flag persistence.

Verifies the layered precedence (default → ``~/.jarvis/runtime_flags.json``
→ environment), the per-flag provenance map, malformed-file safety, the
helper CLI script, and the dashboard surface (``enabled_source`` on
``/api/hrr/status`` and the mental-world facade).
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest


HRR_ENV_KEYS = (
    "ENABLE_HRR_SHADOW",
    "ENABLE_HRR_SPATIAL_SCENE",
    "HRR_SHADOW_DIM",
    "HRR_SHADOW_SAMPLE_EVERY_TICKS",
    "HRR_SPATIAL_SCENE_SAMPLE_EVERY_TICKS",
    "JARVIS_RUNTIME_FLAGS",
)


@pytest.fixture(autouse=True)
def _clean_hrr_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every HRR-relevant env var so each test sees a known baseline."""
    for k in HRR_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture()
def runtime_flags_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the resolver at a fresh tmp file via ``JARVIS_RUNTIME_FLAGS``."""
    target = tmp_path / "runtime_flags.json"
    monkeypatch.setenv("JARVIS_RUNTIME_FLAGS", str(target))
    return target


def _import_fresh_runtime_config():
    from library.vsa import runtime_config

    return importlib.reload(runtime_config)


# ---------------------------------------------------------------------------
# Layered precedence
# ---------------------------------------------------------------------------


def test_default_when_no_file_no_env(runtime_flags_path: Path) -> None:
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is False
    assert cfg.spatial_scene_enabled is False
    assert cfg.spatial_scene_active is False
    assert cfg.enabled_source == rc.SOURCE_DEFAULT
    assert cfg.spatial_scene_enabled_source == rc.SOURCE_DEFAULT
    assert cfg.runtime_flags_error is None
    assert all(v == rc.SOURCE_DEFAULT for v in cfg.flag_sources_dict.values())


def test_runtime_flags_file_enables_both(runtime_flags_path: Path) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {
                "enable_hrr_shadow": True,
                "enable_hrr_spatial_scene": True,
                "hrr_sample_every_ticks": 25,
                "hrr_spatial_scene_sample_every_ticks": 30,
                "hrr_shadow_dim": 2048,
            }
        ),
        encoding="utf-8",
    )
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.spatial_scene_enabled is True
    assert cfg.spatial_scene_active is True
    assert cfg.dim == 2048
    assert cfg.sample_every_ticks == 25
    assert cfg.spatial_scene_sample_every_ticks == 30
    assert cfg.enabled_source == rc.SOURCE_RUNTIME_FLAGS
    assert cfg.spatial_scene_enabled_source == rc.SOURCE_RUNTIME_FLAGS
    sources = cfg.flag_sources_dict
    assert sources["dim"] == rc.SOURCE_RUNTIME_FLAGS
    assert sources["sample_every_ticks"] == rc.SOURCE_RUNTIME_FLAGS
    assert sources["spatial_scene_sample_every_ticks"] == rc.SOURCE_RUNTIME_FLAGS


def test_env_overrides_runtime_flags_to_false(
    runtime_flags_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {"enable_hrr_shadow": True, "enable_hrr_spatial_scene": True}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENABLE_HRR_SHADOW", "0")
    monkeypatch.setenv("ENABLE_HRR_SPATIAL_SCENE", "0")
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is False
    assert cfg.spatial_scene_enabled is False
    assert cfg.spatial_scene_active is False
    assert cfg.enabled_source == rc.SOURCE_ENVIRONMENT
    assert cfg.spatial_scene_enabled_source == rc.SOURCE_ENVIRONMENT


def test_env_overrides_runtime_flags_to_true(
    runtime_flags_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {"enable_hrr_shadow": False, "enable_hrr_spatial_scene": False}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENABLE_HRR_SHADOW", "1")
    monkeypatch.setenv("ENABLE_HRR_SPATIAL_SCENE", "1")
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.spatial_scene_enabled is True
    assert cfg.spatial_scene_active is True
    assert cfg.enabled_source == rc.SOURCE_ENVIRONMENT


def test_partial_file_partial_env_provenance(
    runtime_flags_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File toggles master ON; env independently toggles spatial scene ON."""
    runtime_flags_path.write_text(
        json.dumps({"enable_hrr_shadow": True}), encoding="utf-8"
    )
    monkeypatch.setenv("ENABLE_HRR_SPATIAL_SCENE", "true")
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.spatial_scene_enabled is True
    assert cfg.enabled_source == rc.SOURCE_RUNTIME_FLAGS
    assert cfg.spatial_scene_enabled_source == rc.SOURCE_ENVIRONMENT


# ---------------------------------------------------------------------------
# Malformed / hostile inputs
# ---------------------------------------------------------------------------


def test_malformed_json_falls_back_to_default(runtime_flags_path: Path) -> None:
    runtime_flags_path.write_text("{not: 'valid' json,,}", encoding="utf-8")
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is False
    assert cfg.spatial_scene_enabled is False
    assert cfg.runtime_flags_error is not None
    assert "JSON malformed" in cfg.runtime_flags_error
    assert cfg.enabled_source == rc.SOURCE_DEFAULT


def test_non_object_root_falls_back_to_default(runtime_flags_path: Path) -> None:
    runtime_flags_path.write_text("[1, 2, 3]", encoding="utf-8")
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is False
    assert cfg.runtime_flags_error is not None
    assert "object" in cfg.runtime_flags_error.lower()


def test_unknown_keys_are_ignored(runtime_flags_path: Path) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {
                "enable_hrr_shadow": True,
                "unrelated_key": "ignored",
                "spatial_lane_speed": 9.81,
            }
        ),
        encoding="utf-8",
    )
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.spatial_scene_enabled is False  # not set in file
    assert cfg.runtime_flags_error is None


def test_int_fields_clamped_to_minimum(runtime_flags_path: Path) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {
                "hrr_sample_every_ticks": 0,
                "hrr_spatial_scene_sample_every_ticks": -50,
                "hrr_shadow_dim": 1,
            }
        ),
        encoding="utf-8",
    )
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.sample_every_ticks == 1
    assert cfg.spatial_scene_sample_every_ticks == 1
    assert cfg.dim == 16


def test_string_truthy_coerced(runtime_flags_path: Path) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {"enable_hrr_shadow": "yes", "enable_hrr_spatial_scene": "off"}
        ),
        encoding="utf-8",
    )
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.spatial_scene_enabled is False


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_explicit_path_argument_wins(tmp_path: Path) -> None:
    rc = _import_fresh_runtime_config()
    target = tmp_path / "explicit.json"
    target.write_text(
        json.dumps(
            {"enable_hrr_shadow": True, "enable_hrr_spatial_scene": True}
        ),
        encoding="utf-8",
    )
    cfg = rc.HRRRuntimeConfig.from_env(runtime_flags_path=target)
    assert cfg.enabled is True
    assert cfg.spatial_scene_enabled is True
    assert cfg.runtime_flags_path == str(target)


def test_jarvis_runtime_flags_env_resolves_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "via_env.json"
    target.write_text(json.dumps({"enable_hrr_shadow": True}), encoding="utf-8")
    monkeypatch.setenv("JARVIS_RUNTIME_FLAGS", str(target))
    rc = _import_fresh_runtime_config()
    cfg = rc.HRRRuntimeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.runtime_flags_path == str(target)


# ---------------------------------------------------------------------------
# /api/hrr/status surface
# ---------------------------------------------------------------------------


def test_status_payload_exposes_enabled_source(
    runtime_flags_path: Path,
) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {"enable_hrr_shadow": True, "enable_hrr_spatial_scene": True}
        ),
        encoding="utf-8",
    )
    _import_fresh_runtime_config()
    from library.vsa import status as status_mod

    importlib.reload(status_mod)
    payload = status_mod.get_hrr_status()
    assert payload["enabled"] is True
    assert payload["spatial_scene_enabled"] is True
    assert payload["enabled_source"] == "runtime_flags"
    assert payload["spatial_scene_enabled_source"] == "runtime_flags"
    assert payload["flag_sources"]["enabled"] == "runtime_flags"
    assert payload["runtime_flags_path"].endswith("runtime_flags.json")
    assert payload["runtime_flags_error"] is None
    # Authority block must still be pinned.
    assert payload["policy_influence"] is False
    assert payload["belief_write_enabled"] is False
    assert payload["llm_raw_vector_exposure"] is False


def test_status_payload_when_engine_registers_config(
    runtime_flags_path: Path,
) -> None:
    """A registered config trumps env re-reads at the API layer."""
    rc = _import_fresh_runtime_config()
    from library.vsa import status as status_mod

    importlib.reload(status_mod)

    # Create an explicit "spoofed" config that disagrees with the file.
    pinned = rc.HRRRuntimeConfig(
        enabled=True,
        spatial_scene_enabled=True,
        flag_sources=tuple(
            sorted(
                {
                    "enabled": rc.SOURCE_ENVIRONMENT,
                    "dim": rc.SOURCE_DEFAULT,
                    "sample_every_ticks": rc.SOURCE_DEFAULT,
                    "spatial_scene_enabled": rc.SOURCE_RUNTIME_FLAGS,
                    "spatial_scene_sample_every_ticks": rc.SOURCE_DEFAULT,
                }.items()
            )
        ),
        runtime_flags_path="/fake/path",
    )
    status_mod.register_runtime_config(pinned)
    try:
        payload = status_mod.get_hrr_status()
        assert payload["enabled_source"] == "environment"
        assert payload["spatial_scene_enabled_source"] == "runtime_flags"
        assert payload["runtime_flags_path"] == "/fake/path"
    finally:
        status_mod.register_runtime_config(None)


# ---------------------------------------------------------------------------
# /api/hrr/scene + /api/hrr/scene/history surface
# ---------------------------------------------------------------------------


def test_mental_world_facade_emits_enabled_source(
    runtime_flags_path: Path,
) -> None:
    runtime_flags_path.write_text(
        json.dumps(
            {"enable_hrr_shadow": True, "enable_hrr_spatial_scene": True}
        ),
        encoding="utf-8",
    )
    rc = _import_fresh_runtime_config()
    from library.vsa import status as status_mod

    importlib.reload(status_mod)
    cfg = rc.HRRRuntimeConfig.from_env()
    status_mod.register_runtime_config(cfg)

    from cognition import mental_world

    importlib.reload(mental_world)
    try:
        empty = mental_world.get_state()
        assert empty["enabled_source"] == "runtime_flags"
        assert empty["spatial_scene_enabled_source"] == "runtime_flags"
        assert empty["runtime_flags_path"].endswith("runtime_flags.json")
        assert empty["writes_memory"] is False
        assert empty["llm_raw_vector_exposure"] is False
        assert empty["no_raw_vectors_in_api"] is True

        history = mental_world.get_history(limit=5)
        assert history["enabled_source"] == "runtime_flags"
        assert history["spatial_scene_enabled_source"] == "runtime_flags"
        assert history["no_raw_vectors_in_api"] is True
    finally:
        status_mod.register_runtime_config(None)


def test_mental_world_facade_safe_when_no_config_registered() -> None:
    from library.vsa import status as status_mod
    from cognition import mental_world

    importlib.reload(mental_world)
    status_mod.register_runtime_config(None)
    payload = mental_world.get_state()
    # When no config is registered, provenance keys are simply absent —
    # they must NEVER be reported with stale data.
    assert "enabled_source" not in payload or payload["enabled_source"] in (
        "default",
        "runtime_flags",
        "environment",
    )
    assert payload["writes_memory"] is False
    assert payload["no_raw_vectors_in_api"] is True


# ---------------------------------------------------------------------------
# Helper script CLI
# ---------------------------------------------------------------------------


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "set_hrr_runtime_flags.py"
)


def _run_helper(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        env=full_env,
        check=False,
    )


def test_helper_enable_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "flags.json"
    res = _run_helper(["--enable", "--path", str(target)])
    assert res.returncode == 0, res.stderr
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["enable_hrr_shadow"] is True
    assert payload["enable_hrr_spatial_scene"] is True


def test_helper_disable_writes_false(tmp_path: Path) -> None:
    target = tmp_path / "flags.json"
    target.write_text(
        json.dumps(
            {"enable_hrr_shadow": True, "enable_hrr_spatial_scene": True}
        ),
        encoding="utf-8",
    )
    res = _run_helper(["--disable", "--path", str(target)])
    assert res.returncode == 0, res.stderr
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["enable_hrr_shadow"] is False
    assert payload["enable_hrr_spatial_scene"] is False


def test_helper_status_reports_provenance(tmp_path: Path) -> None:
    target = tmp_path / "flags.json"
    target.write_text(
        json.dumps(
            {"enable_hrr_shadow": True, "enable_hrr_spatial_scene": True}
        ),
        encoding="utf-8",
    )
    res = _run_helper(
        ["--status", "--path", str(target)],
        env={"JARVIS_RUNTIME_FLAGS": str(target)},
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "enabled" in out
    assert "runtime_flags" in out
    assert "spatial_scene_enabled" in out


def test_helper_status_reports_default_with_no_file(tmp_path: Path) -> None:
    target = tmp_path / "flags.json"
    res = _run_helper(
        ["--status", "--path", str(target)],
        env={"JARVIS_RUNTIME_FLAGS": str(target)},
    )
    assert res.returncode == 0, res.stderr
    assert "default" in res.stdout
    assert "False" in res.stdout
