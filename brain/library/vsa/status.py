"""Read-only HRR status aggregator for `/api/hrr/status`.

Pure reader: reads the latest Stage 0 evidence JSON, the boot-time
``HRRRuntimeConfig``, and (once Commits 4-6 land) the bounded ring-buffer
counters from the world encoder / simulation shadow / recall advisor. All
authority flags are hard-coded ``false`` here — enforcement of that is done
structurally by the validation-pack checks.

Consumers MUST NOT mutate anything based on this status; it is a dashboard
read path only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from library.vsa.runtime_config import HRRRuntimeConfig


# ---------------------------------------------------------------------------
# Evidence lookup
# ---------------------------------------------------------------------------

_REPO_ROOT_MARKERS = ("brain", "docs", "pi")


def _repo_root_from_here() -> Optional[Path]:
    """Walk up from this file until we find a directory that looks like the repo root."""
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if all((candidate / m).exists() for m in _REPO_ROOT_MARKERS):
            return candidate
    return None


def _stage0_evidence_path() -> Optional[Path]:
    root = _repo_root_from_here()
    if root is None:
        return None
    return root / "docs" / "validation_reports" / "evidence" / "hrr_stage0.json"


def load_stage0_evidence() -> Optional[Dict[str, Any]]:
    """Load the Stage 0 exercise JSON if it exists and parses. No exceptions raised."""
    path = _stage0_evidence_path()
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Shadow counter registries (populated in Commits 4-6)
# ---------------------------------------------------------------------------

# Each is a callable returning a dict with at least:
#   {"enabled": bool, "samples_total": int, "samples_retained": int, "ring_capacity": int}
# plus any per-shadow metrics (binding_cleanliness, help_rate, ...).
# P5.1: registered boot-time runtime config so every reader / endpoint
# reports the SAME provenance the engine actually booted with. The
# dashboard never re-reads env or the runtime-flags file mid-run; it
# always echoes what the engine pinned at startup.
_REGISTERED_RUNTIME_CONFIG: Optional[HRRRuntimeConfig] = None

_WORLD_SHADOW_READER = None
_SIMULATION_SHADOW_READER = None
_RECALL_ADVISORY_READER = None
_SPATIAL_SCENE_READER = None

# Recent-sample readers: callables that take a single ``n: int`` argument and
# return a list of dicts (newest last) containing the retained metrics for
# each ring. Populated alongside the status readers in Commits 4-6.
_WORLD_SHADOW_RECENT = None
_SIMULATION_SHADOW_RECENT = None
_RECALL_ADVISORY_RECENT = None
_SPATIAL_SCENE_RECENT = None


def register_world_shadow_reader(fn) -> None:
    """Wire in the world-shadow reader. Called once at engine construction."""
    global _WORLD_SHADOW_READER
    _WORLD_SHADOW_READER = fn


def register_simulation_shadow_reader(fn) -> None:
    global _SIMULATION_SHADOW_READER
    _SIMULATION_SHADOW_READER = fn


def register_recall_advisory_reader(fn) -> None:
    global _RECALL_ADVISORY_READER
    _RECALL_ADVISORY_READER = fn


def register_world_shadow_recent(fn) -> None:
    """Wire a ``recent(n)`` reader for world-shadow samples."""
    global _WORLD_SHADOW_RECENT
    _WORLD_SHADOW_RECENT = fn


def register_simulation_shadow_recent(fn) -> None:
    """Wire a ``recent(n)`` reader for simulation-shadow traces."""
    global _SIMULATION_SHADOW_RECENT
    _SIMULATION_SHADOW_RECENT = fn


def register_recall_advisory_recent(fn) -> None:
    """Wire a ``recent(n)`` reader for recall-advisor observations."""
    global _RECALL_ADVISORY_RECENT
    _RECALL_ADVISORY_RECENT = fn


def register_spatial_scene_reader(fn) -> None:
    """Wire in the P5 spatial-scene shadow reader. Called once at engine boot."""
    global _SPATIAL_SCENE_READER
    _SPATIAL_SCENE_READER = fn


def register_spatial_scene_recent(fn) -> None:
    """Wire a ``recent(n)`` reader for P5 spatial-scene samples."""
    global _SPATIAL_SCENE_RECENT
    _SPATIAL_SCENE_RECENT = fn


def register_runtime_config(cfg: Optional[HRRRuntimeConfig]) -> None:
    """Pin the boot-time runtime config so all readers report the same provenance.

    Called once by the consciousness engine after :meth:`HRRRuntimeConfig.from_env`.
    Pass ``None`` to clear (used by tests).
    """
    global _REGISTERED_RUNTIME_CONFIG
    _REGISTERED_RUNTIME_CONFIG = cfg


def get_runtime_config() -> Optional[HRRRuntimeConfig]:
    """Return the engine's registered runtime config, or ``None`` if not booted."""
    return _REGISTERED_RUNTIME_CONFIG


def _safe_recent(fn, n: int) -> list:
    if fn is None:
        return []
    try:
        out = fn(int(n))
    except Exception:
        return []
    if not isinstance(out, list):
        return []
    # Defensive sanitize: only emit dicts with JSON-safe values.
    return [x for x in out if isinstance(x, dict)]


def _safe_call(fn, default: Dict[str, Any]) -> Dict[str, Any]:
    if fn is None:
        return default
    try:
        out = fn()
    except Exception:
        return default
    if not isinstance(out, dict):
        return default
    return out


def _default_world_block() -> Dict[str, Any]:
    return {
        "enabled": False,
        "samples_total": 0,
        "samples_retained": 0,
        "ring_capacity": 500,
        "binding_cleanliness": None,
    }


def _default_simulation_block() -> Dict[str, Any]:
    return {
        "enabled": False,
        "samples_total": 0,
        "samples_retained": 0,
        "ring_capacity": 200,
    }


def _default_recall_block() -> Dict[str, Any]:
    return {
        "enabled": False,
        "samples_total": 0,
        "samples_retained": 0,
        "ring_capacity": 500,
        "help_rate": None,
    }


def _default_spatial_scene_block() -> Dict[str, Any]:
    return {
        "enabled": False,
        "samples_total": 0,
        "samples_retained": 0,
        "ring_capacity": 500,
        "entities_encoded": None,
        "relations_encoded": None,
        "binding_cleanliness": None,
        "cleanup_accuracy": None,
        "relation_recovery": None,
        "cleanup_failures": 0,
        "similarity_to_previous": None,
        "spatial_hrr_side_effects": 0,
        "reason": None,
    }


# ---------------------------------------------------------------------------
# Public composition
# ---------------------------------------------------------------------------

_STATIC_AUTHORITY_FLAGS = {
    "policy_influence": False,
    "belief_write_enabled": False,
    "canonical_memory": False,
    "autonomy_influence": False,
    "llm_raw_vector_exposure": False,
    "soul_integrity_influence": False,
}


def get_hrr_status(config: Optional[HRRRuntimeConfig] = None) -> Dict[str, Any]:
    """Build the payload for `GET /api/hrr/status`.

    The payload is deliberately verbose so dashboards and the validation pack
    can each cherry-pick the fields they need. The ``stage`` field is
    informational only — the **public status marker** remains ``PRE-MATURE``
    and is reported by ``/api/meta/status-markers``, not by this endpoint.
    """
    cfg = (
        config
        if config is not None
        else (_REGISTERED_RUNTIME_CONFIG or HRRRuntimeConfig.from_env())
    )
    evidence = load_stage0_evidence()

    if evidence:
        gates = evidence.get("gates") or {}
        latest_exercise = {
            "generated_at": evidence.get("generated_at"),
            "cleanup_accuracy_at_8": gates.get("cleanup_accuracy_at_8"),
            "cleanup_accuracy_at_16": gates.get("cleanup_accuracy_at_16"),
            "false_positive_rate": gates.get("false_positive_rate"),
            "all_pass": gates.get("all_pass"),
            "hrr_side_effects": evidence.get("hrr_side_effects", 0),
            "schema_version": evidence.get("schema_version"),
        }
    else:
        latest_exercise = None

    world = _safe_call(_WORLD_SHADOW_READER, _default_world_block())
    simulation = _safe_call(_SIMULATION_SHADOW_READER, _default_simulation_block())
    recall = _safe_call(_RECALL_ADVISORY_READER, _default_recall_block())
    spatial_scene = _safe_call(_SPATIAL_SCENE_READER, _default_spatial_scene_block())

    return {
        "status": "PRE-MATURE",
        "stage": "shadow_substrate_operational",
        "enabled": bool(cfg.enabled),
        "dim": int(cfg.dim),
        "backend": "numpy_fft_cpu",
        "sample_every_ticks": int(cfg.sample_every_ticks),
        "sample_interval_s": float(cfg.sample_interval_s),
        "spatial_scene_enabled": bool(getattr(cfg, "spatial_scene_enabled", False)),
        "spatial_scene_sample_every_ticks": int(
            getattr(cfg, "spatial_scene_sample_every_ticks", 50)
        ),
        # P5.1 provenance — purely informational, dashboard surfaces only.
        "enabled_source": getattr(cfg, "enabled_source", "default"),
        "spatial_scene_enabled_source": getattr(
            cfg, "spatial_scene_enabled_source", "default"
        ),
        "flag_sources": dict(getattr(cfg, "flag_sources_dict", {})),
        "runtime_flags_path": getattr(cfg, "runtime_flags_path", None),
        "runtime_flags_error": getattr(cfg, "runtime_flags_error", None),
        "latest_exercise": latest_exercise,
        "world_shadow": world,
        "simulation_shadow": simulation,
        "recall_advisory": recall,
        "spatial_scene": spatial_scene,
        **_STATIC_AUTHORITY_FLAGS,
    }


def get_hrr_samples(
    n_world: int = 20,
    n_simulation: int = 20,
    n_recall: int = 20,
    n_spatial_scene: int = 20,
    config: Optional[HRRRuntimeConfig] = None,
) -> Dict[str, Any]:
    """Build the payload for ``GET /api/hrr/samples``.

    Returns the last ``n_*`` entries of each ring, newest last. When the HRR
    substrate is disabled, returns empty lists.

    Every entry is a dict of **scalar metrics only** — raw HRR vectors are
    never exposed by this endpoint (enforced upstream by each ring owner
    stripping vectors before ``deque.append``).
    """
    cfg = (
        config
        if config is not None
        else (_REGISTERED_RUNTIME_CONFIG or HRRRuntimeConfig.from_env())
    )
    n_world = max(0, min(500, int(n_world)))
    n_simulation = max(0, min(200, int(n_simulation)))
    n_recall = max(0, min(500, int(n_recall)))
    n_spatial_scene = max(0, min(500, int(n_spatial_scene)))

    return {
        "enabled": bool(cfg.enabled),
        "spatial_scene_enabled": bool(getattr(cfg, "spatial_scene_enabled", False)),
        "stage": "shadow_substrate_operational",
        "world_shadow": _safe_recent(_WORLD_SHADOW_RECENT, n_world),
        "simulation_shadow": _safe_recent(_SIMULATION_SHADOW_RECENT, n_simulation),
        "recall_advisory": _safe_recent(_RECALL_ADVISORY_RECENT, n_recall),
        "spatial_scene": _safe_recent(_SPATIAL_SCENE_RECENT, n_spatial_scene),
        **_STATIC_AUTHORITY_FLAGS,
    }
