"""Mental-world facade — read-only service over the P5 spatial HRR shadow.

Thin adapter that sits between :class:`cognition.hrr_spatial_encoder.HRRSpatialShadow`
(owned by the consciousness engine) and the HTTP / dashboard / Pi consumers.

Responsibilities:

* Expose a **read-only** view of the latest derived scene graph and its
  rolling history.
* Pin authority flags to ``false`` in every response — there is no code
  path in this module (or anywhere downstream) that can flip them.
* Never expose raw HRR vectors. The upstream shadow strips them before
  samples land in the ring; this module only forwards the already-
  stripped payloads.

**Zero authority**: this facade never writes canonical memory / beliefs
/ policy / autonomy / identity. The structural scan in
``jarvis_eval.validation_pack._scan_hrr_forbidden_imports`` enforces that
guarantee across the P5 module roots.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Authority contract — pinned false in every payload.
# ---------------------------------------------------------------------------

AUTHORITY_FLAGS: Dict[str, bool] = {
    "writes_memory": False,
    "writes_beliefs": False,
    "influences_policy": False,
    "influences_autonomy": False,
    "soul_integrity_influence": False,
    "llm_raw_vector_exposure": False,
    "no_raw_vectors_in_api": True,
}

EMPTY_REASON: str = "canonical_spatial_state_unavailable"


# ---------------------------------------------------------------------------
# Service (module-level singleton — matches the rest of the HRR lane).
# ---------------------------------------------------------------------------

# The engine (Commit 5) calls register_shadow(...) to wire its HRRSpatialShadow
# instance in at boot time. Until then all reads return the empty scene.
_SHADOW: Optional[Any] = None

# Optional override: a callable returning a dict-shaped "current scene payload"
# used by fixture-based tests. Never used in production.
_STATE_OVERRIDE: Optional[Callable[[], Optional[Dict[str, Any]]]] = None
_HISTORY_OVERRIDE: Optional[Callable[[int], List[Dict[str, Any]]]] = None


def register_shadow(shadow: Any) -> None:
    """Wire the engine's :class:`HRRSpatialShadow` into the facade.

    Called once at engine construction. Safe to call with ``None`` to
    clear (used by tests).
    """
    global _SHADOW
    _SHADOW = shadow


def register_state_override(fn: Optional[Callable[[], Optional[Dict[str, Any]]]]) -> None:
    """Install a test override for :func:`get_state`. Pass ``None`` to clear."""
    global _STATE_OVERRIDE
    _STATE_OVERRIDE = fn


def register_history_override(
    fn: Optional[Callable[[int], List[Dict[str, Any]]]],
) -> None:
    """Install a test override for :func:`get_history`. Pass ``None`` to clear."""
    global _HISTORY_OVERRIDE
    _HISTORY_OVERRIDE = fn


# ---------------------------------------------------------------------------
# Public read API
# ---------------------------------------------------------------------------


def _runtime_provenance() -> Dict[str, Any]:
    """Read the boot-time HRR runtime config (registered by the engine).

    Best-effort; never raises. Returns ``{}`` when the engine has not yet
    registered a config (e.g. fresh imports, tests).
    """
    try:
        from library.vsa.status import get_runtime_config
    except Exception:
        return {}
    cfg = get_runtime_config()
    if cfg is None:
        return {}
    return {
        "enabled_source": getattr(cfg, "enabled_source", "default"),
        "spatial_scene_enabled_source": getattr(
            cfg, "spatial_scene_enabled_source", "default"
        ),
        "runtime_flags_path": getattr(cfg, "runtime_flags_path", None),
        "runtime_flags_error": getattr(cfg, "runtime_flags_error", None),
    }


def _empty_state() -> Dict[str, Any]:
    """Return the canonical empty-scene payload with all authority flags pinned."""
    return {
        "status": "PRE-MATURE",
        "lane": "spatial_hrr_mental_world",
        "enabled": False,
        "timestamp": round(time.time(), 3),
        "entity_count": 0,
        "active_entity_count": 0,
        "removed_entity_count": 0,
        "relation_count": 0,
        "entities": [],
        "relations": [],
        "source": {
            "scene_update_count": 0,
            "track_count": 0,
            "anchor_count": 0,
            "calibration_version": 0,
        },
        "metrics": {
            "entities_encoded": 0,
            "relations_encoded": 0,
            "binding_cleanliness": None,
            "cleanup_accuracy": None,
            "relation_recovery": None,
            "cleanup_failures": 0,
            "similarity_to_previous": None,
            "spatial_hrr_side_effects": 0,
        },
        "reason": EMPTY_REASON,
        "tick": 0,
        **_runtime_provenance(),
        **AUTHORITY_FLAGS,
    }


def _wrap(payload: Dict[str, Any], *, enabled: bool) -> Dict[str, Any]:
    """Wrap a shadow-emitted payload with the mental-world envelope.

    Strips any unexpected ``vector`` / ``ndarray`` keys as a defense-in-
    depth measure (the shadow should have removed them already).
    """
    out = dict(payload)
    # Remove any stray vector keys defensively.
    for k in list(out.keys()):
        if "vector" in k.lower() or "ndarray" in k.lower():
            out.pop(k, None)
    out["status"] = "PRE-MATURE"
    out["lane"] = "spatial_hrr_mental_world"
    out["enabled"] = bool(enabled)
    out.setdefault("entity_count", len(out.get("entities") or []))
    entities = out.get("entities") or []
    stateful_entities = [
        e for e in entities
        if isinstance(e, dict) and "state" in e
    ]
    if stateful_entities:
        out.setdefault(
            "active_entity_count",
            sum(1 for e in stateful_entities if e.get("state") != "removed"),
        )
        out.setdefault(
            "removed_entity_count",
            sum(1 for e in stateful_entities if e.get("state") == "removed"),
        )
    else:
        # Backward-compatible legacy payloads only reported ``entity_count``.
        out.setdefault("active_entity_count", int(out.get("entity_count") or 0))
        out.setdefault("removed_entity_count", 0)
    out.setdefault("relation_count", len(out.get("relations") or []))
    out.update(_runtime_provenance())
    out.update(AUTHORITY_FLAGS)
    return out


def get_state() -> Dict[str, Any]:
    """Return the latest derived mental-world scene payload.

    When the P5 lane is disabled (twin gate off) or no sample has been
    recorded yet, returns :func:`_empty_state` with
    ``reason="canonical_spatial_state_unavailable"``.
    """
    if _STATE_OVERRIDE is not None:
        try:
            override = _STATE_OVERRIDE()
        except Exception:
            override = None
        if override is not None:
            return _wrap(override, enabled=bool(override.get("enabled", True)))
        return _empty_state()

    shadow = _SHADOW
    if shadow is None:
        return _empty_state()

    try:
        enabled = bool(getattr(shadow, "enabled", False))
        payload = shadow.latest_scene_payload() if enabled else None
    except Exception:
        return _empty_state()

    if payload is None:
        empty = _empty_state()
        empty["enabled"] = enabled
        return empty
    return _wrap(payload, enabled=enabled)


def get_history(limit: int = 20) -> Dict[str, Any]:
    """Return the recent mental-world scene payloads.

    ``limit`` is clamped to ``[0, 500]`` to match the upstream ring
    capacity. When the P5 lane is disabled the returned ``scenes`` list
    is empty but authority flags are still pinned.
    """
    limit = max(0, min(500, int(limit)))

    if _HISTORY_OVERRIDE is not None:
        try:
            scenes = _HISTORY_OVERRIDE(limit) or []
        except Exception:
            scenes = []
        return _history_envelope(list(scenes)[:limit], enabled=True)

    shadow = _SHADOW
    if shadow is None:
        return _history_envelope([], enabled=False)

    try:
        enabled = bool(getattr(shadow, "enabled", False))
        if not enabled:
            return _history_envelope([], enabled=False)
        scenes = shadow.recent_scenes(limit)
    except Exception:
        return _history_envelope([], enabled=False)

    # Strip defensive vector keys on every scene.
    cleaned: List[Dict[str, Any]] = []
    for s in scenes:
        if not isinstance(s, dict):
            continue
        cleaned_scene = {
            k: v for k, v in s.items()
            if "vector" not in k.lower() and "ndarray" not in k.lower()
        }
        cleaned.append(cleaned_scene)
    return _history_envelope(cleaned, enabled=enabled)


def _history_envelope(scenes: List[Dict[str, Any]], *, enabled: bool) -> Dict[str, Any]:
    return {
        "status": "PRE-MATURE",
        "lane": "spatial_hrr_mental_world",
        "enabled": bool(enabled),
        "count": len(scenes),
        "scenes": scenes,
        **_runtime_provenance(),
        **AUTHORITY_FLAGS,
    }


def get_authority_flags() -> Dict[str, bool]:
    """Return the hard-pinned authority block for the mental-world lane."""
    return dict(AUTHORITY_FLAGS)


__all__ = [
    "AUTHORITY_FLAGS",
    "EMPTY_REASON",
    "register_shadow",
    "register_state_override",
    "register_history_override",
    "get_state",
    "get_history",
    "get_authority_flags",
]
