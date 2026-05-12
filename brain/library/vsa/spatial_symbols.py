"""HRR symbol vocabulary for the P5 mental-world lane.

Layered on top of the P4 substrate (``library.vsa.hrr`` + ``library.vsa.symbols``).
All symbols generated here use the **shared** :class:`HRRConfig` the engine
constructs for the world / simulation / recall shadows, so vectors produced
by this module are directly bindable with everything else in the HRR
substrate. No independent seed or dimension defaults are introduced.

The vocabulary is organized into four namespaces:

* ``spatial_relation`` — canonical :data:`cognition.spatial_schema.SpatialRelationType`
  plus derived mental-world relations (``facing``, ``moving_toward``,
  ``last_seen_near``, ...). Full canonical coverage including
  ``centered_in`` is asserted by
  :func:`brain.tests.test_hrr_spatial_symbols.test_full_canonical_relation_coverage`.
* ``spatial_axis`` — ``x`` / ``y`` / ``z`` role vectors for optional
  coordinate-triple binding during encoding.
* ``spatial_state`` — entity presence states (``visible``, ``occluded``,
  ``missing``, ``expected_in_view``, ``out_of_view``).
* ``mental_nav`` — symbolic actions used by the Commit 8 mental-navigation
  shadow (``turn_left``, ``turn_right``, ``move_forward``,
  ``object_occluded``, ``return_to_last_seen``). These never drive real-
  world motion.

**Non-negotiable**: this module never imports policy / belief / memory /
autonomy / identity writers. The structural scan in
``jarvis_eval.validation_pack._scan_hrr_forbidden_imports`` mechanically
enforces that guarantee across the P5 module roots.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from library.vsa.hrr import HRRConfig, make_symbol
from library.vsa.symbols import SymbolDictionary


# ---------------------------------------------------------------------------
# Namespace identifiers (stable; embedded into hashed symbol seeds).
# ---------------------------------------------------------------------------

NAMESPACE_RELATION = "spatial_relation"
NAMESPACE_AXIS = "spatial_axis"
NAMESPACE_STATE = "spatial_state"
NAMESPACE_NAV = "mental_nav"
NAMESPACE_ENTITY = "scene_entity"


# ---------------------------------------------------------------------------
# Relation vocabulary
# ---------------------------------------------------------------------------
# Canonical strings mirror cognition.spatial_schema.SpatialRelationType. The
# derived set is mental-world-only and never emitted as a canonical
# SpatialRelationFact.

# Canonical
LEFT_OF = "left_of"
RIGHT_OF = "right_of"
IN_FRONT_OF = "in_front_of"
BEHIND = "behind"
NEAR = "near"
ON = "on"
CENTERED_IN = "centered_in"

# Derived (mental-world only)
FACING = "facing"
MOVING_TOWARD = "moving_toward"
MOVING_AWAY = "moving_away"
OCCLUDED_BY = "occluded_by"
LAST_SEEN_NEAR = "last_seen_near"
EXPECTED_IN_VIEW = "expected_in_view"
OUT_OF_VIEW = "out_of_view"

CANONICAL_RELATION_SYMBOLS: Tuple[str, ...] = (
    LEFT_OF,
    RIGHT_OF,
    IN_FRONT_OF,
    BEHIND,
    NEAR,
    ON,
    CENTERED_IN,
)

DERIVED_RELATION_SYMBOLS: Tuple[str, ...] = (
    FACING,
    MOVING_TOWARD,
    MOVING_AWAY,
    OCCLUDED_BY,
    LAST_SEEN_NEAR,
    EXPECTED_IN_VIEW,
    OUT_OF_VIEW,
)

SPATIAL_RELATION_SYMBOLS: Tuple[str, ...] = (
    CANONICAL_RELATION_SYMBOLS + DERIVED_RELATION_SYMBOLS
)


# ---------------------------------------------------------------------------
# Axis, state, navigation vocabularies
# ---------------------------------------------------------------------------

SPATIAL_AXIS_SYMBOLS: Tuple[str, ...] = ("x", "y", "z")

# Mirrors perception.scene_types.SceneEntity.state values plus the mental-
# world extensions ``expected_in_view`` / ``out_of_view``.
SPATIAL_STATE_SYMBOLS: Tuple[str, ...] = (
    "visible",
    "occluded",
    "missing",
    "expected_in_view",
    "out_of_view",
)

# Five symbolic navigation actions. Purely symbolic: zero real-world authority.
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
MOVE_FORWARD = "move_forward"
OBJECT_OCCLUDED = "object_occluded"
RETURN_TO_LAST_SEEN = "return_to_last_seen"

MENTAL_NAV_SYMBOLS: Tuple[str, ...] = (
    TURN_LEFT,
    TURN_RIGHT,
    MOVE_FORWARD,
    OBJECT_OCCLUDED,
    RETURN_TO_LAST_SEEN,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def relation_string_to_symbol(rel: str, cfg: HRRConfig) -> np.ndarray:
    """Deterministic HRR vector for a spatial relation string.

    Uses the shared P4 :class:`HRRConfig` so vectors are comparable across
    the world / simulation / recall / scene shadows. Accepts any string in
    :data:`SPATIAL_RELATION_SYMBOLS`. Unknown strings are still hashable
    into a distinct vector (so the encoder never crashes on a canonical
    schema extension), but the test pack asserts the expected vocabulary
    is covered.

    The symbol seeding path is identical to the one taken by
    :class:`SymbolDictionary.get` when called with
    ``(NAMESPACE_RELATION, rel)``, so this helper and the dictionary
    lookup return the same vector for the same config.
    """
    if not isinstance(rel, str) or not rel:
        raise ValueError("relation_string_to_symbol: rel must be a non-empty string")
    return make_symbol(f"{NAMESPACE_RELATION}:{rel}", cfg)


def seed_all_symbols(symbols: SymbolDictionary) -> None:
    """Pre-populate ``symbols`` with the full P5 vocabulary.

    Idempotent: :class:`SymbolDictionary` returns cached hits on repeat
    calls. Useful for encoders that want deterministic ordering via
    :meth:`SymbolDictionary.known` and a warm cache before the first
    encode.
    """
    for name in SPATIAL_RELATION_SYMBOLS:
        symbols.get(NAMESPACE_RELATION, name)
    for name in SPATIAL_AXIS_SYMBOLS:
        symbols.get(NAMESPACE_AXIS, name)
    for name in SPATIAL_STATE_SYMBOLS:
        symbols.get(NAMESPACE_STATE, name)
    for name in MENTAL_NAV_SYMBOLS:
        symbols.get(NAMESPACE_NAV, name)


def entity_symbol(entity_id: str, symbols: SymbolDictionary) -> np.ndarray:
    """Return the cached HRR vector for a canonical scene-entity id."""
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("entity_symbol: entity_id must be a non-empty string")
    return symbols.get(NAMESPACE_ENTITY, entity_id)


def relation_symbol(rel: str, symbols: SymbolDictionary) -> np.ndarray:
    """Return the cached HRR vector for a relation string."""
    if not isinstance(rel, str) or not rel:
        raise ValueError("relation_symbol: rel must be a non-empty string")
    return symbols.get(NAMESPACE_RELATION, rel)


def state_symbol(state: str, symbols: SymbolDictionary) -> np.ndarray:
    """Return the cached HRR vector for an entity-state label."""
    if not isinstance(state, str) or not state:
        raise ValueError("state_symbol: state must be a non-empty string")
    return symbols.get(NAMESPACE_STATE, state)


def axis_symbol(axis: str, symbols: SymbolDictionary) -> np.ndarray:
    """Return the cached HRR vector for a coordinate axis label (``x``/``y``/``z``)."""
    if axis not in SPATIAL_AXIS_SYMBOLS:
        raise ValueError(f"axis_symbol: axis must be one of {SPATIAL_AXIS_SYMBOLS}, got {axis!r}")
    return symbols.get(NAMESPACE_AXIS, axis)


def nav_symbol(action: str, symbols: SymbolDictionary) -> np.ndarray:
    """Return the cached HRR vector for a mental-navigation action label."""
    if action not in MENTAL_NAV_SYMBOLS:
        raise ValueError(
            f"nav_symbol: action must be one of {MENTAL_NAV_SYMBOLS}, got {action!r}"
        )
    return symbols.get(NAMESPACE_NAV, action)


__all__ = [
    "NAMESPACE_RELATION",
    "NAMESPACE_AXIS",
    "NAMESPACE_STATE",
    "NAMESPACE_NAV",
    "NAMESPACE_ENTITY",
    "LEFT_OF",
    "RIGHT_OF",
    "IN_FRONT_OF",
    "BEHIND",
    "NEAR",
    "ON",
    "CENTERED_IN",
    "FACING",
    "MOVING_TOWARD",
    "MOVING_AWAY",
    "OCCLUDED_BY",
    "LAST_SEEN_NEAR",
    "EXPECTED_IN_VIEW",
    "OUT_OF_VIEW",
    "CANONICAL_RELATION_SYMBOLS",
    "DERIVED_RELATION_SYMBOLS",
    "SPATIAL_RELATION_SYMBOLS",
    "SPATIAL_AXIS_SYMBOLS",
    "SPATIAL_STATE_SYMBOLS",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_FORWARD",
    "OBJECT_OCCLUDED",
    "RETURN_TO_LAST_SEEN",
    "MENTAL_NAV_SYMBOLS",
    "relation_string_to_symbol",
    "seed_all_symbols",
    "entity_symbol",
    "relation_symbol",
    "state_symbol",
    "axis_symbol",
    "nav_symbol",
]
