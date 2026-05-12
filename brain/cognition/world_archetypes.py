"""Environment archetype packs — prior knowledge for broad environment classes.

Each pack provides entity/zone/relation priors, norm templates, and hazard
templates.  The ``default_registry`` singleton is populated at import time
with the five base archetypes.  Domain packs (football, fishing, etc.) will
extend these later.

Only ``indoor_workspace`` is behaviourally active in the first slice.
The others are registered data — available for matching but not yet
driving runtime norm evaluation.
"""

from __future__ import annotations

from cognition.world_schema import ArchetypePack, ArchetypeRegistry

# ---------------------------------------------------------------------------
# Base archetype packs
# ---------------------------------------------------------------------------

INDOOR_WORKSPACE = ArchetypePack(
    archetype_id="indoor_workspace",
    label="Indoor Workspace",
    entity_priors=(
        "person", "display", "keyboard", "mouse", "camera",
        "microphone", "desk", "chair", "monitor", "laptop",
    ),
    zone_priors=("work_area", "storage_area", "walkway", "private_zone"),
    relation_priors=("on", "near", "facing", "connected_to", "blocking"),
    norm_templates=(
        {"label": "work_surface_should_remain_usable", "kind": "workflow", "priority": 6},
        {"label": "display_layout_is_usually_stable", "kind": "workflow", "priority": 4},
        {"label": "walkway_should_remain_clear", "kind": "safety", "priority": 8},
    ),
    hazard_templates=(
        {"label": "trip_obstruction", "priority": 8},
        {"label": "cable_obstruction", "priority": 5},
    ),
    prediction_rules=(
        "present_user_stays",
        "passive_mode_persists",
        "display_layout_stable",
    ),
)

INDUSTRIAL_WORKSHOP = ArchetypePack(
    archetype_id="industrial_workshop",
    label="Industrial Workshop",
    entity_priors=(
        "person", "tool", "wrench", "hammer", "drill",
        "bench", "machine", "storage", "floor_zone", "vehicle",
    ),
    zone_priors=("work_area", "storage_area", "walkway", "hazard_zone"),
    relation_priors=("on", "near", "belongs_in", "left_on", "hazardous_to", "blocking"),
    norm_templates=(
        {"label": "tools_should_be_stowed_when_inactive", "kind": "safety", "priority": 9},
        {"label": "walkways_should_remain_clear", "kind": "safety", "priority": 9},
        {"label": "ppe_zones_respected", "kind": "safety", "priority": 8},
    ),
    hazard_templates=(
        {"label": "loose_tool_on_floor", "priority": 9},
        {"label": "machine_proximity_risk", "priority": 8},
        {"label": "heat_source_nearby", "priority": 7},
    ),
    prediction_rules=(
        "tool_left_on_floor_persists",
        "walkway_obstruction_persists",
    ),
)

OUTDOOR_NATURE = ArchetypePack(
    archetype_id="outdoor_nature",
    label="Outdoor Nature",
    entity_priors=(
        "person", "water_spot", "shoreline", "dock", "boat",
        "tree", "rock", "trail", "weather_front",
    ),
    zone_priors=("water_edge", "casting_zone", "walkway", "hazard_zone"),
    relation_priors=("near", "located_at", "visited", "facing", "moving_toward"),
    norm_templates=(
        {"label": "footing_safety_near_water", "kind": "safety", "priority": 9},
        {"label": "path_awareness", "kind": "navigation", "priority": 6},
        {"label": "location_continuity", "kind": "preference", "priority": 5},
    ),
    hazard_templates=(
        {"label": "slippery_edge", "priority": 8},
        {"label": "sudden_weather", "priority": 7},
    ),
    prediction_rules=(
        "angler_position_stable",
        "cast_zone_productivity_persists",
    ),
)

MEDICAL_CLINICAL = ArchetypePack(
    archetype_id="medical_clinical",
    label="Medical Clinical",
    entity_priors=(
        "person", "bed", "workstation", "curtain", "monitor",
        "tray", "sink", "instrument", "staff",
    ),
    zone_priors=("work_area", "sterile_zone", "private_zone", "walkway"),
    relation_priors=("near", "inside", "on", "assigned_to", "blocking"),
    norm_templates=(
        {"label": "clear_access_to_patient", "kind": "safety", "priority": 9},
        {"label": "sterile_field_uncontaminated", "kind": "safety", "priority": 10},
        {"label": "equipment_readiness", "kind": "workflow", "priority": 7},
    ),
    hazard_templates=(
        {"label": "blocked_access", "priority": 9},
        {"label": "sterile_field_breach", "priority": 10},
    ),
    prediction_rules=(),
)

SPORTS_FIELD = ArchetypePack(
    archetype_id="sports_field",
    label="Sports Field",
    entity_priors=(
        "person", "player", "ball", "field_region",
        "formation", "sideline", "referee",
    ),
    zone_priors=("field_of_play", "sideline", "bench_area"),
    relation_priors=("near", "facing", "grouped_with", "moving_toward", "blocking"),
    norm_templates=(
        {"label": "formation_shift_precedes_play_start", "kind": "domain_strategy", "priority": 8},
    ),
    hazard_templates=(),
    prediction_rules=(
        "formation_persists_until_snap",
        "motion_precedes_play_direction",
    ),
)

# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

default_registry = ArchetypeRegistry()
default_registry.register(INDOOR_WORKSPACE)
default_registry.register(INDUSTRIAL_WORKSHOP)
default_registry.register(OUTDOOR_NATURE)
default_registry.register(MEDICAL_CLINICAL)
default_registry.register(SPORTS_FIELD)
