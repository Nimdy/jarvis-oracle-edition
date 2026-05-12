"""Canonical world-model schema — the cross-domain substrate.

Seven layers: observation, entity, relation, norm, prediction, outcome,
advisory — plus episodes for bounded event sequences.  These frozen
dataclasses are the durable vocabulary for football, fishing, shop safety,
camera framing, GPS/LiDAR, and any future Pi5 sensor plugins.

The legacy WorldState/WorldDelta/CausalPrediction path remains alive in
parallel; these types project alongside it without replacing it.

See docs/oracle_world_model_spec_v1.md for the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Confidence = float
Timestamp = float

# ---------------------------------------------------------------------------
# Typed literals
# ---------------------------------------------------------------------------

ObservationKind = Literal[
    "vision",
    "audio",
    "identity",
    "presence",
    "attention",
    "system",
    "memory",
    "gps",
    "lidar",
    "imu",
    "sonar",
    "obd2",
    "weather",
    "custom",
]

EntityKind = Literal[
    "person",
    "object",
    "tool",
    "device",
    "vehicle",
    "animal",
    "structure",
    "surface",
    "container",
    "zone",
    "display",
    "ball",
    "player",
    "field_region",
    "water_spot",
    "boat",
    "sensor_anchor",
    "environmental_feature",
    "unknown",
]

RelationKind = Literal[
    "near",
    "inside",
    "on",
    "under",
    "held_by",
    "holding",
    "looking_at",
    "watching",
    "facing",
    "belongs_in",
    "left_on",
    "located_at",
    "attached_to",
    "connected_to",
    "contains",
    "visited",
    "similar_to",
    "hazardous_to",
    "grouped_with",
    "supports",
    "blocks",
    "blocking",
    "moving_toward",
    "assigned_to",
    "part_of",
    "unknown",
]

NormKind = Literal[
    "safety",
    "workflow",
    "preference",
    "task",
    "housekeeping",
    "navigation",
    "domain_strategy",
]

PredictionTargetKind = Literal[
    "entity_state",
    "relation_state",
    "norm_state",
    "facet_projection",
    "advisory_trigger",
]

OutcomeVerdict = Literal[
    "hit",
    "miss",
    "partial",
    "inconclusive",
    "expired",
    "interrupted",
]

AdvisorySeverity = Literal[
    "info",
    "low",
    "medium",
    "high",
    "critical",
]

EpisodeKind = Literal[
    "conversation",
    "room_session",
    "workshop_session",
    "fishing_trip",
    "football_drive",
    "camera_session",
    "generic",
]

ZoneKind = Literal[
    "work_area",
    "storage_area",
    "walkway",
    "hazard_zone",
    "sterile_zone",
    "public_zone",
    "private_zone",
    "water_edge",
    "casting_zone",
    "sideline",
    "field_of_play",
    "bench_area",
    "control_area",
    "vehicle_path",
    "unknown",
]


# ---------------------------------------------------------------------------
# Layer 0 — Sensor observations (normalised input from any source)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorObservation:
    """Raw or normalised perceptual/event input.

    Every sensor — camera, mic, GPS, LiDAR, sonar, OBD-II, API, manual —
    normalises into this contract before fusion.
    """
    observation_id: str
    kind: ObservationKind
    source: str
    ts: Timestamp
    confidence: Confidence
    freshness_s: float = 0.0
    subject_id: str | None = None
    region_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    provenance: str = "live"


# ---------------------------------------------------------------------------
# Layer 1 — Entities (persistent objects / actors / places)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldEntity:
    """Anything that can persist across time: person, wrench, zone, monitor,
    fishing spot, football player, vehicle, camera anchor."""
    entity_id: str
    kind: EntityKind
    label: str
    ts: Timestamp
    confidence: Confidence
    first_seen_ts: Timestamp | None = None
    last_seen_ts: Timestamp | None = None
    stable: bool = False
    mobile: bool = False
    properties: dict[str, Any] = field(default_factory=dict)
    region_id: str | None = None
    source_observation_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Layer 2 — Relations (typed edges between entities)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldRelation:
    """Typed link: near, belongs_in, left_on, hazardous_to, part_of, etc."""
    relation_id: str
    kind: RelationKind
    subject_id: str
    object_id: str
    ts: Timestamp
    confidence: Confidence
    properties: dict[str, Any] = field(default_factory=dict)
    source_observation_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Layer 3 — Norms / affordances (what *should* be true)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldNorm:
    """Expected state or policy: 'wrenches belong in tool chest', 'no bites
    after 20 min → reposition', 'speaker should stay in frame'."""
    norm_id: str
    kind: NormKind
    label: str
    description: str
    scope: str
    priority: int
    expected_state: dict[str, Any] = field(default_factory=dict)
    trigger_conditions: dict[str, Any] = field(default_factory=dict)
    remediation_hint: str = ""
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Layer 4 — Predictions (testable future claims)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionTarget:
    """What the prediction is about — entity state, relation, norm, or facet."""
    kind: PredictionTargetKind
    target_id: str
    field: str
    expected_value: Any = None


@dataclass(frozen=True)
class WorldPrediction:
    """Explicit, testable claim about the near future."""
    prediction_id: str
    rule_id: str
    label: str
    target: PredictionTarget
    created_at: Timestamp
    horizon_s: float
    expires_at: Timestamp
    confidence: Confidence
    basis_entity_ids: tuple[str, ...] = ()
    basis_relation_ids: tuple[str, ...] = ()
    basis_norm_ids: tuple[str, ...] = ()
    legacy_projection: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Layer 5 — Outcomes (prediction validation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldOutcome:
    """Post-hoc validation of a prediction."""
    outcome_id: str
    prediction_id: str
    verdict: OutcomeVerdict
    validated_at: Timestamp
    score: float = 0.0
    actual_value: Any = None
    detail: str = ""
    evidence_observation_ids: tuple[str, ...] = ()
    evidence_entity_ids: tuple[str, ...] = ()
    evidence_relation_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Layer 6 — Advisories (user-facing / planner-facing recommendations)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldAdvisory:
    """Grounded recommendation derived from predictions + norms + confidence."""
    advisory_id: str
    label: str
    severity: AdvisorySeverity
    ts: Timestamp
    confidence: Confidence
    summary: str
    rationale: str = ""
    recommended_action: str = ""
    entity_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    norm_ids: tuple[str, ...] = ()
    prediction_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Episodes (bounded event sequences)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldEpisode:
    """Grouped sequence: one football play, one fishing stop, one shop
    cleanup incident, one camera framing session, one conversation segment."""
    episode_id: str
    kind: EpisodeKind
    label: str
    start_ts: Timestamp
    end_ts: Timestamp | None = None
    entity_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    prediction_ids: tuple[str, ...] = ()
    outcome_ids: tuple[str, ...] = ()
    advisory_ids: tuple[str, ...] = ()
    summary: str = ""
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Spatial position (frame-agnostic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldPosition:
    """Frame-agnostic spatial reference: camera bbox, room coords, GPS, screen."""
    frame: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    bbox: tuple[int, int, int, int] | None = None
    region_label: str | None = None


# ---------------------------------------------------------------------------
# Zones (named regions with norms)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldZone:
    """Named region in the environment with associated norms."""
    zone_id: str
    label: str
    kind: ZoneKind
    region_label: str | None = None
    confidence: Confidence = 1.0
    norms: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Archetype packs (environment priors)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchetypePack:
    """Frozen configuration for an environment archetype.

    Provides entity/zone/relation priors, norm templates, and hazard
    templates for a broad environment class (workspace, workshop, outdoor,
    medical, sports field).  Domain packs specialise these further.
    """
    archetype_id: str
    label: str
    parent_archetypes: tuple[str, ...] = ()
    entity_priors: tuple[str, ...] = ()
    zone_priors: tuple[str, ...] = ()
    relation_priors: tuple[str, ...] = ()
    norm_templates: tuple[dict[str, Any], ...] = ()
    hazard_templates: tuple[dict[str, Any], ...] = ()
    prediction_rules: tuple[str, ...] = ()


class ArchetypeRegistry:
    """Registry for environment archetype packs."""

    def __init__(self) -> None:
        self._packs: dict[str, ArchetypePack] = {}

    def register(self, pack: ArchetypePack) -> None:
        self._packs[pack.archetype_id] = pack

    def get(self, archetype_id: str) -> ArchetypePack | None:
        return self._packs.get(archetype_id)

    def all_packs(self) -> list[ArchetypePack]:
        return list(self._packs.values())

    def match(self, entity_labels: set[str]) -> list[str]:
        """Return archetype IDs whose entity_priors overlap observed labels."""
        matched: list[str] = []
        for pack in self._packs.values():
            overlap = entity_labels & {p.lower() for p in pack.entity_priors}
            if len(overlap) >= 2:
                matched.append(pack.archetype_id)
        return sorted(matched)


# ---------------------------------------------------------------------------
# Canonical world state (full per-tick snapshot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalWorldState:
    """Full per-tick canonical world snapshot — the cross-domain substrate."""
    timestamp: Timestamp
    archetypes_active: tuple[str, ...] = ()
    observations: tuple[SensorObservation, ...] = ()
    entities: tuple[WorldEntity, ...] = ()
    relations: tuple[WorldRelation, ...] = ()
    zones: tuple[WorldZone, ...] = ()
    norms: tuple[WorldNorm, ...] = ()
    predictions: tuple[WorldPrediction, ...] = ()
    advisories: tuple[WorldAdvisory, ...] = ()
