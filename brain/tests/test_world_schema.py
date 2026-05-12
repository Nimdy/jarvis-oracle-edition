"""Tests for the canonical world-model schema and adapter layer.

Validates:
- Canonical dataclass construction and immutability
- Entity projection from scene state
- Relation projection from scene state
- Display surface entity projection
- Tool-on-floor representation
- Legacy WorldModel.update() still works after integration
- get_canonical_state() returns populated data
"""

import time
from dataclasses import FrozenInstanceError

import pytest

from cognition.world_schema import (
    SensorObservation,
    WorldEntity,
    WorldRelation,
    WorldNorm,
    WorldPrediction,
    WorldOutcome,
    WorldAdvisory,
    WorldEpisode,
    PredictionTarget,
    WorldPosition,
    WorldZone,
    ArchetypePack,
    ArchetypeRegistry,
    CanonicalWorldState,
)
from cognition.world_adapters import (
    observations_from_scene_state,
    observations_from_attention,
    observations_from_presence,
    observations_from_identity,
    entities_from_scene_state,
    relations_from_scene_state,
    zones_from_scene_state,
    CanonicalWorldProjector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scene_state_with_person():
    return {
        "entity_count": 2,
        "visible_count": 2,
        "stable_count": 1,
        "entities": [
            {
                "entity_id": "person_1",
                "label": "person",
                "confidence": 0.92,
                "region": "desk_left",
                "stable_cycles": 5,
                "first_seen_ts": 1000.0,
                "last_seen_ts": 1100.0,
            },
            {
                "entity_id": "monitor_1",
                "label": "monitor",
                "confidence": 0.88,
                "region": "monitor_zone",
                "stable_cycles": 10,
                "first_seen_ts": 900.0,
                "last_seen_ts": 1100.0,
                "is_display_surface": True,
            },
        ],
        "display_surfaces": [
            {
                "surface_id": "display_0",
                "label": "monitor",
                "confidence": 0.9,
                "region": "monitor_zone",
            },
        ],
        "display_content": [],
        "region_visibility": {"desk_left": 0.8, "monitor_zone": 1.0},
    }


@pytest.fixture
def scene_state_tool_on_floor():
    return {
        "entity_count": 2,
        "visible_count": 2,
        "stable_count": 2,
        "entities": [
            {
                "entity_id": "wrench_1",
                "label": "wrench",
                "confidence": 0.85,
                "region": "floor_zone",
                "stable_cycles": 3,
            },
            {
                "entity_id": "person_2",
                "label": "person",
                "confidence": 0.9,
                "region": "desk_left",
                "stable_cycles": 8,
            },
        ],
        "display_surfaces": [],
        "display_content": [],
        "region_visibility": {"floor_zone": 0.7, "desk_left": 0.9},
    }


# ---------------------------------------------------------------------------
# Schema construction tests
# ---------------------------------------------------------------------------

class TestSchemaConstruction:
    def test_sensor_observation_immutable(self):
        obs = SensorObservation(
            observation_id="test:1",
            kind="vision",
            source="test",
            ts=1000.0,
            confidence=0.9,
        )
        assert obs.observation_id == "test:1"
        with pytest.raises(FrozenInstanceError):
            obs.confidence = 0.5  # type: ignore[misc]

    def test_world_entity_defaults(self):
        ent = WorldEntity(
            entity_id="e1",
            kind="person",
            label="person",
            ts=1000.0,
            confidence=0.8,
        )
        assert ent.properties == {}
        assert ent.tags == ()
        assert ent.stable is False
        assert ent.mobile is False

    def test_world_relation_construction(self):
        rel = WorldRelation(
            relation_id="r1",
            kind="near",
            subject_id="e1",
            object_id="e2",
            ts=1000.0,
            confidence=0.7,
        )
        assert rel.subject_id == "e1"
        assert rel.object_id == "e2"

    def test_world_norm_construction(self):
        norm = WorldNorm(
            norm_id="n1",
            kind="safety",
            label="tool_not_on_floor",
            description="Tools should not be left on the floor",
            scope="workshop",
            priority=80,
            expected_state={"tool_location": "tool_chest"},
        )
        assert norm.priority == 80

    def test_world_prediction_construction(self):
        target = PredictionTarget(
            kind="entity_state",
            target_id="wrench_1",
            field="region",
            expected_value="tool_chest",
        )
        pred = WorldPrediction(
            prediction_id="p1",
            rule_id="tool_return",
            label="wrench_returned",
            target=target,
            created_at=1000.0,
            horizon_s=30.0,
            expires_at=1035.0,
            confidence=0.6,
        )
        assert pred.target.kind == "entity_state"

    def test_world_outcome_construction(self):
        outcome = WorldOutcome(
            outcome_id="o1",
            prediction_id="p1",
            verdict="hit",
            validated_at=1030.0,
            score=1.0,
        )
        assert outcome.verdict == "hit"

    def test_world_advisory_construction(self):
        adv = WorldAdvisory(
            advisory_id="a1",
            label="return_wrench",
            severity="medium",
            ts=1000.0,
            confidence=0.7,
            summary="Please return the wrench to the tool chest.",
        )
        assert adv.severity == "medium"

    def test_world_episode_construction(self):
        ep = WorldEpisode(
            episode_id="ep1",
            kind="conversation",
            label="morning_chat",
            start_ts=1000.0,
            end_ts=1300.0,
        )
        assert ep.end_ts == 1300.0


# ---------------------------------------------------------------------------
# Observation adapter tests
# ---------------------------------------------------------------------------

class TestObservationAdapters:
    def test_observations_from_scene_state(self, scene_state_with_person):
        obs = observations_from_scene_state(scene_state_with_person, now=1100.0)
        assert len(obs) == 3  # 2 entities + 1 display surface
        assert all(isinstance(o, SensorObservation) for o in obs)
        entity_obs = [o for o in obs if "scene_entity" in o.tags]
        assert len(entity_obs) == 2

    def test_observations_from_attention(self):
        attn = {"presence_confidence": 0.8, "engagement_level": 0.6}
        obs = observations_from_attention(attn, now=1000.0)
        assert len(obs) == 1
        assert obs[0].kind == "attention"
        assert obs[0].confidence == 0.8

    def test_observations_from_empty_attention(self):
        assert observations_from_attention({}, now=1000.0) == []

    def test_observations_from_presence(self):
        pres = {"is_present": True, "confidence": 0.95}
        obs = observations_from_presence(pres, now=1000.0)
        assert len(obs) == 1
        assert obs[0].kind == "presence"

    def test_observations_from_identity(self):
        speaker = {"name": "Alice", "confidence": 0.8}
        obs = observations_from_identity(speaker, now=1000.0)
        assert len(obs) == 1
        assert obs[0].subject_id == "Alice"


# ---------------------------------------------------------------------------
# Entity adapter tests
# ---------------------------------------------------------------------------

class TestEntityAdapters:
    def test_person_entity(self, scene_state_with_person):
        ents = entities_from_scene_state(scene_state_with_person, now=1100.0)
        people = [e for e in ents if e.kind == "person"]
        assert len(people) == 1
        assert people[0].mobile is True
        assert people[0].stable is True  # stable_cycles >= 3

    def test_display_entity(self, scene_state_with_person):
        ents = entities_from_scene_state(scene_state_with_person, now=1100.0)
        displays = [e for e in ents if e.kind == "display"]
        assert len(displays) >= 1
        display = displays[0]
        assert display.stable is True

    def test_display_surface_projection(self, scene_state_with_person):
        ents = entities_from_scene_state(scene_state_with_person, now=1100.0)
        surface_ents = [e for e in ents if "display_surface" in e.tags]
        assert len(surface_ents) == 1
        assert surface_ents[0].entity_id == "display_0"

    def test_tool_on_floor_representation(self, scene_state_tool_on_floor):
        ents = entities_from_scene_state(scene_state_tool_on_floor, now=1100.0)
        tools = [e for e in ents if e.kind == "tool"]
        assert len(tools) == 1
        assert tools[0].label == "wrench"
        assert tools[0].region_id == "floor_zone"

    def test_entity_count_matches_scene(self, scene_state_with_person):
        ents = entities_from_scene_state(scene_state_with_person, now=1100.0)
        # 2 entities + 1 display surface = 3
        assert len(ents) == 3


# ---------------------------------------------------------------------------
# Relation adapter tests
# ---------------------------------------------------------------------------

class TestRelationAdapters:
    def test_region_relations(self, scene_state_with_person):
        rels = relations_from_scene_state(scene_state_with_person, now=1100.0)
        assert len(rels) == 2  # both entities have regions
        assert all(r.kind == "located_at" for r in rels)

    def test_tool_floor_relation(self, scene_state_tool_on_floor):
        rels = relations_from_scene_state(scene_state_tool_on_floor, now=1100.0)
        wrench_rels = [r for r in rels if r.subject_id == "wrench_1"]
        assert len(wrench_rels) == 1
        assert wrench_rels[0].object_id == "region:floor_zone"

    def test_no_relations_without_region(self):
        scene = {
            "entities": [{"label": "cup", "confidence": 0.5}],
            "display_surfaces": [],
        }
        rels = relations_from_scene_state(scene, now=1000.0)
        assert len(rels) == 0


# ---------------------------------------------------------------------------
# WorldModel integration (legacy compatibility)
# ---------------------------------------------------------------------------

class TestWorldModelIntegration:
    def test_update_still_returns_world_state(self):
        from cognition.world_model import WorldModel
        from cognition.world_state import WorldState
        wm = WorldModel()
        result = wm.update()
        assert isinstance(result, WorldState)

    def test_get_state_shape_unchanged(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        wm.update()
        state = wm.get_state()
        assert "physical" in state
        assert "user" in state
        assert "conversation" in state
        assert "system" in state
        assert "causal" in state
        assert "predictions" in state
        assert "promotion" in state

    def test_get_canonical_state_returns_data(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        wm.update()
        canonical = wm.get_canonical_state()
        assert "entities" in canonical
        assert "relations" in canonical
        assert "zones" in canonical
        assert "archetypes_active" in canonical
        assert "observation_count" in canonical
        assert isinstance(canonical["entity_count"], int)
        assert isinstance(canonical["relation_count"], int)
        assert isinstance(canonical["zone_count"], int)

    def test_canonical_populated_with_scene_tracker(self):
        """When scene_tracker provides data, canonical entities/zones/archetypes should appear."""
        from cognition.world_model import WorldModel

        class FakeSceneTracker:
            def get_state(self):
                return {
                    "entity_count": 1,
                    "visible_count": 1,
                    "stable_count": 1,
                    "entities": [
                        {
                            "entity_id": "person_test",
                            "label": "person",
                            "confidence": 0.9,
                            "region": "desk_center",
                            "stable_cycles": 5,
                        }
                    ],
                    "display_surfaces": [],
                    "display_content": [],
                    "region_visibility": {"desk_center": 1.0, "desk_left": 0.8},
                    "person_count": 1,
                }

        wm = WorldModel(scene_tracker=FakeSceneTracker())
        wm.update()
        canonical = wm.get_canonical_state()
        assert canonical["entity_count"] == 1
        assert canonical["relation_count"] == 1
        assert canonical["zone_count"] == 2
        assert canonical["entities"][0]["kind"] == "person"
        assert canonical["entities"][0]["label"] == "person"
        assert canonical["relations"][0]["kind"] == "located_at"
        zone_kinds = {z["kind"] for z in canonical["zones"]}
        assert "work_area" in zone_kinds


# ---------------------------------------------------------------------------
# WorldPosition tests
# ---------------------------------------------------------------------------

class TestWorldPosition:
    def test_construction_minimal(self):
        pos = WorldPosition(frame="camera")
        assert pos.frame == "camera"
        assert pos.x is None
        assert pos.bbox is None

    def test_construction_full(self):
        pos = WorldPosition(
            frame="room", x=1.5, y=2.0, z=0.0,
            bbox=(10, 20, 100, 200), region_label="desk_left",
        )
        assert pos.bbox == (10, 20, 100, 200)
        assert pos.region_label == "desk_left"

    def test_immutable(self):
        pos = WorldPosition(frame="camera")
        with pytest.raises(FrozenInstanceError):
            pos.frame = "gps"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# WorldZone tests
# ---------------------------------------------------------------------------

class TestWorldZone:
    def test_construction(self):
        z = WorldZone(
            zone_id="zone:desk_center",
            label="desk center",
            kind="work_area",
            confidence=0.95,
        )
        assert z.kind == "work_area"
        assert z.norms == ()

    def test_with_norms(self):
        z = WorldZone(
            zone_id="zone:sterile_1",
            label="sterile field",
            kind="sterile_zone",
            norms=("sterile_field_uncontaminated",),
        )
        assert len(z.norms) == 1

    def test_immutable(self):
        z = WorldZone(zone_id="z1", label="test", kind="unknown")
        with pytest.raises(FrozenInstanceError):
            z.label = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ArchetypePack tests
# ---------------------------------------------------------------------------

class TestArchetypePack:
    def test_construction(self):
        pack = ArchetypePack(
            archetype_id="test_pack",
            label="Test Pack",
            entity_priors=("person", "tool"),
        )
        assert pack.archetype_id == "test_pack"
        assert len(pack.entity_priors) == 2
        assert pack.parent_archetypes == ()

    def test_with_parents(self):
        pack = ArchetypePack(
            archetype_id="fishing",
            label="Fishing",
            parent_archetypes=("outdoor_nature",),
            entity_priors=("rod", "boat"),
        )
        assert pack.parent_archetypes == ("outdoor_nature",)

    def test_immutable(self):
        pack = ArchetypePack(archetype_id="x", label="X")
        with pytest.raises(FrozenInstanceError):
            pack.label = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ArchetypeRegistry tests
# ---------------------------------------------------------------------------

class TestArchetypeRegistry:
    def test_register_and_get(self):
        reg = ArchetypeRegistry()
        pack = ArchetypePack(archetype_id="test", label="Test")
        reg.register(pack)
        assert reg.get("test") is pack
        assert reg.get("nonexistent") is None

    def test_all_packs(self):
        reg = ArchetypeRegistry()
        reg.register(ArchetypePack(archetype_id="a", label="A"))
        reg.register(ArchetypePack(archetype_id="b", label="B"))
        assert len(reg.all_packs()) == 2

    def test_match_requires_overlap_of_2(self):
        reg = ArchetypeRegistry()
        reg.register(ArchetypePack(
            archetype_id="workspace",
            label="Workspace",
            entity_priors=("person", "display", "keyboard", "mouse"),
        ))
        assert reg.match({"person", "display", "cup"}) == ["workspace"]
        assert reg.match({"person"}) == []  # only 1 overlap

    def test_match_multiple_archetypes(self):
        reg = ArchetypeRegistry()
        reg.register(ArchetypePack(
            archetype_id="a",
            label="A",
            entity_priors=("person", "tool"),
        ))
        reg.register(ArchetypePack(
            archetype_id="b",
            label="B",
            entity_priors=("person", "display"),
        ))
        result = reg.match({"person", "tool", "display"})
        assert "a" in result
        assert "b" in result

    def test_match_case_insensitive(self):
        reg = ArchetypeRegistry()
        reg.register(ArchetypePack(
            archetype_id="ws",
            label="WS",
            entity_priors=("Person", "Display"),
        ))
        assert reg.match({"person", "display"}) == ["ws"]

    def test_default_registry_loaded(self):
        from cognition.world_archetypes import default_registry
        assert len(default_registry.all_packs()) == 5
        assert default_registry.get("indoor_workspace") is not None
        assert default_registry.get("sports_field") is not None


# ---------------------------------------------------------------------------
# CanonicalWorldState tests
# ---------------------------------------------------------------------------

class TestCanonicalWorldState:
    def test_empty_state(self):
        cs = CanonicalWorldState(timestamp=1000.0)
        assert cs.archetypes_active == ()
        assert cs.observations == ()
        assert cs.entities == ()
        assert cs.zones == ()

    def test_populated_state(self):
        ent = WorldEntity(
            entity_id="e1", kind="person", label="person",
            ts=1000.0, confidence=0.9,
        )
        zone = WorldZone(
            zone_id="z1", label="desk", kind="work_area",
        )
        cs = CanonicalWorldState(
            timestamp=1000.0,
            archetypes_active=("indoor_workspace",),
            entities=(ent,),
            zones=(zone,),
        )
        assert len(cs.entities) == 1
        assert len(cs.zones) == 1
        assert cs.archetypes_active == ("indoor_workspace",)

    def test_immutable(self):
        cs = CanonicalWorldState(timestamp=1000.0)
        with pytest.raises(FrozenInstanceError):
            cs.timestamp = 2000.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Zone adapter tests
# ---------------------------------------------------------------------------

class TestZoneAdapters:
    def test_zones_from_region_visibility(self, scene_state_with_person):
        zones = zones_from_scene_state(scene_state_with_person, now=1100.0)
        assert len(zones) == 2
        labels = {z.label for z in zones}
        assert "desk left" in labels
        assert "monitor zone" in labels

    def test_zone_kind_mapping(self, scene_state_with_person):
        zones = zones_from_scene_state(scene_state_with_person, now=1100.0)
        by_label = {z.zone_id: z for z in zones}
        assert by_label["zone:desk_left"].kind == "work_area"
        assert by_label["zone:monitor_zone"].kind == "work_area"

    def test_zone_confidence_from_visibility(self, scene_state_with_person):
        zones = zones_from_scene_state(scene_state_with_person, now=1100.0)
        desk = [z for z in zones if z.zone_id == "zone:desk_left"][0]
        assert desk.confidence == 0.8

    def test_zones_from_tool_floor(self, scene_state_tool_on_floor):
        zones = zones_from_scene_state(scene_state_tool_on_floor, now=1100.0)
        floor = [z for z in zones if z.zone_id == "zone:floor_zone"][0]
        assert floor.kind == "walkway"

    def test_unknown_region_maps_to_unknown(self):
        scene = {
            "entities": [],
            "display_surfaces": [],
            "region_visibility": {"strange_region": 0.5},
        }
        zones = zones_from_scene_state(scene, now=1000.0)
        assert zones[0].kind == "unknown"

    def test_empty_region_visibility(self):
        scene = {"entities": [], "display_surfaces": [], "region_visibility": {}}
        assert zones_from_scene_state(scene, now=1000.0) == []


# ---------------------------------------------------------------------------
# CanonicalWorldProjector tests
# ---------------------------------------------------------------------------

class TestCanonicalWorldProjector:
    def test_build_empty_inputs(self):
        projector = CanonicalWorldProjector()
        cs = projector.build(
            now=1000.0,
            scene={},
            attention={},
            presence={},
            identity={},
        )
        assert isinstance(cs, CanonicalWorldState)
        assert cs.timestamp == 1000.0
        assert cs.entities == ()
        assert cs.zones == ()
        assert cs.archetypes_active == ()

    def test_build_with_workspace_scene(self):
        """Indoor workspace archetype should match with person + display."""
        scene = {
            "entity_count": 3,
            "visible_count": 3,
            "stable_count": 2,
            "entities": [
                {"entity_id": "p1", "label": "person", "confidence": 0.9,
                 "region": "desk_center", "stable_cycles": 5},
                {"entity_id": "m1", "label": "monitor", "confidence": 0.85,
                 "region": "monitor_zone", "stable_cycles": 10,
                 "is_display_surface": True},
                {"entity_id": "k1", "label": "keyboard", "confidence": 0.8,
                 "region": "desk_center", "stable_cycles": 10},
            ],
            "display_surfaces": [
                {"surface_id": "d0", "label": "monitor", "confidence": 0.9,
                 "region": "monitor_zone"},
            ],
            "display_content": [],
            "region_visibility": {"desk_center": 1.0, "monitor_zone": 0.9},
        }
        projector = CanonicalWorldProjector()
        cs = projector.build(
            now=1100.0,
            scene=scene,
            attention={"presence_confidence": 0.8, "engagement_level": 0.7},
            presence={"is_present": True, "confidence": 0.95},
            identity={"name": "David", "confidence": 0.8},
        )
        assert len(cs.entities) == 4  # 3 entities + 1 display surface
        assert len(cs.zones) == 2
        assert "indoor_workspace" in cs.archetypes_active
        assert len(cs.observations) > 0

    def test_build_workshop_scene(self):
        """Industrial workshop archetype should match with tool entities."""
        scene = {
            "entity_count": 2,
            "visible_count": 2,
            "stable_count": 2,
            "entities": [
                {"entity_id": "w1", "label": "wrench", "confidence": 0.85,
                 "region": "floor_zone", "stable_cycles": 3},
                {"entity_id": "p1", "label": "person", "confidence": 0.9,
                 "region": "work_area", "stable_cycles": 5},
            ],
            "display_surfaces": [],
            "display_content": [],
            "region_visibility": {"floor_zone": 0.7, "work_area": 0.9},
        }
        projector = CanonicalWorldProjector()
        cs = projector.build(now=1000.0, scene=scene, attention={},
                             presence={}, identity={})
        entity_labels = {e.label.lower() for e in cs.entities}
        assert "wrench" in entity_labels
        assert "person" in entity_labels

    def test_build_produces_diagnostics_via_world_model(self):
        """WorldModel.get_diagnostics() should return combined legacy + canonical stats."""
        from cognition.world_model import WorldModel

        class FakeScene:
            def get_state(self):
                return {
                    "entity_count": 2, "visible_count": 2, "stable_count": 1,
                    "entities": [
                        {"entity_id": "p1", "label": "person", "confidence": 0.9,
                         "region": "desk_center", "stable_cycles": 5},
                        {"entity_id": "m1", "label": "monitor", "confidence": 0.85,
                         "region": "monitor_zone", "stable_cycles": 10,
                         "is_display_surface": True},
                    ],
                    "display_surfaces": [
                        {"surface_id": "d0", "label": "monitor", "confidence": 0.9,
                         "region": "monitor_zone"},
                    ],
                    "display_content": [],
                    "region_visibility": {"desk_center": 1.0, "monitor_zone": 0.9},
                    "person_count": 1,
                }

        wm = WorldModel(scene_tracker=FakeScene())
        wm.update()
        diag = wm.get_diagnostics()
        assert "legacy" in diag
        assert "canonical" in diag
        assert "promotion" in diag
        assert diag["canonical"]["entity_count"] >= 2
        assert diag["canonical"]["zone_count"] >= 2
        assert "person" in diag["canonical"]["entity_kinds"]
        assert "work_area" in diag["canonical"]["zone_kinds"]
        assert diag["legacy"]["rules_total"] > 0

    def test_projector_uses_custom_registry(self):
        reg = ArchetypeRegistry()
        reg.register(ArchetypePack(
            archetype_id="custom",
            label="Custom",
            entity_priors=("cat", "dog"),
        ))
        projector = CanonicalWorldProjector(registry=reg)
        scene = {
            "entities": [
                {"entity_id": "c1", "label": "cat", "confidence": 0.9,
                 "region": "room", "stable_cycles": 1},
                {"entity_id": "d1", "label": "dog", "confidence": 0.9,
                 "region": "room", "stable_cycles": 1},
            ],
            "display_surfaces": [],
            "display_content": [],
            "region_visibility": {"room": 1.0},
        }
        cs = projector.build(now=1000.0, scene=scene, attention={},
                             presence={}, identity={})
        assert "custom" in cs.archetypes_active
