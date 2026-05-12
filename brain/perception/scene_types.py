"""Layer 3B Scene Continuity — shared type definitions.

These dataclasses form the contract for the stable scene world model.
Physical entities, display surfaces, and display content are separated
so monitor pixels never become room reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BBox = tuple[int, int, int, int]

EntityState = Literal["candidate", "visible", "occluded", "missing", "removed"]

ContentType = Literal[
    "unknown", "game", "code_editor", "terminal", "browser",
    "video", "chat_app", "document", "dashboard", "mixed_ui",
]

DeltaEvent = Literal[
    "entity_promoted", "entity_occluded", "entity_missing",
    "entity_removed", "entity_moved", "display_content_changed",
]

DISPLAY_SURFACE_LABELS = frozenset({"tv", "laptop", "monitor"})


@dataclass
class SceneDetection:
    """Normalized detection from Pi scene_summary or VLM enrichment."""
    label: str
    confidence: float
    bbox: BBox | None = None
    source: str = "pi"
    hit_count: int = 1
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneEntity:
    """Persistent world-model entity with state machine and permanence."""
    entity_id: str
    label: str
    confidence: float = 0.0
    permanence_confidence: float = 0.0
    bbox: BBox | None = None
    region: str = "unknown"
    state: EntityState = "candidate"
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    unseen_cycles: int = 0
    stable_cycles: int = 0
    source: str = "pi"
    is_display_surface: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "permanence_confidence": round(self.permanence_confidence, 3),
            "bbox": list(self.bbox) if self.bbox else None,
            "region": self.region,
            "state": self.state,
            "first_seen_ts": round(self.first_seen_ts, 1),
            "last_seen_ts": round(self.last_seen_ts, 1),
            "unseen_cycles": self.unseen_cycles,
            "stable_cycles": self.stable_cycles,
            "source": self.source,
            "is_display_surface": self.is_display_surface,
        }


@dataclass
class SceneDelta:
    """Typed scene change event."""
    event: DeltaEvent
    entity_id: str
    label: str
    region: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "entity_id": self.entity_id,
            "label": self.label,
            "region": self.region,
            "confidence": round(self.confidence, 3),
            "details": self.details,
        }


@dataclass
class DisplaySurface:
    """A physical display object in the room (monitor, TV, laptop screen)."""
    surface_id: str
    kind: str
    bbox: BBox | None = None
    confidence: float = 0.0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    stable_for_s: float = 0.0
    physical_state: str = "stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "kind": self.kind,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": round(self.confidence, 3),
            "stable_for_s": round(self.stable_for_s, 1),
            "physical_state": self.physical_state,
        }


@dataclass
class DisplayContentSummary:
    """What a display surface is showing — activity, not physical reality."""
    surface_id: str
    observed_at: float = 0.0
    content_type: ContentType = "unknown"
    confidence: float = 0.0
    activity_label: str = ""
    activity_confidence: float = 0.0
    semantic_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "content_type": self.content_type,
            "confidence": round(self.confidence, 3),
            "activity_label": self.activity_label,
            "activity_confidence": round(self.activity_confidence, 3),
            "semantic_summary": self.semantic_summary,
        }


@dataclass
class SceneSnapshot:
    """Timestamped stable world-state snapshot."""
    timestamp: float
    entities: list[SceneEntity] = field(default_factory=list)
    deltas: list[SceneDelta] = field(default_factory=list)
    display_surfaces: list[DisplaySurface] = field(default_factory=list)
    display_content: list[DisplayContentSummary] = field(default_factory=list)
    region_visibility: dict[str, float] = field(default_factory=dict)
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        physical = [e for e in self.entities if not e.is_display_surface]
        active = [e for e in physical if e.state != "removed"]
        visible = [e for e in physical if e.state == "visible"]
        stable = [e for e in physical if e.state in ("visible", "occluded")]
        removed = [e for e in physical if e.state == "removed"]
        return {
            "timestamp": round(self.timestamp, 1),
            # Backward-compatible total canonical-derived physical entities.
            "entity_count": len(physical),
            # Explicit live/history split for operator displays.
            "active_entity_count": len(active),
            "removed_entity_count": len(removed),
            "visible_count": len(visible),
            "stable_count": len(stable),
            "entities": [e.to_dict() for e in self.entities],
            "deltas": [d.to_dict() for d in self.deltas],
            "display_surfaces": [ds.to_dict() for ds in self.display_surfaces],
            "display_content": [dc.to_dict() for dc in self.display_content],
            "region_visibility": {k: round(v, 2) for k, v in self.region_visibility.items()},
            "update_count": self.update_count,
        }
