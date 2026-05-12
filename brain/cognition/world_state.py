"""Data models for the Unified World Model belief state.

Four facets — physical, user, conversation, system — plus per-facet
uncertainty and a monotonic version counter for race-free consumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

FacetName = Literal["physical", "user", "conversation", "system"]

DeltaEvent = Literal[
    # Physical
    "entity_appeared",
    "entity_disappeared",
    "entity_moved",
    "display_content_changed",
    # User
    "user_arrived",
    "user_departed",
    "emotion_changed",
    "engagement_crossed_threshold",
    "speaker_changed",
    # Conversation
    "conversation_started",
    "conversation_ended",
    "topic_changed",
    "follow_up_started",
    # System
    "mode_changed",
    "health_degraded",
    "health_recovered",
    "goal_promoted",
    "goal_completed",
]


# ---------------------------------------------------------------------------
# Facet dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalState:
    """Room / environment snapshot from SceneTracker."""
    entities: tuple[dict[str, Any], ...] = ()
    entity_count: int = 0
    visible_count: int = 0
    stable_count: int = 0
    display_surfaces: tuple[dict[str, Any], ...] = ()
    display_content: tuple[dict[str, Any], ...] = ()
    region_visibility: dict[str, float] = field(default_factory=dict)
    person_count: int = 0
    last_update_ts: float = 0.0


@dataclass(frozen=True)
class UserState:
    """Behavioural state of the primary user."""
    present: bool = False
    presence_confidence: float = 0.0
    engagement: float = 0.0
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    emotion_trusted: bool = False
    speaker_name: str = ""
    speaker_confidence: float = 0.0
    identity_method: str = ""
    gesture: str = ""
    gesture_confidence: float = 0.0
    seconds_since_last_interaction: float = 0.0
    last_update_ts: float = 0.0


@dataclass(frozen=True)
class ConversationState:
    """Dialogue context."""
    active: bool = False
    topic: str = ""
    last_user_text: str = ""
    last_response_text: str = ""
    conversation_id: str = ""
    turn_count: int = 0
    follow_up_active: bool = False
    current_route: str = ""
    last_update_ts: float = 0.0


@dataclass(frozen=True)
class SystemState:
    """Agent internal state."""
    mode: str = "passive"
    health_score: float = 1.0
    confidence: float = 0.0
    autonomy_level: int = 0
    active_goal_title: str = ""
    active_goal_kind: str = ""
    memory_count: int = 0
    uptime_s: float = 0.0
    last_update_ts: float = 0.0


# ---------------------------------------------------------------------------
# Composite WorldState
# ---------------------------------------------------------------------------

UNCERTAINTY_STALE_THRESHOLD_S: dict[FacetName, float] = {
    "physical": 30.0,
    "user": 60.0,
    "conversation": 120.0,
    "system": 300.0,
}


def _compute_uncertainty(facet: FacetName, staleness_s: float,
                         confidence_signals: list[float]) -> float:
    """Compute facet uncertainty in [0, 1].

    Combines sensor staleness (grows linearly toward 1.0 at the stale
    threshold) with inverse mean confidence of available signals.
    """
    stale_thresh = UNCERTAINTY_STALE_THRESHOLD_S.get(facet, 60.0)
    staleness_factor = min(staleness_s / stale_thresh, 1.0)

    if confidence_signals:
        mean_conf = sum(confidence_signals) / len(confidence_signals)
        confidence_factor = 1.0 - mean_conf
    else:
        confidence_factor = 0.5

    return round(min(staleness_factor * 0.6 + confidence_factor * 0.4, 1.0), 3)


@dataclass(frozen=True)
class WorldState:
    """The agent's complete belief state — one snapshot per update tick."""
    physical: PhysicalState = field(default_factory=PhysicalState)
    user: UserState = field(default_factory=UserState)
    conversation: ConversationState = field(default_factory=ConversationState)
    system: SystemState = field(default_factory=SystemState)

    version: int = 0
    timestamp: float = 0.0
    tick_number: int = 0

    staleness: dict[str, float] = field(default_factory=lambda: {
        "physical": 0.0, "user": 0.0, "conversation": 0.0, "system": 0.0,
    })
    uncertainty: dict[str, float] = field(default_factory=lambda: {
        "physical": 0.5, "user": 0.5, "conversation": 0.5, "system": 0.0,
    })

    def resolve_field(self, dotted_path: str) -> Any:
        """Resolve ``"user.engagement"`` → ``self.user.engagement``.

        Returns *None* when the path doesn't exist (no KeyError).
        """
        parts = dotted_path.split(".", 1)
        facet = getattr(self, parts[0], None)
        if facet is None:
            return None
        if len(parts) == 1:
            return facet
        return getattr(facet, parts[1], None)

    def to_dict(self) -> dict[str, Any]:
        """Dashboard-friendly serialisation."""
        from dataclasses import asdict
        d = asdict(self)
        d["physical"]["entities"] = list(d["physical"]["entities"])
        d["physical"]["display_surfaces"] = list(d["physical"]["display_surfaces"])
        d["physical"]["display_content"] = list(d["physical"]["display_content"])
        return d


# ---------------------------------------------------------------------------
# WorldDelta
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldDelta:
    """A typed change between two consecutive WorldState snapshots."""
    facet: FacetName
    event: str  # one of DeltaEvent values
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
