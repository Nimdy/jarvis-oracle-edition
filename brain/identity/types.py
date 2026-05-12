"""Identity type definitions for Layer 3: Identity Boundary Engine.

Provides the core data structures for dual-axis identity scoping on Memory
(owner = who produced it, subject = who/what it is about).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

IdentityType = Literal[
    "self",
    "primary_user",
    "known_human",
    "guest",
    "external_agent",
    "environment",
    "library",
    "unknown",
]

PARTITION_KEY_FORMAT = "{owner_type}:{owner_id}"

CONFIDENCE_THRESHOLDS = {
    "confident": 0.80,
    "soft": 0.45,
    "quarantine": 0.45,
}


@dataclass(frozen=True)
class IdentitySignal:
    """Individual biometric or contextual identity signal."""

    source: Literal["speaker", "face", "provenance", "actor", "vision", "manual"]
    name: str = "unknown"
    confidence: float = 0.0
    is_known: bool = False


@dataclass(frozen=True)
class IdentityContext:
    """Resolved canonical identity for an interaction or memory event."""

    identity_id: str = ""
    identity_type: IdentityType = "unknown"
    confidence: float = 0.0
    signals: tuple[IdentitySignal, ...] = ()
    resolved_by: str = "none"


@dataclass(frozen=True)
class IdentityScope:
    """Memory-level identity scope with both owner and subject axes.

    owner = who produced/observed the memory
    subject = who/what the memory is about
    """

    owner_id: str = ""
    owner_type: IdentityType = "unknown"
    subject_id: str = ""
    subject_type: IdentityType = "unknown"
    confidence: float = 0.0
    needs_resolution: bool = False

    @property
    def scope_key(self) -> str:
        if not self.owner_type or not self.owner_id:
            return ""
        return PARTITION_KEY_FORMAT.format(
            owner_type=self.owner_type, owner_id=self.owner_id
        )

    @property
    def subject_key(self) -> str:
        if not self.subject_type or not self.subject_id:
            return ""
        return f"{self.subject_type}:{self.subject_id}"


@dataclass(frozen=True)
class RetrievalSignature:
    """Computed (not stored) retrieval context for boundary decisions.

    Keeps identity_scope_key simple for storage/dashboard partitioning
    while giving the boundary engine both axes for retrieval logic.
    """

    owner: tuple[str, str] = ("", "")
    subject: tuple[str, str] = ("", "")
    needs_resolution: bool = False

    @staticmethod
    def from_memory(memory: object) -> RetrievalSignature:
        return RetrievalSignature(
            owner=(
                getattr(memory, "identity_owner_type", ""),
                getattr(memory, "identity_owner", ""),
            ),
            subject=(
                getattr(memory, "identity_subject_type", ""),
                getattr(memory, "identity_subject", ""),
            ),
            needs_resolution=getattr(memory, "identity_needs_resolution", False),
        )
