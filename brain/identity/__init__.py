"""Layer 3: Identity Boundary Engine.

Prevents memory cross-contamination between identities by adding dual-axis
identity scope (owner + subject) to Memory, enforcing retrieval boundaries
with referenced-subject exceptions, and wiring epistemic layers.
"""

from identity.types import (
    IdentityType,
    IdentitySignal,
    IdentityContext,
    IdentityScope,
    RetrievalSignature,
    CONFIDENCE_THRESHOLDS,
    PARTITION_KEY_FORMAT,
)

__all__ = [
    "IdentityType",
    "IdentitySignal",
    "IdentityContext",
    "IdentityScope",
    "RetrievalSignature",
    "CONFIDENCE_THRESHOLDS",
    "PARTITION_KEY_FORMAT",
]
