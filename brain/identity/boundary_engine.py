"""Retrieval boundary engine for Layer 3: Identity Boundary Engine.

Enforces identity-based retrieval policies with three states:
allow, block, and allow_if_referenced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from identity.types import (
    IdentityContext,
    RetrievalSignature,
)

logger = logging.getLogger(__name__)

_ALLOW_ALL_TYPES = frozenset({"self", "environment", "library", "unknown"})


@dataclass(frozen=True)
class BoundaryDecision:
    allow: bool
    reason: str
    confidence: float
    requires_audit: bool = False
    requires_explicit_reference: bool = False


class IdentityBoundaryEngine:
    """Governs which memories are visible to whom during retrieval."""

    _instance: IdentityBoundaryEngine | None = None

    @classmethod
    def get_instance(cls) -> IdentityBoundaryEngine:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def validate_retrieval(
        self,
        query_context: IdentityContext,
        candidate_memory: Any,
        referenced_entities: set[str] | None = None,
    ) -> BoundaryDecision:
        """Check whether a candidate memory is retrievable for the given identity context.

        Returns a BoundaryDecision with allow/block/allow_if_referenced.
        """
        sig = RetrievalSignature.from_memory(candidate_memory)

        if not sig.owner[0] or sig.owner[0] in _ALLOW_ALL_TYPES:
            return BoundaryDecision(allow=True, reason="universal_type", confidence=1.0)

        needs_res = getattr(candidate_memory, "identity_needs_resolution", False)

        querier_type = query_context.identity_type
        querier_id = query_context.identity_id

        if querier_type == "guest":
            return self._policy_guest(sig, needs_res)
        elif querier_type == "self":
            return self._policy_self(sig, referenced_entities, needs_res)
        elif querier_type in ("primary_user", "known_human"):
            return self._policy_human(
                querier_type, querier_id, sig, referenced_entities, needs_res,
            )
        else:
            return BoundaryDecision(allow=True, reason="unknown_querier", confidence=0.5)

    def _policy_guest(
        self, sig: RetrievalSignature, needs_res: bool,
    ) -> BoundaryDecision:
        if sig.owner[0] in _ALLOW_ALL_TYPES:
            return BoundaryDecision(allow=True, reason="guest_universal", confidence=1.0)
        return BoundaryDecision(
            allow=False, reason="guest_blocked_personal", confidence=1.0,
        )

    def _policy_self(
        self,
        sig: RetrievalSignature,
        referenced_entities: set[str] | None,
        needs_res: bool,
    ) -> BoundaryDecision:
        if sig.owner[0] in ("self", "library", "environment"):
            return BoundaryDecision(allow=True, reason="self_own_scope", confidence=1.0)

        if sig.owner[0] in ("guest", "external_agent"):
            return BoundaryDecision(allow=False, reason="self_blocked_external", confidence=1.0)

        if referenced_entities and sig.subject[1] and sig.subject[1] in referenced_entities:
            return BoundaryDecision(
                allow=True, reason="self_referenced_user", confidence=0.8,
                requires_explicit_reference=True,
            )

        return BoundaryDecision(
            allow=False, reason="self_blocked_user_pref", confidence=0.9,
            requires_explicit_reference=True,
        )

    def _policy_human(
        self,
        querier_type: str,
        querier_id: str,
        sig: RetrievalSignature,
        referenced_entities: set[str] | None,
        needs_res: bool,
    ) -> BoundaryDecision:
        owner_type, owner_id = sig.owner
        subject_type, subject_id = sig.subject

        if owner_type in _ALLOW_ALL_TYPES:
            return BoundaryDecision(allow=True, reason="universal_owner", confidence=1.0)

        if owner_id == querier_id and owner_type == querier_type:
            if subject_id == querier_id or not subject_id:
                return BoundaryDecision(allow=True, reason="own_scope", confidence=1.0)

            if referenced_entities and subject_id in referenced_entities:
                return BoundaryDecision(
                    allow=True, reason="own_mem_referenced_subject", confidence=0.9,
                    requires_explicit_reference=True,
                )

            return BoundaryDecision(
                allow=False, reason="cross_subject_not_referenced", confidence=0.8,
                requires_explicit_reference=True, requires_audit=True,
            )

        if owner_type == "external_agent":
            return BoundaryDecision(allow=False, reason="blocked_external_agent", confidence=1.0)

        if subject_id and referenced_entities and subject_id in referenced_entities:
            return BoundaryDecision(
                allow=True, reason="referenced_subject_exception", confidence=0.7,
                requires_explicit_reference=True, requires_audit=True,
            )

        if owner_type in ("primary_user", "known_human", "guest"):
            return BoundaryDecision(
                allow=False, reason="cross_identity_blocked", confidence=0.9,
            )

        return BoundaryDecision(allow=True, reason="fallthrough_allow", confidence=0.5)

    def is_preference_injectable(self, memory: Any) -> bool:
        """Check whether a memory can be injected as a user preference into LLM prompts."""
        if getattr(memory, "identity_needs_resolution", False):
            return False
        if getattr(memory, "identity_confidence", 0.0) < 0.45:
            return False
        return True


identity_boundary_engine = IdentityBoundaryEngine.get_instance()
