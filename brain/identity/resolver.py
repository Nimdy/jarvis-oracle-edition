"""Identity resolver for Layer 3: Identity Boundary Engine.

Wraps the existing IdentityFusion system and adds provenance-based,
actor-based, and payload-based identity resolution. Produces IdentityScope
with both owner and subject axes for every memory write.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from identity.types import (
    IdentityType,
    IdentitySignal,
    IdentityContext,
    IdentityScope,
    CONFIDENCE_THRESHOLDS,
)

logger = logging.getLogger(__name__)

_RELATIONSHIP_PATTERNS = re.compile(
    r"\b(?:my\s+)(wife|husband|partner|girlfriend|boyfriend|"
    r"mom|mother|dad|father|brother|sister|son|daughter|"
    r"friend|roommate|boss|coworker|colleague)\b",
    re.IGNORECASE,
)

_ABOUT_PATTERN = re.compile(
    r"\babout\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

_THIRD_PERSON_SAID = re.compile(
    r"([A-Z][a-z]+)\s+(?:said|says|told|mentioned|likes?|hates?|prefers?|wants?)\b"
)

_REL_ALIAS_PREFIX = "_rel_"


class IdentityResolver:
    """Resolves identity context for memory writes and retrieval queries."""

    _instance: IdentityResolver | None = None

    def __init__(self) -> None:
        self._fusion = None
        self._soul_identity = None
        self._known_names: set[str] = set()

    @classmethod
    def get_instance(cls) -> IdentityResolver:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_fusion(self, fusion: Any) -> None:
        self._fusion = fusion

    def set_soul(self, soul_identity: Any) -> None:
        self._soul_identity = soul_identity
        self._refresh_known_names()

    def _refresh_known_names(self) -> None:
        self._known_names.clear()
        if self._soul_identity and hasattr(self._soul_identity, "relationships"):
            for key in self._soul_identity.relationships:
                self._known_names.add(key.lower().strip())

    def get_known_names(self) -> set[str]:
        self._refresh_known_names()
        return set(self._known_names)

    def resolve_for_memory(
        self,
        provenance: str = "unknown",
        actor: str = "",
        speaker: str = "",
    ) -> IdentityContext:
        """Resolve identity context from available signals.

        Priority order:
        1. Conversation actor == system -> self
        2. Speaker+face fusion
        3. Speaker only / face only
        4. Provenance fallback
        5. Environment
        6. Guest/unknown
        """
        if actor == "system" or actor == "self":
            return IdentityContext(
                identity_id="jarvis",
                identity_type="self",
                confidence=1.0,
                signals=(IdentitySignal(source="actor", name="jarvis", confidence=1.0, is_known=True),),
                resolved_by="actor",
            )

        if self._fusion is not None:
            resolved = getattr(self._fusion, "current", None)
            if resolved and getattr(resolved, "name", "unknown") != "unknown":
                fusion_conf = getattr(resolved, "confidence", 0.0)
                fusion_name = getattr(resolved, "name", "unknown").lower().strip()
                is_known = getattr(resolved, "is_known", False)
                method = getattr(resolved, "method", "none")

                self._refresh_known_names()
                id_type: IdentityType = "primary_user" if is_known or fusion_name in self._known_names else "guest"
                if fusion_conf < CONFIDENCE_THRESHOLDS["soft"]:
                    id_type = "guest"

                return IdentityContext(
                    identity_id=fusion_name,
                    identity_type=id_type,
                    confidence=fusion_conf,
                    signals=(IdentitySignal(source="speaker", name=fusion_name, confidence=fusion_conf, is_known=is_known),),
                    resolved_by=f"fusion:{method}",
                )

        if speaker and speaker.lower().strip() != "unknown":
            s = speaker.lower().strip()
            self._refresh_known_names()
            id_type = "primary_user" if s in self._known_names else "guest"
            return IdentityContext(
                identity_id=s,
                identity_type=id_type,
                confidence=0.6,
                signals=(IdentitySignal(source="speaker", name=s, confidence=0.6, is_known=s in self._known_names),),
                resolved_by="speaker_tag",
            )

        prov_map: dict[str, tuple[str, IdentityType, float]] = {
            "seed": ("jarvis", "self", 1.0),
            "model_inference": ("jarvis", "self", 0.9),
            "experiment_result": ("jarvis", "self", 0.9),
            "external_source": ("external", "library", 0.95),
            "observed": ("scene", "environment", 0.85),
        }
        if provenance in prov_map:
            pid, ptype, pconf = prov_map[provenance]
            return IdentityContext(
                identity_id=pid,
                identity_type=ptype,
                confidence=pconf,
                signals=(IdentitySignal(source="provenance", name=pid, confidence=pconf, is_known=True),),
                resolved_by=f"provenance:{provenance}",
            )

        return IdentityContext(
            identity_id="",
            identity_type="unknown",
            confidence=0.2,
            signals=(),
            resolved_by="fallback",
        )

    def build_scope(
        self,
        context: IdentityContext,
        payload: Any = None,
        memory_type: str = "",
    ) -> IdentityScope:
        """Build full identity scope with owner + subject + confidence gating.

        Subject detection runs on user_preference and user_claim types to find
        third-party references like "my wife hates mushrooms".
        """
        owner_id = context.identity_id
        owner_type = context.identity_type
        subject_id = owner_id
        subject_type = owner_type
        confidence = context.confidence
        needs_resolution = False

        if memory_type in ("user_preference", "user_claim", "conversation") and payload:
            detected = self._detect_subject(payload, owner_id)
            if detected:
                subject_id, subject_type, needs_resolution = detected

        if confidence < CONFIDENCE_THRESHOLDS["quarantine"]:
            if owner_type in ("primary_user", "known_human"):
                owner_type = "guest"
                owner_id = owner_id or "unknown"
                if subject_type in ("primary_user", "known_human") and subject_id == context.identity_id:
                    subject_type = "guest"
                    subject_id = owner_id
            needs_resolution = True
        elif confidence < CONFIDENCE_THRESHOLDS["confident"]:
            needs_resolution = True

        scope_key = f"{owner_type}:{owner_id}" if owner_type and owner_id else ""

        return IdentityScope(
            owner_id=owner_id,
            owner_type=owner_type,
            subject_id=subject_id,
            subject_type=subject_type,
            confidence=confidence,
            needs_resolution=needs_resolution,
        )

    def _detect_subject(
        self, payload: Any, owner_id: str
    ) -> tuple[str, IdentityType, bool] | None:
        """Lightweight subject extraction from payload text.

        Returns (subject_id, subject_type, needs_resolution) or None if
        subject == owner.
        """
        text = payload if isinstance(payload, str) else str(payload)
        if not text:
            return None

        rel_match = _RELATIONSHIP_PATTERNS.search(text)
        if rel_match:
            alias = rel_match.group(1).lower()
            resolved_name = self._resolve_relationship_alias(alias)
            if resolved_name:
                return (resolved_name, "known_human", False)
            return (f"{_REL_ALIAS_PREFIX}{alias}", "known_human", True)

        self._refresh_known_names()

        about_match = _ABOUT_PATTERN.search(text)
        if about_match:
            name = about_match.group(1).lower().strip()
            if name != owner_id and name in self._known_names:
                return (name, "known_human", False)

        said_match = _THIRD_PERSON_SAID.search(text)
        if said_match:
            name = said_match.group(1).lower().strip()
            if name != owner_id and name in self._known_names:
                return (name, "known_human", False)

        return None

    def _resolve_relationship_alias(self, alias: str) -> str | None:
        """Try to resolve a relationship alias (wife, husband, etc.) to a known name.

        Priority:
        1. Explicit role field on Relationship (e.g., role="wife")
        2. Alias appears in notes/preferences text (legacy fallback)
        """
        if not self._soul_identity or not hasattr(self._soul_identity, "relationships"):
            return None

        alias_lower = alias.lower().strip()

        for name, rel in self._soul_identity.relationships.items():
            role = getattr(rel, "role", "")
            if role and role.lower().strip() == alias_lower:
                return name.lower().strip()

        for name, rel in self._soul_identity.relationships.items():
            notes = getattr(rel, "notes", [])
            prefs = getattr(rel, "preferences", {})
            all_text = " ".join(notes) + " " + " ".join(prefs.values())
            if alias_lower in all_text.lower():
                return name.lower().strip()

        return None

    def set_relationship_role(self, person_name: str, role: str) -> bool:
        """Set an explicit relationship role for a known person.

        Called when the user says "my wife is Sarah" or similar. This makes
        alias resolution reliable instead of depending on note scraping.
        """
        if not self._soul_identity or not hasattr(self._soul_identity, "relationships"):
            return False
        key = person_name.lower().strip()
        if key not in self._soul_identity.relationships:
            return False
        rel = self._soul_identity.relationships[key]
        rel.role = role.lower().strip()
        self._refresh_known_names()
        logger.info("Set relationship role: %s = %s", key, role)
        return True


    def expand_reference_aliases(self, raw_refs: set[str]) -> set[str]:
        """Expand raw entity references into all canonical forms.

        Given {"wife"}, returns {"wife", "_rel_wife", "sarah"} (if sarah
        is the resolved wife). This ensures boundary engine subject_id
        matching works regardless of how the memory was originally stored.
        """
        expanded = set(raw_refs)
        for ref in raw_refs:
            ref_lower = ref.lower().strip()
            expanded.add(f"{_REL_ALIAS_PREFIX}{ref_lower}")

            resolved = self._resolve_relationship_alias(ref_lower)
            if resolved:
                expanded.add(resolved)

            if self._soul_identity and hasattr(self._soul_identity, "relationships"):
                for name, rel in self._soul_identity.relationships.items():
                    name_lower = name.lower().strip()
                    role = getattr(rel, "role", "")
                    notes = getattr(rel, "notes", [])
                    prefs = getattr(rel, "preferences", {})
                    all_text = f"{role} " + " ".join(notes) + " " + " ".join(prefs.values())
                    if ref_lower in all_text.lower():
                        expanded.add(name_lower)
                    if name_lower == ref_lower:
                        if role:
                            expanded.add(f"{_REL_ALIAS_PREFIX}{role.lower()}")

        return expanded


identity_resolver = IdentityResolver.get_instance()
