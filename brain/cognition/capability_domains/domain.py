"""Matrix v2 — the Capability Domain object (Phase 0 contract).

A Capability Domain is an isolated, deletable bundle: its own knowledge store,
its own memory namespace, and (later) its own neural sub-consciousness + envelope.
Everything a domain learns lives UNDER ``root_dir`` and nowhere else, so deleting
the domain is a clean ablation — "forget snowboarding, keep everything else"
(docs/MATRIX_V2_CAPABILITY_DOMAINS.md §2.5). This module is pure data + paths; it
holds no behavior authority and never writes outside its own ``root_dir``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

# Lifecycle states (reuse the matrix-ladder vocabulary in spirit; domain-level).
DOMAIN_STATES = (
    "created",      # registered, empty
    "ingesting",    # taking in knowledge
    "learning",     # the sub-consciousness NN is training
    "active",       # matured: recallable on topic
    "retired",      # deleted/ablated
)

# Provenance partitions every fact/rep is tagged with (asymmetric-gate firewall).
PROVENANCE_KINDS = ("ingested", "lived", "synthetic")


@dataclass
class CapabilityDomain:
    """An isolated, deletable capability bundle. Pure record — see registry for I/O."""

    domain_id: str
    name: str
    root_dir: str                       # the ONLY place this domain may write
    kind: str = "document"              # "document" | "physical" (Phase 7+)
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # isolated stores (paths under root_dir)
    knowledge_db: str = ""              # isolated doc/chunk index
    memory_path: str = ""              # isolated memory namespace

    # the sub-consciousness (Phase 2) + envelope (Phase 3, physical domains)
    nn_focus: str = ""
    envelope: dict[str, Any] = field(default_factory=dict)

    # tallies (observability)
    source_count: int = 0
    chunk_count: int = 0
    provenance: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in PROVENANCE_KINDS}
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CapabilityDomain":
        known = {f for f in CapabilityDomain.__dataclass_fields__}  # type: ignore[attr-defined]
        return CapabilityDomain(**{k: v for k, v in d.items() if k in known})

    def public_view(self) -> dict[str, Any]:
        """Honest, read-only summary for /api/domains (no filesystem paths leaked)."""
        return {
            "domain_id": self.domain_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_count": self.source_count,
            "chunk_count": self.chunk_count,
            "nn_focus": self.nn_focus or None,
            "envelope": self.envelope or None,
            "provenance": self.provenance,
        }
