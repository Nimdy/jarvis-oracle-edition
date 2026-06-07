"""Matrix v2 — Capability Domains: isolated, deletable skill sub-consciousnesses.

See ``docs/MATRIX_V2_CAPABILITY_DOMAINS.md``. Phase 1 = the isolation substrate
(registry + per-domain isolated stores + clean deletion). No behavior authority.
"""
from cognition.capability_domains.domain import (
    CapabilityDomain,
    DOMAIN_STATES,
    PROVENANCE_KINDS,
)
from cognition.capability_domains.registry import (
    CapabilityDomainRegistry,
    get_capability_domain_registry,
)

__all__ = [
    "CapabilityDomain",
    "DOMAIN_STATES",
    "PROVENANCE_KINDS",
    "CapabilityDomainRegistry",
    "get_capability_domain_registry",
]
