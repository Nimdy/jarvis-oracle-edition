"""Provenance primitive for the Operational Self-View (OSV P0).

Every self-fact the OSV reports is wrapped in a :class:`Fact` carrying an explicit
provenance level. Crucially, ``is_measurement`` is DERIVED from the provenance — it
cannot be set independently — so an internally-scored / shadow / synthetic value can
never render as a measurement. This is the honesty invariant the whole self-view rests
on (KNOW-not-guess).

Provenance levels are enum-like constants (not loose strings) with a validated set, to
prevent dashboard/schema drift later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Provenance:
    """Allowed provenance levels for an OSV fact (enum-like)."""

    MEASURED = "measured"                 # validated against an external/ground-truth comparator
    INTERNALLY_SCORED = "internally_scored"  # computed by JARVIS from its own outputs, not validated
    SELF_SCORED = "self_scored"           # JARVIS grading itself (e.g. oracle benchmark)
    SHADOW_ONLY = "shadow_only"           # observed in shadow; zero behavioral authority
    SYNTHETIC_ONLY = "synthetic_only"     # produced under a synthetic session; never lived proof
    ADVISORY = "advisory"                 # non-authoritative signal/recommendation
    DORMANT = "dormant"                   # capability exists but is gate-blocked / not active
    UNKNOWN = "unknown"                   # source reachable but value not determinable
    STALE = "stale"                       # value exists but is too old to trust
    GAP = "gap"                           # no comparator / not measured / not yet known (first-class)


ALL_PROVENANCE: frozenset[str] = frozenset({
    Provenance.MEASURED, Provenance.INTERNALLY_SCORED, Provenance.SELF_SCORED,
    Provenance.SHADOW_ONLY, Provenance.SYNTHETIC_ONLY, Provenance.ADVISORY,
    Provenance.DORMANT, Provenance.UNKNOWN, Provenance.STALE, Provenance.GAP,
})

# ONLY this level is a genuine external measurement. is_measurement derives from this set;
# nothing else can ever be true. Keep it a set so the rule is auditable in one place.
MEASUREMENT_LEVELS: frozenset[str] = frozenset({Provenance.MEASURED})

# Levels that represent "we don't actually know this value" — used for gap accounting.
ABSENT_LEVELS: frozenset[str] = frozenset({
    Provenance.UNKNOWN, Provenance.STALE, Provenance.GAP,
})


@dataclass(frozen=True)
class Fact:
    """A single self-fact with explicit, validated provenance.

    ``is_measurement`` is a derived property — it is True iff provenance is a measurement
    level. There is intentionally no way to assert is_measurement independently.
    """

    value: Any
    provenance: str
    note: str = ""
    source: str = ""
    age_s: float | None = None

    def __post_init__(self) -> None:
        # Coerce an unknown/loose provenance to UNKNOWN rather than trust it — a bad label
        # must never silently pass as something stronger.
        if self.provenance not in ALL_PROVENANCE:
            object.__setattr__(self, "provenance", Provenance.UNKNOWN)

    @property
    def is_measurement(self) -> bool:
        return self.provenance in MEASUREMENT_LEVELS

    @property
    def is_absent(self) -> bool:
        return self.provenance in ABSENT_LEVELS

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "value": self.value,
            "provenance": self.provenance,
            "is_measurement": self.is_measurement,
        }
        if self.note:
            d["note"] = self.note
        if self.source:
            d["source"] = self.source
        if self.age_s is not None:
            d["age_s"] = round(self.age_s, 1)
        return d


def gap(reason: str, source: str = "") -> Fact:
    """A first-class gap: we do not know this, and that absence is itself knowledge."""
    return Fact(value=None, provenance=Provenance.GAP, note=reason, source=source)


def unknown(reason: str, source: str = "") -> Fact:
    return Fact(value=None, provenance=Provenance.UNKNOWN, note=reason, source=source)
