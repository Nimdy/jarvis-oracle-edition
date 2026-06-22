"""Spatial Place Consolidator — read-only consolidation of the episodic album into PLACES.

The spatial-episodic album (memory.spatial_episodic_store) is SESSION-keyed: a new
"world" per boot. JARVIS is stationary, so ~N reboots become ~N near-duplicate worlds
of the SAME room. This consolidator recognizes "same room" from the canonical room-frame
geometry already persisted in each record (the stable structural objects + their
position_room_m) and groups the per-session worlds into a small set of PLACES.

It is a READING-layer view. It never touches the capture layer, canonical state, or
authority — see docs/SPATIAL_PLACE_CONSOLIDATION.md.

GOVERNANCE (P4 HRR + P5 mental-world — AGENTS.md "HRR / VSA Governance Rules"):
  * ZERO-AUTHORITY / SHADOW / PRE-MATURE. Produces a derived view; writes nothing
    canonical (memory / beliefs / policy / autonomy / identity). AUTHORITY_FLAGS pinned
    False on every record.
  * DERIVED PROJECTION, NO NEW GEOMETRY. Consumes only the already-canonical scene-graph
    geometry persisted in the album (position_room_m). Thresholds come from
    cognition.spatial_schema (never invented). No perception access, no perc_orch.
  * VECTOR-FREE. Never reads, computes, persists, or exposes an HRR vector; a defensive
    strip runs on every emitted record.
  * STDLIB + spatial_schema ONLY (enforced by the validation-pack forbidden-import scan).
  * DETERMINISTIC + FAIL-CLOSED + CALIBRATION-INVARIANT. Matching is over inter-anchor
    distances (frame re-origin/rotation invariant), so the same physical room stays one place
    across recalibrations; insufficient or unavailable geometry -> its OWN place (never a
    false merge). The only claim is a measurable, auditable compression count, reported from
    live data — never asserted in advance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from cognition.spatial_schema import (
    CLASS_MOVE_THRESHOLDS,
    CONFIDENCE_THRESHOLD_STABLE,
    DEFAULT_MOVE_THRESHOLD,
)

logger = logging.getLogger("jarvis.spatial.place")

JARVIS_DIR = Path.home() / ".jarvis"
ALBUM_DIR = JARVIS_DIR / "spatial" / "episodic"

# Mirror the album's authority envelope verbatim — pinned False on every record.
AUTHORITY_FLAGS: dict[str, bool] = {
    "writes_memory": False,
    "writes_beliefs": False,
    "influences_policy": False,
    "influences_autonomy": False,
    "soul_integrity_influence": False,
    "llm_raw_vector_exposure": False,
    "no_raw_vectors_in_api": True,
}
LANE = "spatial_hrr_mental_world"
STATUS = "PRE-MATURE"

# Stable structural objects that DEFINE a place (low movement). Transient/movable
# objects (person, mouse, cup, bottle, laptop, suitcase) are excluded from the
# fingerprint so they cannot perturb place identity.
_STABLE_PLACE_LABELS = frozenset({"tv", "monitor", "desk", "chair", "keyboard"})
_MIN_STABLE_ANCHORS = 2                            # below this -> fail-closed to own place
_PLACE_MATCH_FRACTION = CONFIDENCE_THRESHOLD_STABLE  # 0.60 — reused, not invented

# Keys that must never appear in an emitted record (defensive vector-free guarantee).
_VECTOR_KEYS = ("vector", "raw_vector", "composite_vector", "ndarray", "hrr_vector")


def _move_threshold(label: str) -> float:
    return CLASS_MOVE_THRESHOLDS.get(label, DEFAULT_MOVE_THRESHOLD)


def _strip_vectors(obj: Any) -> Any:
    """Recursively drop any vector-bearing keys — defensive, never trusts upstream."""
    if isinstance(obj, dict):
        return {k: _strip_vectors(v) for k, v in obj.items() if k not in _VECTOR_KEYS}
    if isinstance(obj, list):
        return [_strip_vectors(v) for v in obj]
    return obj


def _stable_anchors(world: dict) -> list[tuple[str, tuple[float, float, float]]]:
    """(label, room_position) for stable structural objects with a real position."""
    out: list[tuple[str, tuple[float, float, float]]] = []
    for e in (world.get("entities") or []):
        label = e.get("label")
        pos = e.get("position_room_m")
        conf = e.get("confidence", 0.0)
        if (
            label in _STABLE_PLACE_LABELS
            and isinstance(pos, (list, tuple)) and len(pos) == 3
            and all(isinstance(v, (int, float)) for v in pos)
            and isinstance(conf, (int, float)) and conf >= CONFIDENCE_THRESHOLD_STABLE
        ):
            out.append((label, (float(pos[0]), float(pos[1]), float(pos[2]))))
    return out


def _rel_pairs(anchors: list) -> list[tuple[tuple[str, str], float]]:
    """Calibration-INVARIANT signature: (sorted label pair, inter-anchor distance) for
    every pair of stable anchors. Inter-object distances are preserved under a room-frame
    re-origin/rotation (recalibration), so the same physical room stays recognizable across
    the album's ~10 calibration versions — which absolute room positions do not."""
    pairs: list[tuple[tuple[str, str], float]] = []
    n = len(anchors)
    for i in range(n):
        for j in range(i + 1, n):
            (la, pa), (lb, pb) = anchors[i], anchors[j]
            pairs.append((tuple(sorted((la, lb))), math.dist(pa, pb)))
    return pairs


def _pair_tol(lp: tuple[str, str]) -> float:
    """Distance tolerance for a label-pair = the combined endpoint jitter (each anchor may
    drift by its own class move threshold). Derived from spatial_schema, never invented."""
    return _move_threshold(lp[0]) + _move_threshold(lp[1])


def _same_place(a: list, b: list) -> bool:
    """Deterministic, calibration-invariant geometry match: the fraction of A's inter-anchor
    distances that match a same-label-pair distance in B within the combined-jitter tolerance.
    FAIL-CLOSED when either side has <2 anchors (no pair to compare -> cannot confirm the same
    room -> treat as distinct, never a false merge)."""
    pa, pb = _rel_pairs(a), _rel_pairs(b)
    if not pa or not pb:
        return False
    used: set[int] = set()
    matched = 0
    for (lp, d) in pa:
        for k, (lp2, d2) in enumerate(pb):
            if k in used or lp2 != lp:
                continue
            if abs(d - d2) <= _pair_tol(lp):
                matched += 1
                used.add(k)
                break
    return (matched / max(len(pa), len(pb))) >= _PLACE_MATCH_FRACTION


def _place_id(anchors: list) -> str | None:
    """Calibration-invariant id from the relative-distance signature. None when there are
    too few anchors to form a pair (the caller assigns a fail-closed unkeyed id)."""
    pairs = _rel_pairs(anchors)
    if not pairs:
        return None
    sig = ";".join(f"{lp[0]}-{lp[1]}:{round(d,1)}" for lp, d in sorted(pairs))
    return "place_" + hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]


def _key_strength(anchors: list) -> str:
    """Honest confidence of a place key. A 2-anchor place rests on a SINGLE inter-anchor
    distance (one pair) at _PLACE_MATCH_FRACTION — geometrically weak: two distinct rooms
    that happen to share that one distance would merge. Safe in this single-household,
    zero-authority shadow, but >=2 pairs (>=3 anchors) should be REQUIRED before this loop
    is ever granted authority or generalized beyond one home (see docs/SPATIAL_PLACE_
    CONSOLIDATION.md)."""
    n = len(anchors)
    if n < _MIN_STABLE_ANCHORS:
        return "unkeyed"
    return "strong" if n >= 3 else "weak_single_pair"


class SpatialPlaceConsolidator:
    """Groups the per-session episodic album into persistent PLACES (read-only)."""

    def __init__(self, album_dir: Path | None = None) -> None:
        self._dir = Path(album_dir) if album_dir else ALBUM_DIR

    # -- loading -------------------------------------------------------------

    def _load_sessions(self) -> list[dict[str, Any]]:
        """One representative (latest world) per session file. Never raises."""
        sessions: list[dict[str, Any]] = []
        if not self._dir.is_dir():
            return sessions
        for f in sorted(self._dir.glob("*.jsonl")):
            last = None
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = json.loads(line)
                    except ValueError:
                        continue
            except OSError:
                continue
            if not isinstance(last, dict):
                continue
            sessions.append({
                "session_id": last.get("session_id") or f.stem,
                "world": last.get("world") or {},
                "calibration_version": last.get("calibration_version"),
                "captured_ts": last.get("captured_ts") or 0.0,
            })
        return sessions

    # -- consolidation -------------------------------------------------------

    def consolidate(self) -> dict[str, Any]:
        """Group sessions into places by deterministic geometry match. Read-only."""
        sessions = self._load_sessions()
        places: list[dict[str, Any]] = []
        for s in sessions:
            anchors = _stable_anchors(s["world"])
            assigned = None
            for p in places:
                if _same_place(anchors, p["_anchors"]):
                    assigned = p
                    break
            if assigned is None:
                places.append({
                    "_anchors": anchors,
                    "place_id": _place_id(anchors) or f"place_unkeyed_{len(places)}",
                    "record_kind": "place_consolidated",
                    "lane": LANE,
                    "status": STATUS,
                    "authority": dict(AUTHORITY_FLAGS),
                    "loaded_from_store": True,   # aggregate, never a live observation
                    "calibration_versions": sorted({s["calibration_version"]} - {None}),
                    "member_session_ids": [s["session_id"]],
                    "member_count": 1,
                    "first_seen_ts": s["captured_ts"],
                    "last_seen_ts": s["captured_ts"],
                    "anchors": [{"label": l, "position_room_m": list(p)} for l, p in anchors],
                    "fail_closed_unkeyed": len(anchors) < _MIN_STABLE_ANCHORS,
                    "key_strength": _key_strength(anchors),
                })
            else:
                assigned["member_session_ids"].append(s["session_id"])
                assigned["member_count"] += 1
                assigned["first_seen_ts"] = min(assigned["first_seen_ts"], s["captured_ts"])
                assigned["last_seen_ts"] = max(assigned["last_seen_ts"], s["captured_ts"])
                cv = s["calibration_version"]
                if cv is not None and cv not in assigned["calibration_versions"]:
                    assigned["calibration_versions"] = sorted(assigned["calibration_versions"] + [cv])
        # strip the working anchor field + any (impossible) vector keys before emit
        records = [_strip_vectors({k: v for k, v in p.items() if k != "_anchors"}) for p in places]
        n_keyed = sum(1 for p in places if not p["fail_closed_unkeyed"])
        n_strong = sum(1 for p in places if p["key_strength"] == "strong")
        n_weak = sum(1 for p in places if p["key_strength"] == "weak_single_pair")
        return {
            "sessions": len(sessions),
            "places": len(places),
            "places_keyed": n_keyed,
            "places_keyed_strong": n_strong,        # >=3 anchors (>=2 inter-anchor pairs)
            "places_keyed_weak_single_pair": n_weak,  # 2 anchors -> one distance only (weak)
            "places_unkeyed_failclosed": len(places) - n_keyed,
            "compression_ratio": round(len(sessions) / max(1, len(places)), 2),
            "authority": dict(AUTHORITY_FLAGS),
            "no_raw_vectors_in_api": True,
            "status": STATUS,
            "lane": LANE,
            "place_records": records,
            "note": ("derived, read-only consolidation of the shadow album; deterministic "
                     "geometry match with spatial_schema thresholds; fail-closed (insufficient "
                     "geometry -> own place); vector-free; zero authority; PRE-MATURE"),
        }
