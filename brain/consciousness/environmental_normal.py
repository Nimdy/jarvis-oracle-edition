"""
Environmental memory-of-normal — the "be there for the room" half of presence.

Shadow-only. Learns, by quiet observation over time, where each object USUALLY
lives in the room (its dominant region), then notices when a known object is
actually SEEN somewhere other than where it lives, and logs the gentle thing
JARVIS *would* note ("the cup isn't in its usual spot — it usually sits on the
desk, but right now it's on the left"). It is NEVER spoken, it is a hypothesis,
and it is salience-gated (a confident learned normal + a debounced, real sighting
elsewhere).

DESIGN STANCE (data-minimal — decided with David after an adversarial review).
This subsystem makes exactly ONE kind of claim: "moved". It only ever says a
thing is in the WRONG place — and only from a REAL fresh sighting of it there.
It deliberately does NOT claim a thing is GONE/MISSING. Reliable absence ("your
cup is gone") would require distinguishing occlusion (behind your arm, still
here) from removal (actually taken away), which in turn needs person-occlusion
geometry we have chosen NOT to wire — both to avoid piping person-presence data
into perception that doesn't truly need it (sovereignty / data-minimization),
and because "moved to the wrong spot" is the honest, higher-value dignity-anchor
nudge anyway. So: it KNOWS (from a real sighting), it never GUESSES absence.

HOW IT LEARNS. We accrue an object's usual spot from what the tracker still
BELIEVES EXISTS (permanence above a floor, not yet "removed"), not only from what
is freshly detected this frame — so a briefly-undetected object keeps accruing
its spot for a short window instead of resetting instantly.

HONEST LIMITATION (current perception pipeline). The pipeline does NOT feed
person/occlusion geometry, so an undetected object's permanence decays FAST
(there is no slow "occluded" floor), and this engine samples the tracker only
once per ~60 s. So permanence bridges only SHORT gaps, not long occlusions:
a frequently-undetected movable object (the cup/keys this lane most wants) learns
its usual spot SLOWLY, and in a sparse scene this lane may legitimately stay
silent for a long time. That is honest under-firing, not "working" — it FAILS
SAFE (it can still never assert absence, and "moved" requires a confident fresh
sighting). The lane gets meaningfully sharper only as perception matures
(person-aware occlusion, finer regions with hysteresis, faster accrual cadence) —
see get_status()["limitations"].

Complements, and does NOT touch:
  - the novel-object curiosity ask (that lane = "new thing in the room, what is
    it?"); this lane = "a KNOWN thing has been seen away from where it lives".
  - the PRE-MATURE hrr_scene spatial mind's-eye (the heavy spatial graph). This
    is a lightweight per-label accumulator riding the MATURE scene-tracker entities.

Fed once per scene-continuity tick (~60 s) via observe_scene(scene_state). Pure
stdlib; persisted to ~/.jarvis/environmental_normal.json so the learned normal
accrues across days and survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --- Gates (deliberately conservative — see module docstring) -------------
_MIN_OBS = 15            # present-ticks an object must accrue before it can have a "usual spot"
_MIN_DOMINANCE = 0.70    # fraction of present-ticks in one region to call it the usual spot
_CONF_FLOOR = 0.35       # ignore low-confidence FRESH detections (noise) for accrual
_MOVED_CONF = 0.50       # a "moved" sighting elsewhere must be a CONFIDENT fresh detection (anti-jitter)
_PERM_FLOOR = 0.40       # tracker permanence_confidence floor to count an object as still-existing
_DEBOUNCE = 2            # consecutive ticks a deviation must hold before surfaced/counted (and to clear)
_COUNT_CAP = 240         # total region-observations before exponential forgetting (lets a real relocation adapt)
_STALE_TICKS = 720       # evict a never-learned object unseen this many ticks (~12h at 60s)
_MAX_OBJECTS = 200       # hard cap on tracked labels (eviction backstop)
_EXCLUDE_REGIONS = {"unknown", "background"}  # not a place an object "lives" on the desk
_EXIST_EXCLUDE_STATES = {"removed", "candidate"}  # gave-up / unconfirmed — not "still believed to exist"
_SAVE_EVERY = 5          # persist every N observe ticks
_MAX_NORMALS_OUT = 10    # cap learned-normals surfaced in status
_STATE_VERSION = 3       # bumped: v1 visible-only, v2 occluded-premise, v3 permanence-accrual / moved-only

_STATE_DIR = os.path.expanduser("~/.jarvis")
_STATE_PATH = os.path.join(_STATE_DIR, "environmental_normal.json")


def _human(region: str) -> str:
    return (region or "unknown").replace("_", " ")


@dataclass
class ObjectNormal:
    """The learned usual layout of one labelled object."""
    label: str
    region_counts: dict[str, int] = field(default_factory=dict)
    seen_ticks: int = 0          # ticks the tracker believed this object existed (permanence-present)
    first_tick: int = 0
    last_tick: int = 0
    last_region: str = "unknown"
    last_updated: float = 0.0

    def dominant(self) -> tuple[str, float]:
        if not self.region_counts:
            return ("unknown", 0.0)
        total = sum(self.region_counts.values())
        region, count = max(self.region_counts.items(), key=lambda kv: kv[1])
        return (region, (count / total) if total else 0.0)

    def present_frac(self, current_tick: int) -> float:
        if current_tick < self.first_tick:   # only possible via a corrupt/edited state file
            return 0.0
        life = max(1, current_tick - self.first_tick + 1)
        return min(1.0, self.seen_ticks / life)

    def has_normal(self) -> bool:
        if self.seen_ticks < _MIN_OBS:
            return False
        region, frac = self.dominant()
        return frac >= _MIN_DOMINANCE and region not in _EXCLUDE_REGIONS

    def accrue(self, region: str, tick: int, now: float) -> None:
        self.seen_ticks += 1
        self.region_counts[region] = self.region_counts.get(region, 0) + 1
        self.last_region = region
        self.last_tick = tick
        self.last_updated = now
        # Exponential forgetting so a genuine permanent relocation can become the
        # new usual spot over time (and counts stay bounded).
        if sum(self.region_counts.values()) > _COUNT_CAP:
            self.region_counts = {r: c // 2 for r, c in self.region_counts.items() if c // 2 > 0}

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "region_counts": dict(self.region_counts),
            "seen_ticks": self.seen_ticks,
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "last_region": self.last_region,
            "last_updated": round(self.last_updated, 1),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectNormal":
        return cls(
            label=d.get("label", ""),
            region_counts={str(k): int(v) for k, v in (d.get("region_counts") or {}).items()},
            seen_ticks=int(d.get("seen_ticks", 0)),
            first_tick=int(d.get("first_tick", 0)),
            last_tick=int(d.get("last_tick", 0)),
            last_region=d.get("last_region", "unknown"),
            last_updated=float(d.get("last_updated", 0.0)),
        )


class EnvironmentalNormalEngine:
    """Shadow memory-of-normal: learns each object's usual spot from what the tracker
    believes EXISTS, would-note a real, debounced 'moved' sighting. Never claims absence,
    never speaks, never writes a belief, never changes behavior."""

    def __init__(self) -> None:
        self._objects: dict[str, ObjectNormal] = {}
        self._tick: int = 0
        self._flagged_total: int = 0
        self._counted: set[str] = set()        # "label|kind" deviations already counted (persisted)
        self._streak: dict[str, int] = {}      # "label|kind" -> consecutive active ticks (in-memory)
        self._idle: dict[str, int] = {}        # "label|kind" -> consecutive inactive ticks (in-memory)
        self._last_observations: list[dict[str, Any]] = []
        self._last_load_ok: bool = True
        self._last_scene_ts: float | None = None   # staleness guard: skip frozen snapshots
        self._load()

    # --- observation ------------------------------------------------------
    def observe_scene(self, scene_state: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Fold one scene snapshot into the learned normal. Returns current would-notes (shadow)."""
        if not isinstance(scene_state, dict):
            return list(self._last_observations)
        # Staleness guard: get_state() returns the LAST snapshot even if the tracker has not
        # updated (Pi silent / loop stalled). Re-folding a frozen snapshot would re-count the
        # same instant as fresh presence and could hold a stale "moved" forever. Skip it.
        ts = scene_state.get("timestamp")
        if ts is not None and self._last_scene_ts is not None and ts <= self._last_scene_ts:
            return list(self._last_observations)
        self._last_scene_ts = ts
        self._tick += 1
        now = time.time()

        # Per-label aggregation across (possibly multiple) tracks of the same label.
        exist_region: dict[str, str] = {}      # label -> region to accrue (tracker believes it exists)
        exist_best: dict[str, tuple[int, float]] = {}  # label -> (visible_priority, weight) of the chosen track
        fresh_regions: dict[str, dict[str, float]] = {}  # label -> {region: max fresh-visible conf}
        for e in (scene_state.get("entities") or []):
            if not isinstance(e, dict):
                continue
            if e.get("is_display_surface"):
                continue
            label = e.get("label")
            if not label:
                continue
            state = e.get("state")
            region = e.get("region") or "unknown"
            conf = float(e.get("confidence") or 0.0)
            perm = float(e.get("permanence_confidence") or 0.0)
            is_visible = (state == "visible" and conf >= _CONF_FLOOR)

            # Fresh-visible regions + their best confidence (for the "moved" claim — only real,
            # current sightings; confidence is used to gate out low-conf jitter).
            if is_visible and region not in _EXCLUDE_REGIONS:
                fr = fresh_regions.setdefault(label, {})
                if conf > fr.get(region, 0.0):
                    fr[region] = conf

            # Existence accrual: tracker still believes the object is here (permanence),
            # not yet given-up/unconfirmed, and in a real region. Prefer a fresh-visible
            # track's region over a stale remembered one.
            believed_exists = (state not in _EXIST_EXCLUDE_STATES and perm >= _PERM_FLOOR)
            if believed_exists and region not in _EXCLUDE_REGIONS:
                priority = 1 if is_visible else 0
                weight = conf if is_visible else perm
                cur = exist_best.get(label)
                if cur is None or (priority, weight) > cur:
                    exist_best[label] = (priority, weight)
                    exist_region[label] = region

        for label, region in exist_region.items():
            on = self._objects.get(label)
            if on is None:
                on = ObjectNormal(label=label, first_tick=self._tick)
                self._objects[label] = on
            on.accrue(region, self._tick, now)

        self._last_observations = self._debounce(self._compute_candidates(fresh_regions))

        if self._tick % _SAVE_EVERY == 0:
            self._prune()
            self._save()
        return list(self._last_observations)

    def _compute_candidates(self, fresh_regions: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
        """Raw (pre-debounce) 'moved' deviations: a known object CONFIDENTLY, FRESHLY SEEN away
        from its usual spot — and NOT also seen in its usual spot this tick."""
        out: list[dict[str, Any]] = []
        for label, on in self._objects.items():
            if not on.has_normal():
                continue
            seen = fresh_regions.get(label)
            if not seen:
                continue  # no fresh sighting this tick → no claim (we never assert absence)
            dom_region, dom_frac = on.dominant()
            if dom_region in seen:
                continue  # it IS in its usual spot (perhaps also elsewhere) → not "moved"
            # Require a CONFIDENT sighting elsewhere (anti-jitter): a low-conf blip in a
            # neighbouring region must not assert "moved". Pick the most-confident such region.
            elsewhere = [(r, c) for r, c in seen.items()
                         if r not in _EXCLUDE_REGIONS and r != dom_region and c >= _MOVED_CONF]
            if not elsewhere:
                continue
            cur_region = max(elsewhere, key=lambda rc: rc[1])[0]
            out.append({
                "object": label,
                "would_gently_note": ("the %s isn't in its usual spot — it usually sits in the %s, "
                                       "but right now it's in the %s"
                                       % (label, _human(dom_region), _human(cur_region))),
                "kind": "moved",
                "usual_region": dom_region,
                "usual_region_frac": round(dom_frac, 2),
                "current_region": cur_region,
                "observations": on.seen_ticks,
                "basis": "a hypothesis from the room's learned usual layout, not a fact",
                "spoken": False,
                "writes_belief": False,
                "status": "shadow_logged_only",
            })
        return out

    def _debounce(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Surface/count a deviation only once it has held for >=_DEBOUNCE consecutive ticks,
        and only clear it (allowing a future recurrence to count) once it has been gone for
        >=_DEBOUNCE ticks. Kills single-tick jitter both ways; _counted (persisted) keeps a
        standing deviation from being re-counted across restarts. Keyed on (label,kind,region)
        so a sighting that jitters to a DIFFERENT wrong region must itself persist before it counts."""
        active = {"%s|%s|%s" % (c["object"], c["kind"], c.get("current_region")): c for c in candidates}
        surfaced: list[dict[str, Any]] = []
        for k in set(self._streak) | set(self._idle) | set(self._counted) | set(active):
            if k in active:
                self._streak[k] = self._streak.get(k, 0) + 1
                self._idle.pop(k, None)
                if self._streak[k] >= _DEBOUNCE:
                    surfaced.append(active[k])
                    if k not in self._counted:
                        self._flagged_total += 1
                        self._counted.add(k)
            else:
                self._streak.pop(k, None)
                self._idle[k] = self._idle.get(k, 0) + 1
                if self._idle[k] >= _DEBOUNCE:
                    self._counted.discard(k)
                    self._idle.pop(k, None)
        return surfaced

    def _prune(self) -> None:
        """Evict never-learned, long-unseen labels (YOLO long-tail) and enforce a hard cap."""
        stale = [lbl for lbl, on in self._objects.items()
                 if (not on.has_normal()) and (self._tick - on.last_tick) > _STALE_TICKS]
        for lbl in stale:
            del self._objects[lbl]
        if len(self._objects) > _MAX_OBJECTS:
            ranked = sorted(self._objects.items(),
                            key=lambda kv: (kv[1].has_normal(), kv[1].seen_ticks), reverse=True)
            self._objects = dict(ranked[:_MAX_OBJECTS])

    def get_observations(self) -> list[dict[str, Any]]:
        return list(self._last_observations)

    # --- status -----------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        normals = [on for on in self._objects.values() if on.has_normal()]
        normals.sort(key=lambda o: o.seen_ticks, reverse=True)
        learned = []
        for on in normals[:_MAX_NORMALS_OUT]:
            region, frac = on.dominant()
            learned.append({
                "object": on.label,
                "usual_region": region,
                "usual_region_frac": round(frac, 2),
                "observations": on.seen_ticks,
                "present_frac": round(on.present_frac(self._tick), 2),
            })
        return {
            "phase": "environmental_memory_of_normal",
            "lane": "be_there_for_the_room",
            "claim_kinds": ["moved"],
            "claims_absence": False,
            "uses_person_data": False,
            "spoken": False,
            "writes_belief": False,
            "changes_behavior": False,
            "authority": "shadow_logged_only",
            "note": ("shadow, data-minimal — learns each object's usual spot from what the "
                     "perception tracker still BELIEVES EXISTS (permanence, not just freshly "
                     "detected), then would gently note only a real, debounced 'moved' sighting "
                     "('the cup isn't where it usually lives'). It deliberately does NOT claim a "
                     "thing is gone/missing (that would need person-occlusion data we choose not "
                     "to wire). Complements (does not touch) the novel-object curiosity ask and "
                     "the PRE-MATURE hrr_scene mind's-eye."),
            "tick_interval_s": 60,
            "last_load_ok": self._last_load_ok,
            "maturity": "thin_pending_perception",
            "limitations": [
                "no person/occlusion geometry in the pipeline -> undetected objects decay fast, "
                "so permanence bridges only short gaps; sparsely-seen movable objects learn slowly "
                "and this lane may legitimately stay silent (honest under-firing, fails safe).",
                "accrual samples the tracker once per ~60s (the tracker updates ~every 3s) -> coarse.",
                "regions are coarse thirds with no hysteresis -> a confident sighting is required for "
                "'moved', but a near-boundary object could still occasionally read as moved.",
            ],
            "metrics": {
                "ticks_observed": self._tick,
                "objects_tracked": len(self._objects),
                "objects_with_learned_normal": len(normals),
                "moved_noticed_total": self._flagged_total,
                "current_moved": len(self._last_observations),
                "min_observations_for_normal": _MIN_OBS,
                "min_dominance_for_normal": _MIN_DOMINANCE,
                "deviation_debounce_ticks": _DEBOUNCE,
            },
            "learned_normals": learned,
            "observations": self.get_observations(),
        }

    # --- persistence ------------------------------------------------------
    def _save(self) -> None:
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            payload = {
                "version": _STATE_VERSION,
                "tick": self._tick,
                "flagged_total": self._flagged_total,
                "counted": sorted(self._counted),
                "objects": {label: on.to_dict() for label, on in self._objects.items()},
            }
            tmp = _STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, _STATE_PATH)
        except Exception:
            logger.debug("environmental_normal save failed", exc_info=True)

    def _load(self) -> None:
        try:
            if not os.path.exists(_STATE_PATH):
                return
            with open(_STATE_PATH) as f:
                payload = json.load(f)
            try:
                version = int(payload.get("version", 1))
            except (TypeError, ValueError):
                version = 1
            if version != _STATE_VERSION:
                # schema/semantics changed — start fresh rather than carry forward normals
                # learned under different rules. Back up the old file.
                logger.warning(
                    "environmental_normal: state version %s != %s — starting fresh (backing up old file)",
                    version, _STATE_VERSION,
                )
                try:
                    os.replace(_STATE_PATH, _STATE_PATH + ".v%s.bak" % version)
                except Exception:
                    pass
                return
            self._tick = int(payload.get("tick", 0))
            self._flagged_total = int(payload.get("flagged_total", 0))
            self._counted = set(payload.get("counted") or [])
            # Seed streaks so a still-standing deviation re-surfaces on the first post-restart
            # tick (no one-tick "all clear" flicker); a no-longer-active one ages out via idle.
            self._streak = {k: _DEBOUNCE for k in self._counted}
            self._objects = {
                label: ObjectNormal.from_dict(d)
                for label, d in (payload.get("objects") or {}).items()
            }
        except Exception:
            self._last_load_ok = False
            logger.warning("environmental_normal load failed — starting fresh", exc_info=True)


environmental_normal_engine = EnvironmentalNormalEngine()
