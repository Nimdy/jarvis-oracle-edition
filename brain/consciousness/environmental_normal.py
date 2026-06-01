"""
Environmental memory-of-normal — the "be there for the room" half of presence.

Shadow-only. Learns, by quiet observation over time, where each object USUALLY
lives in the room (its dominant region) and how reliably it is present. Then it
notices when a known object is CLEARLY out of its usual spot, or genuinely gone,
and logs the gentle thing JARVIS *would* note ("the cup isn't in its usual spot
— it usually sits on the desk"). It is NEVER spoken, it is a hypothesis, and it
is salience-gated (a confident learned normal + a debounced, persistent shift).

This is the dignity-anchor's environmental half (north star). For someone whose
memory is fading, "your keys aren't where they live" is exactly the gentle,
dignity-preserving nudge — but ONLY if JARVIS actually KNOWS, from a real learned
normal AND from what the perception layer actually BELIEVES. A guessed location,
or "I don't see the cup" when the cup is merely behind your arm, would be the
betrayal we exist to avoid.

THE KEY DISCIPLINE (learned from an adversarial review): the scene tracker
already distinguishes what is genuinely here from what is merely not-detected-
this-frame. Each entity carries a `state` (visible / occluded / missing /
removed) and a `permanence_confidence`. `occluded` means the tracker is CONFIDENT
the object is still present (e.g. a person is standing in front of it); only
`missing` / `removed` mean it believes the object is gone. So:
  - We LEARN an object's usual spot + presence from what the tracker believes is
    PRESENT (visible OR occluded), not just what is freshly detected — otherwise
    movable objects (the cup/keys) that spend most time occluded never accrue.
  - We assert "moved" only from a FRESH visible detection in a different region.
  - We assert "missing" only when the tracker ITSELF believes the object is gone
    (a missing/removed track), never merely because it dropped out of frame.
  - Deviations must persist for >=2 consecutive ticks (debounce) before they are
    surfaced or counted — a single boundary-jitter or detector blink is not a move.

Complements, and does NOT touch:
  - the novel-object curiosity ask (that lane = "new thing in the room, what is
    it?"); this lane = "a KNOWN thing has moved from / vanished from where it lives".
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
_MIN_PRESENCE = 0.60     # fraction of life-ticks present to call an object "usually present"
_CONF_FLOOR = 0.35       # ignore low-confidence fresh detections (noise)
_PERM_FLOOR = 0.35       # tracker's permanence_confidence floor to count an occluded object as present
_DEBOUNCE = 2            # consecutive ticks a deviation must hold before it is surfaced/counted
_STALE_TICKS = 720       # evict a never-learned object unseen this many ticks (~12h at 60s)
_MAX_OBJECTS = 200       # hard cap on tracked labels (eviction backstop)
_EXCLUDE_REGIONS = {"unknown", "background"}  # not a place an object "lives" on the desk
_PRESENT_STATES = {"visible", "occluded"}      # tracker believes the object is here
_GONE_STATES = {"missing", "removed"}          # tracker believes the object is gone
_SAVE_EVERY = 5          # persist every N observe ticks
_MAX_NORMALS_OUT = 10    # cap learned-normals surfaced in status
_STATE_VERSION = 2       # bumped from the visible-only v1 schema

_STATE_DIR = os.path.expanduser("~/.jarvis")
_STATE_PATH = os.path.join(_STATE_DIR, "environmental_normal.json")


def _human(region: str) -> str:
    return (region or "unknown").replace("_", " ")


@dataclass
class ObjectNormal:
    """The learned usual layout of one labelled object."""
    label: str
    region_counts: dict[str, int] = field(default_factory=dict)
    seen_ticks: int = 0          # ticks the tracker believed this object PRESENT (visible|occluded)
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

    def presence_frac(self, current_tick: int) -> float:
        if current_tick < self.first_tick:   # only possible via a corrupt/edited state file
            return 0.0
        life = max(1, current_tick - self.first_tick + 1)
        return min(1.0, self.seen_ticks / life)

    def has_normal(self) -> bool:
        if self.seen_ticks < _MIN_OBS:
            return False
        region, frac = self.dominant()
        return frac >= _MIN_DOMINANCE and region not in _EXCLUDE_REGIONS

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
    believes is present, would-notes a real, debounced deviation. Never speaks, never
    writes a belief, never changes behavior."""

    def __init__(self) -> None:
        self._objects: dict[str, ObjectNormal] = {}
        self._tick: int = 0
        self._flagged_total: int = 0
        self._counted: set[str] = set()        # "label|kind" deviations already counted (persisted)
        self._dev_streak: dict[str, int] = {}  # "label|kind" -> consecutive-tick count (in-memory)
        self._last_observations: list[dict[str, Any]] = []
        self._last_load_ok: bool = True
        self._load()

    # --- observation ------------------------------------------------------
    def observe_scene(self, scene_state: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Fold one scene snapshot into the learned normal. Returns current would-notes (shadow)."""
        if not isinstance(scene_state, dict):
            return list(self._last_observations)
        self._tick += 1
        now = time.time()

        # Per-label aggregation across (possibly multiple) tracks of the same label.
        present_region: dict[str, tuple[str, float]] = {}  # label -> (region, perm/conf) of best PRESENT track
        fresh_region: dict[str, tuple[str, float]] = {}     # label -> (region, conf) of best FRESH VISIBLE track
        gone_labels: set[str] = set()                        # labels with a missing/removed track
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
            if state in _PRESENT_STATES and perm >= _PERM_FLOOR and region not in _EXCLUDE_REGIONS:
                # Existence-confidence: visible carries detection conf; occluded leans on permanence.
                weight = conf if state == "visible" else perm
                cur = present_region.get(label)
                if cur is None or weight > cur[1]:
                    present_region[label] = (region, weight)
            if state == "visible" and conf >= _CONF_FLOOR and region not in _EXCLUDE_REGIONS:
                cur = fresh_region.get(label)
                if cur is None or conf > cur[1]:
                    fresh_region[label] = (region, conf)
            if state in _GONE_STATES:
                gone_labels.add(label)

        # Accrue the usual spot from what the tracker believes is PRESENT.
        for label, (region, _w) in present_region.items():
            on = self._objects.get(label)
            if on is None:
                on = ObjectNormal(label=label, first_tick=self._tick)
                self._objects[label] = on
            on.seen_ticks += 1
            on.region_counts[region] = on.region_counts.get(region, 0) + 1
            on.last_region = region
            on.last_tick = self._tick
            on.last_updated = now

        self._last_observations = self._debounce(
            self._compute_candidates(present_region, fresh_region, gone_labels)
        )

        if self._tick % _SAVE_EVERY == 0:
            self._prune()
            self._save()
        return list(self._last_observations)

    def _compute_candidates(
        self,
        present_region: dict[str, tuple[str, float]],
        fresh_region: dict[str, tuple[str, float]],
        gone_labels: set[str],
    ) -> list[dict[str, Any]]:
        """Raw (pre-debounce) deviations: which known objects are out of their usual spot,
        or believed gone, RIGHT NOW — using the tracker's authoritative belief."""
        out: list[dict[str, Any]] = []
        for label, on in self._objects.items():
            if not on.has_normal():
                continue
            dom_region, dom_frac = on.dominant()
            present_here = label in present_region
            fresh = fresh_region.get(label)
            note = None
            kind = None
            cur_region = None
            if fresh is not None and fresh[0] != dom_region:
                # MOVED: a fresh visible detection places it in a different real region.
                cur_region = fresh[0]
                note = ("the %s isn't in its usual spot — it usually sits in the %s, "
                        "but right now it's in the %s"
                        % (label, _human(dom_region), _human(cur_region)))
                kind = "moved"
            elif (not present_here) and (label in gone_labels) and on.presence_frac(self._tick) >= _MIN_PRESENCE:
                # MISSING: the tracker itself believes it is gone (missing/removed track,
                # and NOT currently visible/occluded) — and it is usually reliably present.
                note = ("i don't see the %s right now — it usually sits in the %s"
                        % (label, _human(dom_region)))
                kind = "missing"
            if note:
                out.append({
                    "object": label,
                    "would_gently_note": note,
                    "kind": kind,
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
        """Only surface/count a deviation once it has held for >=_DEBOUNCE consecutive ticks.
        Kills single-tick boundary jitter / detector blinks. Counts distinct (label,kind) once,
        and (with persisted _counted) does not re-count a standing deviation across restarts."""
        keys_now = {"%s|%s" % (c["object"], c["kind"]): c for c in candidates}
        # bump streaks for active candidates; drop streaks for cleared ones
        for k in list(self._dev_streak.keys()):
            if k not in keys_now:
                del self._dev_streak[k]
                self._counted.discard(k)   # fully cleared → a future recurrence counts as new
        surfaced: list[dict[str, Any]] = []
        for k, c in keys_now.items():
            self._dev_streak[k] = self._dev_streak.get(k, 0) + 1
            if self._dev_streak[k] >= _DEBOUNCE:
                surfaced.append(c)
                if k not in self._counted:
                    self._flagged_total += 1
                    self._counted.add(k)
        return surfaced

    def _prune(self) -> None:
        """Evict never-learned, long-unseen labels (YOLO long-tail) and enforce a hard cap,
        so _objects + the persisted file + objects_observed don't grow without bound."""
        stale = [lbl for lbl, on in self._objects.items()
                 if (not on.has_normal()) and (self._tick - on.last_tick) > _STALE_TICKS]
        for lbl in stale:
            del self._objects[lbl]
        if len(self._objects) > _MAX_OBJECTS:
            # keep the most-observed; drop the rest (learned normals are never dropped first)
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
                "present_frac": round(on.presence_frac(self._tick), 2),
            })
        return {
            "phase": "environmental_memory_of_normal",
            "lane": "be_there_for_the_room",
            "spoken": False,
            "writes_belief": False,
            "changes_behavior": False,
            "authority": "shadow_logged_only",
            "note": ("shadow — learns each object's usual spot from what the perception tracker "
                     "believes is PRESENT (visible OR occluded, not just freshly detected), then "
                     "would gently note a real, debounced deviation ('the cup isn't where it "
                     "usually lives' / 'i don't see the cup — the tracker believes it's gone'). "
                     "Complements (does not touch) the novel-object curiosity ask and the "
                     "PRE-MATURE hrr_scene mind's-eye."),
            "tick_interval_s": 60,
            "last_load_ok": self._last_load_ok,
            "metrics": {
                "ticks_observed": self._tick,
                "objects_tracked": len(self._objects),
                "objects_with_learned_normal": len(normals),
                "deviations_noticed_total": self._flagged_total,
                "current_deviations": len(self._last_observations),
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
            version = int(payload.get("version", 1))
            if version != _STATE_VERSION:
                # schema changed (e.g. v1 visible-only accrual) — start fresh rather than
                # carry forward normals learned under different semantics. Back up the old file.
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
            self._objects = {
                label: ObjectNormal.from_dict(d)
                for label, d in (payload.get("objects") or {}).items()
            }
        except Exception:
            self._last_load_ok = False
            logger.warning("environmental_normal load failed — starting fresh", exc_info=True)


environmental_normal_engine = EnvironmentalNormalEngine()
