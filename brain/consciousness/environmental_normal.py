"""
Environmental memory-of-normal — the "be there for the room" half of presence.

Shadow-only. Learns, by quiet observation over time, where each object USUALLY
lives in the room (its dominant region) and how often it is usually present.
Then it notices when an object is CLEARLY out of its usual spot and logs the
gentle thing JARVIS *would* note ("the cup isn't in its usual spot — it usually
sits on the desk"). It is NEVER spoken, it is a hypothesis, and it is salience-
gated to a real, confident deviation (not noise).

This is the dignity-anchor's environmental half (north star). For someone whose
memory is fading, "your keys aren't where they live" is exactly the gentle,
dignity-preserving nudge — but ONLY if JARVIS actually KNOWS where they live,
from a real learned normal. A guessed "usual spot" would be the betrayal we are
avoiding. So a normal is only trusted after enough observations in a dominant
region; below that bar JARVIS stays silent.

Complements, and does NOT touch:
  - the novel-object curiosity ask (that lane = "new thing in the room, what is
    it?"); this lane = "a KNOWN thing has moved from where it usually lives".
  - the PRE-MATURE hrr_scene spatial mind's-eye (the heavy spatial graph). This
    is a lightweight per-label accumulator riding the existing scene entities.

Fed once per scene-continuity tick (~60 s) via observe_scene(scene_state).
Pure stdlib; persisted to ~/.jarvis/environmental_normal.json so the learned
normal accrues across days and survives restarts.
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
_MIN_OBS = 15            # ticks an object must be seen before it can have a "usual spot"
_MIN_DOMINANCE = 0.70    # fraction of seen-ticks in one region to call it the usual spot
_MIN_PRESENCE = 0.60     # fraction of life-ticks present to flag a usually-present object as missing
_CONF_FLOOR = 0.35       # ignore low-confidence detections (noise)
_EXCLUDE_REGIONS = {"unknown"}  # cannot speak of a "usual spot" that is itself unknown
_SAVE_EVERY = 5          # persist every N observe ticks
_MAX_NORMALS_OUT = 10    # cap learned-normals surfaced in status

_STATE_DIR = os.path.expanduser("~/.jarvis")
_STATE_PATH = os.path.join(_STATE_DIR, "environmental_normal.json")


def _human(region: str) -> str:
    return (region or "unknown").replace("_", " ")


@dataclass
class ObjectNormal:
    """The learned usual layout of one labelled object."""
    label: str
    region_counts: dict[str, int] = field(default_factory=dict)
    seen_ticks: int = 0
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
        life = max(1, current_tick - self.first_tick + 1)
        return self.seen_ticks / life

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
    """Shadow memory-of-normal: learns each object's usual spot, would-notes real deviations."""

    def __init__(self) -> None:
        self._objects: dict[str, ObjectNormal] = {}
        self._tick: int = 0
        self._flagged_total: int = 0
        self._last_visible: dict[str, str] = {}       # label -> current region (this tick)
        self._last_observations: list[dict[str, Any]] = []
        self._prev_flagged: set[str] = set()           # labels flagged last tick (for distinct counting)
        self._load()

    # --- observation ------------------------------------------------------
    def observe_scene(self, scene_state: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Fold one scene snapshot into the learned normal. Returns current would-notes (shadow)."""
        if not isinstance(scene_state, dict):
            return list(self._last_observations)
        self._tick += 1

        # One representative region per visible, confident, non-display label.
        per_label: dict[str, tuple[str, float]] = {}
        for e in (scene_state.get("entities") or []):
            if not isinstance(e, dict):
                continue
            if e.get("is_display_surface"):
                continue
            if e.get("state") != "visible":
                continue
            conf = float(e.get("confidence") or 0.0)
            if conf < _CONF_FLOOR:
                continue
            label = e.get("label")
            if not label:
                continue
            region = e.get("region") or "unknown"
            cur = per_label.get(label)
            if cur is None or conf > cur[1]:
                per_label[label] = (region, conf)

        now = time.time()
        for label, (region, _conf) in per_label.items():
            on = self._objects.get(label)
            if on is None:
                on = ObjectNormal(label=label, first_tick=self._tick)
                self._objects[label] = on
            on.seen_ticks += 1
            on.region_counts[region] = on.region_counts.get(region, 0) + 1
            on.last_region = region
            on.last_tick = self._tick
            on.last_updated = now

        self._last_visible = {label: region for label, (region, _c) in per_label.items()}
        self._last_observations = self._compute_observations()

        # Count distinct, newly-appearing deviations (not every tick they persist).
        flagged_now = {o["object"] for o in self._last_observations}
        for label in flagged_now - self._prev_flagged:
            self._flagged_total += 1
        self._prev_flagged = flagged_now

        if self._tick % _SAVE_EVERY == 0:
            self._save()
        return list(self._last_observations)

    def _compute_observations(self) -> list[dict[str, Any]]:
        """Read-only: which known objects are clearly out of their usual spot right now."""
        out: list[dict[str, Any]] = []
        for label, on in self._objects.items():
            if not on.has_normal():
                continue
            dom_region, dom_frac = on.dominant()
            cur_region = self._last_visible.get(label)
            note = None
            kind = None
            if cur_region is not None:
                if cur_region != dom_region and cur_region not in _EXCLUDE_REGIONS:
                    note = ("the %s isn't in its usual spot — it usually sits in the %s, "
                            "but right now it's in the %s"
                            % (label, _human(dom_region), _human(cur_region)))
                    kind = "moved"
            else:
                # Missing: only if it is usually reliably present.
                if on.presence_frac(self._tick) >= _MIN_PRESENCE:
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
                "presence_frac": round(on.presence_frac(self._tick), 2),
            })
        return {
            "phase": "environmental_memory_of_normal",
            "lane": "be_there_for_the_room",
            "spoken": False,
            "writes_belief": False,
            "changes_behavior": False,
            "authority": "shadow_logged_only",
            "note": ("shadow — learns each object's usual spot by quiet observation, then would "
                     "gently note a real, confident deviation ('the cup isn't where it usually "
                     "lives'). Complements (does not touch) the novel-object curiosity ask and "
                     "the PRE-MATURE hrr_scene mind's-eye."),
            "tick_interval_s": 60,
            "metrics": {
                "ticks_observed": self._tick,
                "objects_observed": len(self._objects),
                "objects_with_learned_normal": len(normals),
                "deviations_noticed_total": self._flagged_total,
                "current_deviations": len(self._last_observations),
                "min_observations_for_normal": _MIN_OBS,
                "min_dominance_for_normal": _MIN_DOMINANCE,
            },
            "learned_normals": learned,
            "observations": self.get_observations(),
        }

    # --- persistence ------------------------------------------------------
    def _save(self) -> None:
        try:
            os.makedirs(_STATE_DIR, exist_ok=True)
            payload = {
                "tick": self._tick,
                "flagged_total": self._flagged_total,
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
            self._tick = int(payload.get("tick", 0))
            self._flagged_total = int(payload.get("flagged_total", 0))
            self._objects = {
                label: ObjectNormal.from_dict(d)
                for label, d in (payload.get("objects") or {}).items()
            }
        except Exception:
            logger.debug("environmental_normal load failed", exc_info=True)


environmental_normal_engine = EnvironmentalNormalEngine()
