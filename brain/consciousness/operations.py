"""Operational Flow Tracker — live view of what Jarvis is doing right now.

Singleton that tracks:
  - subsystems: per-subsystem status board (hot-path writes)
  - timeline: rolling event log (last ~200 entries)
  - interactive_path: conversation pipeline stages (wake→playback)
  - stack: active execution breadcrumb

The synthesize_v2() function (cold-path only) transforms the raw snapshot
into a truthful operator-facing view with priority-derived current state,
normalized status enums, and background/interactive split.

Thread-safe, O(1) hot-path writes. Dashboard reads a snapshot copy.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any

_TIMELINE_CAP = 200
_STACK_CAP = 10
_USER_FACING_STALE_S = 300.0

# ── Standardized status enum ────────────────────────────────────────

OP_ACTIVE = "active"
OP_WAITING = "waiting"
OP_QUEUED = "queued"
OP_BLOCKED = "blocked"
OP_IDLE = "idle"
OP_DONE = "done"
OP_COMPLETED = "completed"
OP_ERROR = "error"

_ACTIVE_STATUSES = frozenset({OP_ACTIVE, OP_WAITING, OP_QUEUED, OP_BLOCKED})

_STATUS_NORM: dict[str, str] = {
    "listening": OP_ACTIVE,
    "transcribing": OP_ACTIVE,
    "searching": OP_ACTIVE,
    "processing": OP_ACTIVE,
    "generating": OP_ACTIVE,
    "synthesizing": OP_ACTIVE,
    "playing": OP_ACTIVE,
    "running": OP_ACTIVE,
    "cooldown": OP_WAITING,
    "queued": OP_QUEUED,
    "waiting": OP_WAITING,
    "idle": OP_IDLE,
    "": OP_IDLE,
    "blocked": OP_BLOCKED,
    "disabled": OP_BLOCKED,
    "error": OP_ERROR,
    "done": OP_DONE,
    "completed": OP_COMPLETED,
}

# ── Subsystem metadata ──────────────────────────────────────────────

SUBSYSTEM_NAMES = (
    "perception", "stt", "retrieval", "reasoning", "speech", "memory",
    "cortex_training", "mutation", "autonomy", "policy",
    "hemisphere", "evolution", "existential", "dream",
    "self_improve", "learning_jobs", "study", "goals",
    "scene_continuity",
)

SUBSYSTEM_META: dict[str, dict[str, Any]] = {
    "perception":     {"priority": 100, "user_facing": True,  "label": "Perception"},
    "stt":            {"priority": 95,  "user_facing": True,  "label": "STT"},
    "retrieval":      {"priority": 80,  "user_facing": True,  "label": "Memory Retrieval"},
    "reasoning":      {"priority": 75,  "user_facing": True,  "label": "Reasoning"},
    "speech":         {"priority": 70,  "user_facing": True,  "label": "Speech"},
    "cortex_training": {"priority": 60, "user_facing": False, "label": "Cortex Training"},
    "policy":         {"priority": 50,  "user_facing": False, "label": "Policy"},
    "autonomy":       {"priority": 45,  "user_facing": False, "label": "Autonomy"},
    "memory":         {"priority": 40,  "user_facing": False, "label": "Memory"},
    "dream":          {"priority": 35,  "user_facing": False, "label": "Dream"},
    "mutation":       {"priority": 34,  "user_facing": False, "label": "Mutation"},
    "hemisphere":     {"priority": 33,  "user_facing": False, "label": "Hemisphere"},
    "evolution":      {"priority": 32,  "user_facing": False, "label": "Evolution"},
    "existential":    {"priority": 31,  "user_facing": False, "label": "Existential"},
    "self_improve":   {"priority": 30,  "user_facing": False, "label": "Self-Improve"},
    "learning_jobs":  {"priority": 29,  "user_facing": False, "label": "Learning Jobs"},
    "study":          {"priority": 28,  "user_facing": False, "label": "Study"},
    "goals":          {"priority": 27,  "user_facing": False, "label": "Goals"},
    "scene_continuity": {"priority": 26, "user_facing": False, "label": "Scene Continuity"},
}

_HUMAN_LABELS: dict[tuple[str, str], str] = {
    ("perception", "listening"):     "Listening for speech",
    ("perception", "transcribing"):  "Transcribing speech",
    ("stt", "transcribing"):         "Running speech-to-text",
    ("reasoning", "processing"):     "Processing request",
    ("reasoning", "generating"):     "Generating response",
    ("retrieval", "searching"):      "Searching memory",
    ("speech", "synthesizing"):      "Speaking",
    ("speech", "playing"):           "Playing audio",
    ("cortex_training", "running"):  "Training cortex models",
    ("cortex_training", "waiting"):  "Waiting for training data",
    ("policy", "running"):           "Training policy network",
    ("autonomy", "running"):         "Running autonomous research",
    ("dream", "running"):            "Dream consolidation",
    ("mutation", "running"):         "Evaluating mutations",
    ("hemisphere", "running"):       "Training hemisphere networks",
    ("evolution", "running"):        "Running evolution cycle",
    ("study", "running"):            "Studying documents",
    ("self_improve", "running"):     "Running self-improvement",
    ("memory", "running"):           "Memory maintenance",
}

# ── Interactive path (conversation pipeline) ─────────────────────────

_INTERACTIVE_STAGES = ("wake", "listen", "stt", "route", "reason", "tts", "playback")

_STAGE_LABELS: dict[str, str] = {
    "wake": "Wake",
    "listen": "Listen",
    "stt": "STT",
    "route": "Route",
    "reason": "Reason",
    "tts": "TTS",
    "playback": "Playback",
}


@dataclass
class StageState:
    status: str = OP_IDLE
    detail: str = ""
    ts: float = 0.0


# ── Core dataclasses ─────────────────────────────────────────────────

@dataclass
class Activity:
    name: str = "idle"
    phase: str = ""
    status: str = "idle"
    started_at: float = 0.0
    trigger: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = round(time.time() - self.started_at, 1) if self.started_at else 0.0
        return d


@dataclass
class SubsystemStatus:
    status: str = "idle"
    detail: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "age_s": round(time.time() - self.updated_at, 1) if self.updated_at else 0.0,
        }


@dataclass
class TimelineEntry:
    ts: float
    subsystem: str
    event: str
    msg: str

    def to_dict(self) -> dict[str, Any]:
        return {"ts": round(self.ts, 3), "subsystem": self.subsystem,
                "event": self.event, "msg": self.msg}


# ── OperationsTracker (singleton, hot-path writes) ──────────────────

class OperationsTracker:
    _instance: OperationsTracker | None = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = Activity()
        self._stack: list[str] = []
        self._subsystems: dict[str, SubsystemStatus] = {
            name: SubsystemStatus() for name in SUBSYSTEM_NAMES
        }
        self._timeline: deque[TimelineEntry] = deque(maxlen=_TIMELINE_CAP)
        self._boot_ts = time.time()
        self._interactive_path: dict[str, StageState] = {
            stage: StageState() for stage in _INTERACTIVE_STAGES
        }
        self._conversation_id: str = ""

    @classmethod
    def get_instance(cls) -> OperationsTracker:
        if cls._instance is None:
            cls._instance = OperationsTracker()
        return cls._instance

    # ── Hot-path writes ──────────────────────────────────────────────

    def begin_activity(self, name: str, phase: str = "", trigger: str = "scheduler",
                       detail: str = "") -> None:
        now = time.time()
        with self._lock:
            self._current = Activity(
                name=name, phase=phase, status="running",
                started_at=now, trigger=trigger, detail=detail,
            )
            if name not in self._stack:
                self._stack.append(name)
                if len(self._stack) > _STACK_CAP:
                    self._stack = self._stack[-_STACK_CAP:]
            self._timeline.append(TimelineEntry(
                ts=now, subsystem=name, event="started",
                msg=detail or f"{name} started",
            ))

    def end_activity(self, name: str, detail: str = "") -> None:
        now = time.time()
        with self._lock:
            if self._current.name == name:
                self._current = Activity()
            if name in self._stack:
                self._stack.remove(name)
            self._timeline.append(TimelineEntry(
                ts=now, subsystem=name, event="completed",
                msg=detail or f"{name} completed",
            ))

    def skip_activity(self, name: str, reason: str) -> None:
        now = time.time()
        with self._lock:
            self._timeline.append(TimelineEntry(
                ts=now, subsystem=name, event="skipped",
                msg=reason,
            ))

    def set_subsystem(self, name: str, status: str, detail: str = "") -> None:
        now = time.time()
        with self._lock:
            ss = self._subsystems.get(name)
            if ss:
                ss.status = status
                ss.detail = detail
                ss.updated_at = now

    def log_event(self, subsystem: str, event: str, msg: str) -> None:
        now = time.time()
        with self._lock:
            self._timeline.append(TimelineEntry(
                ts=now, subsystem=subsystem, event=event, msg=msg,
            ))

    # ── Interactive path (conversation pipeline) ─────────────────────

    def advance_stage(self, key: str, status: str = OP_ACTIVE,
                      detail: str = "", conversation_id: str = "") -> None:
        """Advance the interactive pipeline to a given stage.

        All stages before *key* are auto-marked 'done'.
        All stages after *key* are reset to 'idle'.
        """
        now = time.time()
        with self._lock:
            if conversation_id:
                self._conversation_id = conversation_id
            found = False
            for stage_key in _INTERACTIVE_STAGES:
                ss = self._interactive_path[stage_key]
                if stage_key == key:
                    found = True
                    ss.status = status
                    ss.detail = detail
                    ss.ts = now
                elif not found:
                    if ss.status != OP_DONE:
                        ss.status = OP_DONE
                        if not ss.ts:
                            ss.ts = now
                else:
                    ss.status = OP_IDLE
                    ss.detail = ""

    def reset_interactive_path(self) -> None:
        with self._lock:
            for ss in self._interactive_path.values():
                ss.status = OP_IDLE
                ss.detail = ""
                ss.ts = 0.0
            self._conversation_id = ""

    # ── Cold-path snapshot ───────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            current = Activity(
                name=self._current.name, phase=self._current.phase,
                status=self._current.status, started_at=self._current.started_at,
                trigger=self._current.trigger, detail=self._current.detail,
            )
            stack = list(self._stack)
            subsystems = {
                k: SubsystemStatus(s.status, s.detail, s.updated_at)
                for k, s in self._subsystems.items()
            }
            timeline = list(self._timeline)
            path = {
                k: StageState(s.status, s.detail, s.ts)
                for k, s in self._interactive_path.items()
            }
            conv_id = self._conversation_id

        return {
            "current": current.to_dict(),
            "stack": stack,
            "subsystems": {k: v.to_dict() for k, v in subsystems.items()},
            "timeline": [e.to_dict() for e in timeline[-50:]],
            "interactive_path": {
                "conversation_id": conv_id,
                "stages": [
                    {
                        "key": k,
                        "label": _STAGE_LABELS.get(k, k),
                        "status": path[k].status,
                        "detail": path[k].detail,
                        "ts": round(path[k].ts, 3),
                    }
                    for k in _INTERACTIVE_STAGES
                ],
            },
            "boot_ts": round(self._boot_ts, 3),
        }


ops_tracker = OperationsTracker.get_instance()


# ═════════════════════════════════════════════════════════════════════
# Cold-path synthesis — transforms raw snapshot into v2 operator view
# ═════════════════════════════════════════════════════════════════════

def _normalize_status(raw: str) -> str:
    return _STATUS_NORM.get(raw, raw if raw in _STATUS_NORM.values() else OP_IDLE)


def _human_label(subsystem: str, raw_status: str) -> str:
    label = _HUMAN_LABELS.get((subsystem, raw_status))
    if label:
        return label
    meta = SUBSYSTEM_META.get(subsystem)
    return meta["label"] if meta else subsystem.replace("_", " ").title()


def _clear_stale_user_facing(name: str, ss: dict[str, Any]) -> None:
    """Cold-path display guard for abandoned hot-path perception/STT states."""
    if not ss.get("user_facing"):
        return
    if ss.get("status") not in _ACTIVE_STATUSES:
        return
    age_s = float(ss.get("age_s", 0.0) or 0.0)
    if age_s < _USER_FACING_STALE_S:
        return
    ss["status"] = OP_IDLE
    ss["raw_status"] = OP_IDLE
    ss["detail"] = f"Stale {name} activity cleared after {age_s:.0f}s without an update."
    ss["stale_cleared"] = True


def _clear_stale_interactive_path(path: dict[str, Any], now: float) -> dict[str, Any]:
    stages = []
    for stage in path.get("stages", []):
        item = dict(stage)
        status = _normalize_status(item.get("status", OP_IDLE))
        ts = float(item.get("ts", 0.0) or 0.0)
        if status in _ACTIVE_STATUSES and ts and now - ts >= _USER_FACING_STALE_S:
            item["status"] = OP_IDLE
            item["detail"] = f"Stale stage cleared after {now - ts:.0f}s."
            item["stale_cleared"] = True
        stages.append(item)
    return {
        "conversation_id": path.get("conversation_id", ""),
        "stages": stages,
    }


def _enrich_subsystem_from_context(
    name: str,
    ss: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """Inject richer detail for subsystems that have external state."""
    if name == "policy":
        pol = context.get("policy", {})
        if pol.get("active"):
            mode = pol.get("mode", "shadow")
            arch = pol.get("arch", pol.get("registry_active_arch", ""))
            train_runs = pol.get("train_runs_total", 0)
            shadow_total = pol.get("shadow_ab_total", 0)
            nn_win_rate = pol.get("nn_win_rate", 0)
            if pol.get("auto_disabled"):
                ss["status"] = "disabled"
                ss["detail"] = f"Auto-disabled: {pol.get('last_block_reason', '')}"
            else:
                parts = [f"{mode}"]
                if arch:
                    parts.append(arch)
                if shadow_total:
                    parts.append(f"{shadow_total} evals")
                if nn_win_rate and shadow_total:
                    parts.append(f"win={nn_win_rate:.0%}")
                if train_runs:
                    parts.append(f"{train_runs} trains")
                ss["detail"] = ", ".join(parts)
                if mode != "shadow":
                    ss["status"] = "active"

    elif name == "autonomy":
        aut = context.get("autonomy", {})
        if aut.get("enabled") and aut.get("started"):
            q_size = aut.get("queue_size", 0)
            level = aut.get("autonomy_level_name", "L0")
            if q_size > 0:
                ss["status"] = "queued"
                ss["detail"] = f"{q_size} queued, level {level}"
            else:
                ss["detail"] = f"Level {level}, idle"

    elif name == "cortex_training":
        mc = context.get("memory_cortex", {})
        ts = mc.get("training_status", {})
        ranker_reason = ts.get("ranker_skip_reason", "")
        salience_reason = ts.get("salience_skip_reason", "")
        eval_metrics = mc.get("eval_metrics", {})
        r_pairs = eval_metrics.get("training_pairs_available", 0)
        r_min = eval_metrics.get("training_pairs_min_required", 50)
        salience_metrics = mc.get("salience_metrics", {})
        s_pairs = salience_metrics.get("validated_predictions", 0)
        if not s_pairs and salience_reason:
            m = re.search(r"have (\d+)", salience_reason)
            s_pairs = int(m.group(1)) if m else 0
        detail_parts = []
        detail_parts.append(f"Ranker {r_pairs}/{r_min}")
        detail_parts.append(f"Salience {s_pairs}/100")
        ss["detail"] = ", ".join(detail_parts)
        if ranker_reason or salience_reason:
            ss["status"] = "waiting"

    elif name == "self_improve":
        si = context.get("self_improve", {})
        if not si.get("active", False):
            status_str = si.get("status", "disabled")
            ss["detail"] = f"Frozen / {status_str}"


def synthesize_v2(raw: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Transform a raw ops_tracker.snapshot() into the v2 operator contract.

    Called on the cold path by the dashboard _build_snapshot(), never on hot path.

    Args:
        raw: output of ops_tracker.snapshot()
        context: dict with keys: phase, mode, policy, autonomy, memory_cortex, self_improve
    """
    now = time.time()
    raw_subsystems = raw.get("subsystems", {})

    # 1. Build enriched + normalized subsystem dicts
    subsystems: dict[str, dict[str, Any]] = {}
    for name in SUBSYSTEM_NAMES:
        ss_raw = raw_subsystems.get(name, {"status": "idle", "detail": "", "age_s": 0})
        meta = SUBSYSTEM_META.get(name, {"priority": 0, "user_facing": False, "label": name})
        raw_status = ss_raw.get("status", "idle")
        normalized = _normalize_status(raw_status)

        ss = {
            "status": normalized,
            "raw_status": raw_status,
            "detail": ss_raw.get("detail", ""),
            "age_s": ss_raw.get("age_s", 0),
            "priority": meta["priority"],
            "user_facing": meta["user_facing"],
            "label": meta["label"],
        }
        _enrich_subsystem_from_context(name, ss, context)
        # Re-normalize after enrichment may have changed status
        ss["status"] = _normalize_status(ss["status"])
        _clear_stale_user_facing(name, ss)
        subsystems[name] = ss

    # 2. Derive operations.current from highest-priority non-idle subsystem
    #    Priority order: active > queued > blocked(user-facing) > waiting(fallback)
    #    "waiting" background items should not win over true idle when nothing
    #    is actively executing — the hero card should say "Standing By", not
    #    "Cortex Training: waiting for data".
    _STATUS_RANK = {OP_ACTIVE: 0, OP_QUEUED: 1, OP_BLOCKED: 2, OP_WAITING: 3}
    candidates = [
        (name, ss) for name, ss in subsystems.items()
        if ss["status"] in _ACTIVE_STATUSES
    ]
    if candidates:
        candidates.sort(key=lambda x: (
            _STATUS_RANK.get(x[1]["status"], 9),
            0 if x[1]["user_facing"] else 1,
            -x[1]["priority"],
            x[1]["age_s"],
        ))
        best_name, best_ss = candidates[0]
        # If the only candidates are background "waiting" items, treat as idle
        only_bg_waiting = all(
            not c[1]["user_facing"] and c[1]["status"] == OP_WAITING
            for c in candidates
        )
        if only_bg_waiting:
            # Show idle with a background note
            bg_note = ", ".join(
                f"{c[0]}: {c[1]['detail']}" for c in candidates[:2] if c[1]["detail"]
            )
            current = {
                "name": "idle",
                "label": "Standing By",
                "status": OP_IDLE,
                "phase": "",
                "detail": bg_note or "Background systems waiting",
                "started_at": 0,
                "duration_s": 0,
                "priority": 0,
                "user_facing": False,
                "blocking": False,
            }
        else:
            raw_status = best_ss.get("raw_status", best_ss["status"])
            current = {
                "name": best_name,
                "label": _human_label(best_name, raw_status),
                "status": best_ss["status"],
                "phase": raw_status,
                "detail": best_ss["detail"],
                "started_at": round(now - best_ss["age_s"], 3) if best_ss["age_s"] else 0,
                "duration_s": round(best_ss["age_s"], 1),
                "priority": best_ss["priority"],
                "user_facing": best_ss["user_facing"],
                "blocking": False,
            }
    else:
        # Fall back to freshest timeline event
        timeline = raw.get("timeline", [])
        if timeline:
            last = timeline[-1]
            ss_name = last.get("subsystem", "idle")
            meta = SUBSYSTEM_META.get(ss_name, {"priority": 0, "user_facing": False, "label": ss_name})
            age = round(now - last.get("ts", now), 1)
            current = {
                "name": ss_name,
                "label": f"Recent: {last.get('msg', '')}",
                "status": OP_IDLE,
                "phase": "",
                "detail": last.get("msg", ""),
                "started_at": last.get("ts", 0),
                "duration_s": age,
                "priority": meta["priority"],
                "user_facing": meta["user_facing"],
                "blocking": False,
            }
        else:
            current = {
                "name": "idle",
                "label": "Standing By",
                "status": OP_IDLE,
                "phase": "",
                "detail": "",
                "started_at": 0,
                "duration_s": 0,
                "priority": 0,
                "user_facing": False,
                "blocking": False,
            }

    # 3. Build stack
    phase = context.get("phase", "IDLE")
    mode = context.get("mode", "passive")
    stack = [
        {"layer": "core", "value": phase},
        {"layer": "mode", "value": mode},
    ]
    if current["name"] != "idle":
        active_val = current["name"]
        if current["phase"]:
            active_val += f":{current['phase']}"
        stack.append({"layer": "active", "value": active_val})
    # Add any background items that are active
    for name, ss in subsystems.items():
        if not ss["user_facing"] and ss["status"] in _ACTIVE_STATUSES:
            stack.append({"layer": "background", "value": name})
    if len(stack) > _STACK_CAP:
        stack = stack[:_STACK_CAP]

    # 4. Split background items
    bg_items = []
    bg_active = 0
    for name, ss in subsystems.items():
        if ss["user_facing"]:
            continue
        bg_item = {
            "key": name,
            "label": ss["label"],
            "status": ss["status"],
            "detail": ss["detail"],
            "age_s": ss["age_s"],
        }
        bg_items.append(bg_item)
        if ss["status"] in _ACTIVE_STATUSES:
            bg_active += 1

    # 5. Assemble v2 output
    return {
        "version": 2,
        "updated_at": round(now, 3),
        "current": current,
        "stack": stack,
        "interactive_path": _clear_stale_interactive_path(raw.get("interactive_path", {
            "conversation_id": "",
            "stages": [
                {"key": k, "label": _STAGE_LABELS.get(k, k), "status": OP_IDLE, "detail": "", "ts": 0}
                for k in _INTERACTIVE_STAGES
            ],
        }), now),
        "background": {
            "active_count": bg_active,
            "items": bg_items,
        },
        "subsystems": subsystems,
        "timeline": raw.get("timeline", []),
        "boot_ts": raw.get("boot_ts", 0),
    }
