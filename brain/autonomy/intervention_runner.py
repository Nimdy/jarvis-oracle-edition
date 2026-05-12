"""Intervention Runner — Phase 5.2.

Manages the lifecycle of candidate interventions:
  propose → validate → shadow activate → measure → promote/discard

Backlog safety limits:
  - Max 3 active shadow interventions per subsystem
  - Max 10 unresolved interventions globally
  - 1800s cooldown before same deficit family spawns another intervention

Performance: extraction stays lightweight. Shadow activation and evaluation
happen in a separate tick cycle pass, not inline with research.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any

from autonomy.interventions import (
    CandidateIntervention,
    _ALLOWED_TYPES,
    _DEFERRED_TYPES,
)

logger = logging.getLogger("autonomy.intervention_runner")

_PERSISTENCE_PATH = os.path.expanduser("~/.jarvis/interventions.jsonl")
_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_SHADOW_PER_SUBSYSTEM = 3
_MAX_UNRESOLVED_GLOBAL = 10
_DEFICIT_COOLDOWN_S = 1800.0
_DEFAULT_SHADOW_WINDOW_S = 86400.0  # 24h
_MAX_PROPOSED_AGE_S = 3 * 86400.0  # trim stale backlog after 3 days


class InterventionRunner:
    """Shadow-tests candidate interventions and promotes/discards based on evidence."""

    def __init__(self) -> None:
        self._proposed: deque[CandidateIntervention] = deque(maxlen=200)
        self._shadow_active: list[CandidateIntervention] = []
        self._completed: deque[CandidateIntervention] = deque(maxlen=200)
        self._deficit_cooldowns: dict[str, float] = {}
        self._loaded = False

    def propose(self, intervention: CandidateIntervention) -> bool:
        """Validate and add to the proposed queue. Returns True if accepted."""
        if not self._validate(intervention):
            return False

        if intervention.is_no_action:
            intervention.status = "no_action"
            self._completed.append(intervention)
            self._persist(intervention)
            logger.info("No-action intervention recorded: deficit=%s",
                        intervention.trigger_deficit)
            return True

        intervention.status = "proposed"
        self._proposed.append(intervention)
        self._persist(intervention)
        logger.info("Intervention proposed: id=%s type=%s target=%s",
                     intervention.intervention_id, intervention.change_type,
                     intervention.target_subsystem)
        return True

    def activate_shadow(
        self,
        intervention_id: str,
        metrics: dict[str, float] | None = None,
    ) -> bool:
        """Move a proposed intervention into shadow evaluation mode."""
        iv = self._find_proposed(intervention_id)
        if not iv:
            return False

        subsystem_active = sum(
            1 for s in self._shadow_active
            if s.target_subsystem == iv.target_subsystem
        )
        if subsystem_active >= _MAX_SHADOW_PER_SUBSYSTEM:
            logger.warning("Shadow cap reached for subsystem %s (%d/%d)",
                          iv.target_subsystem, subsystem_active, _MAX_SHADOW_PER_SUBSYSTEM)
            return False

        if len(self._shadow_active) >= _MAX_UNRESOLVED_GLOBAL:
            logger.warning("Global shadow cap reached (%d/%d)",
                          len(self._shadow_active), _MAX_UNRESOLVED_GLOBAL)
            return False

        iv.status = "shadow"
        iv.shadow_start = time.time()
        iv.shadow_end = iv.shadow_start + _DEFAULT_SHADOW_WINDOW_S
        if metrics and iv.expected_metric:
            iv.baseline_value = metrics.get(iv.expected_metric, 0.0)
        self._shadow_active.append(iv)
        self._proposed = deque(
            (p for p in self._proposed if p.intervention_id != intervention_id),
            maxlen=200,
        )
        self._deficit_cooldowns[iv.trigger_deficit] = time.time()
        self._persist(iv)
        logger.info("Intervention activated in shadow: id=%s baseline=%.4f metric=%s",
                     iv.intervention_id, iv.baseline_value, iv.expected_metric)
        return True

    def check_shadow_results(
        self,
        metrics: dict[str, float] | None = None,
    ) -> list[CandidateIntervention]:
        """Check shadow windows that have elapsed and measure results.

        Returns list of interventions that completed shadow evaluation.
        ``metrics`` is the current metric snapshot used to compute deltas
        against the baseline captured at shadow activation time.
        """
        now = time.time()
        ready: list[CandidateIntervention] = []
        remaining: list[CandidateIntervention] = []

        for iv in self._shadow_active:
            if now >= iv.shadow_end:
                iv.status = "measured"
                if metrics and iv.expected_metric:
                    current = metrics.get(iv.expected_metric, 0.0)
                    raw_delta = current - iv.baseline_value
                    if iv.expected_direction == "down":
                        iv.measured_delta = -raw_delta
                    else:
                        iv.measured_delta = raw_delta
                    logger.info(
                        "Intervention measured: id=%s metric=%s baseline=%.4f "
                        "current=%.4f delta=%.4f direction=%s",
                        iv.intervention_id, iv.expected_metric,
                        iv.baseline_value, current, iv.measured_delta,
                        iv.expected_direction,
                    )
                ready.append(iv)
            else:
                remaining.append(iv)

        self._shadow_active = remaining
        for iv in ready:
            self._completed.append(iv)
            self._persist(iv)

        return ready

    def promote(self, intervention_id: str) -> bool:
        """Promote a measured intervention to permanent status."""
        iv = self._find_in_completed(intervention_id)
        if not iv:
            return False
        if iv.status not in ("measured", "shadow"):
            return False
        iv.status = "promoted"
        iv.verdict = "promoted"
        self._persist(iv)
        logger.info("Intervention promoted: id=%s delta=%.4f",
                     iv.intervention_id, iv.measured_delta)

        try:
            from autonomy.source_ledger import get_source_ledger
            for sid in iv.source_ids:
                get_source_ledger().record_intervention(sid, promoted=True)
        except Exception:
            pass
        return True

    def discard(self, intervention_id: str, reason: str = "") -> bool:
        """Discard a measured intervention that didn't help."""
        iv = self._find_in_completed(intervention_id)
        if not iv:
            for i, s in enumerate(self._shadow_active):
                if s.intervention_id == intervention_id:
                    iv = self._shadow_active.pop(i)
                    break
        if not iv:
            return False
        iv.status = "discarded"
        iv.verdict = reason or "no_improvement"
        if iv not in self._completed:
            self._completed.append(iv)
        self._persist(iv)
        logger.info("Intervention discarded: id=%s reason=%s",
                     iv.intervention_id, iv.verdict)
        return True

    # -- auto-activation for proposed interventions -------------------------

    def auto_activate_proposed(
        self,
        metrics: dict[str, float] | None = None,
    ) -> int:
        """Try to activate all proposed interventions that pass safety checks.

        Called periodically from the tick cycle. Returns count activated.
        Pre-filters subsystems already at shadow cap to avoid log spam.
        ``metrics`` is passed through to ``activate_shadow`` for baseline capture.
        """
        activated = 0
        now = time.time()

        subsystem_counts: dict[str, int] = {}
        for s in self._shadow_active:
            subsystem_counts[s.target_subsystem] = subsystem_counts.get(s.target_subsystem, 0) + 1

        to_activate: list[str] = []
        for iv in self._proposed:
            if subsystem_counts.get(iv.target_subsystem, 0) >= _MAX_SHADOW_PER_SUBSYSTEM:
                continue
            if len(self._shadow_active) + len(to_activate) >= _MAX_UNRESOLVED_GLOBAL:
                break
            if iv.trigger_deficit in self._deficit_cooldowns:
                last = self._deficit_cooldowns[iv.trigger_deficit]
                if now - last < _DEFICIT_COOLDOWN_S:
                    continue
            to_activate.append(iv.intervention_id)

        for iid in to_activate:
            if self.activate_shadow(iid, metrics=metrics):
                activated += 1
        return activated

    # -- query methods ------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for iv in list(self._proposed) + self._shadow_active + list(self._completed):
            by_status[iv.status] = by_status.get(iv.status, 0) + 1
        return {
            "proposed_count": len(self._proposed),
            "shadow_active_count": len(self._shadow_active),
            "completed_count": len(self._completed),
            "unresolved_count": len(self._proposed) + len(self._shadow_active),
            "global_unresolved_cap": _MAX_UNRESOLVED_GLOBAL,
            "shadow_cap_per_subsystem": _MAX_SHADOW_PER_SUBSYSTEM,
            "by_status": by_status,
        }

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        recent = list(self._completed)[-limit:]
        return [iv.to_dict() for iv in reversed(recent)]

    # -- internals ----------------------------------------------------------

    def _validate(self, iv: CandidateIntervention) -> bool:
        if iv.change_type in _DEFERRED_TYPES:
            logger.warning("Deferred intervention type rejected: %s", iv.change_type)
            return False
        if iv.change_type not in _ALLOWED_TYPES:
            logger.warning("Unknown intervention type rejected: %s", iv.change_type)
            return False

        unresolved = len(self._proposed) + len(self._shadow_active)
        if unresolved >= _MAX_UNRESOLVED_GLOBAL and not iv.is_no_action:
            logger.warning("Global backlog limit reached (%d), rejecting", unresolved)
            return False

        now = time.time()
        if iv.trigger_deficit in self._deficit_cooldowns:
            last = self._deficit_cooldowns[iv.trigger_deficit]
            if now - last < _DEFICIT_COOLDOWN_S and not iv.is_no_action:
                logger.debug("Deficit cooldown active for %s", iv.trigger_deficit)
                return False

        return True

    def _find_proposed(self, intervention_id: str) -> CandidateIntervention | None:
        for iv in self._proposed:
            if iv.intervention_id == intervention_id:
                return iv
        return None

    def _find_in_completed(self, intervention_id: str) -> CandidateIntervention | None:
        for iv in self._completed:
            if iv.intervention_id == intervention_id:
                return iv
        return None

    # -- persistence --------------------------------------------------------

    def _persist(self, iv: CandidateIntervention) -> None:
        try:
            os.makedirs(os.path.dirname(_PERSISTENCE_PATH), exist_ok=True)
            if os.path.exists(_PERSISTENCE_PATH):
                if os.path.getsize(_PERSISTENCE_PATH) > _MAX_FILE_BYTES:
                    rotated = _PERSISTENCE_PATH + ".1"
                    if os.path.exists(rotated):
                        os.remove(rotated)
                    os.rename(_PERSISTENCE_PATH, rotated)
            with open(_PERSISTENCE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(iv.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to persist intervention: %s", exc)

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(_PERSISTENCE_PATH):
            return
        try:
            by_id: dict[str, CandidateIntervention] = {}
            with open(_PERSISTENCE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        iv = CandidateIntervention.from_dict(d)
                        if not iv.intervention_id:
                            continue
                        # Append-only persistence: keep latest record per intervention id.
                        by_id[iv.intervention_id] = iv
                    except Exception:
                        continue

            self._proposed.clear()
            self._shadow_active.clear()
            self._completed.clear()
            self._deficit_cooldowns.clear()

            now = time.time()
            proposed_pool: list[CandidateIntervention] = []
            shadow_pool: list[CandidateIntervention] = []

            for iv in by_id.values():
                if iv.status == "proposed":
                    age_s = now - float(iv.created_at or now)
                    if age_s > _MAX_PROPOSED_AGE_S:
                        iv.status = "expired"
                        iv.verdict = iv.verdict or "stale_proposed_trimmed_on_load"
                        self._completed.append(iv)
                    else:
                        proposed_pool.append(iv)
                elif iv.status == "shadow":
                    if iv.shadow_end and now < iv.shadow_end:
                        shadow_pool.append(iv)
                    else:
                        iv.status = "expired"
                        iv.verdict = iv.verdict or "shadow_elapsed_on_load"
                        self._completed.append(iv)
                else:
                    self._completed.append(iv)

            # Enforce active shadow caps on load so stale files cannot deadlock
            # future proposals after a restart.
            shadow_pool.sort(
                key=lambda iv: float(iv.shadow_start or iv.created_at or 0.0),
                reverse=True,
            )
            subsystem_counts: dict[str, int] = {}
            for iv in shadow_pool:
                if len(self._shadow_active) >= _MAX_UNRESOLVED_GLOBAL:
                    iv.status = "expired"
                    iv.verdict = iv.verdict or "global_cap_trimmed_on_load"
                    self._completed.append(iv)
                    continue
                current = subsystem_counts.get(iv.target_subsystem, 0)
                if current >= _MAX_SHADOW_PER_SUBSYSTEM:
                    iv.status = "expired"
                    iv.verdict = iv.verdict or "subsystem_cap_trimmed_on_load"
                    self._completed.append(iv)
                    continue
                self._shadow_active.append(iv)
                subsystem_counts[iv.target_subsystem] = current + 1

            remaining_capacity = max(0, _MAX_UNRESOLVED_GLOBAL - len(self._shadow_active))
            proposed_pool.sort(
                key=lambda iv: float(iv.created_at or 0.0),
                reverse=True,
            )
            accepted_proposed = proposed_pool[:remaining_capacity]
            trimmed_proposed = proposed_pool[remaining_capacity:]
            self._proposed = deque(accepted_proposed, maxlen=200)
            for iv in trimmed_proposed:
                iv.status = "expired"
                iv.verdict = iv.verdict or "backlog_trimmed_on_load"
                self._completed.append(iv)

            # Seed cooldowns from unresolved + recent completed interventions.
            for iv in list(self._proposed) + list(self._shadow_active):
                if iv.trigger_deficit:
                    ts = float(iv.shadow_start or iv.created_at or now)
                    self._deficit_cooldowns[iv.trigger_deficit] = max(
                        self._deficit_cooldowns.get(iv.trigger_deficit, 0.0), ts,
                    )
            for iv in list(self._completed):
                if not iv.trigger_deficit:
                    continue
                ts = float(iv.shadow_start or iv.created_at or 0.0)
                if ts <= 0.0 or (now - ts) >= _DEFICIT_COOLDOWN_S:
                    continue
                self._deficit_cooldowns[iv.trigger_deficit] = max(
                    self._deficit_cooldowns.get(iv.trigger_deficit, 0.0), ts,
                )

            logger.info("Loaded %d proposed, %d shadow, %d completed interventions",
                        len(self._proposed), len(self._shadow_active), len(self._completed))
        except Exception as exc:
            logger.warning("Failed to load interventions: %s", exc)


# Singleton
_instance: InterventionRunner | None = None


def get_intervention_runner() -> InterventionRunner:
    global _instance
    if _instance is None:
        _instance = InterventionRunner()
        _instance.load()
    return _instance
