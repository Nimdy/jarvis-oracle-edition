"""Event sequence validator — enforces ordering, timing, and conflict rules."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from consciousness.events import (
    KERNEL_PHASE_CHANGE, MEMORY_WRITE,
    MUTATION_APPLIED, MUTATION_REJECTED, MUTATION_ROLLBACK,
    CONSCIOUSNESS_MUTATION_PROPOSED,
    PERCEPTION_TRANSCRIPTION, PERCEPTION_WAKE_WORD,
    PERCEPTION_AUDIO_STREAM_START, PERCEPTION_BARGE_IN,
    CONVERSATION_USER_MESSAGE, CONVERSATION_RESPONSE,
    PERCEPTION_PLAYBACK_COMPLETE, PERCEPTION_AUDIO_STREAM_END,
    RETRY_SCHEDULED, RETRY_EXECUTED, RETRY_EXHAUSTED,
)

logger = logging.getLogger(__name__)

Severity = Literal["warning", "error", "critical"]


@dataclass(frozen=True)
class SequenceRule:
    id: str
    event_type: str
    must_follow: tuple[str, ...] = ()
    max_time_between_s: float = 0.0
    min_time_between_s: float = 0.0
    conflicts_with: tuple[str, ...] = ()
    severity: Severity = "warning"


@dataclass
class Violation:
    rule_id: str
    event_type: str
    violation_type: str  # missing_prerequisite, timing_violation, conflict_detected
    severity: Severity
    message: str
    timestamp: float = field(default_factory=time.time)


SEQUENCE_RULES: tuple[SequenceRule, ...] = (
    SequenceRule(
        id="phase_change_spacing",
        event_type=KERNEL_PHASE_CHANGE,
        min_time_between_s=0.1,
        severity="warning",
    ),
    SequenceRule(
        id="mutation_follows_proposal",
        event_type=MUTATION_APPLIED,
        must_follow=(CONSCIOUSNESS_MUTATION_PROPOSED,),
        max_time_between_s=10.0,
        severity="error",
    ),
    SequenceRule(
        id="rollback_conflicts_applied",
        event_type=MUTATION_ROLLBACK,
        conflicts_with=(MUTATION_APPLIED,),
        severity="critical",
    ),
    SequenceRule(
        id="transcription_follows_audio",
        event_type=PERCEPTION_TRANSCRIPTION,
        must_follow=(PERCEPTION_WAKE_WORD, PERCEPTION_AUDIO_STREAM_START, PERCEPTION_BARGE_IN),
        max_time_between_s=120.0,
        severity="warning",
    ),
    SequenceRule(
        id="playback_follows_response",
        event_type=PERCEPTION_PLAYBACK_COMPLETE,
        must_follow=(CONVERSATION_RESPONSE,),
        max_time_between_s=240.0,
        severity="error",
    ),
    SequenceRule(
        id="audio_end_follows_response",
        event_type=PERCEPTION_AUDIO_STREAM_END,
        must_follow=(CONVERSATION_RESPONSE,),
        max_time_between_s=240.0,
        severity="warning",
    ),
    SequenceRule(
        id="retry_executed_follows_schedule",
        event_type=RETRY_EXECUTED,
        must_follow=(RETRY_SCHEDULED,),
        max_time_between_s=900.0,
        severity="error",
    ),
    SequenceRule(
        id="retry_exhausted_follows_schedule",
        event_type=RETRY_EXHAUSTED,
        must_follow=(RETRY_SCHEDULED,),
        max_time_between_s=1800.0,
        severity="error",
    ),
    # Association can happen independently of writes (dream consolidation,
    # knowledge upgrades, clustering) — no prerequisite rule needed.
)

# Conflict window: events in conflicts_with must not have fired within this many seconds
_CONFLICT_WINDOW_S = 5.0
_LIFECYCLE_WINDOW_S = 240.0
_INPUT_TO_RESPONSE_WINDOW_S = 180.0
_RETRY_WINDOW_S = 1800.0
_MAX_TRACKED_CONVERSATIONS = 500
_MAX_TRACKED_RETRIES = 1000
_RELEASE_REASON_ALLOWLIST: frozenset[str] = frozenset({
    "proactive",
    "dismiss_command",
    "handle_transcription_timeout",
    "handle_transcription_crash",
})


class EventSequenceValidator:
    """Validates event ordering rules. Hook into EventBus.emit() as a pre-check."""

    def __init__(self) -> None:
        self._event_history: dict[str, float] = {}  # event_type -> last timestamp
        self._violations: list[Violation] = []
        self._total_validated: int = 0
        self._rules_by_event: dict[str, list[SequenceRule]] = {}
        self._conversation_lifecycle: dict[str, dict[str, float | str]] = {}
        self._retry_lifecycle: dict[str, dict[str, float | int | str]] = {}
        for rule in SEQUENCE_RULES:
            self._rules_by_event.setdefault(rule.event_type, []).append(rule)

    def record_event(self, event_type: str) -> None:
        """Record an event timestamp without validating rules.

        Use for callback-based events (wake word, barge-in) that bypass the
        event bus but should still satisfy prerequisite checks for downstream events.
        """
        self._event_history[event_type] = time.time()

    def _record_violation(self, violation: Violation) -> Violation:
        self._violations.append(violation)
        if violation.severity == "critical":
            logger.error("Event BLOCKED: %s", violation.message)
        elif violation.severity == "error":
            logger.error("Event sequence error: %s", violation.message)
        else:
            logger.warning("Event sequence violation: %s", violation.message)
        return violation

    @staticmethod
    def _conversation_id(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        cid = payload.get("conversation_id", "")
        return cid if isinstance(cid, str) else ""

    @staticmethod
    def _retry_id(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        rid = payload.get("retry_id", "")
        return rid if isinstance(rid, str) else ""

    def _prune_state(self, now: float) -> None:
        if len(self._conversation_lifecycle) > _MAX_TRACKED_CONVERSATIONS:
            stale = [
                cid for cid, state in self._conversation_lifecycle.items()
                if now - float(state.get("updated_ts", now)) > _LIFECYCLE_WINDOW_S
            ]
            for cid in stale[: len(stale) or 0]:
                self._conversation_lifecycle.pop(cid, None)
            while len(self._conversation_lifecycle) > _MAX_TRACKED_CONVERSATIONS:
                self._conversation_lifecycle.pop(next(iter(self._conversation_lifecycle)), None)

        if len(self._retry_lifecycle) > _MAX_TRACKED_RETRIES:
            stale_retry = [
                rid for rid, state in self._retry_lifecycle.items()
                if now - float(state.get("updated_ts", now)) > _RETRY_WINDOW_S
            ]
            for rid in stale_retry[: len(stale_retry) or 0]:
                self._retry_lifecycle.pop(rid, None)
            while len(self._retry_lifecycle) > _MAX_TRACKED_RETRIES:
                self._retry_lifecycle.pop(next(iter(self._retry_lifecycle)), None)

    def _validate_response_lifecycle(
        self,
        event_type: str,
        payload: dict[str, Any],
        now: float,
    ) -> Violation | None:
        cid = self._conversation_id(payload)
        if not cid:
            return None

        state = self._conversation_lifecycle.setdefault(cid, {})
        state["updated_ts"] = now

        if event_type == PERCEPTION_TRANSCRIPTION:
            state["transcription_ts"] = now
            return None

        if event_type == CONVERSATION_USER_MESSAGE:
            trans_ts = float(state.get("transcription_ts", 0.0) or 0.0)
            state["user_message_ts"] = now
            if trans_ts <= 0 or (now - trans_ts) > _INPUT_TO_RESPONSE_WINDOW_S:
                return self._record_violation(Violation(
                    rule_id="user_message_follows_transcription",
                    event_type=event_type,
                    violation_type="missing_prerequisite",
                    severity="warning",
                    message=(
                        f"{event_type} for {cid[:20]} missing recent {PERCEPTION_TRANSCRIPTION} "
                        f"(window={_INPUT_TO_RESPONSE_WINDOW_S}s)"
                    ),
                ))
            return None

        if event_type == CONVERSATION_RESPONSE:
            user_ts = float(state.get("user_message_ts", 0.0) or 0.0)
            trans_ts = float(state.get("transcription_ts", 0.0) or 0.0)
            release_reason = str(payload.get("release_reason", "") or "")
            allowed_orphan = release_reason in _RELEASE_REASON_ALLOWLIST
            has_recent_input = (
                (user_ts > 0 and (now - user_ts) <= _INPUT_TO_RESPONSE_WINDOW_S)
                or (trans_ts > 0 and (now - trans_ts) <= _INPUT_TO_RESPONSE_WINDOW_S)
            )
            state["response_ts"] = now
            state["response_id"] = str(payload.get("output_id", "") or "")
            if not has_recent_input and not allowed_orphan:
                return self._record_violation(Violation(
                    rule_id="response_follows_input_or_known_fallback",
                    event_type=event_type,
                    violation_type="missing_prerequisite",
                    severity="warning",
                    message=(
                        f"{event_type} for {cid[:20]} missing recent input "
                        f"({CONVERSATION_USER_MESSAGE}/{PERCEPTION_TRANSCRIPTION})"
                    ),
                ))
            return None

        if event_type in (PERCEPTION_PLAYBACK_COMPLETE, PERCEPTION_AUDIO_STREAM_END):
            resp_ts = float(state.get("response_ts", 0.0) or 0.0)
            if resp_ts <= 0 or (now - resp_ts) > _LIFECYCLE_WINDOW_S:
                return self._record_violation(Violation(
                    rule_id="playback_or_end_follows_response",
                    event_type=event_type,
                    violation_type="missing_prerequisite",
                    severity="error",
                    message=(
                        f"{event_type} for {cid[:20]} missing recent {CONVERSATION_RESPONSE} "
                        f"(window={_LIFECYCLE_WINDOW_S}s)"
                    ),
                ))
            state["playback_ts"] = now
            return None

        return None

    def _validate_retry_contract(
        self,
        event_type: str,
        payload: dict[str, Any],
        now: float,
    ) -> Violation | None:
        if event_type not in (RETRY_SCHEDULED, RETRY_EXECUTED, RETRY_EXHAUSTED):
            return None

        retry_id = self._retry_id(payload)
        if not retry_id:
            return self._record_violation(Violation(
                rule_id="retry_event_requires_retry_id",
                event_type=event_type,
                violation_type="contract_violation",
                severity="error",
                message=f"{event_type} missing retry_id",
            ))

        state = self._retry_lifecycle.get(retry_id)
        target_event_type = str(payload.get("target_event_type", "") or "")
        attempt = int(payload.get("attempt", 0) or 0)

        if event_type == RETRY_SCHEDULED:
            if state is None:
                state = {}
                self._retry_lifecycle[retry_id] = state
            else:
                prev_attempt = int(state.get("attempt", 0) or 0)
                if attempt and prev_attempt and attempt <= prev_attempt:
                    self._record_violation(Violation(
                        rule_id="retry_attempt_monotonic",
                        event_type=event_type,
                        violation_type="timing_violation",
                        severity="warning",
                        message=(
                            f"{event_type} retry_id={retry_id[:20]} non-monotonic attempt "
                            f"{attempt} <= {prev_attempt}"
                        ),
                    ))
            state["scheduled_ts"] = now
            state["attempt"] = attempt
            state["target_event_type"] = target_event_type
            state["updated_ts"] = now
            return None

        if state is None:
            return self._record_violation(Violation(
                rule_id="retry_terminal_requires_schedule",
                event_type=event_type,
                violation_type="missing_prerequisite",
                severity="error",
                message=f"{event_type} retry_id={retry_id[:20]} missing prior {RETRY_SCHEDULED}",
            ))

        scheduled_ts = float(state.get("scheduled_ts", 0.0) or 0.0)
        if scheduled_ts <= 0 or (now - scheduled_ts) > _RETRY_WINDOW_S:
            return self._record_violation(Violation(
                rule_id="retry_terminal_within_window",
                event_type=event_type,
                violation_type="timing_violation",
                severity="error",
                message=(
                    f"{event_type} retry_id={retry_id[:20]} outside retry window "
                    f"({_RETRY_WINDOW_S}s)"
                ),
            ))

        expected_target = str(state.get("target_event_type", "") or "")
        if target_event_type and expected_target and target_event_type != expected_target:
            self._record_violation(Violation(
                rule_id="retry_target_event_consistency",
                event_type=event_type,
                violation_type="contract_violation",
                severity="warning",
                message=(
                    f"{event_type} retry_id={retry_id[:20]} target mismatch "
                    f"{target_event_type} != {expected_target}"
                ),
            ))

        state["attempt"] = attempt
        state["updated_ts"] = now
        if event_type == RETRY_EXECUTED:
            state["executed_ts"] = now
        if event_type == RETRY_EXHAUSTED:
            self._retry_lifecycle.pop(retry_id, None)
        return None

    def validate(self, event_type: str, payload: dict[str, Any] | None = None) -> Violation | None:
        """Check rules for this event type. Returns first violation or None."""
        self._total_validated += 1
        now = time.time()
        event_payload = payload if isinstance(payload, dict) else {}
        rules = self._rules_by_event.get(event_type, [])

        for rule in rules:
            # Check prerequisites
            if rule.must_follow:
                found_prereq = False
                for prereq in rule.must_follow:
                    last = self._event_history.get(prereq)
                    if last is not None:
                        if rule.max_time_between_s <= 0 or (now - last) <= rule.max_time_between_s:
                            found_prereq = True
                            break
                if not found_prereq:
                    v = self._record_violation(Violation(
                        rule_id=rule.id, event_type=event_type,
                        violation_type="missing_prerequisite", severity=rule.severity,
                        message=f"{event_type} requires one of {rule.must_follow} within {rule.max_time_between_s}s",
                    ))
                    if rule.severity == "critical":
                        return v

            # Check minimum time between same-type events
            if rule.min_time_between_s > 0:
                last_same = self._event_history.get(event_type)
                if last_same is not None and (now - last_same) < rule.min_time_between_s:
                    v = self._record_violation(Violation(
                        rule_id=rule.id, event_type=event_type,
                        violation_type="timing_violation", severity=rule.severity,
                        message=f"{event_type} fired too soon (min {rule.min_time_between_s}s)",
                    ))
                    if rule.severity == "critical":
                        return v

            # Check conflicts
            if rule.conflicts_with:
                for conflict in rule.conflicts_with:
                    last_conflict = self._event_history.get(conflict)
                    if last_conflict is not None and (now - last_conflict) < _CONFLICT_WINDOW_S:
                        v = self._record_violation(Violation(
                            rule_id=rule.id, event_type=event_type,
                            violation_type="conflict_detected", severity=rule.severity,
                            message=f"{event_type} conflicts with recent {conflict}",
                        ))
                        if rule.severity == "critical":
                            return v

        lifecycle_violation = self._validate_response_lifecycle(event_type, event_payload, now)
        if lifecycle_violation is not None and lifecycle_violation.severity == "critical":
            return lifecycle_violation

        retry_violation = self._validate_retry_contract(event_type, event_payload, now)
        if retry_violation is not None and retry_violation.severity == "critical":
            return retry_violation

        # Record this event
        self._event_history[event_type] = now
        self._prune_state(now)
        return None

    @property
    def integrity_score(self) -> float:
        if self._total_validated == 0:
            return 1.0
        return max(0.0, 1.0 - len(self._violations) / self._total_validated)

    def get_recent_violations(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {"rule_id": v.rule_id, "event": v.event_type, "type": v.violation_type,
             "severity": v.severity, "message": v.message, "timestamp": v.timestamp}
            for v in self._violations[-limit:]
        ]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_validated": self._total_validated,
            "total_violations": len(self._violations),
            "integrity_score": round(self.integrity_score, 4),
            "violations_by_severity": {
                "warning": sum(1 for v in self._violations if v.severity == "warning"),
                "error": sum(1 for v in self._violations if v.severity == "error"),
                "critical": sum(1 for v in self._violations if v.severity == "critical"),
            },
        }


event_validator = EventSequenceValidator()
