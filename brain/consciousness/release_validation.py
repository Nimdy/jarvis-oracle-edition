"""Release-path output validation with immutable evidence records.

This module enforces a minimal, deterministic contract for user-visible
conversation outputs before they are marked as released.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from consciousness.events import (
    CONVERSATION_RESPONSE,
    OUTPUT_RELEASE_BLOCKED,
    OUTPUT_VALIDATION_RECORDED,
    event_bus,
)

_VALIDATION_DIR = os.path.expanduser("~/.jarvis")
_VALIDATION_PATH = os.path.join(_VALIDATION_DIR, "release_validation.jsonl")
_MAX_FILE_MB = 10
_MAX_RECENT_RECORDS = 200
_MAX_VALIDATED_OUTPUTS = 2000


@dataclass(frozen=True)
class ReleaseValidationDecision:
    validation_id: str
    passed: bool
    effective_release_status: str
    effective_release_reason: str
    violations: tuple[str, ...]


def _short_id(prefix: str) -> str:
    return f"{prefix}_{time.monotonic_ns()}"


class OutputReleaseValidator:
    """Validates and records output release decisions."""

    _instance: OutputReleaseValidator | None = None

    @classmethod
    def get_instance(cls) -> OutputReleaseValidator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def __init__(self, record_path: str = _VALIDATION_PATH) -> None:
        self._lock = threading.Lock()
        self._path = os.path.expanduser(record_path)
        self._recent: deque[dict[str, Any]] = deque(maxlen=_MAX_RECENT_RECORDS)
        self._validated_outputs: set[str] = set()
        self._validated_outputs_order: deque[str] = deque(maxlen=_MAX_VALIDATED_OUTPUTS)
        self._wired = False
        self._cleanup = None
        self._errors = 0
        self._last_error = ""

        self._validation_total = 0
        self._validation_failed = 0
        self._released_total = 0
        self._released_validated = 0
        self._released_without_validation = 0
        self._blocked_by_validation = 0

        os.makedirs(os.path.dirname(self._path) or _VALIDATION_DIR, exist_ok=True)

    def wire(self) -> None:
        if self._wired:
            return
        self._cleanup = event_bus.on(CONVERSATION_RESPONSE, self._on_conversation_response)
        self._wired = True

    def validate_output(
        self,
        *,
        text: str,
        conversation_id: str,
        trace_id: str,
        request_id: str,
        output_id: str,
        release_status: str,
        release_reason: str = "",
        source: str = "",
    ) -> ReleaseValidationDecision:
        requested = (release_status or "").strip().lower() or "released"
        violations: list[str] = []

        if not conversation_id:
            violations.append("missing_conversation_id")
        if not trace_id:
            violations.append("missing_trace_id")
        if not request_id:
            violations.append("missing_request_id")
        if not output_id:
            violations.append("missing_output_id")
        if requested not in ("released", "blocked"):
            violations.append("invalid_release_status")
        if requested == "released" and not str(text or "").strip():
            violations.append("empty_released_text")

        passed = len(violations) == 0
        effective_status = requested
        effective_reason = release_reason or ""
        if requested == "released" and not passed:
            effective_status = "blocked"
            reason_suffix = ",".join(violations[:3])
            effective_reason = f"release_validation_failed:{reason_suffix}"

        validation_id = _short_id("val")
        record = {
            "type": "output_validation",
            "validation_id": validation_id,
            "ts": time.time(),
            "source": source or "unknown",
            "conversation_id": conversation_id,
            "trace_id": trace_id,
            "request_id": request_id,
            "output_id": output_id,
            "release_status_requested": requested,
            "release_status_effective": effective_status,
            "release_reason_effective": effective_reason,
            "passed": passed,
            "violations": list(violations),
            "text_len": len(text or ""),
        }

        with self._lock:
            self._validation_total += 1
            if not passed:
                self._validation_failed += 1
            if requested == "released" and effective_status == "blocked":
                self._blocked_by_validation += 1
            if output_id:
                if output_id not in self._validated_outputs:
                    if len(self._validated_outputs_order) >= _MAX_VALIDATED_OUTPUTS:
                        oldest = self._validated_outputs_order.popleft()
                        self._validated_outputs.discard(oldest)
                    self._validated_outputs.add(output_id)
                    self._validated_outputs_order.append(output_id)
            self._recent.appendleft(record)

        self._append_jsonl(record)
        event_bus.emit(
            OUTPUT_VALIDATION_RECORDED,
            validation_id=validation_id,
            conversation_id=conversation_id,
            trace_id=trace_id,
            request_id=request_id,
            output_id=output_id,
            release_status=requested,
            effective_release_status=effective_status,
            validation_passed=passed,
            violations=list(violations),
        )
        if requested == "released" and effective_status == "blocked":
            event_bus.emit(
                OUTPUT_RELEASE_BLOCKED,
                validation_id=validation_id,
                conversation_id=conversation_id,
                trace_id=trace_id,
                request_id=request_id,
                output_id=output_id,
                reason=effective_reason,
                violations=list(violations),
            )

        return ReleaseValidationDecision(
            validation_id=validation_id,
            passed=passed,
            effective_release_status=effective_status,
            effective_release_reason=effective_reason,
            violations=tuple(violations),
        )

    def _on_conversation_response(self, **kwargs: Any) -> None:
        release_status = str(kwargs.get("release_status", "") or "").strip().lower()
        if release_status != "released":
            return
        output_id = str(kwargs.get("output_id", "") or "")
        validation_id = str(kwargs.get("validation_id", "") or "")
        validation_passed = bool(kwargs.get("validation_passed", False))
        violation: dict[str, Any] | None = None
        with self._lock:
            has_known_output = bool(output_id and output_id in self._validated_outputs)
            validated = bool(validation_id and validation_passed and has_known_output)
            self._released_total += 1
            if validated:
                self._released_validated += 1
                return
            self._released_without_validation += 1
            violation = {
                "type": "released_without_validation",
                "ts": time.time(),
                "conversation_id": str(kwargs.get("conversation_id", "") or ""),
                "trace_id": str(kwargs.get("trace_id", "") or ""),
                "request_id": str(kwargs.get("request_id", "") or ""),
                "output_id": output_id,
                "validation_id": validation_id,
            }
            self._recent.appendleft(violation)
        if violation is not None:
            self._append_jsonl(violation)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "wired": self._wired,
                "validation_total": self._validation_total,
                "validation_failed": self._validation_failed,
                "released_total": self._released_total,
                "released_validated": self._released_validated,
                "released_without_validation": self._released_without_validation,
                "blocked_by_validation": self._blocked_by_validation,
                "recent": list(self._recent)[:20],
                "errors": self._errors,
                "last_error": self._last_error,
            }

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        try:
            self._maybe_rotate()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n")
        except Exception as exc:
            self._errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"

    def _maybe_rotate(self) -> None:
        try:
            if not os.path.exists(self._path):
                return
            size = os.path.getsize(self._path)
            if size < (_MAX_FILE_MB * 1024 * 1024):
                return
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            keep_from = len(lines) // 2
            with open(self._path, "w", encoding="utf-8") as f:
                f.writelines(lines[keep_from:])
        except Exception as exc:
            self._errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"


output_release_validator = OutputReleaseValidator.get_instance()
try:
    output_release_validator.wire()
except Exception:
    # Fail-open: output paths should not crash if validator wire fails.
    pass

