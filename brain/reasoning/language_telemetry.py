"""Live quality telemetry for the Jarvis-native language substrate.

This is separate from the training corpus. The corpus stores supervised examples;
this log stores runtime behavior and outcomes for bounded/native response paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
LANGUAGE_DIR = JARVIS_DIR / "language_corpus"
QUALITY_LOG_PATH = LANGUAGE_DIR / "quality_events.jsonl"
MAX_LOG_SIZE_MB = 25
RECENT_EVENTS_LIMIT = 40


class LanguageQualityTelemetry:
    _instance: LanguageQualityTelemetry | None = None

    def __init__(self, path: str | Path = "") -> None:
        self._path = Path(path) if path else QUALITY_LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._loaded = False
        self._total_events = 0
        self._by_class: Counter[str] = Counter()
        self._by_route: Counter[str] = Counter()
        self._by_outcome: Counter[str] = Counter()
        self._by_feedback: Counter[str] = Counter()
        self._by_provenance: Counter[str] = Counter()
        self._native_used_by_class: Counter[str] = Counter()
        self._fail_closed_by_class: Counter[str] = Counter()
        self._outcomes_by_class: dict[str, Counter[str]] = defaultdict(Counter)
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=RECENT_EVENTS_LIMIT)
        self._shadow_total = 0
        self._shadow_by_class: Counter[str] = Counter()
        self._shadow_by_model: Counter[str] = Counter()
        self._shadow_by_choice: Counter[str] = Counter()
        self._shadow_by_reason: Counter[str] = Counter()
        self._recent_shadow: deque[dict[str, Any]] = deque(maxlen=RECENT_EVENTS_LIMIT)
        self._ambiguous_total = 0
        self._ambiguous_by_selected_route: Counter[str] = Counter()
        self._ambiguous_by_candidate: Counter[str] = Counter()
        self._ambiguous_by_outcome: Counter[str] = Counter()
        self._ambiguous_by_feedback: Counter[str] = Counter()
        self._recent_ambiguous: deque[dict[str, Any]] = deque(maxlen=RECENT_EVENTS_LIMIT)
        self._runtime_total = 0
        self._runtime_live_total = 0
        self._runtime_blocked_total = 0
        self._runtime_unpromoted_live_attempts = 0
        self._runtime_live_red_classes = 0
        self._runtime_live_by_class: Counter[str] = Counter()
        self._runtime_blocked_by_class: Counter[str] = Counter()
        self._runtime_by_mode: Counter[str] = Counter()
        self._runtime_by_reason: Counter[str] = Counter()
        self._runtime_by_level: Counter[str] = Counter()
        self._recent_runtime: deque[dict[str, Any]] = deque(maxlen=RECENT_EVENTS_LIMIT)

    @classmethod
    def get_instance(cls) -> LanguageQualityTelemetry:
        if cls._instance is None:
            cls._instance = LanguageQualityTelemetry()
        return cls._instance

    def record_event(
        self,
        *,
        conversation_id: str,
        route: str,
        response_class: str,
        provenance_verdict: str,
        outcome: str,
        user_feedback: str,
        confidence: float,
        native_used: bool,
        fail_closed: bool,
        safety_flags: list[str] | None = None,
        query: str = "",
        reply: str = "",
        runtime_policy: dict[str, Any] | None = None,
    ) -> None:
        _runtime_policy = None
        if isinstance(runtime_policy, dict):
            _runtime_policy = {
                "bridge_enabled": bool(runtime_policy.get("bridge_enabled", False)),
                "rollout_mode": str(runtime_policy.get("rollout_mode", "off") or "off"),
                "promotion_level": str(runtime_policy.get("promotion_level", "shadow") or "shadow"),
                "gate_color": str(runtime_policy.get("gate_color", "") or ""),
                "native_candidate": bool(runtime_policy.get("native_candidate", False)),
                "native_allowed": bool(runtime_policy.get("native_allowed", False)),
                "strict_native": bool(runtime_policy.get("strict_native", False)),
                "blocked_by_guard": bool(runtime_policy.get("blocked_by_guard", False)),
                "runtime_live": bool(runtime_policy.get("runtime_live", False)),
                "unpromoted_live_attempt": bool(runtime_policy.get("unpromoted_live_attempt", False)),
                "reason": str(runtime_policy.get("reason", "") or ""),
                "response_class": str(runtime_policy.get("response_class", "") or ""),
                "canary_classes": list(runtime_policy.get("canary_classes", []) or []),
            }
        record = {
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "route": route or "unknown",
            "response_class": response_class or "unknown",
            "provenance_verdict": provenance_verdict or "unknown",
            "outcome": outcome or "unknown",
            "user_feedback": user_feedback or "",
            "confidence": max(0.0, min(1.0, float(confidence))),
            "native_used": bool(native_used),
            "fail_closed": bool(fail_closed),
            "safety_flags": list(safety_flags or []),
            "query": (query or "")[:180],
            "reply": (reply or "")[:220],
            "runtime_policy": _runtime_policy,
        }
        with self._lock:
            self._ensure_loaded_locked()
            self._maybe_rotate_locked()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            self._ingest_locked(record)

    def _maybe_rotate_locked(self) -> None:
        try:
            if self._path.exists() and self._path.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
                rotated = self._path.with_suffix(self._path.suffix + ".1")
                if rotated.exists():
                    rotated.unlink()
                self._path.rename(rotated)
                logger.info("Language quality telemetry rotated: %s", self._path.name)
        except Exception:
            logger.warning("Language quality telemetry rotation failed", exc_info=True)

    def _rehydrate_paths_locked(self) -> list[Path]:
        rotated = self._path.with_suffix(self._path.suffix + ".1")
        paths: list[Path] = []
        if rotated.exists():
            paths.append(rotated)
        if self._path.exists():
            paths.append(self._path)
        return paths

    def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        paths = self._rehydrate_paths_locked()
        if not paths:
            return
        try:
            for path in paths:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            continue
                        _record_type = str(record.get("type", "") or "")
                        if _record_type == "shadow_comparison":
                            self._ingest_shadow_locked(record)
                        elif _record_type == "ambiguous_intent_probe":
                            self._ingest_ambiguous_locked(record)
                        else:
                            self._ingest_locked(record)
        except Exception:
            logger.warning("Language quality telemetry rehydrate failed", exc_info=True)

    def _ingest_locked(self, record: dict[str, Any]) -> None:
        response_class = str(record.get("response_class", "") or "unknown")
        route = str(record.get("route", "") or "unknown")
        outcome = str(record.get("outcome", "") or "unknown")
        provenance = str(record.get("provenance_verdict", "") or "unknown")
        feedback = str(record.get("user_feedback", "") or "")
        native_used = bool(record.get("native_used", False))
        fail_closed = bool(record.get("fail_closed", False))
        runtime_policy = record.get("runtime_policy")
        self._total_events += 1
        self._by_class[response_class] += 1
        self._by_route[route] += 1
        self._by_outcome[outcome] += 1
        self._by_provenance[provenance] += 1
        if feedback:
            self._by_feedback[feedback] += 1
        if native_used:
            self._native_used_by_class[response_class] += 1
        if fail_closed:
            self._fail_closed_by_class[response_class] += 1
        self._outcomes_by_class[response_class][outcome] += 1
        if isinstance(runtime_policy, dict):
            self._ingest_runtime_locked(
                runtime_policy,
                response_class=response_class,
                timestamp=float(record.get("timestamp", 0.0) or 0.0),
            )
        self._recent_events.append({
            "timestamp": float(record.get("timestamp", 0.0) or 0.0),
            "response_class": response_class,
            "route": route,
            "outcome": outcome,
            "user_feedback": feedback,
            "provenance_verdict": provenance,
            "native_used": native_used,
            "fail_closed": fail_closed,
            "query": str(record.get("query", "") or "")[:180],
            "reply": str(record.get("reply", "") or "")[:220],
            "confidence": float(record.get("confidence", 0.0) or 0.0),
            "runtime_policy": runtime_policy if isinstance(runtime_policy, dict) else {},
        })

    def _ingest_runtime_locked(
        self,
        runtime_policy: dict[str, Any],
        *,
        response_class: str,
        timestamp: float,
    ) -> None:
        mode = str(runtime_policy.get("rollout_mode", "off") or "off")
        level = str(runtime_policy.get("promotion_level", "shadow") or "shadow")
        reason = str(runtime_policy.get("reason", "") or "")
        gate_color = str(runtime_policy.get("gate_color", "") or "")
        blocked = bool(runtime_policy.get("blocked_by_guard", False))
        runtime_live = bool(runtime_policy.get("runtime_live", False))
        unpromoted_live_attempt = bool(runtime_policy.get("unpromoted_live_attempt", False))

        self._runtime_total += 1
        self._runtime_by_mode[mode] += 1
        self._runtime_by_level[level] += 1
        if reason:
            self._runtime_by_reason[reason] += 1
        if blocked:
            self._runtime_blocked_total += 1
            self._runtime_blocked_by_class[response_class] += 1
        if runtime_live:
            self._runtime_live_total += 1
            self._runtime_live_by_class[response_class] += 1
            if gate_color == "red":
                self._runtime_live_red_classes += 1
        if unpromoted_live_attempt:
            self._runtime_unpromoted_live_attempts += 1

        self._recent_runtime.append({
            "timestamp": float(timestamp or 0.0),
            "response_class": response_class,
            "rollout_mode": mode,
            "promotion_level": level,
            "reason": reason,
            "blocked_by_guard": blocked,
            "runtime_live": runtime_live,
            "unpromoted_live_attempt": unpromoted_live_attempt,
            "gate_color": gate_color,
            "bridge_enabled": bool(runtime_policy.get("bridge_enabled", False)),
            "strict_native": bool(runtime_policy.get("strict_native", False)),
        })

    def _ingest_shadow_locked(self, record: dict[str, Any]) -> None:
        response_class = str(record.get("response_class", "") or "unknown")
        model_family = str(record.get("model_family", "") or "unknown")
        chosen = str(record.get("chosen", "") or "unknown")
        reason = str(record.get("reason", "") or "")
        self._shadow_total += 1
        self._shadow_by_class[response_class] += 1
        self._shadow_by_model[model_family] += 1
        self._shadow_by_choice[chosen] += 1
        if reason:
            self._shadow_by_reason[reason] += 1
        self._recent_shadow.append({
            "timestamp": float(record.get("timestamp", 0.0) or 0.0),
            "response_class": response_class,
            "model_family": model_family,
            "chosen": chosen,
            "reason": reason,
            "bounded_confidence": float(record.get("bounded_confidence", 0.0) or 0.0),
            "query": str(record.get("query", "") or "")[:180],
            "bounded_reply": str(record.get("bounded_reply", "") or "")[:220],
            "llm_reply": str(record.get("llm_reply", "") or "")[:220],
        })

    def _ingest_ambiguous_locked(self, record: dict[str, Any]) -> None:
        selected_route = str(record.get("selected_route", "") or "unknown")
        candidate_intent = str(record.get("candidate_intent", "") or "unknown")
        outcome = str(record.get("outcome", "") or "unknown")
        feedback = str(record.get("user_feedback", "") or "")
        self._ambiguous_total += 1
        self._ambiguous_by_selected_route[selected_route] += 1
        self._ambiguous_by_candidate[candidate_intent] += 1
        self._ambiguous_by_outcome[outcome] += 1
        if feedback:
            self._ambiguous_by_feedback[feedback] += 1
        self._recent_ambiguous.append({
            "timestamp": float(record.get("timestamp", 0.0) or 0.0),
            "selected_route": selected_route,
            "candidate_intent": candidate_intent,
            "trigger": str(record.get("trigger", "") or ""),
            "candidate_confidence": float(record.get("candidate_confidence", 0.0) or 0.0),
            "outcome": outcome,
            "user_feedback": feedback,
            "query": str(record.get("query", "") or "")[:180],
            "shadow_only": bool(record.get("shadow_only", True)),
        })

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded_locked()
            paths = self._rehydrate_paths_locked()
            native_total = sum(self._native_used_by_class.values())
            fail_closed_total = sum(self._fail_closed_by_class.values())
            retained_bytes = sum(path.stat().st_size for path in paths if path.exists())
            return {
                "path": str(self._path),
                "exists": self._path.exists(),
                "file_size_bytes": self._path.stat().st_size if self._path.exists() else 0,
                "retained_file_count": len(paths),
                "retained_file_size_bytes": retained_bytes,
                "total_events": self._total_events,
                "counts_by_response_class": dict(self._by_class),
                "counts_by_route": dict(self._by_route),
                "counts_by_outcome": dict(self._by_outcome),
                "counts_by_feedback": dict(self._by_feedback),
                "counts_by_provenance": dict(self._by_provenance),
                "native_used_by_class": dict(self._native_used_by_class),
                "fail_closed_by_class": dict(self._fail_closed_by_class),
                "native_total": native_total,
                "fail_closed_total": fail_closed_total,
                "native_usage_rate": (native_total / self._total_events) if self._total_events else 0.0,
                "fail_closed_rate": (fail_closed_total / self._total_events) if self._total_events else 0.0,
                "rates_overlap": True,
                "outcomes_by_class": {k: dict(v) for k, v in self._outcomes_by_class.items()},
                "recent_events": list(self._recent_events),
                "last_event_ts": self._recent_events[-1]["timestamp"] if self._recent_events else 0.0,
                "shadow_comparisons": {
                    "total": self._shadow_total,
                    "by_class": dict(self._shadow_by_class),
                    "by_model_family": dict(self._shadow_by_model),
                    "by_choice": dict(self._shadow_by_choice),
                    "by_reason": dict(self._shadow_by_reason),
                    "recent": list(self._recent_shadow),
                    "last_ts": self._recent_shadow[-1]["timestamp"] if self._recent_shadow else 0.0,
                },
                "ambiguous_intent": {
                    "total": self._ambiguous_total,
                    "by_selected_route": dict(self._ambiguous_by_selected_route),
                    "by_candidate_intent": dict(self._ambiguous_by_candidate),
                    "by_outcome": dict(self._ambiguous_by_outcome),
                    "by_feedback": dict(self._ambiguous_by_feedback),
                    "recent": list(self._recent_ambiguous),
                    "last_ts": self._recent_ambiguous[-1]["timestamp"] if self._recent_ambiguous else 0.0,
                },
                "runtime_guard": {
                    "total": self._runtime_total,
                    "live_total": self._runtime_live_total,
                    "blocked_by_guard_count": self._runtime_blocked_total,
                    "unpromoted_live_attempts": self._runtime_unpromoted_live_attempts,
                    "live_red_classes": self._runtime_live_red_classes,
                    "live_by_class": dict(self._runtime_live_by_class),
                    "blocked_by_class": dict(self._runtime_blocked_by_class),
                    "by_rollout_mode": dict(self._runtime_by_mode),
                    "by_reason": dict(self._runtime_by_reason),
                    "by_promotion_level": dict(self._runtime_by_level),
                    "live_rate": (self._runtime_live_total / self._runtime_total) if self._runtime_total else 0.0,
                    "blocked_rate": (self._runtime_blocked_total / self._runtime_total) if self._runtime_total else 0.0,
                    "recent": list(self._recent_runtime),
                    "last_ts": self._recent_runtime[-1]["timestamp"] if self._recent_runtime else 0.0,
                },
            }


    def record_shadow_comparison(
        self,
        *,
        conversation_id: str,
        response_class: str,
        query: str = "",
        bounded_reply: str = "",
        llm_reply: str = "",
        bounded_confidence: float = 0.0,
        chosen: str = "bounded",
        reason: str = "",
        model_family: str = "shadow_model",
    ) -> None:
        """Record a shadow A/B comparison between bounded and LLM paths.

        This is only logged when both paths produce output for the same query,
        allowing offline analysis of which produces better results.
        """
        record = {
            "timestamp": time.time(),
            "type": "shadow_comparison",
            "conversation_id": conversation_id,
            "response_class": response_class or "unknown",
            "query": (query or "")[:180],
            "bounded_reply": (bounded_reply or "")[:300],
            "llm_reply": (llm_reply or "")[:300],
            "bounded_confidence": max(0.0, min(1.0, float(bounded_confidence))),
            "chosen": chosen,
            "reason": reason,
            "model_family": model_family or "shadow_model",
        }
        with self._lock:
            self._ensure_loaded_locked()
            self._maybe_rotate_locked()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            self._ingest_shadow_locked(record)

    def record_ambiguous_intent_probe(
        self,
        *,
        conversation_id: str,
        query: str,
        selected_route: str,
        candidate_intent: str,
        candidate_confidence: float,
        trigger: str = "",
        outcome: str = "",
        user_feedback: str = "",
        shadow_only: bool = True,
    ) -> None:
        """Record ambiguous-intent routing probes for shadow learning only."""
        record = {
            "timestamp": time.time(),
            "type": "ambiguous_intent_probe",
            "conversation_id": conversation_id,
            "query": (query or "")[:180],
            "selected_route": selected_route or "unknown",
            "candidate_intent": candidate_intent or "unknown",
            "candidate_confidence": max(0.0, min(1.0, float(candidate_confidence))),
            "trigger": trigger or "",
            "outcome": outcome or "",
            "user_feedback": user_feedback or "",
            "shadow_only": bool(shadow_only),
        }
        with self._lock:
            self._ensure_loaded_locked()
            self._maybe_rotate_locked()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            self._ingest_ambiguous_locked(record)


language_quality_telemetry = LanguageQualityTelemetry.get_instance()
