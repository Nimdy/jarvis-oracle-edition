"""Grounded language corpus capture for Jarvis-native language training.

Stores append-only supervised examples built from real conversation routes where
we know the grounding payload and final answer. This is the first substrate for
training bounded native articulation and, later, a shadow Jarvis language model.
"""

from __future__ import annotations

from collections import Counter, deque
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
LANGUAGE_CORPUS_DIR = JARVIS_DIR / "language_corpus"
EXAMPLES_PATH = LANGUAGE_CORPUS_DIR / "examples.jsonl"
MAX_LOG_SIZE_MB = 25
RECENT_EXAMPLES_LIMIT = 25


@dataclass
class LanguageCorpusExample:
    example_id: str
    conversation_id: str
    query: str
    route: str
    response_class: str
    meaning_frame: dict[str, Any]
    grounding_payload: Any
    teacher_answer: str
    final_answer: str
    provenance_verdict: str
    user_feedback: str
    confidence: float
    safety_flags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LanguageCorpusStore:
    _instance: LanguageCorpusStore | None = None

    def __init__(self, path: str | Path = "") -> None:
        self._path = Path(path) if path else EXAMPLES_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._total_examples = 0
        self._counts_by_route: Counter[str] = Counter()
        self._counts_by_response_class: Counter[str] = Counter()
        self._counts_by_route_class: Counter[str] = Counter()
        self._counts_by_provenance: Counter[str] = Counter()
        self._counts_by_feedback: Counter[str] = Counter()
        self._counts_by_flag: Counter[str] = Counter()
        self._recent_examples: deque[dict[str, Any]] = deque(maxlen=RECENT_EXAMPLES_LIMIT)
        self._loaded_from_disk = False

    @classmethod
    def get_instance(cls) -> LanguageCorpusStore:
        if cls._instance is None:
            cls._instance = LanguageCorpusStore()
        return cls._instance

    def append_example(
        self,
        *,
        conversation_id: str,
        query: str,
        route: str,
        response_class: str,
        meaning_frame: dict[str, Any] | None,
        grounding_payload: Any,
        teacher_answer: str,
        final_answer: str,
        provenance_verdict: str,
        user_feedback: str = "",
        confidence: float = 1.0,
        safety_flags: list[str] | None = None,
    ) -> str:
        example = LanguageCorpusExample(
            example_id=f"lang_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            query=query[:1000],
            route=route,
            response_class=response_class,
            meaning_frame=dict(meaning_frame or {}),
            grounding_payload=grounding_payload,
            teacher_answer=teacher_answer[:4000],
            final_answer=final_answer[:4000],
            provenance_verdict=provenance_verdict,
            user_feedback=user_feedback,
            confidence=max(0.0, min(1.0, float(confidence))),
            safety_flags=list(safety_flags or []),
        )
        with self._lock:
            self._ensure_loaded_locked()
            self._maybe_rotate()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(example.to_dict(), default=str) + "\n")
            self._record_stats_locked(example.to_dict())
        return example.example_id

    def _record_stats_locked(self, example: dict[str, Any]) -> None:
        self._total_examples += 1
        route = str(example.get("route", "") or "unknown")
        response_class = str(example.get("response_class", "") or "unknown")
        provenance = str(example.get("provenance_verdict", "") or "unknown")
        feedback = str(example.get("user_feedback", "") or "")
        self._counts_by_route[route] += 1
        self._counts_by_response_class[response_class] += 1
        route_class_key = f"{route.lower()}|{response_class.lower()}"
        self._counts_by_route_class[route_class_key] += 1
        self._counts_by_provenance[provenance] += 1
        if feedback:
            self._counts_by_feedback[feedback] += 1
        for flag in example.get("safety_flags", []) or []:
            self._counts_by_flag[str(flag)] += 1
        self._recent_examples.append({
            "example_id": example.get("example_id", ""),
            "timestamp": float(example.get("timestamp", 0.0) or 0.0),
            "route": route,
            "response_class": response_class,
            "provenance_verdict": provenance,
            "user_feedback": feedback,
            "query": str(example.get("query", "") or "")[:180],
            "lead": str((example.get("meaning_frame") or {}).get("lead", "") or "")[:220],
            "confidence": float(example.get("confidence", 0.0) or 0.0),
            "safety_flags": list(example.get("safety_flags", []) or []),
        })

    def _ensure_loaded_locked(self) -> None:
        if self._loaded_from_disk:
            return
        self._loaded_from_disk = True
        rotated = self._path.with_suffix(self._path.suffix + ".1")
        paths: list[Path] = []
        if rotated.exists():
            paths.append(rotated)
        if self._path.exists():
            paths.append(self._path)
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
                        self._record_stats_locked(record)
        except Exception:
            logger.warning("Language corpus rehydrate failed", exc_info=True)

    def _maybe_rotate(self) -> None:
        try:
            if self._path.exists() and self._path.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
                rotated = self._path.with_suffix(self._path.suffix + ".1")
                if rotated.exists():
                    rotated.unlink()
                self._path.rename(rotated)
                logger.info("Language corpus rotated: %s", self._path.name)
        except Exception:
            logger.warning("Language corpus rotation failed", exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_loaded_locked()
            rotated = self._path.with_suffix(self._path.suffix + ".1")
            retained_paths = [p for p in (rotated, self._path) if p.exists()]
            examples_with_frames = sum(1 for ex in self._recent_examples if ex.get("lead"))
            return {
                "path": str(self._path),
                "exists": self._path.exists(),
                "file_size_bytes": self._path.stat().st_size if self._path.exists() else 0,
                "retained_file_count": len(retained_paths),
                "retained_file_size_bytes": sum(p.stat().st_size for p in retained_paths),
                "total_examples": self._total_examples,
                "counts_by_route": dict(self._counts_by_route),
                "counts_by_response_class": dict(self._counts_by_response_class),
                "counts_by_route_class": dict(self._counts_by_route_class),
                "counts_by_provenance": dict(self._counts_by_provenance),
                "counts_by_feedback": dict(self._counts_by_feedback),
                "counts_by_safety_flag": dict(self._counts_by_flag),
                "recent_examples": list(self._recent_examples),
                "recent_example_count": len(self._recent_examples),
                "examples_with_meaning_frame_preview": examples_with_frames,
                "native_response_classes": sorted(self._counts_by_response_class.keys()),
                "last_capture_ts": self._recent_examples[-1]["timestamp"] if self._recent_examples else 0.0,
            }


language_corpus = LanguageCorpusStore.get_instance()
