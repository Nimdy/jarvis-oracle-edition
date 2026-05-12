"""Frozen dataclasses for the eval sidecar.

No imports from core cognition — pure data schemas only.
Every record carries scoring_version and run_id for auditability.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class EvalEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    event_id: str = field(default_factory=_uid)
    timestamp: float = field(default_factory=_now)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalEvent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class EvalSnapshot:
    source: str
    metrics: dict[str, Any] = field(default_factory=dict)
    snapshot_id: str = field(default_factory=_uid)
    timestamp: float = field(default_factory=_now)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalSnapshot:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class EvalScore:
    category: str
    score: float
    sample_size: int = 0
    scoring_version: str = ""
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    score_id: str = field(default_factory=_uid)
    timestamp: float = field(default_factory=_now)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalScore:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class EvalScorecard:
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    scorecard_id: str = field(default_factory=_uid)
    timestamp: float = field(default_factory=_now)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalScorecard:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalRun:
    mode: str = "shadow"
    scoring_version: str = ""
    scenario_pack_version: str = ""
    notes: str = ""
    run_id: str = field(default_factory=_uid)
    started_at: float = field(default_factory=_now)
    ended_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalRun:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
