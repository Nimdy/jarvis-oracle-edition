"""Autonomy Policy Memory — learn what works and what doesn't.

Every completed research job with a measured delta gets recorded as an
experience entry.  Future scoring consults this history to boost topics
that consistently improve metrics and penalize topics that consistently
regress.

Persisted as append-only JSONL to ~/.jarvis/autonomy_policy.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from autonomy.constants import MIN_MEANINGFUL_DELTA, WARMUP_PERIOD_S

logger = logging.getLogger(__name__)

_JARVIS_DIR = os.path.join(Path.home(), ".jarvis")
_POLICY_PATH = os.path.join(_JARVIS_DIR, "autonomy_policy.jsonl")

MAX_ENTRIES = 500
MIN_ENTRIES_FOR_PRIOR = 3
_MAX_JSONL_BYTES = 10 * 1024 * 1024  # 10 MB
_AVOID_STRICT_WIN_RATE = 0.20
_AVOID_REGRESSION_HEAVY_WIN_RATE = 0.35
_AVOID_REGRESSION_HEAVY_LOSS_RATE = 0.50


@dataclass
class PolicyOutcome:
    """A single autonomy experience entry."""

    intent_id: str = ""
    intent_type: str = ""        # "metric:confidence_volatility", "thought:existential", etc.
    tool_used: str = ""          # "web", "codebase", "memory", "introspection"
    topic_tags: tuple[str, ...] = ()
    question_summary: str = ""
    net_delta: float = 0.0       # positive = improved, negative = regressed
    stable: bool = False
    confidence: float = 0.0
    cost_tokens: int = 0
    cost_seconds: float = 0.0
    risk_score: float = 0.0
    worked: bool = False         # net_delta > MIN_MEANINGFUL_DELTA and stable
    warmup: bool = False         # recorded during session warmup period
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["topic_tags"] = list(self.topic_tags)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyOutcome:
        d["topic_tags"] = tuple(d.get("topic_tags", ()))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TopicPrior:
    """Aggregated statistics for a topic cluster."""

    total: int = 0
    wins: int = 0
    losses: int = 0
    avg_delta: float = 0.0
    win_rate: float = 0.0
    last_outcome_ts: float = 0.0


class AutonomyPolicyMemory:
    """Experience-based learning for autonomy scoring and prioritization.

    Outcomes recorded during the warmup period (first WARMUP_PERIOD_S seconds
    after session start) are stored but excluded from priors. This prevents
    cold-start noise — unstable baselines, model loading transients, few data
    points — from poisoning the experience memory that drives future scoring.
    """

    def __init__(self) -> None:
        self._outcomes: list[PolicyOutcome] = []
        self._tag_index: dict[str, list[int]] = defaultdict(list)
        self._tool_index: dict[str, list[int]] = defaultdict(list)
        self._type_index: dict[str, list[int]] = defaultdict(list)
        self._session_start: float = time.time()
        self._warmup_count: int = 0
        self._load()

    def is_warmup(self) -> bool:
        """True if the current session is still in the warmup period."""
        return (time.time() - self._session_start) < WARMUP_PERIOD_S

    def record_outcome(self, outcome: PolicyOutcome) -> None:
        if len(self._outcomes) >= MAX_ENTRIES:
            self._trim()

        stored = outcome
        if self.is_warmup():
            stored = replace(outcome, warmup=True)
            self._warmup_count += 1

        idx = len(self._outcomes)
        self._outcomes.append(stored)
        self._index_entry(idx, stored)
        self._persist_append(stored)

        warmup_tag = " [warmup]" if stored.warmup else ""
        verb = "improved" if stored.worked else "regressed/neutral"
        logger.info(
            "Autonomy policy: %s [%s] delta=%.3f %s%s",
            outcome.intent_type, outcome.tool_used,
            outcome.net_delta, verb, warmup_tag,
        )

    def get_topic_prior(self, tags: tuple[str, ...]) -> TopicPrior:
        """Aggregate prior for a topic cluster (any overlapping tag).

        Warmup outcomes are excluded — they represent cold-start noise,
        not real signal about whether a topic is worth pursuing.
        """
        indices: set[int] = set()
        for tag in tags:
            indices.update(self._tag_index.get(tag, []))

        entries = [
            self._outcomes[i] for i in sorted(indices)
            if i < len(self._outcomes) and not self._outcomes[i].warmup
        ]
        if len(entries) < MIN_ENTRIES_FOR_PRIOR:
            return TopicPrior()

        return self._build_prior(entries)

    def get_tool_prior(self, tool: str) -> TopicPrior:
        """Aggregate prior for a specific tool.

        Warmup outcomes are excluded.
        """
        indices = self._tool_index.get(tool, [])
        entries = [
            self._outcomes[i] for i in indices
            if i < len(self._outcomes) and not self._outcomes[i].warmup
        ]
        if len(entries) < MIN_ENTRIES_FOR_PRIOR:
            return TopicPrior()

        return self._build_prior(entries)

    @staticmethod
    def _build_prior(entries: list[PolicyOutcome]) -> TopicPrior:
        wins = sum(1 for e in entries if e.worked)
        losses = sum(1 for e in entries if not e.worked and e.net_delta < -MIN_MEANINGFUL_DELTA)
        avg_delta = sum(e.net_delta for e in entries) / len(entries)

        return TopicPrior(
            total=len(entries),
            wins=wins,
            losses=losses,
            avg_delta=avg_delta,
            win_rate=wins / len(entries) if entries else 0.0,
            last_outcome_ts=max(e.timestamp for e in entries),
        )

    def get_avoid_patterns(self) -> list[dict[str, Any]]:
        """Topics with consistent regressions.

        Primary signal: very low win-rate clusters (<20% wins, >=3 outcomes).
        Secondary signal: regression-heavy low-win clusters (<=35% wins,
        >=50% meaningful regressions, negative average delta, >=3 outcomes).

        Only non-warmup outcomes are considered.
        """
        avoid: list[dict[str, Any]] = []
        seen_clusters: set[frozenset[str]] = set()

        for outcome in reversed(self._outcomes):
            if outcome.warmup:
                continue
            cluster = frozenset(outcome.topic_tags)
            if cluster in seen_clusters or not cluster:
                continue
            seen_clusters.add(cluster)

            prior = self.get_topic_prior(outcome.topic_tags)
            if self._is_avoid_prior(prior):
                loss_rate = prior.losses / prior.total if prior.total > 0 else 0.0
                avoid.append({
                    "tags": list(outcome.topic_tags),
                    "total": prior.total,
                    "win_rate": round(prior.win_rate, 2),
                    "loss_rate": round(loss_rate, 2),
                    "avg_delta": round(prior.avg_delta, 4),
                })

        return avoid[:10]

    @staticmethod
    def _is_avoid_prior(prior: TopicPrior) -> bool:
        """True when a topic prior indicates repeated failure patterns."""
        if prior.total < MIN_ENTRIES_FOR_PRIOR:
            return False
        if prior.win_rate < _AVOID_STRICT_WIN_RATE:
            return True
        loss_rate = prior.losses / prior.total if prior.total > 0 else 0.0
        return (
            prior.win_rate <= _AVOID_REGRESSION_HEAVY_WIN_RATE
            and loss_rate >= _AVOID_REGRESSION_HEAVY_LOSS_RATE
            and prior.avg_delta < 0.0
        )

    def score_adjustment(self, tags: tuple[str, ...], tool: str) -> float:
        """Return a score multiplier based on historical outcomes.

        Returns a value in [-0.3, +0.3] to add to the opportunity score.
        Positive = this kind of research has historically worked.
        Negative = this kind of research has historically failed.
        """
        topic_prior = self.get_topic_prior(tags)
        tool_prior = self.get_tool_prior(tool)

        adjustment = 0.0

        if topic_prior.total >= MIN_ENTRIES_FOR_PRIOR:
            topic_signal = (topic_prior.win_rate - 0.5) * 0.4
            adjustment += topic_signal

        if tool_prior.total >= MIN_ENTRIES_FOR_PRIOR:
            tool_signal = (tool_prior.win_rate - 0.5) * 0.2
            adjustment += tool_signal

        return max(-0.3, min(0.3, adjustment))

    def get_stats(self) -> dict[str, Any]:
        non_warmup = [o for o in self._outcomes if not o.warmup]
        total = len(non_warmup)
        wins = sum(1 for o in non_warmup if o.worked)
        warmup_total = sum(1 for o in self._outcomes if o.warmup)
        return {
            "total_outcomes": total,
            "total_wins": wins,
            "total_losses": total - wins,
            "overall_win_rate": round(wins / total, 3) if total > 0 else 0.0,
            "warmup_outcomes": warmup_total,
            "in_warmup": self.is_warmup(),
            "warmup_remaining_s": round(max(0.0, WARMUP_PERIOD_S - (time.time() - self._session_start)), 0),
            "avoid_patterns": self.get_avoid_patterns(),
            "unique_tools": list(self._tool_index.keys()),
            "unique_types": list(self._type_index.keys()),
        }

    def _index_entry(self, idx: int, outcome: PolicyOutcome) -> None:
        for tag in outcome.topic_tags:
            self._tag_index[tag].append(idx)
        self._tool_index[outcome.tool_used].append(idx)
        self._type_index[outcome.intent_type].append(idx)

    def _trim(self) -> None:
        """Keep the newest MAX_ENTRIES * 0.8 entries."""
        keep = int(MAX_ENTRIES * 0.8)
        self._outcomes = self._outcomes[-keep:]
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._tag_index.clear()
        self._tool_index.clear()
        self._type_index.clear()
        for idx, outcome in enumerate(self._outcomes):
            self._index_entry(idx, outcome)

    def _load(self) -> None:
        if not os.path.isfile(_POLICY_PATH):
            return
        try:
            with open(_POLICY_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._outcomes.append(PolicyOutcome.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue
            if len(self._outcomes) > MAX_ENTRIES:
                self._outcomes = self._outcomes[-MAX_ENTRIES:]
            self._rebuild_indices()
            logger.info("Loaded %d autonomy policy outcomes", len(self._outcomes))
        except Exception:
            logger.warning("Failed to load autonomy policy memory", exc_info=True)

    def _persist_append(self, outcome: PolicyOutcome) -> None:
        try:
            os.makedirs(_JARVIS_DIR, exist_ok=True)
            with open(_POLICY_PATH, "a") as f:
                f.write(json.dumps(outcome.to_dict()) + "\n")
            self._maybe_rotate()
        except Exception:
            logger.warning("Failed to persist autonomy outcome", exc_info=True)

    @staticmethod
    def _maybe_rotate() -> None:
        """Trim JSONL to last half when file exceeds size limit."""
        try:
            if not os.path.exists(_POLICY_PATH):
                return
            size = os.path.getsize(_POLICY_PATH)
            if size <= _MAX_JSONL_BYTES:
                return
            with open(_POLICY_PATH, "r") as f:
                lines = f.readlines()
            keep = lines[len(lines) // 2:]
            with open(_POLICY_PATH, "w") as f:
                f.writelines(keep)
            logger.info("Rotated autonomy_policy.jsonl: %d→%d lines", len(lines), len(keep))
        except OSError:
            pass
