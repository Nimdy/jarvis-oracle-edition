"""Eval Harness — record and replay autonomy decisions for offline evaluation.

Records "trace episodes" (inputs + events + metrics + decisions + outcomes) as
JSONL.  Replays autonomy decisions against recorded episodes to compare
policies/versions by net delta, time-to-improve, action count, and regression
rate.

Usage:
    # Record (happens automatically in orchestrator)
    recorder = EpisodeRecorder()
    recorder.record(episode)

    # Replay (offline, e.g. from tests/eval_autonomy.py)
    episodes = EpisodeRecorder.load_all()
    report = replay_and_compare(episodes, scorer_a, scorer_b)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_JARVIS_DIR = os.path.join(Path.home(), ".jarvis")
_EPISODES_DIR = os.path.join(_JARVIS_DIR, "autonomy_episodes")
MAX_EPISODES_ON_DISK = 200


@dataclass
class MetricSnapshot:
    """A single metric observation at a point in time."""

    timestamp: float = 0.0
    confidence_avg: float = 0.0
    confidence_volatility: float = 0.0
    tick_p95_ms: float = 0.0
    reasoning_coherence: float = 0.0
    processing_health: float = 0.0
    memory_count: int = 0
    barge_in_rate: float = 0.0
    error_count: int = 0


@dataclass
class TraceEpisode:
    """A complete recorded episode of one autonomy decision cycle."""

    episode_id: str = ""
    intent_id: str = ""
    question: str = ""
    source_event: str = ""
    tool_hint: str = ""
    tag_cluster: tuple[str, ...] = ()
    trigger_count: int = 0
    scope: str = "local_only"

    metrics_before: list[dict[str, Any]] = field(default_factory=list)
    metrics_after: list[dict[str, Any]] = field(default_factory=list)

    events_during: list[dict[str, Any]] = field(default_factory=list)

    score_breakdown: dict[str, float] = field(default_factory=dict)
    governor_decision: str = ""      # "allowed" | "blocked:reason"
    execution_result: str = ""       # "success" | "failed:reason"

    delta_result: dict[str, Any] = field(default_factory=dict)

    autonomy_level: int = 1
    mode_at_decision: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tag_cluster"] = list(self.tag_cluster)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceEpisode:
        d["tag_cluster"] = tuple(d.get("tag_cluster", ()))
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ReplayResult:
    """Result of replaying a single episode against a scoring function."""

    episode_id: str
    original_score: float
    replayed_score: float
    would_execute: bool
    original_executed: bool
    original_delta: float
    score_diff: float = 0.0


@dataclass
class ComparisonReport:
    """A/B comparison of two scoring policies over a set of episodes."""

    episodes_compared: int = 0
    policy_a_name: str = "A"
    policy_b_name: str = "B"
    a_total_score: float = 0.0
    b_total_score: float = 0.0
    a_would_execute: int = 0
    b_would_execute: int = 0
    a_predicted_delta: float = 0.0
    b_predicted_delta: float = 0.0
    agreement_rate: float = 0.0
    a_unique_picks: int = 0
    b_unique_picks: int = 0
    episodes: list[dict[str, Any]] = field(default_factory=list)


class EpisodeRecorder:
    """Records autonomy episodes to disk as JSONL."""

    def __init__(self, episodes_dir: str = _EPISODES_DIR) -> None:
        self._dir = episodes_dir
        self._episode_count = 0

    def record(self, episode: TraceEpisode) -> None:
        try:
            os.makedirs(self._dir, exist_ok=True)
            date_str = time.strftime("%Y-%m-%d")
            path = os.path.join(self._dir, f"episodes_{date_str}.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(episode.to_dict()) + "\n")
            self._episode_count += 1
            self._maybe_prune()
        except Exception:
            logger.warning("Failed to record autonomy episode", exc_info=True)

    def load_all(self) -> list[TraceEpisode]:
        episodes: list[TraceEpisode] = []
        if not os.path.isdir(self._dir):
            return episodes
        for fname in sorted(os.listdir(self._dir)):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            episodes.append(TraceEpisode.from_dict(json.loads(line)))
            except Exception:
                logger.warning("Failed to load episode file %s", fname, exc_info=True)
        return episodes

    def load_recent(self, max_episodes: int = 50) -> list[TraceEpisode]:
        all_eps = self.load_all()
        return all_eps[-max_episodes:]

    def get_stats(self) -> dict[str, Any]:
        count = 0
        successful = 0
        if os.path.isdir(self._dir):
            for fname in os.listdir(self._dir):
                if fname.endswith(".jsonl"):
                    path = os.path.join(self._dir, fname)
                    try:
                        with open(path) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                count += 1
                                try:
                                    row = json.loads(line)
                                except Exception:
                                    continue
                                if row.get("execution_result") == "success":
                                    successful += 1
                    except Exception:
                        pass
        return {
            "total_episodes": count,
            "successful_episodes": successful,
            "session_recorded": self._episode_count,
            "storage_dir": self._dir,
        }

    def _maybe_prune(self) -> None:
        """Remove oldest episode files if total exceeds MAX_EPISODES_ON_DISK."""
        if not os.path.isdir(self._dir):
            return
        files = sorted(
            (f for f in os.listdir(self._dir) if f.endswith(".jsonl")),
        )
        total = 0
        file_counts: list[tuple[str, int]] = []
        for fname in files:
            path = os.path.join(self._dir, fname)
            try:
                with open(path) as f:
                    c = sum(1 for line in f if line.strip())
                file_counts.append((fname, c))
                total += c
            except Exception:
                pass

        while total > MAX_EPISODES_ON_DISK and file_counts:
            oldest_name, oldest_count = file_counts.pop(0)
            try:
                os.remove(os.path.join(self._dir, oldest_name))
                total -= oldest_count
            except Exception:
                break


ScorerFn = Callable[[TraceEpisode], float]


def replay_episodes(
    episodes: list[TraceEpisode],
    scorer: ScorerFn,
    min_score_to_execute: float = 0.05,
) -> list[ReplayResult]:
    """Replay a list of recorded episodes against a scoring function.

    The scorer receives a TraceEpisode and returns a float score (0-1).
    """
    results: list[ReplayResult] = []
    for ep in episodes:
        original_score = ep.score_breakdown.get("total", 0.0)
        original_executed = ep.governor_decision == "allowed"
        original_delta = ep.delta_result.get("net_improvement", 0.0)

        replayed_score = scorer(ep)
        would_execute = replayed_score >= min_score_to_execute

        results.append(ReplayResult(
            episode_id=ep.episode_id,
            original_score=original_score,
            replayed_score=replayed_score,
            would_execute=would_execute,
            original_executed=original_executed,
            original_delta=original_delta,
            score_diff=replayed_score - original_score,
        ))
    return results


def compare_policies(
    episodes: list[TraceEpisode],
    scorer_a: ScorerFn,
    scorer_b: ScorerFn,
    name_a: str = "A",
    name_b: str = "B",
    min_score: float = 0.05,
) -> ComparisonReport:
    """Compare two scoring policies over the same set of recorded episodes.

    Computes which policy would have executed which episodes, and what the
    predicted net delta would be (using the original measured deltas as
    ground truth for episodes that were actually executed).
    """
    report = ComparisonReport(
        episodes_compared=len(episodes),
        policy_a_name=name_a,
        policy_b_name=name_b,
    )

    for ep in episodes:
        sa = scorer_a(ep)
        sb = scorer_b(ep)
        a_exec = sa >= min_score
        b_exec = sb >= min_score
        measured_delta = ep.delta_result.get("net_improvement", 0.0)

        report.a_total_score += sa
        report.b_total_score += sb
        if a_exec:
            report.a_would_execute += 1
            if ep.governor_decision == "allowed":
                report.a_predicted_delta += measured_delta
        if b_exec:
            report.b_would_execute += 1
            if ep.governor_decision == "allowed":
                report.b_predicted_delta += measured_delta
        if a_exec == b_exec:
            report.agreement_rate += 1
        if a_exec and not b_exec:
            report.a_unique_picks += 1
        if b_exec and not a_exec:
            report.b_unique_picks += 1

        report.episodes.append({
            "id": ep.episode_id,
            "score_a": round(sa, 4),
            "score_b": round(sb, 4),
            "a_exec": a_exec,
            "b_exec": b_exec,
            "measured_delta": round(measured_delta, 4),
        })

    n = len(episodes)
    if n > 0:
        report.agreement_rate = round(report.agreement_rate / n, 3)
        report.a_total_score = round(report.a_total_score / n, 4)
        report.b_total_score = round(report.b_total_score / n, 4)

    return report
