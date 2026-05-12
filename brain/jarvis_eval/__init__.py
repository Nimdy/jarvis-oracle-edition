"""Jarvis Evaluation Shadow Pipeline.

The eval sidecar is a read-only observer. It subscribes to events,
snapshots subsystem stats, writes to its own isolated JSONL store
under ~/.jarvis/eval/, and powers dashboard panels. It NEVER writes
to memory, beliefs, dreams, policy, or self-report paths.

The Process Verification Layer (PVL) extends the sidecar with
architectural contract assertions — proving every process fires
and produces expected output during real operation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from jarvis_eval.collector import EvalCollector
from jarvis_eval.config import (
    FLUSH_INTERVAL_S, SCORING_VERSION, SCENARIO_PACK_VERSION,
    ORACLE_SCORECARD_INTERVAL_S, PVL_VERIFY_EVERY_N_FLUSHES, PVL_EVENT_WINDOW,
)
from jarvis_eval.contracts import EvalRun, EvalScorecard
from jarvis_eval.dashboard_adapter import build_dashboard_snapshot
from jarvis_eval.event_tap import EvalEventTap
from jarvis_eval.process_verifier import ProcessVerifier
from jarvis_eval.report import build_report
from jarvis_eval.scorecards import build_oracle_scorecard
from jarvis_eval.store import EvalStore

logger = logging.getLogger(__name__)


def _latest_snapshot_metrics_by_source(
    recent_snapshots: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select the newest snapshot per source using explicit timestamps."""
    latest: dict[str, tuple[float, dict[str, Any]]] = {}
    for snap in recent_snapshots:
        src = snap.get("source", "")
        ts = float(snap.get("timestamp", 0.0) or 0.0)
        metrics = snap.get("metrics", {})
        prev = latest.get(src)
        if prev is None or ts >= prev[0]:
            latest[src] = (ts, metrics if isinstance(metrics, dict) else {})
    return {src: metrics for src, (_ts, metrics) in latest.items()}

__all__ = ["EvalSidecar"]


class EvalSidecar:
    """Facade coordinating event tap, collector, store, verifier, and dashboard adapter."""

    def __init__(self) -> None:
        self._store = EvalStore()
        self._tap = EvalEventTap()
        self._verifier = ProcessVerifier()
        self._collector: EvalCollector | None = None
        self._run: EvalRun | None = None
        self._started_at: float = 0.0
        self._flush_task: asyncio.Task | None = None
        self._running: bool = False
        self._flush_count: int = 0
        self._last_scorecard_ts: float = 0.0

    def start(self, engine: Any) -> None:
        """Wire event tap, start collector, begin flush loop."""
        if self._running:
            return

        self._run = EvalRun(
            mode="shadow",
            scoring_version=SCORING_VERSION,
            scenario_pack_version=SCENARIO_PACK_VERSION,
        )
        self._started_at = time.time()
        self._store.append_run(self._run)

        self._tap.wire(run_id=self._run.run_id)

        self._seed_mode_from_engine(engine)

        self._collector = EvalCollector(engine)
        self._last_scorecard_ts = float(self._store.get_meta().get("last_scorecard_ts") or 0.0)
        try:
            loop = asyncio.get_event_loop()
            self._collector.start(run_id=self._run.run_id, loop=loop)
            self._flush_task = loop.create_task(self._flush_loop())
        except RuntimeError:
            logger.warning("Eval sidecar: no event loop, collector deferred")

        self._running = True
        logger.info(
            "Eval sidecar started (run_id=%s, scoring=%s, pvl=enabled)",
            self._run.run_id, SCORING_VERSION,
        )

    def stop(self) -> None:
        """Unwire event tap, stop collector, final flush, close run."""
        self._tap.unwire()
        if self._collector:
            self._collector.stop()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_once()
        if self._run:
            self._run.ended_at = time.time()
            self._store.close_run(self._run)
        self._running = False
        logger.info("Eval sidecar stopped")

    async def _flush_loop(self) -> None:
        """Background loop: drain event buffer -> store, collect snapshots -> store, run PVL."""
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL_S)
                self._flush_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("Eval flush loop error", exc_info=True)

    def _flush_once(self) -> None:
        """Drain events, write snapshots, run PVL verification, update meta."""
        try:
            events = self._tap.drain()
            for ev in events:
                self._store.append_event(ev)
        except Exception:
            logger.warning("Eval flush: event drain failed", exc_info=True)

        try:
            if self._collector:
                snapshots = self._collector.collect_once()
                for snap in snapshots:
                    self._store.append_snapshot(snap)
        except Exception:
            logger.warning("Eval flush: snapshot collect failed", exc_info=True)

        self._flush_count += 1
        if self._flush_count % PVL_VERIFY_EVERY_N_FLUSHES == 0:
            try:
                self._run_pvl_verification()
            except Exception:
                logger.warning("Eval flush: PVL verification failed", exc_info=True)

        try:
            self._maybe_append_scorecard()
        except Exception:
            logger.warning("Eval flush: scorecard write failed", exc_info=True)

        try:
            self._store.flush_meta()
        except Exception:
            logger.warning("Eval flush: meta write failed", exc_info=True)

    def _seed_mode_from_engine(self, engine: Any) -> None:
        """Seed the tap's current mode and replay missed boot events.

        Several events fire during startup before the eval sidecar wires
        its event tap (e.g. MODE_CHANGE, GESTATION_STARTED). Without
        replaying them, the PVL verifier never sees boot-time events and
        contracts like ``gestation_started`` and ``mode_managed`` stay in
        FAIL even though the processes ran correctly.

        This method reads the engine state at wire time and injects
        synthetic EvalEvent records for boot events that already occurred.
        """
        try:
            state = engine.get_state()
            if not isinstance(state, dict):
                return
        except Exception:
            return

        mode = state.get("mode", "")
        if mode:
            self._tap._current_mode = mode
            logger.debug("Eval sidecar: seeded mode from engine state: %s", mode)

        from jarvis_eval.contracts import EvalEvent

        run_id = self._run.run_id if self._run else ""
        now = time.time()

        if mode:
            self._tap._buffer.append(EvalEvent(
                event_type="mode:change",
                payload={"to_mode": mode, "reason": "boot_seed"},
                mode=mode,
                run_id=run_id,
                timestamp=now,
            ))
            self._tap._events_buffered += 1
            logger.debug("Eval sidecar: injected synthetic mode:change → %s", mode)

        gestation_mgr = getattr(engine, "gestation", None)
        if gestation_mgr is not None:
            is_active = getattr(gestation_mgr, "is_active", False)
            if is_active:
                started_at = getattr(gestation_mgr, "_started_at", now)
                self._tap._buffer.append(EvalEvent(
                    event_type="gestation:started",
                    payload={"resumed": False, "reason": "boot_seed"},
                    mode=mode,
                    run_id=run_id,
                    timestamp=started_at or now,
                ))
                self._tap._events_buffered += 1
                logger.debug("Eval sidecar: injected synthetic gestation:started")

    def _run_pvl_verification(self) -> None:
        """Run one PVL verification pass."""
        if not self._verifier._hydrated:
            try:
                all_events = self._store.read_all_events()
                self._verifier.hydrate_from_history(all_events)
            except Exception:
                logger.warning("PVL hydration failed", exc_info=True)

        recent_events = self._store.read_recent_events(limit=PVL_EVENT_WINDOW)
        recent_snapshots = self._store.read_recent_snapshots(limit=200)

        latest_by_source = _latest_snapshot_metrics_by_source(recent_snapshots)

        current_mode = self._tap.get_stats().get("current_mode", "")

        if not current_mode:
            engine_state = latest_by_source.get("engine_state", {})
            current_mode = engine_state.get("mode", "")

        self._verifier.verify(
            recent_events=recent_events,
            latest_snapshots=latest_by_source,
            current_mode=current_mode,
        )

    def _maybe_append_scorecard(self) -> None:
        if not self._run:
            return
        now = time.time()
        if self._last_scorecard_ts and now - self._last_scorecard_ts < ORACLE_SCORECARD_INTERVAL_S:
            return

        recent_snapshots = self._store.read_recent_snapshots(limit=600)
        latest_by_source = _latest_snapshot_metrics_by_source(recent_snapshots)

        metrics = build_oracle_scorecard(
            latest_by_source=latest_by_source,
            pvl_result=self._verifier.get_last_result().to_dict() if self._verifier.get_last_result() else None,
        )
        scorecard = EvalScorecard(
            metrics=metrics,
            notes=list(metrics.get("notes", [])),
            run_id=self._run.run_id,
        )
        self._store.append_scorecard(scorecard)
        self._last_scorecard_ts = scorecard.timestamp

    def get_dashboard_snapshot(self, main_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build dashboard-ready dict from current data."""
        try:
            pvl_result = self._verifier.get_last_result()
            return build_dashboard_snapshot(
                store_meta=self._store.get_meta(),
                store_file_sizes=self._store.get_file_sizes(),
                recent_snapshots=self._store.read_recent_snapshots(limit=200),
                recent_scorecards=self._store.read_recent_scorecards(limit=128),
                recent_events=self._store.read_recent_events(limit=500),
                recent_scores=self._store.read_recent_scores(limit=20),
                collector_stats=self._collector.get_stats() if self._collector else {},
                tap_stats=self._tap.get_stats(),
                pvl_result=pvl_result.to_dict() if pvl_result else None,
                pvl_stats=self._verifier.get_stats(),
                main_snapshot=main_snapshot,
            )
        except Exception:
            logger.warning("Eval dashboard snapshot failed", exc_info=True)
            return {}

    def get_report(self) -> dict[str, Any]:
        """Build Phase A report."""
        return build_report(self._store, self._started_at)

    @property
    def running(self) -> bool:
        return self._running


_sidecar: EvalSidecar | None = None


def get_eval_sidecar() -> EvalSidecar:
    """Get or create the eval sidecar singleton."""
    global _sidecar
    if _sidecar is None:
        _sidecar = EvalSidecar()
    return _sidecar
