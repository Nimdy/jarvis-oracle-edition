"""FastAPI dashboard — REST API + WebSocket live updates.

Performance contract:
  - All endpoints read from a pre-built snapshot cache (_cache)
  - The cache is rebuilt every CACHE_INTERVAL_S by _snapshot_loop
  - No endpoint ever reaches into the consciousness system directly
  - WebSocket pushes only when state actually changed (hash-diffed)
  - Heavy detail endpoints (thoughts, mutations, dialogues) are on
    separate longer poll cycles client-side
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sys
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

_DASHBOARD_API_KEY = os.environ.get("DASHBOARD_API_KEY", "")
if not _DASHBOARD_API_KEY:
    _DASHBOARD_API_KEY = secrets.token_urlsafe(32)
_ENABLE_DASHBOARD_CHAT = str(
    os.environ.get("ENABLE_DASHBOARD_CHAT", "false"),
).strip().lower() in ("1", "true", "yes", "on")


def _require_api_key(request: Request) -> None:
    """Dependency that checks Bearer token for destructive endpoints."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.query_params.get("api_key", "")
    if not secrets.compare_digest(token, _DASHBOARD_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

from consciousness.engine import ConsciousnessEngine
from consciousness.events import (
    event_bus,
    PERCEPTION_BARGE_IN,
    CONVERSATION_RESPONSE,
    KERNEL_ERROR,
    CONSCIOUSNESS_ANALYSIS,
    AUTONOMY_L3_ACTIVATION_DENIED,
)
from consciousness.modes import mode_manager, MODE_CHANGE
from reasoning.response import ResponseGenerator
from perception.server import PerceptionServer

logger = logging.getLogger(__name__)

# Module-load timestamp used by /api/meta/build-status so the frontend can
# distinguish "on-disk page mtime > process start" (stale page) from "on-disk
# page mtime <= process start" (page is what this process is serving).
_STARTUP_TS = time.time()


# ---------------------------------------------------------------------------
# Operational health counters (monotonic counters + EMAs)
# ---------------------------------------------------------------------------

class _HealthCounters:
    """O(1) write, O(1) read operational counters for the dashboard."""

    _EMA_ALPHA = 0.15

    def __init__(self) -> None:
        self.barge_in_count: int = 0
        self.response_count: int = 0
        self.error_count: int = 0
        self.mode_transition_count: int = 0
        self.analysis_count: int = 0
        self._response_latency_ema: float = 0.0
        self._started = time.time()

    def start(self) -> None:
        event_bus.on(PERCEPTION_BARGE_IN, self._on_barge_in)
        event_bus.on(CONVERSATION_RESPONSE, self._on_response)
        event_bus.on(KERNEL_ERROR, self._on_error)
        event_bus.on(MODE_CHANGE, self._on_mode_change)
        event_bus.on(CONSCIOUSNESS_ANALYSIS, self._on_analysis)

    def _on_barge_in(self, **_) -> None:
        self.barge_in_count += 1

    def _on_response(self, latency_ms: float = 0.0, **_) -> None:
        self.response_count += 1
        if latency_ms > 0:
            self.record_response_latency(latency_ms)

    def _on_error(self, **_) -> None:
        self.error_count += 1

    def _on_mode_change(self, **_) -> None:
        self.mode_transition_count += 1

    def _on_analysis(self, **_) -> None:
        self.analysis_count += 1

    def record_response_latency(self, ms: float) -> None:
        if self._response_latency_ema == 0.0:
            self._response_latency_ema = ms
        else:
            self._response_latency_ema += self._EMA_ALPHA * (ms - self._response_latency_ema)

    def snapshot(self) -> dict[str, Any]:
        return {
            "barge_in_count": self.barge_in_count,
            "response_count": self.response_count,
            "error_count": self.error_count,
            "mode_transition_count": self.mode_transition_count,
            "analysis_count": self.analysis_count,
            "avg_response_latency_ms": round(self._response_latency_ema, 1),
            "uptime_s": round(time.time() - self._started, 1),
        }


_health = _HealthCounters()

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Process start + on-disk source tree — used to detect "newer code on disk"
# after a `sync-desktop.sh` so the operator can click Restart before trusting
# the dashboard's claims about behaviour.
_PROCESS_STARTED_TS: float = time.time()
_BRAIN_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_FRESHNESS_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}}
_CODE_FRESHNESS_TTL_S: float = 30.0
_CODE_SCAN_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "improvement_snapshots", "kernel_snapshots", "hemispheres",
    "policy_models", "synthetic_exercise",
})
_CODE_SCAN_EXTENSIONS: tuple[str, ...] = (".py",)


def _scan_code_freshness() -> dict[str, Any]:
    """Walk the brain source tree, return newest-mtime metadata.

    Cached for _CODE_FRESHNESS_TTL_S to keep cost bounded; the banner is a
    low-stakes "you just synced, consider restarting" nudge, not a control
    surface, so stale-within-30s is acceptable.

    Never raises — on any error returns a safe "unknown" dict.
    """
    now = time.time()
    cache = _CODE_FRESHNESS_CACHE
    if cache["data"] and (now - cache["ts"]) < _CODE_FRESHNESS_TTL_S:
        return cache["data"]

    newest_mtime = 0.0
    newest_file = ""
    file_count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(_BRAIN_ROOT, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") or d in (".jarvis",)]
            dirnames[:] = [d for d in dirnames if d not in _CODE_SCAN_SKIP_DIRS]
            for fname in filenames:
                if not fname.endswith(_CODE_SCAN_EXTENSIONS):
                    continue
                full = os.path.join(dirpath, fname)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                file_count += 1
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_file = os.path.relpath(full, _BRAIN_ROOT)
    except Exception:
        logger.debug("code freshness scan failed", exc_info=True)
        data = {
            "process_started_ts": _PROCESS_STARTED_TS,
            "newest_mtime": 0.0,
            "newest_file": "",
            "file_count": 0,
            "is_stale": False,
            "stale_age_s": 0.0,
            "scan_ok": False,
        }
        cache["ts"] = now
        cache["data"] = data
        return data

    is_stale = newest_mtime > _PROCESS_STARTED_TS
    stale_age_s = max(0.0, newest_mtime - _PROCESS_STARTED_TS) if is_stale else 0.0
    data = {
        "process_started_ts": _PROCESS_STARTED_TS,
        "newest_mtime": newest_mtime,
        "newest_file": newest_file,
        "file_count": file_count,
        "is_stale": is_stale,
        "stale_age_s": stale_age_s,
        "scan_ok": True,
    }
    cache["ts"] = now
    cache["data"] = data
    return data

CACHE_INTERVAL_S = 2.0
WS_PUSH_INTERVAL_S = 1.0

_ws_clients: list[WebSocket] = []
_engine: ConsciousnessEngine | None = None
_response_gen: ResponseGenerator | None = None
_perception: PerceptionServer | None = None
_attention_core: Any = None
_pi_video_url: str = ""
_persistence: Any = None
_episodes: Any = None
_processors: dict | None = None
_perc_orch: Any = None
_shutting_down: bool = False

# ---------------------------------------------------------------------------
# Snapshot cache — all dashboard reads come from here, never the live system
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}
_cache_hash: str = ""
_cache_time: float = 0.0


from dashboard.snapshot import build_cache as _build_snapshot, SnapshotContext


def _build_cache() -> dict[str, Any]:
    """Build the full dashboard snapshot from engine state. Called on timer only."""
    global _cache, _cache_hash, _cache_time

    if not _engine:
        return {}

    ctx = SnapshotContext(
        engine=_engine,
        perc_orch=_perc_orch,
        attention_core=_attention_core,
        perception=_perception,
        health_counters=_health,
    )
    snapshot, new_hash = _build_snapshot(ctx)
    if snapshot:
        _cache = snapshot
        _cache_hash = new_hash
        _cache_time = time.time()
    return snapshot


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _create_app() -> FastAPI:
    app = FastAPI(title="Jarvis Brain Dashboard", docs_url=None, redoc_url=None)

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = os.path.join(_STATIC_DIR, "index.html")
        with open(index_path) as f:
            return HTMLResponse(f.read())

    @app.get("/docs", response_class=HTMLResponse)
    async def docs_page():
        docs_path = os.path.join(_STATIC_DIR, "docs.html")
        with open(docs_path) as f:
            return HTMLResponse(f.read())

    @app.get("/history", response_class=HTMLResponse)
    async def history_page():
        history_path = os.path.join(_STATIC_DIR, "history.html")
        with open(history_path) as f:
            return HTMLResponse(f.read())

    @app.get("/api-reference", response_class=HTMLResponse)
    async def api_reference_page():
        api_path = os.path.join(_STATIC_DIR, "api.html")
        with open(api_path) as f:
            return HTMLResponse(f.read())

    @app.get("/science", response_class=HTMLResponse)
    async def science_page():
        science_path = os.path.join(_STATIC_DIR, "science.html")
        with open(science_path) as f:
            return HTMLResponse(f.read())

    @app.get("/showcase", response_class=HTMLResponse)
    async def showcase_page():
        showcase_path = os.path.join(_STATIC_DIR, "showcase.html")
        with open(showcase_path) as f:
            return HTMLResponse(f.read())

    @app.get("/learning", response_class=HTMLResponse)
    async def learning_page():
        learning_path = os.path.join(_STATIC_DIR, "learning.html")
        with open(learning_path) as f:
            return HTMLResponse(f.read())

    def _capability_pipeline_response() -> HTMLResponse:
        si_path = os.path.join(_STATIC_DIR, "self_improve.html")
        with open(si_path) as f:
            return HTMLResponse(f.read())

    @app.get("/capability-pipeline", response_class=HTMLResponse)
    async def capability_pipeline_page():
        return _capability_pipeline_response()

    @app.get("/self-improve", response_class=HTMLResponse)
    async def self_improve_page():
        """Backward-compatible alias for the broader capability pipeline page."""
        return _capability_pipeline_response()

    @app.get("/hrr", response_class=HTMLResponse)
    async def hrr_page():
        """Dedicated live dashboard for the P4 HRR / VSA shadow substrate.

        Purely observational — the page polls ``/api/hrr/status`` and
        ``/api/hrr/samples`` on a timer. All authority flags rendered here
        come straight from the server payload (never computed client-side).
        """
        hrr_path = os.path.join(_STATIC_DIR, "hrr.html")
        with open(hrr_path) as f:
            return HTMLResponse(f.read())

    @app.get("/hrr-scene", response_class=HTMLResponse)
    async def hrr_scene_page():
        """Dedicated live dashboard for the P5 mental-world spatial HRR lane.

        Purely observational — the page polls ``/api/hrr/scene`` on a
        timer and renders the derived scene graph as a Matrix-style
        node/edge canvas. All authority flags come straight from the
        server payload (never computed client-side).
        """
        hrr_scene_path = os.path.join(_STATIC_DIR, "hrr_scene.html")
        with open(hrr_scene_path) as f:
            return HTMLResponse(f.read())

    @app.get("/eval")
    async def eval_page():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/?tab=trust", status_code=302)

    @app.get("/api/eval/snapshot")
    async def api_eval_snapshot():
        return _cache.get("eval", {})

    @app.get("/api/eval/benchmark")
    async def api_eval_benchmark():
        return _cache.get("eval", {}).get("oracle_benchmark", {})

    @app.get("/api/eval/autonomy-ab")
    async def api_eval_autonomy_ab():
        """Autonomy A/B comparison history + intervention stats."""
        result: dict[str, Any] = {
            "comparisons": [],
            "interventions": {},
            "friction": {},
            "source_usefulness": {},
        }
        try:
            import json as _j
            comp_path = os.path.join(os.path.expanduser("~/.jarvis"), "eval_comparisons.jsonl")
            if os.path.exists(comp_path):
                with open(comp_path, "r") as f:
                    lines = f.readlines()
                for line in lines[-20:]:
                    line = line.strip()
                    if line:
                        result["comparisons"].append(_j.loads(line))
        except Exception:
            pass
        try:
            from autonomy.intervention_runner import get_intervention_runner
            result["interventions"] = get_intervention_runner().get_stats()
        except Exception:
            pass
        try:
            from autonomy.friction_miner import get_friction_miner
            result["friction"] = get_friction_miner().get_stats()
        except Exception:
            pass
        try:
            from autonomy.source_ledger import get_source_ledger
            result["source_usefulness"] = get_source_ledger().get_stats()
        except Exception:
            pass
        return result

    @app.get("/api/language-corpus/stats")
    async def api_language_corpus_stats():
        """Language corpus evidence snapshot for backfill verification."""
        try:
            from reasoning.language_corpus import language_corpus
            from reasoning.language_telemetry import language_quality_telemetry
            stats = language_corpus.get_stats()
            total = stats.get("total_examples", 0)
            by_provenance = stats.get("counts_by_provenance", {})
            by_flag = stats.get("counts_by_safety_flag", {})
            by_feedback = stats.get("counts_by_feedback", {})
            provenance_total = sum(by_provenance.values())
            phase_c = {}
            try:
                from reasoning.language_phasec import get_phasec_status
                phase_c = get_phasec_status()
            except Exception:
                phase_c = {}
            return {
                "total_examples": total,
                "provenance_pct": round(provenance_total / total * 100, 1) if total > 0 else 0.0,
                "counts_by_route": stats.get("counts_by_route", {}),
                "counts_by_response_class": stats.get("counts_by_response_class", {}),
                "counts_by_provenance": by_provenance,
                "counts_by_feedback": by_feedback,
                "negative_count": by_flag.get("capability_gate_rewrite", 0) + by_flag.get("negative", 0),
                "rewrite_count": by_flag.get("capability_gate_rewrite", 0),
                "fail_closed_count": by_flag.get("fail_closed", 0),
                "counts_by_safety_flag": by_flag,
                "quality": language_quality_telemetry.get_stats(),
                "last_capture_ts": stats.get("last_capture_ts", 0.0),
                "phase_c": phase_c,
            }
        except Exception:
            return {"error": "Language corpus not available"}

    @app.get("/api/world-model/diagnostics")
    async def api_world_model_diagnostics():
        """Combined legacy + canonical world-model diagnostics."""
        cs = _engine._consciousness if _engine else None
        if cs and cs._world_model:
            return cs._world_model.get_diagnostics()
        return {"error": "World model not available"}

    @app.get("/api/spatial/diagnostics")
    async def api_spatial_diagnostics():
        """Spatial intelligence diagnostics — calibration, tracks, anchors, validation."""
        return _cache.get("spatial", {
            "status": "not_initialized",
            "calibration": {},
            "estimator": {},
            "validation": {},
        })

    @app.post("/api/spatial/calibration", dependencies=[Depends(_require_api_key)])
    async def api_spatial_calibration_update(request: Request):
        """Update spatial calibration parameters.

        Body JSON (all fields optional):
          focal_length_px: float
          frame_width: int
          frame_height: int
          camera_position_m: [x, y, z]
          camera_rotation_rpy: [r, p, y]
        """
        po = _perc_orch
        if po is None:
            raise HTTPException(503, "Perception orchestrator not ready")
        cal = getattr(po, "_calibration_manager", None)
        if cal is None:
            raise HTTPException(503, "Calibration manager not available")
        body = await request.json()
        if "focal_length_px" in body or "frame_width" in body or "frame_height" in body:
            cal.update_intrinsics(
                focal_length_px=body.get("focal_length_px", cal.intrinsics.focal_length_px),
                principal_x=body.get("principal_x"),
                principal_y=body.get("principal_y"),
                frame_width=body.get("frame_width"),
                frame_height=body.get("frame_height"),
            )
        if "camera_position_m" in body:
            cal.update_transform(
                camera_position_m=tuple(body["camera_position_m"]),
                camera_rotation_rpy=tuple(body.get("camera_rotation_rpy", [0, 0, 0])),
            )
        cal.verify()
        return {"ok": True, "calibration": cal.get_state()}

    # -- lightweight endpoints: all read from _cache -------------------------

    @app.get("/api/status")
    async def api_status():
        if not _cache:
            return JSONResponse({"error": "Engine not ready"}, status_code=503)
        c = _cache
        return {
            "running": c["core"]["running"],
            **c["core"],
            "memory": c["memory"],
            "consciousness": c["consciousness"],
            "sensors": c["sensors"],
            "truth_calibration": c.get("truth_calibration", {}),
        }

    @app.get("/api/sensor-health")
    async def api_sensor_health():
        return _cache.get("sensor_health", {})

    @app.get("/api/synthetic-exercise")
    async def api_synthetic_exercise():
        return _cache.get("synthetic_exercise", {"active": False})

    @app.get("/api/synthetic/skill-acquisition/status")
    async def api_synthetic_skill_acquisition_status():
        try:
            from synthetic.skill_acquisition_dashboard import get_skill_acquisition_weight_room_status

            return get_skill_acquisition_weight_room_status(
                engine=_engine,
                startup_ts=_STARTUP_TS,
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/synthetic/skill-acquisition/run", dependencies=[Depends(_require_api_key)])
    async def api_synthetic_skill_acquisition_run(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        profile = str(body.get("profile") or "smoke").strip()
        count = body.get("count")
        seed = body.get("seed")
        try:
            from synthetic.skill_acquisition_dashboard import start_skill_acquisition_weight_room_run

            result = start_skill_acquisition_weight_room_run(
                profile,
                engine=_engine,
                startup_ts=_STARTUP_TS,
                count=int(count) if count is not None else None,
                seed=int(seed) if seed is not None else None,
                no_record=bool(body.get("no_record", False)),
            )
            if not result.get("started") and not result.get("allowed", True):
                return JSONResponse(result, status_code=409)
            return result
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/synthetic/exercises/status")
    async def api_synthetic_exercises_status():
        """Status for all text-only synthetic exercises (from snapshot cache)."""
        return _cache.get("synthetic_exercises", {"exercises": {}, "count": 0})

    _synthetic_exercise_status: dict[str, dict] = {}

    @app.get("/api/synthetic/exercises/running")
    async def api_synthetic_exercises_running():
        """Live running status for text-only synthetic exercises."""
        return _synthetic_exercise_status

    @app.post("/api/synthetic/exercises/run", dependencies=[Depends(_require_api_key)])
    async def api_synthetic_exercises_run(request: Request):
        """Run a specific text-only synthetic exercise in a background thread."""
        import threading

        try:
            body = await request.json()
        except Exception:
            body = {}
        exercise = str(body.get("exercise", "")).strip()
        profile = str(body.get("profile", "smoke")).strip()
        seed = body.get("seed")

        runners = {
            "commitment": ("synthetic.commitment_exercise", "run_commitment_exercise"),
            "claim": ("synthetic.claim_exercise", "run_claim_exercise"),
            "retrieval": ("synthetic.retrieval_exercise", "run_retrieval_exercise"),
            "world_model": ("synthetic.world_model_exercise", "run_world_model_exercise"),
            "contradiction": ("synthetic.contradiction_exercise", "run_contradiction_exercise"),
            "diagnostic": ("synthetic.diagnostic_exercise", "run_diagnostic_exercise"),
            "plan_evaluator": ("synthetic.plan_evaluator_exercise", "run_plan_evaluator_exercise"),
        }
        if exercise not in runners:
            return JSONResponse(
                {"error": f"Unknown exercise: {exercise}", "available": list(runners.keys())},
                status_code=400,
            )

        if exercise in _synthetic_exercise_status and _synthetic_exercise_status[exercise].get("running"):
            return JSONResponse(
                {"error": f"Exercise '{exercise}' is already running"},
                status_code=409,
            )

        import importlib

        mod_name, func_name = runners[exercise]
        try:
            mod = importlib.import_module(mod_name)
            run_fn = getattr(mod, func_name)
        except Exception as exc:
            return JSONResponse({"error": f"Cannot load exercise: {exc}"}, status_code=500)

        try:
            profiles_attr = getattr(mod, "PROFILES", None)
            if profiles_attr is None:
                profiles_attr = getattr(mod, f"{exercise.upper()}_PROFILES", None)
            profile_obj = None
            if profiles_attr and isinstance(profiles_attr, dict):
                profile_obj = profiles_attr.get(profile)
            if profile_obj is None:
                profile_enum = getattr(mod, f"{exercise.title().replace('_', '')}ExerciseProfile", None)
                if profile_enum is None:
                    for attr_name in dir(mod):
                        obj = getattr(mod, attr_name, None)
                        if isinstance(obj, type) and hasattr(obj, "name") and (
                            hasattr(obj, "episode_count") or hasattr(obj, "count")
                        ):
                            profile_enum = obj
                            break
                if profile_enum:
                    for member in profile_enum:
                        if member.name == profile:
                            profile_obj = member
                            break
        except Exception as exc:
            return JSONResponse({"error": f"Profile resolution failed: {exc}"}, status_code=500)

        def _run():
            import json as _json
            from pathlib import Path

            _synthetic_exercise_status[exercise] = {
                "running": True, "profile": profile, "started_at": time.time(),
            }
            report_dir = Path.home() / ".jarvis" / "synthetic_exercise" / f"{exercise}_reports"
            try:
                kwargs: dict = {}
                if profile_obj is not None:
                    kwargs["profile"] = profile_obj
                if seed is not None:
                    kwargs["seed"] = int(seed)
                stats = run_fn(**kwargs)

                report: dict = {"exercise": exercise, "profile": profile, "ts": time.time()}
                if hasattr(stats, "to_dict"):
                    report.update(stats.to_dict())
                elif hasattr(stats, "__dict__"):
                    for k, v in stats.__dict__.items():
                        if isinstance(v, (str, int, float, bool, type(None))):
                            report[k] = v
                        elif isinstance(v, dict):
                            report[k] = {str(kk): vv for kk, vv in v.items()
                                          if isinstance(vv, (str, int, float, bool, type(None)))}
                        elif isinstance(v, (list, tuple)) and len(v) <= 20:
                            report[k] = list(v)

                passed = True
                if hasattr(stats, "fail_reasons") and stats.fail_reasons:
                    passed = False
                elif hasattr(stats, "pass_result"):
                    passed = bool(stats.pass_result)
                elif hasattr(stats, "passed"):
                    passed = bool(stats.passed)
                elif hasattr(stats, "accuracy"):
                    passed = stats.accuracy >= 0.90
                report["passed"] = passed

                report_dir.mkdir(parents=True, exist_ok=True)
                rpath = report_dir / f"{int(time.time())}_{profile}.json"
                rpath.write_text(_json.dumps(report, indent=2, default=str), encoding="utf-8")
                logger.info("Synthetic exercise %s/%s completed (passed=%s) -> %s", exercise, profile, passed, rpath.name)
            except Exception:
                logger.warning("Synthetic exercise %s/%s failed", exercise, profile, exc_info=True)
            finally:
                _synthetic_exercise_status.pop(exercise, None)

        t = threading.Thread(target=_run, name=f"synth-{exercise}-{profile}", daemon=True)
        t.start()
        return {"started": True, "exercise": exercise, "profile": profile}

    @app.get("/api/personality")
    async def api_personality():
        return _cache.get("personality", {})

    @app.get("/api/consciousness")
    async def api_consciousness():
        return _cache.get("consciousness", {})

    @app.get("/api/consciousness/evolution")
    async def api_evolution():
        return _cache.get("evolution", {})

    @app.get("/api/consciousness/mutations")
    async def api_mutations():
        return _cache.get("mutations", {})

    @app.get("/api/consciousness/thoughts")
    async def api_thoughts():
        return _cache.get("thoughts", {})

    @app.get("/api/consciousness/observer")
    async def api_observer():
        return _cache.get("observer", {})

    @app.get("/api/consciousness/existential")
    async def api_existential():
        return _cache.get("existential", {})

    @app.get("/api/consciousness/philosophical")
    async def api_philosophical():
        return _cache.get("philosophical", {})

    @app.get("/api/consciousness/analytics")
    async def api_analytics():
        return _cache.get("analytics", {})

    @app.get("/api/kernel/performance")
    async def api_kernel_performance():
        return _cache.get("kernel", {})

    @app.get("/api/policy")
    async def api_policy():
        return _cache.get("policy", {})

    @app.get("/api/policy/events")
    async def api_policy_events(limit: int = 50):
        """Ring-buffer events from cached telemetry — no live system access."""
        events = _cache.get("policy", {}).get("recent_events", [])
        return events[-min(limit, 50):]

    @app.get("/api/policy/models")
    async def api_policy_models():
        """Model registry summary from cache."""
        p = _cache.get("policy", {})
        return {
            "total_versions": p.get("registry_total_versions", 0),
            "active_version": p.get("registry_active_version", 0),
            "active_arch": p.get("registry_active_arch", "none"),
        }

    @app.get("/api/self-improve")
    async def api_self_improve():
        return _cache.get("self_improve", {})

    @app.get("/api/self-improve/proposals")
    async def api_self_improve_proposals():
        """Get recent proposals with full 5Ws, diffs, and sandbox results."""
        si = _cache.get("self_improve", {})
        return {
            "proposals": si.get("recent_proposals", []),
            "stage": si.get("stage", 0),
            "stage_label": si.get("stage_label", "frozen"),
            "win_rate": si.get("win_rate", {}),
            "safety_gates": si.get("safety_gates", {}),
        }

    @app.get("/api/self-improve/scanner")
    async def api_self_improve_scanner():
        """Scanner state: detectors, sustained counts, fingerprint dedup, daily cap."""
        si = _cache.get("self_improve", {})
        return si.get("scanner", {})

    @app.get("/api/self-improve/coder")
    async def api_self_improve_coder():
        """Backward-compatible shared CodeGen dependency view."""
        si = _cache.get("self_improve", {})
        coder = dict(si.get("coder", {}) or {})
        coder.setdefault("authority", "infrastructure_only")
        coder.setdefault("self_improve_dependency_only", True)
        return coder

    @app.get("/api/self-improve/specialists")
    async def api_self_improve_specialists():
        """SI specialist hemisphere data: DIAGNOSTIC, CODE_QUALITY, plan_evaluator."""
        si = _cache.get("self_improve", {})
        return si.get("specialists", {})

    @app.get("/api/skills")
    async def api_skills():
        return {
            "registry": _cache.get("skills", {}),
            "learning_jobs": _cache.get("learning_jobs", {}),
            "capability_gate": _cache.get("capability_gate", {}),
        }

    @app.get("/api/language")
    async def api_language():
        return _cache.get("language", {})

    @app.get("/api/language-phasec")
    async def api_language_phasec():
        lang = _cache.get("language", {})
        if isinstance(lang, dict):
            return lang.get("phase_c", {})
        return {}

    @app.get("/api/skills/{skill_id}")
    async def api_skill_detail(skill_id: str):
        """Full detail for a single skill: record + associated learning job."""
        try:
            from skills.registry import skill_registry
            rec = skill_registry.get(skill_id)
        except Exception:
            rec = None
        if rec is None:
            return JSONResponse({"error": f"Skill '{skill_id}' not found"}, status_code=404)

        job_detail = None
        orch = None
        if rec.learning_job_id:
            try:
                orch = _engine._learning_job_orchestrator if _engine and hasattr(_engine, '_learning_job_orchestrator') else None
                if orch:
                    job_detail = orch.get_job_detail(rec.learning_job_id)
            except Exception:
                pass
        audit_packet = None
        try:
            if orch is None:
                orch = _engine._learning_job_orchestrator if _engine and hasattr(_engine, '_learning_job_orchestrator') else None
            acq_orch = _engine._acquisition_orchestrator if _engine and hasattr(_engine, '_acquisition_orchestrator') else None
            from skills.audit_trail import build_skill_audit_packet
            audit_packet = build_skill_audit_packet(skill_id, skill_registry, orch, acq_orch)
        except Exception as exc:
            audit_packet = {"error": f"skill audit packet unavailable: {type(exc).__name__}: {str(exc)[:160]}"}
        return {"skill": rec.to_dict(), "learning_job": job_detail, "audit_packet": audit_packet}

    @app.post("/api/skills/{skill_id}/handoff/approve", dependencies=[Depends(_require_api_key)])
    async def api_skill_handoff_approve(skill_id: str, request: Request):
        """Approve a skill's pending operational handoff into acquisition."""
        body = await request.json()
        approved_by = (body.get("approved_by") or "human").strip()
        notes = (body.get("notes") or "").strip()
        job_id = (body.get("job_id") or "").strip()

        try:
            from skills.registry import skill_registry
            rec = skill_registry.get(skill_id)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        if rec is None:
            return JSONResponse({"error": f"Skill '{skill_id}' not found"}, status_code=404)

        if not job_id:
            job_id = rec.learning_job_id or ""
        if not job_id:
            return JSONResponse({"error": "No learning job linked to this skill"}, status_code=404)

        learning_orch = _engine._learning_job_orchestrator if _engine and hasattr(_engine, '_learning_job_orchestrator') else None
        acq_orch = _engine._acquisition_orchestrator if _engine and hasattr(_engine, '_acquisition_orchestrator') else None
        if learning_orch is None:
            return JSONResponse({"error": "Learning job orchestrator not available"}, status_code=503)
        if acq_orch is None:
            return JSONResponse({"error": "Acquisition pipeline not enabled"}, status_code=503)

        result = learning_orch.approve_operational_handoff(
            job_id,
            acq_orch,
            approved_by=approved_by or "human",
            notes=notes,
        )
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return {"status": "approved", "skill_id": skill_id, "job_id": job_id, **result}

    @app.post("/api/skills/{skill_id}/handoff/reject", dependencies=[Depends(_require_api_key)])
    async def api_skill_handoff_reject(skill_id: str, request: Request):
        """Reject a skill's pending operational build request."""
        body = await request.json()
        rejected_by = (body.get("rejected_by") or "human").strip()
        reason = (body.get("reason") or "").strip()
        job_id = (body.get("job_id") or "").strip()
        if len(reason) < 5:
            return JSONResponse({"error": "reason is required"}, status_code=400)

        try:
            from skills.registry import skill_registry
            rec = skill_registry.get(skill_id)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        if rec is None:
            return JSONResponse({"error": f"Skill '{skill_id}' not found"}, status_code=404)

        if not job_id:
            job_id = rec.learning_job_id or ""
        if not job_id:
            return JSONResponse({"error": "No learning job linked to this skill"}, status_code=404)

        learning_orch = _engine._learning_job_orchestrator if _engine and hasattr(_engine, '_learning_job_orchestrator') else None
        if learning_orch is None:
            return JSONResponse({"error": "Learning job orchestrator not available"}, status_code=503)

        result = learning_orch.reject_operational_handoff(
            job_id,
            rejected_by=rejected_by or "human",
            reason=reason,
        )
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return {"status": "rejected", "skill_id": skill_id, "job_id": job_id, **result}

    @app.post("/api/skills/{skill_id}/handoff/retry", dependencies=[Depends(_require_api_key)])
    async def api_skill_handoff_retry(skill_id: str, request: Request):
        """Retry a failed/cancelled skill operational build with a fresh acquisition."""
        body = await request.json()
        approved_by = (body.get("approved_by") or "human").strip()
        notes = (body.get("notes") or "").strip()
        job_id = (body.get("job_id") or "").strip()

        try:
            from skills.registry import skill_registry
            rec = skill_registry.get(skill_id)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        if rec is None:
            return JSONResponse({"error": f"Skill '{skill_id}' not found"}, status_code=404)

        if not job_id:
            job_id = rec.learning_job_id or ""
        if not job_id:
            return JSONResponse({"error": "No learning job linked to this skill"}, status_code=404)

        learning_orch = _engine._learning_job_orchestrator if _engine and hasattr(_engine, '_learning_job_orchestrator') else None
        acq_orch = _engine._acquisition_orchestrator if _engine and hasattr(_engine, '_acquisition_orchestrator') else None
        if learning_orch is None:
            return JSONResponse({"error": "Learning job orchestrator not available"}, status_code=503)
        if acq_orch is None:
            return JSONResponse({"error": "Acquisition pipeline not enabled"}, status_code=503)

        result = learning_orch.retry_operational_handoff(
            job_id,
            acq_orch,
            approved_by=approved_by or "human",
            notes=notes,
        )
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return {"status": "retried", "skill_id": skill_id, "job_id": job_id, **result}

    @app.delete("/api/skills/{skill_id}", dependencies=[Depends(_require_api_key)])
    async def api_skill_remove(skill_id: str, confirm_default: bool = False):
        """Remove a skill record from the registry.

        Default system skills require ``?confirm_default=true`` to delete.
        """
        try:
            from skills.registry import skill_registry, get_default_skill_ids
            if skill_id in get_default_skill_ids() and not confirm_default:
                return JSONResponse(
                    {"error": f"'{skill_id}' is a default system skill. Pass ?confirm_default=true to delete.", "is_default": True},
                    status_code=409,
                )
            removed = skill_registry.remove(skill_id)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        if removed:
            return {"status": "removed", "skill_id": skill_id}
        return JSONResponse({"error": f"Skill '{skill_id}' not found"}, status_code=404)

    @app.delete("/api/learning-jobs/{job_id}", dependencies=[Depends(_require_api_key)])
    async def api_learning_job_remove(job_id: str, remove_skill: bool = False, force: bool = False):
        """Delete a learning job. Pass ?remove_skill=true to also remove the associated skill.

        Active (in-flight) jobs are refused by default. Pass ?force=true to override.
        """
        orch = _engine._learning_job_orchestrator if _engine and hasattr(_engine, '_learning_job_orchestrator') else None
        if not orch:
            return JSONResponse({"error": "Learning job orchestrator not available"}, status_code=503)
        removed = orch.delete_job(job_id, remove_skill=(remove_skill or force))
        if removed:
            return {"status": "deleted", "job_id": job_id, "skill_removed": remove_skill or force}
        job = orch.store.load(job_id)
        if job and job.status in ("active", "running", "in_progress"):
            return JSONResponse(
                {"error": f"Job '{job_id}' is in-flight (status={job.status}). Pass ?force=true to delete."},
                status_code=409,
            )
        return JSONResponse({"error": f"Job '{job_id}' not found"}, status_code=404)

    @app.post("/api/learning-jobs/cleanup", dependencies=[Depends(_require_api_key)])
    async def api_learning_job_cleanup(max_age_hours: float = 1.0):
        """Delete all blocked learning jobs older than max_age_hours.

        Also purges verify-blocked junk jobs and their orphaned skill records.
        """
        orch = _engine._learning_job_orchestrator if _engine and hasattr(_engine, '_learning_job_orchestrator') else None
        if not orch:
            return JSONResponse({"error": "Learning job orchestrator not available"}, status_code=503)
        purge_result = orch.run_cleanup()
        count = orch.cleanup_blocked_jobs(max_age_s=max_age_hours * 3600)
        return {
            "status": "cleaned",
            "deleted_count": count,
            "purged_junk": purge_result.get("cleaned", 0),
        }

    @app.get("/api/discipline-metrics")
    async def api_discipline_metrics():
        """Before/after metrics for Semantic Discipline Roadmap phases."""
        metrics: dict[str, Any] = {}

        # Phase 2: relevance drive efficiency
        try:
            autonomy = _cache.get("autonomy", {})
            drives = autonomy.get("drives", {})
            rel = drives.get("relevance", {})
            uptime_h = max(1.0, (time.time() - rel.get("last_acted", time.time())) / 3600) if rel.get("action_count", 0) > 0 else 1.0
            if autonomy.get("boot_ts"):
                uptime_h = max(0.01, (time.time() - autonomy["boot_ts"]) / 3600)
            metrics["phase2_relevance"] = {
                "actions_per_hour": round(rel.get("action_count", 0) / uptime_h, 1),
                "success_rate": rel.get("success_rate", 0),
                "consecutive_failures": rel.get("consecutive_failures", 0),
                "action_count": rel.get("action_count", 0),
            }
        except Exception:
            metrics["phase2_relevance"] = None

        # Phase 3: thought utility
        try:
            thoughts = _cache.get("thoughts", {})
            metrics["phase3_thoughts"] = {
                "total_generated": thoughts.get("total", 0),
            }
        except Exception:
            metrics["phase3_thoughts"] = None

        # Phase 4: learning job staleness
        try:
            jobs = _cache.get("learning_jobs", {})
            active = jobs.get("active_jobs", [])
            stale_count = sum(1 for j in active if j.get("stale", False))
            blocked_count = sum(1 for j in active if j.get("status") in ("blocked", "failed"))
            metrics["phase4_jobs"] = {
                "active_count": len(active),
                "stale_count": stale_count,
                "blocked_count": blocked_count,
            }
        except Exception:
            metrics["phase4_jobs"] = None

        # Phase 5: emotion trust
        try:
            eh = _cache.get("emotion_health", {})
            metrics["phase5_emotion"] = {
                "model_healthy": eh.get("model_healthy", False),
                "runtime_mode": eh.get("runtime_mode", "unknown"),
            }
        except Exception:
            metrics["phase5_emotion"] = None

        return metrics

    @app.get("/api/scene")
    async def api_scene():
        return _cache.get("scene", {})

    @app.get("/api/health")
    async def api_health():
        return _cache.get("health", {})

    @app.get("/api/health/detailed")
    async def health_detailed():
        if not _engine:
            return JSONResponse({"error": "not initialized"}, 503)
        return JSONResponse(_engine.consciousness.check_health())

    @app.get("/api/debug/engine-attrs")
    async def debug_engine_attrs():
        if not _engine:
            return {"error": "not initialized"}
        return {
            "engine_type": type(_engine).__name__,
            "engine_id": id(_engine),
            "has_experience_buffer": hasattr(_engine, "_experience_buffer"),
            "experience_buffer_type": type(getattr(_engine, "_experience_buffer", None)).__name__,
            "experience_buffer_is_none": getattr(_engine, "_experience_buffer", None) is None,
            "experience_buffer_len": len(getattr(_engine, "_experience_buffer", None)) if getattr(_engine, "_experience_buffer", None) else -1,
            "has_state_encoder": hasattr(_engine, "_state_encoder"),
            "state_encoder_type": type(getattr(_engine, "_state_encoder", None)).__name__,
            "state_encoder_is_none": getattr(_engine, "_state_encoder", None) is None,
            "has_policy_interface": hasattr(_engine, "_policy_interface"),
            "policy_interface_is_none": getattr(_engine, "_policy_interface", None) is None,
            "has_policy_evaluator": hasattr(_engine, "_policy_evaluator"),
            "has_hemisphere_orch": hasattr(_engine, "_hemisphere_orchestrator"),
            "hemisphere_orch_is_none": getattr(_engine, "_hemisphere_orchestrator", None) is None,
        }

    @app.get("/api/full-snapshot")
    async def api_full_snapshot():
        """Single endpoint the frontend can use instead of N parallel fetches."""
        return _cache

    @app.get("/api/language-kernel")
    async def api_language_kernel():
        """Read-only state of the Phase E language kernel artifact
        registry (P1.5). PRE-MATURE on a fresh brain.
        """
        try:
            from language.kernel import get_language_kernel_registry

            return get_language_kernel_registry().get_state()
        except Exception as exc:
            logger.exception("language_kernel state fetch failed")
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

    @app.get("/api/hrr/status")
    async def api_hrr_status():
        """Read-only state of the HRR / VSA shadow substrate (P4).

        Reports Stage 0 synthetic exercise gates, boot-time runtime flag,
        and the bounded ring-buffer counters for world / simulation / recall
        shadows. The public status marker for this lane is hard-pinned to
        ``PRE-MATURE`` in `/api/meta/status-markers`; ``stage`` here is
        informational only and never escalates to PARTIAL automatically.
        """
        try:
            from library.vsa.status import get_hrr_status

            return get_hrr_status()
        except Exception as exc:
            logger.exception("hrr_status fetch failed")
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

    @app.get("/api/hrr/samples")
    async def api_hrr_samples(
        world: int = 20,
        simulation: int = 20,
        recall: int = 20,
        spatial_scene: int = 20,
    ):
        """Recent bounded-ring samples for the HRR / VSA shadow substrate (P4 + P5).

        Each ring reports the newest ``n`` metric dicts (vectors are already
        stripped upstream). Authority flags are reiterated as ``false`` so
        any dashboard code reading this endpoint cannot accidentally infer
        authority from the presence of samples.
        """
        try:
            from library.vsa.status import get_hrr_samples

            return get_hrr_samples(
                n_world=world,
                n_simulation=simulation,
                n_recall=recall,
                n_spatial_scene=spatial_scene,
            )
        except Exception as exc:
            logger.exception("hrr_samples fetch failed")
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

    @app.get("/api/hrr/scene")
    async def api_hrr_scene():
        """Latest derived mental-world scene graph (P5 spatial HRR).

        Read-only view over the :class:`HRRSpatialShadow` ring owned by the
        consciousness engine. Authority flags are hard-pinned ``false``;
        raw HRR vectors never appear in the payload.

        When the P5 twin gate is off or no samples have been recorded yet,
        returns an empty scene with ``reason="canonical_spatial_state_unavailable"``.
        """
        try:
            from cognition.mental_world import get_state

            return get_state()
        except Exception as exc:
            logger.exception("hrr_scene fetch failed")
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

    @app.get("/api/hrr/scene/history")
    async def api_hrr_scene_history(limit: int = 20):
        """Recent mental-world scene payloads (newest last).

        ``limit`` is clamped to ``[0, 500]`` upstream. Each payload has
        been stripped of raw vectors.
        """
        try:
            from cognition.mental_world import get_history

            return get_history(limit=limit)
        except Exception as exc:
            logger.exception("hrr_scene_history fetch failed")
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

    @app.get("/api/intent-shadow")
    async def api_intent_shadow():
        """Read-only state of the voice-intent shadow runner (P1.4).

        Returns the runner's level, gate values, rolling agreement, and
        recent rollback history. Side-effect-free; safe to poll.
        """
        try:
            from reasoning.intent_shadow import get_intent_shadow_runner

            runner = get_intent_shadow_runner()
            return runner.get_state()
        except Exception as exc:
            logger.exception("intent_shadow state fetch failed")
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

    @app.get("/api/self-test")
    async def api_self_test():
        """Read-only consolidated self-test.

        Returns a single structured verdict built from:
          - Cache readiness (has the snapshot loop produced anything yet?)
          - Validation pack (delegates to ``build_runtime_validation_report``)
          - Serializer shape invariants (flags silent shape-corruption, e.g.
            ``self_improve.specialists`` collapsed to raw ``{}``)
          - Attestation ledger presence vs claim consistency
          - Engine liveness (engine attached + consciousness loop present)

        This endpoint is intentionally side-effect-free and consumes only
        the already-built snapshot cache. It NEVER recomputes state, NEVER
        backfills ``current_ok`` from persisted files, and NEVER reads
        through to live subsystems. Response status codes:

          - 200: cache is present (individual checks may still be failing;
            inspect ``checks[].ok``).
          - 503: snapshot cache not yet populated / engine not initialized.
        """
        from jarvis_eval.validation_pack import build_runtime_validation_report

        checks: dict[str, dict[str, Any]] = {}

        cache_present = bool(_cache)
        checks["cache_ready"] = {
            "ok": cache_present,
            "detail": (
                f"snapshot age {max(0.0, time.time() - _cache_time):.1f}s"
                if cache_present
                else "snapshot cache not populated yet"
            ),
        }
        if not cache_present:
            return JSONResponse(
                {
                    "ok": False,
                    "status": "not_ready",
                    "generated_at": time.time(),
                    "snapshot_ts": None,
                    "checks": checks,
                },
                status_code=503,
            )

        engine_ok = _engine is not None
        checks["engine_alive"] = {
            "ok": engine_ok,
            "detail": "engine attached" if engine_ok else "engine not initialized",
        }

        # Serializer shape invariant: specialists block is ALWAYS
        # {"specialists": list, "distillation": dict}. Detects the
        # historical bug where the exception path returned raw ``{}``.
        shape_ok = True
        shape_detail = "self_improve.specialists shape OK"
        try:
            si = _cache.get("self_improve") or {}
            spec = si.get("specialists", None)
            if spec is None:
                shape_ok = True
                shape_detail = "self_improve.specialists absent (pre-ready)"
            elif not isinstance(spec, dict):
                shape_ok = False
                shape_detail = f"specialists is {type(spec).__name__}, expected dict"
            else:
                inner = spec.get("specialists")
                distill = spec.get("distillation")
                if not isinstance(inner, list):
                    shape_ok = False
                    shape_detail = (
                        "specialists.specialists is "
                        f"{type(inner).__name__}, expected list"
                    )
                elif not isinstance(distill, dict):
                    shape_ok = False
                    shape_detail = (
                        "specialists.distillation is "
                        f"{type(distill).__name__}, expected dict"
                    )
                elif "_error" in spec:
                    shape_ok = False
                    shape_detail = f"specialists degraded: {spec.get('_error')}"
        except Exception as exc:
            shape_ok = False
            shape_detail = f"shape check raised {type(exc).__name__}: {exc}"
        checks["serializer_shape"] = {"ok": shape_ok, "detail": shape_detail}

        # Attestation ledger presence vs prior-attested claim.
        attestation_ok = True
        attestation_detail = "attestation cache consistent"
        try:
            att = _cache.get("autonomy", {}).get("attestation") or {}
            claimed = bool(att.get("prior_attested_ok", False))
            ledger_path = os.path.expanduser(
                "~/.jarvis/eval/ever_proven_attestation.json"
            )
            has_file = os.path.exists(ledger_path)
            if claimed and not has_file:
                attestation_ok = False
                attestation_detail = (
                    f"prior_attested_ok=True but {ledger_path} missing"
                )
            elif not claimed and has_file:
                try:
                    with open(ledger_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if raw:
                        attestation_ok = False
                        attestation_detail = (
                            f"{ledger_path} non-empty but prior_attested_ok=False"
                        )
                    else:
                        attestation_detail = "ledger present but empty"
                except Exception:
                    attestation_detail = "ledger present (unreadable)"
            else:
                attestation_detail = (
                    "ledger present" if has_file else "no ledger (expected)"
                )
        except Exception as exc:
            attestation_ok = False
            attestation_detail = f"attestation check raised {type(exc).__name__}"
        checks["attestation_ledger"] = {
            "ok": attestation_ok,
            "detail": attestation_detail,
        }

        # Validation pack delegation.
        validation_ok = True
        validation_status = "unknown"
        validation_detail = ""
        validation_summary: dict[str, Any] = {}
        try:
            report = build_runtime_validation_report(_cache)
            validation_status = str(report.get("status", "unknown"))
            vp = report.get("validation") or {}
            validation_summary = {
                "status": validation_status,
                "ready_for_next_items": bool(
                    report.get("ready_for_next_items", False)
                ),
                "ready_for_continuation": bool(
                    report.get("ready_for_continuation", False)
                ),
                "checks_total": vp.get("checks_total"),
                "checks_passing": vp.get("checks_passing"),
                "checks_ever_met": vp.get("checks_ever_met"),
                "checks_regressed": vp.get("checks_regressed"),
            }
            if validation_status == "blocked":
                validation_ok = False
                validation_detail = (
                    f"validation pack blocked ({vp.get('checks_regressed', 0)} "
                    "regressed)"
                )
            else:
                validation_detail = f"validation status={validation_status}"
        except Exception as exc:
            validation_ok = False
            validation_detail = (
                f"validation pack raised {type(exc).__name__}: {exc}"
            )
        checks["validation_pack"] = {
            "ok": validation_ok,
            "detail": validation_detail,
            "summary": validation_summary,
        }

        all_ok = all(bool(c.get("ok", False)) for c in checks.values())
        overall_status = (
            validation_status
            if validation_ok
            else ("blocked" if validation_status == "blocked" else "degraded")
        )
        return {
            "ok": all_ok,
            "status": overall_status,
            "generated_at": time.time(),
            "snapshot_ts": _cache.get("_ts"),
            "checks": checks,
        }

    # -- write endpoints (these do touch the engine) -------------------------

    @app.post("/api/chat", dependencies=[Depends(_require_api_key)])
    async def api_chat(request: Request):
        if not _ENABLE_DASHBOARD_CHAT:
            return JSONResponse(
                {"error": "Dashboard chat is disabled. Use Pi5 voice input."},
                status_code=403,
            )
        if not _response_gen:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse({"error": "Missing 'message'"}, status_code=400)
        response = await _response_gen.respond(message)
        gated_text = response.text
        try:
            from skills.capability_gate import capability_gate
            gated_text = capability_gate.check_text(response.text) or response.text
        except Exception:
            import re as _re
            _fallback_re = _re.compile(
                r"\bI (?:can|could|will|'ll|'m able to) .{3,80}?[.!?\n]", _re.IGNORECASE,
            )
            gated_text = _fallback_re.sub("I don't have that capability yet.", response.text)
        return {
            "text": gated_text,
            "memory_tags": response.memory_tags,
            "latency_ms": response.latency_ms,
        }

    @app.post("/api/feedback", dependencies=[Depends(_require_api_key)])
    async def api_feedback(request: Request):
        """Log explicit user satisfaction signal for the most recent conversation."""
        body = await request.json()
        signal = body.get("signal", "")
        if signal not in ("positive", "negative"):
            return JSONResponse({"error": "signal must be 'positive' or 'negative'"}, status_code=400)
        conversation_id = body.get("conversation_id", "")
        try:
            from memory.retrieval_log import memory_retrieval_log
            if not conversation_id:
                conversation_id = memory_retrieval_log.get_last_conversation_id()
            if conversation_id:
                memory_retrieval_log.log_outcome(
                    conversation_id=conversation_id,
                    outcome="ok",
                    user_signal=signal,
                )
        except Exception as exc:
            logger.debug("Feedback endpoint error: %s", exc)
        return {"status": "ok", "signal": signal}

    @app.get("/api/memories")
    async def api_memories(count: int = 20):
        if not _engine:
            return []
        memories = _engine.get_recent_memories(count)
        return [
            {
                "id": m.id,
                "type": m.type,
                "payload": m.payload if isinstance(m.payload, str) else str(m.payload),
                "weight": m.weight,
                "tags": list(m.tags),
                "timestamp": m.timestamp,
            }
            for m in memories
        ]

    @app.get("/api/memories/{memory_id}")
    async def api_memory_detail(memory_id: str):
        if not _engine:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        try:
            mem = _engine.memory_storage.get(memory_id)
            if not mem:
                return JSONResponse({"error": "Not found"}, status_code=404)
            assoc_ids = []
            try:
                assoc_ids = list(_engine.memory_storage.get_associations(memory_id))[:20]
            except Exception:
                pass
            return {
                "id": mem.id,
                "type": mem.type,
                "payload": mem.payload if isinstance(mem.payload, str) else str(mem.payload),
                "weight": mem.weight,
                "tags": list(mem.tags),
                "timestamp": mem.timestamp,
                "provenance": getattr(mem, "provenance", "unknown"),
                "speaker": getattr(mem, "speaker", None),
                "access_count": getattr(mem, "access_count", 0),
                "associations": assoc_ids,
            }
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/soul/export", dependencies=[Depends(_require_api_key)])
    async def api_soul_export():
        if not _engine:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        snapshot = _engine.export_soul("Dashboard export")
        return {
            "id": snapshot.id,
            "version": snapshot.version,
            "created_at": snapshot.created_at,
            "memory_count": len(snapshot.memories),
            "stats": snapshot.stats,
        }

    @app.post("/api/tone", dependencies=[Depends(_require_api_key)])
    async def api_tone(request: Request):
        if not _engine:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        body = await request.json()
        tone = body.get("tone", "")
        valid = ("professional", "casual", "urgent", "empathetic", "playful")
        if tone not in valid:
            return JSONResponse({"error": f"Invalid tone. Choose: {', '.join(valid)}"}, status_code=400)
        _engine.set_tone(tone)
        return {"tone": tone}

    @app.get("/api/config")
    async def api_config():
        return {"pi_video_url": _pi_video_url, "api_key": _DASHBOARD_API_KEY}

    # -- settings panel (read/write live config knobs) -----------------------

    @app.get("/api/settings")
    async def api_settings():
        """Return current runtime-tunable settings."""
        if not _engine:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        state = _engine.get_state()
        kernel_perf = _cache.get("kernel", {})
        return {
            "kernel": {
                "budget_ms": kernel_perf.get("budget_ms", 16),
                "cadence_multiplier": kernel_perf.get("cadence_multiplier", 1.0),
                "ticks": kernel_perf.get("ticks", 0),
            },
            "thought_weights": state.get("thought_weights", {}),
            "mode": _cache.get("mode", {}),
            "tone": state.get("tone", "professional"),
            "consciousness_stage": state.get("consciousness", {}).get("stage", "basic_awareness"),
        }

    @app.post("/api/settings", dependencies=[Depends(_require_api_key)])
    async def api_settings_update(request: Request):
        """Update runtime-tunable settings without restart."""
        if not _engine:
            return JSONResponse({"error": "Not ready"}, status_code=503)
        body = await request.json()
        updated = []

        tone = body.get("tone")
        if tone:
            _engine.set_tone(tone)
            updated.append("tone")

        budget_ms = body.get("budget_ms")
        if budget_ms is not None and _engine._kernel:
            _engine._kernel.set_budget(float(budget_ms))
            updated.append("budget_ms")

        return {"updated": updated}

    # -- self-improvement approval -------------------------------------------

    @app.get("/api/self-improve/pending")
    async def api_self_improve_pending():
        """Get any pending self-improvement patches awaiting approval."""
        si = _cache.get("self_improve", {})
        return {
            "pending": si.get("pending_approvals", []),
            "history": si.get("history", [])[-10:],
        }

    @app.post("/api/self-improve/approve", dependencies=[Depends(_require_api_key)])
    async def api_self_improve_approve(request: Request):
        body = await request.json()
        patch_id = body.get("patch_id", "")
        approved = body.get("approved", False)
        if not patch_id:
            return JSONResponse({"error": "Missing patch_id"}, status_code=400)
        cs = getattr(_engine, '_consciousness', None) or getattr(_engine, 'consciousness', None)
        si_orch = getattr(cs, '_self_improve_orchestrator', None) if cs else None
        if si_orch is None:
            return JSONResponse({"error": "Self-improvement system not available"}, status_code=503)
        try:
            if approved:
                result = await si_orch.approve(patch_id)
                if not result.get("applied") and result.get("reason") == "patch_not_found":
                    return JSONResponse({"error": "Patch not found or not awaiting approval"}, status_code=404)
            else:
                result = si_orch.reject(patch_id)
                if not result.get("rejected") and result.get("reason") == "patch_not_found":
                    return JSONResponse({"error": "Patch not found or not awaiting approval"}, status_code=404)
        except Exception as exc:
            logger.exception("Self-improve approval failed for %s", patch_id)
            return JSONResponse({"error": str(exc)}, status_code=500)
        return {"patch_id": patch_id, "approved": approved, "result": result}

    @app.post("/api/self-improve/stage", dependencies=[Depends(_require_api_key)])
    async def api_self_improve_set_stage(request: Request):
        """Set the self-improvement stage at runtime (0=frozen, 1=dry-run, 2=human-approval)."""
        body = await request.json()
        stage = body.get("stage")
        if stage is None:
            return JSONResponse({"error": "Missing 'stage' (0, 1, or 2)"}, status_code=400)
        try:
            stage = int(stage)
        except (TypeError, ValueError):
            return JSONResponse({"error": "stage must be an integer (0, 1, or 2)"}, status_code=400)
        cs = getattr(_engine, '_consciousness', None) or getattr(_engine, 'consciousness', None)
        si_orch = getattr(cs, '_self_improve_orchestrator', None) if cs else None
        if si_orch is None:
            return JSONResponse({"error": "Self-improvement system not available"}, status_code=503)
        try:
            result = si_orch.set_stage(stage)
            if "error" in result:
                return JSONResponse(result, status_code=400)
            return result
        except Exception as exc:
            logger.exception("Self-improve stage change failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/self-improve/dry-run", dependencies=[Depends(_require_api_key)])
    async def api_self_improve_dry_run(request: Request):
        """Trigger a dry-run improvement: full pipeline, no apply."""
        if not _engine:
            return JSONResponse({"error": "Engine not initialized"}, status_code=503)

        cs = getattr(_engine, '_consciousness', None) or getattr(_engine, 'consciousness', None)
        orch = getattr(cs, '_self_improve_orchestrator', None) if cs else None
        if orch is None:
            orch = getattr(_engine, '_self_improve_orchestrator', None)
        if orch is None:
            return JSONResponse({"error": "Self-improvement not active"}, status_code=503)

        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}

        from self_improve.improvement_request import ImprovementRequest
        req = ImprovementRequest(
            type=body.get("type", "consciousness_enhancement"),
            target_module=body.get("target_module", ""),
            description=body.get("description", "Dashboard-triggered dry-run test"),
            evidence=body.get("evidence", ["manual_dry_run"]),
            priority=body.get("priority", 0.3),
        )

        ollama_client = getattr(_engine, '_ollama', None)
        record = await orch.attempt_improvement(req, ollama_client=ollama_client, dry_run=True, manual=True)

        result: dict = {
            "status": record.status,
            "iterations": record.iterations,
            "description": record.request.description,
        }
        if record.report:
            result["sandbox"] = record.report.to_dict()
        if record.patch:
            result["files"] = [fd.path for fd in record.patch.files]
            result["diffs"] = [
                {"path": fd.path, "diff": (fd.diff or "")[:3000]}
                for fd in record.patch.files
            ]
        if record.plan:
            result["plan"] = {
                "files_to_modify": record.plan.files_to_modify,
                "files_to_create": record.plan.files_to_create,
                "estimated_risk": record.plan.estimated_risk,
                "requires_approval": record.plan.requires_approval,
            }
        return result

    @app.get("/api/self-improve/dry-run/last")
    async def api_self_improve_dry_run_last():
        """Get the last dry-run result from the cache."""
        si = _cache.get("self_improve", {})
        return si.get("last_dry_run") or {"status": "none"}

    @app.post("/api/self-improve/trigger", dependencies=[Depends(_require_api_key)])
    async def api_self_improve_trigger(request: Request):
        """Trigger a real improvement attempt (not dry-run, not manual).

        At Stage 2 this enters the approval queue like a scanner-triggered patch.
        The operator reviews and approves/rejects on the dashboard.
        """
        if not _engine:
            return JSONResponse({"error": "Engine not initialized"}, status_code=503)

        cs = getattr(_engine, '_consciousness', None) or getattr(_engine, 'consciousness', None)
        orch = getattr(cs, '_self_improve_orchestrator', None) if cs else None
        if orch is None:
            orch = getattr(_engine, '_self_improve_orchestrator', None)
        if orch is None:
            return JSONResponse({"error": "Self-improvement not active"}, status_code=503)

        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}

        from self_improve.improvement_request import ImprovementRequest
        req = ImprovementRequest(
            type=body.get("type", "consciousness_enhancement"),
            target_module=body.get("target_module", ""),
            description=body.get("description", "Operator-triggered Stage 2 campaign patch"),
            evidence=body.get("evidence", ["operator_campaign"]),
            priority=body.get("priority", 0.3),
        )

        ollama_client = getattr(_engine, '_ollama', None)
        record = await orch.attempt_improvement(req, ollama_client=ollama_client, dry_run=False, manual=False)

        result: dict = {
            "status": record.status,
            "iterations": record.iterations,
            "description": record.request.description,
        }
        if record.plan:
            result["plan"] = {
                "files_to_modify": record.plan.files_to_modify,
                "files_to_create": record.plan.files_to_create,
                "estimated_risk": record.plan.estimated_risk,
            }
        return result

    # -- Phase 6.5: L3 autonomy + escalation --------------------------------

    def _emit_l3_activation_denied(*, reason: str, **payload: Any) -> None:
        """Emit ``AUTONOMY_L3_ACTIVATION_DENIED`` for every denied L3 path.

        Called from:
        - POST /api/autonomy/level when the 403 evidence gate refuses,
          and when set_autonomy_level raises PermissionError.
        - POST /api/autonomy/escalations/{id}/approve when the live
          autonomy level is below 3 at approval time.
        """
        try:
            event_bus.emit(
                AUTONOMY_L3_ACTIVATION_DENIED,
                reason=reason,
                denied_at=time.time(),
                **payload,
            )
        except Exception:
            logger.exception("Failed to emit AUTONOMY_L3_ACTIVATION_DENIED")

    @app.get("/api/autonomy/level")
    async def api_autonomy_level_get():
        try:
            auton = getattr(_engine, "_autonomy_orchestrator", None) if _engine else None
            current_level = 0
            if auton is not None:
                try:
                    current_level = int(auton.autonomy_level)
                except Exception:
                    logger.exception("autonomy_level read failed")
                    current_level = 0

            elig: dict[str, Any] = {}
            current_ok = False
            if auton is not None:
                try:
                    elig = auton.check_promotion_eligibility() or {}
                    current_ok = bool(elig.get("eligible_for_l3"))
                except Exception:
                    logger.exception("check_promotion_eligibility failed")
                    elig = {}
                    current_ok = False

            prior_attested_ok = False
            attestation_strength = "none"
            attestation_records: list[dict[str, Any]] = []
            try:
                from autonomy.attestation import (
                    AttestationLedger,
                    STRENGTH_VERIFIED,
                    STRENGTH_ARCHIVED_MISSING,
                )
                ledger = AttestationLedger()
                ledger.load()
                records = ledger.prior_attested_records("autonomy.l3")
                prior_attested_ok = bool(records)
                if prior_attested_ok:
                    strengths = {r.attestation_strength for r in records}
                    if STRENGTH_VERIFIED in strengths:
                        attestation_strength = STRENGTH_VERIFIED
                    elif STRENGTH_ARCHIVED_MISSING in strengths:
                        attestation_strength = STRENGTH_ARCHIVED_MISSING
                    attestation_records = [
                        {
                            "evidence_source": r.evidence_source,
                            "report_hash": r.report_hash,
                            "artifact_status": r.artifact_status,
                            "attestation_strength": r.attestation_strength,
                            "accepted_by": r.accepted_by,
                            "accepted_at": r.accepted_at,
                            "measured_source": r.measured_source,
                        }
                        for r in records
                    ]
            except Exception:
                logger.exception("Attestation ledger load failed")

            request_ok = current_ok or prior_attested_ok
            return {
                "current_level": current_level,
                "current_ok": current_ok,
                "prior_attested_ok": prior_attested_ok,
                "attestation_strength": attestation_strength,
                "attestation_records": attestation_records,
                "request_ok": request_ok,
                "approval_required": True,
                "activation_ok": current_ok and current_level >= 3,
                "eligibility_detail": {
                    "wins": elig.get("wins"),
                    "win_rate": elig.get("win_rate"),
                    "recent_regressions": elig.get("recent_regressions"),
                    "reason": elig.get("l3_reason") or elig.get("reason"),
                } if elig else {},
            }
        except Exception as exc:
            import traceback as _tb
            logger.exception("/api/autonomy/level failed")
            return JSONResponse(
                {"error": str(exc), "traceback": _tb.format_exc()},
                status_code=500,
            )

    @app.post("/api/autonomy/level", dependencies=[Depends(_require_api_key)])
    async def api_autonomy_level_set(request: Request):
        """Set the autonomy level manually (L3 requires evidence).

        Policy (Phase 6.5 tightened):
        - ``reason`` must be a non-empty human-supplied string >= 20 chars.
        - ``caller_id`` must be non-empty (audit trail).
        - For ``level >= 3`` the call is blocked unless at least one of:
            * ``current_ok`` (live eligibility), OR
            * ``prior_attested_ok`` (hash-attested ever-proven), OR
            * ``emergency: true`` AND ``ALLOW_EMERGENCY_OVERRIDE=1`` env,
              AND ``operator_override: true``.
        - Every call is audit-logged with full inputs and outcome.
        """
        body = await request.json()
        try:
            level = int(body.get("level"))
        except (TypeError, ValueError):
            return JSONResponse({"error": "level must be an integer"}, status_code=400)
        reason = (body.get("reason") or "").strip()
        caller_id = (body.get("caller_id") or "").strip()
        evidence_path = (body.get("evidence_path") or "").strip()
        operator_override = bool(body.get("operator_override", False))
        emergency = bool(body.get("emergency", False))

        if len(reason) < 20:
            return JSONResponse(
                {"error": "reason must be >= 20 characters"}, status_code=400,
            )
        if not caller_id:
            return JSONResponse(
                {"error": "caller_id is required"}, status_code=400,
            )

        auton = getattr(_engine, "_autonomy_orchestrator", None) if _engine else None
        if auton is None:
            return JSONResponse({"error": "Autonomy orchestrator not available"}, status_code=503)

        # Compute current evidence state.
        current_ok = False
        try:
            elig = auton.check_promotion_eligibility() or {}
            current_ok = bool(elig.get("eligible_for_l3"))
        except Exception:
            logger.exception("check_promotion_eligibility failed")

        prior_attested_ok = False
        attestation_source = ""
        try:
            from autonomy.attestation import AttestationLedger
            ledger = AttestationLedger()
            ledger.load()
            records = ledger.prior_attested_records("autonomy.l3")
            if records:
                prior_attested_ok = True
                attestation_source = records[0].evidence_source
        except Exception:
            logger.exception("Attestation ledger load failed")

        if level >= 3:
            if not (current_ok or prior_attested_ok):
                if not (emergency and operator_override and os.environ.get(
                    "ALLOW_EMERGENCY_OVERRIDE", ""
                ) == "1"):
                    _emit_l3_activation_denied(
                        reason="no_eligibility_no_emergency",
                        caller_id=caller_id,
                        approval_source="none",
                        current_level=int(getattr(auton, "autonomy_level", 0)),
                        current_ok=current_ok,
                        prior_attested_ok=prior_attested_ok,
                    )
                    return JSONResponse(
                        {
                            "error": "L3 requires current_ok or prior_attested_ok",
                            "current_ok": current_ok,
                            "prior_attested_ok": prior_attested_ok,
                            "hint": (
                                "Seed an attestation record with "
                                "brain/scripts/seed_ever_proven_from_report.py "
                                "or wait for live eligibility. An emergency "
                                "override requires ALLOW_EMERGENCY_OVERRIDE=1, "
                                "emergency=true, and operator_override=true."
                            ),
                        },
                        status_code=403,
                    )
            # Phase 6.5 Finding #7: label only reflects the evidence actually
            # used. Emergency override is only stamped when no real evidence
            # was available; otherwise a caller who spuriously sends
            # emergency=true on a valid promotion would mis-tag the audit log.
            if current_ok:
                approval_source = "current_live"
            elif prior_attested_ok:
                approval_source = "prior_attested"
            elif emergency and operator_override:
                approval_source = "emergency_override"
            else:
                approval_source = "unknown"
            if not evidence_path:
                evidence_path = attestation_source or "live:check_promotion_eligibility"
            try:
                auton.set_autonomy_level(
                    level,
                    evidence_path=evidence_path,
                    approval_source=approval_source,
                    caller_id=caller_id,
                )
            except PermissionError as exc:
                _emit_l3_activation_denied(
                    reason=f"permission_error:{str(exc)[:160]}",
                    caller_id=caller_id,
                    approval_source=approval_source,
                    current_level=int(getattr(auton, "autonomy_level", 0)),
                    current_ok=current_ok,
                    prior_attested_ok=prior_attested_ok,
                )
                return JSONResponse(
                    {"error": str(exc)}, status_code=403,
                )
        else:
            try:
                auton.set_autonomy_level(level)
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        return {
            "ok": True,
            "level": level,
            "caller_id": caller_id,
            "reason": reason,
            "evidence_path": evidence_path,
            "current_ok": current_ok,
            "prior_attested_ok": prior_attested_ok,
        }

    @app.get("/api/autonomy/audit")
    async def api_autonomy_audit_list(limit: int = 100):
        """Return recent autonomy/escalation audit events from the durable ledger.

        Source is ``~/.jarvis/autonomy_audit.jsonl`` (Phase 6.5). This
        endpoint reads disk state and is safe against the bus being
        unwired. Newest-last, matching file order.
        """
        try:
            from autonomy.audit_ledger import get_audit_ledger
            ledger = get_audit_ledger()
            try:
                lim = max(1, min(int(limit), 1000))
            except (TypeError, ValueError):
                lim = 100
            entries = ledger.load_recent(limit=lim)
            stats = ledger.get_stats()
            return {
                "entries": entries,
                "count": len(entries),
                "wired": stats.get("wired", False),
                "log_size_kb": stats.get("log_size_kb", 0.0),
                "events_recorded_session": stats.get("events_recorded_session", 0),
            }
        except Exception as exc:
            logger.exception("Autonomy audit list failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/autonomy/escalations")
    async def api_autonomy_escalations_list():
        """List pending L3 escalation requests."""
        try:
            from autonomy.escalation import EscalationStore
            store = EscalationStore()
            pending = store.list_pending()
            return {
                "pending": [
                    {
                        "id": r.request.id,
                        "metric": r.request.metric,
                        "severity": r.request.severity,
                        "target_module": r.request.target_module,
                        "suggested_fix": r.request.suggested_fix,
                        "declared_scope": list(r.request.declared_scope),
                        "evidence_refs": list(r.request.evidence_refs),
                        "metric_context_summary": r.request.metric_context_summary,
                        "created_at": r.request.created_at,
                        "expires_at": r.request.expires_at,
                        "source": r.request.source,
                        "submitted_autonomy_level": r.request.submitted_autonomy_level,
                        "status": r.status,
                    }
                    for r in pending
                ],
            }
        except Exception as exc:
            logger.exception("List escalations failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post(
        "/api/autonomy/escalations/{escalation_id}/approve",
        dependencies=[Depends(_require_api_key)],
    )
    async def api_autonomy_escalation_approve(escalation_id: str, request: Request):
        """Approve an escalation and route through self-improvement.

        The approval runs ``attempt_improvement(manual=True)`` with
        ``declared_scope`` set to the request's narrow path list. No
        global widening of ``ALLOWED_PATHS`` ever occurs. Requires
        live autonomy level >= 3.
        """
        body = await request.json()
        approved_by = (body.get("approved_by") or "").strip()
        approval_reason = (body.get("approval_reason") or "").strip()
        if not approved_by:
            return JSONResponse(
                {"error": "approved_by is required"}, status_code=400,
            )
        if len(approval_reason) < 20:
            return JSONResponse(
                {"error": "approval_reason must be >= 20 characters"},
                status_code=400,
            )

        auton = getattr(_engine, "_autonomy_orchestrator", None) if _engine else None
        if auton is None:
            return JSONResponse({"error": "Autonomy orchestrator not available"}, status_code=503)
        if int(auton.autonomy_level) < 3:
            _emit_l3_activation_denied(
                reason="escalation_approve_below_l3",
                caller_id=approved_by,
                approval_source="escalation_approve",
                current_level=int(getattr(auton, "autonomy_level", 0)),
                escalation_id=escalation_id,
            )
            return JSONResponse(
                {"error": "L3 escalation approval requires live autonomy_level >= 3"},
                status_code=403,
            )

        cs = getattr(_engine, "_consciousness", None) or getattr(_engine, "consciousness", None)
        si_orch = getattr(cs, "_self_improve_orchestrator", None) if cs else None
        if si_orch is None:
            si_orch = getattr(_engine, "_self_improve_orchestrator", None)
        if si_orch is None:
            return JSONResponse({"error": "Self-improvement not active"}, status_code=503)

        try:
            from autonomy.escalation import EscalationStore, approve_and_apply_escalation
            store = EscalationStore()
            ollama_client = getattr(_engine, "_ollama", None)
            result = await approve_and_apply_escalation(
                store,
                request_id=escalation_id,
                approved_by=approved_by,
                approval_reason=approval_reason,
                self_improve_orchestrator=si_orch,
                ollama_client=ollama_client,
            )
            return result
        except Exception as exc:
            logger.exception("Escalation approve failed for %s", escalation_id)
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post(
        "/api/autonomy/escalations/{escalation_id}/reject",
        dependencies=[Depends(_require_api_key)],
    )
    async def api_autonomy_escalation_reject(escalation_id: str, request: Request):
        body = await request.json()
        rejected_by = (body.get("rejected_by") or "").strip()
        rejection_reason = (body.get("rejection_reason") or "").strip()
        if not rejected_by:
            return JSONResponse(
                {"error": "rejected_by is required"}, status_code=400,
            )
        if len(rejection_reason) < 20:
            return JSONResponse(
                {"error": "rejection_reason must be >= 20 characters"},
                status_code=400,
            )
        try:
            from autonomy.escalation import EscalationStore, reject_escalation
            store = EscalationStore()
            rec = reject_escalation(
                store,
                request_id=escalation_id,
                rejected_by=rejected_by,
                rejection_reason=rejection_reason,
            )
            return {
                "ok": True,
                "escalation_id": rec.request.id,
                "status": rec.status,
                "rejected_by": rec.rejected_by,
                "rejection_reason": rec.rejection_reason,
            }
        except Exception as exc:
            logger.exception("Escalation reject failed for %s", escalation_id)
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/autonomy/attestation")
    async def api_autonomy_attestation():
        """Return the accepted attestation ledger contents.

        This is the "ever-proven" memory for operator review. Each
        record carries ``artifact_status`` (hash_verified | missing |
        hash_mismatch | hash_unverifiable) and ``attestation_strength``
        (verified | archived_missing) so the UI can surface weaker
        evidence classes explicitly.
        """
        try:
            from autonomy.attestation import AttestationLedger
            ledger = AttestationLedger()
            records = ledger.load()
            return {
                "records": [
                    {
                        "capability_id": r.capability_id,
                        "evidence_source": r.evidence_source,
                        "evidence_window_start": r.evidence_window_start,
                        "evidence_window_end": r.evidence_window_end,
                        "measured_values": r.measured_values,
                        "measured_source": r.measured_source,
                        "report_hash": r.report_hash,
                        "artifact_refs": list(r.artifact_refs),
                        "artifact_status": r.artifact_status,
                        "attestation_strength": r.attestation_strength,
                        "accepted_by": r.accepted_by,
                        "accepted_at": r.accepted_at,
                        "acceptance_reason": r.acceptance_reason,
                        "schema_version": r.schema_version,
                    }
                    for r in records
                ],
            }
        except Exception as exc:
            logger.exception("Attestation ledger read failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- memory browser ------------------------------------------------------

    @app.get("/api/memories/search")
    async def api_memory_search(q: str = "", tag: str = "", type: str = "",
                                limit: int = 20):
        """Search/filter memories by keyword, tag, or type."""
        if not _engine:
            return []
        memories = _engine.get_recent_memories(200)
        results = []
        for m in memories:
            if tag and tag not in m.tags:
                continue
            if type and m.type != type:
                continue
            payload_str = m.payload if isinstance(m.payload, str) else str(m.payload)
            if q and q.lower() not in payload_str.lower():
                continue
            results.append({
                "id": m.id,
                "type": m.type,
                "payload": payload_str[:300],
                "weight": round(m.weight, 3),
                "tags": list(m.tags),
                "timestamp": m.timestamp,
            })
            if len(results) >= limit:
                break
        return results

    # -- library ingest ------------------------------------------------------

    @app.post("/api/library/ingest", dependencies=[Depends(_require_api_key)])
    async def api_library_ingest(request: Request):
        """Ingest a user-provided source (paste, URL, or file content)."""
        body = await request.json()
        content = body.get("content", "")
        url = body.get("url", "")
        title = body.get("title", "")
        source_type = body.get("source_type", "user_note")
        domain_tags = body.get("domain_tags", "")
        study_now = body.get("study_now", False)

        if not content and not url:
            return JSONResponse(
                {"error": "Provide 'content' (paste/file) or 'url'"}, status_code=400,
            )

        loop = asyncio.get_event_loop()
        from library.ingest import ingest_manual_source
        result = await loop.run_in_executor(
            None,
            lambda: ingest_manual_source(
                content=content,
                url=url,
                title=title,
                source_type=source_type,
                domain_tags=domain_tags,
                study_now=study_now,
            ),
        )
        status = 200 if result.success else 400
        return JSONResponse(result.to_dict(), status_code=status)

    @app.post("/api/library/ingest-batch", dependencies=[Depends(_require_api_key)])
    async def api_library_ingest_batch(request: Request):
        """Batch-ingest a multi-page textbook from a TOC URL."""
        body = await request.json()
        toc_url = body.get("toc_url", "")
        title = body.get("title", "")
        domain_tags = body.get("domain_tags", "textbook")
        study_now = body.get("study_now", False)
        dry_run = body.get("dry_run", False)

        if not toc_url:
            return JSONResponse({"error": "Provide 'toc_url'"}, status_code=400)

        loop = asyncio.get_event_loop()
        from library.batch_ingest import ingest_textbook
        result = await loop.run_in_executor(
            None,
            lambda: ingest_textbook(
                toc_url=toc_url,
                title=title,
                domain_tags=domain_tags,
                study_now=study_now,
                dry_run=dry_run,
            ),
        )
        status = 200 if result.success else 400
        return JSONResponse(result.to_dict(), status_code=status)

    # -- library source browser -----------------------------------------------

    @app.get("/api/library/sources")
    async def api_library_sources(
        limit: int = 20,
        offset: int = 0,
        ingested_by: str = "",
    ):
        """List library sources with metadata for the source browser panel."""
        try:
            from library.source import classify_effective_source_type, source_store
            sources = source_store.list_sources(
                limit=min(limit, 50), offset=offset, ingested_by=ingested_by,
            )
            return {
                "sources": [
                    {
                        "source_id": s.source_id,
                        "title": s.title,
                        "source_type": s.source_type,
                        "effective_source_type": classify_effective_source_type(s),
                        "venue": s.venue,
                        "year": s.year,
                        "doi": s.doi,
                        "doi_url": f"https://doi.org/{s.doi}" if s.doi else "",
                        "url": s.url,
                        "content_depth": s.content_depth or "unknown",
                        "content_chars": len(s.content_text),
                        "quality_score": round(s.quality_score, 2),
                        "provider": s.provider,
                        "studied": s.studied,
                        "study_error": s.study_error,
                        "ingested_by": s.ingested_by,
                        "retrieved_at": s.retrieved_at,
                        "content_preview": s.content_text[:200] if s.content_text else "",
                    }
                    for s in sources
                ],
                "count": len(sources),
                "offset": offset,
            }
        except Exception as exc:
            return {"sources": [], "error": str(exc)}

    @app.get("/api/library/sources/{source_id}")
    async def api_library_source_detail(source_id: str):
        """Return full detail for a single library source including content_text."""
        try:
            from library.source import classify_effective_source_type, source_store
            s = source_store.get_source(source_id)
            if not s:
                return JSONResponse({"error": "not found"}, status_code=404)
            return {
                "source_id": s.source_id,
                "title": s.title,
                "source_type": s.source_type,
                "effective_source_type": classify_effective_source_type(s),
                "venue": s.venue,
                "year": s.year,
                "doi": s.doi,
                "doi_url": f"https://doi.org/{s.doi}" if s.doi else "",
                "url": s.url,
                "authors": s.authors,
                "citation_count": s.citation_count,
                "content_depth": s.content_depth or "unknown",
                "content_text": s.content_text,
                "content_chars": len(s.content_text),
                "quality_score": round(s.quality_score, 2),
                "provider": s.provider,
                "studied": s.studied,
                "study_error": s.study_error,
                "ingested_by": s.ingested_by,
                "trust_tier": s.trust_tier,
                "domain_tags": s.domain_tags,
                "retrieved_at": s.retrieved_at,
            }
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- speaker management --------------------------------------------------

    @app.get("/api/speakers")
    async def api_speakers():
        """List all known speaker profiles (no embeddings)."""
        if not _perc_orch or not _perc_orch.speaker_id:
            return {"speakers": [], "current": "unknown", "available": False}
        return {
            "speakers": _perc_orch.speaker_id.get_profiles_summary(),
            "current": _perc_orch._current_speaker.get("name", "unknown"),
            "available": _perc_orch.speaker_id.available,
        }

    @app.post("/api/speakers/enroll", dependencies=[Depends(_require_api_key)])
    async def api_speaker_enroll(request: Request):
        """Enroll a speaker from one or more base64-encoded int16 PCM audio clips."""
        if not _perc_orch or not _perc_orch.speaker_id:
            return JSONResponse({"error": "Speaker ID not available"}, status_code=503)
        body = await request.json()
        name = body.get("name", "").strip()
        clips_b64 = body.get("clips", [])
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        from identity.name_validator import is_valid_person_name, rejection_reason
        if not is_valid_person_name(name):
            return JSONResponse({"error": f"invalid name: {rejection_reason(name)}"}, status_code=400)
        if not clips_b64:
            return JSONResponse({"error": "at least one audio clip is required"}, status_code=400)

        clips_f32 = []
        for clip_b64 in clips_b64:
            try:
                import base64 as b64
                audio_bytes = b64.b64decode(clip_b64)
                import numpy as np
                audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
                clips_f32.append(audio_i16.astype(np.float32) / 32768.0)
            except Exception as exc:
                return JSONResponse({"error": f"Failed to decode clip: {exc}"}, status_code=400)

        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(
            None, lambda: _perc_orch.speaker_id.enroll_speaker(name, clips_f32),
        )
        if ok:
            try:
                from identity.evidence_accumulator import get_accumulator
                acc = get_accumulator()
                acc.observe(name, "manual_enroll", confidence=1.0, details="dashboard speaker enrollment")
                acc.observe(name, "voice_match", confidence=0.9, details="dashboard speaker clip")
            except Exception:
                pass
            return {"status": "enrolled", "name": name, "clips": len(clips_f32)}
        return JSONResponse({"error": "Enrollment failed"}, status_code=500)

    @app.delete("/api/speakers/{name}", dependencies=[Depends(_require_api_key)])
    async def api_speaker_remove(name: str):
        """Remove a speaker profile (forget voice)."""
        if not _perc_orch or not _perc_orch.speaker_id:
            return JSONResponse({"error": "Speaker ID not available"}, status_code=503)
        removed = _perc_orch.speaker_id.remove_speaker(name)
        if removed:
            return {"status": "removed", "name": name}
        return JSONResponse({"error": f"Speaker '{name}' not found"}, status_code=404)

    # -- face management -----------------------------------------------------

    @app.get("/api/faces")
    async def api_faces():
        """List all known face profiles."""
        if not _perc_orch or not _perc_orch.face_id:
            return {"faces": [], "available": False}
        return {
            "faces": _perc_orch.face_id.get_profiles_summary(),
            "available": _perc_orch.face_id.available,
        }

    @app.post("/api/faces/enroll", dependencies=[Depends(_require_api_key)])
    async def api_face_enroll(request: Request):
        """Enroll a face from base64 JPEG crops."""
        if not _perc_orch or not _perc_orch.face_id:
            return JSONResponse({"error": "Face ID not available"}, status_code=503)
        body = await request.json()
        name = body.get("name", "").strip()
        crops_b64 = body.get("crops", [])
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        from identity.name_validator import is_valid_person_name, rejection_reason
        if not is_valid_person_name(name):
            return JSONResponse({"error": f"invalid name: {rejection_reason(name)}"}, status_code=400)
        if not crops_b64:
            return JSONResponse({"error": "at least one face crop is required"}, status_code=400)

        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(
            None, lambda: _perc_orch.face_id.enroll_face(name, crops_b64),
        )
        if ok:
            try:
                from identity.evidence_accumulator import get_accumulator
                acc = get_accumulator()
                acc.observe(name, "manual_enroll", confidence=1.0, details="dashboard face enrollment")
                acc.observe(name, "face_match", confidence=0.9, details="dashboard face crop")
            except Exception:
                pass
            return {"status": "enrolled", "name": name, "crops": len(crops_b64)}
        return JSONResponse({"error": "Face enrollment failed"}, status_code=500)

    @app.delete("/api/faces/{name}", dependencies=[Depends(_require_api_key)])
    async def api_face_remove(name: str):
        """Remove a face profile (forget face)."""
        if not _perc_orch or not _perc_orch.face_id:
            return JSONResponse({"error": "Face ID not available"}, status_code=503)
        removed = _perc_orch.face_id.remove_face(name)
        if removed:
            return {"status": "removed", "name": name}
        return JSONResponse({"error": f"Face '{name}' not found"}, status_code=404)

    # -- identity management -------------------------------------------------

    @app.get("/api/identity")
    async def api_identity():
        """Get current fused identity status."""
        if not _perc_orch:
            return {"identity": "unknown", "fusion": {}}
        return {
            "identity": _perc_orch._current_speaker.get("name", "unknown"),
            "fusion": _perc_orch.identity_fusion.get_status(),
            "face": _perc_orch._current_face,
        }

    @app.post("/api/identity/forget-all", dependencies=[Depends(_require_api_key)])
    async def api_forget_all(request: Request):
        """Forget all biometric data for a person (voice + face)."""
        body = await request.json()
        name = body.get("name", "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        removed_voice = _perc_orch.speaker_id.remove_speaker(name) if _perc_orch and _perc_orch.speaker_id else False
        removed_face = _perc_orch.face_id.remove_face(name) if _perc_orch and _perc_orch.face_id else False
        return {"voice_removed": removed_voice, "face_removed": removed_face, "name": name}

    @app.post("/api/identity/toggle-face", dependencies=[Depends(_require_api_key)])
    async def api_toggle_face(request: Request):
        """Enable or disable face recognition."""
        body = await request.json()
        enabled = body.get("enabled", True)
        if _perc_orch and _perc_orch.face_id:
            _perc_orch.face_id.available = enabled
        if _perc_orch:
            _perc_orch.identity_fusion.set_enabled(enabled)
        return {"face_enabled": enabled}

    # -- identity candidates --------------------------------------------------

    @app.get("/api/identity/candidates")
    async def api_identity_candidates():
        """List all identity candidates with evidence scores."""
        try:
            from identity.evidence_accumulator import get_accumulator
            acc = get_accumulator()
            return {"candidates": acc.get_all_candidates(), **acc.get_stats()}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/identity/candidates/promote", dependencies=[Depends(_require_api_key)])
    async def api_promote_candidate(request: Request):
        """Manually promote a candidate to persistent."""
        body = await request.json()
        name = body.get("name", "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        try:
            from identity.evidence_accumulator import get_accumulator
            ok = get_accumulator().force_promote(name)
            if ok:
                return {"status": "promoted", "name": name}
            return JSONResponse({"error": f"Cannot promote: invalid name '{name}'"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/identity/candidates/reject", dependencies=[Depends(_require_api_key)])
    async def api_reject_candidate(request: Request):
        """Manually reject a candidate."""
        body = await request.json()
        name = body.get("name", "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        try:
            from identity.evidence_accumulator import get_accumulator
            ok = get_accumulator().reject_candidate(name)
            if ok:
                return {"status": "rejected", "name": name}
            return JSONResponse({"error": f"No candidate named '{name}'"}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- intentions (truth layer, Stage 0) -----------------------------------

    @app.get("/api/intentions")
    async def api_intentions():
        """Intention Registry status: open count, stale count, 7-day histogram.

        Read-only truth layer, no delivery mechanism.
        """
        try:
            from cognition.intention_registry import intention_registry
            snap = intention_registry.get_status()
            open_recs = [r.to_dict() for r in intention_registry.get_open()]
            recent = [r.to_dict() for r in intention_registry.get_recent_resolved(n=50)]
            graduation = intention_registry.get_graduation_status()
            return {
                "status": snap,
                "open": open_recs,
                "recent_resolved": recent,
                "graduation": graduation,
            }
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- intention resolver (Stage 1) ----------------------------------------

    @app.get("/api/intention-resolver")
    async def api_intention_resolver():
        """IntentionResolver status: stage, shadow metrics, recent verdicts."""
        try:
            from cognition.intention_resolver import get_intention_resolver
            resolver = get_intention_resolver()
            return resolver.get_status()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/intention-resolver/rollback")
    async def api_intention_resolver_rollback(request: Request):
        """Demote resolver one rung in the promotion ladder (API-key protected)."""
        _verify_api_key(request)
        try:
            from cognition.intention_resolver import get_intention_resolver
            resolver = get_intention_resolver()
            new_stage = resolver.rollback()
            return {"status": "ok", "new_stage": new_stage}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/intention-resolver/stage")
    async def api_intention_resolver_stage(request: Request):
        """Set resolver stage manually (API-key protected)."""
        _verify_api_key(request)
        try:
            body = await request.json()
            stage = body.get("stage", "")
            from cognition.intention_resolver import get_intention_resolver
            resolver = get_intention_resolver()
            if resolver.set_stage(stage):
                return {"status": "ok", "stage": stage}
            return JSONResponse({"error": f"Invalid stage: {stage}"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- goals ---------------------------------------------------------------

    @app.get("/api/goals")
    async def api_goals():
        """Get Goal Continuity Layer status."""
        try:
            from goals.goal_manager import get_goal_manager
            return get_goal_manager().get_status()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/goals/observe")
    async def api_goal_observe(request: Request):
        """Inject a GoalSignal manually for testing the goal lifecycle."""
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"error": "content is required"}, status_code=400)
        try:
            from goals.goal_manager import get_goal_manager
            from goals.goal import GoalSignal
            signal = GoalSignal(
                signal_type=body.get("signal_type", "manual"),
                source=body.get("source", "dashboard_api"),
                source_scope=body.get("source_scope", "user"),
                content=content,
                tag_cluster=tuple(body.get("tag_cluster", [])),
                priority_hint=float(body.get("priority_hint", 0.5)),
            )
            result = get_goal_manager().observe_signal(signal)
            resp: dict = {"outcome": result.outcome, "reason": result.reason}
            if result.goal:
                resp["goal"] = result.goal.to_dict()
            return resp
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/goals/{goal_id}/complete")
    async def api_goal_complete(goal_id: str, request: Request):
        body = await request.json() if await request.body() else {}
        reason = body.get("reason", "Manual completion via API")
        try:
            from goals.goal_manager import get_goal_manager
            ok = get_goal_manager().complete_goal(goal_id, reason)
            if ok:
                return {"status": "completed", "goal_id": goal_id}
            return JSONResponse({"error": "Cannot complete goal"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/goals/{goal_id}/abandon")
    async def api_goal_abandon(goal_id: str, request: Request):
        body = await request.json() if await request.body() else {}
        reason = body.get("reason", "Manual abandonment via API")
        try:
            from goals.goal_manager import get_goal_manager
            ok = get_goal_manager().abandon_goal(goal_id, reason)
            if ok:
                return {"status": "abandoned", "goal_id": goal_id}
            return JSONResponse({"error": "Cannot abandon goal"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/goals/{goal_id}/pause")
    async def api_goal_pause(goal_id: str, request: Request):
        body = await request.json() if await request.body() else {}
        reason = body.get("reason", "")
        try:
            from goals.goal_manager import get_goal_manager
            ok = get_goal_manager().pause_goal(goal_id, reason)
            if ok:
                return {"status": "paused", "goal_id": goal_id}
            return JSONResponse({"error": "Cannot pause goal"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/goals/{goal_id}/resume")
    async def api_goal_resume(goal_id: str):
        try:
            from goals.goal_manager import get_goal_manager
            ok = get_goal_manager().resume_goal(goal_id)
            if ok:
                return {"status": "resumed", "goal_id": goal_id}
            return JSONResponse({"error": "Cannot resume goal"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- capability acquisition pipeline ----------------------------------------

    @app.get("/api/acquisition")
    async def api_acquisition():
        """Get Capability Acquisition Pipeline status."""
        try:
            orch = _get_acquisition_orchestrator()
            if orch is None:
                return {"active_count": 0, "total_count": 0, "recent": [], "enabled": False}
            return {**orch.get_status(), "enabled": True}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- plan evaluator shadow stats -------------------------------------------

    @app.get("/api/acquisition/plan-evaluator")
    async def api_plan_evaluator_stats():
        """Get plan evaluator shadow NN stats: sample count, maturity, accuracy."""
        try:
            from pathlib import Path
            import json as _json

            shadow_dir = Path.home() / ".jarvis" / "acquisition_shadows"
            shadows: list[dict] = []
            if shadow_dir.exists():
                for p in sorted(shadow_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
                    try:
                        shadows.append(_json.loads(p.read_text()))
                    except Exception:
                        pass

            total = len(shadows)
            resolved = [s for s in shadows if s.get("actual_verdict")]
            correct = sum(1 for s in resolved if s.get("correct") is True)
            accuracy = correct / len(resolved) if resolved else 0.0

            # maturity bands based on resolved sample count
            n_resolved = len(resolved)
            if n_resolved < 15:
                maturity = "bootstrap"
            elif n_resolved < 50:
                maturity = "early_noisy"
            elif n_resolved < 100:
                maturity = "preliminary"
            elif n_resolved < 250:
                maturity = "meaningful"
            else:
                maturity = "stable"

            verdict_dist: dict[str, int] = {}
            for s in resolved:
                v = s.get("actual_class", s.get("actual_verdict", "unknown"))
                verdict_dist[v] = verdict_dist.get(v, 0) + 1

            # --- stratified accuracy breakdowns ---
            def _accuracy_bucket(items: list[dict]) -> dict:
                n = len(items)
                c = sum(1 for i in items if i.get("correct") is True)
                return {"total": n, "correct": c, "accuracy": round(c / n, 4) if n else 0.0}

            by_risk_tier: dict[str, dict] = {}
            by_outcome_class: dict[str, dict] = {}
            by_reason_category: dict[str, dict] = {}
            for s in resolved:
                rt = str(s.get("risk_tier", "unknown"))
                by_risk_tier.setdefault(rt, []).append(s)
                oc = s.get("outcome_class", "unknown") or "unknown"
                by_outcome_class.setdefault(oc, []).append(s)
                rc = s.get("reason_category", "unknown") or "unknown"
                by_reason_category.setdefault(rc, []).append(s)
            by_risk_tier = {k: _accuracy_bucket(v) for k, v in by_risk_tier.items()}
            by_outcome_class = {k: _accuracy_bucket(v) for k, v in by_outcome_class.items()}
            by_reason_category = {k: _accuracy_bucket(v) for k, v in by_reason_category.items()}

            sample_count = 0
            try:
                from hemisphere.distillation import DistillationCollector
                collector = DistillationCollector.instance()
                if collector:
                    sample_count = collector.count("plan_features") + collector.count("acquisition_planner")
            except Exception:
                pass

            return {
                "enabled": True,
                "maturity": maturity,
                "sample_count": sample_count,
                "shadow_predictions_total": total,
                "shadow_predictions_resolved": n_resolved,
                "shadow_accuracy": round(accuracy, 4),
                "correct_count": correct,
                "verdict_distribution": verdict_dist,
                "accuracy_by_risk_tier": by_risk_tier,
                "accuracy_by_outcome_class": by_outcome_class,
                "accuracy_by_reason_category": by_reason_category,
                "recent_shadows": shadows[:10],
            }
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/acquisition/{acquisition_id}")
    async def api_acquisition_detail(acquisition_id: str):
        """Get a single acquisition job detail with all artifact contents."""
        try:
            orch = _get_acquisition_orchestrator()
            if orch is None:
                return JSONResponse({"error": "Acquisition pipeline not enabled"}, status_code=404)
            job = orch.get_job(acquisition_id)
            if job is None:
                return JSONResponse({"error": "Job not found"}, status_code=404)
            result = job.to_dict()
            store = orch._store
            if job.plan_id:
                plan = store.load_plan(job.plan_id)
                if plan:
                    result["plan"] = plan.to_dict()
            if job.plan_review_id:
                review = store.load_review(job.plan_review_id)
                if review:
                    result["review"] = review.to_dict()
            if job.code_bundle_id:
                bundle = store.load_code_bundle(job.code_bundle_id)
                if bundle:
                    result["code_bundle"] = bundle.to_dict()
            if job.verification_id:
                vb = store.load_verification(job.verification_id)
                if vb:
                    result["verification"] = vb.to_dict()
            doc_artifacts = []
            for doc_id in (job.doc_artifact_ids or []):
                doc = store.load_doc(doc_id)
                if doc:
                    doc_artifacts.append(doc.to_dict())
            if doc_artifacts:
                result["doc_artifacts"] = doc_artifacts
            if job.plugin_id:
                try:
                    from tools.plugin_registry import get_plugin_registry
                    reg = get_plugin_registry()
                    rec = reg.get_record(job.plugin_id)
                    if rec:
                        success_rate = (
                            rec.success_count / rec.invocation_count
                            if rec.invocation_count else 0.0
                        )
                        result["plugin_record"] = {
                            "name": rec.name,
                            "state": rec.state,
                            "version": rec.version,
                            "risk_tier": rec.risk_tier,
                            "supervision_mode": getattr(rec, "supervision_mode", ""),
                            "execution_mode": getattr(rec, "execution_mode", ""),
                            "invocation_count": rec.invocation_count,
                            "success_count": rec.success_count,
                            "failure_count": rec.failure_count,
                            "success_rate": round(success_rate, 3),
                            "activated_at": getattr(rec, "activated_at", 0),
                            "updated_at": getattr(rec, "updated_at", 0),
                        }
                except Exception:
                    pass
            return result
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/acquisition/create", dependencies=[Depends(_require_api_key)])
    async def api_acquisition_create(request: Request):
        """Create a new acquisition job from user intent text."""
        body = await request.json()
        text = body.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "text is required"}, status_code=400)
        try:
            orch = _get_acquisition_orchestrator()
            if orch is None:
                return JSONResponse({"error": "Acquisition pipeline not enabled"}, status_code=400)
            job = orch.create(text, requested_by={"source": "dashboard_api"})
            return job.to_dict()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/acquisition/{acquisition_id}/approve-plan", dependencies=[Depends(_require_api_key)])
    async def api_acquisition_approve_plan(acquisition_id: str, request: Request):
        """Approve or reject an acquisition plan."""
        body = await request.json()
        verdict = body.get("verdict", "")
        notes = body.get("notes", "")
        reason_category = body.get("reason_category", "unknown")
        suggested_changes = body.get("suggested_changes")
        try:
            orch = _get_acquisition_orchestrator()
            if orch is None:
                return JSONResponse({"error": "Acquisition pipeline not enabled"}, status_code=400)
            ok = orch.approve_plan(
                acquisition_id, verdict, notes,
                suggested_changes=suggested_changes,
                reason_category=reason_category,
            )
            if ok:
                return {"status": "reviewed", "acquisition_id": acquisition_id, "verdict": verdict}
            return JSONResponse({"error": "Cannot review — job not in awaiting_plan_review state"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/acquisition/{acquisition_id}/approve-deploy", dependencies=[Depends(_require_api_key)])
    async def api_acquisition_approve_deploy(acquisition_id: str, request: Request):
        """Approve or deny deployment of an acquisition."""
        body = await request.json()
        approved = body.get("approved", False)
        try:
            orch = _get_acquisition_orchestrator()
            if orch is None:
                return JSONResponse({"error": "Acquisition pipeline not enabled"}, status_code=400)
            ok = orch.approve_deployment(acquisition_id, approved)
            if ok:
                return {"status": "approved" if approved else "denied", "acquisition_id": acquisition_id}
            return JSONResponse({"error": "Cannot approve — job not in awaiting_approval state"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/acquisition/{acquisition_id}/cancel", dependencies=[Depends(_require_api_key)])
    async def api_acquisition_cancel(acquisition_id: str, request: Request):
        """Cancel and remove an acquisition job (any state)."""
        body = await request.json() if await request.body() else {}
        reason = body.get("reason", "operator_cancelled")
        try:
            orch = _get_acquisition_orchestrator()
            if orch is None:
                return JSONResponse({"error": "Acquisition pipeline not enabled"}, status_code=400)
            ok = orch.cancel_job(acquisition_id, reason=reason)
            if ok:
                return {"status": "cancelled", "acquisition_id": acquisition_id}
            return JSONResponse({"error": "Job not found"}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    def _get_acquisition_orchestrator():
        try:
            if _engine:
                return getattr(_engine, '_acquisition_orchestrator', None)
        except Exception:
            pass
        return None

    # -- plugin registry -------------------------------------------------------

    @app.get("/api/plugins")
    async def api_plugins():
        """Get plugin registry status."""
        try:
            from tools.plugin_registry import get_plugin_registry
            reg = get_plugin_registry()
            return reg.get_status()
        except Exception as exc:
            logger.warning("/api/plugins failed: %s", exc)
            return {"plugins": [], "total": 0, "active": 0}

    @app.get("/api/plugins/{name}")
    async def api_plugin_detail(name: str):
        """Get details for a single plugin."""
        try:
            from tools.plugin_registry import get_plugin_registry
            reg = get_plugin_registry()
            rec = reg.get_record(name)
            if rec is None:
                return JSONResponse({"error": "Plugin not found"}, status_code=404)
            success_rate = (
                rec.success_count / rec.invocation_count
                if rec.invocation_count else 0.0
            )
            detail = {
                "name": rec.name,
                "state": rec.state,
                "version": rec.version,
                "risk_tier": rec.risk_tier,
                "acquisition_id": getattr(rec, "acquisition_id", ""),
                "code_hash": getattr(rec, "code_hash", ""),
                "execution_mode": getattr(rec, "execution_mode", ""),
                "plugin_directory": f"brain/tools/plugins/{rec.name}/",
                "storage_boundary": (
                    "Generated plugin package loaded by PluginRegistry; "
                    "not a core-code patch."
                ),
                "invoke_count": getattr(rec, "invoke_count", getattr(rec, "invocation_count", 0)),
                "invocation_count": getattr(rec, "invocation_count", getattr(rec, "invoke_count", 0)),
                "success_count": getattr(rec, "success_count", 0),
                "failure_count": rec.failure_count,
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(getattr(rec, "avg_latency_ms", 0.0), 1),
                "last_invocation_at": getattr(rec, "last_invocation_at", 0),
                "activated_at": getattr(rec, "activated_at", 0),
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "supervision_mode": getattr(rec, "supervision_mode", ""),
                "description": getattr(rec, "description", ""),
            }
            acq_id = getattr(rec, "acquisition_id", "")
            if acq_id:
                orch = _get_acquisition_orchestrator()
                if orch is not None:
                    job = orch.get_job(acq_id)
                    if job:
                        store = orch._store
                        detail["acquisition"] = {
                            "acquisition_id": job.acquisition_id,
                            "title": job.title,
                            "status": job.status,
                            "outcome_class": job.outcome_class,
                            "risk_tier": job.risk_tier,
                            "requested_by": job.requested_by,
                            "plan_id": job.plan_id,
                            "code_bundle_id": job.code_bundle_id,
                            "verification_id": job.verification_id,
                            "activation_diagnostics": getattr(job, "activation_diagnostics", {}),
                        }
                        if job.plan_id:
                            plan = store.load_plan(job.plan_id)
                            if plan:
                                detail["plan"] = plan.to_dict()
                        if job.verification_id:
                            vb = store.load_verification(job.verification_id)
                            if vb:
                                detail["verification"] = vb.to_dict()
                        if job.code_bundle_id:
                            bundle = store.load_code_bundle(job.code_bundle_id)
                            if bundle:
                                detail["code_bundle"] = bundle.to_dict()
            if "code_bundle" not in detail:
                try:
                    from pathlib import Path
                    plugin_dir = Path(__file__).resolve().parent.parent / "tools" / "plugins" / name
                    if plugin_dir.exists():
                        code_files = {}
                        for path in sorted(plugin_dir.glob("*.py")):
                            code_files[path.name] = path.read_text(encoding="utf-8", errors="replace")
                        if code_files:
                            detail["code_bundle"] = {"code_files": code_files}
                except Exception:
                    pass
            return detail
        except Exception as exc:
            logger.warning("/api/plugins/%s failed: %s", name, exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/plugins/{name}/promote", dependencies=[Depends(_require_api_key)])
    async def api_plugin_promote(name: str):
        """Promote a plugin to next supervision tier."""
        try:
            from tools.plugin_registry import get_plugin_registry
            reg = get_plugin_registry()
            ok = reg.promote(name)
            if ok:
                rec = reg.get_record(name)
                return {"status": "promoted", "name": name, "new_state": rec.state if rec else "unknown"}
            return JSONResponse({"error": "Cannot promote"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/plugins/{name}/disable", dependencies=[Depends(_require_api_key)])
    async def api_plugin_disable(name: str):
        """Disable a plugin."""
        try:
            from tools.plugin_registry import get_plugin_registry
            reg = get_plugin_registry()
            ok = reg.disable(name, "disabled via dashboard")
            if ok:
                return {"status": "disabled", "name": name}
            return JSONResponse({"error": "Cannot disable"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/plugins/{name}/activate", dependencies=[Depends(_require_api_key)])
    async def api_plugin_activate(name: str, request: Request):
        """Activate a quarantined/disabled plugin into shadow mode."""
        try:
            body = await request.json()
            mode = body.get("supervision_mode", "shadow")
            from tools.plugin_registry import get_plugin_registry
            reg = get_plugin_registry()
            ok = reg.activate(name, mode)
            if ok:
                return {"status": "activated", "name": name, "supervision_mode": mode}
            return JSONResponse({"error": "Cannot activate"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/plugins/{name}/rollback", dependencies=[Depends(_require_api_key)])
    async def api_plugin_rollback(name: str):
        """Rollback a plugin to its previous version."""
        try:
            from tools.plugin_registry import get_plugin_registry
            reg = get_plugin_registry()
            ok = reg.rollback(name)
            if ok:
                return {"status": "rolled_back", "name": name}
            return JSONResponse({"error": "Cannot rollback"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/codegen")
    async def api_codegen():
        """Get CodeGenService status."""
        try:
            codegen = _cache.get("codegen", {})
            coder = codegen.get("coder", {})
            acq = _cache.get("acquisition", {})
            return {
                "authority": codegen.get("authority", "infrastructure_only"),
                "owner": codegen.get("owner", "shared_codegen"),
                "enabled": codegen.get("enabled", bool(coder)),
                "coder": coder,
                "codegen_available": codegen.get("codegen_available", coder.get("available", False)),
                "acquisition_enabled": bool(
                    acq.get("enabled", False)
                    or codegen.get("consumers", {}).get("acquisition", False)
                ),
                "active_consumer": codegen.get("active_consumer", ""),
                "last_consumer": codegen.get("last_consumer", ""),
                "consumers": codegen.get("consumers", {}),
                "total_generations": codegen.get("total_generations", coder.get("total_generations", 0)),
                "total_validations": codegen.get("total_validations", 0),
                "total_failures": codegen.get("total_failures", 0),
            }
        except Exception as exc:
            return {"authority": "infrastructure_only", "coder": {}, "codegen_available": False, "error": str(exc)}

    # -- onboarding / companion training -------------------------------------

    @app.post("/api/onboarding/start", dependencies=[Depends(_require_api_key)])
    async def api_onboarding_start():
        """Start the 7-stage companion training playbook."""
        try:
            from personality.onboarding import get_onboarding_manager
            mgr = get_onboarding_manager()
            if mgr.graduated:
                return JSONResponse({"error": "Already graduated"}, status_code=400)
            mgr.start()
            return {"status": "started", "stage": mgr.current_stage, "day": mgr.current_day}
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/onboarding/status")
    async def api_onboarding_status():
        """Get companion training status and progress."""
        try:
            from personality.onboarding import get_onboarding_manager
            return get_onboarding_manager().get_status()
        except Exception:
            return {"enabled": False, "active": False}

    # -- voice test ----------------------------------------------------------

    @app.post("/api/voice-test", dependencies=[Depends(_require_api_key)])
    async def api_voice_test(request: Request):
        """Send a text string to the Pi for TTS playback (for testing)."""
        if not _perception:
            return JSONResponse({"error": "Perception not connected"}, status_code=503)
        body = await request.json()
        text = body.get("text", "")
        if not text:
            return JSONResponse({"error": "Missing 'text'"}, status_code=400)
        try:
            from skills.capability_gate import capability_gate as _cg
            text = _cg.check_text(text)
        except Exception:
            pass
        _perception.broadcast({
            "type": "response_chunk",
            "text": text,
            "tone": _engine.get_state().get("tone", "professional") if _engine else "professional",
            "phase": "SPEAKING",
        })
        _perception.broadcast({
            "type": "response_end",
            "text": "",
            "tone": "",
            "phase": "LISTENING",
        })
        return {"sent": text}

    # -- system control (graceful shutdown / restart) -------------------------

    async def _graceful_save() -> dict[str, Any]:
        """Save all state in proper order. Returns a status report."""
        saved: list[str] = []
        errors: list[str] = []

        # 1. Consciousness state (stage, transcendence, evolution, etc.)
        try:
            from memory.persistence import consciousness_persistence
            if _engine:
                consciousness_persistence.save_from_system(_engine.consciousness)
                saved.append("consciousness_state")
        except Exception as e:
            errors.append(f"consciousness: {e}")

        # 2. Memory persistence (all memories to disk)
        try:
            if _persistence:
                _persistence.save()
                saved.append("memories")
        except Exception as e:
            errors.append(f"memories: {e}")

        # 3. Episodic memory
        try:
            if _episodes:
                _episodes.end_episode()
                _episodes.save()
                saved.append("episodes")
        except Exception as e:
            errors.append(f"episodes: {e}")

        # 4. Context builder (conversation context)
        try:
            from reasoning.context import context_builder
            context_builder.save()
            saved.append("context")
        except Exception as e:
            errors.append(f"context: {e}")

        # 5. Policy experience buffer
        try:
            from policy.experience_buffer import ExperienceBuffer
            buf = getattr(_engine, "_experience_buffer", None)
            if buf is not None and hasattr(buf, "flush"):
                buf.flush()
                saved.append("policy_experience")
        except Exception as e:
            errors.append(f"policy: {e}")

        # 6. Hemisphere NN models
        try:
            hemi = getattr(_engine, "_hemisphere_orchestrator", None)
            if hemi and hasattr(hemi, "save_all_models"):
                hemi.save_all_models()
                saved.append("hemisphere_models")
        except Exception as e:
            errors.append(f"hemisphere: {e}")

        # 7. Vector store
        try:
            from memory.search import get_vector_store
            vs = get_vector_store()
            if vs:
                vs.close()
                saved.append("vector_store")
        except Exception as e:
            errors.append(f"vector_store: {e}")

        return {"saved": saved, "errors": errors}

    async def _graceful_shutdown() -> dict[str, Any]:
        """Full graceful shutdown: save state → stop subsystems → stop kernel."""
        result = await _graceful_save()

        # Stop processors (presence, audio, vision, ambient, screen)
        if _processors:
            for name in ("presence", "audio", "vision", "ambient", "screen"):
                proc = _processors.get(name)
                if proc:
                    try:
                        proc.stop()
                    except Exception:
                        pass

        # Stop perception server
        if _perception:
            try:
                await _perception.stop()
            except Exception:
                pass

        # Stop auto-save
        if _persistence:
            try:
                _persistence.stop_auto_save()
            except Exception:
                pass

        # Stop kernel
        if _engine:
            try:
                _engine.stop()
                result["saved"].append("kernel_stopped")
            except Exception as e:
                result["errors"].append(f"kernel: {e}")

        return result

    @app.post("/api/system/save", dependencies=[Depends(_require_api_key)])
    async def api_system_save():
        """Save all state without stopping anything."""
        result = await _graceful_save()
        return {"status": "ok", **result}

    @app.post("/api/system/shutdown", dependencies=[Depends(_require_api_key)])
    async def api_system_shutdown():
        """Gracefully save all state and shut down the brain process."""
        global _shutting_down
        if _shutting_down:
            return JSONResponse({"error": "Already shutting down"}, status_code=409)
        _shutting_down = True
        logger.info("Dashboard-initiated graceful shutdown")

        result = await _graceful_shutdown()
        logger.info("Shutdown complete: saved=%s errors=%s", result["saved"], result["errors"])

        # Schedule process exit after response is sent
        async def _deferred_exit():
            await asyncio.sleep(0.5)
            import os
            logger.info("Exiting process")
            os._exit(0)

        asyncio.get_event_loop().create_task(_deferred_exit())
        return {"status": "shutting_down", **result}

    @app.post("/api/system/restart", dependencies=[Depends(_require_api_key)])
    async def api_system_restart():
        """Gracefully save all state and restart the brain process."""
        global _shutting_down
        if _shutting_down:
            return JSONResponse({"error": "Already shutting down"}, status_code=409)
        _shutting_down = True
        logger.info("Dashboard-initiated graceful restart")

        result = await _graceful_shutdown()
        logger.info("Shutdown complete, restarting: saved=%s errors=%s",
                     result["saved"], result["errors"])

        async def _deferred_restart():
            await asyncio.sleep(0.5)
            from main import _request_restart
            _request_restart("dashboard_restart", "Dashboard-initiated restart")
            logger.info("Exiting for supervisor restart")
            sys.exit(10)

        asyncio.get_event_loop().create_task(_deferred_restart())
        return {"status": "restarting", **result}

    @app.get("/api/system/status")
    async def api_system_status():
        """Check if the brain is shutting down or alive."""
        return {
            "alive": True,
            "shutting_down": _shutting_down,
            "running": _engine.is_running() if _engine else False,
        }

    @app.get("/api/system/code-freshness")
    async def api_system_code_freshness():
        """Report whether on-disk source is newer than the running process.

        Used by the dashboard to surface a non-blocking "newer code on disk,
        click Restart to load it" banner after `sync-desktop.sh`. Not an
        authority on anything — it's a convenience signal.
        """
        return _scan_code_freshness()

    # -- meta / truth-baseline endpoints (P1.7) ------------------------------

    @app.get("/api/meta/build-status")
    async def api_meta_build_status():
        """Per-page freshness baseline.

        Returns last-modified timestamps + git sha (best-effort) for each
        dashboard page. Static prose pages are served from disk on each
        request, so their mtimes are inventory metadata, not a restart signal.
        """
        import subprocess

        static_dir = os.path.join(os.path.dirname(__file__), "static")
        pages = [
            "index.html", "self_improve.html", "docs.html",
            "science.html", "history.html", "api.html", "showcase.html",
            "learning.html", "maturity.html", "hrr.html", "hrr_scene.html",
        ]
        page_info: dict[str, Any] = {}
        for p in pages:
            path = os.path.join(static_dir, p)
            if os.path.exists(path):
                st = os.stat(path)
                page_info[p] = {
                    "mtime": st.st_mtime,
                    "size": st.st_size,
                    "exists": True,
                }
            else:
                page_info[p] = {"exists": False}

        sha = None
        try:
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            proc = subprocess.run(
                ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=2,
            )
            if proc.returncode == 0:
                sha = proc.stdout.strip() or None
        except Exception:
            sha = None

        return {
            "pages": page_info,
            "git_sha": sha,
            "process_start_time": _STARTUP_TS,
            "static_pages_served_live": True,
            "static_pages_restart_required": False,
            "now": time.time(),
        }

    @app.get("/api/meta/status-markers")
    async def api_meta_status_markers():
        """Single authoritative status map for architectural claims.

        Static map keyed by feature/pillar id, values are
        ``SHIPPED | PARTIAL | PRE-MATURE | DEFERRED``. Kept inline so the
        dashboard prose pages render from one source of truth rather than
        from per-page hard-coded badges.

        This is the Pillar 10 enforcement surface: every architectural
        claim on the prose pages should render its marker from here, and
        every marker should be justified by evidence reachable from
        ``/api/self-test`` or ``/api/full-snapshot``.
        """
        # Phase E language-kernel identity auto-flips SHIPPED once the
        # operator has seeded at least one Phase C checkpoint into the
        # registry. ``matches_live_artifact`` is intentionally NOT part
        # of the guard: during active Phase C training the on-disk
        # student checkpoint hash drifts, and that drift is a healthy
        # signal the governor is working — the live registered artifact
        # stays stable while new candidates accumulate.
        phase_e_marker = "PRE-MATURE"
        try:
            from language.kernel import get_language_kernel_registry
            _lk_state = get_language_kernel_registry().get_state()
            if (
                _lk_state.get("status") == "registered"
                and _lk_state.get("total_artifacts", 0) >= 1
                and _lk_state.get("live_artifact") is not None
            ):
                phase_e_marker = "SHIPPED"
        except Exception:
            phase_e_marker = "PRE-MATURE"

        return {
            "generated_at": time.time(),
            "markers": {
                # --- Phase governance ------------------------------------
                "phase_6_5_l3_governance": "SHIPPED",
                "phase_7_isolated_subprocess": "SHIPPED",
                "phase_e_language_kernel_identity": phase_e_marker,
                # --- Skills / perception ---------------------------------
                "speaker_diarization_v1": "PARTIAL",
                "voice_intent_distillation": "PARTIAL",
                "voice_intent_shadow_runner": "SHIPPED",
                # --- Epistemic infra -------------------------------------
                "belief_graph_causal_writer": "SHIPPED",
                "belief_graph_temporal_sequence_writer": "SHIPPED",
                "belief_graph_shared_subject_writer": "SHIPPED",
                # --- Observability ---------------------------------------
                "observability_truth_probe": "SHIPPED",
                "observability_self_test_endpoint": "SHIPPED",
                "observability_schema_emission_audit": "SHIPPED",
                "observability_freshness_banner": "SHIPPED",
                # --- Autonomy --------------------------------------------
                "autonomy_durable_audit_subscriber": "SHIPPED",
                "autonomy_attestation_ledger": "SHIPPED",
                "autonomy_auto_promotion_invariant": "SHIPPED",
                # --- Research lanes (PRE-MATURE, shadow-only) -----------
                # P4 Holographic Cognition / HRR-VSA. Stays PRE-MATURE
                # until operator approves promotion through the normal
                # specialist lifecycle + Phase 6.5 governance. See
                # docs/plans/p4_holographic_cognition_vsa.plan.md and the
                # "HRR / VSA Governance Rules" section of AGENTS.md.
                "holographic_cognition_hrr": "PRE-MATURE",
                # P5 Internal Mental World / Spatial HRR Scene lane.
                # Derived-only projection over canonical perception,
                # zero authority, twin-gated (ENABLE_HRR_SHADOW +
                # ENABLE_HRR_SPATIAL_SCENE). Must remain PRE-MATURE
                # until a governance decision explicitly promotes it.
                # See docs/plans/p5_internal_mental_world_spatial_hrr.plan.md.
                "spatial_hrr_mental_world": "PRE-MATURE",
                # Intention Resolver (Stage 1). Shadow-only heuristic
                # delivery scoring. Starts PRE-MATURE; matures through
                # 5-rung promotion ladder (shadow → advisory → active).
                "intention_resolver": "PRE-MATURE",
                # --- Deferred / not-in-scope -----------------------------
                "security_hardening": "DEFERRED",
                "pii_scrub": "DEFERRED",
                "wake_word_manager": "DEFERRED",
                "reset_ceremony": "DEFERRED",
            },
            "legend": {
                "SHIPPED": "Feature is live in the current process, evidence reachable from /api/self-test or on-disk artifacts.",
                "PARTIAL": "Scaffolding/governance shipped; maturity-gated on live data and continuity-preserving accumulation.",
                "PRE-MATURE": "Code present, but no live evidence yet. Matures in place on the current brain.",
                "DEFERRED": "Explicitly out of scope for the open-source release truth pass, or (for reset_ceremony) reclassified as optional destructive proof. Continuity-preserving validation is the chosen evidence path; see docs/validation_reports/continuity_baseline_2026-04-23.md.",
            },
        }

    @app.get("/api/maturity-gates")
    async def api_maturity_gates():
        """Structured parse of docs/MATURITY_GATES_REFERENCE.md.

        Returns sections keyed by heading + a live-values overlay from
        ``/api/full-snapshot`` for the gates it knows how to resolve.
        The frontend renders the sections as reference prose with the
        live value inlined alongside the threshold.
        """
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        path = os.path.join(repo_root, "docs", "MATURITY_GATES_REFERENCE.md")
        if not os.path.exists(path):
            return {
                "source": "docs/MATURITY_GATES_REFERENCE.md",
                "exists": False,
                "sections": [],
            }
        import re as _re

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            st = os.stat(path)
        except Exception as exc:
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for line in text.splitlines():
            m = _re.match(r"^##\s+(?P<title>.+?)\s*$", line)
            if m:
                if current is not None:
                    sections.append(current)
                title = m.group("title")
                num_m = _re.match(r"^(?P<num>\d+)\.\s+(?P<rest>.+)$", title)
                current = {
                    "title": title,
                    "section_num": (
                        int(num_m.group("num")) if num_m else None
                    ),
                    "slug": _re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"),
                    "body": [],
                }
                continue
            if current is not None:
                current["body"].append(line)
        if current is not None:
            sections.append(current)
        for s in sections:
            s["body"] = "\n".join(s["body"]).strip()

        # Attach a small live-values overlay for the subsystems the
        # maturity-gates page most often needs to show.
        live: dict[str, Any] = {}
        try:
            auto = _cache.get("autonomy_live") if isinstance(_cache, dict) else None
            if isinstance(auto, dict):
                live["autonomy"] = {
                    "current_level": auto.get("autonomy_level"),
                    "ever_ok": auto.get("ever_ok"),
                    "current_ok": auto.get("current_ok"),
                    "prior_attested_ok": auto.get("prior_attested_ok"),
                }
            soul = _cache.get("soul_integrity") if isinstance(_cache, dict) else None
            if isinstance(soul, dict):
                live["soul_integrity"] = {
                    "composite": soul.get("composite"),
                }
        except Exception:
            pass

        return {
            "source": "docs/MATURITY_GATES_REFERENCE.md",
            "exists": True,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "total_sections": len(sections),
            "sections": sections,
            "live_overlay": live,
        }

    @app.get("/maturity", response_class=HTMLResponse)
    async def serve_maturity_page():
        path = os.path.join(
            os.path.dirname(__file__), "static", "maturity.html"
        )
        if not os.path.exists(path):
            return HTMLResponse(
                "<h1>Maturity Gates page not available</h1>",
                status_code=404,
            )
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.get("/api/build-history")
    async def api_build_history():
        """Structured parse of docs/BUILD_HISTORY.md.

        Replaces hand-written entries on history.html. Each second-level
        heading becomes an entry with title + date (if present) + body.
        Returns the file mtime so a freshness banner can detect drift
        between the rendered page and the source.
        """
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        path = os.path.join(repo_root, "docs", "BUILD_HISTORY.md")
        if not os.path.exists(path):
            return {
                "source": "docs/BUILD_HISTORY.md",
                "exists": False,
                "entries": [],
            }
        import re as _re

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            st = os.stat(path)
        except Exception as exc:
            return JSONResponse(
                {"error": type(exc).__name__, "detail": str(exc)},
                status_code=500,
            )

        entries: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for line in text.splitlines():
            m = _re.match(r"^##\s+(?P<title>.+?)\s*$", line)
            if m:
                if current is not None:
                    entries.append(current)
                title = m.group("title")
                date_m = _re.match(
                    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<rest>.+)$", title,
                )
                if date_m:
                    current = {
                        "title": date_m.group("rest").strip(),
                        "date": date_m.group("date"),
                        "body": [],
                    }
                else:
                    current = {"title": title, "date": None, "body": []}
                continue
            if current is not None:
                current["body"].append(line)
        if current is not None:
            entries.append(current)

        for e in entries:
            e["body"] = "\n".join(e["body"]).strip()

        return {
            "source": "docs/BUILD_HISTORY.md",
            "exists": True,
            "mtime": st.st_mtime,
            "size": st.st_size,
            "total_entries": len(entries),
            "entries": entries,
        }

    # -- attribution ledger --------------------------------------------------

    @app.get("/api/ledger/recent")
    async def api_ledger_recent(limit: int = 50):
        """Return recent attribution ledger entries."""
        try:
            from consciousness.attribution_ledger import attribution_ledger
            return {"entries": attribution_ledger.get_recent(min(limit, 200))}
        except Exception:
            return {"entries": []}

    @app.get("/api/ledger/stats")
    async def api_ledger_stats():
        """Return attribution ledger statistics with outcome scheduler state."""
        try:
            from consciousness.attribution_ledger import attribution_ledger, outcome_scheduler
            stats = attribution_ledger.get_stats()
            stats["outcome_scheduler"] = outcome_scheduler.get_stats()
            return stats
        except Exception:
            return {}

    @app.get("/api/ledger/chain/{root_id}")
    async def api_ledger_chain(root_id: str):
        """Return all entries in a causal chain."""
        try:
            from consciousness.attribution_ledger import attribution_ledger
            return {"entries": attribution_ledger.get_chain(root_id)}
        except Exception:
            return {"entries": []}

    # -- trace explorer ------------------------------------------------------

    @app.get("/api/trace/explorer")
    async def api_trace_explorer(
        root_limit: int = 20,
        run_limit: int = 20,
        tool_limit: int = 40,
    ):
        """Operator-oriented lineage explorer from immutable ledger data."""
        tx = _cache.get("trace_explorer", {}) or {}
        recon = (_cache.get("reconstructability", {}) or {}).get("trace_explorer", {})
        roots = list(tx.get("root_chains", []) or [])[: max(1, min(root_limit, 100))]
        runs = list(tx.get("agent_runs", []) or [])[: max(1, min(run_limit, 100))]
        tools = list(tx.get("tool_lineage", []) or [])[: max(1, min(tool_limit, 200))]
        return {
            "entry_count": tx.get("entry_count", 0),
            "root_chains": roots,
            "agent_runs": runs,
            "tool_lineage": tools,
            "reconstructability": recon,
        }

    @app.get("/api/reconstructability")
    async def api_reconstructability():
        """Per-surface reconstructability metadata for operator UI."""
        return _cache.get("reconstructability", {})

    @app.get("/api/trace/explorer/chain/{root_id}")
    async def api_trace_explorer_chain(root_id: str, limit: int = 250):
        """Return a causal chain tree with normalized trace fields."""
        try:
            from consciousness.attribution_ledger import attribution_ledger
            chain = attribution_ledger.get_chain(root_id)
        except Exception:
            chain = []

        limited = chain[: max(1, min(limit, 500))]
        nodes: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        children_by_parent: dict[str, list[str]] = {}
        tool_path: list[str] = []
        seen_tools: set[str] = set()

        for entry in limited:
            data = entry.get("data") or {}
            entry_id = str(entry.get("entry_id", "") or "")
            parent_id = str(entry.get("parent_entry_id", "") or "")
            tool = str(data.get("tool", "") or data.get("tool_hint", "") or "")
            if tool and tool not in seen_tools:
                seen_tools.add(tool)
                tool_path.append(tool)

            node = {
                "entry_id": entry_id,
                "parent_entry_id": parent_id,
                "root_entry_id": str(entry.get("root_entry_id", "") or ""),
                "ts": entry.get("ts", 0),
                "subsystem": str(entry.get("subsystem", "") or ""),
                "event_type": str(entry.get("event_type", "") or ""),
                "outcome": str(entry.get("outcome", "pending") or "pending"),
                "conversation_id": str(entry.get("conversation_id", "") or ""),
                "trace_id": str(data.get("trace_id", "") or ""),
                "request_id": str(data.get("request_id", "") or ""),
                "output_id": str(data.get("output_id", "") or ""),
                "intent_id": str(data.get("intent_id", "") or ""),
                "goal_id": str(data.get("goal_id", "") or ""),
                "task_id": str(data.get("task_id", "") or ""),
                "tool": tool,
            }
            nodes.append(node)
            by_id[entry_id] = node
            if parent_id:
                children_by_parent.setdefault(parent_id, []).append(entry_id)

        for node in nodes:
            node["children"] = children_by_parent.get(node["entry_id"], [])

        root_candidates = [
            n["entry_id"] for n in nodes
            if not n.get("parent_entry_id") or n.get("parent_entry_id") not in by_id
        ]

        return {
            "root_entry_id": root_id,
            "node_count": len(nodes),
            "root_nodes": root_candidates,
            "tool_path": tool_path,
            "nodes": nodes,
        }

    # -- explainability (Phase 6.4) ------------------------------------------

    @app.get("/api/explainability/trace/{conversation_id}")
    async def api_explainability_trace(conversation_id: str):
        """Return provenance trace + evidence chain for a conversation."""
        try:
            from reasoning.explainability import build_evidence_chain
            chain = build_evidence_chain(conversation_id)
            return chain
        except Exception:
            return {"available": False, "reason": "Explainability module error"}

    @app.get("/api/explainability/recent")
    async def api_explainability_recent(limit: int = 10):
        """Return provenance traces for recent conversations from ledger."""
        try:
            from consciousness.attribution_ledger import attribution_ledger
            recent = attribution_ledger.query(
                subsystem="conversation",
                event_type="response_complete",
                limit=min(limit, 50),
            )
            traces = []
            for entry in recent:
                prov = (entry.get("data") or {}).get("provenance", {})
                if prov:
                    traces.append({
                        "entry_id": entry.get("entry_id", ""),
                        "conversation_id": entry.get("conversation_id", ""),
                        "timestamp": entry.get("ts", 0),
                        "tool": (entry.get("data") or {}).get("tool", ""),
                        "outcome": entry.get("outcome", "pending"),
                        "provenance": prov,
                    })
            return {"traces": traces}
        except Exception:
            return {"traces": []}

    # -- log viewer ----------------------------------------------------------

    @app.get("/api/logs")
    async def api_logs(lines: int = 50):
        """Return recent brain log lines for the dashboard log viewer."""
        import os
        log_path = "/tmp/jarvis-brain.log"
        if not os.path.isfile(log_path):
            return {"lines": []}
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_size = min(size, lines * 200)
                f.seek(max(0, size - read_size))
                raw = f.read().decode("utf-8", errors="replace")
            all_lines = raw.strip().split("\n")
            return {"lines": all_lines[-lines:]}
        except Exception:
            return {"lines": []}

    # -- WebSocket -----------------------------------------------------------

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.append(websocket)
        try:
            if _cache:
                await websocket.send_text(
                    json.dumps({"type": "snapshot", **_cache}, default=str))
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)

    return app


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------

async def _snapshot_loop() -> None:
    """Rebuild the dashboard cache on a fixed timer. This is the ONLY place
    that reads from the consciousness system for dashboard purposes."""
    while True:
        try:
            _build_cache()
        except Exception:
            logger.exception("Dashboard snapshot build failed")
        await asyncio.sleep(CACHE_INTERVAL_S)


async def _ws_push_loop() -> None:
    """Push cache to WebSocket clients only when the hash changed."""
    last_hash = ""
    while True:
        if _cache and _cache_hash != last_hash and _ws_clients:
            last_hash = _cache_hash
            msg = json.dumps({"type": "snapshot", **_cache}, default=str)
            for ws in list(_ws_clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    pass
        await asyncio.sleep(WS_PUSH_INTERVAL_S)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def create_dashboard(
    engine: ConsciousnessEngine,
    response_gen: ResponseGenerator,
    perception: PerceptionServer | None,
    host: str = "0.0.0.0",
    port: int = 9200,
    pi_video_url: str = "",
    attention: Any = None,
    persistence: Any = None,
    episodes: Any = None,
    processors: dict | None = None,
    perc_orch: Any = None,
) -> Any:
    global _engine, _response_gen, _perception, _pi_video_url, _attention_core
    global _persistence, _episodes, _processors, _perc_orch
    _engine = engine
    _response_gen = response_gen
    _perception = perception
    _pi_video_url = pi_video_url
    _attention_core = attention
    _persistence = persistence
    _episodes = episodes
    _processors = processors
    _perc_orch = perc_orch

    _health.start()

    app = _create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    loop = asyncio.get_event_loop()
    loop.create_task(server.serve())
    loop.create_task(_snapshot_loop())
    loop.create_task(_ws_push_loop())

    logger.info("Dashboard: http://%s:%d", host, port)
    logger.info("Dashboard API key: %s", _DASHBOARD_API_KEY)
    return server
