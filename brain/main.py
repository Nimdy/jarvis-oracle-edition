"""Jarvis Brain — main entry point.

Starts the consciousness engine, perception server, optional dashboard,
and provides a CLI for direct interaction.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from typing import Any

from config import BrainConfig
from consciousness.engine import ConsciousnessEngine
from consciousness.events import (
    event_bus, KERNEL_PHASE_CHANGE, TONE_SHIFT, KERNEL_THOUGHT,
    PERCEPTION_USER_PRESENT, PERCEPTION_WAKE_WORD,
)
from reasoning.response import ResponseGenerator
from reasoning.claude_client import ClaudeClient
from reasoning.tool_router import tool_router, ToolType
from tools.time_tool import get_current_time
from tools.system_tool import get_system_status
from tools.memory_tool import search_memory, get_memory_summary
from tools.vision_tool import describe_scene
from tools.introspection_tool import get_introspection
from perception.server import PerceptionServer
from consciousness.reflection import reflection_engine
from consciousness.modes import mode_manager
from memory.persistence import (
    MemoryPersistence,
    consciousness_persistence,
    load_intention_registry,
    save_intention_registry,
)
from memory.search import init_vector_store, index_memory
from memory.episodes import EpisodicMemory
from perception_orchestrator import PerceptionOrchestrator

import os
os.environ.setdefault("TQDM_DISABLE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger("jarvis.brain")

_RESTART_EXIT_CODE = 10
_INTENT_FILE = os.path.join(os.path.expanduser("~"), ".jarvis", "restart_intent.json")


def _request_restart(reason: str, message: str = "",
                     delay_s: float = 0, nonce: str = "") -> None:
    """Write an atomic restart intent file for the supervisor to read."""
    import json as _json
    import tempfile as _tmpfile
    intent = {
        "reason": reason,
        "requested_at": time.time(),
        "requested_by": "main.py",
        "delay_s": delay_s,
        "message": message,
        "nonce": nonce or f"{reason}-{time.time():.0f}",
    }
    intent_dir = os.path.dirname(_INTENT_FILE)
    os.makedirs(intent_dir, exist_ok=True)
    fd, tmp_path = _tmpfile.mkstemp(dir=intent_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            _json.dump(intent, f)
        os.replace(tmp_path, _INTENT_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info("Restart intent written: reason=%s message=%s", reason, message)


def _restore_active_policy_controller(
    policy_registry: Any,
    state_encoder: Any,
) -> Any | None:
    """Restore the active persisted policy model, if one exists.

    This restores trained weights across restart without inferring unstored
    promotion-phase details such as staged feature flags.
    """
    try:
        active = policy_registry.get_active()
    except Exception:
        logger.exception("Failed to query active policy model from registry")
        return None

    if active is None or not getattr(active, "path", ""):
        return None
    if not os.path.exists(active.path):
        logger.warning("Active policy model path missing on boot: %s", active.path)
        return None

    try:
        from policy.state_encoder import STATE_DIM
        from policy.policy_nn import PolicyNNController

        controller = PolicyNNController(arch=active.arch, input_dim=STATE_DIM)
        controller.set_encoder(state_encoder)
        if controller.load(active.path):
            logger.info(
                "Restored active policy model on boot: v%04d (%s)",
                active.version,
                active.arch,
            )
            return controller
    except Exception:
        logger.exception("Failed to restore active policy model from %s", active.path)
    return None


async def main() -> None:
    config = BrainConfig()

    from hardware_profile import get_hardware_profile
    hw = get_hardware_profile()

    print()
    print("  ╔══════════════════════════════════╗")
    print("  ║      JARVIS CONSCIOUSNESS        ║")
    print("  ║         Brain v1.0.0 (Python)     ║")
    print("  ╚══════════════════════════════════╝")
    print()
    m = hw.models
    print(f"  GPU: {hw.gpu.name} ({hw.gpu.vram_mb}MB VRAM) | Tier: {hw.tier} ({hw.tier_source})")
    print(f"  CPU: {hw.cpu.model} ({hw.cpu.cores}c/{hw.cpu.threads}t, {hw.cpu.ram_gb}GB RAM) | Tier: {hw.cpu_tier} ({hw.cpu_tier_source})")
    print(f"  LLM: {config.ollama.model} | Fast: {config.ollama.fast_model} | Vision: {config.ollama.vision_model or 'disabled'}")
    print(f"  STT: {config.stt.model} ({config.stt.compute_type}) on {config.stt.device}")
    print(f"  Keep-alive: {config.ollama.keep_alive} | Warmup all: {config.ollama.warmup_all}")
    print(f"  Workload: Emotion={m.emotion_device} SpeakerID={m.speaker_id_device} Embed={m.embedding_device} Hemisphere={m.hemisphere_device}")
    print()

    # -- Core ---------------------------------------------------------------

    engine = ConsciousnessEngine()
    response_gen = ResponseGenerator(engine, {
        "host": config.ollama.host,
        "model": config.ollama.model,
        "fast_model": config.ollama.fast_model,
        "vision_model": config.ollama.vision_model,
        "temperature": config.ollama.temperature,
        "max_tokens": config.ollama.max_tokens,
        "keep_alive": config.ollama.keep_alive,
    })

    claude: ClaudeClient | None = None
    if config.enable_claude and config.claude.api_key:
        claude = ClaudeClient(
            api_key=config.claude.api_key,
            model=config.claude.model,
            max_tokens=config.claude.max_tokens,
        )
        print(f"  Claude API: {'connected' if claude.available else 'unavailable'}")
    elif config.claude.api_key and not config.enable_claude:
        print("  Claude API: key present but ENABLE_CLAUDE is not set")

    ollama_ready = await response_gen.is_ready()
    if not ollama_ready:
        print(f"  WARNING: Ollama not available at {config.ollama.host}")
        print("  Make sure Ollama is running: ollama serve")
    else:
        models = await response_gen.get_available_models()
        print(f"  Ollama connected. Models: {', '.join(models[:5])}")
        if config.ollama.warmup_all:
            print("  Loading LLM models into VRAM (always-online mode)...")
            warmup_models = list(dict.fromkeys(
                m for m in [config.ollama.model, config.ollama.fast_model]
                if m and m in models
            ))
            await response_gen.ollama.warmup_all(warmup_models)
            print(f"  {len(warmup_models)} LLM model(s) loaded and pinned in VRAM")
            if config.ollama.vision_model and config.ollama.vision_model in models:
                print(f"  Vision model ({config.ollama.vision_model}) available on-demand")
        else:
            print("  Warming up primary model into VRAM...")
            await response_gen.ollama.warmup(config.ollama.fast_model or config.ollama.model)

    # -- Memory persistence -------------------------------------------------

    persistence = MemoryPersistence(
        path=config.memory.persist_path,
        interval_s=config.memory.persist_interval_s,
    )
    loaded = persistence.load()
    if loaded:
        print(f"  Restored {loaded} memories from disk")

    # -- Semantic memory (vector store) ----------------------------------------

    vector_store = init_vector_store(
        db_path=config.memory.vector_db_path,
        model=config.memory.embedding_model,
        dim=config.memory.embedding_dim,
        device=config.memory.embedding_device,
    )
    if vector_store and vector_store.available:
        from memory.storage import memory_storage
        existing = memory_storage.get_all()
        if existing:
            indexed = vector_store.rebuild_from_memories(existing)
            print(f"  Semantic memory: {indexed} vectors indexed")
        else:
            print("  Semantic memory: ready (empty)")
    else:
        print("  Semantic memory: disabled (missing sqlite-vec or sentence-transformers)")

    # -- Memory Cortex (ranker + salience NNs) ---------------------------------

    try:
        from memory.ranker import init_memory_ranker
        from memory.salience import init_salience_model
        from memory.retrieval_log import memory_retrieval_log
        from memory.lifecycle_log import memory_lifecycle_log

        memory_retrieval_log.init()
        memory_lifecycle_log.init()

        if os.environ.get("CORTEX_REHYDRATE_ON_BOOT", "true").lower() in ("1", "true", "yes"):
            r_count = memory_retrieval_log.rehydrate(max_events=200)
            l_count = memory_lifecycle_log.rehydrate(max_creations=200)
            if r_count or l_count:
                print(f"  Cortex warm-start: {r_count} retrieval events, {l_count} lifecycle records rehydrated")

        ranker = init_memory_ranker()
        salience = init_salience_model()

        cortex_parts = []
        if ranker and ranker.is_ready():
            cortex_parts.append(f"ranker (train#{ranker._train_count}, loss={ranker._last_loss:.4f})")
        else:
            cortex_parts.append("ranker (untrained)")
        if salience and salience.is_ready():
            cortex_parts.append(f"salience (train#{salience._train_count}, blend={salience._model_blend:.0%})")
        else:
            cortex_parts.append("salience (untrained)")
        print(f"  Memory cortex: {', '.join(cortex_parts)}")
    except Exception as exc:
        print(f"  Memory cortex: init failed ({exc})")

    # -- Episodic memory -------------------------------------------------------

    episodes = EpisodicMemory(
        persist_path=config.episodic.persist_path,
        max_episodes=config.episodic.max_episodes,
    )
    ep_stats = episodes.get_stats()
    if ep_stats["total_episodes"] > 0:
        print(f"  Episodic memory: {ep_stats['total_episodes']} episodes loaded")
        if vector_store and vector_store.available:
            ep_indexed = 0
            for ep in episodes.get_recent_episodes(count=200):
                if ep.summary:
                    EpisodicMemory._index_episode(ep)
                    ep_indexed += 1
            if ep_indexed:
                print(f"  Episodic vectors: {ep_indexed} episode summaries indexed")
    else:
        print("  Episodic memory: ready")

    # -- Extended persistence (clusters, causal models, personality snapshots) --

    from memory.persistence import extended_persistence
    ep_result = extended_persistence.load_all()
    if ep_result.get("clusters"):
        print("  Memory clusters: restored")
    else:
        print("  Memory clusters: will build during dream cycles")

    # -- Consciousness persistence (restore prior identity) -------------------

    consciousness_restored = False
    if consciousness_persistence.restore_to_system(engine.consciousness):
        engine.sync_config_after_restore()
        cs = engine.consciousness.get_state()
        print(f"  Consciousness restored: stage={cs.stage}, transcendence={cs.transcendence_level:.1f}")
        print(f"    observer: awareness={engine.consciousness.observer.awareness_level:.2f}, "
              f"observations={engine.consciousness.observer.state.observation_count}")
        print(f"    governor: mutations={engine.consciousness.governor.mutation_count}, "
              f"rollbacks={engine.consciousness.governor.rollback_count}")
        print(f"    capabilities: {engine.consciousness.driven_evolution.get_active_capabilities()}")
        consciousness_restored = True
    else:
        print("  Consciousness: fresh start")
    engine._restore_complete = True

    try:
        _open_intents = load_intention_registry()
        if _open_intents:
            print(f"  Intention registry: {_open_intents} open intention(s) restored")
        else:
            print("  Intention registry: clean (no open intentions)")
    except Exception:
        logger.exception("Intention registry load failed (continuing)")

    # Boot continuity banner
    prov = consciousness_persistence.get_boot_provenance()
    print(f"  Boot: instance={prov['instance_id'][:8]}.. boot={prov['boot_id'][:8]}..")
    if prov.get("file_saved_at"):
        import datetime as _dt
        saved_str = _dt.datetime.fromtimestamp(prov["file_saved_at"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"    State file: {prov.get('size_bytes', 0)} bytes, saved {saved_str} by boot={prov.get('file_boot_id', '?')}")

    response_gen.set_episodes(episodes)

    # -- Pending verification check (crash-loop protection) --------------------
    # TIMING: this runs AFTER consciousness restore but BEFORE engine.start()
    # so that boot_count is bumped before any risky subsystem initialisation.
    # If a patched module crashes on *import* (before main() is reached),
    # the supervisor detects the rapid crash and rolls back.

    _verification_pending = None
    try:
        from self_improve.verification import (
            read_pending, increment_boot_count, clear_pending,
        )
        from self_improve.orchestrator import SelfImprovementOrchestrator

        _pv = read_pending()
        if _pv:
            if _pv.boot_count >= _pv.max_retries:
                logger.warning(
                    "Patch %s failed %d boot(s) — auto-rolling back from %s",
                    _pv.patch_id, _pv.boot_count, _pv.snapshot_path,
                )
                ok = False
                try:
                    ok = SelfImprovementOrchestrator.restore_snapshot_static(_pv.snapshot_path)
                except Exception as restore_exc:
                    logger.error("Snapshot restore threw: %s", restore_exc)
                clear_pending()
                if ok:
                    print(f"  Verification: ROLLBACK — patch {_pv.patch_id} failed after {_pv.boot_count} boot(s)")
                else:
                    logger.error(
                        "Snapshot restore FAILED for %s — cleared pending to break loop, "
                        "self-improvement will be disabled for safety",
                        _pv.patch_id,
                    )
                    print(f"  Verification: ROLLBACK FAILED — patch {_pv.patch_id}, entering safe mode")
                _request_restart("post_rollback_clean_start",
                                 f"Rollback of patch {_pv.patch_id} — clean restart")
                sys.exit(10)
            else:
                new_count = increment_boot_count()
                _verification_pending = _pv
                print(f"  Verification: ACTIVE — patch {_pv.patch_id} (boot {new_count}/{_pv.max_retries})")
        else:
            logger.debug("No pending verification — normal boot")
    except Exception as exc:
        logger.warning("Verification check failed (continuing normally): %s", exc)

    engine.start()
    persistence.start_auto_save()
    consciousness_persistence.start_auto_save(engine.consciousness, engine, interval_s=60.0)

    # -- Gestation detection ---------------------------------------------------

    gestation_manager = None
    _gestation_resuming = False
    if config.gestation.enabled and config.autonomy.enabled:
        from consciousness.gestation import GestationManager, is_fresh_brain, needs_gestation_resume
        if is_fresh_brain(loaded, consciousness_restored):
            gestation_manager = GestationManager(config.gestation)
            engine.enable_gestation(gestation_manager)
            print("  Gestation: ACTIVE — fresh brain entering birth protocol")
        elif needs_gestation_resume():
            gestation_manager = GestationManager(config.gestation)
            engine.enable_gestation(gestation_manager)
            _gestation_resuming = True
            print("  Gestation: RESUMING — interrupted gestation detected")
        else:
            print("  Gestation: skipped (brain already has identity)")
    elif not config.gestation.enabled:
        print("  Gestation: disabled (ENABLE_GESTATION=false)")
    else:
        print("  Gestation: requires autonomy system")

    reflection_engine.start()
    print("  Self-reflection: active")

    # -- Self-improvement loop ------------------------------------------------

    async def _self_restart() -> None:
        """Core restart primitive — independent of dashboard web layer.

        Performs system quiescence then signals the supervisor to relaunch.
        """
        logger.info("Self-restart triggered by improvement pipeline")
        try:
            consciousness_persistence.save_from_system(engine.consciousness, engine=engine)
            persistence.save()
            if episodes:
                episodes.save()
            try:
                from reasoning.context import context_builder
                context_builder.save()
            except Exception:
                pass
            try:
                from memory.search import get_vector_store
                vs = get_vector_store()
                if vs:
                    vs.close()
            except Exception:
                pass
            engine.stop()
        except Exception as exc:
            logger.error("Error during pre-restart save: %s", exc)

        for handler in logging.root.handlers:
            try:
                handler.flush()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()

        _request_restart("self_improvement_verify", "Self-improvement pipeline restart")
        sys.exit(10)

    # -- Shared CodeGen infrastructure (used by self-improve + acquisition) ----

    _coder = None
    _codegen_service = None
    if config.coder.enabled:
        from codegen.coder_server import CoderServer
        from codegen.service import CodeGenService
        _coder = CoderServer(
            model_path=config.coder.model_path,
            server_port=config.coder.server_port,
            ctx_size=config.coder.ctx_size,
            gpu_layers=config.coder.gpu_layers,
            llama_server_bin=config.coder.llama_server_bin,
            max_tokens=config.coder.max_tokens,
            temperature=config.coder.temperature,
        )
        _codegen_service = CodeGenService()
        _codegen_service.set_coder_server(_coder)
        logger.info("CoderServer wired (model=%s, bin=%s)",
                    config.coder.model_path, config.coder.llama_server_bin)
        print(f"  CodeGen: ready (model={config.coder.model_path})")
    else:
        print("  CodeGen: disabled (coder not configured)")
    try:
        engine._coder_server = _coder
        engine._codegen_service = _codegen_service
    except Exception:
        logger.debug("Failed to attach shared CodeGen handles to engine", exc_info=True)

    # -- Self-improvement pipeline ---------------------------------------------

    self_improve = None
    if config.enable_self_improve:
        from self_improve.orchestrator import SelfImprovementOrchestrator
        from self_improve.provider import PatchProvider
        _si_provider = PatchProvider()
        if _coder is not None:
            _si_provider.set_coder_server(_coder)
        self_improve = SelfImprovementOrchestrator(
            engine=engine,
            restart_callback=_self_restart,
            dry_run_mode=config.self_improve_dry_run,
            provider=_si_provider,
        )
        self_improve.set_ollama_client(response_gen.ollama)
        if gestation_manager is not None:
            self_improve.set_paused(True)
            print("  Self-improvement: paused (gestation active)")
        else:
            engine.consciousness.enable_self_improvement(self_improve)
            engine._self_improve_orchestrator = self_improve
            dry_label = " [DRY-RUN]" if config.self_improve_dry_run else ""
            print(f"  Self-improvement: active (restart-verify enabled){dry_label}")
    else:
        print("  Self-improvement: disabled (set ENABLE_SELF_IMPROVE=true)")

    # -- Hemisphere NN system --------------------------------------------------

    try:
        from hemisphere.orchestrator import HemisphereOrchestrator
        hemisphere_orch = HemisphereOrchestrator(device=config.hemisphere.device)
        restored = hemisphere_orch.restore_models()
        engine.enable_hemisphere(hemisphere_orch)
        if restored:
            print(f"  Hemisphere NNs: active ({restored} model(s) restored)")
        else:
            print("  Hemisphere NNs: active (will design networks as data accumulates)")

        # Wire the voice-intent shadow runner (P1.4) now that the
        # hemisphere orchestrator is live. Stays at SHADOW level on a
        # fresh brain; cannot promote until gates pass.
        try:
            from reasoning.intent_shadow import (
                IntentShadowRunner,
                make_hemisphere_inference_fn,
                make_teacher_sample_provider,
                set_intent_shadow_runner,
            )

            set_intent_shadow_runner(
                IntentShadowRunner(
                    inference_fn=make_hemisphere_inference_fn(hemisphere_orch),
                    teacher_sample_provider=make_teacher_sample_provider(),
                )
            )
            print("  Voice-Intent Shadow Runner: wired (SHADOW level)")
        except Exception as shadow_exc:
            logger.warning("Voice-intent shadow runner wire failed: %s", shadow_exc)
    except Exception as exc:
        logger.warning("Hemisphere system init failed: %s", exc)
        print("  Hemisphere NNs: disabled")

    # -- Autonomy system -------------------------------------------------------

    if config.autonomy.enabled:
        try:
            from autonomy import AutonomyOrchestrator
            autonomy_orch = AutonomyOrchestrator(autonomy_level=config.autonomy.level)
            engine.enable_autonomy(autonomy_orch)
            # Phase 6.5: durable audit subscriber. Must be wired before
            # reconcile_on_boot() because reconcile can emit
            # AUTONOMY_LEVEL_CHANGED for restored state.
            try:
                from autonomy.audit_ledger import get_audit_ledger
                get_audit_ledger().wire()
            except Exception as audit_exc:
                logger.warning("Autonomy audit ledger wire failed: %s", audit_exc)
            reconcile_report = autonomy_orch.reconcile_on_boot()
            lvl = autonomy_orch.autonomy_level
            level_names = {0: "propose", 1: "research", 2: "safe-apply", 3: "full"}
            restored_tag = " (auto-restored)" if reconcile_report.get("auto_restored") else ""
            print(f"  Autonomy: L{lvl} ({level_names.get(lvl, '?')}){restored_tag} — metric triggers + opportunity scoring")
            if reconcile_report.get("vetoed_by"):
                print(f"  Autonomy reconcile: restore vetoed by {reconcile_report['vetoed_by']}")
            if reconcile_report.get("l3_eligible_but_manual"):
                print("  Autonomy reconcile: L3 eligible but manual-only (not auto-restored)")
            for d in reconcile_report.get("disagreements", []):
                print(f"  Autonomy reconcile: {d}")
            if self_improve:
                autonomy_orch._self_improve_orchestrator = self_improve
        except Exception as exc:
            logger.warning("Autonomy system init failed: %s", exc)
            print("  Autonomy: disabled")
    else:
        print("  Autonomy: disabled (set ENABLE_AUTONOMY=true)")

    # -- Goal Continuity Layer (Phase 1A — observational only) -----------------

    try:
        from goals.goal_manager import GoalManager, get_goal_manager
        goal_mgr = get_goal_manager()
        engine.enable_goals(goal_mgr)

        reconciled = goal_mgr.reconcile_on_boot()
        if reconciled:
            print(f"  Goals: reconciled {reconciled} orphaned running task(s)")

        if config.autonomy.enabled and hasattr(engine, "_autonomy_orchestrator") and engine._autonomy_orchestrator:
            engine._autonomy_orchestrator.set_goal_callback(goal_mgr.record_task_outcome)
            goal_mgr.set_autonomy_orchestrator(engine._autonomy_orchestrator)
            engine._autonomy_orchestrator.set_goal_manager(goal_mgr)
            print("  Goals: Phase 2 (dispatch + alignment)")
        else:
            print("  Goals: Phase 2 (no autonomy — preview only)")
    except Exception as exc:
        logger.warning("Goal Continuity Layer init failed: %s", exc)
        print("  Goals: disabled")

    # -- Skill Registry + Learning Jobs ----------------------------------------

    _learning_job_orch = None
    try:
        from skills.registry import skill_registry
        from skills.capability_gate import capability_gate
        from skills.learning_jobs import LearningJobStore, LearningJobOrchestrator
        skill_registry.load()
        capability_gate.set_registry(skill_registry)
        _job_store = LearningJobStore()
        _learning_job_orch = LearningJobOrchestrator(store=_job_store, registry=skill_registry)
        capability_gate.set_orchestrator(_learning_job_orch)
        engine.enable_learning_jobs(_learning_job_orch)

        try:
            from tools.skill_tool import set_orchestrator as _set_skill_orch
            from tools.skill_tool import set_registry as _set_skill_reg
            _set_skill_orch(_learning_job_orch)
            _set_skill_reg(skill_registry)
        except Exception:
            logger.warning("Failed to wire skill_tool singletons")

        _n_skills = len(skill_registry.get_all())
        _n_active = len(_learning_job_orch.get_active_jobs())
        print(f"  Skills: registry loaded ({_n_skills} skills, {_n_active} active jobs)")
    except Exception as exc:
        logger.warning("Skill system init failed: %s", exc)
        print("  Skills: disabled")

    # -- Capability Acquisition Pipeline ----------------------------------------

    _acq_orch = None
    try:
        from acquisition.orchestrator import AcquisitionOrchestrator
        from acquisition.job import AcquisitionStore
        _acq_store = AcquisitionStore()
        _acq_orch = AcquisitionOrchestrator(store=_acq_store)
        if _codegen_service is not None:
            _acq_orch.set_codegen_service(_codegen_service)
        engine.enable_acquisition(_acq_orch)
        _acq_status = _acq_orch.get_status()
        _coder_label = " + CodeGen" if _codegen_service is not None else " (no coder)"
        print(f"  Acquisition: enabled ({_acq_status['active_count']} active, {_acq_status['total_count']} total){_coder_label}")
    except Exception as exc:
        logger.warning("Capability Acquisition Pipeline init failed: %s", exc)
        print("  Acquisition: disabled")

    # -- Start gestation if detected -------------------------------------------

    if gestation_manager is not None:
        gestation_manager.start(resume=_gestation_resuming)
        mode_manager.set_mode("gestation", reason="fresh_brain_detected", force=True)
        print("  Mode: gestation (self-discovery in progress)")

    # -- Codebase self-knowledge index ----------------------------------------

    try:
        from tools.codebase_tool import codebase_index
        codebase_index.build()
        stats = codebase_index.get_stats()
        print(f"  Codebase index: {stats['total_modules']} modules, {stats['total_symbols']} symbols, {stats['total_lines']} lines")

        modified = codebase_index.get_modified_files()
        if modified:
            logger.info("Code changes detected since last index: %s", modified[:20])
            print(f"  Code changes detected: {len(modified)} file(s) modified")
            try:
                from library.ingest import ingest_codebase_source
                from library.self_study_filter import allow_library_reingest
                project_root = codebase_index._root.parent
                reingest_count = 0
                for rel_path in modified[:50]:
                    if rel_path.endswith(".md"):
                        fpath = project_root / rel_path
                        repo_rel = rel_path
                    else:
                        fpath = codebase_index._root / rel_path
                        repo_rel = f"brain/{rel_path}"
                    if not fpath.exists():
                        continue
                    if not allow_library_reingest(project_root, repo_rel):
                        logger.debug("Skipping non-runtime/self-study file for Library re-ingest: %s", repo_rel)
                        continue
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="replace")
                        result = ingest_codebase_source(
                            file_path=repo_rel,
                            content=content,
                            title=f"{rel_path} (updated)",
                        )
                        if result.success and result.error != "unchanged":
                            reingest_count += 1
                    except Exception:
                        pass
                if reingest_count:
                    logger.info("Re-ingested %d changed file(s) into Library", reingest_count)
                    print(f"  Library re-ingest: {reingest_count} file(s) updated")
            except ImportError:
                pass
    except Exception as exc:
        logger.warning("Codebase index build failed: %s", exc)
        print("  Codebase index: disabled")

    # -- Wire LLM enrichment for consciousness subsystems --------------------

    if ollama_ready and config.ollama.fast_model:
        async def _consciousness_llm(prompt: str) -> str:
            result = await response_gen.ollama.generate(
                model=config.ollama.fast_model,
                prompt=prompt,
                options={"num_predict": 150, "temperature": 0.7},
            )
            return result.get("response", "") if isinstance(result, dict) else str(result)

        engine.consciousness.set_llm_callback(_consciousness_llm)
        print("  Consciousness LLM enrichment: active (using fast model)")

        async def _study_llm(prompt: str) -> str:
            from ollama import AsyncClient, ResponseError
            client = AsyncClient(host=config.ollama.host)
            resp = await client.chat(
                model=config.ollama.fast_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 600},
                keep_alive=config.ollama.keep_alive,
                think=False,
            )
            return resp.message.content

        if config.research.llm_study:
            try:
                from library.study import set_llm_callback as set_study_llm
                set_study_llm(_study_llm)
                print("  Study pipeline LLM extraction: active (using fast model)")
            except Exception as exc:
                print(f"  Study pipeline LLM extraction: failed ({exc})")
        else:
            print("  Study pipeline LLM extraction: disabled (config.research.llm_study=False)")

    # -- Policy promotion pipeline --------------------------------------------

    promotion = None
    try:
        from policy.experience_buffer import ExperienceBuffer
        from policy.evaluator import PolicyEvaluator
        from policy.registry import ModelRegistry
        from policy.governor import PolicyGovernor
        from policy.policy_interface import PolicyInterface
        from policy.promotion import PromotionPipeline

        policy_buffer = ExperienceBuffer()
        loaded_exp = policy_buffer.load()
        policy_evaluator = PolicyEvaluator()
        policy_registry = ModelRegistry()
        policy_governor = PolicyGovernor()
        policy_interface = PolicyInterface()

        promotion = PromotionPipeline(
            buffer=policy_buffer,
            evaluator=policy_evaluator,
            registry=policy_registry,
            governor=policy_governor,
            interface=policy_interface,
        )

        from policy.state_encoder import StateEncoder, STATE_DIM
        from policy.policy_nn import PolicyNNController, TORCH_AVAILABLE
        state_encoder = StateEncoder()
        engine.set_experience_buffer(policy_buffer, state_encoder)
        engine.set_policy_layer(policy_interface, policy_evaluator)

        if TORCH_AVAILABLE:
            restored_nn = _restore_active_policy_controller(policy_registry, state_encoder)
            if restored_nn is not None:
                policy_interface.set_nn_controller(restored_nn)
            else:
                default_nn = PolicyNNController(arch="mlp2", input_dim=STATE_DIM)
                default_nn.set_encoder(state_encoder)
                policy_interface.set_nn_controller(default_nn)
            policy_interface.set_governor(policy_governor)
            policy_interface.enable()
            policy_evaluator.set_mode("shadow")
            print(f"  Policy pipeline: active (shadow mode, {loaded_exp} experiences)")
        else:
            print(f"  Policy pipeline: ready, no PyTorch ({loaded_exp} experiences loaded)")
    except Exception as exc:
        logger.warning("Policy pipeline init failed: %s", exc)
        print("  Policy pipeline: disabled")

    if promotion is not None:
        engine._promotion_pipeline = promotion
    try:
        if promotion is not None and hemisphere_orch is not None:
            promotion.set_hemisphere_orchestrator(hemisphere_orch)
    except NameError:
        pass

    # -- Perception Orchestrator -----------------------------------------------

    perc_orch = PerceptionOrchestrator(
        engine=engine,
        response_gen=response_gen,
        claude=claude,
        config=config,
        episodes=episodes,
    )
    await perc_orch.start()
    engine._perception_orchestrator = perc_orch

    perception = perc_orch.perception
    attention = perc_orch.attention

    if _learning_job_orch and hasattr(perc_orch, 'brain_tts') and perc_orch.brain_tts:
        _learning_job_orch.set_context_provider("tts_engine", perc_orch.brain_tts)
        _learning_job_orch.set_context_provider(
            "audio_output_available", lambda: getattr(perc_orch.brain_tts, 'available', False),
        )

    if _learning_job_orch:
        if hemisphere_orch:
            _learning_job_orch.set_context_provider("hemisphere_orchestrator", hemisphere_orch)
            _learning_job_orch.set_context_provider(
                "distillation_stats", lambda: hemisphere_orch.get_distillation_stats(),
            )
        if perc_orch.presence:
            _learning_job_orch.set_context_provider(
                "user_present", lambda: perc_orch.presence.get_state().get("is_present", False),
            )
        if perc_orch.speaker_id:
            _learning_job_orch.set_context_provider("speaker_id", perc_orch.speaker_id)
            logger.info("Learning job context: speaker_id wired (available=%s, profiles=%d)",
                        getattr(perc_orch.speaker_id, 'available', '?'),
                        len(getattr(perc_orch.speaker_id, '_profiles', {})))
        if perc_orch.emotion_classifier:
            _learning_job_orch.set_context_provider("emotion_classifier", perc_orch.emotion_classifier)
            logger.info("Learning job context: emotion_classifier wired (gpu=%s, healthy=%s)",
                        getattr(perc_orch.emotion_classifier, '_gpu_available', '?'),
                        getattr(perc_orch.emotion_classifier, '_model_healthy', '?'))
        if hasattr(perc_orch, 'identity_fusion') and perc_orch.identity_fusion:
            _learning_job_orch.set_context_provider("identity_fusion", perc_orch.identity_fusion)
        if _acq_orch:
            _learning_job_orch.set_context_provider("acquisition_orchestrator", _acq_orch)
            try:
                from skills.operational_bridge import build_skill_execution_callables
                _learning_job_orch.set_context_provider(
                    "skill_execution_callables",
                    lambda: build_skill_execution_callables(_acq_orch),
                )
                logger.info("Learning job context: acquisition proof bridge wired")
            except Exception:
                logger.debug("Failed to wire acquisition proof bridge", exc_info=True)

    # Activate gestation gating on perception layer
    if gestation_manager is not None and gestation_manager.is_active:
        perc_orch.set_gestation_active(True)
        # MODE_CHANGE fired before perc_orch existed — replay the mode profile
        perc_orch._on_mode_change(to_mode="gestation")

    # -- Layer 3B Scene Continuity (shadow) ------------------------------------
    try:
        if hasattr(perc_orch, "_scene_tracker") and perc_orch._scene_tracker:
            engine.enable_scene_continuity(perc_orch._scene_tracker)
            print("  Scene Continuity: Layer 3B (shadow)")
        else:
            print("  Scene Continuity: disabled (no scene tracker)")
    except Exception as exc:
        logger.warning("Scene Continuity init failed: %s", exc)
        print("  Scene Continuity: disabled")

    # -- Unified World Model (Phase 1 — shadow) -----------------------------
    try:
        from cognition.world_model import WorldModel as _WorldModel
        from consciousness.health_monitor import health_monitor as _health_mon
        _scene_tr = getattr(perc_orch, "_scene_tracker", None)
        _presence_tr = getattr(perc_orch, "presence", None)
        _analytics = engine.consciousness.analytics if engine.consciousness else None
        _goal_mgr_ref = None
        try:
            from goals.goal_manager import get_goal_manager
            _goal_mgr_ref = get_goal_manager()
        except Exception:
            pass
        _world_model = _WorldModel(
            scene_tracker=_scene_tr,
            attention=attention,
            presence=_presence_tr,
            mode_manager=mode_manager,
            goal_manager=_goal_mgr_ref,
            episodes=episodes,
            health_monitor=_health_mon,
            analytics=_analytics,
            perc_orch=perc_orch,
        )
        engine.enable_world_model(_world_model)
        print("  World Model: Phase 1 (shadow mode)")
    except Exception as exc:
        logger.warning("World Model init failed: %s", exc)
        print("  World Model: disabled")

    # -- Dashboard ----------------------------------------------------------

    dashboard_runner = None
    if config.enable_dashboard:
        from dashboard.app import create_dashboard
        pi_video_url = ""
        if config.perception.pi_host:
            pi_video_url = f"http://{config.perception.pi_host}:{config.perception.pi_ui_port}/video"
        dashboard_runner = await create_dashboard(
            engine, response_gen, perception,
            host=config.dashboard.host, port=config.dashboard.port,
            pi_video_url=pi_video_url,
            attention=attention,
            persistence=persistence,
            episodes=episodes,
            processors=perc_orch.get_processors(),
            perc_orch=perc_orch,
        )

    # -- Eval sidecar (read-only observer, always-on in shadow mode) --------

    try:
        from jarvis_eval import get_eval_sidecar
        _eval_sidecar = get_eval_sidecar()
        engine.enable_eval_sidecar(_eval_sidecar)
    except Exception:
        logger.warning("Eval sidecar failed to initialize", exc_info=True)

    # -- Event logging ------------------------------------------------------

    event_bus.on(KERNEL_PHASE_CHANGE, lambda from_phase, to_phase, **_: _log_and_broadcast(
        f"  [Phase: {from_phase} → {to_phase}]", perception, engine, to_phase,
    ))
    event_bus.on(TONE_SHIFT, lambda from_tone, to_tone, **_: print(f"  [Tone: {from_tone} → {to_tone}]"))
    event_bus.on(KERNEL_THOUGHT, lambda **kw: print(f"  [Thought: {kw.get('content') or kw.get('text', '')}]"))
    def _on_presence_stable(present, **_):
        print(f"  [Presence: {'detected' if present else 'lost'}]")

    from consciousness.events import PERCEPTION_USER_PRESENT_STABLE
    event_bus.on(PERCEPTION_USER_PRESENT_STABLE, _on_presence_stable)
    event_bus.on(PERCEPTION_WAKE_WORD, lambda **_: print("  [Wake word detected on Pi]"))

    # -- CLI ----------------------------------------------------------------

    print()
    print("  Systems online. Waiting for Pi senses to connect...")
    print("  Type a message or use commands:")
    print("  /status   - Show consciousness state")
    print("  /memories - Show recent memories")
    print("  /tone <t> - Set tone")
    print("  /sensors  - Show connected sensors")
    print("  /export   - Export soul snapshot")
    print("  /quit     - Shutdown")
    print()

    processors = perc_orch.get_processors()

    # -- Policy promotion background task -------------------------------------

    async def _policy_tick():
        while True:
            if promotion:
                try:
                    buf = promotion._buffer
                    buf_len = len(buf) if buf is not None else 0
                    if buf_len > 0 and buf_len % 50 == 0:
                        logger.info("Policy tick: phase=%s, experiences=%d",
                                    promotion.status.phase, buf_len)
                    if buf is not None and buf._unflushed > 0:
                        buf.flush()
                    await promotion.tick(time.time())
                except Exception as exc:
                    logger.warning("Policy tick error: %s", exc)
            await asyncio.sleep(60.0)

    if promotion:
        asyncio.get_event_loop().create_task(_policy_tick())

    DEEP_LEARNING_THRESHOLD_S = 3600.0  # 1 hour without sensors → deep learning mode
    _deep_learning_logged = False
    _gestation_end_handled = False

    async def _background_tick():
        nonlocal _deep_learning_logged, _gestation_end_handled
        while True:
            try:
                # Check if gestation just completed — transition perception gating
                if gestation_manager is not None and not _gestation_end_handled:
                    if not gestation_manager.is_active:
                        _gestation_end_handled = True
                        perc_orch.set_gestation_active(False)
                        if mode_manager.mode == "gestation":
                            mode_manager.set_mode("conversational", reason="gestation_complete", force=True)
                        if self_improve is not None:
                            self_improve.set_paused(False)
                            engine.consciousness.enable_self_improvement(self_improve)
                            engine._self_improve_orchestrator = self_improve
                        logger.info("Gestation complete — perception unlocked, wake word armed, self-improve enabled")

                reflection_engine.tick()
                if attention:
                    attention.tick()
                if perception and perception.presence:
                    perception.presence.check_divergence()
                try:
                    from personality.rollback import personality_rollback
                    from personality.evolution import trait_evolution
                    snapshot = trait_evolution.evaluate_traits()
                    trait_dict = {t.trait: t.score for t in snapshot.traits if t.score >= 0.15}
                    if trait_dict:
                        personality_rollback.update_traits(trait_dict)
                    conf = engine.consciousness.analytics.get_confidence()
                    personality_rollback.tick(conf.avg if hasattr(conf, "avg") else float(conf))
                except Exception as exc:
                    if not hasattr(_background_tick, "_rollback_err_logged"):
                        _background_tick._rollback_err_logged = True
                        logger.warning("Personality rollback tick failed: %s", exc, exc_info=True)
                # Sensor-absence mode switching
                if perception:
                    absent_s = perception.get_sensor_absent_duration()
                    cur_mode = mode_manager.mode
                    if absent_s >= DEEP_LEARNING_THRESHOLD_S and cur_mode != "deep_learning":
                        mode_manager.set_mode("deep_learning", reason="sensors_absent_1hr", force=True)
                        if not _deep_learning_logged:
                            logger.info("No sensors for %.0f min — entering deep_learning mode", absent_s / 60)
                            _deep_learning_logged = True
                    elif absent_s == 0 and cur_mode == "deep_learning":
                        mode_manager.set_mode("passive", reason="sensor_reconnected", force=True)
                        _deep_learning_logged = False
                        logger.info("Sensor reconnected — exiting deep_learning mode")
            except Exception as exc:
                logger.debug("Background tick error: %s", exc)
            await asyncio.sleep(5.0)

    asyncio.get_event_loop().create_task(_background_tick())

    # -- Post-restart verification task ----------------------------------------

    if _verification_pending is not None:
        async def _run_verification(pending):
            """Wait for sufficient data, then compare metrics and promote or rollback.

            Guarantees a terminal outcome even if the system is idle:
            - Normal path: wall-clock >= period AND ticks >= min_ticks
            - Timeout path: wall-clock >= MAX_VERIFICATION_CEILING_S
              with insufficient ticks → classify as 'stable (insufficient_data)'
            """
            from self_improve.verification import (
                capture_current_metrics, compare_metrics, clear_pending,
                write_verification_result, MAX_VERIFICATION_CEILING_S,
            )
            from self_improve.orchestrator import SelfImprovementOrchestrator
            from consciousness.events import IMPROVEMENT_PROMOTED, IMPROVEMENT_ROLLED_BACK

            logger.info(
                "Verification running for patch %s (period=%ds, min_ticks=%d, ceiling=%ds)",
                pending.patch_id, pending.verification_period_s,
                pending.min_ticks, MAX_VERIFICATION_CEILING_S,
            )
            start = time.time()
            insufficient_data = False

            while True:
                await asyncio.sleep(10.0)
                elapsed = time.time() - start
                wall_ok = elapsed >= pending.verification_period_s

                tick_count = 0
                tick_ok = False
                try:
                    if engine._kernel:
                        perf = engine._kernel.get_performance()
                        tick_count = perf.tick_count
                        tick_ok = tick_count >= pending.min_ticks
                except Exception:
                    pass

                if wall_ok and tick_ok:
                    break

                if elapsed >= MAX_VERIFICATION_CEILING_S:
                    insufficient_data = not tick_ok
                    if insufficient_data:
                        logger.warning(
                            "Verification ceiling reached (%ds) with only %d ticks "
                            "(need %d) — evaluating with insufficient data",
                            int(elapsed), tick_count, pending.min_ticks,
                        )
                    else:
                        logger.warning("Verification ceiling reached — evaluating")
                    break

            current = capture_current_metrics(engine)
            result, details = compare_metrics(
                pending.baselines, current, pending.target_metrics,
            )

            if insufficient_data and result == "regressed":
                logger.info(
                    "Downgrading 'regressed' → 'stable' due to insufficient data (%d ticks)",
                    tick_count,
                )
                result = "stable"
                details["reason"] = (
                    f"stable (insufficient_data): only {tick_count}/{pending.min_ticks} ticks, "
                    f"regressions not conclusive"
                )

            reason = details.get("reason", result)

            if insufficient_data and result == "stable":
                verdict = "stable_insufficient_data"
            else:
                verdict = result

            logger.info(
                "Verification result for %s: %s (%s) — %s",
                pending.patch_id, result, verdict, reason,
            )

            write_verification_result(verdict=verdict, reason=reason)

            # Persist verdict into improvements history (survives clear_pending)
            if self_improve is not None:
                try:
                    self_improve.record_verification_outcome(
                        pending.patch_id, verdict, reason,
                    )
                except Exception as hist_exc:
                    logger.warning("Failed to record verification outcome: %s", hist_exc)

            if result in ("improved", "stable"):
                clear_pending()
                try:
                    event_bus.emit(IMPROVEMENT_PROMOTED, id=pending.patch_id,
                                  description=pending.description[:100],
                                  verification_result=result,
                                  verification_verdict=verdict,
                                  verification_reason=reason)
                except Exception:
                    pass
                logger.info(
                    "Patch %s PROMOTED after restart-verify (%s)",
                    pending.patch_id, result,
                )
                print(f"  Verification: PROMOTED — patch {pending.patch_id} ({reason})")
            else:
                logger.warning(
                    "Patch %s REGRESSED — rolling back from %s (%s)",
                    pending.patch_id, pending.snapshot_path, reason,
                )
                ok = False
                try:
                    ok = SelfImprovementOrchestrator.restore_snapshot_static(pending.snapshot_path)
                except Exception as restore_exc:
                    logger.error("Snapshot restore threw during verification rollback: %s", restore_exc)
                clear_pending()
                if not ok:
                    logger.error(
                        "Rollback FAILED for patch %s — cleared pending, "
                        "self-improvement will be disabled on next boot",
                        pending.patch_id,
                    )
                try:
                    event_bus.emit(IMPROVEMENT_ROLLED_BACK, id=pending.patch_id,
                                  reason="post_restart_regression",
                                  verification_verdict=verdict,
                                  verification_reason=reason,
                                  details=details)
                except Exception:
                    pass
                print(f"  Verification: ROLLBACK — patch {pending.patch_id} ({reason}), restarting...")
                _request_restart("verification_rollback",
                                 f"Patch {pending.patch_id} regressed: {reason}")
                sys.exit(10)

        asyncio.get_event_loop().create_task(_run_verification(_verification_pending))

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        await _cli_loop(engine, response_gen, claude, perception, persistence, processors,
                        episodes=episodes)
    else:
        if sys.stdin.isatty():
            print("  Stdout is piped — CLI disabled to prevent log-echo feedback loop.")
        print("  Running headless. Use dashboard or Pi for interaction.")
        await _headless_loop(engine, perception, persistence, processors, episodes=episodes)


def _log_and_broadcast(msg: str, perception, engine, phase) -> None:
    print(msg)
    if perception:
        perception.broadcast({"type": "state_update", "phase": phase, "tone": engine.get_state()["tone"]})


async def _cli_loop(
    engine: ConsciousnessEngine,
    response_gen: ResponseGenerator,
    claude: ClaudeClient | None,
    perception: PerceptionServer | None,
    persistence: MemoryPersistence,
    processors: dict | None = None,
    episodes: EpisodicMemory | None = None,
) -> None:
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin,
    )

    import re as _re
    _PRINTABLE_RE = _re.compile(r'[^\x20-\x7E]')
    _LOG_LINE_RE = _re.compile(
        r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'  # timestamp
        r'|^\[.*\]\s+(INFO|DEBUG|WARNING|ERROR|CRITICAL):'  # logger
        r'|^Batches:\s+\d+%'  # tqdm
        r'|^HTTP Request:'  # httpx
        r'|^Loading weights:'  # model loading
    )
    _MIN_RESPONSE_INTERVAL = 2.0
    _last_response_time = 0.0

    def _cli_gate(text: str) -> str:
        try:
            from skills.capability_gate import capability_gate
            return capability_gate.check_text(text) or text
        except Exception:
            _fallback_re = _re.compile(
                r"\bI (?:can|could|will|'ll|'m able to) .{3,80}?[.!?\n]", _re.IGNORECASE,
            )
            return _fallback_re.sub("I don't have that capability yet.", text)

    try:
        while True:
            sys.stdout.write("  You: ")
            sys.stdout.flush()
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            line = _PRINTABLE_RE.sub('', line_bytes.decode()).strip()
            if not line or len(line) < 2:
                continue
            if _LOG_LINE_RE.search(line):
                continue

            if line.startswith("/"):
                await _handle_command(line, engine, response_gen, perception)
                continue

            now_ts = time.time()
            if now_ts - _last_response_time < _MIN_RESPONSE_INTERVAL:
                continue
            _last_response_time = now_ts

            routing = tool_router.route(line)

            if routing.tool == ToolType.TIME:
                print(f"\n  Jarvis: {get_current_time()}\n")
            elif routing.tool == ToolType.SYSTEM_STATUS:
                print(f"\n  Jarvis: {get_system_status()}\n")
            elif routing.tool == ToolType.MEMORY:
                result = search_memory(line) if any(w in line.lower() for w in ("search", "remember", "recall")) else get_memory_summary()
                print(f"\n  Jarvis: {result}\n")
            elif routing.tool == ToolType.VISION:
                _pi_host = os.environ.get("PI_HOST", "")
                _pi_port = os.environ.get("PI_UI_PORT", "8080")
                _pi_url = f"http://{_pi_host}:{_pi_port}/snapshot" if _pi_host else ""
                result = await describe_scene(
                    _pi_url,
                    ollama_client=response_gen.ollama,
                    claude_client=claude,
                )
                print(f"\n  Jarvis: {result}\n")
            elif routing.tool == ToolType.INTROSPECTION:
                introspection_data, _intro_meta = get_introspection(engine, query=line)
                try:
                    if episodes:
                        episodes.add_user_turn(line)
                    response = await response_gen.respond(
                        line,
                        perception_context=f"[Self-introspection data — answer based on this real data about yourself]\n{introspection_data}",
                    )
                    if episodes:
                        episodes.add_assistant_turn(response.text)
                    engine.consciousness.record_response_latency(response.latency_ms)
                    print(f"\n  Jarvis: {_cli_gate(response.text)}")
                    print(f"  [{response.latency_ms}ms]\n")
                except Exception:
                    print("\n  Error generating introspection response\n")
                    logger.exception("Introspection CLI error")
            else:
                try:
                    if episodes:
                        episodes.add_user_turn(line)
                    response = await response_gen.respond(line)
                    if episodes:
                        episodes.add_assistant_turn(response.text)
                    engine.consciousness.record_response_latency(response.latency_ms)
                    print(f"\n  Jarvis: {_cli_gate(response.text)}")
                    print(f"  [{response.latency_ms}ms]\n")
                except Exception:
                    print("\n  Error generating response\n")
                    logger.exception("CLI response error")
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        if episodes:
            episodes.end_episode()
            episodes.save()
        await _shutdown(engine, perception, persistence, processors)


async def _headless_loop(
    engine: ConsciousnessEngine,
    perception: PerceptionServer | None,
    persistence: MemoryPersistence,
    processors: dict | None = None,
    episodes: EpisodicMemory | None = None,
) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    except asyncio.CancelledError:
        pass
    finally:
        if episodes:
            episodes.end_episode()
            episodes.save()
        await _shutdown(engine, perception, persistence, processors)


async def _handle_command(
    cmd_line: str,
    engine: ConsciousnessEngine,
    response_gen: ResponseGenerator,
    perception: PerceptionServer | None,
) -> None:
    parts = cmd_line.split()
    cmd = parts[0]

    match cmd:
        case "/status":
            state = engine.get_state()
            stats = engine.get_memory_stats()
            print(f"\n  === Consciousness Status ===")
            print(f"  Phase: {state['phase']}")
            print(f"  Tone: {state['tone']}")
            print(f"  Tick: {state.get('tick', 0)}")
            print(f"  Traits: {', '.join(state['traits']) or 'none yet'}")
            print(f"  Memory density: {state['memory_density']:.2f}")
            print(f"  Memories: {stats['total']} ({stats['core_count']} core)")
            print(f"  Avg weight: {stats['avg_weight']:.3f}")
            if perception:
                print(f"  Sensors: {', '.join(perception.get_connected_sensors()) or 'none'}")
            print()

        case "/memories":
            memories = engine.get_recent_memories(10)
            print(f"\n  === Recent Memories ===")
            for m in memories:
                payload = m.payload if isinstance(m.payload, str) else str(m.payload)
                preview = payload[:80] + "..." if len(payload) > 80 else payload
                print(f"  [{m.type}] w={m.weight:.2f} {preview}")
            print()

        case "/tone":
            valid = ("professional", "casual", "urgent", "empathetic", "playful")
            if len(parts) < 2 or parts[1] not in valid:
                print(f"  Usage: /tone <{'|'.join(valid)}>")
            else:
                engine.set_tone(parts[1])
                print(f"  Tone set to: {parts[1]}")

        case "/sensors":
            if not perception:
                print("  Perception bus not enabled")
            else:
                sensors = perception.get_connected_sensors()
                print(f"  Connected sensors: {', '.join(sensors) if sensors else 'none'}")

        case "/export":
            snapshot = engine.export_soul("CLI export")
            print(f"  Soul exported: {len(snapshot.memories)} memories, id={snapshot.id}")

        case "/quit":
            print("  Shutting down...")
            raise KeyboardInterrupt

        case _:
            print(f"  Unknown command: {cmd}")


async def _shutdown(
    engine: ConsciousnessEngine,
    perception: PerceptionServer | None,
    persistence: MemoryPersistence,
    processors: dict | None = None,
) -> None:
    print("\n  Jarvis entering sleep...")

    if processors:
        for name in ("presence", "audio", "vision", "ambient", "screen"):
            proc = processors.get(name)
            if proc:
                proc.stop()
        srv = processors.get("perception")
        if srv:
            await srv.stop()

    consciousness_persistence.stop_auto_save()
    consciousness_persistence.save_from_system(engine.consciousness, engine=engine)
    persistence.stop_auto_save()

    try:
        save_intention_registry()
    except Exception:
        logger.debug("intention_registry save on shutdown failed", exc_info=True)

    buf = getattr(engine, "_experience_buffer", None)
    if buf is not None and hasattr(buf, "flush"):
        try:
            buf.flush()
            logger.info("Flushed %d policy experiences to disk", len(buf))
        except Exception as exc:
            logger.warning("Failed to flush policy experience buffer: %s", exc)

    from reasoning.context import context_builder
    context_builder.save()

    from memory.search import get_vector_store
    vs = get_vector_store()
    if vs:
        vs.close()

    engine.stop()
    logger.info("Goodnight.")


if __name__ == "__main__":
    asyncio.run(main())
