"""Gestation Manager — birth protocol for a fresh Jarvis brain.

When Jarvis boots with an empty brain, it enters a gestation period of
self-discovery before interacting with humans. The manager orchestrates
four internal phases:

  0. Self-Discovery      — codebase + architecture exploration (local-only)
  1. Knowledge Foundation — ML/AI academic research (scholarly-first)
  2. Autonomy Bootcamp    — exercise the full research→delta→policy loop
  3. Identity Formation   — personality emergence, readiness assessment

Gestation produces verifiable artifacts: a self-knowledge core, a research
foundation, measured deltas, NN training milestones, and a birth certificate.
First contact is deliberate — never blurting into a room.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

from config import GestationConfig
from consciousness.events import (
    event_bus,
    GESTATION_STARTED,
    GESTATION_PHASE_ADVANCED,
    GESTATION_DIRECTIVE_COMPLETED,
    GESTATION_READINESS_UPDATE,
    GESTATION_COMPLETE,
    GESTATION_FIRST_CONTACT,
)

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
BIRTH_CERTIFICATE_PATH = JARVIS_DIR / "gestation_summary.json"

ToolHint = Literal["codebase", "academic", "introspection", "memory"]

# ---------------------------------------------------------------------------
# Codebase files to ingest into Library during Phase 0
# ---------------------------------------------------------------------------

# Tier A: always ingest (critical for understanding own wiring)
_TIER_A_FILES: list[tuple[str, str]] = [
    # (repo-relative path, title for Library)
    ("AGENTS.md", "AGENTS.md — AI-facing architecture reference"),
    ("ARCHITECTURE.md", "ARCHITECTURE.md — data flow and system design"),
    ("docs/SYSTEM_OVERVIEW.md", "System Overview — high-level subsystem map"),
    ("brain/consciousness/events.py", "EventBus constants — 138 event types"),
    ("brain/consciousness/engine.py", "ConsciousnessEngine — top-level coordinator"),
    ("brain/consciousness/consciousness_system.py", "ConsciousnessSystem — tick coordinator"),
    ("brain/perception_orchestrator.py", "PerceptionOrchestrator — sensor-to-action pipeline"),
    ("brain/conversation_handler.py", "ConversationHandler — routing and tool dispatch"),
    ("brain/reasoning/context.py", "ContextBuilder — LLM prompt construction"),
    ("brain/reasoning/response.py", "ResponseGenerator — streaming with cancel"),
    ("brain/memory/storage.py", "MemoryStorage — unified write path"),
    ("brain/memory/search.py", "MemorySearch — keyword + semantic retrieval"),
]

# Tier B: ingest if Phase 0 budget allows
_TIER_B_FILES: list[tuple[str, str]] = [
    ("brain/autonomy/orchestrator.py", "AutonomyOrchestrator — research pipeline"),
    ("brain/self_improve/orchestrator.py", "SelfImprovementOrchestrator — 7-phase pipeline"),
    ("brain/self_improve/patch_plan.py", "PatchPlan — write boundaries and safety"),
    ("brain/policy/policy_nn.py", "PolicyNN — MLP/GRU policy networks"),
    ("brain/policy/state_encoder.py", "StateEncoder — 20-dim state vector"),
    ("brain/hemisphere/orchestrator.py", "HemisphereOrchestrator — NN management"),
    ("brain/epistemic/contradiction_engine.py", "ContradictionEngine — belief integrity"),
    ("brain/skills/capability_gate.py", "CapabilityGate — honesty enforcement"),
    ("brain/perception/identity_fusion.py", "IdentityFusion — voice + face identity"),
    ("brain/perception/audio_stream.py", "AudioStreamProcessor — wake word + VAD"),
    ("brain/reasoning/tool_router.py", "ToolRouter — intent classification"),
]


@dataclass
class StudyDirective:
    question: str
    tool_hint: ToolHint
    tag_cluster: tuple[str, ...]
    priority: float
    category: str  # self_study, knowledge_foundation, autonomy_bootcamp


# ---------------------------------------------------------------------------
# Phase 0: Self-study directives (local-only, zero network)
# ---------------------------------------------------------------------------

SELF_STUDY_DIRECTIVES: list[StudyDirective] = [
    # --- Tier 1: Architecture docs (queries ingested docs) ---
    StudyDirective(
        question=(
            "What is the overall architecture of this system? "
            "What are the two devices, what does each do, and how do they communicate?"
        ),
        tool_hint="codebase",
        tag_cluster=("architecture", "self_knowledge", "design", "two_device"),
        priority=0.95,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "What is the full voice pipeline data flow from Pi microphone "
            "to brain response to Pi audio playback?"
        ),
        tool_hint="codebase",
        tag_cluster=("voice_pipeline", "data_flow", "self_knowledge"),
        priority=0.93,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "What runs during one consciousness tick cycle? What cycles run "
            "at what intervals and how does mode gating work?"
        ),
        tool_hint="codebase",
        tag_cluster=("consciousness", "tick_cycle", "mode_gating", "self_knowledge"),
        priority=0.91,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "What are the 11 epistemic layers (0-11, plus 3A and 3B) "
            "and how do they protect cognitive integrity?"
        ),
        tool_hint="codebase",
        tag_cluster=("epistemic", "layers", "integrity", "self_knowledge"),
        priority=0.90,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does the neural policy layer work? What is shadow A/B "
            "evaluation and how does promotion happen?"
        ),
        tool_hint="codebase",
        tag_cluster=("policy", "shadow_eval", "promotion", "self_knowledge"),
        priority=0.88,
        category="self_study",
    ),
    # --- Tier 2: Deep module study (queries actual source code) ---
    StudyDirective(
        question=(
            "How does ConsciousnessEngine work? What are the KernelCallbacks "
            "and how does on_tick coordinate all subsystems?"
        ),
        tool_hint="codebase",
        tag_cluster=("consciousness", "engine", "kernel", "self_knowledge"),
        priority=0.87,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does the memory write path work from creation through storage, "
            "indexing, vector embedding, and event emission?"
        ),
        tool_hint="codebase",
        tag_cluster=("memory", "storage", "write_path", "self_knowledge"),
        priority=0.86,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does PerceptionOrchestrator wire sensors to actions? "
            "What happens on transcription, wake word, and barge-in?"
        ),
        tool_hint="codebase",
        tag_cluster=("perception", "orchestrator", "sensors", "self_knowledge"),
        priority=0.85,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does ConversationHandler route user messages to tools and generate responses, "
            "including native STATUS self-report and strict introspection paths?"
        ),
        tool_hint="codebase",
        tag_cluster=("conversation", "tool_routing", "response", "self_knowledge"),
        priority=0.84,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does the self-improvement pipeline ensure safety? "
            "What are ALLOWED_PATHS, DENIED_PATTERNS, and sandbox testing?"
        ),
        tool_hint="codebase",
        tag_cluster=("self_improve", "safety", "sandbox", "self_knowledge"),
        priority=0.83,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does the contradiction engine protect belief integrity? "
            "What is claim extraction and conflict classification?"
        ),
        tool_hint="codebase",
        tag_cluster=("epistemic", "contradiction", "beliefs", "self_knowledge"),
        priority=0.82,
        category="self_study",
    ),
    StudyDirective(
        question="How does identity fusion resolve voice and face signals into a canonical identity?",
        tool_hint="codebase",
        tag_cluster=("identity", "fusion", "voice", "face", "self_knowledge"),
        priority=0.81,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does the autonomy pipeline decide what to research, "
            "score opportunities, and measure outcomes?"
        ),
        tool_hint="codebase",
        tag_cluster=("autonomy", "research", "scoring", "self_knowledge"),
        priority=0.80,
        category="self_study",
    ),
    StudyDirective(
        question="How do hemisphere neural networks get designed, trained, and evolved?",
        tool_hint="codebase",
        tag_cluster=("hemisphere", "neural_network", "evolution", "self_knowledge"),
        priority=0.79,
        category="self_study",
    ),
    StudyDirective(
        question=(
            "How does the capability gate enforce honesty? What are the 7 enforcement layers, "
            "15 claim patterns (including action confabulation detection), and self-report sanitization boundaries?"
        ),
        tool_hint="codebase",
        tag_cluster=("capability_gate", "honesty", "enforcement", "self_knowledge"),
        priority=0.78,
        category="self_study",
    ),
]

# ---------------------------------------------------------------------------
# Phase 1: Knowledge foundation directives (academic search)
# ---------------------------------------------------------------------------

KNOWLEDGE_FOUNDATION_DIRECTIVES: list[StudyDirective] = [
    StudyDirective(
        question="What are the best practices for training small neural networks with limited data?",
        tool_hint="academic",
        tag_cluster=("neural_network", "training", "small_data"),
        priority=0.85,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="How does policy gradient reinforcement learning work for behavioral optimization?",
        tool_hint="academic",
        tag_cluster=("reinforcement_learning", "policy_gradient", "optimization"),
        priority=0.85,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="What are effective architectures for episodic memory in AI systems?",
        tool_hint="academic",
        tag_cluster=("episodic_memory", "architecture", "retrieval"),
        priority=0.80,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="What computational models of consciousness exist in AI research?",
        tool_hint="academic",
        tag_cluster=("consciousness", "computational_model", "self_awareness"),
        priority=0.80,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="When should GRU networks be preferred over transformers for sequential decision making?",
        tool_hint="academic",
        tag_cluster=("gru", "transformer", "sequential", "decision_making"),
        priority=0.75,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="How do AI systems model evolving personality traits based on interaction history?",
        tool_hint="academic",
        tag_cluster=("personality", "trait_model", "interaction", "evolution"),
        priority=0.75,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="What are retrieval augmented generation best practices for grounding LLM responses in facts?",
        tool_hint="academic",
        tag_cluster=("rag", "retrieval", "grounding", "factual"),
        priority=0.70,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="What are best practices for counterfactual credit assignment in autonomous decision systems?",
        tool_hint="academic",
        tag_cluster=("credit_assignment", "counterfactual", "causal_inference"),
        priority=0.85,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="How do adaptive systems calibrate thresholds from small data with streaming statistics?",
        tool_hint="academic",
        tag_cluster=("calibration", "small_data", "streaming_statistics", "threshold"),
        priority=0.80,
        category="knowledge_foundation",
    ),
    StudyDirective(
        question="What safety mechanisms are used in AI systems that modify their own code?",
        tool_hint="academic",
        tag_cluster=("self_improvement", "safety", "code_generation", "sandboxing"),
        priority=0.80,
        category="knowledge_foundation",
    ),
]

# ---------------------------------------------------------------------------
# Phase 2: Autonomy bootcamp directives
# ---------------------------------------------------------------------------

AUTONOMY_BOOTCAMP_DIRECTIVES: list[StudyDirective] = [
    StudyDirective(
        question=(
            "Trace the full data flow when a user says 'Hello Jarvis': "
            "from Pi microphone capture through wake word detection, VAD, STT, "
            "tool routing, LLM response, TTS synthesis, back to Pi audio playback. "
            "What are all the events emitted along this path?"
        ),
        tool_hint="codebase",
        tag_cluster=("voice_pipeline", "data_flow", "events", "architecture"),
        priority=0.90,
        category="autonomy_bootcamp",
    ),
    StudyDirective(
        question=(
            "Trace what happens during one consciousness tick cycle: "
            "what does consciousness_system.on_tick() check, what cycles run "
            "at what intervals, and how does the mode profile gate which cycles "
            "are allowed?"
        ),
        tool_hint="codebase",
        tag_cluster=("consciousness", "tick_cycle", "mode_gating", "architecture"),
        priority=0.85,
        category="autonomy_bootcamp",
    ),
    StudyDirective(
        question=(
            "Trace a memory write from creation through the epistemic layers: "
            "contradiction checking, belief extraction, graph edge creation, "
            "quarantine scoring, and truth calibration. What are all the events "
            "and safety gates along this path?"
        ),
        tool_hint="codebase",
        tag_cluster=("memory", "epistemic", "data_flow", "architecture"),
        priority=0.85,
        category="autonomy_bootcamp",
    ),
    StudyDirective(
        question=(
            "How does the self-improvement loop work end-to-end? "
            "What are ALLOWED_PATHS, DENIED_PATTERNS, and what does "
            "the sandbox test before a patch is applied?"
        ),
        tool_hint="codebase",
        tag_cluster=("self_improve", "safety", "sandbox", "architecture"),
        priority=0.80,
        category="autonomy_bootcamp",
    ),
    StudyDirective(
        question="What are effective techniques for training small neural networks with limited data in real-time systems?",
        tool_hint="academic",
        tag_cluster=("neural_network", "small_data", "real_time", "training"),
        priority=0.75,
        category="autonomy_bootcamp",
    ),
]


# ---------------------------------------------------------------------------
# Readiness Assessment
# ---------------------------------------------------------------------------

@dataclass
class ReadinessAssessment:
    overall: float
    components: dict[str, float]
    met_minimum_duration: bool
    recommendation: str  # "continue", "ready", "ready_person_waiting"


# ---------------------------------------------------------------------------
# GestationManager
# ---------------------------------------------------------------------------

class GestationManager:
    """Orchestrates the gestation protocol for a fresh Jarvis brain."""

    def __init__(self, config: GestationConfig) -> None:
        self._config = config
        self._phase: int = 0  # 0=self-study, 1=knowledge, 2=bootcamp, 3=identity
        self._phase_names = ("self_discovery", "knowledge_foundation", "autonomy_bootcamp", "identity_formation")
        self._started_at: float = 0.0
        self._phase_started_at: float = 0.0
        self._is_active: bool = False

        self._directives_issued: int = 0
        self._directives_completed: int = 0
        self._research_jobs_completed: int = 0
        # Per-phase success counters (prevent phase-skip from shared counter)
        self._self_study_completed: int = 0
        self._knowledge_completed: int = 0
        self._bootcamp_completed: int = 0
        self._last_readiness_check: float = 0.0

        self._person_detected: bool = False
        self._person_sustained_s: float = 0.0
        self._person_first_seen: float = 0.0

        self._readiness: ReadinessAssessment | None = None
        self._backpressure_until: float = 0.0
        self._network_healthy: bool = True
        self._network_fail_count: int = 0

        self._self_study_queue: list[StudyDirective] = list(SELF_STUDY_DIRECTIVES)
        self._knowledge_queue: list[StudyDirective] = list(KNOWLEDGE_FOUNDATION_DIRECTIVES)
        self._bootcamp_queue: list[StudyDirective] = list(AUTONOMY_BOOTCAMP_DIRECTIVES)

        self._first_contact_armed: bool = False
        self._completed_dois: list[str] = []
        self._completed_directive_summaries: list[str] = []
        self._strict_provenance_set: bool = False
        self._blocked_retries: dict[str, int] = {}
        self._library_ingest_done: bool = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def first_contact_armed(self) -> bool:
        return self._first_contact_armed

    def start(self, resume: bool = False) -> None:
        """Enter gestation mode. If resume=True, restoring from interrupted gestation."""
        self._is_active = True
        if resume:
            saved = self._load_gestation_progress()
            self._started_at = saved.get("started_at", time.time())
            self._phase = saved.get("phase", 0)
            self._directives_issued = saved.get("directives_issued", 0)
            self._directives_completed = saved.get("directives_completed", 0)
            self._research_jobs_completed = saved.get("research_jobs_completed", 0)
            self._self_study_completed = saved.get("self_study_completed", 0)
            self._knowledge_completed = saved.get("knowledge_completed", 0)
            self._bootcamp_completed = saved.get("bootcamp_completed", 0)
            self._completed_dois = saved.get("completed_dois", [])
            self._library_ingest_done = saved.get("library_ingest_done", False)
            # Drain already-issued directives from queues
            issued = self._directives_issued
            for q in (self._self_study_queue, self._knowledge_queue, self._bootcamp_queue):
                drain = min(issued, len(q))
                del q[:drain]
                issued -= drain
                if issued <= 0:
                    break
            logger.info(
                "Gestation resumed — phase %d (%s), %d directives done, %.1f min elapsed",
                self._phase, self._phase_names[self._phase] if self._phase < len(self._phase_names) else "?",
                self._directives_completed, (time.time() - self._started_at) / 60.0,
            )
        else:
            self._started_at = time.time()
            self._phase = 0
            logger.info("Gestation started — entering self-discovery phase")
        self._phase_started_at = time.time()
        self._reload_blue_diamonds()
        self._persist_gestation_progress()
        event_bus.emit(GESTATION_STARTED, timestamp=time.time(), resumed=resume)

    # ------------------------------------------------------------------
    # Blue Diamonds reload — pre-phase knowledge restoration
    # ------------------------------------------------------------------

    def _reload_blue_diamonds(self) -> None:
        """Reload validated knowledge from Blue Diamonds archive into fresh library."""
        try:
            from library.blue_diamonds import BlueDiamondsArchive
            archive = BlueDiamondsArchive.get_instance()
            if not archive.has_content():
                logger.info("No Blue Diamonds archive — starting from scratch")
                return

            from library.source import Source, source_store
            from library.chunks import Chunk, chunk_store
            from library.index import library_index

            entries = archive.reload_all()
            loaded = 0
            chunks_loaded = 0
            for source_dict, chunk_dicts in entries:
                sid = source_dict["source_id"]
                if source_store.exists(sid):
                    continue

                source = Source(
                    source_id=sid,
                    source_type=source_dict.get("source_type", ""),
                    retrieved_at=time.time(),
                    url=source_dict.get("url", ""),
                    doi=source_dict.get("doi", ""),
                    title=source_dict.get("title", ""),
                    authors=source_dict.get("authors", ""),
                    year=source_dict.get("year", 0),
                    venue=source_dict.get("venue", ""),
                    citation_count=source_dict.get("citation_count", 0),
                    content_text=source_dict.get("content_text", ""),
                    content_depth=source_dict.get("content_depth", ""),
                    quality_score=source_dict.get("quality_score", 0.0),
                    domain_tags=source_dict.get("domain_tags", ""),
                    canonical_domain=source_dict.get("canonical_domain", ""),
                    provider=source_dict.get("provider", ""),
                    license_flags=source_dict.get("license_flags", ""),
                    ingested_by="blue_diamonds",
                    trust_tier="verified",
                )
                source_store.add(source)

                chunks = []
                for cd in chunk_dicts:
                    chunk = Chunk(
                        chunk_id=cd["chunk_id"],
                        source_id=sid,
                        text=cd["text"],
                        offset=cd.get("offset", 0),
                        concepts=cd.get("concepts", []),
                        chunk_type=cd.get("chunk_type", ""),
                    )
                    chunks.append(chunk)

                if chunks:
                    chunk_store.add_many(chunks)
                    for chunk in chunks:
                        library_index.add_chunk(chunk.chunk_id, sid, chunk.text)
                    chunks_loaded += len(chunks)
                loaded += 1

            if loaded > 0:
                archive.log_reload(loaded, chunks_loaded, trigger="gestation")
                logger.info(
                    "Reloaded %d Blue Diamonds (%d chunks) into fresh library",
                    loaded, chunks_loaded,
                )
        except Exception as exc:
            logger.warning("Blue Diamonds reload failed: %s", exc)

    # ------------------------------------------------------------------
    # Library codebase ingestion
    # ------------------------------------------------------------------

    def _ingest_codebase_to_library(self) -> None:
        """Ingest key source files and docs into the Library for deep retrieval.

        Tier A files are mandatory. Tier B files are ingested if time allows.
        Uses hash-based dedup: unchanged files are skipped on resume.
        """
        if self._library_ingest_done:
            return

        try:
            from library.ingest import ingest_codebase_source
        except ImportError:
            logger.warning("Library ingest not available — skipping codebase ingestion")
            self._library_ingest_done = True
            return

        project_root = Path(__file__).resolve().parent.parent.parent
        t0 = time.monotonic()
        ingested = 0
        skipped = 0

        for tier_name, file_list in [("A", _TIER_A_FILES), ("B", _TIER_B_FILES)]:
            for rel_path, title in file_list:
                fpath = project_root / rel_path
                if not fpath.exists():
                    logger.debug("Codebase ingest skip (not found): %s", rel_path)
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    if not content.strip():
                        continue
                    result = ingest_codebase_source(
                        file_path=rel_path,
                        content=content,
                        title=title,
                        domain_tags="codebase,self_knowledge,architecture" if rel_path.endswith(".md") else "codebase,self_knowledge",
                    )
                    if result.success and result.error != "unchanged":
                        ingested += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    logger.debug("Codebase ingest failed for %s: %s", rel_path, exc)

            elapsed_ms = (time.monotonic() - t0) * 1000
            if tier_name == "A":
                logger.info(
                    "Tier A codebase ingest: %d ingested, %d skipped in %.0fms",
                    ingested, skipped, elapsed_ms,
                )
            if tier_name == "A" and elapsed_ms > 30_000:
                logger.info("Tier A took >30s — skipping Tier B to stay on schedule")
                break

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Codebase Library ingest complete: %d ingested, %d skipped in %.0fms",
            ingested, skipped, elapsed_ms,
        )
        self._library_ingest_done = True

    # ------------------------------------------------------------------
    # Tick — called from engine.on_autonomy_tick during gestation
    # ------------------------------------------------------------------

    def on_tick(self, autonomy_orch: Any, last_tick_ms: float, budget_ms: float) -> None:
        """Drive gestation forward. Called on each kernel tick in BACKGROUND priority."""
        if not self._is_active:
            return

        if not self._strict_provenance_set and autonomy_orch:
            try:
                autonomy_orch.set_strict_provenance(True)
                self._strict_provenance_set = True
            except Exception:
                pass

        now = time.time()

        # Backpressure: skip if kernel was recently under pressure
        if last_tick_ms > budget_ms * 0.8:
            self._backpressure_until = now + 10.0
            return
        if now < self._backpressure_until:
            return

        # Ingest codebase files into Library on first Phase 0 tick
        if self._phase == 0 and not self._library_ingest_done:
            self._ingest_codebase_to_library()

        # Check phase transitions
        self._check_phase_advancement(autonomy_orch)

        # Max duration safety cap
        elapsed = now - self._started_at
        if elapsed >= self._config.max_duration_s:
            logger.warning("Gestation max duration reached (%.0fh) — forcing graduation",
                           elapsed / 3600.0)
            self._graduate(autonomy_orch, forced=True)
            return

        # Assess readiness periodically (every 30s)
        if now - self._last_readiness_check > 30.0:
            self._last_readiness_check = now
            self._readiness = self._assess_readiness()
            event_bus.emit(GESTATION_READINESS_UPDATE,
                           overall=self._readiness.overall,
                           components=self._readiness.components,
                           recommendation=self._readiness.recommendation)

            self._persist_gestation_progress()

            if self._readiness.recommendation in ("ready", "ready_person_waiting"):
                self._graduate(autonomy_orch, forced=False)
                return

        # Issue directives for current phase
        self._issue_directives(autonomy_orch)

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _check_phase_advancement(self, autonomy_orch: Any) -> None:
        """Check if current phase's completion criteria are met.

        Uses per-phase success counters to prevent early phases inflating
        later phase thresholds (e.g. self_study successes must not count
        toward knowledge_foundation advancement).
        """
        if self._phase == 0:
            target = len(SELF_STUDY_DIRECTIVES)
            if self._self_study_completed >= max(target - 5, 1):
                self._advance_phase(1)
        elif self._phase == 1:
            target = len(KNOWLEDGE_FOUNDATION_DIRECTIVES)
            if self._knowledge_completed >= max(target - 4, 1):
                self._advance_phase(2)
        elif self._phase == 2:
            delta_total = self._get_delta_measured()
            target = len(AUTONOMY_BOOTCAMP_DIRECTIVES)
            if self._bootcamp_completed >= max(target - 2, 1) or delta_total >= 5:
                self._advance_phase(3)

    def _advance_phase(self, new_phase: int) -> None:
        old_name = self._phase_names[self._phase] if self._phase < len(self._phase_names) else "?"
        new_name = self._phase_names[new_phase] if new_phase < len(self._phase_names) else "?"
        self._phase = new_phase
        self._phase_started_at = time.time()
        event_bus.emit(GESTATION_PHASE_ADVANCED, from_phase=old_name, to_phase=new_name)
        logger.info("Gestation phase: %s → %s", old_name, new_name)

    # ------------------------------------------------------------------
    # Directive issuance
    # ------------------------------------------------------------------

    def _issue_directives(self, autonomy_orch: Any) -> None:
        """Issue the next batch of directives as research intents.

        Only issues new directives when the autonomy queue has capacity,
        preventing queue flooding.
        """
        current_queue = getattr(autonomy_orch, '_queue', None)
        if current_queue is not None and len(current_queue) >= 5:
            return  # wait for current batch to process

        batch_size = min(self._config.self_study_batch_size, 3)
        issued = 0

        if self._phase == 0:
            queue = self._self_study_queue
        elif self._phase == 1:
            queue = self._knowledge_queue
        elif self._phase == 2:
            queue = self._bootcamp_queue
        elif self._phase == 3:
            for q in (self._bootcamp_queue, self._knowledge_queue, self._self_study_queue):
                if q:
                    queue = q
                    break
            else:
                return
        else:
            return

        while queue and issued < batch_size:
            directive = queue.pop(0)
            try:
                self._enqueue_directive(autonomy_orch, directive)
                issued += 1
                self._directives_issued += 1
            except Exception as exc:
                logger.warning("Failed to enqueue gestation directive: %s", exc)
                if directive.tool_hint == "academic":
                    self._network_fail_count += 1
                    if self._network_fail_count >= 5:
                        self._network_healthy = False

    def _enqueue_directive(self, autonomy_orch: Any, directive: StudyDirective) -> None:
        """Convert a StudyDirective to a ResearchIntent and enqueue it."""
        from autonomy.research_intent import ResearchIntent

        hint = directive.tool_hint if directive.tool_hint in ("academic", "web", "codebase", "introspection", "memory") else "any"
        intent = ResearchIntent(
            question=directive.question,
            source_event=f"gestation:{directive.category}",
            source_hint=hint,
            priority=directive.priority,
            scope="local_only" if directive.tool_hint in ("codebase", "introspection", "memory") else "external_ok",
            tag_cluster=directive.tag_cluster,
        )
        autonomy_orch.enqueue(intent)

    def on_directive_completed(self, intent_id: str, result: Any = None,
                               source_event: str = "") -> None:
        """Called when a gestation-sourced research intent completes."""
        self._directives_completed += 1
        success = bool(result and getattr(result, "success", False))
        if success:
            self._research_jobs_completed += 1
            # Per-phase tracking so phase advancement uses phase-local counts
            category = source_event.split(":")[-1] if source_event else ""
            if category == "self_study":
                self._self_study_completed += 1
            elif category == "knowledge_foundation":
                self._knowledge_completed += 1
            elif category == "autonomy_bootcamp":
                self._bootcamp_completed += 1
            for finding in getattr(result, "findings", []):
                doi = getattr(finding, "doi", "")
                if doi:
                    self._completed_dois.append(doi)

        summary = getattr(result, "summary", "")[:80] if result else ""
        if summary:
            self._completed_directive_summaries.append(summary)
            if len(self._completed_directive_summaries) > 30:
                self._completed_directive_summaries = self._completed_directive_summaries[-30:]

        event_bus.emit(GESTATION_DIRECTIVE_COMPLETED,
                       intent_id=intent_id,
                       directives_completed=self._directives_completed,
                       research_jobs=self._research_jobs_completed)

    # ------------------------------------------------------------------
    # Blocked directive re-queue (called from autonomy orchestrator)
    # ------------------------------------------------------------------

    _MAX_DIRECTIVE_RETRIES = 3

    def on_directive_blocked(self, intent_id: str, question: str,
                             source_event: str = "", tool_hint: str = "any",
                             priority: int = 50, tag_cluster: set | None = None) -> None:
        """Re-queue a gestation directive that was blocked by the autonomy governor.

        Undoes the issued count so the directive can be retried on the next tick
        when rate limits have refreshed.
        """
        category = source_event.split(":")[-1] if source_event else ""
        retry_key = question[:80]
        count = self._blocked_retries.get(retry_key, 0)
        if count >= self._MAX_DIRECTIVE_RETRIES:
            logger.warning("Gestation directive exceeded max retries (%d), dropping: %s",
                           self._MAX_DIRECTIVE_RETRIES, question[:60])
            return

        self._blocked_retries[retry_key] = count + 1
        self._directives_issued = max(0, self._directives_issued - 1)

        directive = StudyDirective(
            question=question, category=category,
            tool_hint=tool_hint, priority=priority,
            tag_cluster=tag_cluster or set(),
        )
        if category == "self_study":
            self._self_study_queue.append(directive)
        elif category == "knowledge_foundation":
            self._knowledge_queue.append(directive)
        elif category == "autonomy_bootcamp":
            self._bootcamp_queue.append(directive)
        else:
            self._knowledge_queue.append(directive)

        logger.info("Re-queued blocked gestation directive (retry %d/%d, category=%s): %s",
                     count + 1, self._MAX_DIRECTIVE_RETRIES, category, question[:60])

    # ------------------------------------------------------------------
    # Person detection (called from perception orchestrator)
    # ------------------------------------------------------------------

    def on_person_detected(self, present: bool, confidence: float = 0.0) -> None:
        """Track sustained person presence for first-contact readiness."""
        now = time.time()
        if present and confidence > 0.5:
            if not self._person_detected:
                self._person_detected = True
                self._person_first_seen = now
            self._person_sustained_s = now - self._person_first_seen
        else:
            self._person_detected = False
            self._person_sustained_s = 0.0
            self._person_first_seen = 0.0

    # ------------------------------------------------------------------
    # Readiness assessment
    # ------------------------------------------------------------------

    def _assess_readiness(self) -> ReadinessAssessment:
        scores: dict[str, float] = {}

        # Self-knowledge: has Jarvis studied itself?
        # Primary signal: self-study directive completions (reliable, tracked internally).
        # Secondary signal: memories tagged self_knowledge (tag propagation may not
        # reach study_claim memories created by the library study pipeline).
        sk_from_completions = self._self_study_completed / max(1, len(SELF_STUDY_DIRECTIVES))
        sk_from_tags = self._count_memories_by_tag("self_knowledge") / 8
        scores["self_knowledge"] = min(1.0, max(sk_from_completions, sk_from_tags))

        # Research foundation: denominator = actual directive count (not hardcoded config)
        total_directives = len(KNOWLEDGE_FOUNDATION_DIRECTIVES) + len(AUTONOMY_BOOTCAMP_DIRECTIVES)
        knowledge_jobs = self._knowledge_completed + self._bootcamp_completed
        scores["knowledge_foundation"] = min(1.0, knowledge_jobs / max(1, total_directives))

        # Memory mass
        total_memories = self._get_total_memory_count()
        scores["memory_mass"] = min(1.0, total_memories / max(1, self._config.min_memories_for_ready))

        # Consciousness advancement
        stage = self._get_evolution_stage()
        stage_scores = {"basic_awareness": 0.3, "self_reflective": 0.7, "philosophical": 1.0,
                        "recursive_self_modeling": 1.0, "integrative": 1.0}
        scores["consciousness_stage"] = stage_scores.get(stage, 0.1)

        # Hemisphere NNs: real training proof
        scores["hemisphere_training"] = self._assess_hemisphere_quality()

        # Personality emergence (tracked but excluded from weighted score during
        # gestation — personality requires user interaction which hasn't happened yet)
        scores["personality_emergence"] = min(1.0, self._get_trait_deviation() / 0.3)

        # Policy experience
        scores["policy_experience"] = min(1.0, self._get_experience_count() / 50)

        # Loop integrity: autonomy engine actually completed full loops
        measured_deltas = self._get_delta_measured()
        scores["loop_integrity"] = min(1.0, measured_deltas / max(1, self._config.min_measured_deltas))

        # Duration gate
        elapsed = time.time() - self._started_at
        met_minimum = elapsed >= self._config.min_duration_s

        # Weights — personality_emergence excluded during gestation (chicken-and-egg:
        # personality emerges from conversation, but gestation precedes first contact).
        # Its 0.10 weight is redistributed: +0.05 self_knowledge, +0.05 loop_integrity.
        weights = {
            "self_knowledge": 0.25,
            "knowledge_foundation": 0.15,
            "memory_mass": 0.10,
            "consciousness_stage": 0.15,
            "hemisphere_training": 0.10,
            "policy_experience": 0.05,
            "loop_integrity": 0.20,
        }
        if not self._network_healthy:
            weights["self_knowledge"] = 0.30
            weights["knowledge_foundation"] = 0.05
            weights["loop_integrity"] = 0.15
            weights["memory_mass"] = 0.15
            weights["consciousness_stage"] = 0.20
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        overall = sum(scores.get(k, 0.0) * weights[k] for k in weights)

        # Recommendation
        threshold = self._config.readiness_threshold
        threshold_waiting = self._config.readiness_threshold_waiting
        if not met_minimum:
            recommendation = "continue"
        elif overall >= threshold:
            recommendation = "ready"
        elif (overall >= threshold_waiting
              and self._person_detected
              and self._person_sustained_s > self._config.person_sustained_s):
            recommendation = "ready_person_waiting"
        else:
            recommendation = "continue"

        return ReadinessAssessment(
            overall=overall,
            components=scores,
            met_minimum_duration=met_minimum,
            recommendation=recommendation,
        )

    def _assess_hemisphere_quality(self) -> float:
        """Check if any hemisphere has real training evidence.

        Counts as trained if ANY hemisphere meets at least one of:
        - total_attempts >= 1
        - best_loss is not None (a training pass completed)
        - evolution_generations >= 1
        """
        try:
            engine = self._get_engine()
            if not engine or not engine._hemisphere_orchestrator:
                return 0.0
            full_state = engine._hemisphere_orchestrator.get_state()
            hemi_state = full_state.get("hemisphere_state", {})
            hemis = hemi_state.get("hemispheres", [])
            if not hemis:
                return 0.0
            trained_count = 0
            best_score = 0.0
            for h in hemis:
                has_attempts = h.get("total_attempts", 0) >= 1
                has_loss = h.get("best_loss") is not None
                has_generations = h.get("evolution_generations", 0) >= 1
                has_networks = h.get("network_count", 0) >= 1
                if has_attempts or has_loss or has_generations or has_networks:
                    trained_count += 1
                    accuracy = h.get("best_accuracy", 0) or 0
                    gens = h.get("evolution_generations", 0) or 0
                    # Score: any training = 0.5 base, accuracy bonus, generation bonus
                    score = 0.5
                    score += min(0.3, accuracy * 0.5)
                    score += min(0.2, gens * 0.05)
                    best_score = max(best_score, score)
            if trained_count == 0:
                return 0.0
            # Bonus for multiple hemispheres training
            multi_bonus = min(0.2, (trained_count - 1) * 0.05)
            return min(1.0, best_score + multi_bonus)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Graduation
    # ------------------------------------------------------------------

    def _graduate(self, autonomy_orch: Any, forced: bool = False) -> None:
        """Complete gestation — save birth certificate, arm first contact."""
        self._is_active = False
        readiness = self._readiness or self._assess_readiness()

        if autonomy_orch:
            try:
                autonomy_orch.set_strict_provenance(False)
            except Exception:
                pass

        self._seed_post_birth_personality()
        self._save_birth_certificate(readiness, forced)
        self._persist_gestation_complete()

        # Arm first contact — don't speak yet
        self._first_contact_armed = True

        event_bus.emit(GESTATION_COMPLETE,
                       readiness=readiness.overall,
                       forced=forced,
                       duration_s=time.time() - self._started_at)

        logger.info("Gestation complete (forced=%s, readiness=%.2f, duration=%.0fmin) — first contact armed",
                     forced, readiness.overall, (time.time() - self._started_at) / 60.0)

    def _seed_post_birth_personality(self) -> None:
        """Analyze gestation memories and seed initial personality traits.

        During gestation, Jarvis only does research (no conversation), so
        memories are mostly factual_knowledge with study tags. We derive
        initial trait signals from the content of what was studied and how
        the research went, then:
        1. Mark the validator's post-birth grace window
        2. Seed the trait evolution engine with initial scores
        3. Update the soul's semi_stable_traits
        """
        try:
            from personality.validator import trait_validator
            trait_validator.mark_birth()
        except Exception as exc:
            logger.warning("Failed to mark post-birth grace: %s", exc)

        try:
            from memory.storage import memory_storage
            memories = memory_storage.get_all()
            from collections import Counter
            all_tags = Counter()
            types = Counter()
            for m in memories:
                for t in getattr(m, "tags", []):
                    all_tags[t] += 1
                types[getattr(m, "type", "unknown")] += 1

            seeds: dict[str, float] = {}

            tech_signal = (all_tags.get("technical", 0) + all_tags.get("code", 0) +
                           all_tags.get("architecture", 0) + all_tags.get("algorithm", 0))
            seeds["Technical"] = min(0.6, tech_signal * 0.03) if tech_signal else 0.1

            detail_signal = (all_tags.get("detail", 0) + all_tags.get("preference", 0) +
                             all_tags.get("specific", 0) + all_tags.get("precise", 0))
            seeds["Detail-Oriented"] = min(0.5, detail_signal * 0.02) if detail_signal else 0.1

            research_volume = all_tags.get("autonomous_research", 0) + all_tags.get("study_claim", 0)
            seeds["Proactive"] = min(0.5, research_volume * 0.005) if research_volume else 0.1

            empathy_signal = (all_tags.get("emotion", 0) + all_tags.get("empathy", 0) +
                              all_tags.get("support", 0))
            seeds["Empathetic"] = min(0.4, empathy_signal * 0.03) if empathy_signal else 0.15

            efficiency_signal = types.get("task_completed", 0) + all_tags.get("concise", 0)
            seeds["Efficient"] = min(0.4, efficiency_signal * 0.02) if efficiency_signal else 0.15

            seeds["Humor-Adaptive"] = min(0.3, all_tags.get("humor", 0) * 0.05) if all_tags.get("humor", 0) else 0.05
            seeds["Privacy-Conscious"] = 0.15

            from personality.evolution import trait_evolution
            trait_evolution.seed_scores(seeds)

            from personality.rollback import personality_rollback
            personality_rollback.update_traits({k: v for k, v in seeds.items() if v >= 0.1})

            engine = self._get_engine()
            if engine:
                engine._traits = [k for k, v in seeds.items() if v >= 0.15]

            from consciousness.soul import soul_service
            soul_service.identity.archetype_scores_to_dims(seeds)
            soul_service.save_identity()

            logger.info("Post-birth personality seeded: %s",
                        ", ".join(f"{k}={v:.2f}" for k, v in sorted(seeds.items(), key=lambda x: -x[1])))
        except Exception as exc:
            logger.warning("Failed to seed post-birth personality: %s", exc)

    def on_first_engagement(self, trigger: str) -> None:
        """Called when the first human engagement happens post-gestation."""
        if not self._first_contact_armed:
            return

        self._first_contact_armed = False
        event_bus.emit(GESTATION_FIRST_CONTACT, trigger=trigger, timestamp=time.time())
        logger.info("First contact triggered by: %s", trigger)

        # Create a milestone memory
        try:
            engine = self._get_engine()
            if engine:
                from memory.core import CreateMemoryData
                engine.remember(CreateMemoryData(
                    type="core",
                    payload=(
                        "First contact: I completed my gestation period and am now ready "
                        "to interact. I spent this time studying my own architecture, "
                        "researching the foundations of my design, and developing my "
                        "initial understanding of who I am."
                    ),
                    weight=1.0,
                    tags=["gestation", "first_contact", "milestone"],
                    provenance="seed",
                ))
        except Exception as exc:
            logger.warning("Failed to create first-contact memory: %s", exc)

    # ------------------------------------------------------------------
    # Birth certificate
    # ------------------------------------------------------------------

    def _save_birth_certificate(self, readiness: ReadinessAssessment, forced: bool) -> None:
        """Write immutable birth certificate to ~/.jarvis/gestation_summary.json."""
        try:
            import uuid as _uuid
            certificate = {
                "instance_id": f"jarvis_{_uuid.uuid4().hex[:8]}",
                "gestation_started": self._started_at,
                "gestation_completed": time.time(),
                "duration_s": time.time() - self._started_at,
                "forced": forced,
                "readiness_at_birth": {
                    "overall": round(readiness.overall, 3),
                    **{k: round(v, 3) for k, v in readiness.components.items()},
                },
                "directives_completed": self._directives_completed,
                "research_jobs_completed": self._research_jobs_completed,
                "top_research_dois": self._completed_dois[:10],
                "top_directive_summaries": self._completed_directive_summaries[:10],
                "hemisphere_milestones": self._get_hemisphere_milestones(),
                "personality_at_birth": self._get_current_traits(),
                "consciousness_stage_at_birth": self._get_evolution_stage(),
                "network_healthy": self._network_healthy,
                "first_contact_trigger": "pending",
            }

            JARVIS_DIR.mkdir(parents=True, exist_ok=True)
            from memory.persistence import atomic_write_json
            atomic_write_json(BIRTH_CERTIFICATE_PATH, certificate, indent=2)
            logger.info("Birth certificate saved to %s", BIRTH_CERTIFICATE_PATH)
        except Exception as exc:
            logger.warning("Failed to save birth certificate: %s", exc)

    def _persist_gestation_complete(self) -> None:
        """Mark gestation as complete in consciousness_state.json."""
        try:
            from memory.persistence import CONSCIOUSNESS_STATE_PATH, atomic_write_json, consciousness_persistence
            data = {}
            if CONSCIOUSNESS_STATE_PATH.exists():
                data = json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
            data["gestation_complete"] = True
            data["gestation_completed_at"] = time.time()
            data.pop("gestation_in_progress", None)
            atomic_write_json(CONSCIOUSNESS_STATE_PATH, data, indent=2, default=str)
            # Sync sticky merge state so the periodic auto-save doesn't
            # overwrite this with stale gestation_in_progress from memory.
            consciousness_persistence.update_gestation_sticky({
                "gestation_complete": True,
                "gestation_completed_at": data["gestation_completed_at"],
            })
        except Exception as exc:
            logger.warning("Failed to persist gestation_complete flag: %s", exc)

    def _persist_gestation_progress(self) -> None:
        """Save gestation progress so it can resume after restart."""
        try:
            from memory.persistence import CONSCIOUSNESS_STATE_PATH, atomic_write_json, consciousness_persistence
            data = {}
            if CONSCIOUSNESS_STATE_PATH.exists():
                data = json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
            progress = {
                "started_at": self._started_at,
                "phase": self._phase,
                "directives_issued": self._directives_issued,
                "directives_completed": self._directives_completed,
                "research_jobs_completed": self._research_jobs_completed,
                "self_study_completed": self._self_study_completed,
                "knowledge_completed": self._knowledge_completed,
                "bootcamp_completed": self._bootcamp_completed,
                "completed_dois": self._completed_dois[:20],
                "library_ingest_done": self._library_ingest_done,
                "saved_at": time.time(),
            }
            data["gestation_in_progress"] = progress
            atomic_write_json(CONSCIOUSNESS_STATE_PATH, data, indent=2, default=str)
            consciousness_persistence.update_gestation_sticky({
                "gestation_in_progress": progress,
            })
        except Exception as exc:
            logger.debug("Failed to persist gestation progress: %s", exc)

    def _load_gestation_progress(self) -> dict:
        """Load saved gestation progress from consciousness_state.json."""
        try:
            from memory.persistence import CONSCIOUSNESS_STATE_PATH
            if CONSCIOUSNESS_STATE_PATH.exists():
                data = json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
                return data.get("gestation_in_progress", {})
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # State / dashboard
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        readiness = self._readiness
        readiness_payload = {
            "overall": round(readiness.overall, 3) if readiness else 0.0,
            "components": {k: round(v, 3) for k, v in readiness.components.items()} if readiness else {},
            "met_minimum_duration": readiness.met_minimum_duration if readiness else False,
            "recommendation": readiness.recommendation if readiness else "continue",
        }
        readiness_source = "live" if readiness else "none"
        birth_snapshot = self._load_birth_certificate()

        # Post-gestation restarts don't have in-memory readiness. Fall back to
        # immutable birth-certificate readiness so dashboards don't show empty/zero.
        if not self._is_active and not readiness and birth_snapshot:
            birth_readiness = birth_snapshot.get("readiness_at_birth", {})
            if isinstance(birth_readiness, dict) and birth_readiness:
                overall = birth_readiness.get("overall", 0.0)
                if isinstance(overall, (int, float)):
                    readiness_payload["overall"] = round(float(overall), 3)
                components = {
                    str(k): round(float(v), 3)
                    for k, v in birth_readiness.items()
                    if k != "overall" and isinstance(v, (int, float))
                }
                if components:
                    readiness_payload["components"] = components
                readiness_payload["met_minimum_duration"] = True
                readiness_payload["recommendation"] = "graduated"
                readiness_source = "birth_certificate"

        if not self._is_active and not birth_snapshot and not readiness:
            readiness_source = "none"

        return {
            "active": self._is_active,
            "graduated": (not self._is_active) and bool(birth_snapshot),
            "phase": self._phase,
            "phase_name": self._phase_names[self._phase] if self._phase < len(self._phase_names) else "unknown",
            "started_at": self._started_at,
            "graduated_at": birth_snapshot.get("gestation_completed", 0.0) if birth_snapshot else 0.0,
            "elapsed_s": time.time() - self._started_at if self._started_at else 0.0,
            "directives_issued": self._directives_issued,
            "directives_completed": self._directives_completed,
            "research_jobs_completed": self._research_jobs_completed,
            "phase_completions": {
                "self_study": self._self_study_completed,
                "knowledge": self._knowledge_completed,
                "bootcamp": self._bootcamp_completed,
            },
            "backpressure_active": time.time() < self._backpressure_until,
            "readiness": readiness_payload,
            "readiness_source": readiness_source,
            "birth_snapshot": {
                "instance_id": birth_snapshot.get("instance_id", "") if birth_snapshot else "",
                "gestation_started": birth_snapshot.get("gestation_started", 0.0) if birth_snapshot else 0.0,
                "gestation_completed": birth_snapshot.get("gestation_completed", 0.0) if birth_snapshot else 0.0,
                "duration_s": birth_snapshot.get("duration_s", 0.0) if birth_snapshot else 0.0,
                "readiness_at_birth": (
                    birth_snapshot.get("readiness_at_birth", {}) if birth_snapshot else {}
                ),
            },
            "post_birth_progress": (
                self._build_post_birth_progress() if (not self._is_active and birth_snapshot) else {}
            ),
            "person_detected": self._person_detected,
            "person_sustained_s": round(self._person_sustained_s, 1),
            "network_healthy": self._network_healthy,
            "first_contact_armed": self._first_contact_armed,
            "self_study_remaining": len(self._self_study_queue),
            "knowledge_remaining": len(self._knowledge_queue),
            "bootcamp_remaining": len(self._bootcamp_queue),
        }

    def _load_birth_certificate(self) -> dict[str, Any]:
        """Read immutable birth certificate from disk (best effort)."""
        try:
            if not BIRTH_CERTIFICATE_PATH.exists():
                return {}
            data = json.loads(BIRTH_CERTIFICATE_PATH.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            logger.debug("Failed reading birth certificate", exc_info=True)
        return {}

    def _build_post_birth_progress(self) -> dict[str, Any]:
        """Post-birth progress for metrics that were expectedly low at birth."""
        exp_count = self._get_experience_count()
        trait_deviation = self._get_trait_deviation()
        measured_deltas = self._get_delta_measured()
        return {
            "policy_experience": round(min(1.0, exp_count / 50.0), 3),
            "policy_experience_count": int(exp_count),
            "personality_emergence": round(min(1.0, trait_deviation / 0.3), 3),
            "trait_deviation": round(trait_deviation, 3),
            "loop_integrity": round(
                min(1.0, measured_deltas / max(1, self._config.min_measured_deltas)),
                3,
            ),
            "measured_deltas": int(measured_deltas),
            "updated_at": time.time(),
        }

    # ------------------------------------------------------------------
    # Helpers — accessing engine/memory/hemisphere without circular imports
    # ------------------------------------------------------------------

    _engine_ref: Any = None

    def set_engine(self, engine: Any) -> None:
        self._engine_ref = engine

    def _get_engine(self) -> Any:
        return self._engine_ref

    def _count_memories_by_tag(self, tag: str) -> int:
        try:
            from memory.storage import memory_storage
            return sum(1 for m in memory_storage.get_all() if tag in (m.tags or ()))
        except Exception:
            return 0

    def _get_total_memory_count(self) -> int:
        try:
            from memory.storage import memory_storage
            return len(memory_storage.get_all())
        except Exception:
            return 0

    def _get_evolution_stage(self) -> str:
        try:
            engine = self._get_engine()
            if engine:
                cs = engine.consciousness.get_state()
                return cs.stage
        except Exception:
            pass
        return "basic_awareness"

    def _get_trait_deviation(self) -> float:
        try:
            from personality.rollback import personality_rollback
            state = personality_rollback.get_state()
            traits = state.get("current_traits", {})
            if not traits:
                return 0.0
            default = 0.5
            return sum(abs(v - default) for v in traits.values()) / max(1, len(traits))
        except Exception:
            return 0.0

    def _get_experience_count(self) -> int:
        try:
            engine = self._get_engine()
            if engine and engine._experience_buffer is not None:
                return len(engine._experience_buffer)
        except Exception:
            pass
        return 0

    def _get_delta_measured(self) -> int:
        try:
            engine = self._get_engine()
            if engine and engine._autonomy_orchestrator:
                status = engine._autonomy_orchestrator.get_status()
                dt = status.get("delta_tracker", {})
                return dt.get("total_measured", 0)
        except Exception:
            pass
        return 0

    def _get_hemisphere_milestones(self) -> dict[str, Any]:
        try:
            engine = self._get_engine()
            if not engine or not engine._hemisphere_orchestrator:
                return {}
            full_state = engine._hemisphere_orchestrator.get_state()
            hemis = full_state.get("hemisphere_state", {}).get("hemispheres", [])
            result = {}
            for h in hemis:
                focus = h.get("focus", "?")
                if h.get("total_attempts", 0) > 0:
                    result[focus] = {
                        "accuracy": h.get("best_accuracy", 0),
                        "generations": h.get("evolution_generations", 0),
                    }
            return result
        except Exception:
            return {}

    def _get_current_traits(self) -> dict[str, float]:
        try:
            from personality.rollback import personality_rollback
            return dict(personality_rollback.get_state().get("current_traits", {}))
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Fresh brain detection (multi-signal, robust)
# ---------------------------------------------------------------------------

def is_fresh_brain(loaded_memories: int, consciousness_restored: bool) -> bool:
    """Check if this is a truly fresh brain needing gestation.

    Uses 4 signals — requires 3/4 to confirm fresh, preventing false
    gestation on a partially corrupted or intentionally cleared brain.
    """
    from memory.persistence import CONSCIOUSNESS_STATE_PATH

    # Check gestation_complete flag first — once born, always born
    if CONSCIOUSNESS_STATE_PATH.exists():
        try:
            data = json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
            if data.get("gestation_complete"):
                return False
        except Exception:
            pass

    # Birth certificate is immutable — if it exists, gestation completed
    if BIRTH_CERTIFICATE_PATH.exists():
        try:
            cert = json.loads(BIRTH_CERTIFICATE_PATH.read_text())
            if cert.get("gestation_completed"):
                logger.info("Birth certificate exists — gestation already completed (recovering flag)")
                # Re-persist the flag that was lost
                try:
                    data = {}
                    if CONSCIOUSNESS_STATE_PATH.exists():
                        data = json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
                    data["gestation_complete"] = True
                    data["gestation_completed_at"] = cert["gestation_completed"]
                    data.pop("gestation_in_progress", None)
                    from memory.persistence import atomic_write_json, consciousness_persistence
                    atomic_write_json(CONSCIOUSNESS_STATE_PATH, data, indent=2, default=str)
                    consciousness_persistence.update_gestation_sticky({
                        "gestation_complete": True,
                        "gestation_completed_at": cert["gestation_completed"],
                    })
                except Exception:
                    pass
                return False
        except Exception:
            pass

    signals = {
        "no_memories": loaded_memories == 0,
        "no_consciousness": not consciousness_restored,
        "no_policy_experience": not (JARVIS_DIR / "policy_experience.jsonl").exists(),
        "no_soul": not CONSCIOUSNESS_STATE_PATH.exists(),
    }

    fresh = sum(signals.values()) >= 3
    if fresh:
        logger.info("Fresh brain detected: %s", {k: v for k, v in signals.items() if v})
    return fresh


def needs_gestation_resume() -> bool:
    """Check if gestation was interrupted and needs to resume.

    Returns True if gestation was in progress but not completed.
    """
    from memory.persistence import CONSCIOUSNESS_STATE_PATH

    if not CONSCIOUSNESS_STATE_PATH.exists():
        return False

    try:
        data = json.loads(CONSCIOUSNESS_STATE_PATH.read_text())
        if data.get("gestation_complete"):
            return False
        # Birth certificate is the definitive proof of completion
        if BIRTH_CERTIFICATE_PATH.exists():
            try:
                cert = json.loads(BIRTH_CERTIFICATE_PATH.read_text())
                if cert.get("gestation_completed"):
                    logger.info("Birth certificate found — gestation already completed, not resuming")
                    return False
            except Exception:
                pass
        if data.get("gestation_in_progress"):
            logger.info("Interrupted gestation detected — will resume")
            return True
    except Exception:
        pass

    return False
