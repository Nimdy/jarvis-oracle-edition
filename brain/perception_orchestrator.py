"""Perception orchestrator — sets up all sensor processing, attention, and mode wiring."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Any

from config import BrainConfig
from consciousness.engine import ConsciousnessEngine
from consciousness.events import (
    event_bus,
    PERCEPTION_SPEAKER_IDENTIFIED,
    PERCEPTION_USER_EMOTION,
    PERCEPTION_FACE_IDENTIFIED,
    PERCEPTION_TRANSCRIPTION,
    PERCEPTION_RAW_AUDIO,
    PERCEPTION_SCENE_SUMMARY,
    PERCEPTION_USER_PRESENT_STABLE,
    CONVERSATION_RESPONSE,
    PERCEPTION_PLAYBACK_COMPLETE,
    PERCEPTION_SENSOR_DISCONNECTED,
    SYNTHETIC_EXERCISE_STATE,
    SPATIAL_TRACK_UPDATED,
    SPATIAL_DELTA_PROMOTED,
    SPATIAL_CALIBRATION_CHANGED,
    MEMORY_WRITE,
)
from consciousness.modes import mode_manager, MODE_CHANGE
from consciousness.release_validation import output_release_validator
from consciousness.trace_context import build_trace_context
from perception.server import PerceptionServer
from perception.presence import PresenceTracker, PRESENCE_USER_ARRIVED
from perception.screen import ScreenProcessor
from perception.audio import AudioProcessor
from perception.vision import VisionProcessor
from perception.ambient import AmbientAudioProcessor
from perception.attention import AttentionCore, ATTENTION_UPDATE
from perception.audio_stream import AudioStreamProcessor
from reasoning.response import ResponseGenerator
from reasoning.claude_client import ClaudeClient
from memory.episodes import EpisodicMemory
from conversation_handler import handle_transcription
from perception.stt import LaptopSTT
from personality.proactive import proactive_behavior, ProactiveSuggestion
from perception.emotion import AudioEmotionClassifier
from perception.speaker_id import SpeakerIdentifier
from perception.trait_perception import trait_perception
from perception.identity_fusion import IdentityFusion, IDENTITY_RESOLVED
from perception.scene_tracker import SceneTracker
from perception.scene_types import SceneDetection
from perception.display_classifier import DisplayClassifier
from perception.calibration import CalibrationManager
from perception.spatial import SpatialEstimator
from cognition.spatial_fusion import SpatialFusion
from cognition.spatial_validation import SpatialValidator
from reasoning.tts import BrainTTS
from reasoning.voice_policy import OracleVoicePolicy, VoicePolicyConfig

logger = logging.getLogger("jarvis.perception")


# Bounded scalar keys allowed in the consciousness-feed `scene` block.
# This is the contract enforced by ``test_consciousness_feed_scene_block``:
# any addition / removal must happen here in a single place and is also
# the wire format the Pi particle visualizer reads. **No** entity or relation
# arrays. **No** raw vectors. **No** authority flags (those stay brain-side).
_SCENE_FEED_KEYS: tuple[str, ...] = (
    "enabled",
    "entity_count",
    "relation_count",
    "cleanup_accuracy",
    "relation_recovery",
    "similarity_to_previous",
    "spatial_hrr_side_effects",
)


def _build_scene_block() -> dict[str, Any]:
    """Compact, vector-free, authority-free scene scalars for the Pi feed.

    Reads the in-process ``cognition.mental_world.get_state()`` snapshot
    (already vector-stripped and authority-pinned upstream) and projects
    it down to the seven scalars in :data:`_SCENE_FEED_KEYS`. Always
    returns the full key set so the wire format is stable; missing /
    unavailable values default to ``0`` (or ``False`` for ``enabled``).
    """
    block: dict[str, Any] = {
        "enabled": False,
        "entity_count": 0,
        "relation_count": 0,
        "cleanup_accuracy": 0.0,
        "relation_recovery": 0.0,
        "similarity_to_previous": 0.0,
        "spatial_hrr_side_effects": 0,
    }
    try:
        from cognition.mental_world import get_state as _scene_state
        state = _scene_state()
    except Exception:
        return block
    if not isinstance(state, dict):
        return block

    block["enabled"] = bool(state.get("enabled", False))
    block["entity_count"] = int(
        state.get("active_entity_count", state.get("entity_count") or 0) or 0
    )
    block["relation_count"] = int(state.get("relation_count") or 0)

    metrics = state.get("metrics") or {}
    if isinstance(metrics, dict):
        for src_key, dst_key in (
            ("cleanup_accuracy", "cleanup_accuracy"),
            ("relation_recovery", "relation_recovery"),
            ("similarity_to_previous", "similarity_to_previous"),
        ):
            v = metrics.get(src_key)
            if isinstance(v, (int, float)):
                block[dst_key] = round(float(v), 3)
        side_effects = metrics.get("spatial_hrr_side_effects")
        if isinstance(side_effects, (int, float)):
            block["spatial_hrr_side_effects"] = int(side_effects)
    return block


class PerceptionOrchestrator:
    """Encapsulates all perception wiring so main.py stays clean."""

    def __init__(
        self,
        engine: ConsciousnessEngine,
        response_gen: ResponseGenerator,
        claude: ClaudeClient | None,
        config: BrainConfig,
        episodes: EpisodicMemory | None = None,
    ) -> None:
        self.engine = engine
        self.response_gen = response_gen
        self.claude = claude
        self.config = config
        self.episodes = episodes

        self.perception: PerceptionServer | None = None
        self.presence: PresenceTracker | None = None
        self.audio_proc: AudioProcessor | None = None
        self.vision_proc: VisionProcessor | None = None
        self.ambient_proc: AmbientAudioProcessor | None = None
        self.screen_proc: ScreenProcessor | None = None
        self.attention: AttentionCore | None = None
        self.stt: LaptopSTT | None = None
        self.audio_stream: AudioStreamProcessor | None = None

        self._ollama = response_gen.ollama
        pi_host = config.perception.pi_host
        pi_port = config.perception.pi_ui_port
        self._pi_snapshot_url = f"http://{pi_host}:{pi_port}/snapshot" if pi_host else ""

        self._current_speaker: dict[str, str] = {"name": "unknown"}
        self._current_emotion: dict[str, str] = {"emotion": "neutral"}
        self._current_face: dict[str, Any] = {"name": "unknown", "confidence": 0.0, "is_known": False}
        self._speakers_seen_this_session: set[str] = set()
        self._previous_speaker_name: str = "unknown"
        self.identity_fusion = IdentityFusion()
        # Replace this dict per conversation so in-flight handlers keep their
        # own cancel state instead of sharing one mutable object globally.
        self._active_conversation: dict[str, Any] = {"id": "", "cancelled": False}
        self._conv_lock = threading.Lock()
        self._latest_stt_conv_id: str = ""
        self._last_response_text: str = ""
        self._last_response_confidence: float | None = None
        self._last_response_set_time: float = 0.0
        self._playback_complete_time: float = 0.0
        self._last_gesture_response_time: float = 0.0
        self._gesture_cooldown_s: float = 300.0
        self._wave_consecutive_count: int = 0
        self._wave_required_consecutive: int = 5
        self._wave_first_seen_time: float = 0.0
        self._wave_min_duration_s: float = 1.5

        self._speaking_safety_timer: threading.Timer | None = None
        self._SPEAKING_SAFETY_TIMEOUT_S = 60.0
        self._speaking_conv_id: str = ""

        # Unknown speaker tracking for curiosity system
        self._unknown_speaker_events: deque[dict[str, Any]] = deque(maxlen=10)
        self._UNKNOWN_SPEAKER_COOLDOWN_S = 120.0
        self._last_unknown_speaker_at: float = 0.0
        self._stt_lock = asyncio.Lock()

        from perception.addressee import AddresseeGate
        self._addressee_gate = AddresseeGate()

        self._last_speech_audio: Any = None
        self._last_speech_sr: int = 16000
        self._recent_speech_clips: list[Any] = []
        # Enrollment density directly drives speaker_id measurement accuracy.
        # The buffer is filled by every inbound speech chunk; when the user says
        # "my name is X" the current contents are averaged into the ECAPA
        # centroid stored in speakers.json. A thin buffer (N=5) collapses voice
        # variance into a single point; a denser buffer (N=20) gives ~2x lower
        # standard error on the centroid and a richer anchor set for the
        # multi-anchor upgrade. Memory cost at 20 × ~3s × 16kHz × float32 is
        # ~3.8MB max, well within budget.
        self._MAX_ENROLLMENT_CLIPS = 20

        tts_cfg = config.tts
        tts_device = "cuda" if tts_cfg.engine == "kokoro_gpu" else "cpu"
        voice_policy = OracleVoicePolicy(VoicePolicyConfig(
            base_voice=tts_cfg.voice,
            base_speed=tts_cfg.speed,
            solemn_voice=tts_cfg.oracle_solemn_voice or tts_cfg.voice,
            empathetic_voice=tts_cfg.oracle_empathetic_voice or tts_cfg.oracle_solemn_voice or tts_cfg.voice,
            urgent_voice=tts_cfg.oracle_urgent_voice or tts_cfg.oracle_solemn_voice or tts_cfg.voice,
            observational_voice=tts_cfg.oracle_observational_voice or tts_cfg.oracle_solemn_voice or tts_cfg.voice,
            guarded_voice=tts_cfg.oracle_guarded_voice or tts_cfg.oracle_solemn_voice or tts_cfg.voice,
        ))
        self.brain_tts = BrainTTS(
            model_path=tts_cfg.model_path,
            voice=tts_cfg.voice,
            speed=tts_cfg.speed,
            device=tts_device,
            voice_policy=voice_policy,
        ) if tts_cfg.engine != "none" else BrainTTS()

        self.speaker_id = SpeakerIdentifier(
            device=config.speaker_id.device,
        ) if config.speaker_id.enabled else None

        self.emotion_classifier = AudioEmotionClassifier(
            device=config.emotion.device,
        ) if config.emotion.enabled else AudioEmotionClassifier()

        self.face_id = None
        try:
            from perception.face_id import FaceIdentifier
            self.face_id = FaceIdentifier(device=config.speaker_id.device)
            if not self.face_id.available:
                logger.info("Face ID model not available — face identification disabled")
                self.face_id = None
        except Exception as exc:
            logger.info("Face ID initialization skipped: %s", exc)

        self._last_scene_analysis_time: float = 0.0
        self._scene_interval_away: float = 300.0
        self._scene_interval_present: float = 1800.0
        self._object_memory: dict[str, dict] = {}
        self._last_scene_description: str = ""
        self._scene_analysis_in_progress: bool = False
        self._gestation_active: bool = False

        self._synthetic_sources: set[str] = set()
        self._synthetic_cooldown_until: float = 0.0
        self._current_conv_synthetic: bool = False
        self._synthetic_ledger: dict[str, Any] = {
            "utterances_stt": 0,
            "routes_produced": 0,
            "distillation_records": 0,
            "blocked_side_effects": 0,
            "route_histogram": {},
            "stt_texts": [],
            "recent_route_examples": [],
            "llm_leaks": 0,
            "tts_leaks": 0,
            "transcription_emit_leaks": 0,
            "memory_side_effects": 0,
            "identity_side_effects": 0,
            "total_runs": 0,
            "last_run_time": 0.0,
        }

        self._scene_tracker = SceneTracker()
        self._display_classifier = DisplayClassifier()
        self._last_scene_snapshot: Any = None

        self._calibration_manager = CalibrationManager()
        self._calibration_manager.setup_pi_camera_defaults(
            frame_width=640, frame_height=480,
            camera_position_m=(0.0, 1.2, 0.0),
        )
        self._spatial_estimator = SpatialEstimator(self._calibration_manager)
        self._spatial_validator = SpatialValidator()
        self._spatial_fusion = SpatialFusion()
        self._spatial_fused: dict[str, Any] = {}
        self._spatial_update_count: int = 0
        self._spatial_scene_signature: dict[str, Any] = {}
        self._spatial_relocalization_threshold: float = 0.55
        self._spatial_relocalization_interval_s: float = 20.0
        self._spatial_profile_sync_interval_s: float = 60.0
        self._last_spatial_relocalization_ts: float = 0.0
        self._last_spatial_profile_sync_ts: float = 0.0
        self._last_spatial_relocalization: dict[str, Any] = {}
        self._spatial_last_scene_change_score: float = 0.0
        self._spatial_post_handoff_stable_windows: int = 0
        self._spatial_post_handoff_verify_min_stable_tracks: int = 2
        self._spatial_post_handoff_verify_required_windows: int = 3
        self._spatial_post_handoff_verify_max_scene_change: float = 0.25

    async def start(self) -> None:
        engine = self.engine
        cfg = self.config
        self._loop = asyncio.get_running_loop()

        event_bus.on(PERCEPTION_SPEAKER_IDENTIFIED, self._on_speaker_id)
        event_bus.on(PERCEPTION_USER_EMOTION, self._on_emotion)
        event_bus.on(PERCEPTION_FACE_IDENTIFIED, self._on_face_id)
        event_bus.on(IDENTITY_RESOLVED, self._on_identity_resolved)
        event_bus.on(PERCEPTION_SCENE_SUMMARY, self._on_scene_summary)
        self.identity_fusion.start()

        try:
            from identity.resolver import identity_resolver
            identity_resolver.set_fusion(self.identity_fusion)
            if hasattr(engine, '_soul') and engine._soul:
                identity_resolver.set_soul(engine._soul)
            logger.info("Identity resolver wired to fusion + soul")
        except Exception as exc:
            logger.warning("Identity resolver wiring failed: %s", exc)

        if not cfg.enable_perception:
            return

        self.perception = PerceptionServer(host=cfg.perception.host, port=cfg.perception.port)
        if self.face_id:
            self.perception.set_face_identifier(self.face_id)
        await self.perception.start()

        def _proactive_speak(text: str) -> None:
            if self._gestation_active:
                return
            try:
                from skills.capability_gate import capability_gate
                text = capability_gate.check_text(text)
            except Exception:
                pass
            self._speak_proactive(text)
        engine.set_proactive_speech_callback(_proactive_speak)
        engine.set_proactive_evaluator(self.evaluate_proactive)

        self.presence = PresenceTracker(engine)
        self.presence.start()

        self.audio_proc = AudioProcessor(engine)
        self.audio_proc.start()

        self.vision_proc = VisionProcessor(engine)
        self.vision_proc.start()

        self.ambient_proc = AmbientAudioProcessor(engine)
        self.ambient_proc.start()

        self.attention = AttentionCore()
        self.attention.start()
        engine.set_attention_core(self.attention)

        event_bus.on(ATTENTION_UPDATE, self._on_attention_update)
        event_bus.on(MODE_CHANGE, self._on_mode_change)
        event_bus.on(CONVERSATION_RESPONSE, self._on_conversation_response)
        event_bus.on(PERCEPTION_PLAYBACK_COMPLETE, self._on_playback_complete)
        event_bus.on(PERCEPTION_SENSOR_DISCONNECTED, lambda **_: self._on_sensor_disconnect())
        event_bus.on(PERCEPTION_TRANSCRIPTION, self._on_transcription)
        event_bus.on(PRESENCE_USER_ARRIVED, self._on_user_arrived)
        event_bus.on(PERCEPTION_USER_PRESENT_STABLE, self._on_presence_stable)

        traits_list = engine.get_state().get("traits", [])
        if isinstance(traits_list, list) and traits_list:
            trait_perception.set_traits({t: 1.0 for t in traits_list})
        elif isinstance(traits_list, dict):
            trait_perception.set_traits(traits_list)

        # --- STT ---
        stt_cfg = getattr(cfg, "stt", None)
        if stt_cfg:
            stt_model = stt_cfg.model
            stt_compute = stt_cfg.compute_type
            stt_device = stt_cfg.device
        else:
            stt_model = getattr(cfg, "stt_model", "large-v3-turbo")
            stt_compute = "int8_float16"
            stt_device = "cuda"
        self.stt = LaptopSTT(
            model_size=stt_model, device=stt_device, compute_type=stt_compute,
        )
        if self.stt.available:
            print(f"  LaptopSTT: ready ({stt_model} on {stt_device}, {stt_compute})")
        else:
            print("  LaptopSTT: unavailable")

        # --- Audio stream processor (wake word + VAD on brain) ---
        ww = cfg.wake_word
        self.audio_stream = AudioStreamProcessor(
            keyword=ww.keyword,
            threshold=ww.threshold,
            speaking_threshold_mult=ww.speaking_threshold_mult,
            speaking_hits_required=ww.speaking_hits_required,
            cooldown_s=ww.cooldown_s,
            silence_duration_s=ww.silence_duration_s,
            max_record_s=ww.max_record_s,
            follow_up_timeout_s=ww.follow_up_timeout_s,
            on_wake=self._on_wake_word_detected,
            on_speech_ready=self._on_speech_ready,
            on_barge_in=self._on_barge_in,
        )
        self.audio_stream.start()
        event_bus.on(PERCEPTION_RAW_AUDIO, self._on_raw_audio)
        event_bus.on(SYNTHETIC_EXERCISE_STATE, self._on_synthetic_exercise_state)

        # Invariant leak watchers: observer-only counters that fire when an event
        # known to violate the synthetic truth boundary (LLM response, TTS, memory
        # write, identity record, post-hard-stop transcription) is emitted while a
        # synthetic exercise session is active. Fails open — never blocks emitters.
        # See docs/AWAKENING_PROTOCOL.md + Pillar 9 (Observability and Proof).
        event_bus.on(PERCEPTION_TRANSCRIPTION, self._synthetic_leak_transcription)
        event_bus.on(MEMORY_WRITE, self._synthetic_leak_memory)
        event_bus.on(IDENTITY_RESOLVED, self._synthetic_leak_identity)
        event_bus.on(PERCEPTION_SPEAKER_IDENTIFIED, self._synthetic_leak_identity)
        event_bus.on(PERCEPTION_FACE_IDENTIFIED, self._synthetic_leak_identity)
        event_bus.on(CONVERSATION_RESPONSE, self._synthetic_leak_llm)
        print(f"  AudioStream: wake_word={ww.keyword} threshold={ww.threshold}")

        print("  AttentionCore: active")
        print(f"  Mode: {mode_manager.mode}")

        if cfg.enable_screen_awareness:
            self.screen_proc = ScreenProcessor()
            self.screen_proc.set_engine(engine)
            self.screen_proc.start()

        if self._pi_snapshot_url:
            asyncio.get_event_loop().create_task(self._periodic_scene_analysis())

        asyncio.get_event_loop().create_task(self._consciousness_feed_loop())

    def set_gestation_active(self, active: bool) -> None:
        """Enable/disable gestation gating (wake word disarm, greeting suppress)."""
        self._gestation_active = active
        if active and self.audio_stream:
            self.audio_stream.set_wake_armed(False)
        elif not active and self.audio_stream:
            self.audio_stream.set_wake_armed(True)
        logger.info("Perception gestation gating: %s", "active" if active else "inactive")

    async def stop(self) -> None:
        if self.audio_stream:
            self.audio_stream.stop()
        for proc in (self.presence, self.audio_proc, self.vision_proc,
                     self.ambient_proc, self.screen_proc):
            if proc:
                proc.stop()
        if self.perception:
            await self.perception.stop()

    def get_processors(self) -> dict[str, Any]:
        return {
            "perception": self.perception,
            "presence": self.presence,
            "audio": self.audio_proc,
            "vision": self.vision_proc,
            "ambient": self.ambient_proc,
            "screen": self.screen_proc,
        }

    # ------------------------------------------------------------------
    # Consciousness feed — periodic rich state to Pi for particle viz
    # ------------------------------------------------------------------

    async def _consciousness_feed_loop(self) -> None:
        """Broadcast consciousness state to Pi every 2s for particle visualization."""
        while True:
            try:
                await asyncio.sleep(2.0)
                if not self.perception or not self.perception.any_sensor_connected:
                    continue
                feed = self._build_consciousness_feed()
                if feed:
                    self.perception.broadcast({
                        "type": "consciousness_feed",
                        "data": feed,
                    })
            except Exception:
                logger.debug("Consciousness feed error", exc_info=True)

    def _build_consciousness_feed(self) -> dict[str, Any]:
        """Build a compact consciousness snapshot for the Pi particle visualizer."""
        engine = self.engine
        cs = engine.consciousness
        state = engine.get_state()

        cs_state = cs.get_state()
        analytics = cs.analytics
        confidence = analytics.get_confidence()
        reasoning = analytics.get_reasoning_quality()
        health = analytics.get_system_health()

        thoughts = cs.meta_thoughts.get_recent_thoughts(3)
        thought_list = [
            {"type": t.thought_type, "depth": t.depth, "text": t.text[:80]}
            for t in thoughts
        ]

        hemisphere_signals = cs.get_hemisphere_signals()

        hemi_state = cs.get_hemisphere_state()
        hemi_summary = {}
        if hemi_state:
            models = hemi_state.get("models", {})
            for focus, info in models.items():
                hemi_summary[focus] = {
                    "accuracy": info.get("best_accuracy", 0),
                    "generation": info.get("generation", 0),
                    "training": info.get("training_active", False),
                }

        kernel_perf = engine._kernel.get_performance() if engine._kernel else None

        feed: dict[str, Any] = {
            "phase": state["phase"],
            "tone": state["tone"],
            "mode": state["mode"],
            "stage": cs_state.stage,
            "transcendence": round(cs_state.transcendence_level, 2),
            "awareness": round(cs_state.awareness_level, 2),
            "confidence": round(confidence.avg, 2),
            "reasoning": round(reasoning.overall, 2),
            "healthy": health.healthy,
            "mutation_count": cs_state.mutation_count,
            "observation_count": cs_state.observation_count,
            "emergent_count": cs_state.emergent_behavior_count,
            "memory_count": state.get("memory_count", 0),
            "memory_density": round(state.get("memory_density", 0), 2),
            "thoughts": thought_list,
            "focus": cs_state.current_focus[:60] if cs_state.current_focus else "",
            "hemisphere_signals": {k: round(v, 3) for k, v in hemisphere_signals.items()},
            "hemisphere": hemi_summary,
            "kernel_tick_ms": round(kernel_perf.avg_tick_ms, 1) if kernel_perf else 0,
            "kernel_tick": state.get("tick", 0),
            "traits": state.get("traits", [])[:5],
            "capabilities": cs_state.active_capabilities[:5],
            "scene": _build_scene_block(),
        }
        return feed

    # ------------------------------------------------------------------
    # Raw audio from Pi -> AudioStreamProcessor
    # ------------------------------------------------------------------

    def _on_raw_audio(self, pcm_bytes: bytes, **_):
        if self.audio_stream:
            self.audio_stream.feed(pcm_bytes)

    _SYNTHETIC_COOLDOWN_S = 15.0

    def _on_synthetic_exercise_state(self, active: bool, sensor_id: str = "", **_):
        """Session-level sticky flag set by synthetic_exercise_start/end control messages."""
        sid = sensor_id or "unknown"
        if active:
            self._synthetic_sources.add(sid)
            self._synthetic_cooldown_until = 0.0
        else:
            self._synthetic_sources.discard(sid)
            if not self._synthetic_sources:
                self._synthetic_cooldown_until = time.time() + self._SYNTHETIC_COOLDOWN_S
                self._synthetic_ledger["total_runs"] += 1
                self._synthetic_ledger["last_run_time"] = time.time()
        is_active = bool(self._synthetic_sources)
        if self.audio_stream:
            self.audio_stream.synthetic_active = is_active
        # Truth-boundary propagation: wire synthetic state to the authoritative
        # memory gate and to IdentityFusion's event gate. CueGate blocks
        # observation + consolidation writes during synthetic sessions; fusion
        # suppresses IDENTITY_RESOLVED emissions. Both are reversible on end.
        try:
            from memory.gate import memory_gate as _mg
            if active:
                _mg.begin_synthetic_session(sid)
            else:
                _mg.end_synthetic_session(sid)
        except Exception:
            logger.debug("memory_gate synthetic wiring failed", exc_info=True)
        try:
            if self.identity_fusion is not None:
                self.identity_fusion.set_synthetic_active(is_active)
        except Exception:
            logger.debug("identity_fusion synthetic wiring failed", exc_info=True)
        logger.info(
            "Synthetic exercise %s (sources=%s%s)",
            "ACTIVE" if is_active else "ended",
            self._synthetic_sources or "none",
            f", cooldown {self._SYNTHETIC_COOLDOWN_S}s" if not is_active and self._synthetic_cooldown_until > 0 else "",
        )

    def _record_synthetic_leak(self, counter_name: str, event_name: str, payload_keys: list[str]) -> None:
        """Increment an invariant leak counter and log a WARNING.

        Observer-only. Fails open — if ``_synthetic_sources`` is empty, nothing
        happens. Payload *keys* are logged (not values) to avoid leaking any
        audio/text/embedding content into log files.
        """
        if not self._synthetic_sources:
            return
        self._synthetic_ledger[counter_name] = int(self._synthetic_ledger.get(counter_name, 0)) + 1
        fail_reasons = self._synthetic_ledger.setdefault("fail_reasons", [])
        reason = {
            "counter": counter_name,
            "event": event_name,
            "payload_keys": sorted(payload_keys),
            "sources": sorted(self._synthetic_sources),
            "ts": time.time(),
        }
        fail_reasons.append(reason)
        if len(fail_reasons) > 50:
            del fail_reasons[:-50]
        logger.warning(
            "Synthetic truth boundary leak: counter=%s event=%s payload_keys=%s sources=%s",
            counter_name, event_name, sorted(payload_keys), sorted(self._synthetic_sources),
        )

    def _synthetic_leak_transcription(self, **kwargs: Any) -> None:
        self._record_synthetic_leak("transcription_emit_leaks", PERCEPTION_TRANSCRIPTION, list(kwargs.keys()))

    def _synthetic_leak_memory(self, **kwargs: Any) -> None:
        self._record_synthetic_leak("memory_side_effects", MEMORY_WRITE, list(kwargs.keys()))

    def _synthetic_leak_identity(self, **kwargs: Any) -> None:
        # Any of IDENTITY_RESOLVED / PERCEPTION_SPEAKER_IDENTIFIED /
        # PERCEPTION_FACE_IDENTIFIED counts as an identity side-effect during a
        # synthetic run. Event name is inferred from payload where possible; we
        # surface a generic label otherwise — the counter is the primary signal.
        event_label = str(kwargs.get("event") or "identity:event")
        self._record_synthetic_leak("identity_side_effects", event_label, list(kwargs.keys()))

    def _synthetic_leak_llm(self, **kwargs: Any) -> None:
        self._record_synthetic_leak("llm_leaks", CONVERSATION_RESPONSE, list(kwargs.keys()))

    def _synthetic_leak_tts(self, text_chars: int = 0) -> None:
        """Called from the TTS synthesis call sites (see _send_response paths).

        Fails open — only increments when a synthetic session is active. We pass
        ``text_chars`` for sanitized diagnostic context instead of the raw text.
        """
        if not self._synthetic_sources:
            return
        self._record_synthetic_leak("tts_leaks", "brain_tts:synthesize", [f"text_chars={text_chars}"])

    def _on_wake_word_detected(self):
        """Called from AudioStreamProcessor worker thread when wake word is detected."""
        self._cancel_speaking_safety_timer()
        from consciousness.operations import ops_tracker
        for _stale in ("retrieval", "reasoning", "speech", "stt"):
            ops_tracker.set_subsystem(_stale, "idle", "")
        ops_tracker.set_subsystem("perception", "listening", "wake word detected")
        ops_tracker.log_event("perception", "wake_detected", "Wake word detected")
        ops_tracker.advance_stage("wake", "done", "Wake word detected")
        ops_tracker.advance_stage("listen", "active", "Capturing speech")
        gm = self.engine._gestation_manager
        if gm and gm.first_contact_armed:
            gm.on_first_engagement("wake_word")
        logger.info("Wake word detected (brain-side)")
        try:
            from consciousness.event_validator import event_validator
            from consciousness.events import PERCEPTION_WAKE_WORD
            event_validator.record_event(PERCEPTION_WAKE_WORD)
            event_bus.emit(PERCEPTION_WAKE_WORD, source="brain")
        except Exception:
            pass
        modulation = trait_perception.process_event("wake_word", {})
        if modulation and modulation.thought_trigger:
            from consciousness.events import KERNEL_THOUGHT
            event_bus.emit(KERNEL_THOUGHT, content=modulation.thought_trigger, tone="contemplative")
        if self.perception:
            self.perception.broadcast({
                "type": "command",
                "data": {"action": "wake_detected"},
                "phase": "LISTENING",
            })

    def _on_speech_ready(self, audio_f32, conversation_id: str):
        """Called from AudioStreamProcessor worker thread with complete speech for STT."""
        in_cooldown = (
            not self._synthetic_sources
            and self._synthetic_cooldown_until > 0
            and time.time() < self._synthetic_cooldown_until
        )
        self._current_conv_synthetic = bool(self._synthetic_sources) or in_cooldown
        if in_cooldown:
            logger.info("Synthetic cooldown active — treating speech as synthetic (conv=%s)", conversation_id)
        from consciousness.operations import ops_tracker
        ops_tracker.set_subsystem("perception", "transcribing", "speech captured, running STT")
        ops_tracker.set_subsystem("stt", "transcribing", "Running STT")
        ops_tracker.log_event("perception", "speech_ready", f"Speech buffer ready ({len(audio_f32)/16000:.1f}s)")
        ops_tracker.advance_stage("listen", "done", "Speech captured")
        ops_tracker.advance_stage("stt", "active", "Transcribing speech")
        self._latest_stt_conv_id = conversation_id
        with self._conv_lock:
            self._active_conversation = {"id": conversation_id, "cancelled": False}

        if self.perception:
            self.perception.broadcast({
                "type": "command",
                "data": {"action": "thinking"},
                "conversation_id": conversation_id,
            })

        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(
                loop.create_task,
                self._transcribe_speech(audio_f32, conversation_id),
            )
        else:
            logger.error("No running event loop — cannot dispatch STT")

    def _on_barge_in(self, conversation_id: str):
        """Called from AudioStreamProcessor worker thread on barge-in detection."""
        from consciousness.operations import ops_tracker
        ops_tracker.log_event("perception", "barge_in", "User interrupted — cancelling response")
        ops_tracker.set_subsystem("speech", "idle", "barge-in cancelled playback")
        self._cancel_speaking_safety_timer()
        if self.audio_stream:
            self.audio_stream.set_speaking(False)
        self._speaking_conv_id = ""
        with self._conv_lock:
            if self._active_conversation.get("id") == conversation_id:
                self._active_conversation["cancelled"] = True
        try:
            from consciousness.event_validator import event_validator
            from consciousness.events import PERCEPTION_BARGE_IN
            event_validator.record_event(PERCEPTION_BARGE_IN)
            event_bus.emit(PERCEPTION_BARGE_IN, conversation_id=conversation_id)
        except Exception:
            pass
        modulation = trait_perception.process_event("barge_in", {"conversation_id": conversation_id})
        if modulation and modulation.thought_trigger:
            from consciousness.events import KERNEL_THOUGHT
            event_bus.emit(KERNEL_THOUGHT, content=modulation.thought_trigger, tone="contemplative")
        logger.info("Barge-in received — cancelling conv %s",
                     conversation_id[:8] if conversation_id else "?")
        if self.perception:
            self.perception.broadcast({
                "type": "command",
                "data": {"action": "barge_in"},
                "conversation_id": conversation_id,
            })

    # ------------------------------------------------------------------
    # STT pipeline
    # ------------------------------------------------------------------

    _VRAM_STT_HEADROOM_MB = 3500
    _VRAM_COEXIST_TIERS = frozenset({"premium", "ultra", "extreme"})

    async def _ensure_vram_for_stt(self) -> None:
        if not self._ollama:
            return
        import time as _t
        t0 = _t.monotonic()
        tier = getattr(self.config, "gpu_tier", "")
        try:
            if tier in self._VRAM_COEXIST_TIERS:
                unloaded = await self._ollama.unload_non_essential()
                if unloaded:
                    await asyncio.sleep(0.2)
                elapsed = (_t.monotonic() - t0) * 1000
                logger.info("VRAM prep (coexist): unloaded %d non-essential, "
                            "kept text LLM warm (%.0fms)", unloaded, elapsed)
            else:
                await self._ollama.unload_all()
                await asyncio.sleep(0.5)
                elapsed = (_t.monotonic() - t0) * 1000
                logger.info("VRAM prep (full clear): unloaded all (%.0fms)", elapsed)
        except Exception as exc:
            logger.warning("VRAM prep failed, falling back to full clear: %s", exc)
            try:
                await self._ollama.unload_all()
                await asyncio.sleep(0.5)
            except Exception:
                pass

    async def _release_vram_for_background(self, mode: str) -> None:
        """Release text LLM from VRAM when entering background modes."""
        try:
            await self._ollama.unload_all()
            logger.info("VRAM released for %s mode — text LLM unloaded", mode)
        except Exception as exc:
            logger.debug("VRAM release for %s failed (non-fatal): %s", mode, exc)

    async def _transcribe_speech(self, audio_f32, conversation_id: str) -> None:
        """Run STT on speech buffer dispatched by AudioStreamProcessor.

        Serialized via _stt_lock to prevent concurrent VRAM management and
        duplicate transcription events from overlapping wake detections.
        """
        async with self._stt_lock:
            await self._transcribe_speech_inner(audio_f32, conversation_id)

    _STT_TIMEOUT_S = 30.0

    async def _transcribe_speech_inner(self, audio_f32, conversation_id: str) -> None:
        import numpy as np
        import time as _t
        duration = len(audio_f32) / 16000
        _stt_wall_start = _t.monotonic()
        logger.info("Transcribing %.1fs speech (conv=%s)",
                     duration, conversation_id[:8] if conversation_id else "?")

        if not self.stt or not self.stt.available:
            logger.warning("Speech received but LaptopSTT unavailable")
            self._resume_listening(conversation_id)
            return

        await self._ensure_vram_for_stt()
        _stt_prep_ms = (_t.monotonic() - _stt_wall_start) * 1000

        text = None
        for attempt in range(2):
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(self.stt.transcribe, audio_f32, 16000),
                    timeout=self._STT_TIMEOUT_S,
                )
                break
            except asyncio.TimeoutError:
                logger.error(
                    "STT stuck-state watchdog: transcription exceeded %.0fs timeout (conv=%s)",
                    self._STT_TIMEOUT_S, conversation_id[:8] if conversation_id else "?",
                )
                self._resume_listening(conversation_id)
                if self.perception:
                    self.perception.broadcast({
                        "type": "command",
                        "data": {"action": "stt_failed"},
                        "conversation_id": conversation_id,
                    })
                return
            except Exception as exc:
                if attempt == 0 and "out of memory" in str(exc).lower():
                    logger.warning("STT CUDA OOM on attempt 1, retrying after VRAM settle...")
                    await asyncio.sleep(1.5)
                    continue
                logger.exception("STT failed for conv=%s: %s",
                                 conversation_id[:8] if conversation_id else "?", exc)
                self._resume_listening(conversation_id)
                if self.perception:
                    self.perception.broadcast({
                        "type": "command",
                        "data": {"action": "stt_failed"},
                        "conversation_id": conversation_id,
                    })
                return

        if self._latest_stt_conv_id != conversation_id:
            logger.warning("Stale STT result discarded: conv=%s", conversation_id[:8])
            self._resume_listening(conversation_id)
            return

        if text.strip():
            if self._is_echo(text.strip()):
                self._resume_listening(conversation_id)
                return

            is_synthetic = self._current_conv_synthetic

            if not is_synthetic:
                self._last_speech_audio = audio_f32
                self._last_speech_sr = 16000
                self._recent_speech_clips.append(audio_f32)
                if len(self._recent_speech_clips) > self._MAX_ENROLLMENT_CLIPS:
                    self._recent_speech_clips.pop(0)

            self._identify_speaker_and_emotion(audio_f32, synthetic=is_synthetic)

            if is_synthetic:
                self._synthetic_ledger["utterances_stt"] += 1
                self._synthetic_route_only(text, conversation_id)
                self._resume_listening(conversation_id)
                return

            _stt_total_ms = (_t.monotonic() - _stt_wall_start) * 1000
            _stt_transcribe_ms = _stt_total_ms - _stt_prep_ms
            logger.info("[LATENCY] STT pipeline: prep=%.0fms transcribe=%.0fms total=%.0fms (conv=%s)",
                        _stt_prep_ms, _stt_transcribe_ms, _stt_total_ms,
                        conversation_id[:8] if conversation_id else "?")
            logger.info("STT result (conv=%s): %s",
                         conversation_id[:8] if conversation_id else "?", text[:80])
            event_bus.emit(
                PERCEPTION_TRANSCRIPTION,
                text=text,
                timestamp=time.time(),
                conversation_id=conversation_id,
            )
        else:
            logger.info("STT returned empty (conv=%s)",
                         conversation_id[:8] if conversation_id else "?")
            self._resume_listening(conversation_id)
            if self.perception:
                self.perception.broadcast({
                    "type": "command",
                    "data": {"action": "stt_failed"},
                    "conversation_id": conversation_id,
                })

    # ------------------------------------------------------------------
    # Speaker ID & emotion
    # ------------------------------------------------------------------

    def _identify_speaker_and_emotion(self, audio_f32, *, synthetic: bool = False) -> None:
        """Run speaker ID and audio emotion classification on a speech buffer.

        Called synchronously from _transcribe_speech after STT succeeds.
        Both models run on GPU and are fast (~50-100ms each). Results are
        emitted as events which the existing _on_speaker_id / _on_emotion
        handlers pick up to set _current_speaker / _current_emotion state
        before the conversation handler processes the transcription.

        When synthetic=True, the GPU models still run (distillation signals
        fire inside them), but the state-mutating event emissions are skipped
        to prevent identity/tone/trait contamination.
        """
        origin = "synthetic" if synthetic else "mic"

        if self.speaker_id and self.speaker_id.available:
            try:
                result = self.speaker_id.identify(audio_f32, 16000, origin=origin)
                if not synthetic and result.get("confidence", 0) > 0:
                    event_bus.emit(
                        PERCEPTION_SPEAKER_IDENTIFIED,
                        name=result["name"],
                        confidence=result["confidence"],
                        is_known=result["is_known"],
                        embedding_id=result.get("embedding_id", ""),
                        closest_match=result.get("closest_match", ""),
                    )
                elif synthetic:
                    self._synthetic_ledger["distillation_records"] += 1
                    self._synthetic_ledger["blocked_side_effects"] += 1
            except Exception as exc:
                logger.debug("Speaker ID failed: %s", exc)

        if (self.emotion_classifier
                and getattr(self.emotion_classifier, "_gpu_available", False)
                and getattr(self.emotion_classifier, "_model_healthy", False)):
            try:
                result = self.emotion_classifier._classify_waveform(audio_f32, 16000, origin=origin)
                if not synthetic and result and result.emotion != "neutral":
                    trust = "high" if result.confidence > 0.6 else "medium"
                    event_bus.emit(
                        PERCEPTION_USER_EMOTION,
                        emotion=result.emotion,
                        confidence=result.confidence,
                        source="audio",
                        trust=trust,
                    )
                elif synthetic:
                    self._synthetic_ledger["distillation_records"] += 1
                    self._synthetic_ledger["blocked_side_effects"] += 1
            except Exception as exc:
                logger.debug("Audio emotion classification failed: %s", exc)
        elif self.emotion_classifier and not getattr(self.emotion_classifier, "_model_healthy", True):
            if not synthetic:
                try:
                    import numpy as np
                    rms = float(np.sqrt(np.mean(audio_f32 ** 2)) + 1e-9)
                    duration_s = len(audio_f32) / 16000.0
                    speech_rate = 0.0
                    pitch_hz = 0.0
                    spectral_centroid = 0.0
                    try:
                        fft = np.fft.rfft(audio_f32[:16000])
                        magnitudes = np.abs(fft)
                        freqs = np.fft.rfftfreq(min(len(audio_f32), 16000), 1.0 / 16000)
                        total_mag = magnitudes.sum() + 1e-9
                        spectral_centroid = float(np.sum(freqs * magnitudes) / total_mag)
                        corr = np.correlate(audio_f32[:4000], audio_f32[:4000], mode='full')
                        corr = corr[len(corr) // 2:]
                        d = np.diff(corr)
                        starts = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
                        if len(starts) > 0:
                            first_peak = starts[0] + 1
                            if first_peak > 0:
                                pitch_hz = float(16000.0 / first_peak)
                    except Exception:
                        pass
                    result = self.emotion_classifier.classify(
                        rms=rms, duration_s=duration_s,
                        pitch_hz=pitch_hz, spectral_centroid=spectral_centroid,
                        speech_rate=speech_rate,
                    )
                    if result and result.emotion != "neutral":
                        event_bus.emit(
                            PERCEPTION_USER_EMOTION,
                            emotion=result.emotion,
                            confidence=result.confidence,
                            source="heuristic",
                            trust="low",
                        )
                except Exception as exc:
                    logger.debug("Heuristic emotion classification failed: %s", exc)

    def _synthetic_route_only(self, text: str, conversation_id: str) -> None:
        """Run tool router in isolation for distillation — no conversation handler."""
        try:
            from reasoning.tool_router import tool_router
            result = tool_router.route(text, synthetic=True)
            route_name = result.tool.value if hasattr(result, "tool") else str(result)
            self._synthetic_ledger["routes_produced"] += 1
            self._synthetic_ledger["distillation_records"] += 2

            hist = self._synthetic_ledger["route_histogram"]
            hist[route_name] = hist.get(route_name, 0) + 1

            stt_texts = self._synthetic_ledger["stt_texts"]
            if len(stt_texts) < 20:
                stt_texts.append(text[:80])

            examples = self._synthetic_ledger["recent_route_examples"]
            examples.append({"text": text[:80], "route": route_name})
            if len(examples) > 20:
                del examples[:-20]

            logger.info(
                "Synthetic exercise: STT='%s' route=%s (conv=%s) — HARD STOP",
                text[:60], route_name, conversation_id[:8] if conversation_id else "?",
            )
        except Exception as exc:
            logger.debug("Synthetic route-only failed: %s", exc)

    def get_synthetic_exercise_stats(self) -> dict[str, Any]:
        """Return current synthetic exercise ledger for dashboard."""
        cooldown_remaining = max(0.0, self._synthetic_cooldown_until - time.time())
        return {
            "active": bool(self._synthetic_sources),
            "cooldown_s": round(cooldown_remaining, 1),
            **self._synthetic_ledger,
        }

    def _on_face_id(self, name, confidence, is_known, track_id=-1, closest_match="", **_):
        self._current_face = {"name": name if is_known else "unknown", "confidence": confidence, "is_known": is_known}
        if self.vision_proc:
            self.identity_fusion.set_visible_persons(
                self.vision_proc._persons_present
            )
        if is_known:
            print(f"  [Face: {name} ({confidence:.0%})]")
            try:
                from identity.evidence_accumulator import get_accumulator
                get_accumulator().observe(
                    name, "face_match", confidence=confidence,
                    details="face_id match",
                )
            except Exception:
                pass

    def _on_identity_resolved(self, name="unknown", confidence=0.0, is_known=False,
                               method="", conflict=False, **_):
        """Canonical identity from fusion of voice + face.

        Updates _current_speaker so soul/relationships use the best
        available identity, not just the last raw voice signal.
        When face confirms a speaker that voice didn't match, updates
        the voice profile's interaction count and does a gentle EMA.
        """
        if is_known:
            self._current_speaker["name"] = name
            self._current_speaker["identity_method"] = method
            self._current_speaker["confidence"] = float(confidence)
            self._current_speaker["is_known"] = True
            if method == "face_only" and self.speaker_id and self.speaker_id.available:
                voice_known = self._current_speaker.get("voice_is_known")
                if voice_known == "False" or voice_known is None:
                    voice_score = float(self._current_speaker.get("voice_confidence", 0))
                    embedding = self.speaker_id.last_embedding
                    self.speaker_id.record_fused_interaction(name, voice_score=voice_score, embedding=embedding)
                    self._current_speaker["voice_is_known"] = "fused"
            if conflict:
                print(f"  [Identity CONFLICT: voice/face disagree, using {name} via {method}]")
                try:
                    from identity.evidence_accumulator import get_accumulator
                    get_accumulator().observe(
                        name, "contradiction", confidence=0.5,
                        details=f"voice/face conflict resolved via {method}",
                    )
                except Exception:
                    pass
            else:
                print(f"  [Identity: {name} ({confidence:.0%}, {method})]")
        else:
            self._current_speaker["name"] = "unknown"
            self._current_speaker["identity_method"] = method
            self._current_speaker["confidence"] = 0.0
            self._current_speaker["is_known"] = False
            face_name = _.get("face_name", "unknown")
            if method == "face_present_voice_unknown" and face_name != "unknown":
                self._current_speaker["visible_face"] = face_name
                print(f"  [Identity: unknown speaker ({face_name} visible on camera)]")
            else:
                self._current_speaker.pop("visible_face", None)
        try:
            from skills.capability_gate import capability_gate
            capability_gate.set_identity_confirmed(is_known and not conflict)
        except Exception:
            pass

    def _on_speaker_id(self, name, confidence, is_known, **_):
        resolved = name if is_known else "unknown"
        self._current_speaker["voice_name"] = resolved
        self._current_speaker["voice_confidence"] = float(confidence)
        self._current_speaker["voice_is_known"] = str(is_known)  # three-state: "True"/"False"/"fused"
        if is_known:
            self._current_speaker["name"] = resolved
            self._current_speaker["confidence"] = float(confidence)
            self._current_speaker["is_known"] = bool(is_known)

            try:
                from identity.evidence_accumulator import get_accumulator
                get_accumulator().observe(
                    resolved, "voice_match", confidence=confidence,
                    details="speaker_id match",
                )
            except Exception:
                pass

        if is_known and resolved != "unknown":
            is_new_this_session = resolved.lower() not in self._speakers_seen_this_session
            speaker_changed = self._previous_speaker_name.lower() != resolved.lower()
            self._current_speaker["first_this_session"] = bool(is_new_this_session)
            self._current_speaker["speaker_changed"] = bool(speaker_changed)
            self._speakers_seen_this_session.add(resolved.lower())
            self._previous_speaker_name = resolved
            print(f"  [Speaker: {name} ({confidence:.0%}){' NEW' if is_new_this_session else ''}]")

            if is_new_this_session or speaker_changed:
                primary = self._get_primary_companion()
                from consciousness.events import KERNEL_THOUGHT
                if primary and resolved.lower() != primary.lower():
                    event_bus.emit(KERNEL_THOUGHT,
                                  content=f"I notice {resolved} is speaking — that's not {primary}. "
                                          f"I'm curious what brings them here.",
                                  tone="curious")
                elif not primary:
                    event_bus.emit(KERNEL_THOUGHT,
                                  content=f"I hear {resolved}'s voice for the first time this session.",
                                  tone="curious")
        else:
            # Check if this is a genuinely new unknown voice: either the
            # previous speaker was known, OR identity fusion currently
            # knows someone (e.g. David identified by face but hasn't spoken)
            is_new_unknown = self._previous_speaker_name != "unknown"
            if not is_new_unknown:
                try:
                    fused = self.identity_fusion.current
                    is_new_unknown = fused.is_known and fused.name != "unknown"
                except Exception:
                    pass

            self._current_speaker["first_this_session"] = True
            self._current_speaker["speaker_changed"] = bool(is_new_unknown)

            if is_new_unknown:
                primary = self._get_primary_companion()
                known_user = primary
                if not known_user:
                    try:
                        fused = self.identity_fusion.current
                        if fused.is_known and fused.name != "unknown":
                            known_user = fused.name
                    except Exception:
                        pass

                from consciousness.events import KERNEL_THOUGHT
                if known_user:
                    event_bus.emit(KERNEL_THOUGHT,
                                  content=(
                                      f"Unrecognized speaker signal detected (not matching {known_user}). "
                                      "Waiting for a clearer sample before updating identity state."
                                  ),
                                  tone="curious")
                else:
                    event_bus.emit(KERNEL_THOUGHT,
                                  content=(
                                      "Unrecognized speaker signal detected. "
                                      "Waiting for a clearer sample before updating identity state."
                                  ),
                                  tone="curious")

                # Record unknown speaker event for curiosity system
                now_ts = time.time()
                if now_ts - self._last_unknown_speaker_at > self._UNKNOWN_SPEAKER_COOLDOWN_S:
                    self._unknown_speaker_events.append({
                        "timestamp": now_ts,
                        "primary_user": known_user or "",
                        "raw_name": name,
                        "confidence": float(confidence),
                        "had_known_user": bool(known_user),
                    })
                    self._last_unknown_speaker_at = now_ts

                    # Signal tension to policy state encoder for NN learning
                    encoder = getattr(self.engine, "_state_encoder", None)
                    if encoder and hasattr(encoder, "set_unknown_speaker_tension"):
                        encoder.set_unknown_speaker_tension(0.9)

        modulation = trait_perception.process_event("speaker_identified",
                                                    {"speaker_name": name, "confidence": confidence, "is_known": is_known})
        if modulation and modulation.thought_trigger:
            from consciousness.events import KERNEL_THOUGHT
            event_bus.emit(KERNEL_THOUGHT, content=modulation.thought_trigger, tone="contemplative")

    _EMOTION_TRUST_THRESHOLD = 0.35

    def _on_emotion(self, emotion, confidence, text_sentiment="", source="", trust="medium", **_):
        self._current_emotion["emotion"] = emotion
        self._current_emotion["confidence"] = confidence
        self._current_emotion["trust"] = trust
        self._current_emotion["source"] = source

        trusted = trust != "low" and confidence >= self._EMOTION_TRUST_THRESHOLD
        self._current_emotion["trusted"] = trusted

        if not trusted:
            return

        modulation = trait_perception.process_event("emotion_detected",
                                                    {"emotion": emotion, "confidence": confidence})
        if modulation and modulation.thought_trigger:
            from consciousness.events import KERNEL_THOUGHT
            event_bus.emit(KERNEL_THOUGHT, content=modulation.thought_trigger, tone="contemplative")
        if emotion != "neutral" and source != "heuristic":
            self.engine.calibrate_tone(
                time_of_day=int(time.strftime("%H")),
                user_emotion=emotion,
            )

    # ------------------------------------------------------------------
    # Scene summary (Layer 3B)
    # ------------------------------------------------------------------

    def _on_scene_summary(self, detections=None, frame_size=None, scene_change_score=0.0, **_):
        """Handle scene_summary events from the Pi aggregator."""
        if not detections:
            return
        fw, fh = (frame_size or [640, 480])[:2]

        scene_dets: list[SceneDetection] = []
        for d in detections:
            bbox_raw = d.get("bbox")
            bbox = tuple(bbox_raw) if bbox_raw and len(bbox_raw) == 4 else None
            scene_dets.append(SceneDetection(
                label=d.get("label", "unknown"),
                confidence=d.get("confidence", 0.0),
                bbox=bbox,
                source="pi",
                hit_count=d.get("hit_count", 1),
            ))

        person_bboxes: list[tuple[int, int, int, int]] = []

        snapshot = self._scene_tracker.update(scene_dets, fw, fh, person_bboxes)
        self._last_scene_snapshot = snapshot

        entity_ct = len(snapshot.entities)
        display_ct = len(snapshot.display_surfaces)
        labels = [f"{d.label}:{d.confidence:.2f}" for d in scene_dets[:6]]
        if entity_ct > 0 or self._scene_tracker._update_count <= 5:
            logger.info(
                "Scene update #%d: %d entities, %d displays, change=%.2f [%s]",
                self._scene_tracker._update_count, entity_ct, display_ct,
                scene_change_score, ", ".join(labels),
            )

        if snapshot.display_surfaces:
            desc = self._last_scene_description
            if desc:
                content_summaries = self._display_classifier.classify_from_description(
                    desc, snapshot.display_surfaces,
                )
                snapshot.display_content = content_summaries

        scene_signature = self._build_spatial_scene_signature(snapshot)
        self._spatial_scene_signature = scene_signature
        self._spatial_last_scene_change_score = float(scene_change_score or 0.0)
        if scene_signature:
            now = time.time()
            persist_profile = (
                now - self._last_spatial_profile_sync_ts
            ) >= self._spatial_profile_sync_interval_s
            self._calibration_manager.upsert_profile(
                self._calibration_manager.active_profile_id,
                scene_signature=scene_signature,
                persist=persist_profile,
            )
            if persist_profile:
                self._last_spatial_profile_sync_ts = now
        self._maybe_relocalize_spatial(
            scene_change_score=float(scene_change_score or 0.0),
            scene_signature=scene_signature,
        )

        self._process_spatial(snapshot)

    # ------------------------------------------------------------------
    # Spatial Intelligence Pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spatial_scene_signature(snapshot: Any) -> dict[str, Any]:
        labels: set[str] = set()
        regions: set[str] = set()
        display_kinds: set[str] = set()
        entity_count = 0
        display_count = 0

        entities = list(getattr(snapshot, "entities", []) or [])
        for ent in entities:
            state = str(getattr(ent, "state", "") or "")
            if state not in ("visible", "occluded"):
                continue
            label = str(getattr(ent, "label", "") or "").strip().lower()
            region = str(getattr(ent, "region", "") or "").strip().lower()
            if label:
                labels.add(label)
            if region:
                regions.add(region)
            entity_count += 1

        surfaces = list(getattr(snapshot, "display_surfaces", []) or [])
        for surf in surfaces:
            kind = str(getattr(surf, "kind", "") or "").strip().lower()
            if kind:
                display_kinds.add(kind)
            display_count += 1

        return {
            "labels": sorted(labels),
            "regions": sorted(regions),
            "display_kinds": sorted(display_kinds),
            "entity_count": int(entity_count),
            "display_count": int(display_count),
        }

    def _maybe_relocalize_spatial(
        self,
        *,
        scene_change_score: float,
        scene_signature: dict[str, Any],
    ) -> None:
        if not scene_signature:
            return
        now = time.time()
        if now - self._last_spatial_relocalization_ts < self._spatial_relocalization_interval_s:
            return

        cal_state = self._calibration_manager.get_state()
        state = str(cal_state.get("state", "") or "")
        reason = str(cal_state.get("reason", "") or "")
        local_only = state == "stale" and reason in {
            "verification_expired_local_only",
            "profile_handoff_local_only",
        }
        if not local_only:
            return
        if scene_change_score < self._spatial_relocalization_threshold:
            return

        active_profile = self._calibration_manager.active_profile_id
        target_profile = active_profile
        match_score = 0.0
        handoff_reason = "no_match"

        match = self._calibration_manager.match_profile(
            scene_signature,
            min_score=self._spatial_relocalization_threshold,
            include_active=True,
        )
        if match is not None:
            target_profile, match_score = match
            handoff_reason = "matched_profile"
        else:
            target_profile = self._calibration_manager.suggest_profile_id(scene_signature)
            self._calibration_manager.upsert_profile(
                target_profile,
                scene_signature=scene_signature,
                persist=True,
            )
            handoff_reason = "new_profile"

        if target_profile == active_profile:
            self._last_spatial_relocalization_ts = now
            self._last_spatial_relocalization = {
                "attempted_at": round(now, 1),
                "from_profile": active_profile,
                "to_profile": target_profile,
                "scene_change_score": round(scene_change_score, 3),
                "match_score": round(match_score, 3),
                "status": "no_handoff_same_profile",
                "reason": handoff_reason,
            }
            return

        activated = self._calibration_manager.activate_profile(
            target_profile,
            stale_reason="profile_handoff_local_only",
            persist=True,
        )
        self._last_spatial_relocalization_ts = now
        if not activated:
            self._last_spatial_relocalization = {
                "attempted_at": round(now, 1),
                "from_profile": active_profile,
                "to_profile": target_profile,
                "scene_change_score": round(scene_change_score, 3),
                "match_score": round(match_score, 3),
                "status": "activation_failed",
                "reason": handoff_reason,
            }
            return

        self._spatial_estimator.reset_for_relocalization(
            profile_id=target_profile,
            reason=handoff_reason,
        )
        self._spatial_validator.reset_for_relocalization(
            profile_id=target_profile,
            reason=handoff_reason,
        )
        self._spatial_post_handoff_stable_windows = 0
        self._spatial_fused = {}
        event_bus.emit(
            SPATIAL_CALIBRATION_CHANGED,
            from_profile=active_profile,
            to_profile=target_profile,
            scene_change_score=scene_change_score,
            match_score=match_score,
            handoff_reason=handoff_reason,
        )
        logger.info(
            "Spatial relocalization handoff: %s -> %s (change=%.3f, match=%.3f, reason=%s)",
            active_profile,
            target_profile,
            scene_change_score,
            match_score,
            handoff_reason,
        )
        self._last_spatial_relocalization = {
            "attempted_at": round(now, 1),
            "from_profile": active_profile,
            "to_profile": target_profile,
            "scene_change_score": round(scene_change_score, 3),
            "match_score": round(match_score, 3),
            "status": "handoff_applied",
            "reason": handoff_reason,
        }

    def _process_spatial(self, snapshot: Any) -> None:
        """Run spatial estimation on scene entities with bounding boxes."""
        if self._spatial_estimator is None:
            return
        try:
            now = time.time()
            tracked_ids: set[str] = set()

            for ent in snapshot.entities:
                if ent.state not in ("visible", "occluded"):
                    continue
                if not ent.bbox:
                    continue

                obs = self._spatial_estimator.estimate(
                    entity_id=ent.entity_id,
                    label=ent.label,
                    bbox=ent.bbox,
                    confidence=ent.confidence,
                    timestamp=now,
                )
                if obs is None:
                    continue

                track = self._spatial_estimator.update_track(obs)
                tracked_ids.add(ent.entity_id)

                if track.track_status == "stable":
                    delta = self._spatial_validator.validate_track_to_delta(
                        track,
                        self._spatial_estimator.get_anchors(),
                        calibration_version=self._calibration_manager.version,
                    )
                    if delta:
                        event_bus.emit(SPATIAL_DELTA_PROMOTED, delta=delta.to_dict())

            self._spatial_estimator.decay_stale_tracks()

            scene_dict = {
                "entities": [e.to_dict() for e in snapshot.entities],
                "display_surfaces": [
                    ds.to_dict() if hasattr(ds, "to_dict") else {}
                    for ds in snapshot.display_surfaces
                ],
                "display_content": [
                    dc.to_dict() if hasattr(dc, "to_dict") else {}
                    for dc in getattr(snapshot, "display_content", [])
                ],
                "region_visibility": getattr(snapshot, "region_visibility", {}),
            }
            self._spatial_fused = self._spatial_fusion.fuse(
                scene_dict,
                self._spatial_estimator.get_tracks(),
                self._spatial_estimator.get_anchors(),
            )
            self._spatial_update_count += 1

            if self._spatial_update_count <= 3 or self._spatial_update_count % 50 == 0:
                tracks = self._spatial_estimator.get_tracks()
                stable_ct = sum(1 for t in tracks.values() if t.track_status == "stable")
                logger.info(
                    "Spatial update #%d: %d tracks (%d stable), cal=%s",
                    self._spatial_update_count,
                    len(tracks),
                    stable_ct,
                    self._calibration_manager.state,
                )
            self._maybe_verify_spatial_handoff()
        except Exception:
            logger.debug("Spatial pipeline error", exc_info=True)

    def _maybe_verify_spatial_handoff(self) -> None:
        """Promote handoff-local calibration back to valid after stable windows."""
        cal = self._calibration_manager.get_state()
        if str(cal.get("state", "") or "") != "stale":
            self._spatial_post_handoff_stable_windows = 0
            return
        if str(cal.get("reason", "") or "") != "profile_handoff_local_only":
            self._spatial_post_handoff_stable_windows = 0
            return
        if not bool(cal.get("handoff_pending_verify", False)):
            self._spatial_post_handoff_stable_windows = 0
            return

        est = self._spatial_estimator.get_state()
        stable_tracks = int(est.get("stable_tracks", 0) or 0)
        scene_change = float(self._spatial_last_scene_change_score or 0.0)
        if (
            stable_tracks < self._spatial_post_handoff_verify_min_stable_tracks
            or scene_change > self._spatial_post_handoff_verify_max_scene_change
        ):
            self._spatial_post_handoff_stable_windows = 0
            return

        self._spatial_post_handoff_stable_windows += 1
        if (
            self._spatial_post_handoff_stable_windows
            < self._spatial_post_handoff_verify_required_windows
        ):
            return

        self._spatial_post_handoff_stable_windows = 0
        self._calibration_manager.verify(anchor_consistency_ok=True)
        logger.info(
            "Spatial handoff auto-verify: profile=%s stable_tracks=%d scene_change=%.3f",
            self._calibration_manager.active_profile_id,
            stable_tracks,
            scene_change,
        )

    def get_spatial_state(self) -> dict[str, Any]:
        """Return spatial intelligence state for dashboard/snapshot."""
        if self._spatial_estimator is None:
            return {"status": "not_initialized"}
        est_state = self._spatial_estimator.get_state()
        return {
            "status": "active",
            "calibration": self._calibration_manager.get_state(),
            "estimator": {
                "track_count": est_state.get("total_tracks", 0),
                "stable_tracks": est_state.get("stable_tracks", 0),
                "observation_count": est_state.get("observations_total", 0),
                "anchor_count": est_state.get("total_anchors", 0),
                "tracks": {
                    t.get("entity_id", f"t{i}"): t
                    for i, t in enumerate(est_state.get("tracks", []))
                },
            },
            "validation": self._spatial_validator.get_state(),
            "fused_entity_count": len(self._spatial_fused.get("entities", [])),
            "fused_relation_count": len(self._spatial_fused.get("spatial_relations", [])),
            "update_count": self._spatial_update_count,
            "scene_signature": dict(self._spatial_scene_signature),
            "relocalization": dict(self._last_spatial_relocalization),
        }

    # ------------------------------------------------------------------
    # Attention & mode
    # ------------------------------------------------------------------

    def _on_attention_update(self, state, **_):
        if self.ambient_proc and self.ambient_proc.get_state().get("in_meeting"):
            mode_manager.set_mode("passive", reason="user_in_meeting")
            return
        suggested = mode_manager.suggest_mode_from_attention(state)
        mode_manager.set_mode(suggested, reason="attention_heuristic")

    _VRAM_RELEASE_MODES = frozenset({"sleep", "dreaming", "deep_learning"})

    def _on_mode_change(self, to_mode="", **_):
        profile = mode_manager.profile
        if self.engine._kernel:
            self.engine._kernel.set_cadence_multiplier(profile.tick_cadence_multiplier)
        self.engine._policy_response_length = profile.response_depth_hint
        from memory.storage import memory_storage
        memory_storage.set_reinforcement_multiplier(profile.memory_reinforcement_multiplier)
        from memory.gate import memory_gate
        memory_gate.set_mode(mode_manager.mode)
        from personality.proactive import proactive_behavior
        proactive_behavior.set_cooldown_override(profile.proactivity_cooldown_s)
        if self.attention:
            self.attention.set_interruption_threshold(1.0 - profile.interruption_sensitivity)

        if to_mode in self._VRAM_RELEASE_MODES and self._ollama:
            tier = getattr(self.config, "gpu_tier", "")
            if tier in self._VRAM_COEXIST_TIERS:
                asyncio.ensure_future(self._release_vram_for_background(to_mode))

    def _on_conversation_response(self, text="", playback_estimate_s: float = 0.0, **_):
        from consciousness.operations import ops_tracker
        ops_tracker.set_subsystem("reasoning", "idle", f"response complete ({len(text)} chars)")
        ops_tracker.set_subsystem("speech", "idle", "playback complete")
        ops_tracker.log_event("reasoning", "response_complete", f"Response: {len(text)} chars")
        ops_tracker.advance_stage("playback", "done", f"Response: {len(text)} chars")
        ops_tracker.reset_interactive_path()
        # Feed back to addressee telemetry whether a response was actually produced
        if self._addressee_gate:
            self._addressee_gate.mark_response_generated(bool(text))
        if text:
            self._last_response_text = text
            self._last_response_set_time = time.monotonic()
            # New response text supersedes any prior playback timestamp.
            # Keep echo guard active until either a fresh playback_complete
            # arrives or stale timeout elapses.
            self._playback_complete_time = 0.0
        if self.audio_stream and not self.brain_tts.available:
            self.audio_stream.set_speaking(False)
            self.audio_stream.set_follow_up()
        elif self.audio_stream and self.brain_tts.available and self.audio_stream.is_speaking:
            timeout_s = max(
                self._SPEAKING_SAFETY_TIMEOUT_S,
                float(playback_estimate_s or 0.0) + 20.0,
            )
            self._start_speaking_safety_timer(timeout_s=timeout_s)
        elif self.audio_stream:
            self.audio_stream.set_follow_up()

    def _on_playback_complete(self, conversation_id: str = "", **_):
        """Pi finished playing TTS audio — now safe to enter follow-up mode.

        If conversation_id is provided and doesn't match the current speaking
        conversation, this is a late arrival from a previous conversation and
        is ignored to prevent corrupting the active conversation's state.
        """
        current_conv = self._speaking_conv_id
        if conversation_id and current_conv and conversation_id != current_conv:
            logger.debug("Stale playback_complete for conv %s (current: %s) — ignoring",
                         conversation_id[:8], current_conv[:8])
            return
        logger.debug("Pi playback complete — entering follow-up mode")
        self._playback_complete_time = time.monotonic()
        self._speaking_conv_id = ""
        self._cancel_speaking_safety_timer()
        if self.audio_stream:
            self.audio_stream.set_speaking(False)
            self.audio_stream.set_follow_up()

    def _start_speaking_safety_timer(self, timeout_s: float | None = None) -> None:
        """Start a safety timer that clears speaking if playback_complete never arrives."""
        self._cancel_speaking_safety_timer()
        timeout = float(timeout_s or self._SPEAKING_SAFETY_TIMEOUT_S)
        self._speaking_safety_timer = threading.Timer(
            timeout, self._speaking_safety_expired)
        self._speaking_safety_timer.daemon = True
        self._speaking_safety_timer.start()

    def _cancel_speaking_safety_timer(self) -> None:
        timer = self._speaking_safety_timer
        self._speaking_safety_timer = None
        if timer:
            timer.cancel()

    def _speaking_safety_expired(self) -> None:
        """Forcibly clear speaking flag — playback_complete never arrived."""
        logger.warning("Speaking safety timeout — forcing set_speaking(False)")
        self._speaking_safety_timer = None
        self._speaking_conv_id = ""
        if self.audio_stream:
            self.audio_stream.set_speaking(False)
            self.audio_stream.set_follow_up()

    def _on_sensor_disconnect(self) -> None:
        """Called when all sensors disconnect — clear stuck speaking state."""
        if self.audio_stream and self.audio_stream.is_speaking:
            logger.warning("Sensor disconnected while speaking — clearing speaking flag")
            self._cancel_speaking_safety_timer()
            self._speaking_conv_id = ""
            self.audio_stream.set_speaking(False)

    def _emit_conversation_response(
        self,
        *,
        text: str,
        conversation_id: str = "",
        release_status: str = "released",
        release_reason: str = "",
    ) -> str:
        """Emit terminal response events with minimum lineage metadata."""
        resolved_conversation_id = (
            conversation_id
            or self._speaking_conv_id
            or f"conv_{time.monotonic_ns()}"
        )
        trace_ctx = build_trace_context(resolved_conversation_id)
        output_id = f"out_{time.monotonic_ns()}"
        validation_decision = output_release_validator.validate_output(
            text=text,
            conversation_id=trace_ctx.conversation_id,
            trace_id=trace_ctx.trace_id,
            request_id=trace_ctx.request_id,
            output_id=output_id,
            release_status=release_status,
            release_reason=release_reason,
            source="perception_orchestrator",
        )
        payload: dict[str, Any] = {
            "text": text,
            "conversation_id": trace_ctx.conversation_id,
            "trace_id": trace_ctx.trace_id,
            "request_id": trace_ctx.request_id,
            "output_id": output_id,
            "validation_id": validation_decision.validation_id,
            "validation_passed": validation_decision.passed,
            "validation_violations": list(validation_decision.violations),
            "release_status": validation_decision.effective_release_status,
            "provenance": {
                "provenance": "fallback_system_response",
                "source": "fallback:perception_orchestrator",
                "confidence": 0.0,
                "native": True,
                "response_class": "system_fallback",
                "fallback": True,
            },
        }
        if validation_decision.effective_release_reason:
            payload["release_reason"] = validation_decision.effective_release_reason
        try:
            event_bus.emit(CONVERSATION_RESPONSE, **payload)
        except Exception:
            logger.debug("Failed to emit CONVERSATION_RESPONSE lineage payload", exc_info=True)
        return output_id

    def _speak_proactive(self, text: str) -> None:
        """Broadcast proactive/greeting TTS through the full speaking lifecycle.

        Enters speaking state, broadcasts the speak command with TTS audio,
        sends response_end, updates echo detection state, and emits
        CONVERSATION_RESPONSE so the normal cleanup path (safety timer,
        follow-up mode) fires correctly.
        """
        if not self.perception or not text:
            return
        conv_id = f"proactive_{time.monotonic()}"
        self._speaking_conv_id = conv_id
        if self.audio_stream:
            self.audio_stream.set_speaking(True)
        try:
            msg: dict[str, Any] = {
                "type": "command",
                "data": {"action": "speak"},
                "text": text,
                "conversation_id": conv_id,
            }
            if self.brain_tts and self.brain_tts.available:
                profile = self.brain_tts.get_voice_profile(
                    tone=self.engine.get_state().get("tone", "professional"),
                    user_emotion=self._current_emotion.get("emotion", "neutral"),
                    emotion_trusted=bool(self._current_emotion.get("trusted", False)),
                    speaker_name=(self._current_speaker or {}).get("name", ""),
                    proactive=True,
                )
                self._synthetic_leak_tts(text_chars=len(text or ""))
                audio_b64 = self.brain_tts.synthesize_b64(text, profile=profile)
                if audio_b64:
                    msg["data"]["audio_b64"] = audio_b64
            self.perception.broadcast(msg)
        except Exception:
            logger.exception("_speak_proactive TTS/broadcast error")
        try:
            self.perception.broadcast({
                "type": "response_end", "text": "", "tone": "",
                "phase": "LISTENING", "conversation_id": conv_id,
            })
        except Exception:
            logger.exception("_speak_proactive response_end broadcast error")
        self._last_response_text = text
        self._last_response_set_time = time.monotonic()
        self._playback_complete_time = 0.0
        self._emit_conversation_response(
            text=text,
            conversation_id=conv_id,
            release_status="released",
            release_reason="proactive",
        )

    def _resume_listening(self, conversation_id: str = "") -> None:
        """Tell the Pi to return to listening after a discarded transcription."""
        if self.audio_stream and self.audio_stream._speaking:
            self.audio_stream.set_speaking(False)
        if self.perception:
            self.perception.broadcast({
                "type": "command",
                "data": {"action": "phase_update", "phase": "LISTENING"},
                "conversation_id": conversation_id,
            })
        self.engine.set_phase("LISTENING")

    _ECHO_SIMILARITY_THRESHOLD = 0.70
    _ECHO_PARTIAL_THRESHOLD = 0.50
    _ECHO_CONVERSATIONAL_MARKERS = (
        "i asked", "did you", "you said", "i said",
        "you just", "i just", "what did you", "repeat",
    )
    _ECHO_WINDOW_S = 12.0
    _ECHO_STALE_S = 30.0

    def _is_echo_expired(self) -> bool:
        """Check if the echo reference has expired (stale response text)."""
        if not self._last_response_text:
            return True
        now = time.monotonic()
        # If response text was set after the last playback_complete marker,
        # treat it as a fresh response still in the echo-guard window.
        if self._last_response_set_time > 0 and (
            self._playback_complete_time <= 0
            or self._last_response_set_time > self._playback_complete_time
        ):
            return (now - self._last_response_set_time) > self._ECHO_STALE_S
        if self._playback_complete_time > 0:
            return (now - self._playback_complete_time) > self._ECHO_WINDOW_S
        return False

    def _is_echo(self, text: str) -> bool:
        if not self._last_response_text or not text:
            return False

        if self._is_echo_expired():
            self._last_response_text = ""
            return False

        lower = text.strip().lower()

        if any(marker in lower for marker in self._ECHO_CONVERSATIONAL_MARKERS):
            return False

        ref_lower = self._last_response_text.strip().lower()

        is_substring = lower in ref_lower and len(lower) >= 10
        if is_substring:
            gap = time.monotonic() - self._playback_complete_time if self._playback_complete_time else -1
            logger.warning("Echo detected (substring match, gap=%.1fs) — discarding: %s",
                           gap, text[:80])
            try:
                event_bus.emit("echo:detected", text=text[:80], similarity=1.0,
                               gap_s=gap, guard="substring")
            except Exception:
                pass
            return True

        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, ref_lower, lower).ratio()

        threshold = self._ECHO_SIMILARITY_THRESHOLD
        if len(lower) < len(ref_lower) * 0.5 and len(lower) >= 10:
            threshold = self._ECHO_PARTIAL_THRESHOLD

        if similarity >= threshold:
            gap = time.monotonic() - self._playback_complete_time if self._playback_complete_time else -1
            logger.warning("Echo detected (%.0f%% similar, threshold=%.0f%%, gap=%.1fs) — discarding: %s",
                           similarity * 100, threshold * 100, gap, text[:80])
            try:
                event_bus.emit("echo:detected", text=text[:80], similarity=similarity,
                               gap_s=gap, guard="text_similarity")
            except Exception:
                pass
            return True
        return False

    _SPEAKER_ECHO_SIMILARITY = 0.60
    _SPEAKER_ECHO_WINDOW_S = 15.0

    def _is_speaker_echo(self, text: str) -> bool:
        """Second line of defense: if speaker_id says 'unknown' and the text
        closely matches the last response, this is almost certainly the mic
        picking up TTS playback. Catches echoes that slip past _is_echo()
        due to timing races."""
        if not self._last_response_text or not text:
            return False

        if self._is_echo_expired():
            return False

        speaker = self._current_speaker.get("name", "unknown") if self._current_speaker else "unknown"
        is_known = self._current_speaker.get("is_known", False) if self._current_speaker else False
        if is_known and speaker != "unknown":
            return False

        ref_lower = self._last_response_text.strip().lower()
        lower = text.strip().lower()

        if lower in ref_lower and len(lower) >= 8:
            logger.warning(
                "ECHO-GUARD: unknown speaker substring match — blocking: %s", text[:80],
            )
            try:
                event_bus.emit("echo:detected", text=text[:80], similarity=1.0,
                               speaker=speaker, guard="speaker_substring")
            except Exception:
                pass
            return True

        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, ref_lower, lower).ratio()
        if similarity >= self._SPEAKER_ECHO_SIMILARITY:
            logger.warning(
                "ECHO-GUARD: unknown speaker (%.0f%% match to last response) — blocking: %s",
                similarity * 100, text[:80],
            )
            try:
                event_bus.emit("echo:detected", text=text[:80], similarity=similarity,
                               speaker=speaker, guard="speaker_mismatch")
            except Exception:
                pass
            return True
        return False

    # ------------------------------------------------------------------
    # Transcription handling
    # ------------------------------------------------------------------

    _DISMISS_PHRASES = frozenset({
        "stop listening", "stop", "cancel", "never mind", "nevermind",
        "go to sleep", "go away", "shut up", "be quiet", "that's all",
        "that's it", "goodbye", "good night", "dismiss",
    })
    _DISMISS_PREFIXES = ("stop listen", "never mind", "go to sleep", "go away")

    def _is_dismiss_command(self, text: str) -> bool:
        lower = text.strip().lower().rstrip(".!?,")
        return lower in self._DISMISS_PHRASES or any(
            lower.startswith(p) for p in self._DISMISS_PREFIXES
        )

    def _on_transcription(self, text, timestamp, conversation_id="", **_):
        from consciousness.operations import ops_tracker
        ops_tracker.set_subsystem("perception", "idle", "transcription delivered")
        ops_tracker.set_subsystem("stt", "idle", f"Transcribed: {text[:40]}")
        ops_tracker.log_event("perception", "transcription", f"STT: \"{text[:60]}\"")
        ops_tracker.advance_stage("stt", "done", f"Transcribed: {text[:40]}")
        ops_tracker.advance_stage("route", "active", "Routing request")
        if self._is_dismiss_command(text):
            logger.info("Dismiss command detected: '%s'", text[:40])
            with self._conv_lock:
                self._active_conversation = {"id": conversation_id, "cancelled": True}
            if self.audio_stream:
                self.audio_stream.set_speaking(True)
            self._speaking_conv_id = conversation_id
            if self.perception:
                dismiss_msg: dict[str, Any] = {
                    "type": "command",
                    "data": {"action": "speak"},
                    "text": "Going quiet. Say hey Jarvis when you need me.",
                    "conversation_id": conversation_id,
                }
                if self.brain_tts and self.brain_tts.available:
                    profile = self.brain_tts.get_voice_profile(
                        tone=self.engine.get_state().get("tone", "professional"),
                        user_emotion=self._current_emotion.get("emotion", "neutral"),
                        emotion_trusted=bool(self._current_emotion.get("trusted", False)),
                        speaker_name=(self._current_speaker or {}).get("name", ""),
                    )
                    self._synthetic_leak_tts(text_chars=len(dismiss_msg.get("text", "") or ""))
                    audio_b64 = self.brain_tts.synthesize_b64(dismiss_msg["text"], profile=profile)
                    if audio_b64:
                        dismiss_msg["data"]["audio_b64"] = audio_b64
                self.perception.broadcast(dismiss_msg)
                self.perception.broadcast({
                    "type": "response_end", "text": "", "tone": "",
                    "phase": "LISTENING", "conversation_id": conversation_id,
                })
            self._last_response_text = dismiss_msg["text"]
            self._last_response_set_time = time.monotonic()
            self._playback_complete_time = 0.0
            self._emit_conversation_response(
                text=dismiss_msg["text"],
                conversation_id=conversation_id,
                release_status="released",
                release_reason="dismiss_command",
            )
            if self.brain_tts and self.brain_tts.available:
                self._start_speaking_safety_timer()
            else:
                if self.audio_stream:
                    self.audio_stream.set_speaking(False)
            return

        if self._is_speaker_echo(text):
            logger.warning("Speaker-echo guard: unknown speaker echoed last response — discarding: %s", text[:80])
            self._resume_listening(conversation_id)
            return

        is_follow_up = bool(self.audio_stream and self.audio_stream.was_follow_up)
        addressee = self._addressee_gate.check(
            text,
            is_follow_up=is_follow_up,
            speaker_name=self._current_speaker.get("name", "unknown") if self._current_speaker else "unknown",
            had_wake_word=not is_follow_up,
        )
        if addressee.suppressed:
            logger.info("Addressee gate suppressed (%s, %.2f): '%s'",
                        addressee.reason, addressee.confidence, text[:60])
            ops_tracker.log_event("perception", "addressee_suppressed",
                                  f"Not addressed ({addressee.reason}): \"{text[:40]}\"")
            self._resume_listening(conversation_id)
            return

        with self._conv_lock:
            cancel_state = {"id": conversation_id, "cancelled": False}
            self._active_conversation = cancel_state

        traits_raw = self.engine.get_state().get("traits", [])
        if isinstance(traits_raw, list) and traits_raw:
            trait_perception.set_traits({t: 1.0 for t in traits_raw})
        elif isinstance(traits_raw, dict) and traits_raw:
            trait_perception.set_traits(traits_raw)
        modulation = trait_perception.process_event("transcription", {"text": text})
        if modulation and modulation.thought_trigger:
            from consciousness.events import KERNEL_THOUGHT
            event_bus.emit(KERNEL_THOUGHT, content=modulation.thought_trigger, tone="contemplative")

        if self.audio_stream:
            self.audio_stream.set_speaking(True)
        self._speaking_conv_id = conversation_id

        speaker_snapshot = dict(self._current_speaker) if self._current_speaker else {}
        emotion_snapshot = dict(self._current_emotion) if self._current_emotion else {}

        _HANDLE_TIMEOUT_S = 120.0

        async def _safe_handle():
            try:
                await asyncio.wait_for(
                    handle_transcription(
                        text, self.engine, self.response_gen, self.claude, self.perception,
                        self.episodes, speaker_snapshot, emotion_snapshot,
                        conversation_id=conversation_id,
                        cancel_flag=cancel_state,
                        ollama=self._ollama,
                        pi_snapshot_url=self._pi_snapshot_url,
                        brain_tts=self.brain_tts,
                        scene_context=self.get_scene_context(),
                        follow_up=is_follow_up,
                        enroll_callback=self.enroll_current_user,
                        identity_callback=self.get_identity_status,
                    ),
                    timeout=_HANDLE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.error("handle_transcription timed out after %.0fs — clearing speaking flag",
                             _HANDLE_TIMEOUT_S)
                self._cancel_speaking_safety_timer()
                self._speaking_conv_id = ""
                if self.audio_stream:
                    self.audio_stream.set_speaking(False)
                if self.perception:
                    self.perception.broadcast({
                        "type": "response_end", "text": "", "tone": "",
                        "phase": "LISTENING", "conversation_id": conversation_id,
                    })
                self._emit_conversation_response(
                    text="",
                    conversation_id=conversation_id,
                    release_status="blocked",
                    release_reason="handle_transcription_timeout",
                )
            except Exception:
                logger.exception("handle_transcription crashed — clearing speaking flag")
                self._cancel_speaking_safety_timer()
                self._speaking_conv_id = ""
                if self.audio_stream:
                    self.audio_stream.set_speaking(False)
                if self.perception:
                    self.perception.broadcast({
                        "type": "response_end", "text": "", "tone": "",
                        "phase": "LISTENING", "conversation_id": conversation_id,
                    })
                self._emit_conversation_response(
                    text="",
                    conversation_id=conversation_id,
                    release_status="blocked",
                    release_reason="handle_transcription_crash",
                )

        asyncio.get_event_loop().create_task(_safe_handle())

    # ------------------------------------------------------------------
    # Identity enrollment & query
    # ------------------------------------------------------------------

    def enroll_current_user(self, name: str) -> dict[str, Any]:
        """Enroll the current speaker's voice and face under the given name.

        Uses the last speech audio buffer for voice enrollment and
        the most recent face crops for face enrollment.
        Returns a summary dict of what was enrolled.
        """
        result: dict[str, Any] = {
            "name": name,
            "voice_enrolled": False,
            "face_enrolled": False,
            "voice_detail": "",
            "face_detail": "",
        }

        if self.speaker_id and self.speaker_id.available and self._last_speech_audio is not None:
            try:
                import numpy as np
                clips = []
                for clip in self._recent_speech_clips:
                    c = clip if isinstance(clip, np.ndarray) else np.array(clip, dtype=np.float32)
                    clips.append(c)
                if not clips:
                    audio = self._last_speech_audio
                    if not isinstance(audio, np.ndarray):
                        audio = np.array(audio, dtype=np.float32)
                    clips = [audio]
                success = self.speaker_id.enroll_speaker(name, clips, self._last_speech_sr)
                result["voice_enrolled"] = success
                result["voice_detail"] = f"enrolled from {len(clips)} clip(s)" if success else "enrollment failed"
            except Exception as exc:
                logger.warning("Voice enrollment failed for '%s': %s", name, exc)
                result["voice_detail"] = f"error: {exc}"
        elif self._last_speech_audio is None:
            result["voice_detail"] = "no recent speech audio available"
        else:
            result["voice_detail"] = "speaker ID not available"

        if self.face_id and self.face_id.available:
            try:
                crops = self.face_id.get_recent_crops(max_crops=3)
                if crops:
                    success = self.face_id.enroll_face(name, crops)
                    result["face_enrolled"] = success
                    result["face_detail"] = f"enrolled from {len(crops)} recent crop(s)" if success else "enrollment failed"
                else:
                    result["face_detail"] = "no recent face crops available"
            except Exception as exc:
                logger.warning("Face enrollment failed for '%s': %s", name, exc)
                result["face_detail"] = f"error: {exc}"
        else:
            result["face_detail"] = "face ID not available"

        logger.info("Enrollment result for '%s': voice=%s, face=%s",
                     name, result["voice_enrolled"], result["face_enrolled"])
        return result

    def reconcile_identity(self, source_name: str, target_name: str,
                             engine: Any = None) -> dict[str, Any]:
        """Merge source identity into target across all subsystems.

        Merges speaker profiles, face profiles, evidence candidates,
        re-tags memories, and writes an alias tombstone for audit.
        Returns a summary of what was merged.
        """
        result: dict[str, Any] = {
            "source": source_name, "target": target_name,
            "voice_merged": False, "face_merged": False,
            "evidence_merged": False, "memories_retagged": 0,
            "alias_recorded": False,
        }

        if self.speaker_id and self.speaker_id.available:
            if self.speaker_id.has_profile(source_name):
                if self.speaker_id.has_profile(target_name):
                    result["voice_merged"] = self.speaker_id.merge_into(source_name, target_name)
                else:
                    logger.info("Reconcile: target '%s' has no voice profile, skipping voice merge", target_name)

        if self.face_id and self.face_id.available:
            if self.face_id.has_profile(source_name):
                if self.face_id.has_profile(target_name):
                    result["face_merged"] = self.face_id.merge_into(source_name, target_name)
                else:
                    logger.info("Reconcile: target '%s' has no face profile, skipping face merge", target_name)

        try:
            from identity.evidence_accumulator import get_accumulator
            acc = get_accumulator()
            result["evidence_merged"] = acc.merge_candidate(source_name, target_name)
        except Exception:
            logger.debug("Evidence merge failed", exc_info=True)

        if engine:
            try:
                src_tag = f"speaker:{source_name.lower()}"
                tgt_tag = f"speaker:{target_name.lower()}"
                storage = getattr(engine, "_memory", None) or getattr(engine, "_storage", None)
                if storage is None:
                    from memory.storage import MemoryStorage
                    storage = MemoryStorage._instance if hasattr(MemoryStorage, "_instance") else None
                if storage and hasattr(storage, "get_all"):
                    count = 0
                    for mem in storage.get_all():
                        changed = False
                        if src_tag in (mem.tags or []):
                            mem.tags.remove(src_tag)
                            if tgt_tag not in mem.tags:
                                mem.tags.append(tgt_tag)
                            changed = True
                        if hasattr(mem, "speaker") and getattr(mem, "speaker", "") == source_name:
                            mem.speaker = target_name
                            changed = True
                        if changed:
                            count += 1
                    result["memories_retagged"] = count
                    if count:
                        logger.info("Reconcile: re-tagged %d memories from '%s' to '%s'",
                                     count, source_name, target_name)
            except Exception:
                logger.debug("Memory re-tagging failed", exc_info=True)

        try:
            import json as _json
            from pathlib import Path as _Path
            tombstone_path = _Path.home() / ".jarvis" / "identity_aliases.jsonl"
            tombstone_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": time.time(),
                "action": "reconcile",
                "source": source_name,
                "target": target_name,
                "result": {k: v for k, v in result.items() if k != "alias_recorded"},
            }
            with open(tombstone_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")
            result["alias_recorded"] = True
            logger.info("Alias tombstone recorded: '%s' → '%s'", source_name, target_name)
        except Exception:
            logger.debug("Alias tombstone write failed", exc_info=True)

        if self.identity_fusion:
            try:
                with self.identity_fusion._lock:
                    if getattr(self.identity_fusion, "_last_known", "") == source_name:
                        self.identity_fusion._last_known = target_name
                    if getattr(self.identity_fusion, "_confirmed_name", "") == source_name:
                        self.identity_fusion._confirmed_name = target_name
            except Exception:
                logger.debug("Identity fusion state update failed", exc_info=True)

        logger.info("Identity reconciliation complete: %s → %s | %s", source_name, target_name, result)
        return result

    def get_identity_status(self) -> dict[str, Any]:
        """Return current identity state for 'who am I' queries."""
        voice = dict(self._current_speaker)
        face = dict(self._current_face)
        fusion = self.identity_fusion.get_state() if hasattr(self.identity_fusion, "get_state") else {}

        voice_profiles = []
        if self.speaker_id and self.speaker_id.available:
            voice_profiles = self.speaker_id.get_known_speakers() if hasattr(self.speaker_id, "get_known_speakers") else list(getattr(self.speaker_id, "_profiles", {}).keys())

        face_profiles = []
        if self.face_id and self.face_id.available:
            face_profiles = self.face_id.get_known_faces()

        return {
            "current_voice": voice,
            "current_face": face,
            "fusion": fusion,
            "enrolled_voices": voice_profiles,
            "enrolled_faces": face_profiles,
        }

    # ------------------------------------------------------------------
    # Presence / greetings
    # ------------------------------------------------------------------

    def _get_primary_companion(self) -> str:
        """Return the name of the person with highest familiarity (primary companion)."""
        try:
            from consciousness.soul import soul_service
            best_name, best_fam = "", 0.0
            for _key, rel in soul_service.identity.relationships.items():
                if rel.familiarity > best_fam:
                    best_fam = rel.familiarity
                    best_name = rel.name
            return best_name
        except Exception:
            return ""

    def get_unknown_speaker_events(self, max_age_s: float = 600.0) -> list[dict[str, Any]]:
        """Return recent unknown speaker events for the curiosity system."""
        now = time.time()
        return [
            e for e in self._unknown_speaker_events
            if now - e["timestamp"] < max_age_s
        ]

    def clear_unknown_speaker_event(self, timestamp: float) -> None:
        """Mark an unknown speaker event as handled (asked about)."""
        self._unknown_speaker_events = deque(
            (e for e in self._unknown_speaker_events if e["timestamp"] != timestamp),
            maxlen=10,
        )

    def _on_user_arrived(self, absence_duration_s: float = 0, confidence: float = 0, **_):
        if not self.perception:
            return
        if self._gestation_active:
            gm = self.engine._gestation_manager
            if gm:
                gm.on_person_detected(True, confidence)
            return
        gm = self.engine._gestation_manager
        if gm and gm.first_contact_armed:
            gm.on_first_engagement("sustained_attention")
        minutes = int(absence_duration_s / 60)
        hour = time.localtime().tm_hour
        if hour < 12:
            time_greeting = "Good morning"
        elif hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"

        face_name = self._current_face.get("name", "unknown")
        face_known = self._current_face.get("is_known", False)
        primary = self._get_primary_companion()

        if face_known and face_name and face_name != "unknown":
            is_primary = primary and face_name.lower() == primary.lower()
            if is_primary:
                if minutes > 60:
                    msg = f"{time_greeting} {face_name}! Welcome back, you were away for about {minutes // 60} hours."
                elif minutes > 10:
                    msg = f"{time_greeting} {face_name}! Welcome back."
                else:
                    msg = f"{time_greeting} {face_name}!"
            else:
                if primary:
                    msg = f"{time_greeting} {face_name}! Nice to see you. Is {primary} around?"
                else:
                    msg = f"{time_greeting} {face_name}!"
        else:
            if minutes > 60:
                msg = f"{time_greeting}! Welcome back, you were away for about {minutes // 60} hours."
            elif minutes > 10:
                msg = f"{time_greeting}! Welcome back."
            else:
                msg = f"{time_greeting}!"

        logger.info("Proactive arrival greeting (absent %.0fs): %s", absence_duration_s, msg)
        self._speak_proactive(msg)

    def _on_presence_stable(self, present: bool = False, confidence: float = 0, **_):
        """Feed stable presence into gestation person tracking (bypasses greeting gate)."""
        if not self._gestation_active:
            return
        gm = self.engine._gestation_manager
        if gm:
            gm.on_person_detected(present, confidence)

    # ------------------------------------------------------------------
    # Proactive behavior
    # ------------------------------------------------------------------

    def evaluate_proactive(self) -> None:
        if not self.perception or not self.presence:
            return
        if self._gestation_active:
            return
        state = self.engine.get_state()
        modulation = self.engine.get_trait_modulation()
        presence_state = self.presence.get_state()

        suggestion = proactive_behavior.evaluate(
            phase=state.get("phase", "IDLE"),
            is_user_present=presence_state.get("is_present", False),
            traits=list(modulation.applied_traits),
            engagement_level=presence_state.get("attention_level", 0.5),
        )
        if suggestion:
            logger.info("Proactive suggestion: [%s] %s", suggestion.type, suggestion.message)
            msg_text = suggestion.message
            try:
                from skills.capability_gate import capability_gate
                msg_text = capability_gate.check_text(msg_text)
            except Exception:
                pass
            self._speak_proactive(msg_text)
            return

        onboarding = self.engine._check_onboarding_prompt()
        if onboarding:
            msg_text = onboarding.message
            try:
                from skills.capability_gate import capability_gate
                msg_text = capability_gate.check_text(msg_text)
            except Exception:
                pass
            self._speak_proactive(msg_text)
            return

        curiosity = self.engine._check_curiosity_question()
        if curiosity:
            msg_text = curiosity.message
            try:
                from skills.capability_gate import capability_gate
                msg_text = capability_gate.check_text(msg_text)
            except Exception:
                pass
            logger.info("Curiosity question: %s", msg_text[:80])
            self._speak_proactive(msg_text)

    # ------------------------------------------------------------------
    # Scene analysis
    # ------------------------------------------------------------------

    async def _periodic_scene_analysis(self) -> None:
        await asyncio.sleep(30)
        while True:
            try:
                now = time.time()
                user_here = self.engine._is_user_present
                interval = self._scene_interval_present if user_here else self._scene_interval_away
                if now - self._last_scene_analysis_time >= interval:
                    self._last_scene_analysis_time = now
                    await self._analyze_scene()
            except Exception as exc:
                logger.debug("Scene analysis error: %s", exc)
            await asyncio.sleep(60)

    async def _analyze_scene(self) -> None:
        if not self._pi_snapshot_url or not self._ollama:
            return
        with self._conv_lock:
            conv_active = self._active_conversation.get("id") and not self._active_conversation.get("cancelled")
        if conv_active:
            logger.debug("Skipping scene analysis — active conversation in progress")
            return

        self._scene_analysis_in_progress = True
        try:
            from tools.vision_tool import describe_scene
            description = await describe_scene(
                self._pi_snapshot_url,
                ollama_client=self._ollama,
                prompt="Describe what you see briefly. List any people, objects, or activities. Be factual.",
            )
        finally:
            self._scene_analysis_in_progress = False
            try:
                await self._ollama.unload_model(self._ollama.vision_model)
            except Exception:
                pass

        if not description or "can't see" in description.lower():
            return
        self._last_scene_description = description
        logger.info("Scene analysis: %s", description[:100])

        self._update_object_memory(description)
        self._feed_vlm_to_tracker(description)

        if self._last_scene_snapshot and self._last_scene_snapshot.display_surfaces:
            content = self._display_classifier.classify_from_description(
                description, self._last_scene_snapshot.display_surfaces,
            )
            self._last_scene_snapshot.display_content = content

    def _update_object_memory(self, description: str) -> None:
        common_objects = [
            "person", "people", "laptop", "phone", "book", "cup", "mug",
            "chair", "desk", "monitor", "keyboard", "cat", "dog", "plant",
            "bottle", "headphones", "camera", "window", "door",
        ]
        now = time.time()
        desc_lower = description.lower()
        for obj in common_objects:
            if obj in desc_lower:
                if obj in self._object_memory:
                    self._object_memory[obj]["last_seen"] = now
                    self._object_memory[obj]["count"] += 1
                else:
                    self._object_memory[obj] = {"last_seen": now, "count": 1, "first_seen": now}

    def _feed_vlm_to_tracker(self, description: str) -> None:
        """Parse a VLM description and inject enrichment detections into the scene tracker."""
        vlm_objects = [
            "laptop", "phone", "book", "cup", "mug", "chair", "desk",
            "monitor", "keyboard", "cat", "dog", "plant", "bottle",
            "headphones", "camera", "tv", "mouse", "clock", "vase",
        ]
        desc_lower = description.lower()
        enrichments: list[SceneDetection] = []
        for obj in vlm_objects:
            if obj in desc_lower:
                enrichments.append(SceneDetection(
                    label=obj, confidence=0.5, bbox=None, source="vlm",
                ))
        if enrichments and self._last_scene_snapshot:
            fw = 640
            fh = 480
            fs = self._last_scene_snapshot.to_dict().get("region_visibility")
            if fs:
                fw = 1920
                fh = 1080
            self._scene_tracker.update(enrichments, fw, fh)

    # --- Camera control ---

    def send_camera_control(self, control: str, **kwargs) -> None:
        """Send a camera control command to the Pi sensor."""
        if self.perception:
            self.perception.send_camera_control(control, **kwargs)

    def zoom_to_person(self, bbox: list[int] | None = None) -> None:
        """Zoom camera to a detected person's bounding box, or reset if None."""
        if bbox and len(bbox) == 4:
            self.send_camera_control("zoom_to", region=bbox, padding=1.8)
        else:
            self.send_camera_control("reset")

    def get_scene_snapshot(self) -> Any:
        """Public read-only accessor for the latest :class:`SceneSnapshot`.

        Returns a **deep copy** so downstream consumers (the P5 mental-
        world lane in particular) cannot accidentally mutate canonical
        perception state even if they try. Belt-and-suspenders on top of
        the frozen :class:`MentalWorld*` dataclasses the adapter emits.
        Returns ``None`` if perception has not yet produced a snapshot.
        The returned object is the canonical snapshot held by this
        orchestrator; callers MUST NOT mutate it (it is shared with the
        next perception tick). The P5 mental-world lane asserts read-only
        discipline in :mod:`brain.tests.test_mental_world_hook`.
        """
        import copy as _copy

        snap = self._last_scene_snapshot
        if snap is None:
            return None
        try:
            return _copy.deepcopy(snap)
        except Exception:
            return snap

    def get_spatial_tracks(self) -> dict[str, Any]:
        """Public read-only accessor for current :class:`SpatialTrack` map.

        Returns a shallow copy of the estimator's track map, or an empty
        dict if the estimator is not initialized. Safe to iterate.
        """
        est = self._spatial_estimator
        if est is None:
            return {}
        return est.get_tracks()

    def get_spatial_anchors(self) -> dict[str, Any]:
        """Public read-only accessor for current :class:`SpatialAnchor` map.

        Returns a shallow copy of the estimator's anchor map, or an empty
        dict if the estimator is not initialized.
        """
        est = self._spatial_estimator
        if est is None:
            return {}
        return est.get_anchors()

    def get_scene_context(self) -> str:
        parts: list[str] = []

        snap = self._last_scene_snapshot
        if snap is not None:
            physical = [e for e in snap.entities
                        if not e.is_display_surface and e.state in ("visible", "occluded")]
            if physical:
                labels = sorted({e.label for e in physical})
                parts.append(f"Physical: {', '.join(labels)}")

            if snap.display_surfaces:
                dkinds = sorted({ds.kind for ds in snap.display_surfaces})
                parts.append(f"Displays: {', '.join(dkinds)}")

            if snap.display_content:
                activities = [dc for dc in snap.display_content
                              if dc.content_type != "unknown" and dc.confidence > 0.3]
                if activities:
                    act_parts = [f"{dc.content_type}" +
                                 (f" ({dc.activity_label})" if dc.activity_label else "")
                                 for dc in activities]
                    parts.append(f"Display content: {', '.join(act_parts)}")

        if not parts and self._last_scene_description:
            parts.append(f"Visual: {self._last_scene_description[:150]}")

        if not parts and self._object_memory:
            recent = sorted(
                self._object_memory.items(),
                key=lambda x: x[1]["last_seen"],
                reverse=True,
            )[:5]
            parts.append("Objects: " + ", ".join(f"{k} (seen {v['count']}x)" for k, v in recent))

        if self.ambient_proc:
            ambient = self.ambient_proc.get_context_string()
            if ambient and ambient != "Quiet environment":
                parts.append(f"Audio: {ambient}")
        if self.presence:
            ps = self.presence.get_state()
            if ps.get("is_present"):
                dur = int(ps.get("duration_s", 0))
                if dur > 60:
                    parts.append(f"User present for {dur // 60}min")
        return "; ".join(parts) if parts else ""
