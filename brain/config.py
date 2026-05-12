"""Jarvis Brain — Configuration.

Model selection is hardware-aware: on startup, the GPU is detected and a
VRAM tier is resolved (see hardware_profile.py). Models, compute types,
and keep-alive durations are set automatically for each tier. Any value
can be overridden via environment variables or .env — explicit settings
always win over the hardware profile defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


_LAPTOP_ROOT = str(Path(__file__).resolve().parent)
_JARVIS_DIR = Path(os.environ.get("JARVIS_HOME", str(Path.home() / ".jarvis")))
_MODELS_DIR = _JARVIS_DIR / "models"


def get_models_dir() -> Path:
    """Return the consolidated model storage directory (~/.jarvis/models/)."""
    return _MODELS_DIR


def _load_dotenv(path: str) -> None:
    """Load a .env file into os.environ (no dependency required)."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            raw = value.strip()
            if (raw.startswith("'") and raw.endswith("'")) or \
               (raw.startswith('"') and raw.endswith('"')):
                value = raw[1:-1]
            else:
                if "#" in raw:
                    raw = raw[:raw.index("#")].rstrip()
                value = raw
            if key and key not in os.environ:
                os.environ[key] = value


def _clean_env_value(value: str) -> str:
    """Normalize values from either our dotenv loader or systemd EnvironmentFile."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if (raw.startswith("'") and raw.endswith("'")) or \
       (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1].strip()
    if "#" in raw:
        raw = raw[:raw.index("#")].rstrip()
    return raw.strip()


def _env(name: str, default: str = "") -> str:
    """Read an environment value with systemd EnvironmentFile comments stripped."""
    return _clean_env_value(os.getenv(name, default))


_load_dotenv(os.path.join(_LAPTOP_ROOT, ".env"))


# Resolve hardware profile *after* dotenv so JARVIS_GPU_TIER can be in .env
from hardware_profile import get_hardware_profile as _get_hw_profile

_hw = _get_hw_profile()
_mp = _hw.models  # ModelProfile for this tier


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = _mp.llm_model
    fast_model: str = _mp.llm_fast_model
    vision_model: str = _mp.vision_model
    temperature: float = _mp.temperature
    max_tokens: int = _mp.max_tokens
    keep_alive: str = _mp.keep_alive
    warmup_all: bool = _mp.warmup_all


class ClaudeConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024


class PerceptionConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9100
    pi_host: str = ""
    pi_ui_port: int = 8080


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9200


class MemoryConfig(BaseModel):
    max_capacity: int = 500
    persist_path: str = ""
    persist_interval_s: float = 60.0
    vector_db_path: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_device: str = _mp.embedding_device
    semantic_search_top_k: int = 5


class MultimodalConfig(BaseModel):
    enabled: bool = False
    model_name: str = "microsoft/Phi-4-multimodal-instruct"
    device: str = "cuda"
    max_tokens: int = 512


class EpisodicConfig(BaseModel):
    enabled: bool = True
    max_episodes: int = 200
    persist_path: str = ""


class STTConfig(BaseModel):
    model: str = _mp.stt_model
    compute_type: str = _mp.stt_compute_type
    device: str = _mp.stt_device


class TTSConfig(BaseModel):
    engine: str = _mp.tts_engine          # "kokoro_gpu", "kokoro_cpu", "none"
    voice: str = _mp.tts_voice
    speed: float = 1.0
    oracle_solemn_voice: str = ""
    oracle_empathetic_voice: str = ""
    oracle_urgent_voice: str = ""
    oracle_observational_voice: str = ""
    oracle_guarded_voice: str = ""
    model_path: str = ""                  # auto-resolved to brain/models/kokoro-v1.0.onnx


class SpeakerIDConfig(BaseModel):
    enabled: bool = _mp.enable_speaker_id
    device: str = _mp.speaker_id_device


class WakeWordConfig(BaseModel):
    keyword: str = "hey_jarvis"
    threshold: float = 0.5
    speaking_threshold_mult: float = 2.0
    speaking_hits_required: int = 3
    cooldown_s: float = 2.0
    silence_duration_s: float = 2.0
    max_record_s: float = 30.0
    follow_up_timeout_s: float = 4.0


class EmotionConfig(BaseModel):
    enabled: bool = _mp.enable_emotion
    device: str = _mp.emotion_device


class HemisphereConfig(BaseModel):
    device: str = _mp.hemisphere_device


class CodingConfig(BaseModel):
    model: str = _mp.coding_model
    backend: str = "ollama-cpu"         # "ollama-cpu" or "llama-cpp"
    ollama_host: str = "http://localhost:11435"
    max_tokens: int = 8192
    temperature: float = 0.3
    enabled: bool = bool(_mp.coding_model)


class CoderConfig(BaseModel):
    """Self-improvement coder: Qwen3-Coder-Next via on-demand llama-server.

    Model quant is auto-selected based on system RAM:
      56GB+ → UD-Q4_K_XL (~46GB)
      48GB+ → UD-IQ4_XS  (~38GB)
      32GB+ → UD-IQ2_M   (~25GB)
      <32GB → disabled
    """
    enabled: bool = False
    model_path: str = ""
    server_port: int = 8081
    ctx_size: int = 16384
    gpu_layers: int = 0
    llama_server_bin: str = "llama-server"
    max_tokens: int = 16384
    temperature: float = 0.3


class AutonomyConfig(BaseModel):
    enabled: bool = True
    level: int = 1  # L0=propose, L1=research, L2=safe-apply, L3=full
    max_research_per_hour: int = 8
    max_research_per_day: int = 30
    max_web_per_hour: int = 3
    process_interval_s: float = 30.0
    topic_cooldown_s: float = 600.0
    repetition_threshold: int = 3
    metric_trigger_enabled: bool = True
    delta_tracking_enabled: bool = True
    l2_required_positive_deltas: int = 10
    l2_min_win_rate: float = 0.4
    l2_allowed_paths: list[str] = [
        "brain/dashboard/",
        "brain/tests/",
    ]
    l3_required_positive_deltas: int = 25
    l3_min_win_rate: float = 0.5


class ResearchConfig(BaseModel):
    enabled: bool = True
    s2_api_key: str = ""
    crossref_mailto: str = ""
    min_content_chars: int = 200
    max_content_chars: int = 10000
    fetch_tldr: bool = True
    fetch_open_access: bool = True
    fetch_full_text: bool = True
    llm_study: bool = True
    enrich_on_ingest: bool = True
    detail_fetch_timeout: int = 10


class SyntheticExerciseConfig(BaseModel):
    enabled: bool = False
    rate_limit_per_hour: int = 120


class LanguageRuntimeConfig(BaseModel):
    enable_promotion_bridge: bool = False
    rollout_mode: str = "off"  # off | canary | full
    canary_classes: list[str] = ["self_introspection"]


class GestationConfig(BaseModel):
    enabled: bool = True
    min_duration_s: float = 7200.0       # 2 hour minimum
    max_duration_s: float = 172800.0     # 48 hour safety cap
    research_rate_limit: int = 30        # aggressive learning during gestation
    web_rate_limit: int = 10             # elevated for gestation (normal: 3/hr)
    self_study_batch_size: int = 5       # directives per gestation tick
    min_memories_for_ready: int = 50     # readiness: accumulated memories
    min_research_jobs: int = 15          # readiness: completed research
    min_measured_deltas: int = 10        # readiness: loop integrity
    readiness_threshold: float = 0.8     # composite score to graduate
    readiness_threshold_waiting: float = 0.6  # lower if person waiting
    vision_triggers_birth: bool = True   # person detection can trigger transition
    person_sustained_s: float = 30.0     # sustained attention before birth greeting


class BrainConfig(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    claude: ClaudeConfig = ClaudeConfig()
    perception: PerceptionConfig = PerceptionConfig()
    dashboard: DashboardConfig = DashboardConfig()
    memory: MemoryConfig = MemoryConfig()
    multimodal: MultimodalConfig = MultimodalConfig()
    episodic: EpisodicConfig = EpisodicConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    speaker_id: SpeakerIDConfig = SpeakerIDConfig()
    emotion: EmotionConfig = EmotionConfig()
    hemisphere: HemisphereConfig = HemisphereConfig()
    coding: CodingConfig = CodingConfig()
    coder: CoderConfig = CoderConfig()
    autonomy: AutonomyConfig = AutonomyConfig()
    research: ResearchConfig = ResearchConfig()
    gestation: GestationConfig = GestationConfig()
    synthetic_exercise: SyntheticExerciseConfig = SyntheticExerciseConfig()
    language_runtime: LanguageRuntimeConfig = LanguageRuntimeConfig()
    wake_word: WakeWordConfig = WakeWordConfig()
    enable_perception: bool = True
    enable_screen_awareness: bool = False
    enable_dashboard: bool = True
    enable_claude: bool = False
    enable_self_improve: bool = False
    self_improve_dry_run: bool = False
    self_improve_stage: int = 0
    stt_model: str = _mp.stt_model             # legacy compat
    gpu_tier: str = _hw.tier
    gpu_name: str = _hw.gpu.name
    gpu_vram_mb: int = _hw.gpu.vram_mb
    cpu_tier: str = _hw.cpu_tier
    cpu_model: str = _hw.cpu.model
    cpu_cores: int = _hw.cpu.cores
    cpu_threads: int = _hw.cpu.threads
    cpu_ram_gb: int = _hw.cpu.ram_gb
    log_level: str = "INFO"
    project_root: str = _LAPTOP_ROOT

    # Matrix Protocol — configurable trigger aliases
    matrix_trigger_aliases: list[str] = [
        "use the matrix to learn",
        "matrix learn",
        "matrix style",
        "matrix-style",
    ]

    def model_post_init(self, __context):
        jarvis_dir = str(_JARVIS_DIR)
        if not self.memory.persist_path:
            self.memory.persist_path = os.path.join(jarvis_dir, "memories.json")
        if not self.memory.vector_db_path:
            self.memory.vector_db_path = os.path.join(jarvis_dir, "vector_memory.db")
        if not self.episodic.persist_path:
            self.episodic.persist_path = os.path.join(jarvis_dir, "episodes.json")

        api_key = _env("ANTHROPIC_API_KEY")
        if api_key:
            self.claude.api_key = api_key

        # Env vars override hardware-profile defaults
        ollama_host = _env("OLLAMA_HOST")
        if ollama_host:
            self.ollama.host = ollama_host
        ollama_model = _env("OLLAMA_MODEL")
        if ollama_model:
            self.ollama.model = ollama_model
        fast_model = _env("OLLAMA_FAST_MODEL")
        if fast_model:
            self.ollama.fast_model = fast_model
        vision_model = _env("OLLAMA_VISION_MODEL")
        if vision_model:
            self.ollama.vision_model = vision_model
        keep_alive = _env("OLLAMA_KEEP_ALIVE")
        if keep_alive:
            self.ollama.keep_alive = keep_alive

        perception_port = _env("PERCEPTION_PORT")
        if perception_port:
            self.perception.port = int(perception_port)
        pi_host = _env("PI_HOST")
        if pi_host:
            self.perception.pi_host = pi_host
        pi_ui_port = _env("PI_UI_PORT")
        if pi_ui_port:
            self.perception.pi_ui_port = int(pi_ui_port)

        dashboard_port = _env("DASHBOARD_PORT")
        if dashboard_port:
            self.dashboard.port = int(dashboard_port)

        if _env("ENABLE_PERCEPTION") == "false":
            self.enable_perception = False
        if _env("ENABLE_SCREEN") == "true":
            self.enable_screen_awareness = True
        if _env("ENABLE_CLAUDE") == "true":
            self.enable_claude = True
        if _env("ENABLE_SELF_IMPROVE") == "true":
            self.enable_self_improve = True
        if _env("SELF_IMPROVE_DRY_RUN") == "true":
            self.self_improve_dry_run = True
        si_stage = _env("SELF_IMPROVE_STAGE")
        if si_stage:
            try:
                self.self_improve_stage = int(si_stage)
            except ValueError:
                pass
        stt_model = _env("STT_MODEL")
        if stt_model:
            self.stt_model = stt_model
            self.stt.model = stt_model
        stt_compute = _env("STT_COMPUTE_TYPE")
        if stt_compute:
            self.stt.compute_type = stt_compute
        if _env("ENABLE_MULTIMODAL") == "true":
            self.multimodal.enabled = True

        tts_engine = _env("TTS_ENGINE")
        if tts_engine:
            self.tts.engine = tts_engine
        tts_voice = _env("TTS_VOICE")
        if tts_voice:
            self.tts.voice = tts_voice
        tts_speed = _env("TTS_SPEED")
        if tts_speed:
            self.tts.speed = float(tts_speed)
        solemn_voice = _env("TTS_VOICE_SOLEMN")
        if solemn_voice:
            self.tts.oracle_solemn_voice = solemn_voice
        empathetic_voice = _env("TTS_VOICE_EMPATHETIC")
        if empathetic_voice:
            self.tts.oracle_empathetic_voice = empathetic_voice
        urgent_voice = _env("TTS_VOICE_URGENT")
        if urgent_voice:
            self.tts.oracle_urgent_voice = urgent_voice
        observational_voice = _env("TTS_VOICE_OBSERVATIONAL")
        if observational_voice:
            self.tts.oracle_observational_voice = observational_voice
        guarded_voice = _env("TTS_VOICE_GUARDED")
        if guarded_voice:
            self.tts.oracle_guarded_voice = guarded_voice
        if _env("ENABLE_SPEAKER_ID") == "true":
            self.speaker_id.enabled = True
        elif _env("ENABLE_SPEAKER_ID") == "false":
            self.speaker_id.enabled = False
        if _env("ENABLE_EMOTION") == "true":
            self.emotion.enabled = True
        elif _env("ENABLE_EMOTION") == "false":
            self.emotion.enabled = False

        # Coding LLM overrides
        coding_model = _env("CODING_MODEL")
        if coding_model:
            self.coding.model = coding_model
            self.coding.enabled = True
        coding_host = _env("CODING_OLLAMA_HOST")
        if coding_host:
            self.coding.ollama_host = coding_host
        coding_backend = _env("CODING_BACKEND")
        if coding_backend:
            self.coding.backend = coding_backend

        # Self-improvement coder overrides (Qwen3-Coder-Next via llama-server)
        import shutil
        from hardware_profile import resolve_coder_profile, CODER_MIN_RAM_GB
        ram_gb = _hw.cpu.ram_gb if hasattr(_hw, "cpu") else 0
        coder_ok, coder_quant, coder_filename, coder_min_ram = resolve_coder_profile(ram_gb)

        if not self.coder.model_path:
            self.coder.model_path = str(
                Path.home() / ".jarvis/models" / coder_filename
            ) if coder_filename else ""

        coder_path = _env("CODER_MODEL_PATH")
        if coder_path:
            self.coder.model_path = str(Path(coder_path).expanduser())
        coder_port = _env("CODER_SERVER_PORT")
        if coder_port:
            self.coder.server_port = int(coder_port)
        coder_ctx = _env("CODER_CTX_SIZE")
        if coder_ctx:
            self.coder.ctx_size = int(coder_ctx)
        coder_gpu = _env("CODER_GPU_LAYERS")
        if coder_gpu:
            self.coder.gpu_layers = int(coder_gpu)
        coder_bin = _env("CODER_LLAMA_SERVER")
        if coder_bin:
            self.coder.llama_server_bin = coder_bin
        elif shutil.which(self.coder.llama_server_bin) is None:
            local_build = Path.home() / ".local/share/llama.cpp/build/bin/llama-server"
            if local_build.exists():
                self.coder.llama_server_bin = str(local_build)
        if _env("ENABLE_CODER_MODEL") == "true":
            if ram_gb and ram_gb < CODER_MIN_RAM_GB:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "ENABLE_CODER_MODEL=true but system has only %dGB RAM "
                    "(minimum %dGB) — coder may OOM", ram_gb, CODER_MIN_RAM_GB)
            self.coder.enabled = True
        elif _env("ENABLE_CODER_MODEL") == "false":
            self.coder.enabled = False
        elif not self.coder.enabled:
            model_exists = bool(self.coder.model_path) and Path(self.coder.model_path).expanduser().exists()
            binary_exists = shutil.which(self.coder.llama_server_bin) is not None or Path(self.coder.llama_server_bin).exists()
            self.coder.enabled = coder_ok and model_exists and binary_exists

        # Autonomy overrides
        if _env("ENABLE_AUTONOMY") == "false":
            self.autonomy.enabled = False
        elif _env("ENABLE_AUTONOMY") == "true":
            self.autonomy.enabled = True

        # Research overrides
        s2_key = _env("S2_API_KEY")
        if s2_key:
            self.research.s2_api_key = s2_key
        cr_mailto = _env("CROSSREF_MAILTO")
        if cr_mailto:
            self.research.crossref_mailto = cr_mailto
        if _env("RESEARCH_FETCH_OPEN_ACCESS") == "true":
            self.research.fetch_open_access = True
        if _env("RESEARCH_FETCH_OPEN_ACCESS") == "false":
            self.research.fetch_open_access = False
        if _env("RESEARCH_FETCH_FULL_TEXT") == "false":
            self.research.fetch_full_text = False
        if _env("RESEARCH_LLM_STUDY") == "false":
            self.research.llm_study = False

        # Gestation overrides
        if _env("ENABLE_GESTATION") == "false":
            self.gestation.enabled = False
        gestation_min = _env("GESTATION_MIN_DURATION")
        if gestation_min:
            self.gestation.min_duration_s = float(gestation_min)

        # Synthetic exercise
        if _env("ENABLE_SYNTHETIC_EXERCISE") == "true":
            self.synthetic_exercise.enabled = True
        synth_rate = _env("SYNTHETIC_EXERCISE_RATE_LIMIT")
        if synth_rate:
            self.synthetic_exercise.rate_limit_per_hour = int(synth_rate)

        # Phase D guarded runtime-consumption bridge (default off).
        if _env("ENABLE_LANGUAGE_RUNTIME_BRIDGE") == "true":
            self.language_runtime.enable_promotion_bridge = True
        elif _env("ENABLE_LANGUAGE_RUNTIME_BRIDGE") == "false":
            self.language_runtime.enable_promotion_bridge = False
        runtime_mode = _env("LANGUAGE_RUNTIME_ROLLOUT_MODE").lower()
        if runtime_mode:
            self.language_runtime.rollout_mode = runtime_mode if runtime_mode in {"off", "canary", "full"} else "off"
        canary_classes = _env("LANGUAGE_RUNTIME_CANARY_CLASSES")
        if canary_classes:
            parsed = [c.strip() for c in canary_classes.split(",") if c.strip()]
            if parsed:
                self.language_runtime.canary_classes = parsed

        # Per-component device overrides (cpu / cuda)
        emotion_dev = _env("EMOTION_DEVICE")
        if emotion_dev:
            self.emotion.device = emotion_dev
        speaker_id_dev = _env("SPEAKER_ID_DEVICE")
        if speaker_id_dev:
            self.speaker_id.device = speaker_id_dev
        embedding_dev = _env("EMBEDDING_DEVICE")
        if embedding_dev:
            self.memory.embedding_device = embedding_dev
        hemisphere_dev = _env("HEMISPHERE_DEVICE")
        if hemisphere_dev:
            self.hemisphere.device = hemisphere_dev
