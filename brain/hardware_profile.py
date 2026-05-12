"""Hardware Profile — GPU and CPU aware configuration for different machines.

Auto-detects both GPU (VRAM) and CPU (cores, threads, RAM) capabilities,
then selects appropriate models, compute types, memory budgets, and
per-component device assignments (cuda vs cpu). Designed for portability:
a Raspberry Pi gets minimal models on CPU; a Ryzen 9 + RTX 4080 offloads
ancillary ML to the beefy CPU and reserves GPU for latency-critical work.

GPU Tiers (by VRAM):
  minimal   (0-4 GB)   — CPU-only or tiny GPU, smallest models
  low       (4-6 GB)   — GTX 1650/1660, small models
  medium    (6-8 GB)   — RTX 3060 8GB / 4060, mid-range models
  high      (8-12 GB)  — RTX 3060 12GB / 4070, large models w/ swapping
  premium   (12-16 GB) — RTX 4080 / 3080 Ti, large models always loaded
  ultra     (16-24 GB) — RTX 4090 / 3090, multiple large models loaded
  extreme   (24+ GB)   — A100 / H100, everything loaded

CPU Tiers (by thread count + RAM):
  weak      (1-4 threads)    — SBCs, cheap VPS
  standard  (4-8 threads)    — laptop i5, older desktops
  strong    (8-16 threads)   — desktop i7/Ryzen 7, M1 Pro
  beast     (16+ threads)    — Ryzen 9, Threadripper, Xeon, M2 Ultra

Override via JARVIS_GPU_TIER / JARVIS_CPU_TIER env vars.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

GpuTier = Literal[
    "minimal", "low", "medium", "high", "premium", "ultra", "extreme",
]

ALL_GPU_TIERS: tuple[GpuTier, ...] = (
    "minimal", "low", "medium", "high", "premium", "ultra", "extreme",
)

# Backwards compat alias
ALL_TIERS = ALL_GPU_TIERS

CpuTier = Literal["weak", "standard", "strong", "beast"]

ALL_CPU_TIERS: tuple[CpuTier, ...] = ("weak", "standard", "strong", "beast")


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    name: str = "none"
    vram_mb: int = 0
    driver_version: str = ""
    cuda_available: bool = False
    gpu_count: int = 0


def detect_gpu() -> GpuInfo:
    """Detect GPU via nvidia-smi (no torch dependency at import time)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return GpuInfo()

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return GpuInfo()

        parts = lines[0].split(",")
        name = parts[0].strip() if len(parts) > 0 else "unknown"
        vram_mb = int(float(parts[1].strip())) if len(parts) > 1 else 0
        driver = parts[2].strip() if len(parts) > 2 else ""

        return GpuInfo(
            name=name,
            vram_mb=vram_mb,
            driver_version=driver,
            cuda_available=True,
            gpu_count=len(lines),
        )
    except FileNotFoundError:
        return GpuInfo()
    except Exception as exc:
        logger.warning("GPU detection failed: %s", exc)
        return GpuInfo()


def vram_to_tier(vram_mb: int) -> GpuTier:
    if vram_mb < 4000:
        return "minimal"
    if vram_mb < 6000:
        return "low"
    if vram_mb < 8000:
        return "medium"
    if vram_mb < 12000:
        return "high"
    if vram_mb < 16500:
        return "premium"
    if vram_mb < 24500:
        return "ultra"
    return "extreme"


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

@dataclass
class CpuInfo:
    model: str = "unknown"
    cores: int = 1
    threads: int = 1
    ram_gb: int = 4
    max_mhz: float = 0.0


def detect_cpu() -> CpuInfo:
    """Detect CPU model, core/thread count, and RAM. Linux /proc, fallback os."""
    model = "unknown"
    cores = os.cpu_count() or 1
    threads = cores
    ram_gb = 4
    max_mhz = 0.0

    # /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        for line in cpuinfo.splitlines():
            if line.startswith("model name") and model == "unknown":
                model = line.split(":", 1)[1].strip()
            if line.startswith("cpu MHz") and max_mhz == 0.0:
                try:
                    max_mhz = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        # physical cores vs logical threads
        physical_ids: set[str] = set()
        core_ids: set[tuple[str, str]] = set()
        current_physical = "0"
        for line in cpuinfo.splitlines():
            if line.startswith("physical id"):
                current_physical = line.split(":", 1)[1].strip()
                physical_ids.add(current_physical)
            if line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
                core_ids.add((current_physical, core_id))
        if core_ids:
            cores = len(core_ids)
            # Count "processor" lines for threads
            threads = sum(1 for l in cpuinfo.splitlines() if l.startswith("processor"))
    except FileNotFoundError:
        pass

    # Try lscpu for max MHz (more reliable than /proc/cpuinfo)
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "CPU max MHz" in line:
                try:
                    max_mhz = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    except Exception:
        pass

    # RAM from /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    ram_gb = max(1, kb // (1024 * 1024))
                    break
    except FileNotFoundError:
        pass

    return CpuInfo(
        model=model, cores=cores, threads=threads,
        ram_gb=ram_gb, max_mhz=max_mhz,
    )


def cpu_to_tier(cpu: CpuInfo) -> CpuTier:
    """Map CPU capabilities to a tier. Threads are the primary signal, RAM is secondary."""
    if cpu.threads >= 16 and cpu.ram_gb >= 16:
        return "beast"
    if cpu.threads >= 8 and cpu.ram_gb >= 8:
        return "strong"
    if cpu.threads >= 4:
        return "standard"
    return "weak"


# ---------------------------------------------------------------------------
# Model profiles per tier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelProfile:
    """Which models and devices to use at a given hardware tier."""
    llm_model: str
    llm_fast_model: str
    vision_model: str
    stt_model: str
    stt_compute_type: str
    stt_device: str
    embedding_model: str
    keep_alive: str                     # Ollama keep_alive: "5m", "30m", "-1" (infinite)
    warmup_all: bool                    # Pre-load all models into VRAM at startup
    max_tokens: int = 1024
    temperature: float = 0.7
    tts_engine: str = "none"            # "kokoro_gpu", "kokoro_cpu", "none"
    tts_voice: str = "af_bella"
    enable_speaker_id: bool = False     # ECAPA-TDNN speaker identification (~300MB)
    enable_emotion: bool = False        # wav2vec2 emotion detection (~500MB)
    # Per-component device assignment — "cuda" or "cpu"
    emotion_device: str = "cpu"
    speaker_id_device: str = "cpu"
    embedding_device: str = "cpu"
    hemisphere_device: str = "cpu"
    # CPU-resident coding LLM (never touches GPU VRAM)
    coding_model: str = ""              # empty = disabled for this tier
    coding_device: str = "cpu"          # always CPU
    vram_budget_note: str = ""
    # Self-improvement coder (Qwen3-Coder-Next via llama-server, RAM-gated)
    coder_enabled: bool = False         # auto-set by RAM in resolve_coder_profile()
    coder_quant: str = ""               # GGUF quant suffix (e.g. "UD-Q4_K_XL")
    coder_min_ram_gb: int = 0           # minimum RAM for this quant


# Base GPU tier profiles. The CPU overlay in _apply_cpu_overlay() adjusts
# device fields when a strong CPU is available.
_BASE_TIER_PROFILES: dict[GpuTier, ModelProfile] = {
    "minimal": ModelProfile(
        llm_model="qwen3:1.7b",
        llm_fast_model="qwen3:1.7b",
        vision_model="",
        stt_model="tiny",
        stt_compute_type="int8",
        stt_device="cpu",
        embedding_model="all-MiniLM-L6-v2",
        keep_alive="5m",
        warmup_all=False,
        max_tokens=512,
        tts_engine="none",
        enable_speaker_id=False,
        enable_emotion=False,
        emotion_device="cpu",
        speaker_id_device="cpu",
        embedding_device="cpu",
        hemisphere_device="cpu",
        coding_model="",
        coding_device="cpu",
        vram_budget_note="CPU-only or <4GB — minimal models, no vision/TTS",
    ),
    "low": ModelProfile(
        llm_model="qwen3:4b",
        llm_fast_model="qwen3:1.7b",
        vision_model="",
        stt_model="small",
        stt_compute_type="int8",
        stt_device="cuda",
        embedding_model="all-MiniLM-L6-v2",
        keep_alive="5m",
        warmup_all=False,
        max_tokens=768,
        tts_engine="none",
        enable_speaker_id=False,
        enable_emotion=False,
        emotion_device="cpu",
        speaker_id_device="cpu",
        embedding_device="cpu",
        hemisphere_device="cpu",
        coding_model="",
        coding_device="cpu",
        vram_budget_note="4-6GB — small LLM, no vision/TTS, STT on GPU",
    ),
    "medium": ModelProfile(
        llm_model="qwen3:8b",
        llm_fast_model="qwen3:4b",
        vision_model="",
        stt_model="medium",
        stt_compute_type="int8_float16",
        stt_device="cuda",
        embedding_model="all-MiniLM-L6-v2",
        keep_alive="5m",
        warmup_all=False,
        max_tokens=1024,
        tts_engine="kokoro_cpu",
        enable_speaker_id=False,
        enable_emotion=False,
        emotion_device="cpu",
        speaker_id_device="cpu",
        embedding_device="cpu",
        hemisphere_device="cpu",
        coding_model="qwen2.5-coder:3b",
        coding_device="cpu",
        vram_budget_note="6-8GB — 8B LLM (swapped), CPU TTS, no vision",
    ),
    "high": ModelProfile(
        llm_model="qwen3:8b",
        llm_fast_model="qwen3:4b",
        vision_model="qwen2.5vl:7b",
        stt_model="large-v3-turbo",
        stt_compute_type="int8_float16",
        stt_device="cuda",
        embedding_model="all-MiniLM-L6-v2",
        keep_alive="10m",
        warmup_all=False,
        max_tokens=1024,
        tts_engine="kokoro_cpu",
        enable_speaker_id=False,
        enable_emotion=False,
        emotion_device="cuda",
        speaker_id_device="cuda",
        embedding_device="cuda",
        hemisphere_device="cuda",
        coding_model="qwen2.5-coder:7b",
        coding_device="cpu",
        vram_budget_note="8-12GB — full model set, CPU TTS, vision swaps with LLM",
    ),
    "premium": ModelProfile(
        llm_model="qwen3:8b",
        llm_fast_model="qwen3:8b",
        vision_model="qwen2.5vl:7b",
        stt_model="large-v3",
        stt_compute_type="int8_float16",
        stt_device="cuda",
        embedding_model="all-MiniLM-L6-v2",
        keep_alive="30m",
        warmup_all=True,
        max_tokens=1536,
        tts_engine="kokoro_gpu",
        enable_speaker_id=True,
        enable_emotion=True,
        emotion_device="cuda",
        speaker_id_device="cuda",
        embedding_device="cuda",
        hemisphere_device="cuda",
        coding_model="qwen2.5-coder:7b",
        coding_device="cpu",
        vram_budget_note="12-16GB — text LLM kept warm, STT coexists (~8GB combined), vision on-demand",
    ),
    "ultra": ModelProfile(
        llm_model="qwen3:14b",
        llm_fast_model="qwen3:8b",
        vision_model="qwen2.5vl:7b",
        stt_model="large-v3",
        stt_compute_type="float16",
        stt_device="cuda",
        embedding_model="all-MiniLM-L6-v2",
        keep_alive="-1s",
        warmup_all=True,
        max_tokens=2048,
        tts_engine="kokoro_gpu",
        enable_speaker_id=True,
        enable_emotion=True,
        emotion_device="cuda",
        speaker_id_device="cuda",
        embedding_device="cuda",
        hemisphere_device="cuda",
        coding_model="qwen2.5-coder:7b",
        coding_device="cpu",
        vram_budget_note="16-24GB — 14B LLM, GPU TTS, full perception",
    ),
    "extreme": ModelProfile(
        llm_model="qwen3:32b",
        llm_fast_model="qwen3:14b",
        vision_model="qwen2.5vl:7b",
        stt_model="large-v3",
        stt_compute_type="float16",
        stt_device="cuda",
        embedding_model="all-MiniLM-L6-v2",
        keep_alive="-1s",
        warmup_all=True,
        max_tokens=4096,
        tts_engine="kokoro_gpu",
        enable_speaker_id=True,
        enable_emotion=True,
        emotion_device="cuda",
        speaker_id_device="cuda",
        embedding_device="cuda",
        hemisphere_device="cuda",
        coding_model="qwen2.5-coder:7b",
        coding_device="cpu",
        vram_budget_note="24GB+ — 32B LLM, GPU TTS, full perception",
    ),
}

# Public alias — resolve_profile() replaces the ModelProfile with a
# CPU-overlay-adjusted copy, but callers that import TIER_PROFILES
# directly get the base (GPU-only) defaults.
TIER_PROFILES = _BASE_TIER_PROFILES


def _apply_cpu_overlay(base: ModelProfile, cpu_tier: CpuTier, gpu_tier: GpuTier) -> ModelProfile:
    """Adjust device assignments based on CPU strength.

    On a strong/beast CPU, offload ancillary ML (emotion, speaker ID,
    embeddings, hemisphere training) to CPU — frees ~1-1.5 GB VRAM.
    On a weak/standard CPU, keep them on GPU if VRAM allows.
    """
    if cpu_tier in ("beast", "strong"):
        return ModelProfile(
            **{
                **{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()},
                "emotion_device": "cpu",
                "speaker_id_device": "cpu",
                "embedding_device": "cpu",
                "hemisphere_device": "cpu",
            }
        )
    # weak/standard CPU: keep on GPU where VRAM allows (high+ tiers)
    return base


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    """Resolved hardware profile for this machine."""
    gpu: GpuInfo
    cpu: CpuInfo
    tier: GpuTier
    cpu_tier: CpuTier
    models: ModelProfile
    tier_source: str        # "auto", "env_override", "config_override"
    cpu_tier_source: str    # "auto", "env_override"

    def summary(self) -> str:
        m = self.models
        gpu_part = (
            f"GPU: {self.gpu.name} ({self.gpu.vram_mb}MB VRAM) | Tier: {self.tier}"
            if self.gpu.cuda_available
            else "GPU: none | Tier: minimal"
        )
        cpu_part = (
            f"CPU: {self.cpu.model} ({self.cpu.cores}c/{self.cpu.threads}t, "
            f"{self.cpu.ram_gb}GB RAM) | Tier: {self.cpu_tier}"
        )
        workload = (
            f"STT={m.stt_device} LLM=cuda TTS={m.tts_engine} "
            f"Emotion={m.emotion_device} SpeakerID={m.speaker_id_device} "
            f"Embed={m.embedding_device} Hemisphere={m.hemisphere_device}"
        )
        return f"{gpu_part} | {cpu_part} | {workload}"

    def to_dict(self) -> dict[str, Any]:
        m = self.models
        return {
            "gpu_name": self.gpu.name,
            "vram_mb": self.gpu.vram_mb,
            "tier": self.tier,
            "tier_source": self.tier_source,
            "cpu_model": self.cpu.model,
            "cpu_cores": self.cpu.cores,
            "cpu_threads": self.cpu.threads,
            "cpu_ram_gb": self.cpu.ram_gb,
            "cpu_tier": self.cpu_tier,
            "cpu_tier_source": self.cpu_tier_source,
            "llm_model": m.llm_model,
            "fast_model": m.llm_fast_model,
            "vision_model": m.vision_model,
            "stt_model": m.stt_model,
            "stt_compute": m.stt_compute_type,
            "stt_device": m.stt_device,
            "tts_engine": m.tts_engine,
            "tts_voice": m.tts_voice,
            "enable_speaker_id": m.enable_speaker_id,
            "enable_emotion": m.enable_emotion,
            "emotion_device": m.emotion_device,
            "speaker_id_device": m.speaker_id_device,
            "embedding_device": m.embedding_device,
            "hemisphere_device": m.hemisphere_device,
            "keep_alive": m.keep_alive,
            "warmup_all": m.warmup_all,
            "max_tokens": m.max_tokens,
            "coder_enabled": m.coder_enabled,
            "coder_quant": m.coder_quant,
            "coder_min_ram_gb": m.coder_min_ram_gb,
            "note": m.vram_budget_note,
        }


def resolve_profile(
    tier_override: str | None = None,
    cpu_tier_override: str | None = None,
) -> HardwareProfile:
    """Detect GPU + CPU, resolve tiers, apply CPU overlay, return profile.

    Priority (GPU):
      1. tier_override argument
      2. JARVIS_GPU_TIER env var
      3. Auto-detected from VRAM

    Priority (CPU):
      1. cpu_tier_override argument
      2. JARVIS_CPU_TIER env var
      3. Auto-detected from cores/threads/RAM
    """
    gpu = detect_gpu()
    cpu = detect_cpu()

    # --- GPU tier ---
    env_gpu = os.getenv("JARVIS_GPU_TIER", "").lower().strip()
    gpu_ov = tier_override or env_gpu
    if gpu_ov and gpu_ov in ALL_GPU_TIERS:
        gpu_tier: GpuTier = gpu_ov  # type: ignore[assignment]
        gpu_src = "env_override" if env_gpu == gpu_ov else "config_override"
    else:
        gpu_tier = vram_to_tier(gpu.vram_mb)
        gpu_src = "auto"
        if gpu_ov and gpu_ov not in ALL_GPU_TIERS:
            logger.warning("Unknown GPU tier '%s' — auto-detecting", gpu_ov)

    # --- CPU tier ---
    env_cpu = os.getenv("JARVIS_CPU_TIER", "").lower().strip()
    cpu_ov = cpu_tier_override or env_cpu
    if cpu_ov and cpu_ov in ALL_CPU_TIERS:
        c_tier: CpuTier = cpu_ov  # type: ignore[assignment]
        cpu_src = "env_override"
    else:
        c_tier = cpu_to_tier(cpu)
        cpu_src = "auto"
        if cpu_ov and cpu_ov not in ALL_CPU_TIERS:
            logger.warning("Unknown CPU tier '%s' — auto-detecting", cpu_ov)

    base_profile = _BASE_TIER_PROFILES[gpu_tier]
    profile = _apply_cpu_overlay(base_profile, c_tier, gpu_tier)

    # Apply RAM-aware coder settings
    coder_ok, coder_quant, _coder_fn, coder_min_ram = resolve_coder_profile(cpu.ram_gb)
    if coder_ok:
        profile = ModelProfile(
            **{
                **{f.name: getattr(profile, f.name) for f in profile.__dataclass_fields__.values()},
                "coder_enabled": True,
                "coder_quant": coder_quant,
                "coder_min_ram_gb": coder_min_ram,
            }
        )

    hw = HardwareProfile(
        gpu=gpu, cpu=cpu, tier=gpu_tier, cpu_tier=c_tier,
        models=profile, tier_source=gpu_src, cpu_tier_source=cpu_src,
    )
    logger.info("Hardware profile: %s", hw.summary())
    return hw


# ---------------------------------------------------------------------------
# Coder model RAM tiers (Qwen3-Coder-Next GGUF quants)
# ---------------------------------------------------------------------------

_CODER_RAM_TIERS: list[tuple[int, str, str, int]] = [
    # (min_ram_gb, quant_suffix, gguf_filename, model_size_gb)
    (56, "UD-Q4_K_XL", "Qwen3-Coder-Next-UD-Q4_K_XL.gguf", 46),
    (48, "UD-IQ4_XS",  "Qwen3-Coder-Next-UD-IQ4_XS.gguf",  38),
    (32, "UD-IQ2_M",   "Qwen3-Coder-Next-UD-IQ2_M.gguf",    25),
]

CODER_MIN_RAM_GB = 32


def resolve_coder_profile(ram_gb: int) -> tuple[bool, str, str, int]:
    """Select the best coder quant for available system RAM.

    Returns (enabled, quant_suffix, gguf_filename, min_ram_gb).
    Leaves ~10-15GB headroom for OS + Jarvis + Ollama.
    """
    for min_ram, quant, filename, model_size in _CODER_RAM_TIERS:
        if ram_gb >= min_ram:
            return True, quant, filename, min_ram
    return False, "", "", 0


# Singleton — resolved once on import
_profile: HardwareProfile | None = None


def get_hardware_profile() -> HardwareProfile:
    global _profile
    if _profile is None:
        _profile = resolve_profile()
    return _profile
