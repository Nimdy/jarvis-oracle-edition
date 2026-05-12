"""Brain-side speaker identification using SpeechBrain ECAPA-TDNN on CUDA.

Extracts speaker embeddings from audio clips received from the Pi,
compares against stored speaker profiles, and emits PERCEPTION_SPEAKER_IDENTIFIED
events. New speakers are assigned temporary IDs until explicitly labelled.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SPEECHBRAIN_AVAILABLE = False
try:
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        from speechbrain.pretrained import EncoderClassifier
        SPEECHBRAIN_AVAILABLE = True
    except (ImportError, AttributeError):
        pass


class SpeakerIdentifier:
    """Identifies speakers from audio embeddings using ECAPA-TDNN."""

    SIMILARITY_THRESHOLD = 0.50
    SCORE_EMA_ALPHA = 0.35
    PROFILES_FILENAME = "speakers.json"

    def __init__(
        self,
        device: str = "cuda",
        persist_dir: str = "",
    ) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self.available = False
        self._profiles: dict[str, dict] = {}
        self._next_unknown_id = 1
        self._last_embedding: np.ndarray | None = None
        self._score_ema: dict[str, float] = {}

        if not persist_dir:
            persist_dir = str(Path.home() / ".jarvis")
        self._profiles_path = Path(persist_dir) / self.PROFILES_FILENAME

        if not SPEECHBRAIN_AVAILABLE:
            logger.info("speechbrain not installed — speaker ID disabled")
            return

        try:
            import torch
            use_device = device if torch.cuda.is_available() else "cpu"

            self._model = self._load_ecapa_model(use_device)
            self.available = True
            self._load_profiles()
            logger.info("SpeakerID ready: ECAPA-TDNN on %s (%d profiles loaded)",
                        use_device, len(self._profiles))
        except Exception as exc:
            logger.error("Failed to load speaker ID model: %s", exc)

    @staticmethod
    def _load_ecapa_model(device: str):
        """Load ECAPA-TDNN with compatibility patches for SpeechBrain 1.0+ / huggingface_hub."""
        import huggingface_hub

        _orig_hf_download = huggingface_hub.hf_hub_download

        def _compat_hf_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            filename = kwargs.get("filename") or (args[1] if len(args) > 1 else "")
            local_kwargs = dict(kwargs)
            local_kwargs["local_files_only"] = True
            try:
                return _orig_hf_download(*args, **local_kwargs)
            except Exception:
                pass
            try:
                return _orig_hf_download(*args, **kwargs)
            except Exception as e:
                if "404" in str(e) and "custom" in str(filename):
                    raise ValueError(f"{filename} not found in repo (expected)")
                raise

        huggingface_hub.hf_hub_download = _compat_hf_download
        try:
            from config import get_models_dir
            save_dir = str(get_models_dir() / "ecapa-tdnn")
            return EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=save_dir,
                run_opts={"device": device},
            )
        finally:
            huggingface_hub.hf_hub_download = _orig_hf_download

    def _load_profiles(self) -> None:
        if self._profiles_path.exists():
            try:
                data = json.loads(self._profiles_path.read_text())
                from identity.name_validator import is_valid_person_name
                cleaned = {}
                for name, profile in data.items():
                    if not is_valid_person_name(name):
                        logger.warning("Removing invalid speaker profile on load: %r", name)
                        continue
                    profile["embedding"] = np.array(profile["embedding"], dtype=np.float32)
                    cleaned[name] = profile
                self._profiles = cleaned
                if len(cleaned) < len(data):
                    self._save_profiles()
            except Exception as exc:
                logger.warning("Failed to load speaker profiles: %s", exc)

    def _save_profiles(self) -> None:
        try:
            self._profiles_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for name, profile in self._profiles.items():
                data[name] = {
                    **profile,
                    "embedding": profile["embedding"].tolist()
                    if isinstance(profile["embedding"], np.ndarray)
                    else profile["embedding"],
                }
            self._profiles_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("Failed to save speaker profiles: %s", exc)

    def identify_b64(self, audio_b64: str, sample_rate: int = 16000) -> dict:
        """Identify speaker from base64-encoded int16 PCM audio.

        Returns dict with keys: name, confidence, is_known, embedding_id.
        """
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
        except Exception as exc:
            logger.error("Speaker ID audio decode failed: %s", exc)
            return {"name": "unknown", "confidence": 0.0, "is_known": False, "embedding_id": "", "raw_score": 0.0, "closest_match": ""}

        return self.identify(audio_f32, sample_rate)

    EMA_ALPHA = 0.1
    EMA_MIN_CONFIDENCE = 0.55
    ADAPT_MIN_SCORE = 0.35
    ADAPT_ALPHA_SCALE = 0.25

    def _extract_embedding(self, audio_f32: np.ndarray) -> np.ndarray | None:
        """Extract speaker embedding from float32 audio. Returns unit-norm vector."""
        import torch
        waveform = torch.tensor(audio_f32).unsqueeze(0)
        raw = self._model.encode_batch(waveform).squeeze().cpu().numpy()
        norm = np.linalg.norm(raw)
        if norm < 1e-8:
            return None
        return (raw / norm).astype(np.float32)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def identify(self, audio_f32: np.ndarray, sample_rate: int = 16000, origin: str = "mic") -> dict:
        """Identify speaker from float32 audio waveform.

        Returns dict with keys: name, confidence (smoothed), is_known,
        embedding_id, raw_score (unsmoothed cosine), closest_match (profile
        name of best match regardless of threshold).
        """
        _unknown = {"name": "unknown", "confidence": 0.0, "is_known": False, "embedding_id": "", "raw_score": 0.0, "closest_match": ""}
        if not self.available or self._model is None:
            return _unknown

        if len(audio_f32) < sample_rate * 0.5:
            return _unknown

        with self._lock:
            try:
                embedding = self._extract_embedding(audio_f32)
                if embedding is None:
                    return _unknown

                self._last_embedding = embedding

                best_name = "unknown"
                best_score = 0.0
                best_raw = 0.0
                is_known = False

                for name, profile in self._profiles.items():
                    stored = profile["embedding"]
                    if isinstance(stored, list):
                        stored = np.array(stored, dtype=np.float32)
                    raw = self._cosine_sim(embedding, stored)

                    prev = self._score_ema.get(name)
                    if prev is None:
                        smoothed = raw
                    else:
                        smoothed = self.SCORE_EMA_ALPHA * raw + (1 - self.SCORE_EMA_ALPHA) * prev
                    self._score_ema[name] = smoothed

                    if smoothed > best_score:
                        best_score = smoothed
                        best_raw = raw
                        best_name = name

                closest_match = best_name if best_name != "unknown" else ""
                is_known = best_score >= self.SIMILARITY_THRESHOLD
                if is_known:
                    # Truth-boundary guard: synthetic TTS audio must not drift real
                    # enrolled profiles or persist to ~/.jarvis/speakers.json.
                    # Recognition (is_known, name, closest_match, distillation signal)
                    # is preserved; only the persistent profile-mutation path is gated.
                    if origin != "synthetic":
                        self._profiles[best_name]["last_seen"] = time.time()
                        self._profiles[best_name]["interaction_count"] = (
                            self._profiles[best_name].get("interaction_count", 0) + 1
                        )
                        if best_raw >= self.EMA_MIN_CONFIDENCE:
                            stored = self._profiles[best_name]["embedding"]
                            if isinstance(stored, list):
                                stored = np.array(stored, dtype=np.float32)
                            updated = (1 - self.EMA_ALPHA) * stored + self.EMA_ALPHA * embedding
                            updated = updated / (np.linalg.norm(updated) + 1e-8)
                            self._profiles[best_name]["embedding"] = updated
                            self._save_profiles()
                elif best_raw >= self.ADAPT_MIN_SCORE and best_name != "unknown":
                    # Same truth-boundary guard for the low-confidence adapt branch.
                    if origin != "synthetic":
                        stored = self._profiles[best_name]["embedding"]
                        if isinstance(stored, list):
                            stored = np.array(stored, dtype=np.float32)
                        gentle = self.EMA_ALPHA * self.ADAPT_ALPHA_SCALE
                        updated = (1 - gentle) * stored + gentle * embedding
                        updated = updated / (np.linalg.norm(updated) + 1e-8)
                        self._profiles[best_name]["embedding"] = updated
                        self._save_profiles()
                    best_name = f"speaker_{self._next_unknown_id}"
                    self._next_unknown_id += 1
                else:
                    best_name = f"speaker_{self._next_unknown_id}"
                    self._next_unknown_id += 1

                logger.info("Speaker ID: %s (raw=%.3f, smoothed=%.3f, known=%s)",
                            best_name, best_raw, best_score, is_known)

                try:
                    from hemisphere.distillation import distillation_collector
                    fidelity = min(1.0, best_raw + 0.2) if is_known else best_raw
                    if origin == "synthetic":
                        fidelity = min(fidelity, 0.7)
                    distillation_collector.record(
                        "ecapa_tdnn", "embedding", embedding.tolist(),
                        {"speaker": best_name, "confidence": round(best_raw, 3)},
                        origin=origin, fidelity=fidelity,
                    )
                except Exception:
                    pass

                return {
                    "name": best_name,
                    "confidence": best_score,
                    "is_known": is_known,
                    "embedding_id": best_name,
                    "raw_score": best_raw,
                    "closest_match": closest_match,
                }
            except Exception as exc:
                logger.error("Speaker identification failed: %s", exc)
                return _unknown

    @property
    def last_embedding(self) -> np.ndarray | None:
        """Most recent speaker embedding from the last identify() call."""
        return self._last_embedding

    def register_speaker(self, name: str, audio_b64: str, sample_rate: int = 16000) -> bool:
        """Register a named speaker from a single audio sample (base64 int16 PCM)."""
        if not self.available or self._model is None:
            return False
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            return self.enroll_speaker(name, [audio_f32], sample_rate)
        except Exception as exc:
            logger.error("Speaker registration failed: %s", exc)
            return False

    def enroll_speaker(
        self, name: str, clips: list[np.ndarray], sample_rate: int = 16000,
    ) -> bool:
        """Enroll a speaker from one or more float32 audio clips.

        Averages embeddings from all valid clips to produce a robust centroid.
        More clips (3-10) yield better discrimination.
        """
        if not self.available or self._model is None:
            return False

        embeddings: list[np.ndarray] = []
        with self._lock:
            for clip in clips:
                if len(clip) < sample_rate * 0.5:
                    continue
                try:
                    emb = self._extract_embedding(clip)
                    if emb is not None:
                        embeddings.append(emb)
                except Exception as exc:
                    logger.debug("Enrollment clip failed: %s", exc)

            if not embeddings:
                logger.warning("No valid clips for enrollment of '%s'", name)
                return False

            centroid = np.mean(embeddings, axis=0).astype(np.float32)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            for existing_name, profile in self._profiles.items():
                if existing_name.lower() == name.lower():
                    continue
                stored = profile["embedding"]
                if isinstance(stored, list):
                    stored = np.array(stored, dtype=np.float32)
                sim = self._cosine_sim(centroid, stored)
                if sim > 0.45:
                    logger.warning(
                        "Enrollment dedup: '%s' is %.1f%% similar to existing '%s' — "
                        "proceeding but flagging for potential merge",
                        name, sim * 100, existing_name,
                    )

            self._profiles[name] = {
                "embedding": centroid,
                "registered": time.time(),
                "last_seen": time.time(),
                "interaction_count": 0,
                "enrollment_clips": len(embeddings),
            }
            self._score_ema.pop(name, None)
            self._save_profiles()
            logger.info("Enrolled speaker '%s' from %d clip(s)", name, len(embeddings))
            return True

    def merge_into(self, source_name: str, target_name: str) -> bool:
        """Merge source profile into target, weighted by interaction count.

        Preserves the older registration date and sums interaction counts.
        Returns True if merge succeeded.
        """
        with self._lock:
            src = self._profiles.get(source_name)
            tgt = self._profiles.get(target_name)
            if not src or not tgt:
                logger.warning("Merge failed: source=%s(%s) target=%s(%s)",
                               source_name, bool(src), target_name, bool(tgt))
                return False

            src_emb = src["embedding"]
            tgt_emb = tgt["embedding"]
            if isinstance(src_emb, list):
                src_emb = np.array(src_emb, dtype=np.float32)
            if isinstance(tgt_emb, list):
                tgt_emb = np.array(tgt_emb, dtype=np.float32)

            src_count = max(src.get("interaction_count", 1), 1)
            tgt_count = max(tgt.get("interaction_count", 1), 1)
            total = src_count + tgt_count
            merged = (tgt_emb * tgt_count + src_emb * src_count) / total
            merged = merged / (np.linalg.norm(merged) + 1e-8)

            tgt["embedding"] = merged
            tgt["interaction_count"] = total
            tgt["registered"] = min(src.get("registered", time.time()),
                                    tgt.get("registered", time.time()))
            tgt["last_seen"] = max(src.get("last_seen", 0), tgt.get("last_seen", 0))
            tgt["enrollment_clips"] = (src.get("enrollment_clips", 0)
                                       + tgt.get("enrollment_clips", 0))

            del self._profiles[source_name]
            self._score_ema.pop(source_name, None)
            self._score_ema.pop(target_name, None)
            self._save_profiles()
            logger.info("Merged speaker '%s' into '%s' (weight %d/%d)",
                        source_name, target_name, src_count, total)
            return True

    def has_profile(self, name: str) -> bool:
        return name in self._profiles

    def remove_speaker(self, name: str) -> bool:
        """Remove a speaker profile. Returns True if the profile existed."""
        with self._lock:
            if name in self._profiles:
                del self._profiles[name]
                self._score_ema.pop(name, None)
                self._save_profiles()
                logger.info("Removed speaker profile: %s", name)
                return True
        return False

    def record_fused_interaction(self, name: str, voice_score: float = 0.0,
                                 embedding: "np.ndarray | None" = None) -> None:
        """Record an interaction for a speaker confirmed by identity fusion.

        Called when face recognition confirms the speaker but voice alone didn't
        match. Increments interaction_count and does a cautious EMA update when
        the voice score was reasonably close (>0.3), helping the profile converge.
        """
        with self._lock:
            if name not in self._profiles:
                return
            self._profiles[name]["last_seen"] = time.time()
            self._profiles[name]["interaction_count"] = (
                self._profiles[name].get("interaction_count", 0) + 1
            )
            if embedding is not None and voice_score > 0.3:
                stored = self._profiles[name]["embedding"]
                if isinstance(stored, list):
                    stored = np.array(stored, dtype=np.float32)
                gentle_alpha = self.EMA_ALPHA * 0.5
                updated = (1 - gentle_alpha) * stored + gentle_alpha * embedding
                updated = updated / (np.linalg.norm(updated) + 1e-8)
                self._profiles[name]["embedding"] = updated
                self._save_profiles()

    def get_known_speakers(self) -> list[str]:
        with self._lock:
            return list(self._profiles.keys())

    def last_score_for(self, name: str) -> float | None:
        """Return the smoothed EMA score for a known speaker, or None."""
        return self._score_ema.get(name)

    def get_profiles_summary(self) -> list[dict]:
        """Return a summary of all speaker profiles (no embeddings) for dashboard."""
        with self._lock:
            items = list(self._profiles.items())
        result = []
        for name, profile in items:
            result.append({
                "name": name,
                "registered": profile.get("registered", 0),
                "last_seen": profile.get("last_seen", 0),
                "interaction_count": profile.get("interaction_count", 0),
                "enrollment_clips": profile.get("enrollment_clips", 1),
            })
        return result
