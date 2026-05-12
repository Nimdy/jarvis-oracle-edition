"""Brain-side face identification using ArcFace/MobileFaceNet on CUDA.

Receives face crops (112x112 JPEG) from the Pi via WebSocket, extracts
embeddings, and matches against stored face profiles. Uses the same
enrollment/EMA/forget pattern as speaker_id.py.

All models run locally — no cloud calls.
"""

from __future__ import annotations

import base64
import collections
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass


class FaceIdentifier:
    """Identifies faces from 112x112 crops using a local ONNX embedding model."""

    SIMILARITY_THRESHOLD = 0.55
    EMA_ALPHA = 0.1
    EMA_MIN_CONFIDENCE = 0.70
    PROFILES_FILENAME = "face_profiles.json"
    SCORE_EMA_ALPHA = 0.35

    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "",
        persist_dir: str = "",
    ) -> None:
        self._lock = threading.Lock()
        self._session: Any = None
        self.available = False
        self._profiles: dict[str, dict] = {}
        self._next_unknown_id = 1
        self._input_name: str = ""
        self._input_shape: tuple = ()
        self._recent_crops_b64: collections.deque[str] = collections.deque(maxlen=5)
        self._score_ema: dict[str, float] = {}

        if not persist_dir:
            persist_dir = str(Path.home() / ".jarvis")
        self._persist_dir = Path(persist_dir)
        self._profiles_path = self._persist_dir / self.PROFILES_FILENAME

        if not ONNX_AVAILABLE:
            logger.info("onnxruntime not installed — face ID disabled")
            return

        if not model_path:
            model_path = str(self._persist_dir / "models" / "mobilefacenet.onnx")

        self._try_load_model(model_path, device)

    def _try_load_model(self, model_path: str, device: str) -> None:
        try:
            providers = []
            if device == "cuda":
                providers.append(("CUDAExecutionProvider", {}))
            providers.append(("CPUExecutionProvider", {}))

            if not Path(model_path).exists():
                logger.warning("Face embedding model not found at %s — face ID disabled. "
                               "Run setup.sh to download it.", model_path)
                return

            self._session = ort.InferenceSession(model_path, providers=providers)
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_shape = tuple(inp.shape)
            self.available = True
            self._load_profiles()

            actual_provider = self._session.get_providers()[0] if self._session.get_providers() else "unknown"
            logger.info("FaceID ready: %s on %s (%d profiles loaded)",
                        Path(model_path).name, actual_provider, len(self._profiles))
        except Exception as exc:
            logger.error("Failed to load face embedding model: %s", exc)

    def _load_profiles(self) -> None:
        if self._profiles_path.exists():
            try:
                data = json.loads(self._profiles_path.read_text())
                from identity.name_validator import is_valid_person_name
                cleaned = {}
                for name, profile in data.items():
                    if not is_valid_person_name(name):
                        logger.warning("Removing invalid face profile on load: %r", name)
                        continue
                    profile["embedding"] = np.array(profile["embedding"], dtype=np.float32)
                    cleaned[name] = profile
                self._profiles = cleaned
                if len(cleaned) < len(data):
                    self._save_profiles()
            except Exception as exc:
                logger.warning("Failed to load face profiles: %s", exc)

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
            logger.warning("Failed to save face profiles: %s", exc)

    def _decode_crop(self, crop_b64: str) -> np.ndarray | None:
        """Decode a base64 JPEG face crop to a 112x112 numpy array."""
        if not CV2_AVAILABLE:
            return None
        try:
            raw = base64.b64decode(crop_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112))
            return img
        except Exception as exc:
            logger.debug("Face crop decode failed: %s", exc)
            return None

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Extract face embedding from a 112x112 BGR crop."""
        if self._session is None:
            return None

        img = crop.astype(np.float32)
        img = (img - 127.5) / 128.0
        if len(self._input_shape) == 4 and self._input_shape[1] == 3:
            img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        outputs = self._session.run(None, {self._input_name: img})
        emb = outputs[0].flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def identify_b64(self, crop_b64: str) -> dict:
        """Identify a face from a base64 JPEG crop.

        Returns dict with keys: name, confidence, is_known, closest_match.
        """
        if not self.available:
            return {"name": "unknown", "confidence": 0.0, "is_known": False, "closest_match": ""}

        crop = self._decode_crop(crop_b64)
        if crop is None:
            return {"name": "unknown", "confidence": 0.0, "is_known": False, "closest_match": ""}

        self._recent_crops_b64.append(crop_b64)
        return self.identify(crop)

    def get_recent_crops(self, max_crops: int = 3) -> list[str]:
        """Return the most recent face crop base64 strings for enrollment."""
        return list(self._recent_crops_b64)[-max_crops:]

    def identify(self, crop: np.ndarray) -> dict:
        """Identify a face from a 112x112 BGR crop.

        NOTE (truth boundary): there is no synthetic vision lane today — the
        synthetic perception exercise is audio-only, so face profiles cannot
        be drifted by synthetic input at present. If a synthetic vision lane
        is ever added, this method must grow an ``origin`` parameter mirroring
        ``speaker_id.identify()`` and gate the EMA adapt + ``_save_profiles``
        paths with ``if origin != "synthetic":`` before any synthetic frame
        reaches this code. See docs/AWAKENING_PROTOCOL.md + AGENTS.md
        "Sequencing" subsection.
        """
        if not self.available or self._session is None:
            return {"name": "unknown", "confidence": 0.0, "is_known": False, "closest_match": ""}

        with self._lock:
            try:
                embedding = self._extract_embedding(crop)
                if embedding is None:
                    return {"name": "unknown", "confidence": 0.0, "is_known": False, "closest_match": ""}

                best_name = "unknown"
                best_score = 0.0
                best_raw = 0.0

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
                else:
                    best_name = f"face_{self._next_unknown_id}"
                    self._next_unknown_id += 1

                logger.info("Face ID: %s (raw=%.3f, smoothed=%.3f, known=%s)",
                            best_name, best_raw, best_score, is_known)

                try:
                    from hemisphere.distillation import distillation_collector
                    fidelity = min(1.0, best_raw + 0.1) if is_known else best_raw
                    distillation_collector.record(
                        "mobilefacenet", "embedding", embedding.tolist(),
                        {"name": best_name, "confidence": round(best_raw, 3)},
                        origin="camera", fidelity=fidelity,
                    )
                except Exception:
                    pass

                return {
                    "name": best_name,
                    "confidence": best_score,
                    "is_known": is_known,
                    "closest_match": closest_match,
                }
            except Exception as exc:
                logger.error("Face identification failed: %s", exc)
                return {"name": "unknown", "confidence": 0.0, "is_known": False, "closest_match": ""}

    def enroll_face(self, name: str, crops_b64: list[str]) -> bool:
        """Enroll a face from one or more base64 JPEG crops."""
        if not self.available or self._session is None:
            return False

        embeddings: list[np.ndarray] = []
        with self._lock:
            for crop_b64 in crops_b64:
                crop = self._decode_crop(crop_b64)
                if crop is None:
                    continue
                try:
                    emb = self._extract_embedding(crop)
                    if emb is not None:
                        embeddings.append(emb)
                except Exception as exc:
                    logger.debug("Face enrollment crop failed: %s", exc)

            if not embeddings:
                logger.warning("No valid face crops for enrollment of '%s'", name)
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
                        "Face enrollment dedup: '%s' is %.1f%% similar to existing '%s' — "
                        "proceeding but flagging for potential merge",
                        name, sim * 100, existing_name,
                    )

            self._profiles[name] = {
                "embedding": centroid,
                "registered": time.time(),
                "last_seen": time.time(),
                "interaction_count": 0,
                "enrollment_crops": len(embeddings),
            }
            self._score_ema.pop(name, None)
            self._save_profiles()
            logger.info("Enrolled face '%s' from %d crop(s)", name, len(embeddings))
            return True

    def merge_into(self, source_name: str, target_name: str) -> bool:
        """Merge source face profile into target, weighted by interaction count.

        Preserves the older registration date and sums interaction counts.
        Returns True if merge succeeded.
        """
        with self._lock:
            src = self._profiles.get(source_name)
            tgt = self._profiles.get(target_name)
            if not src or not tgt:
                logger.warning("Face merge failed: source=%s(%s) target=%s(%s)",
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
            self._save_profiles()
            logger.info("Merged face '%s' into '%s' (weight %d/%d)",
                        source_name, target_name, src_count, total)
            return True

    def has_profile(self, name: str) -> bool:
        return name in self._profiles

    def remove_face(self, name: str) -> bool:
        """Remove a face profile (forget face)."""
        with self._lock:
            if name in self._profiles:
                del self._profiles[name]
                self._score_ema.pop(name, None)
                self._save_profiles()
                logger.info("Removed face profile: %s", name)
                return True
        return False

    def get_known_faces(self) -> list[str]:
        with self._lock:
            return list(self._profiles.keys())

    def get_profiles_summary(self) -> list[dict]:
        """Return summary of all face profiles (no embeddings) for dashboard."""
        with self._lock:
            items = list(self._profiles.items())
        result = []
        for name, profile in items:
            result.append({
                "name": name,
                "registered": profile.get("registered", 0),
                "last_seen": profile.get("last_seen", 0),
                "interaction_count": profile.get("interaction_count", 0),
                "enrollment_crops": profile.get("enrollment_crops", 1),
            })
        return result
