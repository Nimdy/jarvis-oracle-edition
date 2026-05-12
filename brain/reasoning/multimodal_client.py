"""Phi-4-Multimodal client — text + vision + audio in one model.

Secondary reasoning engine for complex queries that benefit from
multimodal context (camera frames, voice tone, combined text+image).
Runs on laptop GPU alongside Ollama.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False


class MultimodalClient:
    """Phi-4-Multimodal for combined text + vision + audio reasoning."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        device: str = "cuda",
        max_tokens: int = 512,
    ):
        self._model_name = model_name
        self._device = device
        self._max_tokens = max_tokens
        self._model = None
        self._processor = None
        self.available = False

        if not MULTIMODAL_AVAILABLE:
            logger.info("transformers/torch not available — multimodal disabled")
            return

        if device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
            logger.info("CUDA not available, multimodal will use CPU")

    async def load(self) -> bool:
        """Lazy-load the model (heavy, only load when first needed)."""
        if self._model is not None:
            return True
        if not MULTIMODAL_AVAILABLE:
            return False

        try:
            from config import get_models_dir
            _cache = str(get_models_dir() / "huggingface")
            logger.info("Loading multimodal model: %s (device=%s)...", self._model_name, self._device)
            self._processor = AutoProcessor.from_pretrained(
                self._model_name, trust_remote_code=True, cache_dir=_cache,
            )
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
                device_map=self._device if self._device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=_cache,
            )
            if self._device != "cuda":
                self._model = self._model.to(self._device)

            self.available = True
            logger.info("Multimodal model loaded successfully")
            return True
        except Exception as exc:
            logger.error("Failed to load multimodal model: %s", exc)
            return False

    async def reason(
        self,
        text: str,
        image_bytes: bytes | None = None,
        audio_array: Any = None,
        system_prompt: str = "",
    ) -> str:
        """Run multimodal reasoning with optional image and audio context."""
        if not await self.load():
            return ""

        start = time.time()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            content_parts = []

            if image_bytes:
                img_b64 = base64.b64encode(image_bytes).decode()
                content_parts.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{img_b64}",
                })

            content_parts.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": content_parts})

            inputs = self._processor(
                messages,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens,
                    do_sample=True,
                    temperature=0.7,
                )

            response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self._processor.decode(response_ids, skip_special_tokens=True)

            latency = int((time.time() - start) * 1000)
            logger.info("Multimodal response: %dms, %d tokens", latency, len(response_ids))
            return response.strip()

        except Exception as exc:
            logger.error("Multimodal reasoning failed: %s", exc)
            return ""

    async def describe_image(self, image_bytes: bytes, question: str = "What do you see?") -> str:
        """Describe an image using multimodal understanding."""
        return await self.reason(question, image_bytes=image_bytes)

    def unload(self) -> None:
        """Free GPU memory by unloading the model."""
        self._model = None
        self._processor = None
        self.available = False
        if MULTIMODAL_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Multimodal model unloaded")

    def get_status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "loaded": self._model is not None,
            "model": self._model_name,
            "device": self._device,
        }
