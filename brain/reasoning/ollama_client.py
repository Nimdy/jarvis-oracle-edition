"""Ollama LLM client — async chat, streaming, and vision."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from ollama import AsyncClient, ChatResponse, ResponseError

logger = logging.getLogger(__name__)


def _default_model() -> str:
    try:
        from hardware_profile import get_hardware_profile
        return get_hardware_profile().models.llm_model
    except Exception:
        return "qwen3:8b"


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str | None = None,
        vision_model: str = "qwen2.5vl:7b",
        temperature: float = 0.7,
        max_tokens: int = 512,
        keep_alive: str = "5m",
    ) -> None:
        if model is None:
            model = _default_model()
        self._host = host
        self._model = model
        self._vision_model = vision_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._keep_alive = keep_alive
        self._client = AsyncClient(host=host)

    async def chat(self, messages: list[dict[str, Any]], system_prompt: str | None = None,
                   model_override: str | None = None) -> str:
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        use_model = model_override or self._model
        try:
            response: ChatResponse = await self._client.chat(
                model=use_model,
                messages=all_messages,
                options={"temperature": self._temperature, "num_predict": self._max_tokens},
                keep_alive=self._keep_alive,
                think=False,
            )
            return response.message.content
        except ResponseError as exc:
            logger.error("Ollama chat failed: %s (status %s)", exc.error, exc.status_code)
            raise
        except ConnectionError:
            logger.error("Ollama connection failed — is it running? %s", self._host)
            raise

    async def chat_stream(
        self, messages: list[dict[str, Any]], system_prompt: str | None = None,
        model_override: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        use_model = model_override or self._model
        num_predict = max_tokens or self._max_tokens
        use_temp = temperature if temperature is not None else self._temperature
        try:
            stream = await self._client.chat(
                model=use_model,
                messages=all_messages,
                stream=True,
                options={"temperature": use_temp, "num_predict": num_predict},
                keep_alive=self._keep_alive,
                think=False,
            )
            async for chunk in stream:
                yield chunk.message.content
        except ResponseError as exc:
            logger.error("Ollama stream failed (model=%s): %s (status %s)",
                         use_model, exc.error, exc.status_code)
            raise
        except (ConnectionError, OSError) as exc:
            logger.error("Ollama connection failed (model=%s, host=%s): %s",
                         use_model, self._host, exc)
            raise

    # -- Vision (image description) ------------------------------------------

    async def describe_image(
        self, image_b64: str, prompt: str = "Describe what you see in this image.",
        model: str | None = None,
    ) -> str:
        """Send a base64 JPEG to a vision model and return the description."""
        use_model = model or self._vision_model
        try:
            response: ChatResponse = await self._client.chat(
                model=use_model,
                messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
                options={"temperature": 0.4, "num_predict": self._max_tokens},
                keep_alive="5m",
            )
            return response.message.content
        except ResponseError as exc:
            logger.error("Ollama vision failed: %s (status %s)", exc.error, exc.status_code)
            raise

    async def describe_image_stream(
        self, image_b64: str, prompt: str = "Describe what you see in this image.",
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a vision description token-by-token."""
        use_model = model or self._vision_model
        try:
            stream = await self._client.chat(
                model=use_model,
                messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
                stream=True,
                options={"temperature": 0.4, "num_predict": self._max_tokens},
                keep_alive="5m",
            )
            async for chunk in stream:
                yield chunk.message.content
        except ResponseError as exc:
            logger.error("Ollama vision stream failed: %s", exc.error)
            raise

    # -- CPU-resident coding LLM -------------------------------------------------

    async def code_chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        coding_host: str = "http://localhost:11435",
        coding_model: str = "qwen2.5-coder:7b",
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> str:
        """Chat with the CPU-resident coding LLM on a separate Ollama instance.

        This connects to a dedicated CPU-only Ollama server (launched with
        CUDA_VISIBLE_DEVICES="" on a separate port) to avoid GPU VRAM contention.
        """
        cpu_client = AsyncClient(host=coding_host)

        all_messages: list[dict[str, Any]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        try:
            response: ChatResponse = await cpu_client.chat(
                model=coding_model,
                messages=all_messages,
                options={"temperature": temperature, "num_predict": max_tokens},
                keep_alive="10m",
                think=False,
            )
            return response.message.content
        except ResponseError as exc:
            logger.error("Coding LLM failed (model=%s, host=%s): %s",
                         coding_model, coding_host, exc.error)
            raise
        except ConnectionError:
            logger.error("Coding LLM connection failed — is CPU Ollama running? %s", coding_host)
            raise

    @property
    def coding_model(self) -> str:
        return "qwen2.5-coder:7b"

    async def is_available(self) -> bool:
        try:
            await self._client.list()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            result = await self._client.list()
            return [m.model for m in result.models]
        except Exception:
            return []

    async def warmup(self, model_override: str | None = None) -> None:
        """Load model into VRAM with a tiny prompt so first real call is fast."""
        use_model = model_override or self._model
        try:
            t0 = __import__("time").monotonic()
            await self._client.chat(
                model=use_model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
                keep_alive=self._keep_alive,
            )
            elapsed = __import__("time").monotonic() - t0
            logger.info("Model '%s' warmed up in %.1fs (keep_alive=%s)",
                        use_model, elapsed, self._keep_alive)
        except Exception as exc:
            logger.warning("Model warmup failed for '%s': %s", use_model, exc)

    async def warmup_all(self, models: list[str]) -> None:
        """Pre-load multiple models into VRAM. For always-online tiers."""
        for model in models:
            if model:
                await self.warmup(model)

    def set_model(self, model: str) -> None:
        self._model = model

    def set_temperature(self, temp: float) -> None:
        self._temperature = max(0.0, min(2.0, temp))

    @property
    def model(self) -> str:
        return self._model

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def host(self) -> str:
        return self._host

    async def unload_model(self, model: str) -> None:
        """Immediately unload a model from VRAM (keep_alive=0)."""
        try:
            await self._client.generate(model=model, prompt="", keep_alive=0)
            logger.info("Unloaded model '%s' from VRAM", model)
        except Exception as exc:
            logger.debug("Could not unload '%s': %s", model, exc)

    async def unload_all(self) -> None:
        """Unload all currently loaded models to free VRAM (e.g. before STT)."""
        try:
            result = await self._client.ps()
            loaded = [m.model for m in result.models]
        except Exception:
            loaded = []
        for m in loaded:
            await self.unload_model(m)
        if loaded:
            logger.info("Freed VRAM: unloaded %d model(s)", len(loaded))

    async def unload_non_essential(self) -> int:
        """Unload secondary models (vision, coding) but keep the text LLM warm.

        Returns the number of models unloaded.
        """
        try:
            result = await self._client.ps()
            loaded = [m.model for m in result.models]
        except Exception:
            loaded = []
        keep = {self._model}
        unloaded = 0
        for m in loaded:
            if m not in keep:
                await self.unload_model(m)
                unloaded += 1
        if unloaded:
            logger.info("Freed non-essential VRAM: unloaded %d model(s), kept %s",
                        unloaded, self._model)
        return unloaded

    async def get_loaded_vram_mb(self) -> int:
        """Return approximate total VRAM used by currently loaded Ollama models."""
        try:
            result = await self._client.ps()
            total = 0
            for m in result.models:
                total += getattr(m, "size", 0) // (1024 * 1024)
            return total
        except Exception:
            return 0
