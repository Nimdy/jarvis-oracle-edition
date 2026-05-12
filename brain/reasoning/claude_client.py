"""Claude API client — async text chat and multimodal vision."""

from __future__ import annotations

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Async wrapper around the Anthropic Python SDK.

    Provides ``chat()`` for text conversations and ``describe_image()``
    for base64-encoded image analysis.  Falls back gracefully when no
    API key is configured.
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-5-20250929", max_tokens: int = 1024) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any | None = None

        if api_key:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=api_key)
                logger.info("Claude client initialised (model=%s)", model)
            except ImportError:
                logger.warning("anthropic package not installed — Claude features disabled")
            except Exception:
                logger.exception("Failed to initialise Claude client")

    @property
    def available(self) -> bool:
        return self._client is not None

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        if not self._client:
            raise RuntimeError("Claude client not available (missing API key or package)")

        api_messages: list[dict[str, Any]] = []
        for m in messages:
            api_messages.append({"role": m["role"], "content": m["content"]})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def describe_image(
        self,
        image_bytes: bytes,
        prompt: str = "Describe what you see in this image in detail.",
        media_type: str = "image/jpeg",
    ) -> str:
        if not self._client:
            raise RuntimeError("Claude client not available (missing API key or package)")

        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                ],
            }],
        )
        return response.content[0].text
