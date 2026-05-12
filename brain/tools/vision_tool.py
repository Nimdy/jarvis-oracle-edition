"""Vision tool — fetches a snapshot from Pi and describes it via Ollama VLM."""

from __future__ import annotations

import base64
import logging
from typing import AsyncGenerator

import aiohttp

logger = logging.getLogger(__name__)

_SNAPSHOT_TIMEOUT = aiohttp.ClientTimeout(total=5)


async def _fetch_snapshot(url: str) -> bytes | None:
    """GET a JPEG snapshot from the Pi's HTTP server."""
    try:
        async with aiohttp.ClientSession(timeout=_SNAPSHOT_TIMEOUT) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("Snapshot fetch failed: HTTP %d from %s", resp.status, url)
                    return None
                return await resp.read()
    except Exception as exc:
        logger.warning("Snapshot fetch error (%s): %s", url, exc)
        return None


async def describe_scene(
    pi_snapshot_url: str,
    ollama_client=None,
    claude_client=None,
    prompt: str = "Describe what you see in this image concisely. Focus on people, objects, and activity.",
) -> str:
    """Fetch a camera frame from the Pi and describe it with a vision model.

    Priority: Ollama VLM (local) -> Claude API (cloud) -> unavailable message.
    """
    jpeg_bytes = await _fetch_snapshot(pi_snapshot_url)
    if jpeg_bytes is None:
        return "I can't see anything right now — the camera isn't reachable."

    image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    logger.info("Snapshot fetched: %d bytes from %s", len(jpeg_bytes), pi_snapshot_url)

    if ollama_client:
        try:
            return await ollama_client.describe_image(image_b64, prompt)
        except Exception as exc:
            logger.warning("Ollama vision failed, trying Claude fallback: %s", exc)

    if claude_client and claude_client.available:
        try:
            return await claude_client.describe_image(jpeg_bytes, prompt)
        except Exception as exc:
            logger.warning("Claude vision also failed: %s", exc)

    return "Vision models aren't available right now."


async def describe_scene_stream(
    pi_snapshot_url: str,
    ollama_client=None,
    prompt: str = "Describe what you see in this image concisely. Focus on people, objects, and activity.",
) -> AsyncGenerator[str, None]:
    """Stream a scene description token-by-token from Ollama VLM.

    Falls back to a single-shot yield if streaming isn't possible.
    """
    jpeg_bytes = await _fetch_snapshot(pi_snapshot_url)
    if jpeg_bytes is None:
        yield "I can't see anything right now — the camera isn't reachable."
        return

    image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    logger.info("Snapshot fetched for stream: %d bytes", len(jpeg_bytes))

    if not ollama_client:
        yield "Vision models aren't available right now."
        return

    async for token in ollama_client.describe_image_stream(image_b64, prompt):
        yield token
