"""Vision tool — fetches a snapshot from Pi and describes it via Ollama VLM."""

from __future__ import annotations

import base64
import hashlib
import logging
from typing import AsyncGenerator

import aiohttp

logger = logging.getLogger(__name__)

_SNAPSHOT_TIMEOUT = aiohttp.ClientTimeout(total=5)

# Generic look / Golden VISION STATUS / periodic room inventory.
GENERIC_SCENE_PROMPT = (
    "Describe what you see in this image concisely. "
    "Focus on people, objects, and activity."
)

# Targeted VQA (#24): the VLM answers THIS question from the live frame.
# Text LLM must not invent a count/color from a generic caption.
_VQA_PROMPT = (
    "Answer the user's question from this camera frame only. "
    "If the frame does not clearly show the answer, say you cannot tell. "
    "Do not guess. Do not invent rooms, objects, or counts that are not visible.\n\n"
    "Question: {question}"
)
_VQA_RETRY_PROMPT = (
    "Answer the original visual question from this NEW camera frame only. "
    "The operator said the previous answer was wrong. Use their correction as a "
    "hint only if the frame supports it. If the frame is unclear, say you cannot "
    "tell. Do not guess. Do not invent rooms, objects, or counts that are not visible.\n\n"
    "Original question: {question}\n"
    "Operator correction: {correction}"
)


def vqa_prompt(user_text: str, correction: str | None = None) -> str:
    """Wrap a spoken visual question as the describe_image prompt."""
    question = " ".join((user_text or "").split())
    if not question:
        return GENERIC_SCENE_PROMPT
    corr = " ".join((correction or "").split())
    if corr:
        return _VQA_RETRY_PROMPT.format(question=question, correction=corr)
    return _VQA_PROMPT.format(question=question)


def _snapshot_url(url: str, fresh: bool) -> str:
    if not fresh:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}grab=1"


async def fetch_snapshot(url: str, fresh: bool = False) -> bytes | None:
    """GET a JPEG snapshot from the Pi's HTTP server.

    fresh=True asks the Pi to grab a new camera frame (`?grab=1`) instead of
    encoding the last Hailo-processed buffer.
    """
    get_url = _snapshot_url(url, fresh)
    try:
        async with aiohttp.ClientSession(timeout=_SNAPSHOT_TIMEOUT) as session:
            async with session.get(get_url) as resp:
                if resp.status != 200:
                    logger.warning("Snapshot fetch failed: HTTP %d from %s", resp.status, get_url)
                    return None
                data = await resp.read()
                age = resp.headers.get("X-Frame-Age-Ms", "?")
                digest = hashlib.sha256(data).hexdigest()[:12]
                logger.info(
                    "Snapshot fetched: %d bytes sha=%s age_ms=%s fresh=%s from %s",
                    len(data), digest, age, fresh, get_url,
                )
                return data
    except Exception as exc:
        logger.warning("Snapshot fetch error (%s): %s", get_url, exc)
        return None


async def describe_jpeg(
    jpeg_bytes: bytes,
    ollama_client=None,
    claude_client=None,
    prompt: str | None = None,
) -> str:
    """Run the VLM on an already-fetched JPEG."""
    prompt = prompt or GENERIC_SCENE_PROMPT
    image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
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


async def describe_scene(
    pi_snapshot_url: str,
    ollama_client=None,
    claude_client=None,
    prompt: str | None = None,
    fresh: bool = False,
) -> str:
    """Fetch a camera frame from the Pi and describe it with a vision model.

    Priority: Ollama VLM (local) -> Claude API (cloud) -> unavailable message.
    Pass `prompt` for targeted VQA; default is a generic scene caption.
    fresh=True grabs a new Pi frame (VQA / Golden VISION STATUS).
    """
    jpeg_bytes = await fetch_snapshot(pi_snapshot_url, fresh=fresh)
    if jpeg_bytes is None:
        return "I can't see anything right now — the camera isn't reachable."
    return await describe_jpeg(jpeg_bytes, ollama_client, claude_client, prompt)


async def describe_scene_stream(
    pi_snapshot_url: str,
    ollama_client=None,
    prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a scene description token-by-token from Ollama VLM.

    Falls back to a single-shot yield if streaming isn't possible.
    """
    prompt = prompt or GENERIC_SCENE_PROMPT
    jpeg_bytes = await fetch_snapshot(pi_snapshot_url, fresh=True)
    if jpeg_bytes is None:
        yield "I can't see anything right now — the camera isn't reachable."
        return

    image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    if not ollama_client:
        yield "Vision models aren't available right now."
        return

    try:
        async for token in ollama_client.describe_image_stream(image_b64, prompt):
            yield token
    except Exception as exc:
        logger.warning("Vision stream failed (cold-load/timeout?): %s", exc)
        yield "I can't see clearly right now."
