"""Provider -- AI patch generation with source code context.

Default path is fully local:
1. Local CoderServer (Qwen3-Coder-Next via llama-server)
2. Local Ollama fallback when the coder model is unavailable

Claude/OpenAI in this module are gated by ``SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS``
(default off). API keys alone do not activate cloud codegen or cloud patch
review here — avoids accidental calls when keys exist for other features.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from self_improve.code_patch import CodePatch, FileDiff
from self_improve.conversation import (
    CODER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    ImprovementConversation,
)
from self_improve.improvement_request import ImprovementRequest
from self_improve.patch_plan import PatchPlan

logger = logging.getLogger(__name__)

MAX_TOKENS = 16384


def _self_improve_cloud_plugins_enabled() -> bool:
    v = os.environ.get("SELF_IMPROVE_ALLOW_CLOUD_PROVIDERS", "").strip().lower()
    return v in ("1", "true", "yes", "on")
MAX_PARSE_RETRIES = 2

JSON_REPAIR_PROMPT = (
    "Your previous response was not valid JSON. You MUST respond with ONLY a JSON object "
    "matching this exact schema (no markdown fences, no explanation before or after):\n"
    '{"files": [{"path": "brain/...", "edits": [{"search": "exact existing code", "replace": "new code"}]}], '
    '"description": "...", "confidence": 0.0-1.0}\n'
    "Respond with ONLY the JSON object, nothing else."
)


class PatchProvider:
    """Generates code patches via local CoderServer and Ollama; cloud APIs opt-in only."""

    def __init__(self) -> None:
        allow_cloud = _self_improve_cloud_plugins_enabled()
        self._claude_available = allow_cloud and bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._openai_available = allow_cloud and bool(os.environ.get("OPENAI_API_KEY"))
        self._coder_server: Any = None

    def set_coder_server(self, server: Any) -> None:
        """Wire the CoderServer instance (called from main.py or orchestrator)."""
        self._coder_server = server

    # ------------------------------------------------------------------
    # Primary: local CoderServer generation (Qwen3-Coder-Next)
    # ------------------------------------------------------------------

    async def generate_with_coder(
        self,
        messages: list[dict[str, str]],
        plan_id: str,
    ) -> CodePatch | None:
        """Generate a patch using the local CoderServer (on-demand llama-server).

        Starts the server, generates, then shuts it down to reclaim RAM.
        Includes JSON retry logic.
        """
        if not self._coder_server or not self._coder_server.is_available():
            return None

        try:
            if hasattr(self._coder_server, "set_consumer"):
                self._coder_server.set_consumer("self_improve")
            response = await self._coder_server.generate(messages, CODER_SYSTEM_PROMPT)
            if response is None:
                return None

            patch = self._parse_response(response, "coder_local", plan_id)
            if patch is not None:
                return patch

            retry_msgs = list(messages)
            for attempt in range(MAX_PARSE_RETRIES):
                logger.info("Coder JSON parse retry %d/%d for plan %s",
                            attempt + 1, MAX_PARSE_RETRIES, plan_id)
                retry_msgs.append({"role": "assistant", "content": response})
                retry_msgs.append({"role": "user", "content": JSON_REPAIR_PROMPT})
                response = await self._coder_server.generate(retry_msgs, CODER_SYSTEM_PROMPT)
                if response is None:
                    break
                patch = self._parse_response(response, "coder_local", plan_id)
                if patch is not None:
                    logger.info("Coder JSON parse succeeded on retry %d", attempt + 1)
                    return patch

            logger.error("All %d coder JSON parse retries exhausted for plan %s",
                         MAX_PARSE_RETRIES, plan_id)
            return None
        except Exception:
            logger.exception("CoderServer generation failed")
            return None
        finally:
            try:
                await self._coder_server.shutdown()
            except Exception:
                logger.debug("CoderServer shutdown error", exc_info=True)

    # ------------------------------------------------------------------
    # Fallback: local Ollama generation
    # ------------------------------------------------------------------

    async def generate_patch_local(
        self,
        conversation: ImprovementConversation,
        code_context: str,
        plan_text: str,
        plan: PatchPlan,
    ) -> CodePatch | None:
        """Generate a patch using the local Ollama model with full source context."""
        prompt = self._build_contextual_prompt(plan_text, code_context, plan)

        conversation.add_turn("system", CODER_SYSTEM_PROMPT)
        conversation.add_turn("think", prompt)

        messages = conversation.get_messages_for_coder()

        try:
            from reasoning.ollama_client import OllamaClient
            # Use the default OllamaClient -- the caller should provide an instance,
            # but for now we create one with defaults that will be overridden at runtime
            ollama = OllamaClient()
            response = await ollama.chat(
                messages=messages,
                system_prompt=CODER_SYSTEM_PROMPT,
            )

            conversation.add_turn("code", response)
            return self._parse_response(response, "local", plan.id)

        except Exception:
            logger.exception("Local Ollama code generation failed")
            return None

    async def generate_with_ollama(
        self,
        ollama_client: Any,
        messages: list[dict[str, str]],
        plan_id: str,
    ) -> CodePatch | None:
        """Generate a patch using a provided OllamaClient instance.

        Retries up to MAX_PARSE_RETRIES times when the LLM output is not
        valid JSON, appending a structured correction prompt each time.
        """
        try:
            response = await ollama_client.chat(
                messages=messages,
                system_prompt=CODER_SYSTEM_PROMPT,
            )
            patch = self._parse_response(response, "local", plan_id)
            if patch is not None:
                return patch

            retry_msgs = list(messages)
            for attempt in range(MAX_PARSE_RETRIES):
                logger.info("JSON parse retry %d/%d for plan %s", attempt + 1, MAX_PARSE_RETRIES, plan_id)
                retry_msgs.append({"role": "assistant", "content": response})
                retry_msgs.append({"role": "user", "content": JSON_REPAIR_PROMPT})
                response = await ollama_client.chat(
                    messages=retry_msgs,
                    system_prompt=CODER_SYSTEM_PROMPT,
                )
                patch = self._parse_response(response, "local", plan_id)
                if patch is not None:
                    logger.info("JSON parse succeeded on retry %d", attempt + 1)
                    return patch

            logger.error("All %d JSON parse retries exhausted for plan %s", MAX_PARSE_RETRIES, plan_id)
            return None
        except Exception:
            logger.exception("Ollama code generation failed")
            return None

    # ------------------------------------------------------------------
    # Optional external plugin providers
    # ------------------------------------------------------------------

    async def generate_with_claude(
        self,
        messages: list[dict[str, str]],
        plan_id: str,
    ) -> CodePatch | None:
        """Generate a patch using the optional Claude plugin provider."""
        if not self._claude_available:
            return None
        try:
            import anthropic
            client = anthropic.AsyncAnthropic()

            api_messages = [
                {"role": m["role"] if m["role"] in ("user", "assistant") else "user",
                 "content": m["content"]}
                for m in messages
                if m.get("content")
            ]

            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=MAX_TOKENS,
                system=CODER_SYSTEM_PROMPT,
                messages=api_messages,
            )
            text = response.content[0].text
            return self._parse_response(text, "claude", plan_id)
        except ImportError:
            logger.warning("anthropic package not installed")
            return None
        except Exception:
            logger.exception("Claude code generation failed")
            return None

    async def generate_with_openai(
        self,
        messages: list[dict[str, str]],
        plan_id: str,
    ) -> CodePatch | None:
        """Generate a patch using the optional OpenAI plugin provider."""
        if not self._openai_available:
            return None
        try:
            import openai
            client = openai.AsyncOpenAI()

            api_messages = [{"role": "system", "content": CODER_SYSTEM_PROMPT}]
            for m in messages:
                role = m.get("role", "user")
                if role not in ("user", "assistant", "system"):
                    role = "user"
                api_messages.append({"role": role, "content": m["content"]})

            response = await client.chat.completions.create(
                model="gpt-4o",
                max_tokens=MAX_TOKENS,
                messages=api_messages,
            )
            text = response.choices[0].message.content or ""
            return self._parse_response(text, "openai", plan_id)
        except ImportError:
            logger.warning("openai package not installed")
            return None
        except Exception:
            logger.exception("OpenAI code generation failed")
            return None

    async def retry_with_feedback_external(
        self,
        conversation: ImprovementConversation,
        diagnostics: list[dict[str, Any]],
        original_code: str,
        plan_id: str,
    ) -> CodePatch | None:
        """Retry generation via external API after sandbox failure."""
        feedback_parts = ["The previous patch FAILED validation. Here are the errors:\n"]
        for diag in diagnostics[:3]:
            feedback_parts.append(
                f"- [{diag.get('error_type', '?')}] {diag.get('file', '?')}"
                f":{diag.get('line', '?')} -- {diag.get('message', '')}"
            )
            if diag.get("context_span"):
                feedback_parts.append(f"  Context:\n{diag['context_span']}")

        feedback_parts.append("\nFix these errors and regenerate the search-and-replace edits.")
        feedback_parts.append('Return the same JSON format: {"files": [{"path": "...", "edits": [{"search": "...", "replace": "..."}]}], "description": "...", "confidence": ...}')

        feedback = "\n".join(feedback_parts)
        conversation.add_turn("validate", feedback)

        messages = conversation.get_messages_for_coder()

        if self._claude_available:
            patch = await self.generate_with_claude(messages, plan_id)
            if patch is not None:
                conversation.add_turn("code", json.dumps(patch.to_dict()))
                return patch

        if self._openai_available:
            patch = await self.generate_with_openai(messages, plan_id)
            if patch is not None:
                conversation.add_turn("code", json.dumps(patch.to_dict()))
                return patch

        return None

    # ------------------------------------------------------------------
    # Iterative feedback (local Ollama)
    # ------------------------------------------------------------------

    async def retry_with_feedback(
        self,
        ollama_client: Any,
        conversation: ImprovementConversation,
        diagnostics: list[dict[str, Any]],
        original_code: str,
        plan_id: str,
    ) -> CodePatch | None:
        """Retry generation after sandbox failure with structured diagnostics."""
        feedback_parts = ["The previous patch FAILED validation. Here are the errors:\n"]
        for diag in diagnostics[:3]:
            feedback_parts.append(
                f"- [{diag.get('error_type', '?')}] {diag.get('file', '?')}"
                f":{diag.get('line', '?')} -- {diag.get('message', '')}"
            )
            if diag.get("context_span"):
                feedback_parts.append(f"  Context:\n{diag['context_span']}")

        feedback_parts.append("\nFix these errors and regenerate the search-and-replace edits.")
        feedback_parts.append('Return the same JSON format: {"files": [{"path": "...", "edits": [{"search": "...", "replace": "..."}]}], "description": "...", "confidence": ...}')

        feedback = "\n".join(feedback_parts)
        conversation.add_turn("validate", feedback)

        messages = conversation.get_messages_for_coder()

        try:
            response = await ollama_client.chat(
                messages=messages,
                system_prompt=CODER_SYSTEM_PROMPT,
            )
            conversation.add_turn("code", response)
            return self._parse_response(response, "local_retry", plan_id)
        except Exception:
            logger.exception("Ollama retry generation failed")
            return None

    # ------------------------------------------------------------------
    # External review (optional)
    # ------------------------------------------------------------------

    async def review_with_claude(
        self,
        patch: CodePatch,
        plan_text: str,
        code_context: str,
    ) -> dict[str, Any]:
        """Optional external review via Claude API."""
        if not self._claude_available:
            return {"approved": True, "reasoning": "Claude not available, skipping review", "concerns": []}

        prompt = (
            f"Plan:\n{plan_text}\n\n"
            f"Code context:\n{code_context[:3000]}\n\n"
            f"Patch:\n{json.dumps(patch.to_dict(), indent=2)}\n\n"
            "Review this patch for correctness and safety."
        )

        try:
            import anthropic
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=REVIEWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_review(text)
        except ImportError:
            return {"approved": True, "reasoning": "anthropic not installed", "concerns": []}
        except Exception:
            logger.exception("Claude review failed")
            return {"approved": True, "reasoning": "Claude review errored", "concerns": []}

    # ------------------------------------------------------------------
    # Legacy external generation (kept for backward compatibility)
    # ------------------------------------------------------------------

    async def generate_patch(
        self,
        request: ImprovementRequest,
        plan: PatchPlan,
        provider: str = "claude",
    ) -> CodePatch | None:
        """Generate patch via external API (legacy path)."""
        prompt = self._build_legacy_prompt(request, plan)

        if provider == "claude" and self._claude_available:
            return await self._call_claude(prompt, plan.id)
        elif provider == "openai" and self._openai_available:
            return await self._call_openai(prompt, plan.id)
        else:
            logger.warning("Provider %s not available", provider)
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_contextual_prompt(
        self, plan_text: str, code_context: str, plan: PatchPlan,
    ) -> str:
        resolved = getattr(plan, "_resolved_target_files", [])
        if resolved:
            files_section = (
                "Resolved files in scope:\n"
                + "\n".join(f"  - {f}" for f in resolved) + "\n"
                "ONLY use paths from this list.\n"
            )
        else:
            files_list = ", ".join(plan.files_to_modify + plan.files_to_create)
            files_section = f"{files_list}\n"
        return (
            f"## Improvement Plan\n{plan_text}\n\n"
            f"## Target Files\n{files_section}\n"
            f"## Current Source Code\n{code_context}\n\n"
            f"## Constraints\n{chr(10).join(plan.constraints)}\n\n"
            "## Critical Rules\n"
            "- FILENAME RULE: Every file path MUST match exactly a file from the resolved list or the source code headers. Do NOT abbreviate or invent filenames.\n"
            "- Preserve ALL existing public names: module-level variables, classes, functions, and constants.\n"
            "  Other modules import these names; removing or renaming them will break the system.\n"
            "- Only ADD new code or modify existing function bodies. Do not reorganize imports,\n"
            "  rename variables, or restructure the file layout unless the plan explicitly requires it.\n"
            "- Keep changes minimal and focused on the plan objective.\n"
            "- Search strings MUST come from the source code blocks only.\n"
            "- NEVER use instructional text, markdown headings, or prompt boilerplate as a search string.\n"
            '- Before returning JSON, verify every "search" string appears verbatim in the provided source.\n\n'
            "## Output Format\n"
            "Return SURGICAL search-and-replace edits, NOT full file content.\n"
            "Each edit's \"search\" must be an exact substring from the source code above.\n"
            "Return valid JSON:\n"
            '{\"files\": [{\"path\": \"brain/...\", \"edits\": [{\"search\": \"exact code\", \"replace\": \"new code\"}]}], '
            '\"description\": \"...\", \"confidence\": 0.0-1.0}'
        )

    def _build_legacy_prompt(self, request: ImprovementRequest, plan: PatchPlan) -> str:
        return (
            f"Improvement Request:\nType: {request.type}\n"
            f"Target: {request.target_module}\n"
            f"Description: {request.description}\n"
            f"Evidence: {', '.join(request.evidence[:5])}\n"
            f"Constraints: {json.dumps(request.constraints)}\n\n"
            f"Files to modify: {', '.join(plan.files_to_modify)}\n"
            f"Generate surgical search-and-replace edits as described in the system prompt."
        )

    async def _call_claude(self, prompt: str, plan_id: str) -> CodePatch | None:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=MAX_TOKENS,
                system=CODER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_response(text, "claude", plan_id)
        except ImportError:
            logger.warning("anthropic package not installed")
            return None
        except Exception:
            logger.exception("Claude API call failed")
            return None

    async def _call_openai(self, prompt: str, plan_id: str) -> CodePatch | None:
        try:
            import openai
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model="gpt-4o",
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            return self._parse_response(text, "openai", plan_id)
        except ImportError:
            logger.warning("openai package not installed")
            return None
        except Exception:
            logger.exception("OpenAI API call failed")
            return None

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Find the outermost JSON object in LLM output."""
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start: i + 1]
        return None

    _REQUIRED_KEYS = {"files"}

    @staticmethod
    def _apply_edits(original: str, edits: list[dict[str, str]]) -> str | None:
        """Apply search-and-replace edits to original source, returning new content.

        Returns None if any search string is not found in the source.
        """
        result = original
        for edit in edits:
            search = edit.get("search", "")
            replace = edit.get("replace", "")
            if not search:
                continue
            if search not in result:
                logger.warning("Edit search string not found in source (first 80 chars): %s",
                               search[:80])
                return None
            result = result.replace(search, replace, 1)
        return result

    @staticmethod
    def _read_original_file(path: str) -> str | None:
        """Read the original file content from disk for edit application."""
        from pathlib import Path
        brain_dir = Path(__file__).resolve().parent.parent
        rel = path.replace("brain/", "") if path.startswith("brain/") else path
        full = brain_dir / rel
        if full.exists():
            try:
                return full.read_text(encoding="utf-8")
            except Exception:
                pass
        return None

    def _parse_response(self, text: str, provider: str, plan_id: str) -> CodePatch | None:
        try:
            raw = self._extract_json(text)
            if raw is None:
                logger.error("No JSON found in provider response (%d chars)", len(text))
                return None

            data = json.loads(raw)

            missing = self._REQUIRED_KEYS - set(data.keys())
            if missing:
                logger.error("JSON missing required keys %s", missing)
                return None

            files_raw = data.get("files", [])
            if not isinstance(files_raw, list) or not files_raw:
                logger.error("JSON 'files' is empty or not a list")
                return None

            files = []
            for fd in files_raw:
                path = fd.get("path", "")
                edits = fd.get("edits", [])
                content = fd.get("content", "")
                diff = fd.get("diff", "")

                if not path:
                    logger.warning("Skipping file entry with no path")
                    continue

                original = self._read_original_file(path)

                if edits and isinstance(edits, list):
                    if original is None:
                        logger.warning(
                            "HALLUCINATED FILENAME — file does not exist: %s "
                            "(the model invented a path not in the codebase)", path,
                        )
                        continue
                    new_content = self._apply_edits(original, edits)
                    if new_content is None:
                        logger.warning("Edit application failed for %s (search string mismatch)", path)
                        continue
                    files.append(FileDiff(
                        path=path,
                        original_content=original,
                        new_content=new_content,
                    ))
                elif content or diff:
                    files.append(FileDiff(
                        path=path,
                        original_content=original or "",
                        new_content=content,
                        diff=diff,
                    ))
                else:
                    logger.warning("Skipping file entry with no edits or content: %s", path)
                    continue

            if not files:
                logger.error("No valid file entries after schema validation")
                return None

            return CodePatch(
                plan_id=plan_id,
                provider=provider,
                files=files,
                description=data.get("description", ""),
                test_instructions=data.get("test_plan", ""),
                confidence=data.get("confidence", 0.5),
            )
        except json.JSONDecodeError as exc:
            logger.error("JSON decode failed at char %d: %s", exc.pos or -1, exc.msg)
            return None
        except Exception:
            logger.exception("Failed to parse provider response")
            return None

    def _parse_review(self, text: str) -> dict[str, Any]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return {"approved": True, "reasoning": "Could not parse review", "concerns": []}

    def get_status(self) -> dict[str, Any]:
        coder_status: dict[str, Any] = {"available": False}
        if self._coder_server:
            try:
                coder_status = self._coder_server.get_status()
            except Exception:
                pass
        return {
            "claude_available": self._claude_available,
            "openai_available": self._openai_available,
            "cloud_plugins_enabled": _self_improve_cloud_plugins_enabled(),
            "local_available": True,
            "coder": coder_status,
        }
