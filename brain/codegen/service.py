"""CodeGenService — unified API for code generation and validation.

Shared service consumed by both self-improvement and the capability acquisition
pipeline.  Wraps CoderServer (llama-server lifecycle) and Sandbox (AST + lint +
pytest + kernel simulation) behind a single ``generate_and_validate()`` call.

All generated code passes through the same PatchPlan safety validation
(denied patterns, AST checks, capability escalation detection, diff budget)
regardless of the caller.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BRAIN_DIR = Path(__file__).resolve().parent.parent


class CodeGenService:
    """Unified code generation + sandbox validation service.

    Usage::

        service = CodeGenService()
        service.set_coder_server(coder)  # inject CoderServer from main.py

        result = await service.generate_and_validate(
            messages=[{"role": "user", "content": prompt}],
            write_category="skill_plugin",
            evidence_bundle=["art_abc123"],
            risk_tier=1,
        )
    """

    def __init__(self) -> None:
        self._coder_server: Any = None
        self._total_generations: int = 0
        self._total_validations: int = 0
        self._total_failures: int = 0
        self._active_consumer: str = ""
        self._last_consumer: str = ""

    def set_coder_server(self, server: Any) -> None:
        """Wire the CoderServer instance (called during initialization)."""
        self._coder_server = server

    def set_consumer(self, consumer: str) -> None:
        """Record the current infrastructure consumer for dashboard truth."""
        self._active_consumer = consumer
        self._last_consumer = consumer
        if self._coder_server and hasattr(self._coder_server, "set_consumer"):
            self._coder_server.set_consumer(consumer)

    @property
    def coder_available(self) -> bool:
        return self._coder_server is not None and self._coder_server.is_available()

    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
    ) -> str | None:
        """Generate code via CoderServer.  Returns raw text or None."""
        if not self._coder_server:
            logger.warning("CodeGenService: no coder server configured")
            self._active_consumer = ""
            return None

        if not self._coder_server.is_available():
            logger.warning("CodeGenService: coder server not available")
            self._active_consumer = ""
            return None

        try:
            result = await self._coder_server.generate(messages, system_prompt)
            if result is not None:
                self._total_generations += 1
            return result
        finally:
            try:
                await self._coder_server.shutdown()
            except Exception:
                logger.debug("CodeGenService coder shutdown failed", exc_info=True)
            self._active_consumer = ""

    async def generate_and_validate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        write_category: str = "self_improve",
        evidence_bundle: list[str] | None = None,
        risk_tier: int = 0,
    ) -> dict[str, Any]:
        """Generate code and run sandbox validation.

        Returns a result dict with keys:
          - success: bool
          - raw_text: str | None (raw LLM output)
          - patch: CodePatch | None (parsed patch if valid JSON)
          - sandbox_report: EvaluationReport | None
          - validation_errors: list[str]
          - evidence_check: dict (sufficiency status)
        """
        result: dict[str, Any] = {
            "success": False,
            "raw_text": None,
            "patch": None,
            "sandbox_report": None,
            "validation_errors": [],
            "evidence_check": {},
        }

        # Evidence sufficiency check
        ev_check = self._check_evidence_sufficiency(risk_tier, evidence_bundle or [])
        result["evidence_check"] = ev_check
        if not ev_check.get("sufficient", True):
            result["validation_errors"].append(f"Evidence insufficient: {ev_check.get('reason', '')}")
            self._active_consumer = ""
            return result

        # Generate
        raw_text = await self.generate(messages, system_prompt)
        result["raw_text"] = raw_text
        if raw_text is None:
            result["validation_errors"].append("Code generation returned None")
            self._total_failures += 1
            return result

        # Parse into CodePatch
        try:
            patch = self._parse_raw_output(raw_text)
            result["patch"] = patch
        except Exception as exc:
            result["validation_errors"].append(f"Failed to parse code generation output: {exc}")
            self._total_failures += 1
            return result

        if result["patch"] is None:
            result["validation_errors"].append("Parsed patch is None")
            self._total_failures += 1
            return result

        if write_category == "skill_plugin":
            self._normalize_skill_plugin_patch_paths(result["patch"])

        # Safety validation via PatchPlan
        try:
            from self_improve.patch_plan import PatchPlan, check_denied_patterns

            paths = [fd.path for fd in result["patch"].files]
            plan = PatchPlan(write_category=write_category)
            plan.files_to_modify = paths
            plan.files_to_create = []

            scope_errors = plan.validate_scope()
            if scope_errors:
                result["validation_errors"].extend(scope_errors)

            boundary_errors = plan.validate_write_boundaries()
            if boundary_errors:
                result["validation_errors"].extend(boundary_errors)

            for fd in result["patch"].files:
                denied = check_denied_patterns(fd.new_content)
                if denied:
                    result["validation_errors"].extend(
                        [f"Denied pattern in {fd.path}: {d}" for d in denied]
                    )
        except Exception as exc:
            result["validation_errors"].append(f"Safety validation error: {exc}")

        if result["validation_errors"]:
            self._total_failures += 1
            return result

        # Sandbox evaluation
        try:
            from codegen.sandbox import Sandbox
            sandbox = Sandbox()
            report = await sandbox.evaluate(result["patch"])
            result["sandbox_report"] = report
            self._total_validations += 1

            if not report.overall_passed:
                result["validation_errors"].append("Sandbox validation failed")
                result["success"] = False
            else:
                result["success"] = True
        except Exception as exc:
            result["validation_errors"].append(f"Sandbox evaluation error: {exc}")
            self._total_failures += 1

        return result

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Find the outermost JSON object in raw LLM output.

        Falls back to reconstructing JSON from labelled markdown code blocks
        when the model ignores the JSON-only instruction and returns bare code
        (common with MoE models).
        """
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        depth = 0
        start = -1
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    return cleaned[start : i + 1]

        # Fallback: reconstruct from labelled code blocks
        # Pattern: ```python\n# handler.py\n...code...\n```
        #   or:    ## handler.py\n```python\n...code...\n```
        blocks = re.findall(
            r"(?:^|\n)(?:#+ *(\S+\.py)\s*\n)?```(?:python)?\s*\n"
            r"(?:#\s*(\S+\.py)\s*\n)?"
            r"(.*?)```",
            cleaned,
            re.DOTALL,
        )
        if blocks:
            files = []
            for heading_name, comment_name, code in blocks:
                fname = heading_name or comment_name or ""
                code = code.strip()
                if not fname and code:
                    if "def run(" in code:
                        fname = "handler.py"
                    elif "def handle(" in code or "PLUGIN_MANIFEST" in code:
                        fname = "__init__.py"
                if fname and code:
                    files.append({"path": fname, "content": code})
            if files:
                logger.info(
                    "JSON extraction fallback: reconstructed %d file(s) from code blocks",
                    len(files),
                )
                return json.dumps({"files": files})

        logger.debug("Raw LLM output (first 500 chars): %s", text[:500])
        return None

    @staticmethod
    def _apply_edits(original: str, edits: list[dict[str, str]]) -> str | None:
        """Apply search-and-replace edits to *original*, returning new content.

        Returns ``None`` if any search string is not found.
        """
        result = original
        for edit in edits:
            search = edit.get("search", "")
            replace = edit.get("replace", "")
            if not search:
                continue
            if search not in result:
                logger.warning(
                    "Edit search string not found (first 80 chars): %s",
                    search[:80],
                )
                return None
            result = result.replace(search, replace, 1)
        return result

    @staticmethod
    def _read_original_file(path: str) -> str | None:
        """Read original file content from the brain directory."""
        rel = path.replace("brain/", "") if path.startswith("brain/") else path
        full = _BRAIN_DIR / rel
        if full.exists():
            try:
                return full.read_text(encoding="utf-8")
            except Exception:
                pass
        return None

    _REQUIRED_KEYS = frozenset({"files"})

    @staticmethod
    def _parse_raw_output(raw_text: str, plan_id: str = "") -> "CodePatch | None":
        """Parse raw LLM output into a :class:`CodePatch`.

        Stateless — does not depend on PatchProvider or any external instance.
        Reuses the same JSON contract that the CoderServer prompt specifies.
        """
        from self_improve.code_patch import CodePatch, FileDiff

        try:
            raw_json = CodeGenService._extract_json(raw_text)
            if raw_json is None:
                logger.error(
                    "No JSON object found in codegen output (%d chars)", len(raw_text)
                )
                return None

            data = json.loads(raw_json)

            if "files" not in data:
                file_map = {
                    k: v for k, v in data.items()
                    if isinstance(k, str) and k.endswith(".py") and isinstance(v, str)
                }
                if file_map:
                    data["files"] = [
                        {"path": path, "content": content}
                        for path, content in file_map.items()
                    ]
                else:
                    missing = CodeGenService._REQUIRED_KEYS - set(data.keys())
                    logger.error("Codegen JSON missing required keys %s", missing)
                    return None

            files_raw = data.get("files", [])
            if isinstance(files_raw, dict):
                files_raw = [
                    {"path": path, "content": content}
                    for path, content in files_raw.items()
                    if isinstance(path, str) and isinstance(content, str)
                ]
            if not isinstance(files_raw, list) or not files_raw:
                logger.error("Codegen JSON 'files' is empty or not a list")
                return None

            files: list[FileDiff] = []
            for fd in files_raw:
                path = fd.get("path", "")
                edits = fd.get("edits", [])
                content = fd.get("content", "")
                diff = fd.get("diff", "")

                if not path:
                    logger.warning("Skipping file entry with no path")
                    continue

                if edits and isinstance(edits, list):
                    original = CodeGenService._read_original_file(path)
                    if original is None:
                        logger.warning(
                            "Cannot read original for edit application: %s", path
                        )
                        continue
                    new_content = CodeGenService._apply_edits(original, edits)
                    if new_content is None:
                        logger.warning(
                            "Edit application failed for %s (search mismatch)", path
                        )
                        continue
                    files.append(
                        FileDiff(
                            path=path,
                            original_content=original,
                            new_content=new_content,
                        )
                    )
                elif content or diff:
                    files.append(
                        FileDiff(path=path, new_content=content, diff=diff)
                    )
                else:
                    logger.warning(
                        "Skipping file entry with no edits or content: %s", path
                    )
                    continue

            if not files:
                logger.error("No valid file entries after parsing codegen output")
                return None

            return CodePatch(
                plan_id=plan_id,
                provider="codegen_service",
                files=files,
                description=data.get("description", ""),
                test_instructions=data.get("test_plan", ""),
                confidence=data.get("confidence", 0.5),
            )
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON decode failed at char %d: %s", exc.pos or -1, exc.msg
            )
            return None
        except Exception:
            logger.exception("Failed to parse codegen output")
            return None

    @staticmethod
    def _normalize_skill_plugin_patch_paths(patch: Any) -> None:
        """Make sandbox paths match deployed plugin package layout."""
        files = getattr(patch, "files", []) or []
        for fd in files:
            path = getattr(fd, "path", "")
            if not path:
                continue
            if path.startswith("brain/"):
                continue
            filename = path.split("/")[-1]
            fd.path = f"brain/tools/plugins/_gen/{filename}"

    def _check_evidence_sufficiency(
        self, risk_tier: int, evidence_bundle: list[Any]
    ) -> dict[str, Any]:
        """Check if evidence is sufficient for the given risk tier.

        Contract 2 from the plan: CodeGen MUST NOT run on weak evidence.
        """
        if risk_tier == 0:
            return {"sufficient": True, "tier": 0, "reason": "tier 0 — minimal evidence ok"}

        if not evidence_bundle:
            return {
                "sufficient": False,
                "tier": risk_tier,
                "reason": f"tier {risk_tier} requires at least 1 evidence artifact",
            }

        structured = [e for e in evidence_bundle if isinstance(e, dict)]
        if structured:
            meaningful = [
                e for e in structured
                if e.get("source_type") != "none_found"
                and (e.get("citations") or float(e.get("relevance", 0.0) or 0.0) > 0.0)
            ]
            if not meaningful:
                return {
                    "sufficient": False,
                    "tier": risk_tier,
                    "reason": f"tier {risk_tier} requires meaningful documentation evidence",
                    "evidence_count": len(evidence_bundle),
                }

        return {"sufficient": True, "tier": risk_tier, "evidence_count": len(evidence_bundle)}

    def get_status(self) -> dict[str, Any]:
        """Status for dashboard."""
        coder_status: dict[str, Any] = {"available": False}
        if self._coder_server:
            try:
                coder_status = self._coder_server.get_status()
            except Exception:
                pass

        return {
            "coder": coder_status,
            "total_generations": self._total_generations,
            "total_validations": self._total_validations,
            "total_failures": self._total_failures,
            "active_consumer": coder_status.get("active_consumer", "") or self._active_consumer,
            "last_consumer": coder_status.get("last_consumer", "") or self._last_consumer,
        }

    async def shutdown(self) -> None:
        """Shut down the coder server if running."""
        if self._coder_server:
            try:
                await self._coder_server.shutdown()
            except Exception:
                pass
