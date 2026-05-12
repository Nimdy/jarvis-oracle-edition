"""Internal Query Interface — headless tool runtime for autonomous research.

Executes tools (academic search, web search, codebase, memory) without
requiring user speech, Pi broadcast, or TTS. Returns structured
ResearchResult objects.

This is the "motor cortex" — it turns ResearchIntent into action.

Search policy: scholarly-first. Academic search (Semantic Scholar + Crossref)
is the default for all autonomous research. DDG web search is only used for
realtime queries (weather, news, sports) or explicit user requests.
Autonomy scholarly lane gets NO DDG fallback -- if academic returns 0 hits,
the query fails cleanly rather than falling back to poisonable web results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from autonomy.research_intent import (
    ResearchIntent, ResearchResult, ResearchFinding, ToolHint,
)

logger = logging.getLogger(__name__)


class InternalQueryInterface:
    """Headless tool runtime: executes research queries without conversation."""

    def __init__(self) -> None:
        self._queries_executed: int = 0
        self._last_query_time: float = 0.0

    async def execute(self, intent: ResearchIntent) -> ResearchResult:
        """Route an intent to the appropriate tool and return results."""
        t0 = time.monotonic()
        intent.status = "running"
        intent.started_at = time.time()

        hint = intent.source_hint
        if hint == "any":
            hint = self._auto_route(intent.question)

        try:
            if hint == "academic":
                result = await self._execute_academic(intent)
            elif hint == "web":
                from tools.academic_search_tool import classify_search_lane
                lane = classify_search_lane(intent.question)
                if lane == "scholarly":
                    result = await self._execute_academic(intent)
                    if not result.findings:
                        logger.info("Academic search returned 0 hits, no DDG fallback for autonomy")
                else:
                    result = await self._execute_web(intent)
            elif hint == "codebase":
                result = await self._execute_codebase(intent)
            elif hint == "memory":
                result = await self._execute_memory(intent)
            elif hint == "introspection":
                result = await self._execute_introspection(intent)
            else:
                result = await self._execute_codebase(intent)
                if not result.findings:
                    result = await self._execute_memory(intent)

            elapsed = (time.monotonic() - t0) * 1000
            self._queries_executed += 1
            self._last_query_time = time.time()

            intent.status = "completed" if result.success else "failed"
            intent.completed_at = time.time()
            intent.result = result

            logger.info(
                "Research query completed: %s -> %d findings (%.0fms, tool=%s)",
                intent.question[:50], len(result.findings), elapsed, result.tool_used,
            )
            return result

        except Exception as exc:
            intent.status = "failed"
            intent.completed_at = time.time()
            result = ResearchResult(
                intent_id=intent.id,
                tool_used=hint or "unknown",
                success=False,
                error=str(exc)[:200],
            )
            intent.result = result
            logger.warning("Research query failed: %s — %s", intent.question[:50], exc)
            return result

    @property
    def queries_executed(self) -> int:
        return self._queries_executed

    def get_stats(self) -> dict[str, Any]:
        return {
            "queries_executed": self._queries_executed,
            "last_query_time": self._last_query_time,
        }

    # -- tool executors -------------------------------------------------------

    async def _execute_academic(self, intent: ResearchIntent) -> ResearchResult:
        """Search academic databases (Semantic Scholar + Crossref).

        Uses bulk search with minCitationCount + year filters and batch
        enrichment instead of individual detail calls.
        """
        from tools.academic_search_tool import (
            search_academic, confidence_for_result,
        )

        results = await search_academic(intent.question, max_results=intent.max_results)

        findings = []
        enriched_count = 0
        for r in results:
            conf = confidence_for_result(r)
            source_type = "peer_reviewed" if r.is_peer_reviewed else (
                "preprint" if r.is_preprint else "unverified"
            )
            authors_str = ", ".join(r.authors[:5])
            if len(r.authors) > 5:
                authors_str += " et al."

            if r.abstract and len(r.abstract) >= 100:
                content = r.abstract[:1500]
                content_depth = "abstract"
            elif r.tldr:
                content = r.tldr
                content_depth = "tldr"
            else:
                content = r.title
                content_depth = "title_only"

            if content_depth != "title_only":
                enriched_count += 1

            provenance_parts = ["academic:"]
            if r.doi:
                provenance_parts.append(r.doi)
            elif r.url:
                provenance_parts.append(r.url)
            if r.venue and r.year:
                provenance_parts.append(f"({r.venue}, {r.year})")
            elif r.venue:
                provenance_parts.append(f"({r.venue})")
            elif r.year:
                provenance_parts.append(f"({r.year})")
            provenance_parts.append(f"[depth:{content_depth}]")

            findings.append(ResearchFinding(
                content=content,
                provenance=" ".join(provenance_parts),
                confidence=conf,
                url=r.url or r.doi_url,
                doi=r.doi,
                doi_url=r.doi_url,
                authors=authors_str,
                year=r.year,
                venue=r.venue,
                citation_count=r.citation_count,
                influential_citation_count=r.influential_citation_count,
                source_type=source_type,
                source_provider=r.source_provider,
                open_access_pdf_url=r.open_access_pdf_url,
            ))

        summary = ""
        if findings:
            titles = [r.title for r in results[:3]]
            depth_note = f" ({enriched_count}/{len(findings)} with content)" if findings else ""
            summary = f"Found {len(findings)} scholarly results{depth_note}. Top: {'; '.join(titles)}"
        else:
            summary = "No scholarly sources found for this query."

        return ResearchResult(
            intent_id=intent.id,
            tool_used="academic_search",
            findings=findings,
            summary=summary,
            raw_query=intent.question,
            success=len(findings) > 0,
        )

    async def _execute_web(self, intent: ResearchIntent) -> ResearchResult:
        """Search the web via DuckDuckGo (fenced)."""
        from tools.web_search_tool import search_web

        results = await search_web(intent.question, max_results=intent.max_results)

        findings = [
            ResearchFinding(
                content=r.snippet[:300],
                provenance=f"web_search: {r.url}",
                confidence=0.35,
                url=r.url,
                source_type="web",
                source_provider="ddg",
            )
            for r in results
        ]

        summary = ""
        if findings:
            titles = [r.title for r in results[:3]]
            summary = f"Found {len(findings)} web results. Top: {'; '.join(titles)}"

        return ResearchResult(
            intent_id=intent.id,
            tool_used="web_search",
            findings=findings,
            summary=summary,
            raw_query=intent.question,
            success=len(findings) > 0,
        )

    async def _execute_codebase(self, intent: ResearchIntent) -> ResearchResult:
        """Query the codebase via AST index + budgeted source context + Library chunks.

        Three-layer retrieval:
        1. AST index keyword search -> matched modules and symbols
        2. get_budgeted_context() -> actual source code spans (up to 6000 tokens)
        3. Library semantic search -> relevant codebase/doc chunks
        """
        from tools.codebase_tool import codebase_index

        if not codebase_index._modules:
            codebase_index.build()

        findings: list[ResearchFinding] = []
        matched_files: list[str] = []
        matched_symbols: list[str] = []

        stop_words = {
            "what", "how", "does", "do", "is", "are", "my", "the", "a", "an",
            "and", "or", "in", "of", "to", "for", "with", "from", "about",
            "read", "summarize", "explain", "describe", "work", "have", "has",
            "can", "when", "where", "which", "its", "it", "this", "that",
            "overall", "current", "known", "key", "all", "each", "their",
            "trace", "list", "every",
        }
        keywords = [
            w.strip("?.,!:;()\"'")
            for w in intent.question.split()
            if len(w.strip("?.,!:;()\"'")) > 2
            and w.strip("?.,!:;()\"'").lower() not in stop_words
        ]

        for kw in keywords:
            for sym in codebase_index.search(kw, limit=5):
                if sym.file and sym.file not in matched_files:
                    matched_files.append(sym.file)
                if sym.fqn and sym.fqn not in matched_symbols:
                    matched_symbols.append(sym.fqn)

        if not matched_files:
            for kw in keywords:
                kw_lower = kw.lower()
                for fqn, info in codebase_index._modules.items():
                    if kw_lower in fqn.lower() or kw_lower in info.docstring.lower():
                        if info.file not in matched_files:
                            matched_files.append(info.file)

        if matched_files:
            budgeted = codebase_index.get_budgeted_context(
                target_files=matched_files[:5],
                referenced_symbols=matched_symbols[:10],
                max_tokens=6000,
            )
            if budgeted.strip():
                findings.append(ResearchFinding(
                    content=budgeted,
                    provenance="codebase_budgeted_context",
                    confidence=0.85,
                    source_type="codebase",
                    source_provider="codebase",
                ))

        try:
            from library.index import library_index
            if library_index.available:
                search_results = library_index.search(
                    intent.question, top_k=5, source_id_filter="",
                )
                from library.chunks import chunk_store as lib_chunk_store
                from library.source import source_store as lib_source_store
                for sr in search_results:
                    sid = sr.get("source_id", "")
                    chunk_id = sr.get("chunk_id", "")
                    score = sr.get("distance", 1.0)
                    src = lib_source_store.get(sid)
                    is_codebase = src and src.source_type == "codebase" if src else False
                    if not is_codebase and score >= 0.8:
                        continue
                    full_chunk = lib_chunk_store.get(chunk_id)
                    text = full_chunk.text if full_chunk else sr.get("text_preview", "")
                    if text and len(text) > 50:
                        findings.append(ResearchFinding(
                            content=text[:3000],
                            provenance=f"library:{sid}",
                            confidence=0.80 if is_codebase else 0.65,
                        ))
        except Exception as exc:
            logger.debug("Library search in codebase query failed: %s", exc)

        if not findings:
            answer = codebase_index.answer_query(intent.question)
            if answer and "No symbols found" not in answer:
                findings.append(ResearchFinding(
                    content=answer[:2000],
                    provenance="codebase_index",
                    confidence=0.7,
                    source_type="codebase",
                    source_provider="codebase",
                ))

        summary_parts = []
        if matched_files:
            summary_parts.append(f"Matched {len(matched_files)} module(s): {', '.join(matched_files[:5])}")
        if findings:
            summary_parts.append(f"{len(findings)} finding(s) with source context")
        summary = ". ".join(summary_parts) if summary_parts else "No relevant code found."

        return ResearchResult(
            intent_id=intent.id,
            tool_used="codebase",
            findings=findings,
            summary=summary[:300],
            raw_query=intent.question,
            success=len(findings) > 0,
        )

    async def _execute_memory(self, intent: ResearchIntent) -> ResearchResult:
        """Search memories by keyword and semantic similarity."""
        from memory.storage import memory_storage

        keywords = intent.question.lower().split()
        relevant_tags = [w for w in keywords if len(w) > 3][:5]

        findings = []
        seen_ids: set[str] = set()

        for tag in relevant_tags:
            for mem in memory_storage.get_by_tag(tag):
                if mem.id in seen_ids:
                    continue
                seen_ids.add(mem.id)
                payload_str = str(mem.payload)[:300]
                findings.append(ResearchFinding(
                    content=payload_str,
                    provenance=f"memory:{mem.type}:{mem.id}",
                    confidence=mem.weight,
                    source_type="memory",
                    source_provider="memory",
                ))
                if len(findings) >= intent.max_results:
                    break
            if len(findings) >= intent.max_results:
                break

        try:
            from memory.search import semantic_search
            semantic_results = semantic_search(intent.question, top_k=3)
            for mem in semantic_results:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    findings.append(ResearchFinding(
                        content=str(mem.payload)[:300],
                        provenance=f"semantic_search:{mem.type}:{mem.id}",
                        confidence=min(0.8, mem.weight),
                        source_type="memory",
                        source_provider="memory",
                    ))
        except Exception as exc:
            logger.warning("Memory semantic search failed in research: %s", exc)

        summary = f"Found {len(findings)} relevant memories." if findings else "No relevant memories found."

        return ResearchResult(
            intent_id=intent.id,
            tool_used="memory",
            findings=findings,
            summary=summary,
            raw_query=intent.question,
            success=len(findings) > 0,
        )

    async def _execute_introspection(self, intent: ResearchIntent) -> ResearchResult:
        """Gather information from internal subsystem state."""
        findings = []

        try:
            from consciousness.modes import mode_manager
            findings.append(ResearchFinding(
                content=f"Current mode: {mode_manager.mode}, profile: {mode_manager.get_state()}",
                provenance="introspection:mode_manager",
                confidence=0.9,
                source_type="introspection",
                source_provider="introspection",
            ))
        except Exception:
            pass

        try:
            from memory.storage import memory_storage
            stats = memory_storage.get_stats()
            findings.append(ResearchFinding(
                content=f"Memory stats: {stats}",
                provenance="introspection:memory",
                confidence=0.9,
                source_type="introspection",
                source_provider="introspection",
            ))
        except Exception:
            pass

        return ResearchResult(
            intent_id=intent.id,
            tool_used="introspection",
            findings=findings,
            summary=f"Gathered {len(findings)} introspection data points.",
            raw_query=intent.question,
            success=len(findings) > 0,
        )

    # -- routing heuristic ----------------------------------------------------

    @staticmethod
    def _auto_route(question: str) -> ToolHint:
        q = question.lower()
        codebase_signals = ("code", "module", "function", "class", "file", "implementation",
                            "orchestrator", "kernel", "pipeline", "hemisphere", "policy")
        academic_signals = ("research", "technique", "approach", "theory", "framework",
                            "heuristic", "best practice", "how do humans", "literature",
                            "paper", "study", "journal", "mechanism", "model",
                            "evidence", "cognitive", "neural", "consciousness")
        memory_signals = ("remember", "recall", "past", "previous", "earlier", "history")

        code_score = sum(1 for s in codebase_signals if s in q)
        academic_score = sum(1 for s in academic_signals if s in q)
        mem_score = sum(1 for s in memory_signals if s in q)

        if code_score > academic_score and code_score > mem_score:
            return "codebase"
        if academic_score > code_score and academic_score > mem_score:
            return "academic"
        if mem_score > 0:
            return "memory"
        return "academic"
