"""Study pipeline — concept extraction and structured claim generation.

Processes unstudied Sources from the library and produces:
  1. Concept tags on each chunk (keyword + pattern extraction)
  2. Derived claim memories (structured pointer payloads with chunk_ids)
  3. Concept graph entries (co-occurring concepts within chunks)

When an LLM callback is available and the source has sufficient content,
uses LLM-based structured extraction (understanding argumentative structure:
what was tried, what failed, what succeeded, conclusions). Falls back to
regex-only extraction when the LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

MAX_CONCEPTS_PER_SOURCE = 20
MAX_CLAIMS_REGEX = 3
MAX_CLAIMS_LLM = 8
MAX_CHUNK_REFS_PER_CLAIM = 3
LLM_MIN_CONTENT_CHARS = 300

_llm_callback: Callable[[str], Awaitable[str]] | None = None


def _try_post_study_graduation(
    source: Any, chunks: list[Any],
    concepts: list[str], claim_texts: list[str],
) -> None:
    """Graduate studied source to Blue Diamonds if it meets quality criteria."""
    try:
        # Codebase sources are always re-obtainable from the filesystem
        # and don't belong in the permanent curated knowledge archive.
        if getattr(source, "source_type", "") == "codebase":
            return

        from library.blue_diamonds import (
            BlueDiamondsArchive, GRADUATION_MIN_QUALITY,
            GRADUATION_ELIGIBLE_DEPTHS,
        )
        archive = BlueDiamondsArchive.get_instance()

        if archive.is_archived(source.source_id):
            return

        if source.content_depth not in GRADUATION_ELIGIBLE_DEPTHS:
            return
        if source.quality_score < GRADUATION_MIN_QUALITY:
            archive.log_rejection(
                source.source_id,
                f"post_study:quality_{source.quality_score:.2f}"
            )
            return

        reason = f"post_study+quality_{source.quality_score:.2f}+{source.content_depth}"
        archive.graduate(source, chunks, concepts, claim_texts, reason)
    except Exception as exc:
        logger.debug("Post-study Blue Diamond graduation failed: %s", exc)

# ── Study telemetry (read by eval sidecar PVL) ─────────────────────
_study_telemetry: dict[str, int | float] = {
    "llm_extractions": 0,
    "regex_fallbacks": 0,
    "total_claims": 0,
    "total_concepts": 0,
    "sources_studied": 0,
    "sources_skipped_codebase_claims": 0,
}


def get_study_telemetry() -> dict[str, int | float]:
    """Return study pipeline counters for the eval collector.

    Session counters track this-session activity. Cumulative fields are
    pulled from the library DB so PVL contracts don't reset on restart.
    """
    result = dict(_study_telemetry)
    try:
        from library.source import source_store
        if source_store:
            stats = source_store.get_stats()
            result["cumulative_studied"] = stats.get("studied", 0)
            result["cumulative_total"] = stats.get("total", 0)
    except Exception:
        pass
    return result

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "and", "but", "if", "or", "because",
    "while", "what", "which", "who", "whom", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "you",
    "also", "using", "based", "used", "propose", "show", "paper", "results",
})

_CODE_STOP_WORDS = frozenset({
    "return", "none", "true", "false", "class", "self", "except", "finally",
    "raise", "yield", "async", "await", "lambda", "assert", "global",
    "nonlocal", "elif", "pass", "break", "continue", "with", "else", "del",
    "from", "try", "import", "init", "args", "kwargs", "super", "isinstance",
    "getattr", "setattr", "hasattr", "staticmethod", "property", "logger",
    "exception", "error", "traceback", "dict", "list", "tuple", "print",
    "type", "object", "string", "float", "bool", "bytes", "info", "debug",
    "warning", "optional", "union", "callable", "classmethod", "abstractmethod",
    "dataclass", "field", "frozen", "slots", "override", "protocol",
})

_DEFINITION_RE = re.compile(
    r"([A-Z][\w\s-]{2,40}?)\s+(?:is|are)\s+defined\s+as\s+(.{10,200}?)(?:\.|$)",
    re.IGNORECASE,
)
_PROPOSAL_RE = re.compile(
    r"(?:we\s+)?(?:propose|present|introduce)\s+(.{5,150}?)(?:\.|,|$)",
    re.IGNORECASE,
)
_RESULT_RE = re.compile(
    r"(?:achieves?|outperforms?|improves?)\s+(.{5,150}?)(?:\.|,|$)",
    re.IGNORECASE,
)
_METHOD_RE = re.compile(
    r"(?:our\s+)?(?:method|approach|algorithm|technique|framework)\s+(.{5,100}?)(?:\.|,|$)",
    re.IGNORECASE,
)
_CAPITALIZED_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

@dataclass
class Claim:
    """A structured claim extracted from a source."""
    claim_type: str
    text: str
    chunk_ids: list[str]
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "text": self.text,
            "chunk_ids": self.chunk_ids,
            "confidence": self.confidence,
        }


@dataclass
class StudyResult:
    """Output of studying a single source."""
    source_id: str
    concepts: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0
    extraction_method: str = "regex"


_STRUCTURED_EXTRACTION_PROMPT = """You are a research analysis system. Given the following academic paper content, extract structured knowledge claims.

PAPER CONTENT:
{content}

Extract the following as a JSON object. Be precise — only include what the paper actually states:

{{
  "problem": "What problem does this paper address? (1-2 sentences)",
  "approaches_tried": ["List of approaches/methods tried or compared"],
  "what_failed": ["Any approaches that were found inadequate, with brief reason"],
  "what_worked": "The main finding or successful approach (1-2 sentences)",
  "key_metrics": ["Specific quantitative results, benchmarks, or measurements"],
  "conclusion": "The paper's main conclusion or recommendation (1-2 sentences)",
  "limitations": ["Known limitations or caveats mentioned by the authors"],
  "novel_concepts": ["New terms, architectures, or techniques introduced"]
}}

Rules:
- Only extract what is explicitly stated in the text
- If a field has no content, use an empty string or empty list
- Keep each entry concise (under 150 characters)
- Do not invent or infer claims not in the text
- Return ONLY the JSON object, no other text"""


def set_llm_callback(callback: Callable[[str], Awaitable[str]] | None) -> None:
    """Set the LLM function for structured extraction. Called from main.py."""
    global _llm_callback
    _llm_callback = callback
    if callback:
        logger.info("Study pipeline: LLM-based extraction enabled")


def study_source(source_id: str) -> StudyResult:
    """Process a single unstudied source: extract concepts and claims.

    Uses LLM-based structured extraction when available and content is
    sufficient. Falls back to regex extraction otherwise.
    """
    t0 = time.time()
    result = StudyResult(source_id=source_id)

    try:
        from library.source import source_store
        from library.chunks import chunk_store

        source = source_store.get(source_id)
        if not source:
            result.error = "source not found"
            return result

        if source.content_depth in ("metadata_only", "title_only"):
            source_store.mark_studied(source_id)
            result.error = f"skipped:insufficient_content ({source.content_depth})"
            result.duration_ms = (time.time() - t0) * 1000
            logger.debug("Skipping study for %s: %s", source_id[:12], result.error)
            return result

        chunks = chunk_store.get_for_source(source_id)
        if not chunks:
            result.error = "no chunks"
            source_store.mark_study_error(source_id, "no chunks available")
            return result

        if getattr(source, "source_type", "") == "codebase":
            result.extraction_method = "skipped:codebase"
            _study_telemetry["sources_studied"] += 1
            _study_telemetry["sources_skipped_codebase_claims"] += 1
            source_store.mark_studied(source_id)
            logger.info(
                "Studied codebase source %s: concepts+claims skipped (%.0fms)",
                source_id[:12], (time.time() - t0) * 1000,
            )
            result.duration_ms = (time.time() - t0) * 1000
            return result

        all_text = "\n".join(c.text for c in chunks)
        all_concepts: list[str] = []
        per_chunk_concepts: dict[str, list[str]] = {}

        for chunk in chunks:
            chunk_concepts = _extract_concepts(chunk.text)
            deduped = _deduplicate_concepts(chunk_concepts)[:10]
            per_chunk_concepts[chunk.chunk_id] = deduped
            all_concepts.extend(chunk_concepts)

            if deduped:
                chunk_store.update_concepts(chunk.chunk_id, deduped)

        unique_concepts = _deduplicate_concepts(all_concepts)[:MAX_CONCEPTS_PER_SOURCE]
        result.concepts = unique_concepts

        try:
            from library.concept_graph import concept_graph
            for chunk_id, concepts_for_chunk in per_chunk_concepts.items():
                if len(concepts_for_chunk) >= 2:
                    concept_graph.add_concepts(chunk_id, concepts_for_chunk)
        except Exception:
            pass

        llm_claims = _try_llm_extraction(all_text, chunks, source)
        if llm_claims:
            unique_claims = _deduplicate_claims(llm_claims)[:MAX_CLAIMS_LLM]
            result.extraction_method = "llm"
            _study_telemetry["llm_extractions"] += 1
        else:
            all_claims: list[Claim] = []
            for chunk in chunks:
                chunk_claims = _extract_claims(chunk.text, chunk.chunk_id)
                all_claims.extend(chunk_claims)
            unique_claims = _deduplicate_claims(all_claims)[:MAX_CLAIMS_REGEX]
            result.extraction_method = "regex"
            _study_telemetry["regex_fallbacks"] += 1

        result.claims = unique_claims
        _study_telemetry["total_claims"] += len(unique_claims)
        _study_telemetry["total_concepts"] += len(unique_concepts)
        _study_telemetry["sources_studied"] += 1

        _create_claim_memories(source, unique_claims)
        source_store.mark_studied(source_id)

        _try_post_study_graduation(source, chunks, unique_concepts,
                                   [c.text for c in unique_claims])

        logger.info(
            "Studied source %s: %d concepts, %d claims (%s, %.0fms)",
            source_id[:12], len(unique_concepts), len(unique_claims),
            result.extraction_method, (time.time() - t0) * 1000,
        )

    except Exception as exc:
        result.error = str(exc)[:200]
        try:
            from library.source import source_store
            source_store.mark_study_error(source_id, result.error)
        except Exception:
            pass
        logger.warning("Study failed for %s: %s", source_id, exc)

    result.duration_ms = (time.time() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# LLM-based structured extraction
# ---------------------------------------------------------------------------

def _try_llm_extraction(
    all_text: str, chunks: list, source: Any,
) -> list[Claim] | None:
    """Attempt LLM-based structured extraction. Returns None if unavailable or fails."""
    if not _llm_callback:
        return None

    if len(all_text) < LLM_MIN_CONTENT_CHARS:
        return None

    try:
        import asyncio
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        content_for_llm = all_text[:3000]
        prompt = _STRUCTURED_EXTRACTION_PROMPT.format(content=content_for_llm)

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_llm_sync, prompt)
                raw_response = future.result(timeout=60)
        else:
            raw_response = asyncio.run(_llm_callback(prompt))

        if not raw_response:
            return None

        return _parse_llm_claims(raw_response, chunks)

    except Exception as exc:
        logger.debug("LLM extraction failed, falling back to regex: %s", exc)
        return None


def _run_llm_sync(prompt: str) -> str:
    """Run the async LLM callback from a sync context."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_llm_callback(prompt))
    finally:
        loop.close()


def _parse_llm_claims(raw: str, chunks: list) -> list[Claim] | None:
    """Parse the LLM's JSON response into Claim objects."""
    json_str = raw.strip()
    if "```" in json_str:
        parts = json_str.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                json_str = part
                break

    start = json_str.find("{")
    end = json_str.rfind("}")
    if start == -1 or end == -1:
        return None
    json_str = json_str[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    chunk_ids = [c.chunk_id for c in chunks[:MAX_CHUNK_REFS_PER_CLAIM]]
    claims: list[Claim] = []

    if problem := data.get("problem", ""):
        if len(str(problem)) > 10:
            claims.append(Claim(
                claim_type="problem",
                text=f"Problem: {str(problem)[:200]}",
                chunk_ids=chunk_ids,
                confidence=0.70,
            ))

    if worked := data.get("what_worked", ""):
        if len(str(worked)) > 10:
            claims.append(Claim(
                claim_type="result",
                text=f"Finding: {str(worked)[:200]}",
                chunk_ids=chunk_ids,
                confidence=0.75,
            ))

    if conclusion := data.get("conclusion", ""):
        if len(str(conclusion)) > 10:
            claims.append(Claim(
                claim_type="conclusion",
                text=f"Conclusion: {str(conclusion)[:200]}",
                chunk_ids=chunk_ids,
                confidence=0.75,
            ))

    for metric in _ensure_list(data.get("key_metrics", [])):
        if len(str(metric)) > 5:
            claims.append(Claim(
                claim_type="metric",
                text=f"Metric: {str(metric)[:150]}",
                chunk_ids=chunk_ids,
                confidence=0.70,
            ))

    for failed in _ensure_list(data.get("what_failed", [])):
        if len(str(failed)) > 10:
            claims.append(Claim(
                claim_type="negative_result",
                text=f"Negative: {str(failed)[:150]}",
                chunk_ids=chunk_ids,
                confidence=0.65,
            ))

    for approach in _ensure_list(data.get("approaches_tried", [])):
        if len(str(approach)) > 10:
            claims.append(Claim(
                claim_type="method",
                text=f"Method: {str(approach)[:150]}",
                chunk_ids=chunk_ids,
                confidence=0.60,
            ))

    for concept in _ensure_list(data.get("novel_concepts", [])):
        if len(str(concept)) > 3:
            claims.append(Claim(
                claim_type="definition",
                text=f"Concept: {str(concept)[:150]}",
                chunk_ids=chunk_ids,
                confidence=0.65,
            ))

    for limitation in _ensure_list(data.get("limitations", [])):
        if len(str(limitation)) > 10:
            claims.append(Claim(
                claim_type="limitation",
                text=f"Limitation: {str(limitation)[:150]}",
                chunk_ids=chunk_ids,
                confidence=0.60,
            ))

    return claims if claims else None


def _ensure_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val:
        return [val]
    return []


# ---------------------------------------------------------------------------
# Regex-based extraction (fallback)
# ---------------------------------------------------------------------------

def _extract_concepts(text: str) -> list[str]:
    """Extract concept candidates from chunk text using TF + patterns."""
    concepts: list[str] = []

    words = text.lower().split()
    word_freq: dict[str, int] = {}
    for w in words:
        w = w.strip(".,;:!?()[]\"'")
        if len(w) > 3 and w not in _STOP_WORDS and w not in _CODE_STOP_WORDS and w.isalpha():
            word_freq[w] = word_freq.get(w, 0) + 1

    top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    concepts.extend(t for t, _ in top_terms if _ >= 2)

    for match in _CAPITALIZED_PHRASE_RE.finditer(text):
        phrase = match.group(1).lower()
        if len(phrase) > 5 and phrase not in _STOP_WORDS:
            concepts.append(phrase)

    for match in _DEFINITION_RE.finditer(text):
        term = match.group(1).strip().lower()
        if len(term) > 2:
            concepts.append(term)

    return concepts


def _extract_claims(text: str, chunk_id: str) -> list[Claim]:
    """Extract structured claims from chunk text using regex patterns."""
    claims: list[Claim] = []

    for match in _DEFINITION_RE.finditer(text):
        term = match.group(1).strip()
        definition = match.group(2).strip()
        claims.append(Claim(
            claim_type="definition",
            text=f"{term} is defined as {definition}",
            chunk_ids=[chunk_id],
            confidence=0.7,
        ))

    for match in _PROPOSAL_RE.finditer(text):
        proposal = match.group(1).strip()
        if len(proposal) > 10:
            claims.append(Claim(
                claim_type="proposal",
                text=f"Proposed: {proposal}",
                chunk_ids=[chunk_id],
                confidence=0.6,
            ))

    for match in _RESULT_RE.finditer(text):
        result_text = match.group(1).strip()
        if len(result_text) > 10:
            claims.append(Claim(
                claim_type="result",
                text=f"Result: {result_text}",
                chunk_ids=[chunk_id],
                confidence=0.65,
            ))

    for match in _METHOD_RE.finditer(text):
        method = match.group(1).strip()
        if len(method) > 10:
            claims.append(Claim(
                claim_type="method",
                text=f"Method: {method}",
                chunk_ids=[chunk_id],
                confidence=0.6,
            ))

    return claims


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate_concepts(concepts: list[str]) -> list[str]:
    """Deduplicate and normalize concepts."""
    seen: set[str] = set()
    unique: list[str] = []
    for c in concepts:
        c = c.strip().lower()
        if c and c not in seen and len(c) > 2:
            seen.add(c)
            unique.append(c)
    return unique


def _deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    """Deduplicate claims by normalized text."""
    seen: set[str] = set()
    unique: list[Claim] = []
    for claim in claims:
        key = claim.text.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(claim)
    return unique


# ---------------------------------------------------------------------------
# Memory creation
# ---------------------------------------------------------------------------

def _get_existing_claims_for_source(memory_storage: Any, source_id: str) -> set[str]:
    """Return lowercased claim texts already stored for this source_id."""
    try:
        existing = memory_storage.get_by_tag(f"source_id:{source_id}")
        claims: set[str] = set()
        for mem in existing:
            p = mem.payload
            if isinstance(p, dict):
                c = p.get("claim", "")
            else:
                c = str(p)
            claims.add(c.strip().lower())
        return claims
    except Exception:
        return set()


def _create_claim_memories(source: Any, claims: list[Claim]) -> None:
    """Create pointer memories for each claim, referencing the source and chunks."""
    if not claims:
        return

    try:
        from memory.core import memory_core, CreateMemoryData
        from memory.storage import memory_storage
        from memory.index import memory_index

        existing_claims = _get_existing_claims_for_source(memory_storage, source.source_id)

        for claim in claims:
            claim_key = claim.text[:200].strip().lower()
            if claim_key in existing_claims:
                continue

            payload = {
                "type": "study_claim",
                "source_id": source.source_id,
                "claim": claim.text[:200],
                "claim_type": claim.claim_type,
                "chunk_ids": claim.chunk_ids[:MAX_CHUNK_REFS_PER_CLAIM],
                "confidence": claim.confidence,
            }
            if source.doi:
                payload["doi"] = source.doi
            if source.venue:
                payload["venue"] = source.venue

            tags = [
                "study_claim",
                f"source_id:{source.source_id}",
                f"claim_type:{claim.claim_type}",
                "autonomous_research",
            ]
            if getattr(source, "domain_tags", ""):
                for dt in source.domain_tags.split(","):
                    dt = dt.strip()
                    if dt and dt not in tags and not dt.startswith("source_id:"):
                        tags.append(dt)

            from memory.core import canonical_remember
            canonical_remember(CreateMemoryData(
                type="factual_knowledge",
                payload=payload,
                weight=0.45,
                tags=tags,
                decay_rate=0.003,
                provenance="external_source",
                identity_owner="external",
                identity_owner_type="library",
                identity_subject="external",
                identity_subject_type="library",
                identity_scope_key="library:external",
                identity_confidence=0.95,
            ))

    except Exception as exc:
        logger.warning("Failed to create claim memories: %s", exc)
