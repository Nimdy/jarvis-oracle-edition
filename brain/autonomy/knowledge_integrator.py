"""Knowledge Integrator — stores research findings in the document library.

Two integration paths:
  1. Library + pointer memory: findings become Sources in the document library
     (with chunks and embeddings).  A lightweight pointer memory is created in
     the memory system referencing the source_id + claim.
  2. Evidence feed: compact summaries are fed into existential/philosophical
     systems as new evidence, so future thoughts reference fresh material.

Memory type rules (anti-poisoning):
  - Academic with DOI -> factual_knowledge pointer (highest weight)
  - Academic without DOI -> contextual_insight pointer (unless corroborated)
  - Preprint -> factual_knowledge pointer at reduced weight cap
  - Web -> ALWAYS contextual_insight pointer, NEVER factual_knowledge
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from autonomy.research_intent import ResearchIntent, ResearchResult, ResearchFinding

logger = logging.getLogger(__name__)

MAX_INTEGRATION_HISTORY = 100
MAX_INTENT_SOURCE_TRACK = 500
FINDING_WEIGHT_BASE = 0.3
FINDING_WEIGHT_WEB = 0.2
FINDING_WEIGHT_ACADEMIC = 0.55
FINDING_WEIGHT_ACADEMIC_PREPRINT = 0.4
FINDING_WEIGHT_CODEBASE = 0.5
FINDING_WEIGHT_MEMORY = 0.4
DEDUP_OVERLAP_THRESHOLD = 0.65
PRIOR_KNOWLEDGE_THRESHOLD = 0.6
CONFLICT_OVERLAP_THRESHOLD = 0.3
IMMEDIATE_GRADUATION_MIN_QUALITY = 0.55

_NON_LIBRARY_SOURCE_TYPES: frozenset[str] = frozenset({
    "memory", "introspection", "codebase",
})

# Domain-relevance gate: findings must contain at least one indicator term
# from the AI/ML/cognition core set OR from companion interest terms.
# This prevents tort law, business continuity, etc. from entering the brain
# when the query keywords happen to overlap with non-AI fields.
_AI_DOMAIN_INDICATORS: frozenset[str] = frozenset({
    "neural", "network", "machine learning", "deep learning", "reinforcement",
    "transformer", "attention mechanism", "gradient", "backpropagation",
    "embedding", "latent", "encoder", "decoder", "generative", "discriminative",
    "classification", "regression", "clustering", "supervised", "unsupervised",
    "self-supervised", "pre-training", "fine-tuning", "inference",
    "convolutional", "recurrent", "lstm", "gru", "bert", "gpt", "llm",
    "language model", "computer vision", "nlp", "natural language",
    "artificial intelligence", "cognitive", "autonomous agent",
    "robot", "perception", "reasoning", "knowledge graph",
    "ontology", "semantic", "vector", "retrieval", "memory system",
    "consciousness", "self-awareness", "metacognition", "introspection",
    "policy gradient", "reward", "exploration", "exploitation",
    "multi-agent", "simulation", "loss function",
    "activation", "dropout", "batch normalization", "learning rate",
    "algorithm", "computational", "software architecture",
    "neural architecture", "model training", "dataset", "benchmark",
    "accuracy", "f1 score", "perplexity",
    "tokenizer", "token", "prompt", "context window", "rag",
    "retrieval augmented", "knowledge distillation", "pruning",
    "quantization", "federated learning", "continual learning",
    "meta-learning", "few-shot", "zero-shot", "transfer learning",
    "representation learning", "feature extraction", "latent space",
    "autoencoder", "diffusion", "contrastive learning",
    "speech recognition", "text-to-speech", "speaker identification",
    "emotion detection", "sentiment analysis", "named entity",
    "information extraction", "question answering", "summarization",
    "code generation", "program synthesis",
})

_PAYWALL_MARKERS = (
    "sign in", "log in", "create account", "purchase details",
    "payment options", "order history", "ieee account",
    "access denied", "subscription required", "buy this article",
    "institutional access", "get full access", "rent this article",
    "add to cart", "purchase pdf", "view purchased documents",
    "cookie policy", "cookie preferences", "accept cookies",
    "your privacy choices", "manage preferences",
    "all rights reserved", "terms of use", "privacy policy",
    "javascript is required", "enable javascript", "please enable cookies",
    "we use cookies", "terms of service", "terms and conditions",
)
_PAYWALL_MARKER_THRESHOLD = 2

_ACADEMIC_INDICATORS = (
    "abstract", "introduction", "methodology", "method", "results",
    "discussion", "conclusion", "references", "experiment",
    "we propose", "we present", "our approach", "state-of-the-art",
    "related work", "evaluation", "dataset", "baseline", "ablation",
    "hypothesis", "framework", "empirical", "quantitative",
)

_BOILERPLATE_INDICATORS = (
    "cookie", "privacy", "terms of", "sign in", "log in", "subscribe",
    "newsletter", "copyright", "all rights reserved", "javascript",
    "your browser", "enable cookies", "accept all", "manage preferences",
    "navigation", "menu", "sidebar", "footer", "header",
)

_PDF_RESIDUE_MARKERS = (">>stream", "/Filter", "/FlateDecode", "endobj", "/Type /Page")


def _validate_content_quality(text: str, source_label: str = "") -> tuple[bool, str]:
    """Check that text is actual readable content, not binary garbage.

    Returns (is_valid, rejection_reason). Empty reason on success.
    """
    if not text or len(text.strip()) < 100:
        return False, "too_short"

    sample = text[:2000]
    printable = sum(1 for c in sample if c.isprintable() or c in "\n\r\t")
    ratio = printable / len(sample)
    if ratio < 0.85:
        return False, f"binary_garbage:printable_ratio={ratio:.2f}"

    fffd_count = sample.count("\ufffd")
    if fffd_count > 10:
        return False, f"encoding_garbage:replacement_chars={fffd_count}"

    pdf_hits = sum(1 for m in _PDF_RESIDUE_MARKERS if m in text[:1000])
    if pdf_hits >= 2:
        return False, f"pdf_binary_residue:markers={pdf_hits}"

    return True, ""


def _score_academic_content(text: str) -> tuple[int, int]:
    """Score text for academic vs boilerplate indicators. Returns (academic, boilerplate)."""
    lower = text.lower()
    academic = sum(1 for ind in _ACADEMIC_INDICATORS if ind in lower)
    boilerplate = sum(1 for ind in _BOILERPLATE_INDICATORS if ind in lower)
    return academic, boilerplate


def _has_academic_substance(text: str) -> bool:
    """Check whether text contains meaningful academic content.

    Returns False when the text has zero academic indicators — even if it's
    not a paywall page, content with no methodological, experimental, or
    theoretical language is unlikely to be a useful research source.
    """
    academic, _ = _score_academic_content(text)
    if academic >= 2:
        return True

    words = text.lower().split()
    if len(words) < 80:
        return False

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    substantive = sum(1 for p in paragraphs if len(p.split()) >= 30)
    if substantive >= 3:
        return True

    return False


def _is_paywall_garbage(text: str) -> bool:
    """Detect paywall, login, cookie-consent, or boilerplate pages."""
    lower = text.lower()
    hits = sum(1 for marker in _PAYWALL_MARKERS if marker in lower)
    if hits >= _PAYWALL_MARKER_THRESHOLD:
        return True

    words = lower.split()
    if len(words) < 50:
        return True

    academic, boilerplate = _score_academic_content(text)

    if academic == 0 and boilerplate == 0 and len(words) < 200:
        return True

    if academic == 0 and hits >= 1:
        return True

    if boilerplate > academic and boilerplate >= 3:
        return True

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        avg_para_words = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
        if avg_para_words < 20 and boilerplate >= 2:
            return True

    return False


class KnowledgeIntegrator:
    """Integrates research results into the consciousness + memory systems."""

    def __init__(self) -> None:
        self._engine_ref: Any = None
        self._history: deque[dict[str, Any]] = deque(maxlen=MAX_INTEGRATION_HISTORY)
        self._total_memories_created: int = 0
        self._total_evidence_fed: int = 0
        self._total_skipped_known: int = 0
        self._total_conflicts_detected: int = 0
        self._strict_provenance: bool = False
        self._intent_source_ids: dict[str, list[str]] = {}
        self._intent_source_order: deque[str] = deque(maxlen=MAX_INTENT_SOURCE_TRACK)
        self._ingest_stats: dict[str, int] = {
            "sources_ingested": 0,
            "sources_rejected_binary": 0,
            "sources_rejected_paywall": 0,
            "sources_rejected_boilerplate": 0,
            "sources_rejected_title_only": 0,
            "sources_rejected_domain": 0,
            "sources_rejected_low_relevance": 0,
            "sources_full_text": 0,
            "sources_abstract": 0,
            "sources_tldr": 0,
            "sources_metadata_only": 0,
            "pdf_fetched": 0,
            "pdf_failed": 0,
            "diamonds_graduated": 0,
            "diamonds_rejected": 0,
        }

    def set_engine(self, engine: Any) -> None:
        self._engine_ref = engine

    def set_strict_provenance(self, strict: bool) -> None:
        """Enable strict provenance gating (gestation mode).

        When strict:
        - Requires DOI + venue + year for factual_knowledge
        - Preprints → contextual_insight (not factual_knowledge)
        - Unverified academic → contextual_insight
        """
        self._strict_provenance = strict
        logger.info("Strict provenance mode: %s", strict)

    def integrate(self, intent: ResearchIntent, result: ResearchResult) -> int:
        """Integrate research results. Returns number of memories created."""
        if not result.success or not result.findings:
            return 0

        created = 0

        created += self._store_findings(intent, result)
        self._feed_evidence(intent, result)
        self._record_integration(intent, result, created)

        return created

    def get_recent_learnings(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(self._history)[-limit:]

    def get_source_ids_for_intent(self, intent_id: str, limit: int = 5) -> list[str]:
        """Return recent library source_ids associated with a research intent."""
        if not intent_id:
            return []
        source_ids = self._intent_source_ids.get(intent_id, [])
        return list(source_ids[:max(0, int(limit))]) if limit > 0 else []

    def get_evidence_summary(self, limit: int = 3) -> str:
        """Compact summary of recent learnings for consciousness context injection."""
        recent = list(self._history)[-limit:]
        if not recent:
            return ""
        parts = []
        for entry in recent:
            ts = time.strftime("%H:%M", time.localtime(entry["timestamp"]))
            parts.append(f"[{ts}] {entry['question'][:60]} → {entry['summary'][:80]}")
        return "Recent autonomous research:\n" + "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_memories_created": self._total_memories_created,
            "total_evidence_fed": self._total_evidence_fed,
            "total_skipped_known": self._total_skipped_known,
            "total_conflicts_detected": self._total_conflicts_detected,
            "integration_history_len": len(self._history),
            "intent_source_rows": len(self._intent_source_ids),
            "recent_learnings": self.get_recent_learnings(3),
            "ingestion": dict(self._ingest_stats),
        }

    def get_ingestion_stats(self) -> dict[str, int]:
        """Snapshot of content ingestion quality counters for dashboard."""
        return dict(self._ingest_stats)

    # -- pre-research knowledge check ----------------------------------------

    def check_prior_knowledge(self, question: str) -> dict[str, Any]:
        """Check if we already have strong knowledge on a topic before researching.

        Returns:
            {
                "already_known": bool,
                "existing_memories": list of matching memory summaries,
                "best_confidence": float,
                "recommendation": "skip" | "research" | "verify",
            }
        """
        try:
            from memory.search import semantic_search, keyword_search
            from memory.storage import memory_storage

            existing = []

            sem_results = semantic_search(question, top_k=5)
            for mem in sem_results:
                if mem.type in ("factual_knowledge", "contextual_insight"):
                    existing.append(mem)

            kw_results = keyword_search(question, limit=5)
            seen_ids = {m.id for m in existing}
            for mem in kw_results:
                if mem.id not in seen_ids and mem.type in ("factual_knowledge", "contextual_insight"):
                    existing.append(mem)
                    seen_ids.add(mem.id)

            if not existing:
                return {
                    "already_known": False,
                    "existing_memories": [],
                    "best_confidence": 0.0,
                    "recommendation": "research",
                }

            research_memories = [
                m for m in existing
                if "autonomous_research" in str(m.payload).lower() or any(
                    "autonomous_research" in t for t in getattr(m, "tags", [])
                )
            ]

            best_weight = max(m.weight for m in existing)
            has_peer_reviewed = any(
                "evidence:peer_reviewed" in getattr(m, "tags", [])
                for m in research_memories
            )

            q_words = self._extract_keywords(question)
            high_relevance = []
            for mem in existing:
                m_words = self._extract_keywords(str(mem.payload))
                if q_words and m_words:
                    overlap = len(q_words & m_words) / len(q_words)
                    if overlap >= PRIOR_KNOWLEDGE_THRESHOLD:
                        high_relevance.append(mem)

            summaries = [
                {
                    "id": m.id,
                    "type": m.type,
                    "weight": round(m.weight, 2),
                    "snippet": str(m.payload)[:120],
                    "has_provenance": "autonomous_research" in str(m.payload).lower(),
                }
                for m in high_relevance[:3]
            ]

            import hashlib
            _sig_ids = sorted(s["id"] for s in summaries)
            match_signature = hashlib.md5("|".join(_sig_ids).encode()).hexdigest()[:12]

            if has_peer_reviewed and best_weight >= 0.6 and len(high_relevance) >= 2:
                self._total_skipped_known += 1
                return {
                    "already_known": True,
                    "existing_memories": summaries,
                    "best_confidence": best_weight,
                    "recommendation": "skip",
                    "match_signature": match_signature,
                }

            age_hours = min(
                (time.time() - getattr(m, "created_at", time.time())) / 3600
                for m in high_relevance
            ) if high_relevance else 999

            if high_relevance and age_hours > 168:
                return {
                    "already_known": True,
                    "existing_memories": summaries,
                    "best_confidence": best_weight,
                    "recommendation": "verify",
                    "reason": f"Knowledge exists but is {age_hours:.0f}h old — verify freshness",
                    "match_signature": match_signature,
                    "age_hours": age_hours,
                }

            if best_weight >= 0.5 and len(high_relevance) >= 1:
                self._total_skipped_known += 1
                return {
                    "already_known": True,
                    "existing_memories": summaries,
                    "best_confidence": best_weight,
                    "recommendation": "skip",
                    "match_signature": match_signature,
                }

            return {
                "already_known": False,
                "existing_memories": summaries,
                "best_confidence": best_weight,
                "recommendation": "research",
                "match_signature": match_signature,
            }

        except Exception as exc:
            logger.warning("Prior knowledge check failed: %s", exc)
            return {
                "already_known": False,
                "existing_memories": [],
                "best_confidence": 0.0,
                "recommendation": "research",
            }

    # -- conflict detection --------------------------------------------------

    def detect_conflicts(
        self, question: str, new_findings: list[ResearchFinding],
    ) -> list[dict[str, Any]]:
        """Compare new findings against existing memories on the same topic.

        Returns a list of detected conflicts with both sides for logging/resolution.
        """
        conflicts: list[dict[str, Any]] = []
        try:
            from memory.search import semantic_search

            existing = semantic_search(question, top_k=5)
            if not existing:
                return conflicts

            research_existing = [
                m for m in existing
                if m.type in ("factual_knowledge", "contextual_insight")
            ]
            if not research_existing:
                return conflicts

            existing_keywords = set()
            for mem in research_existing:
                existing_keywords |= self._extract_keywords(str(mem.payload))

            for finding in new_findings:
                finding_keywords = self._extract_keywords(finding.content)
                if not finding_keywords or not existing_keywords:
                    continue

                topic_overlap = len(finding_keywords & existing_keywords) / max(
                    len(finding_keywords), 1
                )
                if topic_overlap < CONFLICT_OVERLAP_THRESHOLD:
                    continue

                for mem in research_existing:
                    mem_keywords = self._extract_keywords(str(mem.payload))
                    shared = finding_keywords & mem_keywords
                    if len(shared) < 3:
                        continue

                    finding_unique = finding_keywords - mem_keywords
                    mem_unique = mem_keywords - finding_keywords

                    if len(finding_unique) > 5 and len(mem_unique) > 5:
                        new_confidence = finding.confidence
                        old_confidence = mem.weight
                        is_upgrade = new_confidence > old_confidence + 0.1

                        conflicts.append({
                            "topic_overlap": round(topic_overlap, 2),
                            "shared_concepts": list(shared)[:5],
                            "new_unique": list(finding_unique)[:5],
                            "existing_unique": list(mem_unique)[:5],
                            "new_confidence": round(new_confidence, 2),
                            "existing_confidence": round(old_confidence, 2),
                            "existing_memory_id": mem.id,
                            "is_upgrade": is_upgrade,
                            "new_snippet": finding.content[:100],
                            "existing_snippet": str(mem.payload)[:100],
                        })
                        self._total_conflicts_detected += 1

        except Exception as exc:
            logger.warning("Conflict detection failed: %s", exc)

        return conflicts

    def apply_upgrades(self, conflicts: list[dict[str, Any]]) -> int:
        """For conflicts flagged as upgrades, accelerate decay on the old memory.

        This ensures superseded knowledge fades faster while the new, higher-quality
        finding takes precedence in retrieval. Returns count of memories updated.
        """
        updated = 0
        try:
            from memory.storage import memory_storage
            for conflict in conflicts:
                if not conflict.get("is_upgrade"):
                    continue
                mem_id = conflict.get("existing_memory_id")
                if not mem_id:
                    continue
                mem = memory_storage.get(mem_id)
                if not mem:
                    continue
                old_weight = mem.weight
                old_decay = mem.decay_rate
                if memory_storage.downweight(mem_id, weight_factor=0.6, decay_rate_factor=3.0):
                    updated += 1
                    new_mem = memory_storage.get(mem_id)
                    logger.info(
                        "Superseded memory %s (weight %.2f→%.2f, decay %.3f→%.3f): %s",
                        mem_id, old_weight,
                        new_mem.weight if new_mem else old_weight * 0.6,
                        old_decay,
                        new_mem.decay_rate if new_mem else min(0.1, old_decay * 3.0),
                        conflict["existing_snippet"][:60],
                    )
        except Exception as exc:
            logger.warning("Failed to apply knowledge upgrades: %s", exc)
        return updated

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "into", "through", "during",
            "and", "but", "if", "or", "because", "while", "what", "which",
            "who", "this", "that", "these", "those", "it", "its", "they",
            "autonomous", "research", "finding", "question", "summary",
        }
        words = {
            w.lower().strip("?.,!:;|[]()\"'")
            for w in text.split()
        }
        return {w for w in words if len(w) > 2 and w not in stop_words}

    # -- memory write path ---------------------------------------------------

    def _store_findings(self, intent: ResearchIntent, result: ResearchResult) -> int:
        """Store findings in the document library and create pointer memories.

        Each finding becomes a Source (with chunks + embeddings) in library.db.
        A lightweight pointer memory is created in the memory system referencing
        the source_id + a one-sentence claim derived from the finding.

        Anti-poisoning rules still apply to memory type classification.
        """
        if not self._engine_ref:
            logger.debug("No engine ref, skipping memory storage")
            return 0

        created = 0
        tool = result.tool_used

        best_findings = sorted(result.findings, key=lambda f: f.confidence, reverse=True)[:5]

        for idx, finding in enumerate(best_findings):
            if finding.source_type in _NON_LIBRARY_SOURCE_TYPES:
                logger.debug(
                    "Skipping non-library source_type '%s': %.60s...",
                    finding.source_type, finding.content,
                )
                continue

            relevance = self._compute_relevance(intent.question, finding.content)
            if relevance < 0.15:
                logger.debug("Dropping irrelevant finding (%.2f): %.60s...", relevance, finding.content)
                self._ingest_stats["sources_rejected_low_relevance"] += 1
                continue

            if not self._has_domain_relevance(finding.content, intent):
                logger.info(
                    "Domain gate rejected finding (no AI/companion indicators): %.80s...",
                    finding.content,
                )
                self._ingest_stats["sources_rejected_domain"] += 1
                continue

            source_id = self._ingest_to_library(finding, tool, intent.tag_cluster)
            if not source_id:
                continue
            self._track_intent_source_id(intent.id, source_id)

            claim = self._extract_claim(intent.question, finding)
            pointer_payload = self._build_pointer_payload(
                source_id, claim, finding,
                intent=intent, finding_idx=idx, tool=tool,
            )

            if self._is_duplicate_pointer(source_id):
                logger.debug("Skipping duplicate source pointer: %s", source_id)
                continue

            tags = self._build_tags(intent, finding, tool)
            tags.append(f"source_id:{source_id}")
            mem_type, weight = self._classify_memory(finding, tool)
            if relevance < 0.25:
                mem_type = "contextual_insight"
                weight = min(weight, 0.2)

            try:
                from memory.core import CreateMemoryData
                mem = self._engine_ref.remember(CreateMemoryData(
                    type=mem_type,
                    payload=pointer_payload,
                    weight=weight,
                    tags=tags,
                    decay_rate=0.005,
                    provenance="external_source",
                ))
                if mem:
                    created += 1
                    self._total_memories_created += 1
                    try:
                        from autonomy.source_ledger import get_source_ledger
                        get_source_ledger().record_source(
                            source_id=source_id,
                            intent_id=intent.id,
                            trigger_deficit=intent.source_event.split(":")[0] if intent.source_event else "",
                            memories_created=1,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Failed to create pointer memory: %s", exc)

        if result.summary and created > 0:
            try:
                from memory.core import CreateMemoryData
                summary_payload = {
                    "type": "research_summary",
                    "question": intent.question[:120],
                    "tool": tool,
                    "finding_count": len(result.findings),
                    "summary": result.summary[:250],
                }
                summary_tags = ["autonomous_research", "summary", intent.source_event.split(":")[0]]
                summary_tags.extend(intent.tag_cluster[:5])
                self._engine_ref.remember(CreateMemoryData(
                    type="contextual_insight",
                    payload=summary_payload,
                    weight=0.35,
                    tags=summary_tags,
                    provenance="external_source",
                ))
            except Exception as exc:
                logger.warning("Failed to create research summary memory: %s", exc)

        return created

    def _track_intent_source_id(self, intent_id: str, source_id: str) -> None:
        if not intent_id or not source_id:
            return
        if intent_id not in self._intent_source_ids:
            if len(self._intent_source_order) >= MAX_INTENT_SOURCE_TRACK:
                oldest = self._intent_source_order.popleft()
                self._intent_source_ids.pop(oldest, None)
            self._intent_source_order.append(intent_id)
            self._intent_source_ids[intent_id] = []

        row = self._intent_source_ids[intent_id]
        if source_id in row:
            return
        row.append(source_id)
        if len(row) > 10:
            del row[10:]

    @staticmethod
    def _infer_content_depth(finding: ResearchFinding) -> str:
        """Determine content depth from provenance tags and content length."""
        provenance = finding.provenance or ""
        content_len = len(finding.content) if finding.content else 0

        if "[depth:tldr]" in provenance:
            return "tldr"
        if "[depth:abstract]" in provenance:
            return "abstract"
        if "[depth:title_only]" in provenance:
            return "title_only"

        if content_len >= 800:
            return "abstract"
        if content_len >= 200:
            return "tldr"
        return "title_only"

    @staticmethod
    def _try_fetch_full_text(finding: ResearchFinding, max_chars: int = 5000) -> tuple[str, str]:
        """Attempt to fetch full paper text from open-access URL or DOI.

        Uses Content-Type routing (not URL extension) to detect PDFs.
        Returns (content, depth) where depth is 'full_text' on success or
        empty string on failure (caller keeps original content).
        """
        fetch_url = finding.open_access_pdf_url or ""
        if not fetch_url and finding.doi_url:
            fetch_url = finding.doi_url

        if not fetch_url:
            return "", ""

        try:
            from config import BrainConfig
            if not BrainConfig().research.fetch_full_text:
                return "", ""
        except Exception:
            pass

        try:
            from library.ingest import _fetch_url, _validate_url, _extract_pdf_text
        except ImportError:
            return "", ""

        ssrf_err = _validate_url(fetch_url)
        if ssrf_err:
            logger.debug("SSRF blocked full-text fetch for %s: %s", fetch_url[:60], ssrf_err)
            return "", ""

        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(fetch_url, headers={
                "User-Agent": "Jarvis-ResearchBot/1.0 (academic paper fetch)",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "").lower()
                raw = resp.read(2_000_000)

                if "pdf" in content_type or raw[:5] == b"%PDF-":
                    text, pdf_err = _extract_pdf_text(raw, max_chars)
                    if text:
                        valid, reason = _validate_content_quality(text)
                        if not valid:
                            logger.debug("PDF content failed validation: %s", reason)
                            return "", ""
                        logger.info(
                            "Fetched full text from PDF (%d chars): %s",
                            len(text), fetch_url[:80],
                        )
                        return text, "full_text"
                    logger.debug("PDF extraction failed: %s", pdf_err)
                    return "", ""

                if "html" in content_type:
                    from library.ingest import _strip_html
                    text = _strip_html(raw.decode("utf-8", errors="replace"))
                else:
                    text = raw.decode("utf-8", errors="replace")

        except (urllib.error.HTTPError, urllib.error.URLError, Exception) as exc:
            logger.debug("Full-text fetch failed for %s: %s", fetch_url[:60], exc)
            return "", ""

        text = text.strip()
        if len(text) < 500:
            return "", ""

        valid, reason = _validate_content_quality(text)
        if not valid:
            logger.debug("Fetched content failed validation (%s): %s", reason, fetch_url[:60])
            return "", ""

        if _is_paywall_garbage(text):
            logger.debug("Rejected paywall/login page content from %s", fetch_url[:60])
            return "", ""

        academic, boilerplate = _score_academic_content(text)
        if boilerplate > academic and boilerplate >= 3:
            logger.debug("Rejected boilerplate page (acad=%d, boiler=%d): %s",
                         academic, boilerplate, fetch_url[:60])
            return "", ""

        if not _has_academic_substance(text):
            logger.debug("Rejected non-academic content (acad=%d): %s",
                         academic, fetch_url[:60])
            return "", ""

        logger.info(
            "Fetched full text from URL (%d chars): %s",
            len(text), fetch_url[:80],
        )
        return text[:max_chars], "full_text"

    def _ingest_to_library(self, finding: ResearchFinding, tool: str,
                           tag_cluster: tuple[str, ...] = ()) -> str:
        """Create a Source + Chunks in the document library.  Returns source_id or "".

        Attempts to fetch full paper text when an open-access URL is available.
        Computes content_depth and applies a quality floor: sources below
        min_content_chars are stored as metadata_only (valid citation ref)
        but are NOT study-eligible and won't generate concepts/claims.
        """
        try:
            from library.source import Source, source_store, make_source_id
            from library.chunks import chunk_text, chunk_store
            from library.index import library_index

            title = finding.content.split(".")[0].strip()[:120] if finding.content else ""
            if title.startswith("{") or title.startswith("Current mode:"):
                logger.debug("Rejecting raw-dump title: %.60s", title)
                return ""
            source_id = make_source_id(
                doi=finding.doi, url=finding.url, title=title,
            )

            if source_store.exists(source_id):
                return source_id

            tags_parts: list[str] = []
            if tag_cluster:
                tags_parts.extend(t for t in tag_cluster if t)
            if finding.venue:
                tags_parts.append(finding.venue.lower().replace(" ", "_"))
            if finding.source_type and finding.source_type not in ("url", "web"):
                tags_parts.append(finding.source_type)
            seen: set[str] = set()
            domain_tags = ",".join(
                t for t in tags_parts
                if t and t not in seen and not seen.add(t)  # type: ignore[func-returns-value]
            )[:200]

            try:
                from config import BrainConfig
                min_chars = BrainConfig().research.min_content_chars
                max_chars = BrainConfig().research.max_content_chars
            except Exception:
                min_chars = 200
                max_chars = 5000

            full_text, full_depth = self._try_fetch_full_text(finding, max_chars)
            if full_text and full_depth:
                content_text = full_text
                content_depth = full_depth
                license_flags = "open_access_full_text"
                self._ingest_stats["pdf_fetched"] += 1
            else:
                if finding.open_access_pdf_url:
                    self._ingest_stats["pdf_failed"] += 1
                content_text = finding.content[:max_chars]
                content_depth = self._infer_content_depth(finding)
                license_flags = "fair_use_abstract" if finding.doi else "unknown"

            valid, rejection_reason = _validate_content_quality(content_text)
            if not valid:
                content_depth = "metadata_only"
                self._ingest_stats["sources_rejected_binary"] += 1
                logger.info("Content validation failed for %s: %s", title[:40], rejection_reason)

            if content_depth == "full_text":
                academic, boilerplate = _score_academic_content(content_text)
                if boilerplate > academic and boilerplate >= 3:
                    content_depth = "metadata_only"
                    self._ingest_stats["sources_rejected_boilerplate"] += 1
                    logger.info("Boilerplate content rejected (acad=%d boiler=%d): %s",
                                academic, boilerplate, title[:40])
                elif not _has_academic_substance(content_text):
                    content_depth = "metadata_only"
                    self._ingest_stats["sources_rejected_boilerplate"] += 1
                    logger.info("No academic substance in full_text (acad=%d): %s",
                                academic, title[:40])

            if len(content_text) < min_chars:
                content_depth = "metadata_only"

            depth_key = {
                "full_text": "sources_full_text",
                "abstract": "sources_abstract",
                "tldr": "sources_tldr",
                "metadata_only": "sources_metadata_only",
            }.get(content_depth)
            if depth_key:
                self._ingest_stats[depth_key] += 1
            self._ingest_stats["sources_ingested"] += 1

            source = Source(
                source_id=source_id,
                source_type=finding.source_type or ("doi" if finding.doi else "url"),
                retrieved_at=time.time(),
                url=finding.url or "",
                doi=finding.doi or "",
                title=title,
                authors=finding.authors or "",
                year=finding.year or 0,
                venue=finding.venue or "",
                citation_count=finding.citation_count or 0,
                content_text=content_text,
                license_flags=license_flags,
                quality_score=finding.confidence,
                provider=finding.source_provider or tool,
                domain_tags=domain_tags,
                content_depth=content_depth,
            )
            source_store.add(source)

            if content_depth in ("metadata_only", "title_only"):
                logger.debug(
                    "Ingested source %s (metadata_only, %d chars, no chunks): %s",
                    source_id, len(content_text), title[:60],
                )
                return source_id

            chunks = chunk_text(
                content_text, source_id,
                chunk_type="full_text" if content_depth == "full_text" else (
                    "abstract" if finding.doi else ""
                ),
            )
            if chunks:
                chunk_store.add_many(chunks)
                for chunk in chunks:
                    library_index.add_chunk(chunk.chunk_id, source_id, chunk.text)

            if (content_depth == "full_text"
                    and finding.confidence >= IMMEDIATE_GRADUATION_MIN_QUALITY
                    and chunks):
                self._try_graduate_to_blue_diamonds(source, chunks)

            logger.debug(
                "Ingested source %s (%d chunks, depth=%s): %s",
                source_id, len(chunks), content_depth, title[:60],
            )
            return source_id

        except Exception as exc:
            logger.warning("Library ingestion failed: %s", exc)
            return ""

    @staticmethod
    def _extract_claim(question: str, finding: ResearchFinding) -> str:
        """Derive a one-sentence claim from the finding for the pointer memory."""
        content_first_sentence = finding.content.split(".")[0].strip()
        if len(content_first_sentence) > 20:
            return f"{content_first_sentence[:300]} (re: {question[:100]})"
        return f"Research finding on: {question[:150]}"

    @staticmethod
    def _build_pointer_payload(
        source_id: str, claim: str, finding: ResearchFinding,
        intent: Any | None = None, finding_idx: int = 0, tool: str = "",
    ) -> dict[str, Any]:
        """Build a structured pointer payload instead of a flat text blob."""
        payload: dict[str, Any] = {
            "type": "library_pointer",
            "source_id": source_id,
            "claim": claim,
        }
        if finding.doi:
            payload["doi"] = finding.doi
        if finding.venue:
            payload["venue"] = finding.venue
        if finding.year:
            payload["year"] = finding.year
        if finding.citation_count:
            payload["citations"] = finding.citation_count
        if intent is not None:
            payload["source_lineage"] = {
                "intent_id": getattr(intent, "id", ""),
                "source_id": source_id,
                "finding_index": finding_idx,
                "tool_used": tool,
                "trigger_event": getattr(intent, "source_event", ""),
                "tag_cluster": list(getattr(intent, "tag_cluster", []))[:5],
            }
        return payload

    def _classify_memory(self, finding: ResearchFinding, tool: str) -> tuple[str, float]:
        """Determine memory type and weight based on provenance.

        When strict_provenance is enabled (gestation mode):
        - Requires DOI + venue + year for factual_knowledge from academic sources
        - Preprints always become contextual_insight
        - Unverified academic findings become contextual_insight

        Returns (memory_type, weight).
        """
        st = finding.source_type
        strict = self._strict_provenance

        if tool == "web_search" or st == "web":
            weight = min(0.5, FINDING_WEIGHT_WEB * finding.confidence * 2.0)
            return "contextual_insight", weight

        if tool == "academic_search":
            has_full_provenance = bool(finding.doi and finding.venue and finding.year)

            if st == "peer_reviewed" and finding.doi:
                if strict and not has_full_provenance:
                    weight = min(0.5, FINDING_WEIGHT_ACADEMIC * finding.confidence * 1.5)
                    return "contextual_insight", weight
                weight = min(0.8, FINDING_WEIGHT_ACADEMIC * finding.confidence * 2.0)
                return "factual_knowledge", weight

            if st == "preprint":
                if strict:
                    weight = min(0.35, FINDING_WEIGHT_ACADEMIC_PREPRINT * finding.confidence * 1.5)
                    return "contextual_insight", weight
                weight = min(0.5, FINDING_WEIGHT_ACADEMIC_PREPRINT * finding.confidence * 2.0)
                return "factual_knowledge", weight

            if finding.doi:
                if strict and not has_full_provenance:
                    weight = min(0.45, FINDING_WEIGHT_BASE * finding.confidence * 2.0)
                    return "contextual_insight", weight
                weight = min(0.7, FINDING_WEIGHT_ACADEMIC * finding.confidence * 2.0)
                return "factual_knowledge", weight

            weight = min(0.5, FINDING_WEIGHT_BASE * finding.confidence * 2.0)
            return "contextual_insight", weight

        if tool == "codebase":
            weight = min(0.8, FINDING_WEIGHT_CODEBASE * finding.confidence * 2.0)
            return "factual_knowledge", weight
        if tool == "memory":
            weight = min(0.8, FINDING_WEIGHT_MEMORY * finding.confidence * 2.0)
            return "factual_knowledge", weight

        weight = min(0.6, FINDING_WEIGHT_BASE * finding.confidence * 2.0)
        return "contextual_insight", weight

    # -- Blue Diamonds graduation ---------------------------------------------

    def _try_graduate_to_blue_diamonds(
        self, source: Any, chunks: list[Any],
        concepts: list[str] | None = None, claims: list[str] | None = None,
        reason: str = "",
    ) -> bool:
        """Attempt to graduate a source to the Blue Diamonds archive."""
        try:
            if getattr(source, "source_type", "") == "codebase":
                return False

            from library.blue_diamonds import (
                BlueDiamondsArchive, GRADUATION_MIN_QUALITY,
                GRADUATION_ELIGIBLE_DEPTHS,
            )
            archive = BlueDiamondsArchive.get_instance()

            if source.content_depth not in GRADUATION_ELIGIBLE_DEPTHS:
                archive.log_rejection(source.source_id, f"depth:{source.content_depth}")
                self._ingest_stats["diamonds_rejected"] += 1
                return False

            if source.quality_score < GRADUATION_MIN_QUALITY:
                archive.log_rejection(source.source_id,
                                      f"quality:{source.quality_score:.2f}<{GRADUATION_MIN_QUALITY}")
                self._ingest_stats["diamonds_rejected"] += 1
                return False

            if archive.graduate(source, chunks, concepts, claims, reason):
                self._ingest_stats["diamonds_graduated"] += 1
                return True
            return False

        except Exception as exc:
            logger.debug("Blue Diamond graduation attempt failed: %s", exc)
            return False

    # -- evidence feed path --------------------------------------------------

    def _feed_evidence(self, intent: ResearchIntent, result: ResearchResult) -> None:
        """Feed compact summaries to existential/epistemic systems."""
        if not result.summary:
            return

        try:
            from consciousness.events import event_bus, KERNEL_THOUGHT
            event_bus.emit(
                KERNEL_THOUGHT,
                thought_type="research_finding",
                depth="deep",
                text=f"[Autonomous research] {intent.question[:60]}: {result.summary[:120]}",
            )
            self._total_evidence_fed += 1
        except Exception as exc:
            logger.debug("Failed to emit research finding thought: %s", exc)

    # -- deduplication -------------------------------------------------------

    def _is_duplicate_pointer(self, source_id: str) -> bool:
        """Check if a pointer memory for this source_id already exists."""
        try:
            from memory.storage import memory_storage
            tag = f"source_id:{source_id}"
            existing = memory_storage.get_by_tag(tag)
            return len(existing) > 0
        except Exception as exc:
            logger.warning("Pointer dedup check failed: %s", exc)
        return False

    def _is_duplicate_payload(self, payload: str) -> bool:
        """Legacy dedup for old-format string payloads."""
        try:
            from memory.storage import memory_storage
            payload_words = set(payload.lower().split())
            if not payload_words:
                return False
            for mem in memory_storage.get_all():
                if mem.type not in ("factual_knowledge", "contextual_insight"):
                    continue
                mem_str = str(mem.payload).lower()
                if "autonomous_research" not in mem_str and "research summary" not in mem_str:
                    continue
                mem_words = set(mem_str.split())
                if not mem_words:
                    continue
                overlap = len(payload_words & mem_words) / max(len(payload_words), len(mem_words))
                if overlap >= DEDUP_OVERLAP_THRESHOLD:
                    return True
        except Exception as exc:
            logger.warning("Dedup check failed: %s", exc)
        return False

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _compute_relevance(question: str, content: str) -> float:
        """Compute keyword overlap between question and finding content.

        Returns 0.0-1.0 indicating how relevant the finding is to the question.
        Low scores indicate the academic search returned off-topic results.
        """
        stop_words = {
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
        }
        q_words = {w.lower().strip("?.,!") for w in question.split()} - stop_words
        c_words = {w.lower().strip("?.,!") for w in content.split()} - stop_words
        q_words = {w for w in q_words if len(w) > 2}
        c_words = {w for w in c_words if len(w) > 2}
        if not q_words or not c_words:
            return 0.0
        overlap = len(q_words & c_words)
        return overlap / len(q_words)

    def _has_domain_relevance(
        self, content: str, intent: ResearchIntent | None = None,
    ) -> bool:
        """Check that content falls within an accepted knowledge domain.

        Accepted domains:
        - AI/ML/cognition/consciousness core (always)
        - Companion interest terms (post-gestation only)
        - Exact intent tag_cluster terms (always, when intent provided)
        """
        content_lower = content.lower()

        for indicator in _AI_DOMAIN_INDICATORS:
            if indicator in content_lower:
                return True

        if intent and intent.tag_cluster:
            cluster_terms = {t.strip().lower() for t in intent.tag_cluster if len(t.strip()) > 2}
            for term in cluster_terms:
                if term in content_lower:
                    return True

        companion_terms = self._get_companion_domain_terms()
        for term in companion_terms:
            if term in content_lower:
                return True

        return False

    def _get_companion_domain_terms(self) -> set[str]:
        """Derive additional accepted domain terms from companion preferences.

        Only activates after gestation when user_preference memories exist.
        Returns an empty set during gestation, keeping the domain gate
        restricted to AI/core topics.
        """
        try:
            engine = self._engine_ref
            if engine is None:
                return set()

            mode_mgr = getattr(engine, "mode_manager", None)
            if mode_mgr is not None:
                current_mode = getattr(mode_mgr, "current_mode", None)
                if current_mode is not None and str(current_mode).lower() == "gestation":
                    return set()

            storage = getattr(engine, "memory", None) or getattr(engine, "storage", None)
            if storage is None:
                try:
                    from memory.storage import memory_storage
                    storage = memory_storage
                except Exception:
                    return set()

            terms: set[str] = set()
            for mem in storage.get_all():
                if "user_preference" in (getattr(mem, "tags", None) or []):
                    payload = str(getattr(mem, "payload", "")).lower()
                    words = {
                        w.strip(".,!?;:'\"()[]{}") for w in payload.split()
                        if len(w.strip(".,!?;:'\"()[]{}")) > 3
                    }
                    terms.update(words)
            return terms
        except Exception as exc:
            logger.debug("Failed to read companion domain terms: %s", exc)
            return set()

    @staticmethod
    def _build_payload(intent: ResearchIntent, finding: ResearchFinding, result: ResearchResult) -> str:
        parts = [f"[Autonomous research: {result.tool_used}]"]
        parts.append(f"Question: {intent.question[:120]}")
        parts.append(f"Finding: {finding.content[:300]}")
        if finding.doi:
            parts.append(f"DOI: {finding.doi}")
        if finding.authors:
            parts.append(f"Authors: {finding.authors}")
        if finding.venue and finding.year:
            parts.append(f"Venue: {finding.venue} ({finding.year})")
        elif finding.venue:
            parts.append(f"Venue: {finding.venue}")
        elif finding.year:
            parts.append(f"Year: {finding.year}")
        if finding.citation_count > 0:
            parts.append(f"Citations: {finding.citation_count}")
        if not finding.doi and finding.url:
            parts.append(f"Source: {finding.url}")
        elif finding.file_path:
            parts.append(f"Source: {finding.file_path}:{finding.line_range}")
        parts.append(f"Confidence: {finding.confidence:.2f}")
        return " | ".join(parts)

    @staticmethod
    def _build_tags(intent: ResearchIntent, finding: ResearchFinding, tool: str) -> list[str]:
        tags = ["autonomous_research", f"tool:{tool}"]

        st = finding.source_type
        if st == "peer_reviewed":
            tags.append("evidence:peer_reviewed")
        elif st == "preprint":
            tags.append("evidence:preprint")
        elif st == "web":
            tags.append("evidence:web")
        elif st == "unverified":
            tags.append("evidence:unverified")
        elif finding.file_path:
            tags.append("evidence:codebase")

        if finding.doi:
            tags.append(f"doi:{finding.doi}")
        if finding.venue:
            tags.append(f"venue:{finding.venue[:40]}")
        if finding.year:
            tags.append(f"year:{finding.year}")
        if finding.source_provider:
            tags.append(f"provider:{finding.source_provider}")

        if finding.url and not finding.doi:
            tags.append("web_sourced")
        if finding.file_path:
            tags.append("code_sourced")
        tags.extend(intent.tag_cluster[:5])
        event_prefix = intent.source_event.split(":")[0]
        if event_prefix:
            tags.append(f"trigger:{event_prefix}")
        return tags

    def _record_integration(self, intent: ResearchIntent, result: ResearchResult, memories_created: int) -> None:
        self._history.append({
            "intent_id": intent.id,
            "question": intent.question,
            "tool": result.tool_used,
            "findings": len(result.findings),
            "memories_created": memories_created,
            "summary": result.summary[:200],
            "timestamp": time.time(),
        })
