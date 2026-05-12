"""Blue Diamonds — persistent curated knowledge archive.

Stores validated, studied knowledge that survives brain resets. Lives at
``~/.jarvis_blue_diamonds/`` (overridable via ``BLUE_DIAMONDS_PATH`` env var),
completely separate from ``~/.jarvis/``.

Architecture:
  - SQLite DB (archive.db): structured storage for graduated sources + chunks
  - JSONL audit trail (audit.jsonl): append-only log of graduations/rejections/reloads

Quality gates enforced inside ``graduate()``:
  1. Language — English stop-word frequency ≥ 15% of words
  2. Relevance — domain tags or content keywords overlap with Jarvis architecture
  3. Content dedup — rejects if first 200 chars match an existing diamond
  4. Quality floor — 0.55 general, 0.70 for unverified sources
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".jarvis_blue_diamonds"
BLUE_DIAMONDS_PATH = Path(os.environ.get("BLUE_DIAMONDS_PATH", str(_DEFAULT_PATH)))

GRADUATION_MIN_QUALITY = 0.55
GRADUATION_MIN_QUALITY_UNVERIFIED = 0.70
GRADUATION_ELIGIBLE_DEPTHS = frozenset({"abstract", "full_text"})
IMMEDIATE_GRADUATION_MIN_QUALITY = 0.55

# ── Language detection ────────────────────────────────────────────────
_ENGLISH_STOP_WORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "can", "like", "no", "just", "him",
    "know", "some", "could", "them", "other", "than", "then", "now",
    "only", "its", "over", "also", "after", "use", "how", "our",
    "because", "any", "these", "most", "us", "is", "are", "was",
    "were", "been", "being", "has", "had", "does", "did", "may",
    "might", "shall", "should", "must", "used", "using", "such",
    "each", "into", "through", "during", "before", "between",
    "both", "under", "while", "where", "those", "same", "more",
})
_MIN_ENGLISH_RATIO = 0.15


def _is_english(text: str) -> bool:
    """Heuristic English detection via stop-word frequency."""
    words = text.lower().split()[:300]
    if len(words) < 20:
        return True  # too short to judge
    hits = sum(1 for w in words if w.strip(".,;:!?()\"'") in _ENGLISH_STOP_WORDS)
    return (hits / len(words)) >= _MIN_ENGLISH_RATIO


# ── Relevance detection ──────────────────────────────────────────────
_RELEVANT_TAG_DOMAINS = frozenset({
    "artificial_intelligence", "machine_learning", "deep_learning",
    "neural_network", "neural_networks", "reinforcement_learning",
    "transformer", "attention_mechanism", "gru", "lstm", "rnn",
    "consciousness", "self_awareness", "computational_model",
    "cognitive_architecture", "cognitive_science", "metacognition",
    "episodic_memory", "memory_consolidation", "recall", "retrieval",
    "semantic_search", "vector_database", "indexing", "rag",
    "knowledge_representation", "knowledge_graph",
    "natural_language_processing", "nlp", "language_model",
    "grounding", "factual", "speech_recognition", "tts",
    "training", "small_data", "optimization", "policy_gradient",
    "credit_assignment", "counterfactual", "causal_inference",
    "learning", "transfer_learning", "distillation",
    "deep_pattern_matching", "sequential", "decision_making",
    "computer_vision", "object_detection", "face_recognition",
    "speaker_identification", "emotion_recognition", "multimodal",
    "perception", "identity", "personality", "agency",
    "self_directed_inquiry", "emergence", "existential",
    "self_improvement", "safety", "code_generation", "sandboxing",
    "alignment", "evaluation", "event_driven", "architecture",
    "tick_latency", "real_time",
})

_RELEVANT_CONTENT_KEYWORDS = (
    "neural network", "deep learning", "machine learning",
    "reinforcement learning", "artificial intelligence",
    "consciousness", "self-awareness", "cognitive architecture",
    "episodic memory", "memory consolidation", "memory retrieval",
    "language model", "natural language processing",
    "speech recognition", "text-to-speech", "knowledge graph",
    "semantic search", "attention mechanism", "transformer",
    "embedding", "policy gradient", "reward function",
    "credit assignment", "object detection", "face recognition",
    "emotion recognition", "self-improvement", "code generation",
    "multi-agent", "autonomous agent", "decision making",
    "knowledge representation", "belief network", "reasoning",
    "causal inference", "counterfactual", "prediction model",
    "dialogue system", "retrieval augmented", "vector database",
    "recurrent neural", "convolutional neural",
    "distillation", "transfer learning", "few-shot", "meta-learning",
    "sentiment analysis", "personality model",
)


def _is_relevant_to_jarvis(domain_tags: str, content_text: str) -> bool:
    """Check if content relates to AI, consciousness, memory, or related fields."""
    tags = set(t.strip().lower() for t in domain_tags.split(",") if t.strip())
    non_meta_tags = tags - {"peer_reviewed", "unverified"}
    venue_like = {t for t in non_meta_tags if len(t) > 30}
    semantic_tags = non_meta_tags - venue_like

    if semantic_tags & _RELEVANT_TAG_DOMAINS:
        return True

    lower = content_text.lower()[:2000]
    hits = sum(1 for kw in _RELEVANT_CONTENT_KEYWORDS if kw in lower)
    return hits >= 2


@dataclass
class Diamond:
    diamond_id: str
    source_type: str
    doi: str
    url: str
    title: str
    authors: str
    year: int
    venue: str
    citation_count: int
    content_text: str
    content_depth: str
    quality_score: float
    domain_tags: str
    canonical_domain: str
    provider: str
    concepts: str
    claims: str  # JSON array
    graduated_at: float
    source_provenance: str
    graduation_reason: str


@dataclass
class DiamondChunk:
    chunk_id: str
    diamond_id: str
    text: str
    chunk_type: str
    concepts: str
    offset: int


class BlueDiamondsArchive:
    """Persistent curated knowledge archive that survives brain resets."""

    _instance: BlueDiamondsArchive | None = None
    _lock = threading.Lock()

    def __init__(self, db_path: Path | None = None) -> None:
        self._base_path = db_path or BLUE_DIAMONDS_PATH
        self._db_path = self._base_path / "archive.db"
        self._audit_path = self._base_path / "audit.jsonl"
        self._conn: sqlite3.Connection | None = None
        self._initialized = False
        self._write_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> BlueDiamondsArchive:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = BlueDiamondsArchive()
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()
        self._initialized = True
        logger.info("Blue Diamonds archive initialized at %s", self._base_path)

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS diamonds (
                diamond_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL DEFAULT '',
                doi TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                authors TEXT NOT NULL DEFAULT '',
                year INTEGER NOT NULL DEFAULT 0,
                venue TEXT NOT NULL DEFAULT '',
                citation_count INTEGER NOT NULL DEFAULT 0,
                content_text TEXT NOT NULL DEFAULT '',
                content_depth TEXT NOT NULL DEFAULT '',
                quality_score REAL NOT NULL DEFAULT 0.0,
                domain_tags TEXT NOT NULL DEFAULT '',
                canonical_domain TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                concepts TEXT NOT NULL DEFAULT '',
                claims TEXT NOT NULL DEFAULT '[]',
                graduated_at REAL NOT NULL DEFAULT 0.0,
                source_provenance TEXT NOT NULL DEFAULT '',
                graduation_reason TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS diamond_chunks (
                chunk_id TEXT PRIMARY KEY,
                diamond_id TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                chunk_type TEXT NOT NULL DEFAULT '',
                concepts TEXT NOT NULL DEFAULT '',
                offset_idx INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (diamond_id) REFERENCES diamonds(diamond_id)
            );

            CREATE TABLE IF NOT EXISTS reload_log (
                reload_id INTEGER PRIMARY KEY AUTOINCREMENT,
                reloaded_at REAL NOT NULL,
                diamonds_loaded INTEGER NOT NULL DEFAULT 0,
                chunks_loaded INTEGER NOT NULL DEFAULT 0,
                trigger TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_diamond_chunks_did
                ON diamond_chunks(diamond_id);
            CREATE INDEX IF NOT EXISTS idx_diamonds_depth
                ON diamonds(content_depth);
        """)

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.init()

    # -- core operations -------------------------------------------------------

    def graduate(
        self,
        source: Any,
        chunks: list[Any],
        concepts: list[str] | None = None,
        claims: list[str] | None = None,
        reason: str = "",
    ) -> bool:
        """Archive a validated source + chunks into Blue Diamonds.

        Enforces quality gates before writing:
          1. Dedup by source_id
          2. Higher quality floor for unverified sources
          3. English language check
          4. Relevance to Jarvis architecture
          5. Content dedup (first 200 chars against existing diamonds)

        Returns True if graduated successfully, False if rejected or failed.
        """
        self._ensure_init()
        assert self._conn is not None

        diamond_id = source.source_id
        if self.is_archived(diamond_id):
            return False

        # Gate: higher bar for unverified sources
        source_type = getattr(source, "source_type", "")
        if source_type == "unverified":
            if source.quality_score < GRADUATION_MIN_QUALITY_UNVERIFIED:
                self._audit("rejected", {
                    "diamond_id": diamond_id,
                    "reason": f"unverified_quality:{source.quality_score:.2f}<{GRADUATION_MIN_QUALITY_UNVERIFIED}",
                })
                logger.debug("Blue Diamond rejected (unverified quality): %s", diamond_id)
                return False

        # Gate: English language check
        content = getattr(source, "content_text", "") or ""
        if content and not _is_english(content):
            self._audit("rejected", {
                "diamond_id": diamond_id,
                "reason": "non_english",
                "title": (getattr(source, "title", "") or "")[:80],
            })
            logger.debug("Blue Diamond rejected (non-English): %s", diamond_id)
            return False

        # Gate: relevance to Jarvis architecture
        domain_tags = getattr(source, "domain_tags", "") or ""
        if not _is_relevant_to_jarvis(domain_tags, content):
            self._audit("rejected", {
                "diamond_id": diamond_id,
                "reason": "irrelevant_domain",
                "title": (getattr(source, "title", "") or "")[:80],
                "tags": domain_tags[:200],
            })
            logger.debug("Blue Diamond rejected (irrelevant): %s", diamond_id)
            return False

        # Gate: content dedup against existing archive
        if content and self._has_duplicate_content(content):
            self._audit("rejected", {
                "diamond_id": diamond_id,
                "reason": "duplicate_content",
                "title": (getattr(source, "title", "") or "")[:80],
            })
            logger.debug("Blue Diamond rejected (duplicate content): %s", diamond_id)
            return False

        concepts_str = ",".join(concepts) if concepts else ""
        claims_json = json.dumps(claims or [])
        if not reason:
            reason = f"studied+quality_{source.quality_score:.2f}+{source.content_depth}"

        now = time.time()
        try:
            with self._write_lock:
                self._conn.execute("""
                    INSERT INTO diamonds (
                        diamond_id, source_type, doi, url, title, authors,
                        year, venue, citation_count, content_text, content_depth,
                        quality_score, domain_tags, canonical_domain, provider,
                        concepts, claims, graduated_at, source_provenance,
                        graduation_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    diamond_id, source.source_type, source.doi, source.url,
                    source.title, source.authors, source.year, source.venue,
                    source.citation_count, source.content_text, source.content_depth,
                    source.quality_score, source.domain_tags,
                    getattr(source, "canonical_domain", ""),
                    source.provider, concepts_str, claims_json, now,
                    getattr(source, "license_flags", ""),
                    reason,
                ))

                for chunk in chunks:
                    chunk_concepts = ",".join(chunk.concepts) if hasattr(chunk, "concepts") and chunk.concepts else ""
                    self._conn.execute("""
                        INSERT OR IGNORE INTO diamond_chunks
                        (chunk_id, diamond_id, text, chunk_type, concepts, offset_idx)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.chunk_id, diamond_id, chunk.text,
                        getattr(chunk, "chunk_type", ""),
                        chunk_concepts,
                        getattr(chunk, "offset", 0),
                    ))

                self._conn.commit()

            self._audit("graduated", {
                "diamond_id": diamond_id,
                "title": source.title[:120],
                "depth": source.content_depth,
                "quality_score": round(source.quality_score, 3),
                "concepts": (concepts or [])[:10],
                "chunks": len(chunks),
                "reason": reason,
            })

            logger.info(
                "Blue Diamond graduated: %s (depth=%s, quality=%.2f, chunks=%d)",
                diamond_id, source.content_depth, source.quality_score, len(chunks),
            )
            return True

        except Exception as exc:
            logger.warning("Blue Diamond graduation failed for %s: %s", diamond_id, exc)
            return False

    def _has_duplicate_content(self, content_text: str) -> bool:
        """Check if the first 200 chars of content match an existing diamond."""
        assert self._conn is not None
        prefix = content_text.strip()[:200]
        if len(prefix) < 50:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM diamonds WHERE SUBSTR(TRIM(content_text), 1, 200) = ? LIMIT 1",
            (prefix,)
        ).fetchone()
        return row is not None

    def purge_diamond(self, diamond_id: str, reason: str = "manual_purge") -> bool:
        """Remove a diamond and its chunks from the archive."""
        self._ensure_init()
        assert self._conn is not None
        try:
            with self._write_lock:
                title_row = self._conn.execute(
                    "SELECT title FROM diamonds WHERE diamond_id = ?", (diamond_id,)
                ).fetchone()
                if not title_row:
                    return False
                self._conn.execute(
                    "DELETE FROM diamond_chunks WHERE diamond_id = ?", (diamond_id,)
                )
                self._conn.execute(
                    "DELETE FROM diamonds WHERE diamond_id = ?", (diamond_id,)
                )
                self._conn.commit()

            self._audit("purged", {
                "diamond_id": diamond_id,
                "title": (title_row[0] or "")[:120],
                "reason": reason,
            })
            logger.info("Blue Diamond purged: %s (%s)", diamond_id, reason)
            return True
        except Exception as exc:
            logger.warning("Blue Diamond purge failed for %s: %s", diamond_id, exc)
            return False

    def is_archived(self, source_id: str) -> bool:
        self._ensure_init()
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT 1 FROM diamonds WHERE diamond_id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def has_content(self) -> bool:
        self._ensure_init()
        assert self._conn is not None
        row = self._conn.execute("SELECT COUNT(*) FROM diamonds").fetchone()
        return (row[0] or 0) > 0

    def reload_all(self) -> list[tuple[Any, list[Any]]]:
        """Load all archived diamonds for re-ingestion into a fresh library.

        Returns list of (Source-like dict, list of Chunk-like dicts).
        """
        self._ensure_init()
        assert self._conn is not None

        results = []
        rows = self._conn.execute("""
            SELECT diamond_id, source_type, doi, url, title, authors,
                   year, venue, citation_count, content_text, content_depth,
                   quality_score, domain_tags, canonical_domain, provider,
                   source_provenance
            FROM diamonds
            ORDER BY graduated_at ASC
        """).fetchall()

        for row in rows:
            did = row[0]
            source_dict = {
                "source_id": did,
                "source_type": row[1],
                "doi": row[2],
                "url": row[3],
                "title": row[4],
                "authors": row[5],
                "year": row[6],
                "venue": row[7],
                "citation_count": row[8],
                "content_text": row[9],
                "content_depth": row[10],
                "quality_score": row[11],
                "domain_tags": row[12],
                "canonical_domain": row[13],
                "provider": row[14],
                "license_flags": row[15],
            }

            chunk_rows = self._conn.execute("""
                SELECT chunk_id, text, chunk_type, concepts, offset_idx
                FROM diamond_chunks
                WHERE diamond_id = ?
                ORDER BY offset_idx ASC
            """, (did,)).fetchall()

            chunk_dicts = []
            for cr in chunk_rows:
                chunk_dicts.append({
                    "chunk_id": cr[0],
                    "source_id": did,
                    "text": cr[1],
                    "chunk_type": cr[2],
                    "concepts": cr[3].split(",") if cr[3] else [],
                    "offset": cr[4],
                })

            results.append((source_dict, chunk_dicts))

        return results

    def log_reload(self, count: int, chunks_loaded: int = 0, trigger: str = "gestation") -> None:
        self._ensure_init()
        assert self._conn is not None
        with self._write_lock:
            self._conn.execute("""
                INSERT INTO reload_log (reloaded_at, diamonds_loaded, chunks_loaded, trigger)
                VALUES (?, ?, ?, ?)
            """, (time.time(), count, chunks_loaded, trigger))
            self._conn.commit()

        self._audit("reloaded", {
            "count": count,
            "chunks_loaded": chunks_loaded,
            "trigger": trigger,
        })

    def log_rejection(self, source_id: str, reason: str) -> None:
        self._audit("rejected", {
            "source_id": source_id,
            "reason": reason,
        })

    def get_stats(self) -> dict[str, Any]:
        self._ensure_init()
        assert self._conn is not None

        total = self._conn.execute("SELECT COUNT(*) FROM diamonds").fetchone()[0] or 0
        by_depth = {}
        for row in self._conn.execute(
            "SELECT content_depth, COUNT(*) FROM diamonds GROUP BY content_depth"
        ).fetchall():
            by_depth[row[0]] = row[1]

        last_grad = self._conn.execute(
            "SELECT MAX(graduated_at) FROM diamonds"
        ).fetchone()[0]

        reload_rows = self._conn.execute("""
            SELECT reloaded_at, diamonds_loaded, chunks_loaded, trigger
            FROM reload_log ORDER BY reloaded_at DESC LIMIT 5
        """).fetchall()
        reload_history = [
            {"reloaded_at": r[0], "diamonds_loaded": r[1],
             "chunks_loaded": r[2], "trigger": r[3]}
            for r in reload_rows
        ]

        chunk_count = self._conn.execute(
            "SELECT COUNT(*) FROM diamond_chunks"
        ).fetchone()[0] or 0

        return {
            "total_diamonds": total,
            "total_chunks": chunk_count,
            "by_depth": by_depth,
            "last_graduated_at": last_grad,
            "reload_history": reload_history,
            "archive_path": str(self._base_path),
        }

    # -- audit trail -----------------------------------------------------------

    def _audit(self, event: str, data: dict[str, Any]) -> None:
        try:
            entry = {"event": event, "ts": time.time(), **data}
            with open(self._audit_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._initialized = False
