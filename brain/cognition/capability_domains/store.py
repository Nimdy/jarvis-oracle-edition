"""Matrix v2 Phase 2 — a domain's ISOLATED knowledge store.

One sqlite file per Capability Domain (under the domain's ``root_dir``). All of a
domain's ingested chunks live here and nowhere else, so deleting the domain dir is
a clean ablation (the registry already does that). No shared library.db, no core
memory writes — that is the anti-pollution invariant. Retrieval is keyword-scored
(deterministic, no embedding-model dependency for the tracer); embeddings can be
layered on later without changing the isolation contract.
"""
from __future__ import annotations

import re
import sqlite3
import time
import uuid

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in on for is are was were be been it this that with as at by "
    "from i you he she they we my your his her their our its do does did have has had".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


class DomainKnowledgeStore:
    """Isolated per-domain chunk store (sqlite). Deterministic keyword retrieval."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT,
                source_title TEXT,
                text TEXT,
                provenance TEXT,
                ts REAL
            )"""
        )
        self._conn.commit()

    def add_chunks(self, source_id: str, source_title: str, chunks: list[str],
                   provenance: str = "ingested") -> int:
        now = time.time()
        rows = [
            (uuid.uuid4().hex, source_id, source_title, c, provenance, now)
            for c in chunks if c and c.strip()
        ]
        if rows:
            self._conn.executemany(
                "INSERT INTO chunks (chunk_id, source_id, source_title, text, provenance, ts) "
                "VALUES (?,?,?,?,?,?)", rows,
            )
            self._conn.commit()
        return len(rows)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def source_count(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(DISTINCT source_id) FROM chunks").fetchone()[0])

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return up to k chunks ranked by query-term overlap (isolated to this domain)."""
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return []
        scored: list[tuple[float, dict]] = []
        for chunk_id, src, title, text in self._conn.execute(
            "SELECT chunk_id, source_id, source_title, text FROM chunks"
        ):
            toks = _tokens(text)
            if not toks:
                continue
            hits = sum(1 for t in toks if t in q_tokens)
            if hits <= 0:
                continue
            # overlap normalized by chunk length (favours focused chunks)
            score = hits / (1.0 + 0.01 * len(toks))
            scored.append((score, {
                "chunk_id": chunk_id, "source_id": src,
                "source_title": title, "text": text, "score": round(score, 4),
            }))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    def topic_terms(self, top_n: int = 12) -> list[str]:
        """The store's most frequent content terms — the domain's 'topic signature'
        (used to decide when a conversation is ABOUT this domain)."""
        from collections import Counter
        counter: Counter = Counter()
        for (text,) in self._conn.execute("SELECT text FROM chunks"):
            counter.update(_tokens(text))
        return [t for t, _ in counter.most_common(top_n)]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
