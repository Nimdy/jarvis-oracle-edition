"""Memory search — keyword search + semantic vector search over memories.

Retrieval pipeline: vector search (candidate generator) -> ranker rerank (learned
or heuristic fallback) -> top-k selection. Only conversation-scoped retrievals are
logged to MemoryRetrievalLog for ranker training and calibration; background/internal
semantic lookups intentionally bypass that closed-loop telemetry.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from consciousness.events import Memory
from memory.storage import memory_storage

logger = logging.getLogger(__name__)

_vector_store = None
_last_retrieval_event_id: str = ""
_baseline_probe_counter: int = 0
_BASELINE_PROBE_INTERVAL: int = 20


def get_last_retrieval_event_id() -> str:
    """Return the event_id from the most recent _hybrid_search call."""
    return _last_retrieval_event_id


def init_vector_store(
    db_path: str = "", model: str = "all-MiniLM-L6-v2",
    dim: int = 384, device: str = "cpu",
):
    """Initialize the vector store for semantic search (call once at startup)."""
    global _vector_store
    from memory.vector_store import VectorStore
    _vector_store = VectorStore(
        db_path=db_path, embedding_model=model,
        embedding_dim=dim, device=device,
    )
    return _vector_store


def get_vector_store():
    return _vector_store


def search(query: str, limit: int = 20, speaker: str = "",
           conversation_id: str = "",
           identity_context: object | None = None,
           referenced_entities: set[str] | None = None) -> list[Memory]:
    """Hybrid search: combine semantic similarity with recency and weight."""
    try:
        from memory.gate import memory_gate
        _reason = "semantic_search" if (_vector_store and _vector_store.available) else "keyword_search"
        with memory_gate.session(_reason, actor="memory.search"):
            if _vector_store and _vector_store.available:
                results = _hybrid_search(query, top_k=limit, speaker=speaker,
                                         conversation_id=conversation_id,
                                         identity_context=identity_context,
                                         referenced_entities=referenced_entities)
                if results:
                    return results

            return keyword_search(query, limit, speaker=speaker,
                                  identity_context=identity_context,
                                  referenced_entities=referenced_entities)
    except Exception:
        if _vector_store and _vector_store.available:
            results = _hybrid_search(query, top_k=limit, speaker=speaker,
                                     conversation_id=conversation_id,
                                     identity_context=identity_context,
                                     referenced_entities=referenced_entities)
            if results:
                return results

        return keyword_search(query, limit, speaker=speaker,
                              identity_context=identity_context,
                              referenced_entities=referenced_entities)


def _compute_candidate_features(
    r: dict, mem: Memory, speaker: str, now: float,
) -> tuple[float, float, float, float, bool]:
    """Compute decomposed features for a single candidate.

    Returns (similarity, recency_score, speaker_boost, provenance_boost, speaker_match).
    """
    from consciousness.events import resolve_provenance_boost

    similarity = r.get("similarity", 0.5)
    age_hours = max(0.01, (now - mem.timestamp) / 3600.0)
    recency_score = 1.0 / (1.0 + age_hours * 0.1)

    speaker_match = False
    speaker_boost = 0.0
    if speaker and speaker != "unknown":
        tags_str = " ".join(mem.tags).lower()
        payload_str = mem.payload if isinstance(mem.payload, str) else str(mem.payload)
        if speaker.lower() in tags_str or speaker.lower() in payload_str.lower():
            speaker_boost = 0.15
            speaker_match = True

    provenance_boost = resolve_provenance_boost(mem)

    return similarity, recency_score, speaker_boost, provenance_boost, speaker_match


def _heuristic_score(
    similarity: float, recency_score: float, weight: float,
    speaker_boost: float, provenance_boost: float,
) -> float:
    """Original hardcoded scoring formula — preserved as feature and fallback."""
    return ((similarity * 0.5) + (recency_score * 0.25) + (weight * 0.15)
            + speaker_boost + provenance_boost + 0.1 * weight)


def _hybrid_search(query: str, top_k: int = 20, speaker: str = "",
                   conversation_id: str = "",
                   identity_context: object | None = None,
                   referenced_entities: set[str] | None = None) -> list[Memory]:
    """Vector search -> identity pre-filter -> ranker rerank -> top-k.

    Identity boundary filtering runs *before* top-k selection (Layer 3
    invariant) so cross-subject memories never consume ranking slots.
    """
    global _last_retrieval_event_id

    if not _vector_store or not _vector_store.available:
        return []

    fetch_multiplier = 4 if identity_context is not None else 2
    raw = _vector_store.search(query, top_k=top_k * fetch_multiplier, min_weight=0.0)
    if not raw:
        return []

    now = time.time()

    from memory.retrieval_log import CandidateRecord
    candidates: list[tuple[CandidateRecord, Memory]] = []
    boundary_blocked: int = 0

    for r in raw:
        mem = memory_storage.get(r["memory_id"])
        if not mem:
            continue

        if identity_context is not None:
            decision = _check_identity_boundary(identity_context, mem, referenced_entities)
            if not decision.allow:
                boundary_blocked += 1
                if decision.requires_audit:
                    try:
                        from identity.audit import identity_audit
                        identity_audit.record_boundary_block(
                            mem.id, decision.reason,
                            getattr(identity_context, "identity_id", ""),
                        )
                    except Exception:
                        pass
                continue
            if decision.requires_explicit_reference:
                try:
                    from identity.audit import identity_audit
                    identity_audit.record_referenced_allow(
                        mem.id,
                        getattr(mem, "identity_subject", ""),
                        getattr(identity_context, "identity_id", ""),
                    )
                except Exception:
                    pass

        sim, recency, spk_boost, prov_boost, spk_match = _compute_candidate_features(
            r, mem, speaker, now,
        )
        hs = _heuristic_score(sim, recency, mem.weight, spk_boost, prov_boost)

        record = CandidateRecord(
            memory_id=mem.id,
            similarity=sim,
            recency_score=recency,
            weight=mem.weight,
            memory_type=mem.type,
            tag_count=len(mem.tags),
            association_count=len(mem.associations),
            priority=mem.priority,
            provenance_boost=prov_boost,
            speaker_match=spk_match,
            heuristic_score=hs,
            selected=False,
        )
        candidates.append((record, mem))

    if boundary_blocked > 0 or identity_context is not None:
        logger.info("Identity pre-filter: %d/%d blocked, %d candidates passed",
                    boundary_blocked, len(raw), len(candidates))

    global _baseline_probe_counter
    _baseline_probe_counter += 1
    force_heuristic = (_baseline_probe_counter % _BASELINE_PROBE_INTERVAL == 0)

    ranker_used = False
    try:
        from memory.ranker import get_memory_ranker
        ranker = get_memory_ranker()
        if ranker and ranker.is_ready() and not force_heuristic:
            feature_batch = []
            for rec, mem in candidates:
                fv = rec.to_feature_vector()
                fv[9] = 1.0 if mem.is_core else 0.0
                feature_batch.append(fv)
            scores = ranker.score_batch(feature_batch)
            scored = list(zip(scores, candidates))
            ranker_used = True
        else:
            scored = [(rec.heuristic_score, (rec, mem)) for rec, mem in candidates]
    except Exception:
        scored = [(rec.heuristic_score, (rec, mem)) for rec, mem in candidates]

    scored.sort(key=lambda x: x[0], reverse=True)
    selected_pairs = scored[:top_k]

    selected_id_set: set[str] = set()
    results: list[Memory] = []
    for _, pair in selected_pairs:
        rec, mem = pair
        selected_id_set.add(rec.memory_id)
        results.append(mem)

    all_records: list[CandidateRecord] = []
    for rec, mem in candidates:
        rec.selected = rec.memory_id in selected_id_set
        all_records.append(rec)

    try:
        from memory.retrieval_log import memory_retrieval_log
        from memory.lifecycle_log import memory_lifecycle_log
        if conversation_id:
            event_id = memory_retrieval_log.log_retrieval(
                conversation_id=conversation_id,
                query_text=query,
                candidates=all_records,
                selected_memory_ids=list(selected_id_set),
                ranker_used=ranker_used,
            )
            _last_retrieval_event_id = event_id

        for rec in all_records:
            if rec.selected:
                memory_lifecycle_log.log_retrieved(rec.memory_id)
    except Exception as exc:
        logger.debug("Retrieval logging failed: %s", exc)

    return results


def _check_identity_boundary(
    identity_context: object,
    memory: Memory,
    referenced_entities: set[str] | None,
) -> Any:
    """Run boundary engine check, returning a BoundaryDecision-like object."""
    try:
        from identity.boundary_engine import identity_boundary_engine, BoundaryDecision
        return identity_boundary_engine.validate_retrieval(
            identity_context, memory, referenced_entities,
        )
    except Exception:
        class _Allow:
            allow = True
            reason = "boundary_unavailable"
            confidence = 0.5
            requires_audit = False
            requires_explicit_reference = False
        return _Allow()


def semantic_search(query: str, top_k: int = 5, speaker: str = "",
                    conversation_id: str = "",
                    identity_context: object | None = None,
                    referenced_entities: set[str] | None = None) -> list[Memory]:
    """Search memories by meaning using vector embeddings with hybrid ranking.

    Identity boundary filtering runs *before* top-k selection so that
    cross-subject memories don't consume ranking slots (Layer 3 invariant).
    """
    if not _vector_store or not _vector_store.available:
        return []

    results = _hybrid_search(query, top_k=top_k, speaker=speaker,
                             conversation_id=conversation_id,
                             identity_context=identity_context,
                             referenced_entities=referenced_entities)
    return results


def keyword_search(query: str, limit: int = 20, speaker: str = "",
                   identity_context: object | None = None,
                   referenced_entities: set[str] | None = None) -> list[Memory]:
    """Return memories whose payload or tags match query (case-insensitive).

    Identity boundary filtering runs before limit selection (Layer 3 invariant).
    """
    query_lower = query.lower()
    scored: list[tuple[float, Memory]] = []

    for m in memory_storage.get_all():
        if identity_context is not None:
            decision = _check_identity_boundary(identity_context, m, referenced_entities)
            if not decision.allow:
                continue

        payload_str = m.payload if isinstance(m.payload, str) else str(m.payload)
        tag_str = " ".join(m.tags)
        combined = f"{payload_str} {tag_str}".lower()

        if query_lower in combined:
            score = m.weight
            if speaker and speaker != "unknown" and speaker.lower() in combined:
                score += 0.15
            scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]


def search_by_tag(tag: str) -> list[Memory]:
    return memory_storage.get_by_tag(tag)


def search_by_type(mem_type: str) -> list[Memory]:
    return memory_storage.get_by_type(mem_type)


def _extract_embedding_text(payload: Any) -> str:
    """Extract clean embedding text from a memory payload.

    Structured dict payloads (research pointers, summaries) carry
    human-readable fields that embed far better than Python repr.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        ptype = payload.get("type", "")
        if ptype == "library_pointer" and payload.get("claim"):
            return payload["claim"]
        if ptype == "research_summary":
            parts = []
            if payload.get("question"):
                parts.append(payload["question"])
            if payload.get("summary"):
                parts.append(payload["summary"])
            if parts:
                return " ".join(parts)
        for field in ("text", "content", "claim", "summary", "description"):
            val = payload.get(field)
            if isinstance(val, str) and len(val) > 10:
                return val
    return str(payload)


def index_memory(memory: Memory) -> None:
    """Index a memory into the vector store (call when memories are created)."""
    if not _vector_store or not _vector_store.available:
        return
    text = _extract_embedding_text(memory.payload)
    _vector_store.add(memory.id, text, memory.type, memory.weight)
