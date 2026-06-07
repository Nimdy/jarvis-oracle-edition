"""Matrix v2 Phase 2 — topic-triggered, domain-scoped recall.

Answers "do you know about X?" from a domain's ISOLATED store only — recall crosses
the domain boundary just when the conversation is about that domain's topic, and
returns only that domain's chunks (no leakage to core or sibling domains). Honesty:
this is "I know about X" (ingested knowledge), never "I can do X" (which is earned).
"""
from __future__ import annotations

import logging

from cognition.capability_domains.store import DomainKnowledgeStore, _tokens

logger = logging.getLogger(__name__)

# how much query/topic overlap counts as "this conversation is about the domain"
_TOPIC_MATCH_MIN = 2


def domain_match_score(domain, query: str) -> int:
    """How many of the query's content tokens hit the domain (name + topic signature)."""
    q = set(_tokens(query))
    if not q:
        return 0
    name_terms = set(_tokens(domain.name))
    score = len(q & name_terms)
    try:
        store = DomainKnowledgeStore(domain.knowledge_db)
        try:
            topic = set(store.topic_terms())
        finally:
            store.close()
        score += len(q & topic)
    except Exception:
        logger.debug("topic_terms failed for %s", domain.domain_id, exc_info=True)
    return score


def best_domain_for(registry, query: str):
    """Pick the domain a query is most about (or None). Domain-scoped routing."""
    best, best_score = None, 0
    for d in registry.list():
        if d.status in ("retired",):
            continue
        s = domain_match_score(d, query)
        if s > best_score:
            best, best_score = d, s
    return best if best_score >= _TOPIC_MATCH_MIN else None


def recall(domain, query: str, k: int = 5) -> list[dict]:
    """Retrieve top-k chunks from the domain's ISOLATED store."""
    store = DomainKnowledgeStore(domain.knowledge_db)
    try:
        return store.search(query, k=k)
    finally:
        store.close()


def recall_answer(registry, query: str, k: int = 3) -> dict | None:
    """Topic-triggered: if the query is about a known domain, return grounded chunks
    from that domain only. Returns None when no domain matches (caller falls back to
    normal handling — no confabulation)."""
    domain = best_domain_for(registry, query)
    if domain is None:
        return None
    chunks = recall(domain, query, k=k)
    if not chunks:
        return None
    return {
        "domain_id": domain.domain_id,
        "domain_name": domain.name,
        "claim_scope": "know_about",   # NOT "can_do" — honesty boundary
        "chunks": chunks,
    }
