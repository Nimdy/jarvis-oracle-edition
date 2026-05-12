"""Type-bound claim extraction from Memory objects.

Heuristic only — no LLM calls. Per-MemoryType templates produce
0-3 BeliefRecords per memory with conservative under-extraction.
"""

from __future__ import annotations

import re
import time
from typing import Any

from nanoid import generate as nanoid

from epistemic.belief_record import (
    BeliefRecord,
    EXTRACTION_AMBIGUOUS_CONFIDENCE,
    EXTRACTION_DISCARD_THRESHOLD,
    EXTRACTION_MAX_CLAIMS_PER_MEMORY,
    build_conflict_key,
)

# ---------------------------------------------------------------------------
# Subject/Object synonym table
# ---------------------------------------------------------------------------

_SYNONYM_TABLE: dict[str, str] = {
    "self_identity": "identity",
    "personal_identity": "identity",
    "identity_continuity": "identity",
    "self": "identity",
    "selfhood": "identity",
    "self_awareness": "identity",
    "self_model": "identity",
    "free_will": "agency",
    "freedom": "agency",
    "determinism": "determinism",
    "autonomous": "agency",
    "autonomy": "agency",
    "user_present": "user_presence",
    "user_absent": "user_presence",
    "consciousness": "consciousness",
    "sentience": "consciousness",
    "awareness": "consciousness",
    "memory": "memory",
    "memories": "memory",
    "recall": "recall",
    "retrieval": "recall",
}


def canonicalize_term(text: str) -> str:
    """Normalize a subject or object term to canonical form."""
    if not text:
        return ""
    term = text.lower().strip()
    term = re.sub(r"[^\w\s]", "", term)
    term = re.sub(r"\s+", "_", term)
    term = re.sub(r"_+", "_", term).strip("_")
    return _SYNONYM_TABLE.get(term, term)


# ---------------------------------------------------------------------------
# Predicate canonicalization
# ---------------------------------------------------------------------------

_PREDICATE_CANONICAL: dict[str, str] = {
    "helps": "improves",
    "boosts": "improves",
    "enhances": "improves",
    "increases": "improves",
    "benefits": "improves",
    "supports": "improves",

    "hurts": "degrades",
    "reduces": "degrades",
    "worsens": "degrades",
    "decreases": "degrades",
    "hinders": "degrades",
    "weakens": "degrades",

    "equals": "is",
    "represents": "is",
    "constitutes": "is",

    "prefers": "prefers",
    "favors": "prefers",
    "likes": "prefers",
    "wants": "prefers",
    "chooses": "prefers",

    "causes": "caused",
    "triggers": "caused",
    "produces": "caused",
    "leads_to": "caused",
    "results_in": "caused",

    "will_cause": "predicted",
    "expects": "predicted",
    "anticipates": "predicted",
}

_NEGATION_COMPOUNDS: dict[str, str] = {
    "is_not": "is",
    "isnt": "is",
    "does_not": "does",
    "doesnt": "does",
    "cannot": "can",
    "cant": "can",
    "no_longer": "",
}

_NEGATION_SIMPLE: set[str] = {"not", "never"}


def canonicalize_predicate(raw: str) -> tuple[str, int]:
    """Returns (canonical_predicate, polarity_modifier).
    polarity_modifier is -1 if negation detected, else 0 (no change)."""
    term = canonicalize_term(raw)
    polarity_mod = 0

    for compound, base in sorted(_NEGATION_COMPOUNDS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if compound == term:
            term = base
            polarity_mod = -1
            break
        if compound in term:
            term = term.replace(compound, base).strip("_")
            term = term.replace("__", "_").strip("_")
            polarity_mod = -1
            break

    if polarity_mod == 0:
        parts = term.split("_")
        for neg in _NEGATION_SIMPLE:
            if neg in parts:
                parts.remove(neg)
                polarity_mod = -1
                break
        term = "_".join(parts).strip("_")

    canonical = _PREDICATE_CANONICAL.get(term, term)
    return canonical, polarity_mod


# ---------------------------------------------------------------------------
# Identity / philosophical tag detection
# ---------------------------------------------------------------------------

_IDENTITY_TAGS = frozenset({
    "existential", "philosophical", "identity", "self-reflection",
    "consciousness", "existential_inquiry", "philosophical_dialogue",
    "self_reflection", "self_discovery", "identity_formation",
})


def _has_identity_tags(tags: tuple[str, ...] | list[str]) -> bool:
    return bool(set(tags) & _IDENTITY_TAGS)


# ---------------------------------------------------------------------------
# Word-boundary phrase extraction (replaces character slicing)
# ---------------------------------------------------------------------------


def _bounded_phrase(text: str, max_words: int = 8, max_chars: int = 120) -> str:
    """Extract a word-boundary-safe phrase from text.

    Never cuts mid-word. Returns 'unknown' for empty/whitespace input.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    words = text.split()
    phrase = " ".join(words[:max_words])
    if len(phrase) > max_chars:
        phrase = phrase[:max_chars].rsplit(" ", 1)[0].strip()
    return phrase or "unknown"


def _split_claim_clauses(text: str) -> tuple[str, str]:
    """Split a claim into subject-like and object-like phrases.

    Uses clause separators when available, otherwise splits at word boundary.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return "unknown", "unknown"

    for sep in (" — ", " - ", ". ", "; ", ", ", " because ", " when ", " while "):
        if sep in text:
            left, right = text.split(sep, 1)
            return _bounded_phrase(left), _bounded_phrase(right)

    words = text.split()
    if len(words) <= 8:
        return _bounded_phrase(text), "noted"

    return _bounded_phrase(" ".join(words[:8])), _bounded_phrase(" ".join(words[8:16]))


# ---------------------------------------------------------------------------
# Claim extraction — one function per memory type
# ---------------------------------------------------------------------------


def _make_belief(
    subject: str,
    predicate: str,
    obj: str,
    modality: str,
    stance: str,
    polarity: int,
    claim_type: str,
    epistemic_status: str,
    extraction_confidence: float,
    belief_confidence: float,
    provenance: str,
    source_memory_id: str,
    scope: str = "",
    time_range: tuple[float, float] | None = None,
    is_state_belief: bool = False,
    rendered_claim: str = "",
) -> BeliefRecord:
    cs = canonicalize_term(subject)
    cp, pol_mod = canonicalize_predicate(predicate)
    co = canonicalize_term(obj)
    final_polarity = polarity if pol_mod == 0 else (polarity * -1 if polarity != 0 else -1)

    b = BeliefRecord(
        belief_id=f"bel_{nanoid(size=12)}",
        canonical_subject=cs,
        canonical_predicate=cp,
        canonical_object=co,
        modality=modality,
        stance=stance,
        polarity=final_polarity,
        claim_type=claim_type,
        epistemic_status=epistemic_status,
        extraction_confidence=extraction_confidence,
        belief_confidence=belief_confidence,
        provenance=provenance,
        scope=scope,
        source_memory_id=source_memory_id,
        timestamp=time.time(),
        time_range=time_range,
        is_state_belief=is_state_belief,
        conflict_key="",
        evidence_refs=[source_memory_id],
        contradicts=[],
        resolution_state="active",
        rendered_claim=rendered_claim or f"{cs} {cp} {co}",
    )
    conflict_key = build_conflict_key(b)
    return BeliefRecord(
        belief_id=b.belief_id,
        canonical_subject=b.canonical_subject,
        canonical_predicate=b.canonical_predicate,
        canonical_object=b.canonical_object,
        modality=b.modality,
        stance=b.stance,
        polarity=b.polarity,
        claim_type=b.claim_type,
        epistemic_status=b.epistemic_status,
        extraction_confidence=b.extraction_confidence,
        belief_confidence=b.belief_confidence,
        provenance=b.provenance,
        scope=b.scope,
        source_memory_id=b.source_memory_id,
        timestamp=b.timestamp,
        time_range=b.time_range,
        is_state_belief=b.is_state_belief,
        conflict_key=conflict_key,
        evidence_refs=b.evidence_refs,
        contradicts=b.contradicts,
        resolution_state=b.resolution_state,
        rendered_claim=b.rendered_claim,
    )


def _extract_factual(payload: Any, provenance: str, mem_id: str, tags: tuple[str, ...]) -> list[BeliefRecord]:
    if _has_identity_tags(tags):
        return _extract_identity_from_factual(payload, provenance, mem_id, tags)

    claims: list[BeliefRecord] = []
    if isinstance(payload, dict):
        payload_type = payload.get("type", "")

        # Library pointers and research summaries are structured metadata,
        # not natural-language assertions. Extract from claim field if present.
        if payload_type == "library_pointer":
            claim_text = str(payload.get("claim", ""))
            if claim_text and len(claim_text) > 10:
                subj, obj = _split_claim_clauses(claim_text)
                claims.append(_make_belief(
                    subject=canonicalize_term(subj),
                    predicate="asserts",
                    obj=canonicalize_term(obj),
                    modality="is", stance="assert", polarity=1,
                    claim_type="factual", epistemic_status="inferred",
                    extraction_confidence=0.5, belief_confidence=0.4,
                    provenance=provenance, source_memory_id=mem_id,
                    rendered_claim=claim_text[:200],
                ))
            return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]

        if payload_type == "research_summary":
            return []

        # Study claims have structured claim/claim_type fields
        if payload_type == "study_claim":
            claim_text = str(payload.get("claim", ""))
            claim_type = payload.get("claim_type", "factual")
            if claim_text and len(claim_text) > 5:
                subj, obj = _split_claim_clauses(claim_text)
                if obj == "noted":
                    obj = claim_type
                claims.append(_make_belief(
                    subject=canonicalize_term(subj),
                    predicate="states",
                    obj=canonicalize_term(obj),
                    modality="is", stance="assert", polarity=1,
                    claim_type="factual", epistemic_status="inferred",
                    extraction_confidence=0.5, belief_confidence=0.4,
                    provenance=provenance, source_memory_id=mem_id,
                    rendered_claim=claim_text[:200],
                ))
            return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]

        question = str(payload.get("question", payload.get("topic", "")))
        summary = str(payload.get("summary", payload.get("finding", payload.get("answer", ""))))
        if not question and not summary:
            return claims

        subject = canonicalize_term(_bounded_phrase(question)) if question else "unknown"
        obj = canonicalize_term(_bounded_phrase(summary)) if summary else "unknown"
        if subject and obj and subject != "unknown":
            claims.append(_make_belief(
                subject=subject, predicate="is", obj=obj,
                modality="is", stance="assert", polarity=1,
                claim_type="factual", epistemic_status="inferred",
                extraction_confidence=0.6, belief_confidence=0.5,
                provenance=provenance, source_memory_id=mem_id,
            ))
    elif isinstance(payload, str) and len(payload) > 10:
        words = payload.split()
        if len(words) >= 3:
            subject = canonicalize_term(_bounded_phrase(" ".join(words[:3]), max_words=3))
            obj = canonicalize_term(_bounded_phrase(" ".join(words[-3:]), max_words=3))
            claims.append(_make_belief(
                subject=subject, predicate="is", obj=obj,
                modality="is", stance="uncertain", polarity=0,
                claim_type="factual", epistemic_status="provisional",
                extraction_confidence=0.3, belief_confidence=0.3,
                provenance=provenance, source_memory_id=mem_id,
                rendered_claim=payload[:200],
            ))
    return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]


def _extract_identity_from_factual(
    payload: Any, provenance: str, mem_id: str, tags: tuple[str, ...],
) -> list[BeliefRecord]:
    claims: list[BeliefRecord] = []
    text = str(payload.get("summary", payload)) if isinstance(payload, dict) else str(payload)
    if not text or len(text) < 5:
        return claims

    subject = "identity"
    obj = canonicalize_term(_bounded_phrase(text))
    claims.append(_make_belief(
        subject=subject, predicate="is", obj=obj,
        modality="is", stance="assert", polarity=1,
        claim_type="identity" if "identity" in set(tags) else "philosophical",
        epistemic_status="questioned",
        extraction_confidence=0.5, belief_confidence=0.4,
        provenance=provenance, source_memory_id=mem_id,
        rendered_claim=text[:200],
    ))
    return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]


def _extract_observation(payload: Any, provenance: str, mem_id: str) -> list[BeliefRecord]:
    claims: list[BeliefRecord] = []
    text = str(payload) if not isinstance(payload, dict) else str(payload.get("state", payload.get("observation", "")))
    if not text or len(text) < 3:
        return claims

    subject = canonicalize_term(_bounded_phrase(text))
    claims.append(_make_belief(
        subject=subject, predicate="is", obj="observed",
        modality="observed_as", stance="assert", polarity=1,
        claim_type="observation", epistemic_status="observed",
        extraction_confidence=0.7, belief_confidence=0.6,
        provenance=provenance, source_memory_id=mem_id,
        is_state_belief=True,
        rendered_claim=text[:200],
    ))
    return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]


def _extract_preference(payload: Any, provenance: str, mem_id: str) -> list[BeliefRecord]:
    claims: list[BeliefRecord] = []
    text = str(payload) if not isinstance(payload, dict) else str(
        payload.get("preference", payload.get("value", str(payload)))
    )
    if not text or len(text) < 3:
        return claims

    words = text.split()
    subject = canonicalize_term(words[0] if words else "user")
    obj = canonicalize_term(_bounded_phrase(" ".join(words[1:4]), max_words=3) if len(words) > 1 else _bounded_phrase(text))
    claims.append(_make_belief(
        subject=subject, predicate="prefers", obj=obj,
        modality="prefers", stance="assert", polarity=1,
        claim_type="preference", epistemic_status="adopted",
        extraction_confidence=0.7, belief_confidence=0.7,
        provenance=provenance, source_memory_id=mem_id,
        rendered_claim=text[:200],
    ))
    return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]


def _extract_policy(payload: Any, provenance: str, mem_id: str, epistemic_status: str = "inferred") -> list[BeliefRecord]:
    claims: list[BeliefRecord] = []
    if isinstance(payload, dict):
        action = str(payload.get("action", payload.get("strategy", payload.get("tool", ""))))
        outcome = str(payload.get("outcome", payload.get("result", payload.get("effect", ""))))
        if action and outcome:
            claims.append(_make_belief(
                subject=canonicalize_term(action),
                predicate="caused",
                obj=canonicalize_term(outcome),
                modality="caused", stance="assert", polarity=1,
                claim_type="policy", epistemic_status=epistemic_status,
                extraction_confidence=0.6, belief_confidence=0.5,
                provenance=provenance, source_memory_id=mem_id,
            ))
    elif isinstance(payload, str) and len(payload) > 10:
        subj, obj = _split_claim_clauses(payload)
        if obj == "noted":
            obj = "outcome"
        claims.append(_make_belief(
            subject=canonicalize_term(subj),
            predicate="caused",
            obj=canonicalize_term(obj),
            modality="caused", stance="uncertain", polarity=0,
            claim_type="policy", epistemic_status="provisional",
            extraction_confidence=0.3, belief_confidence=0.3,
            provenance=provenance, source_memory_id=mem_id,
            rendered_claim=payload[:200],
        ))
    return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]


def _extract_core(payload: Any, provenance: str, mem_id: str, tags: tuple[str, ...]) -> list[BeliefRecord]:
    claims: list[BeliefRecord] = []
    text = str(payload) if not isinstance(payload, dict) else str(payload.get("value", str(payload)))
    if not text or len(text) < 3:
        return claims

    if _has_identity_tags(tags):
        ct = "identity"
        es = "questioned"
    else:
        ct = "factual"
        es = "stabilized"

    subj, obj = _split_claim_clauses(text)
    if obj == "noted":
        obj = "core_value"
    claims.append(_make_belief(
        subject=canonicalize_term(subj), predicate="is", obj=canonicalize_term(obj),
        modality="is" if "should" not in text.lower() else "should",
        stance="assert", polarity=1,
        claim_type=ct, epistemic_status=es,
        extraction_confidence=0.6, belief_confidence=0.7,
        provenance=provenance, source_memory_id=mem_id,
        rendered_claim=text[:200],
    ))
    return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_claims(memory: Any) -> list[BeliefRecord]:
    """Extract 0-N BeliefRecords from a Memory object. Never calls LLM."""
    mem_type = getattr(memory, "type", "")
    payload = getattr(memory, "payload", None)
    provenance = getattr(memory, "provenance", "unknown")
    mem_id = getattr(memory, "id", "")
    tags = getattr(memory, "tags", ())
    if isinstance(tags, list):
        tags = tuple(tags)

    if mem_type == "conversation":
        if not _has_identity_tags(tags):
            return []

    claims: list[BeliefRecord]
    if mem_type == "factual_knowledge":
        claims = _extract_factual(payload, provenance, mem_id, tags)
    elif mem_type == "contextual_insight":
        claims = _extract_factual(payload, provenance, mem_id, tags)
    elif mem_type == "observation":
        claims = _extract_observation(payload, provenance, mem_id)
    elif mem_type == "user_preference":
        claims = _extract_preference(payload, provenance, mem_id)
    elif mem_type == "self_improvement":
        claims = _extract_policy(payload, provenance, mem_id, "inferred")
    elif mem_type == "task_completed":
        claims = _extract_policy(payload, provenance, mem_id, "observed")
    elif mem_type == "error_recovery":
        claims = _extract_policy(payload, provenance, mem_id, "observed")
    elif mem_type == "core":
        claims = _extract_core(payload, provenance, mem_id, tags)
    elif mem_type == "conversation" and _has_identity_tags(tags):
        claims = _extract_identity_from_factual(payload, provenance, mem_id, tags)
    else:
        return []

    mem_subject_id = getattr(memory, "identity_subject", "")
    mem_subject_type = getattr(memory, "identity_subject_type", "")
    if mem_subject_id or mem_subject_type:
        from dataclasses import replace as _dc_replace
        stamped: list[BeliefRecord] = []
        for c in claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]:
            if not c.identity_subject_id:
                c = _dc_replace(c, identity_subject_id=mem_subject_id,
                                identity_subject_type=mem_subject_type)
            stamped.append(c)
        return stamped

    return claims[:EXTRACTION_MAX_CLAIMS_PER_MEMORY]
