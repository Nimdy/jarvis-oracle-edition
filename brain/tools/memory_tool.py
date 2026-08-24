"""Memory tool — search and recall from observation + episodic memory."""

from __future__ import annotations

import re

_last_memory_tool_summary: dict[str, object] = {
    "count": 0,
    "subjects": {},
    "types": {},
    "route_type": "memory_tool_idle",
    "search_scope": "none",
}


_EPISODE_PATTERNS = re.compile(
    r"\b(conversation|episode|discussion|chat|session|talked about)\b", re.I)
_PERSONAL_ACTIVITY_QUERY_RE = re.compile(
    r"\b(?:what\s+did\s+(?:i|we)\s+do|tell\s+me\s+what\s+(?:i|we)\s+did|what\s+(?:i|we)\s+did|"
    r"where\s+did\s+(?:i|we)\s+go|when\s+did\s+(?:i|we)\s+(?:go|do))\b",
    re.I,
)
_SYSTEM_SELF_MEMORY_TYPES = frozenset({
    "observation",
    "contextual_insight",
    "self_improvement",
    "core",
    "error_recovery",
})
_TOKEN_RE = re.compile(r"[a-z0-9']+", re.I)
_TEMPORAL_TOKENS = frozenset({
    "today", "tonight", "yesterday", "tomorrow",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "week", "weekend", "morning", "afternoon", "evening", "night",
})

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "are", "am", "be", "been",
    "do", "did", "does", "have", "has", "had", "can", "could", "will",
    "would", "shall", "should", "may", "might", "must", "about", "above",
    "after", "again", "all", "also", "and", "any", "at", "because", "before",
    "between", "both", "but", "by", "down", "during", "each", "for", "from",
    "get", "got", "her", "here", "him", "his", "how", "if", "in", "into",
    "it", "its", "just", "like", "make", "me", "more", "most", "my", "no",
    "not", "now", "of", "on", "one", "only", "or", "other", "our", "out",
    "over", "own", "same", "she", "so", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "up", "very", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "you", "your",
    "tell", "remember", "recall", "know", "think",
})


def _build_identity_context(speaker: str = "") -> object | None:
    """Best-effort retrieval identity context for explicit memory queries."""
    if not speaker or speaker == "unknown":
        return None
    try:
        from identity.resolver import identity_resolver
        return identity_resolver.resolve_for_memory(
            provenance="conversation",
            speaker=speaker,
        )
    except Exception:
        return None


def _extract_referenced_entities(query: str) -> set[str]:
    """Extract known names mentioned in the query for boundary exceptions."""
    refs: set[str] = set()
    try:
        from identity.resolver import identity_resolver

        for name in identity_resolver.get_known_names():
            if name and re.search(rf"\b{re.escape(name)}\b", query, re.I):
                refs.add(name)
    except Exception:
        return refs
    return refs


# Topical "about X" for MEMORY aboutness only — not identity known-names,
# not Layer 3 boundary. Lived 2026-08-24 12:50: Skyler is a remembered dog,
# so get_known_names() was {david} and the first-sentence cut never ran.
_ABOUT_SUBJECT_RE = re.compile(r"\babout\s+([A-Za-z][A-Za-z'-]*)\b", re.I)
_ABOUT_STOP = _STOP_WORDS | frozenset({
    "something", "anything", "everything", "stuff", "things",
    "myself", "yourself", "himself", "herself", "itself",
})
_SPEAKER_ABOUT = frozenset({"me", "myself"})
_NONPERSONAL_PROVENANCE = frozenset({"external_source", "web_scrap", "library"})
_NONPERSONAL_SUBJECT_TYPES = frozenset({"library", "environment"})
_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _extract_about_subjects(query: str, speaker: str = "") -> set[str]:
    """Subject tokens after 'about' in a recall query.

    'me'/'myself' resolve to the speaker when known — they are not identity
    enrollments. Other pronouns stay empty (KNOW-not-guess). Does not write
    names into identity. Matching later is case-insensitive.
    """
    if not query:
        return set()
    speaker_name = (speaker or "").strip()
    if speaker_name.lower() in ("", "unknown"):
        speaker_name = ""
    out: set[str] = set()
    for match in _ABOUT_SUBJECT_RE.finditer(query):
        token = (match.group(1) or "").strip()
        if len(token) < 2:
            continue
        low = token.lower()
        if low in _SPEAKER_ABOUT:
            if speaker_name:
                out.add(speaker_name)
            continue
        if low in _ABOUT_STOP:
            continue
        out.add(token)
    return out


def get_last_memory_tool_summary() -> dict[str, object]:
    return dict(_last_memory_tool_summary)


def _set_memory_tool_summary(
    *,
    count: int,
    route_type: str,
    search_scope: str,
    types: dict[str, int] | None = None,
) -> None:
    global _last_memory_tool_summary
    _last_memory_tool_summary = {
        "count": count,
        "subjects": {},
        "types": dict(types or {}),
        "route_type": route_type,
        "search_scope": search_scope,
    }


def _extract_type_from_preview(preview: str) -> str:
    m = re.match(r"^\[([^\]]+)\]", preview)
    return m.group(1) if m else "memory"


def _is_personal_activity_query(query: str) -> bool:
    return bool(_PERSONAL_ACTIVITY_QUERY_RE.search(query or ""))


def _tokenize_query(query: str) -> list[str]:
    tokens = _TOKEN_RE.findall((query or "").lower())
    cleaned: list[str] = []
    for token in tokens:
        t = token.strip("'")
        if t:
            cleaned.append(t)
    return cleaned


def _build_keyword_seeds(words: list[str], *, prefer_temporal: bool) -> list[str]:
    if not words:
        return []
    seeds: list[str] = []
    if prefer_temporal:
        for word in words:
            if word in _TEMPORAL_TOKENS:
                seeds.append(word)
                break
    if words:
        longest = max(words, key=len)
        if longest not in seeds:
            seeds.append(longest)
    for word in words:
        if word not in seeds:
            seeds.append(word)
        if len(seeds) >= 3:
            break
    return seeds


def _load_osv_for_continuity() -> dict | None:
    """Read-only OSV snapshot for recall-time continuity checks. None = unreadable."""
    try:
        from cognition.self_view import load_self_view
        return load_self_view()
    except Exception:
        return None


_LEAD_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _payload_lead_text(memory_obj) -> str:
    payload = getattr(memory_obj, "payload", None)
    if isinstance(payload, dict):
        for key in ("response", "summary", "text", "message", "user_message"):
            val = str(payload.get(key) or "").strip()
            if val:
                return val
    return str(payload or "").strip()


def _first_sentence(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    return _LEAD_SENTENCE_SPLIT.split(raw, maxsplit=1)[0].strip()


def _leads_with_referenced_subject(memory_obj, referenced_entities: set[str] | None) -> bool:
    """Whether a memory may be declared as recall for a topical 'about X' query.

    Lived 2026-08-24 12:16: a courtesy closer that namedrops Skyler in the
    tail was spoken as 'here's what I remember about Skyler'. Whole-payload
    contains() would keep that line. Aboutness is the FIRST sentence.

    Empty referenced_entities → True (KNOW-not-guess; do not invent a cut).
    Does not touch fractal recall, HRR, ranker weights, or identity boundary.
    """
    if not referenced_entities:
        return True
    lead = _first_sentence(_payload_lead_text(memory_obj))
    if not lead:
        return False
    low = lead.lower()
    return any(str(ent).lower() in low for ent in referenced_entities if ent)


def _is_self_aboutness(aboutness: set[str] | None, speaker: str) -> bool:
    sp = (speaker or "").strip().lower()
    if sp in ("", "unknown") or not aboutness:
        return False
    return any(str(a).lower() == sp for a in aboutness)


def _competing_proper_names(first_sentence: str, speaker: str) -> set[str]:
    """Proper-name tokens in the first sentence after the opener, excluding speaker.

    Used only for about-me. Does not delete memories. 'Ah, Skyler — your
    border collie' is about Skyler even though the owner is David.
    """
    words = _NAME_TOKEN_RE.findall(first_sentence or "")
    if len(words) < 2:
        return set()
    speaker_l = (speaker or "").strip().lower()
    skip = _ABOUT_STOP | {speaker_l, "jarvis", "user"}
    out: set[str] = set()
    for w in words[1:]:
        if len(w) < 3:
            continue
        if w.lower() in skip:
            continue
        if not w[0].isupper():
            continue
        out.add(w)
    return out


def _is_nonpersonal_knowledge(memory_obj) -> bool:
    """Library / external study is not autobiography of the speaker."""
    prov = str(getattr(memory_obj, "provenance", "") or "").strip().lower()
    if prov in _NONPERSONAL_PROVENANCE:
        return True
    st = str(getattr(memory_obj, "identity_subject_type", "") or "").strip().lower()
    return st in _NONPERSONAL_SUBJECT_TYPES


def _matches_aboutness(
    memory_obj,
    aboutness: set[str] | None,
    speaker: str = "",
) -> bool:
    """Recall-time aboutness, including about-me scope. Never mutates the store."""
    if not _leads_with_referenced_subject(memory_obj, aboutness):
        return False
    if not _is_self_aboutness(aboutness, speaker):
        return True
    lead = _first_sentence(_payload_lead_text(memory_obj))
    if _competing_proper_names(lead, speaker):
        return False
    if _is_nonpersonal_knowledge(memory_obj):
        return False
    return True


def _is_contradicted_wipe_memory(memory_obj, model: dict | None = None) -> bool:
    """True when a conversation memory's stored reply asserts a wipe the OSV refutes.

    The memory stays in the store (never discard). Callers skip injecting it as
    recalled fact (never declare). KNOW-not-guess if the OSV is unreadable.
    """
    payload = getattr(memory_obj, "payload", None)
    if not isinstance(payload, dict):
        return False
    response = str(payload.get("response") or "")
    if not response:
        return False
    try:
        from cognition.self_view.articulate import contradicts_measured_continuity
    except Exception:
        return False
    if model is None:
        model = _load_osv_for_continuity()
    return bool(contradicts_measured_continuity(response, model))


def _is_system_self_memory(memory_obj) -> bool:
    memory_type = str(getattr(memory_obj, "type", "") or "").strip().lower()
    subject = str(getattr(memory_obj, "identity_subject", "") or "").strip().lower()
    subject_type = str(getattr(memory_obj, "identity_subject_type", "") or "").strip().lower()
    owner_type = str(getattr(memory_obj, "identity_owner_type", "") or "").strip().lower()
    tags = {
        str(tag).strip().lower()
        for tag in getattr(memory_obj, "tags", ())
        if str(tag).strip()
    }

    if subject in {"jarvis", "system", "assistant"}:
        return True
    if subject_type in {"self", "system"}:
        return True
    if owner_type in {"self", "system"} and memory_type in _SYSTEM_SELF_MEMORY_TYPES:
        return True
    if "speaker:jarvis" in tags and memory_type in _SYSTEM_SELF_MEMORY_TYPES:
        return True
    return False


def _format_payload_preview(memory_obj, max_len: int = 220) -> str:
    payload = getattr(memory_obj, "payload", "")
    memory_type = str(getattr(memory_obj, "type", "") or "")

    if isinstance(payload, dict):
        if memory_type == "conversation":
            user_msg = str(payload.get("user_message", "") or "").strip()
            assistant_msg = str(payload.get("response", "") or "").strip()
            # Prefer assistant response for recall summaries so we avoid
            # replaying user first-person phrasing that can trigger
            # capability-claim gates ("I'm going to ...") during speech.
            # Exception: an OSV-contradicted wipe/blank-slate claim is not
            # autobiography — do not declare it as "Jarvis recalled".
            if assistant_msg and _is_contradicted_wipe_memory(memory_obj):
                assistant_msg = ""
            if assistant_msg:
                text = f"Jarvis recalled: {assistant_msg}"
            elif user_msg:
                text = f"User said: {user_msg}"
            else:
                text = ", ".join(
                    f"{k}={str(v).strip()}"
                    for k, v in payload.items()
                    if str(v).strip()
                )
        else:
            preferred_keys = ("summary", "message", "text", "note", "insight", "topic", "title")
            text = ""
            for key in preferred_keys:
                val = str(payload.get(key, "") or "").strip()
                if val:
                    text = val
                    break
            if not text:
                text = ", ".join(
                    f"{k}={str(v).strip()}"
                    for k, v in payload.items()
                    if str(v).strip()
                )
    else:
        text = str(payload or "").strip()

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def search_memory(query: str, limit: int = 8, speaker: str = "") -> str:
    """Search memories via semantic search first, keyword fallback second."""
    query_lower = query.lower()
    identity_context = _build_identity_context(speaker)
    referenced_entities = _extract_referenced_entities(query)
    aboutness_entities = _extract_about_subjects(query, speaker=speaker)

    if _EPISODE_PATTERNS.search(query):
        ep_results = _search_episodes(query_lower, limit)
        if ep_results and "No " not in ep_results[:4]:
            episode_count = 0
            match = re.match(r"Found (\d+) episode", ep_results)
            if match:
                episode_count = int(match.group(1))
            _set_memory_tool_summary(
                count=episode_count,
                route_type="episodic_recall",
                search_scope="episode_summaries",
                types={"episode": episode_count} if episode_count > 0 else {},
            )
            return ep_results

    results = _semantic_search(
        query,
        limit,
        speaker=speaker,
        identity_context=identity_context,
        referenced_entities=referenced_entities,
        aboutness_entities=aboutness_entities,
    )

    if len(results) < limit:
        kw_results = _keyword_search(
            query_lower,
            limit - len(results),
            speaker=speaker,
            identity_context=identity_context,
            referenced_entities=referenced_entities,
            aboutness_entities=aboutness_entities,
        )
        seen = {preview for _, preview in results}
        # Keyword is a lexical FALLBACK, scored by memory weight (not query
        # similarity, and sometimes >1.0 for core memories). Re-map it onto the
        # 0..1 relevance scale STRICTLY below the weakest semantic hit so the
        # similarity-ranked semantic results always lead — otherwise a
        # high-weight core/boilerplate memory could outrank a real topical
        # match in the downstream renderer (same weight-as-relevance bug).
        sem_floor = min((s for s, _ in results), default=0.30)
        kw_base = min(sem_floor, 0.30)
        rank = 0
        for _, preview in kw_results:
            if preview not in seen:
                rank += 1
                results.append((max(0.0, kw_base - 0.01 * rank), preview))
                seen.add(preview)

    if not results:
        _set_memory_tool_summary(
            count=0,
            route_type="memory_tool_search",
            search_scope="semantic_keyword",
            types={},
        )
        return f"No memories found for this query."

    # Do NOT globally re-sort: semantic hits already arrive in ranker order
    # (learned relevance) keyed by true query SIMILARITY, and keyword fill is
    # appended after as the lower-priority fallback. Re-sorting by raw score
    # here was part of the recall bug — it mixed similarity (0..1) with the
    # keyword path's weight-based score and let boilerplate float to the top.
    types: dict[str, int] = {}
    for _, preview in results[:limit]:
        t = _extract_type_from_preview(preview)
        types[t] = types.get(t, 0) + 1
    _set_memory_tool_summary(
        count=min(len(results), limit),
        route_type="memory_tool_search",
        search_scope="semantic_keyword",
        types=types,
    )
    lines = [f"Found {len(results)} relevant memory(ies):"]
    for score, preview in results[:limit]:
        lines.append(f"  - (relevance={score:.2f}) {preview}")
    return "\n".join(lines)


def _semantic_search(
    query: str,
    limit: int = 8,
    speaker: str = "",
    identity_context: object | None = None,
    referenced_entities: set[str] | None = None,
    aboutness_entities: set[str] | None = None,
) -> list[tuple[float, str]]:
    """Primary search path using embeddings.

    Returns ``(similarity, preview)`` pairs. The score is the raw vector
    similarity (cosine 0..1) from the ranked pipeline — NOT memory weight.
    Using weight here was the recall bug: it re-sorted topical matches
    beneath high-weight boilerplate ("User's name is David", library chunks),
    so "what do you remember about Skylar" surfaced identity facts instead of
    the dog. Relevance must be query-similarity, not intrinsic importance.
    """
    try:
        from memory.search import semantic_search_scored
        is_personal_activity = _is_personal_activity_query(query)
        osv = _load_osv_for_continuity()
        hits = semantic_search_scored(
            query,
            top_k=limit,
            speaker=speaker,
            identity_context=identity_context,
            referenced_entities=referenced_entities,
        )
        results = []
        for sim, m in hits:
            if is_personal_activity and _is_system_self_memory(m):
                continue
            if _is_contradicted_wipe_memory(m, osv):
                continue
            aboutness = aboutness_entities if aboutness_entities is not None else referenced_entities
            if not _matches_aboutness(m, aboutness, speaker=speaker):
                continue
            payload_str = _format_payload_preview(m)
            results.append((float(sim), f"[{m.type}] {payload_str[:200]}"))
        return results
    except Exception:
        return []


def _keyword_search(
    query_lower: str,
    limit: int = 5,
    speaker: str = "",
    identity_context: object | None = None,
    referenced_entities: set[str] | None = None,
    aboutness_entities: set[str] | None = None,
) -> list[tuple[float, str]]:
    """Fallback keyword search with stop-word filtering."""
    results: list[tuple[float, str]] = []
    is_personal_activity = _is_personal_activity_query(query_lower)
    words = [w for w in _tokenize_query(query_lower) if len(w) > 2 and w not in _STOP_WORDS]
    words = list(dict.fromkeys(words))
    if not words:
        return results

    seed_terms = _build_keyword_seeds(words, prefer_temporal=is_personal_activity)
    hits = []
    seen_ids: set[str] = set()
    try:
        from memory.search import keyword_search
        for term in seed_terms:
            term_hits = keyword_search(
                term,
                limit=max(limit * 4, 12),
                speaker=speaker,
                identity_context=identity_context,
                referenced_entities=referenced_entities,
            )
            for m in term_hits:
                mem_id = str(getattr(m, "id", ""))
                if mem_id and mem_id in seen_ids:
                    continue
                if mem_id:
                    seen_ids.add(mem_id)
                hits.append(m)
    except Exception:
        hits = []

    osv = _load_osv_for_continuity()
    for m in hits:
        if is_personal_activity and _is_system_self_memory(m):
            continue
        if _is_contradicted_wipe_memory(m, osv):
            continue
        aboutness = aboutness_entities if aboutness_entities is not None else referenced_entities
        if not _matches_aboutness(m, aboutness, speaker=speaker):
            continue
        payload_str = _format_payload_preview(m)
        tag_str = " ".join(m.tags)
        combined = f"{payload_str} {tag_str}".lower()

        score = 0.0
        for w in words:
            if w in combined:
                score += m.weight
        if score > 0:
            results.append((score, f"[{m.type}] {payload_str[:200]}"))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:limit]


def _search_episodes(query: str, limit: int = 3) -> str:
    """Search episode summaries."""
    try:
        from memory.episodes import episodic_memory
        if not episodic_memory:
            return "No episodic memory available."
        episodes = episodic_memory.search_episodes(query, limit=limit)
        if not episodes:
            return f"No conversation episodes matching this query."
        lines = [f"Found {len(episodes)} episode(s):"]
        for ep in episodes:
            summary = ep.get("summary", "")[:200]
            ts = ep.get("started", "")
            lines.append(f"  - [{ts}] {summary}")
        return "\n".join(lines)
    except Exception:
        return _fallback_memory_search(query, limit)


def _fallback_memory_search(query: str, limit: int = 5) -> str:
    from memory.storage import memory_storage
    results: list[tuple[float, str]] = []
    for m in memory_storage.get_all():
        if m.type == "episode_summary":
            payload_str = m.payload if isinstance(m.payload, str) else str(m.payload)
            results.append((m.weight, f"[episode] {payload_str[:200]}"))
    results.sort(key=lambda x: x[0], reverse=True)
    if not results:
        return "No episode summaries found."
    lines = [f"Found {len(results)} episode summary(ies):"]
    for w, preview in results[:limit]:
        lines.append(f"  - {preview}")
    return "\n".join(lines)


def get_memory_summary() -> str:
    """Brief summary of current memory state."""
    from memory.storage import memory_storage
    stats = memory_storage.get_stats()
    tag_freq = memory_storage.get_tag_frequency()
    top_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:5]

    parts = [
        f"I have {stats['total']} memories ({stats['core_count']} core).",
        f"Average weight: {stats['avg_weight']:.2f}.",
    ]
    if top_tags:
        parts.append(f"Top themes: {', '.join(t[0] for t in top_tags)}.")

    try:
        from memory.episodes import episodic_memory
        if episodic_memory:
            ep_count = episodic_memory.get_episode_count()
            if ep_count > 0:
                parts.append(f"I have {ep_count} conversation episode(s) recorded.")
    except Exception:
        pass

    _set_memory_tool_summary(
        count=0,
        route_type="memory_summary",
        search_scope="summary_only",
        types={},
    )
    return " ".join(parts)
