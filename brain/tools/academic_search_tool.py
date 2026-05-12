"""Academic Search Tool -- Semantic Scholar + Crossref scholarly retrieval.

Provides poisoning-resistant search by querying curated bibliographic databases
instead of the open web. Results include structured provenance (DOI, venue,
authors, year, citation count) that cannot be gamed via SEO.

Two providers:
  - Semantic Scholar (S2): papers, abstracts, citations, wide coverage
  - Crossref: DOI metadata, journal verification, fills S2 gaps

Search policy router classifies queries into "scholarly" (default) or
"realtime" (weather/news/sports exception) lanes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import aiohttp

logger = logging.getLogger(__name__)

SearchLane = Literal["scholarly", "realtime"]

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_FIELDS = "paperId,title,year,venue,externalIds,citationCount,influentialCitationCount,isOpenAccess"
S2_DETAIL_FIELDS = "abstract,authors,tldr,openAccessPdf"
S2_BATCH_FIELDS = "title,abstract,authors,year,venue,externalIds,citationCount,influentialCitationCount,isOpenAccess,tldr,openAccessPdf"
S2_TIMEOUT = 15
S2_DETAIL_TIMEOUT = 10
S2_BATCH_TIMEOUT = 20
S2_MIN_CITATION_COUNT = 3
S2_DEFAULT_YEAR_RANGE = "2018-"

CROSSREF_BASE = "https://api.crossref.org/works"
CROSSREF_SELECT = "DOI,title,author,published-print,published-online,container-title,is-referenced-by-count,abstract"
CROSSREF_TIMEOUT = 15
CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO", "")

CACHE_FILE = Path("~/.jarvis/academic_search_cache.json").expanduser()
CACHE_TTL_S = 21600.0  # 6 hours
MAX_CACHE_ENTRIES = 200

_PREPRINT_VENUES: frozenset[str] = frozenset({
    "arxiv", "biorxiv", "medrxiv", "ssrn", "preprints",
    "research square", "authorea", "techrxiv", "chemrxiv",
})

_STRONG_REALTIME = frozenset({
    "weather", "forecast", "radar", "hurricane", "storm",
    "traffic", "stock price", "open now",
})

_WEAK_REALTIME = frozenset({
    "latest", "today", "this week", "breaking", "current",
    "right now", "tonight", "yesterday",
})

_REALTIME_DOMAINS = frozenset({
    "news", "score", "scores", "election", "price", "prices",
    "forecast", "sports", "results", "standings", "market",
    "update", "updates",
})


@dataclass
class AcademicResult:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    doi: str = ""
    doi_url: str = ""
    url: str = ""
    citation_count: int = 0
    influential_citation_count: int = 0
    source_provider: str = ""  # semantic_scholar / crossref
    is_peer_reviewed: bool = False
    is_preprint: bool = False
    tldr: str = ""
    is_open_access: bool = False
    open_access_pdf_url: str = ""
    paper_id: str = ""


class _AcademicCache:
    """In-memory + disk cache for academic search results."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if CACHE_FILE.exists():
                self._data = json.loads(CACHE_FILE.read_text())
        except Exception:
            self._data = {}

    def get(self, query: str) -> list[dict] | None:
        self._load()
        key = hashlib.md5(query.lower().encode()).hexdigest()[:16]
        entry = self._data.get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > CACHE_TTL_S:
            del self._data[key]
            return None
        return entry.get("results")

    def put(self, query: str, results: list[dict]) -> None:
        self._load()
        key = hashlib.md5(query.lower().encode()).hexdigest()[:16]
        self._data[key] = {"ts": time.time(), "results": results}
        if len(self._data) > MAX_CACHE_ENTRIES:
            oldest = sorted(self._data.items(), key=lambda x: x[1].get("ts", 0))
            for k, _ in oldest[: len(self._data) - MAX_CACHE_ENTRIES]:
                del self._data[k]
        self._save()

    def _save(self) -> None:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data))
            tmp.replace(CACHE_FILE)
        except Exception:
            logger.debug("Failed to save academic cache", exc_info=True)


_cache = _AcademicCache()


_s2_key_disabled: bool = False
_S2_AUTH_MIN_INTERVAL_S = max(0.0, float(os.environ.get("S2_AUTH_MIN_INTERVAL_S", "1.05")))
_s2_last_auth_request_monotonic: float = 0.0
_s2_auth_rate_lock: asyncio.Lock | None = None


def _get_s2_auth_rate_lock() -> asyncio.Lock:
    global _s2_auth_rate_lock
    if _s2_auth_rate_lock is None:
        _s2_auth_rate_lock = asyncio.Lock()
    return _s2_auth_rate_lock


def _has_s2_auth_header(headers: dict[str, str]) -> bool:
    auth = str(headers.get("Authorization", "")).strip()
    legacy = str(headers.get("x-api-key", "")).strip()
    return bool(auth or legacy)


async def _enforce_s2_auth_rate_limit(headers: dict[str, str]) -> None:
    """Throttle authenticated S2 requests to roughly 1 req/sec.

    Semantic Scholar API keys are cumulative across endpoints, so one bulk query
    followed immediately by a detail/batch query can exceed the key allowance.
    We only throttle when an authenticated header is present.
    """
    if not _has_s2_auth_header(headers):
        return
    global _s2_last_auth_request_monotonic
    lock = _get_s2_auth_rate_lock()
    async with lock:
        now = time.monotonic()
        wait_s = _S2_AUTH_MIN_INTERVAL_S - (now - _s2_last_auth_request_monotonic)
        if wait_s > 0:
            await asyncio.sleep(wait_s)
            now = time.monotonic()
        _s2_last_auth_request_monotonic = now


def _s2_headers() -> dict[str, str]:
    """Build S2 request headers, including API key if configured.

    If a previous request got 403 with the key, stop sending it for the
    rest of the session so requests fall back to unauthenticated access.

    NOTE: Semantic Scholar key auth currently works via Bearer authorization.
    The legacy x-api-key header may be rejected with 403 on some accounts.
    """
    if _s2_key_disabled:
        return {}
    headers: dict[str, str] = {}
    try:
        from config import BrainConfig
        cfg = BrainConfig()
        key = cfg.research.s2_api_key
    except Exception:
        key = os.environ.get("S2_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _disable_s2_key() -> None:
    """Mark the S2 API key as rejected (403) for this session."""
    global _s2_key_disabled
    if not _s2_key_disabled:
        _s2_key_disabled = True
        logger.warning(
            "S2 API key disabled for session (got 403 with authenticated header). "
            "Falling back to unauthenticated.",
        )


async def batch_enrich_results(results: list[AcademicResult]) -> list[AcademicResult]:
    """Fetch missing abstract/TLDR from S2 batch endpoint for multiple papers.

    Uses POST /paper/batch instead of individual detail calls, dramatically
    reducing API load (1 call instead of N).
    """
    needs_enrichment: list[tuple[int, str]] = []
    for i, r in enumerate(results):
        if (not r.abstract or len(r.abstract) < 100) and r.paper_id:
            needs_enrichment.append((i, r.paper_id))

    if not needs_enrichment:
        return results

    paper_ids = [pid for _, pid in needs_enrichment]
    url = f"{S2_BASE}/paper/batch"
    params = {"fields": S2_DETAIL_FIELDS}
    headers = _s2_headers()
    headers["Content-Type"] = "application/json"

    try:
        timeout = aiohttp.ClientTimeout(total=S2_BATCH_TIMEOUT)
        async with aiohttp.ClientSession(headers=headers) as session:
            await _enforce_s2_auth_rate_limit(headers)
            async with session.post(
                url, params=params, json={"ids": paper_ids}, timeout=timeout,
            ) as resp:
                if resp.status == 403 and _has_s2_auth_header(headers):
                    _disable_s2_key()
                    async with aiohttp.ClientSession() as anon_session:
                        async with anon_session.post(
                            url,
                            params=params,
                            json={"ids": paper_ids},
                            timeout=timeout,
                        ) as anon_resp:
                            if anon_resp.status != 200:
                                logger.warning("S2 batch enrich anon retry returned %d", anon_resp.status)
                                return results
                            data = await anon_resp.json()
                elif resp.status != 200:
                    logger.warning("S2 batch enrich returned %d", resp.status)
                    return results
                else:
                    data = await resp.json()
    except Exception as exc:
        logger.debug("S2 batch enrich failed: %s", exc)
        return results

    if not isinstance(data, list):
        return results

    for (idx, _), paper_data in zip(needs_enrichment, data):
        if not paper_data:
            continue
        r = results[idx]
        if not r.abstract:
            r.abstract = (paper_data.get("abstract") or "")[:2000]
        if not r.tldr:
            tldr_obj = paper_data.get("tldr") or {}
            r.tldr = (tldr_obj.get("text") or "")[:500]
        if not r.open_access_pdf_url:
            oa_pdf = paper_data.get("openAccessPdf") or {}
            r.open_access_pdf_url = oa_pdf.get("url") or ""

    return results


async def enrich_result(result: AcademicResult) -> AcademicResult:
    """Fetch missing abstract/TLDR from S2 detail endpoint for a single paper.

    Legacy single-paper fallback. Prefer batch_enrich_results() for multiple papers.
    """
    if result.abstract and len(result.abstract) >= 100:
        return result

    paper_id = result.paper_id
    if not paper_id and result.url and "semanticscholar.org/paper/" in result.url:
        paper_id = result.url.rsplit("/", 1)[-1]
    if not paper_id and result.doi:
        paper_id = f"DOI:{result.doi}"
    if not paper_id:
        return result

    detail_url = f"{S2_BASE}/paper/{paper_id}"
    params = {"fields": S2_DETAIL_FIELDS}
    headers = _s2_headers()

    try:
        timeout = aiohttp.ClientTimeout(total=S2_DETAIL_TIMEOUT)
        async with aiohttp.ClientSession(headers=headers) as session:
            await _enforce_s2_auth_rate_limit(headers)
            async with session.get(detail_url, params=params, timeout=timeout) as resp:
                if resp.status == 403 and _has_s2_auth_header(headers):
                    _disable_s2_key()
                    async with aiohttp.ClientSession() as anon_session:
                        async with anon_session.get(detail_url, params=params, timeout=timeout) as anon_resp:
                            if anon_resp.status != 200:
                                return result
                            data = await anon_resp.json()
                elif resp.status != 200:
                    return result
                else:
                    data = await resp.json()
    except Exception as exc:
        logger.debug("S2 detail fetch failed for %s: %s", paper_id[:30], exc)
        return result

    if not result.abstract:
        result.abstract = (data.get("abstract") or "")[:2000]
    if not result.tldr:
        tldr_obj = data.get("tldr") or {}
        result.tldr = (tldr_obj.get("text") or "")[:500]
    if not result.open_access_pdf_url:
        oa_pdf = data.get("openAccessPdf") or {}
        result.open_access_pdf_url = oa_pdf.get("url") or ""

    return result


def normalize_doi(raw: str) -> str:
    """Normalize a DOI: lowercase, strip URL prefix, trim."""
    if not raw:
        return ""
    d = raw.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def _classify_provenance(venue: str, doi: str) -> tuple[bool, bool]:
    """Return (is_preprint, is_peer_reviewed) using venue heuristic."""
    if not venue and not doi:
        return False, False
    v_lower = venue.lower().strip() if venue else ""
    is_preprint = any(pv in v_lower for pv in _PREPRINT_VENUES)
    is_peer_reviewed = bool(doi) and bool(v_lower) and not is_preprint
    return is_preprint, is_peer_reviewed


_TOP_ML_VENUES: frozenset[str] = frozenset({
    "neurips", "nips", "icml", "iclr", "aaai", "cvpr", "iccv", "eccv",
    "acl", "emnlp", "naacl", "sigir", "kdd", "www", "ijcai", "uai",
    "jmlr", "pami", "nature", "science", "cell",
    "transactions on pattern analysis", "journal of machine learning",
    "artificial intelligence", "neural computation",
    "ieee transactions on neural networks",
})


def confidence_for_result(result: AcademicResult) -> float:
    """Compute confidence score based on provenance + citation impact + recency.

    Scoring formula uses three dimensions:
      1. Provenance base (peer-reviewed > preprint > DOI-only > unknown)
      2. Citation impact (influentialCitationCount weighted 2x over raw count)
      3. Recency bonus (last 3 years get +0.05, last 5 years get +0.02)

    Top ML/AI venues get elevated base scores.
    """
    venue_lower = (result.venue or "").lower()
    is_top_venue = any(v in venue_lower for v in _TOP_ML_VENUES)

    influential = result.influential_citation_count or 0
    regular = result.citation_count or 0
    impact = influential * 2.0 + regular
    cite_bonus = min(0.15, impact * 0.0005)

    current_year = time.localtime().tm_year
    if result.year and result.year >= current_year - 2:
        recency_bonus = 0.05
    elif result.year and result.year >= current_year - 5:
        recency_bonus = 0.02
    else:
        recency_bonus = 0.0

    if result.is_peer_reviewed:
        if is_top_venue:
            base = 0.80
        elif regular > 0 or influential > 0:
            base = 0.72
        else:
            base = 0.68
    elif result.is_preprint:
        if is_top_venue and regular > 50:
            base = 0.55
        elif regular > 0 or influential > 0:
            base = 0.45
        else:
            base = 0.40
    elif result.doi:
        base = 0.35
    else:
        base = 0.25

    return min(0.95, base + cite_bonus + recency_bonus)


def classify_search_lane(query: str) -> SearchLane:
    """Classify a query into scholarly (default) or realtime lane.

    Realtime requires either a strong signal keyword or a weak signal
    keyword paired with a realtime domain keyword. This prevents
    "latest paper on X" from escaping to DDG.
    """
    q = query.lower()

    for kw in _STRONG_REALTIME:
        if kw in q:
            return "realtime"

    has_weak = any(kw in q for kw in _WEAK_REALTIME)
    if has_weak:
        has_domain = any(kw in q for kw in _REALTIME_DOMAINS)
        if has_domain:
            return "realtime"

    return "scholarly"


_QUESTION_STRIP_PREFIXES = [
    "what techniques could", "what techniques", "what are the best practices for",
    "what are best practices for", "what are the most effective",
    "how can", "how do", "how does", "how should", "what patterns cause",
    "what causes", "what frameworks exist for", "what are effective",
    "what optimization would", "what do current theories say about",
    "what is the relationship between", "when should",
]

_QUERY_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "and", "but", "if", "or", "because", "while", "what", "which", "who",
    "whom", "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "you", "your", "he", "she", "his", "her",
    "how", "when", "where", "why", "not", "no", "so", "than", "too",
    "very", "just", "about", "also", "between", "after", "before",
    "such", "each", "some", "any", "all", "most", "more", "other",
    "there", "then", "here", "up", "out", "over",
})


def _clean_query_for_search(query: str) -> str:
    """Convert natural-language questions into keyword queries for S2.

    S2's bulk search works best with concise keyword queries rather than
    long natural-language phrases.  Steps:
      1. Strip question prefix ("How do", "What are the best practices for")
      2. Extract content words, dropping stop words
      3. Keep hyphenated terms intact (e.g. "self-modification")
      4. Cap at 10 keywords to stay in S2's sweet spot
    """
    q = query.strip().rstrip("?").strip()
    lower = q.lower()
    for prefix in _QUESTION_STRIP_PREFIXES:
        if lower.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    words = q.split()
    keywords: list[str] = []
    for w in words:
        cleaned = w.strip(".,;:!?()[]\"'")
        if not cleaned:
            continue
        if "-" in cleaned and len(cleaned) > 3:
            keywords.append(cleaned)
            continue
        if cleaned.lower() in _QUERY_STOP_WORDS:
            continue
        if len(cleaned) <= 2:
            continue
        keywords.append(cleaned)

    if not keywords:
        return q[:200]

    return " ".join(keywords[:10])


def _parse_s2_paper(paper: dict[str, Any]) -> AcademicResult:
    """Parse a single S2 API paper object into an AcademicResult."""
    doi_raw = ""
    ext_ids = paper.get("externalIds") or {}
    if ext_ids.get("DOI"):
        doi_raw = ext_ids["DOI"]

    doi = normalize_doi(doi_raw)
    venue = paper.get("venue") or ""
    is_preprint, is_peer_reviewed = _classify_provenance(venue, doi)

    paper_id = paper.get("paperId") or ""
    s2_url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""

    author_names = []
    for a in (paper.get("authors") or []):
        name = a.get("name", "")
        if name:
            author_names.append(name)

    tldr_obj = paper.get("tldr") or {}
    tldr_text = (tldr_obj.get("text") or "")[:500]

    oa_pdf = paper.get("openAccessPdf") or {}
    oa_pdf_url = oa_pdf.get("url") or ""

    return AcademicResult(
        title=paper.get("title") or "",
        abstract=(paper.get("abstract") or "")[:2000],
        authors=author_names[:10],
        year=paper.get("year") or 0,
        venue=venue,
        doi=doi,
        doi_url=f"https://doi.org/{doi}" if doi else "",
        url=s2_url,
        citation_count=paper.get("citationCount") or 0,
        influential_citation_count=paper.get("influentialCitationCount") or 0,
        source_provider="semantic_scholar",
        is_peer_reviewed=is_peer_reviewed,
        is_preprint=is_preprint,
        tldr=tldr_text,
        is_open_access=bool(paper.get("isOpenAccess")),
        open_access_pdf_url=oa_pdf_url,
        paper_id=paper_id,
    )


async def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    min_citations: int | None = None,
    year_range: str | None = None,
) -> list[AcademicResult]:
    """Search Semantic Scholar via bulk search endpoint.

    Uses /paper/search/bulk for better filtering (minCitationCount, year,
    sort) and lower server-side cost. Then enriches top results via
    /paper/batch for abstract/TLDR data not returned in the lightweight
    discovery phase.

    Args:
        query: Search terms.
        max_results: Number of results to return.
        min_citations: Minimum citation count filter (default: S2_MIN_CITATION_COUNT).
        year_range: Year filter, e.g. "2018-" or "2020-2024" (default: S2_DEFAULT_YEAR_RANGE).
    """
    results: list[AcademicResult] = []
    url = f"{S2_BASE}/paper/search/bulk"
    cleaned_query = _clean_query_for_search(query)

    effective_min_cites = min_citations if min_citations is not None else S2_MIN_CITATION_COUNT
    effective_year = year_range or S2_DEFAULT_YEAR_RANGE

    params: dict[str, str] = {
        "query": cleaned_query,
        "fields": S2_SEARCH_FIELDS,
        "fieldsOfStudy": "Computer Science",
        "sort": "citationCount:desc",
    }
    if effective_min_cites > 0:
        params["minCitationCount"] = str(effective_min_cites)
    if effective_year:
        params["year"] = effective_year

    headers = _s2_headers()

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            await _enforce_s2_auth_rate_limit(headers)
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=S2_TIMEOUT)) as resp:
                if resp.status == 400:
                    logger.debug("S2 bulk search 400, retrying without minCitationCount")
                    params.pop("minCitationCount", None)
                    params.pop("sort", None)
                    await _enforce_s2_auth_rate_limit(headers)
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=S2_TIMEOUT)) as retry_resp:
                        if retry_resp.status != 200:
                            logger.warning("Semantic Scholar bulk retry returned %d for: %s", retry_resp.status, query[:60])
                            return results
                        data = await retry_resp.json()
                elif resp.status == 403 and _has_s2_auth_header(headers):
                    _disable_s2_key()
                    async with aiohttp.ClientSession() as anon_session:
                        async with anon_session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=S2_TIMEOUT)) as anon_resp:
                            if anon_resp.status != 200:
                                logger.warning("Semantic Scholar anon retry returned %d for: %s", anon_resp.status, query[:60])
                                return results
                            data = await anon_resp.json()
                elif resp.status != 200:
                    logger.warning("Semantic Scholar returned %d for: %s", resp.status, query[:60])
                    return results
                else:
                    data = await resp.json()
    except Exception as exc:
        logger.warning("Semantic Scholar request failed: %s", exc)
        return results

    papers = data.get("data", [])[:max_results]
    if not papers:
        return results

    lightweight_results = [_parse_s2_paper(p) for p in papers]

    paper_ids = [r.paper_id for r in lightweight_results if r.paper_id]
    if paper_ids:
        detail_map = await _batch_fetch_details(paper_ids, headers)
        for r in lightweight_results:
            if r.paper_id in detail_map:
                _apply_detail(r, detail_map[r.paper_id])

    return lightweight_results


async def _batch_fetch_details(
    paper_ids: list[str], headers: dict[str, str],
) -> dict[str, dict]:
    """Fetch abstract/TLDR/PDF for multiple papers via POST /paper/batch."""
    url = f"{S2_BASE}/paper/batch"
    params = {"fields": S2_DETAIL_FIELDS}
    batch_headers = {**headers, "Content-Type": "application/json"}

    try:
        timeout = aiohttp.ClientTimeout(total=S2_BATCH_TIMEOUT)
        async with aiohttp.ClientSession(headers=batch_headers) as session:
            await _enforce_s2_auth_rate_limit(batch_headers)
            async with session.post(
                url, params=params, json={"ids": paper_ids}, timeout=timeout,
            ) as resp:
                if resp.status == 403 and _has_s2_auth_header(batch_headers):
                    _disable_s2_key()
                    async with aiohttp.ClientSession() as anon_session:
                        async with anon_session.post(
                            url,
                            params=params,
                            json={"ids": paper_ids},
                            timeout=timeout,
                        ) as anon_resp:
                            if anon_resp.status != 200:
                                logger.debug("S2 batch detail anon retry returned %d", anon_resp.status)
                                return {}
                            data = await anon_resp.json()
                elif resp.status != 200:
                    logger.debug("S2 batch detail returned %d", resp.status)
                    return {}
                else:
                    data = await resp.json()
    except Exception as exc:
        logger.debug("S2 batch detail failed: %s", exc)
        return {}

    result_map: dict[str, dict] = {}
    if isinstance(data, list):
        for pid, detail in zip(paper_ids, data):
            if detail:
                result_map[pid] = detail
    return result_map


def _apply_detail(result: AcademicResult, detail: dict) -> None:
    """Apply batch-fetched detail data to an AcademicResult."""
    if not result.abstract:
        result.abstract = (detail.get("abstract") or "")[:2000]
    if not result.authors:
        author_names = []
        for a in (detail.get("authors") or []):
            name = a.get("name", "")
            if name:
                author_names.append(name)
        if author_names:
            result.authors = author_names[:10]
    if not result.tldr:
        tldr_obj = detail.get("tldr") or {}
        result.tldr = (tldr_obj.get("text") or "")[:500]
    if not result.open_access_pdf_url:
        oa_pdf = detail.get("openAccessPdf") or {}
        result.open_access_pdf_url = oa_pdf.get("url") or ""


async def search_crossref(
    query: str,
    max_results: int = 5,
) -> list[AcademicResult]:
    """Search Crossref metadata API with polite pool headers."""
    results: list[AcademicResult] = []
    cleaned_query = _clean_query_for_search(query)
    params = {
        "query": cleaned_query,
        "rows": str(max_results),
        "select": CROSSREF_SELECT,
    }
    headers: dict[str, str] = {}
    if CROSSREF_MAILTO:
        headers["User-Agent"] = f"jarvis/1.0 (mailto:{CROSSREF_MAILTO})"
    else:
        headers["User-Agent"] = "jarvis/1.0"

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(CROSSREF_BASE, params=params, timeout=aiohttp.ClientTimeout(total=CROSSREF_TIMEOUT)) as resp:
                if resp.status != 200:
                    logger.warning("Crossref returned %d for: %s", resp.status, query[:60])
                    return results
                data = await resp.json()
    except Exception as exc:
        logger.warning("Crossref request failed: %s", exc)
        return results

    for item in data.get("message", {}).get("items", []):
        doi_raw = item.get("DOI", "")
        doi = normalize_doi(doi_raw)

        titles = item.get("title", [])
        title = titles[0] if titles else ""

        containers = item.get("container-title", [])
        venue = containers[0] if containers else ""

        year = 0
        for date_key in ("published-print", "published-online"):
            date_parts = (item.get(date_key) or {}).get("date-parts", [[]])
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = int(date_parts[0][0])
                break

        author_names = []
        for a in (item.get("author") or []):
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                author_names.append(name)

        is_preprint, is_peer_reviewed = _classify_provenance(venue, doi)
        cite_count = item.get("is-referenced-by-count", 0) or 0

        abstract_raw = item.get("abstract", "")
        if abstract_raw:
            import re as _re
            import html as _html
            abstract_raw = _html.unescape(_re.sub(r"<[^>]+>", "", abstract_raw)).strip()

        results.append(AcademicResult(
            title=title,
            abstract=abstract_raw,
            authors=author_names[:10],
            year=year,
            venue=venue,
            doi=doi,
            doi_url=f"https://doi.org/{doi}" if doi else "",
            url=f"https://doi.org/{doi}" if doi else "",
            citation_count=cite_count,
            source_provider="crossref",
            is_peer_reviewed=is_peer_reviewed,
            is_preprint=is_preprint,
        ))

    return results


async def search_academic(
    query: str,
    max_results: int = 5,
) -> list[AcademicResult]:
    """Orchestrate scholarly search: S2 first, Crossref fills gaps, dedup by DOI.

    Returns merged results sorted by citation count (descending).
    """
    cached = _cache.get(query)
    if cached is not None:
        return [AcademicResult(**r) for r in cached]

    s2_results = await search_semantic_scholar(query, max_results=max_results)
    cr_results = await search_crossref(query, max_results=max_results)

    seen_dois: set[str] = set()
    merged: list[AcademicResult] = []

    for r in s2_results:
        if r.doi:
            seen_dois.add(r.doi)
        merged.append(r)

    for r in cr_results:
        if r.doi and r.doi in seen_dois:
            continue
        if r.doi:
            seen_dois.add(r.doi)
        merged.append(r)

    merged.sort(key=lambda r: r.citation_count, reverse=True)
    merged = merged[:max_results]

    try:
        cache_data = []
        for r in merged:
            cache_data.append({
                "title": r.title,
                "abstract": r.abstract[:500],
                "authors": r.authors,
                "year": r.year,
                "venue": r.venue,
                "doi": r.doi,
                "doi_url": r.doi_url,
                "url": r.url,
                "citation_count": r.citation_count,
                "influential_citation_count": r.influential_citation_count,
                "source_provider": r.source_provider,
                "is_peer_reviewed": r.is_peer_reviewed,
                "is_preprint": r.is_preprint,
                "tldr": r.tldr,
                "is_open_access": r.is_open_access,
                "open_access_pdf_url": r.open_access_pdf_url,
                "paper_id": r.paper_id,
            })
        _cache.put(query, cache_data)
    except Exception as exc:
        logger.debug("Failed to cache academic results: %s", exc)

    return merged


def format_academic_results_for_llm(
    results: list[AcademicResult],
    max_chars: int = 2000,
) -> str:
    """Format academic results as fenced scholarly reference material."""
    if not results:
        return "[No scholarly results found]"

    lines = [
        "=== SCHOLARLY SEARCH RESULTS (from academic databases, NOT the open web) ===",
        "These are peer-reviewed papers and academic publications. Cite provenance when answering.\n",
    ]
    total = sum(len(l) for l in lines)

    for i, r in enumerate(results, 1):
        authors_str = ", ".join(r.authors[:3])
        if len(r.authors) > 3:
            authors_str += " et al."

        provenance = ""
        if r.is_peer_reviewed:
            provenance = "[Peer-reviewed]"
        elif r.is_preprint:
            provenance = "[Preprint]"

        block = f"[{i}] {r.title}\n"
        if authors_str:
            block += f"    Authors: {authors_str}\n"
        if r.venue:
            block += f"    Venue: {r.venue}"
            if r.year:
                block += f" ({r.year})"
            block += "\n"
        elif r.year:
            block += f"    Year: {r.year}\n"
        if r.doi:
            block += f"    DOI: {r.doi}\n"
        if r.citation_count > 0 or r.influential_citation_count > 0:
            cite_str = f"{r.citation_count}"
            if r.influential_citation_count > 0:
                cite_str += f" ({r.influential_citation_count} influential)"
            block += f"    Citations: {cite_str}\n"
        if provenance:
            block += f"    Status: {provenance}\n"
        if r.tldr:
            block += f"    TLDR: {r.tldr[:300]}\n"
        elif r.abstract:
            abstract_trunc = r.abstract[:300]
            if len(r.abstract) > 300:
                abstract_trunc += "..."
            block += f"    Abstract: {abstract_trunc}\n"
        block += "\n"

        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)

    return "\n".join(lines)


def format_academic_results_for_user(
    results: list[AcademicResult],
    query: str = "",
    limit: int = 3,
) -> str:
    """Format scholarly results into a direct user-facing fallback reply."""
    if not results:
        return "I couldn't find any academic results right now."

    lines: list[str] = []
    if query:
        lines.append(f"Scholarly results for '{query}':")
    else:
        lines.append("Scholarly results:")

    for idx, result in enumerate(results[:limit], 1):
        title = result.title.strip() or "Untitled paper"
        parts = [f"{idx}. {title}"]
        if result.venue and result.year:
            parts.append(f"{result.venue} ({result.year})")
        elif result.venue:
            parts.append(result.venue)
        elif result.year:
            parts.append(str(result.year))

        if result.doi:
            parts.append(f"DOI {result.doi}")

        summary = result.tldr.strip() or result.abstract.strip()
        summary = summary[:220].rstrip() + ("..." if len(summary) > 220 else "") if summary else ""

        line = " — ".join(parts)
        if summary:
            line += f" — {summary}"
        lines.append(line)

    return "\n".join(lines)
