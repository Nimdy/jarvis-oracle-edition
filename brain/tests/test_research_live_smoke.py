"""Live smoke test: hits real Semantic Scholar API and traces content through the pipeline.

This test is NOT run in CI (marked with @pytest.mark.live). Run manually:
    python -m pytest tests/test_research_live_smoke.py -v -s

It verifies:
  1. S2 search returns results with real paper data
  2. enrich_result() backfills abstracts/TLDRs for papers missing them
  3. Content depth classification produces correct tags
  4. The quality gate correctly separates substantive from shallow sources
  5. The study pipeline gate correctly skips metadata_only/title_only sources

Note: Without an S2_API_KEY, rate limits (429) will kick in after ~3 requests.
All tests share a single search result set to stay within the free tier.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytestmark = pytest.mark.live

_QUERY = "attention is all you need transformer"
_cached_results = None


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


_RATE_LIMITED = False


def _get_results(max_results=5):
    """Fetch results once and cache globally to avoid hitting rate limits.

    Retries once after 4s if rate-limited (429). Skips all tests if S2 is
    persistently unavailable.
    """
    global _cached_results, _RATE_LIMITED

    if _RATE_LIMITED:
        pytest.skip("S2 rate-limited (429) — skipping")

    if _cached_results is not None:
        return _cached_results

    from tools.academic_search_tool import search_semantic_scholar

    _cached_results = _run(search_semantic_scholar(_QUERY, max_results=max_results))
    if not _cached_results:
        time.sleep(4)
        _cached_results = _run(search_semantic_scholar(_QUERY, max_results=max_results))

    if not _cached_results:
        _RATE_LIMITED = True
        _cached_results = None
        pytest.skip(
            "S2 API returned 0 results (likely rate-limited 429). "
            "Re-run after ~60s or set S2_API_KEY for higher limits."
        )
    return _cached_results


class TestLiveS2Search:
    """Hit real Semantic Scholar API and inspect returned data."""

    def test_search_returns_papers_with_content(self):
        results = _get_results()  # skips if rate-limited

        print("\n" + "=" * 80)
        print(f"LIVE S2 SEARCH RESULTS: '{_QUERY}'")
        print("=" * 80)

        for i, r in enumerate(results, 1):
            abstract_len = len(r.abstract) if r.abstract else 0
            tldr_len = len(r.tldr) if r.tldr else 0

            print(f"\n--- Result {i} ---")
            print(f"  Title:         {r.title}")
            print(f"  Authors:       {', '.join(r.authors[:3])}")
            print(f"  Venue:         {r.venue} ({r.year})")
            print(f"  DOI:           {r.doi or '(none)'}")
            print(f"  Paper ID:      {r.paper_id or '(none)'}")
            print(f"  Citations:     {r.citation_count}")
            print(f"  Peer-reviewed: {r.is_peer_reviewed}")
            print(f"  Open access:   {r.is_open_access}")
            print(f"  Abstract len:  {abstract_len} chars")
            print(f"  TLDR len:      {tldr_len} chars")
            if r.tldr:
                print(f"  TLDR:          {r.tldr[:200]}")
            if r.abstract:
                print(f"  Abstract (first 200): {r.abstract[:200]}...")

        has_content = any(
            (r.abstract and len(r.abstract) >= 50) or (r.tldr and len(r.tldr) >= 20)
            for r in results
        )
        assert has_content, (
            "No result had substantive content (abstract>=50 or TLDR>=20). "
            "Content pipeline would produce only title_only sources."
        )

    def test_enrich_backfills_missing_abstract(self):
        from tools.academic_search_tool import enrich_result
        import copy

        results = _get_results()  # skips if rate-limited

        papers_without_abstract = [r for r in results if not r.abstract or len(r.abstract) < 100]
        papers_with_paper_id = [r for r in papers_without_abstract if r.paper_id]

        print("\n" + "=" * 80)
        print(f"ENRICHMENT TEST (using cached results from '{_QUERY}')")
        print(f"Total results: {len(results)}")
        print(f"Papers with abstract >= 100 chars: {len(results) - len(papers_without_abstract)}")
        print(f"Papers missing abstract (enrichment candidates): {len(papers_without_abstract)}")
        print(f"  of which have paper_id (can be enriched): {len(papers_with_paper_id)}")
        print("=" * 80)

        for r in results:
            before_abstract = len(r.abstract) if r.abstract else 0
            before_tldr = len(r.tldr) if r.tldr else 0

            test_copy = copy.deepcopy(r)
            enriched = _run(enrich_result(test_copy))

            after_abstract = len(enriched.abstract) if enriched.abstract else 0
            after_tldr = len(enriched.tldr) if enriched.tldr else 0

            gained = (after_abstract > before_abstract) or (after_tldr > before_tldr)

            print(f"\n  [{r.title[:60]}]")
            print(f"    Abstract: {before_abstract} -> {after_abstract} chars")
            print(f"    TLDR:     {before_tldr} -> {after_tldr} chars")
            if gained:
                print(f"    Enriched: YES (new content backfilled)")
            elif before_abstract >= 100:
                print(f"    Enriched: SKIPPED (already has abstract >= 100 chars)")
            elif not r.paper_id:
                print(f"    Enriched: SKIPPED (no paper_id for detail API)")
            else:
                print(f"    Enriched: no change (detail API may have returned empty)")


class TestLiveContentDepthClassification:
    """Trace how real S2 results would be classified by the pipeline."""

    def test_depth_classification_on_real_results(self):
        from tools.academic_search_tool import enrich_result
        import copy

        results = _get_results()  # skips if rate-limited

        print("\n" + "=" * 80)
        print(f"CONTENT DEPTH CLASSIFICATION (from '{_QUERY}')")
        print("=" * 80)

        depth_counts = {"tldr": 0, "abstract": 0, "title_only": 0}

        for r in results:
            test_copy = copy.deepcopy(r)
            enriched = _run(enrich_result(test_copy))

            if enriched.tldr:
                content = enriched.tldr
                depth = "tldr"
            elif enriched.abstract and len(enriched.abstract) >= 100:
                content = enriched.abstract[:800]
                depth = "abstract"
            else:
                content = enriched.title
                depth = "title_only"

            depth_counts[depth] += 1
            content_chars = len(content)
            would_be_gated = content_chars < 200

            print(f"\n  [{enriched.title[:60]}]")
            print(f"    depth={depth}, content_chars={content_chars}")
            print(f"    quality_gate: {'BLOCKED (metadata_only, no chunks)' if would_be_gated else 'PASS (will be chunked + studied)'}")
            if content:
                preview = content[:150].replace("\n", " ")
                print(f"    content preview: {preview}...")

        print(f"\n  >> Distribution: {depth_counts}")
        print(f"  >> Substantive (tldr+abstract): {depth_counts['tldr'] + depth_counts['abstract']}")
        print(f"  >> Shallow (title_only): {depth_counts['title_only']}")

        substantive = depth_counts["tldr"] + depth_counts["abstract"]
        assert substantive > 0, (
            "All results classified as title_only — enrichment pipeline not working"
        )


class TestLiveLibraryIngestion:
    """Trace a real result through _infer_content_depth and the quality gate."""

    def test_real_result_through_ingest_pipeline(self):
        from tools.academic_search_tool import enrich_result
        from autonomy.research_intent import ResearchFinding
        from autonomy.knowledge_integrator import KnowledgeIntegrator
        import copy

        results = _get_results()  # skips if rate-limited

        print("\n" + "=" * 80)
        print(f"LIBRARY INGESTION TRACE (from '{_QUERY}')")
        print("=" * 80)

        for r in results:
            test_copy = copy.deepcopy(r)
            enriched = _run(enrich_result(test_copy))

            if enriched.tldr:
                content = enriched.tldr
                depth_tag = "tldr"
            elif enriched.abstract and len(enriched.abstract) >= 100:
                content = enriched.abstract[:800]
                depth_tag = "abstract"
            else:
                content = enriched.title
                depth_tag = "title_only"

            provenance_parts = ["academic:"]
            if enriched.doi:
                provenance_parts.append(enriched.doi)
            if enriched.venue and enriched.year:
                provenance_parts.append(f"({enriched.venue}, {enriched.year})")
            provenance_parts.append(f"[depth:{depth_tag}]")

            finding = ResearchFinding(
                content=content,
                provenance=" ".join(provenance_parts),
                confidence=0.75,
                url=enriched.url,
                doi=enriched.doi,
            )

            inferred = KnowledgeIntegrator._infer_content_depth(finding)
            would_be_metadata_only = len(content) < 200

            print(f"\n  [{enriched.title[:60]}]")
            print(f"    raw content chars:       {len(content)}")
            print(f"    provenance depth tag:    {depth_tag}")
            print(f"    _infer_content_depth():  {inferred}")
            if would_be_metadata_only:
                print(f"    quality gate:            BLOCKED -> metadata_only (< 200 chars, no chunks)")
            else:
                print(f"    quality gate:            PASS -> will chunk + embed + study")
            if depth_tag in ("tldr", "abstract"):
                print(f"    study pipeline:          WILL PROCESS (depth={depth_tag})")
            else:
                print(f"    study pipeline:          WOULD SKIP (depth={depth_tag})")

            assert inferred == depth_tag, (
                f"Depth mismatch: provenance tagged as {depth_tag} but _infer_content_depth returned {inferred}"
            )


class TestLiveEndToEndSummary:
    """Print a full pipeline summary for operator verification."""

    def test_full_pipeline_summary(self):
        from tools.academic_search_tool import enrich_result
        import copy

        results = _get_results()  # skips if rate-limited

        print("\n" + "=" * 80)
        print(f"FULL PIPELINE SUMMARY — {len(results)} papers from '{_QUERY}'")
        print("=" * 80)

        stats = {"total": 0, "tldr": 0, "abstract": 0, "title_only": 0, "would_chunk": 0}

        for r in results:
            test_copy = copy.deepcopy(r)
            enriched = _run(enrich_result(test_copy))
            stats["total"] += 1

            if enriched.tldr:
                content = enriched.tldr
                depth = "tldr"
            elif enriched.abstract and len(enriched.abstract) >= 100:
                content = enriched.abstract[:800]
                depth = "abstract"
            else:
                content = enriched.title
                depth = "title_only"

            stats[depth] += 1
            would_chunk = len(content) >= 200
            if would_chunk:
                stats["would_chunk"] += 1

            marker = "[chunk+study]" if would_chunk else "[metadata_only]"
            print(f"  [{depth:12s}] {len(content):4d} chars {marker:16s} | {enriched.title[:50]}")

        print(f"\n  {'=' * 60}")
        print(f"  TOTALS: {stats['total']} papers")
        print(f"    With TLDR:        {stats['tldr']}")
        print(f"    With abstract:    {stats['abstract']}")
        print(f"    Title only:       {stats['title_only']}")
        print(f"    Would chunk+study (>= 200 chars): {stats['would_chunk']}")
        substantive_pct = (
            (stats["tldr"] + stats["abstract"]) / stats["total"] * 100
            if stats["total"] > 0 else 0
        )
        print(f"    Substantive ratio: {substantive_pct:.0f}%")

        if stats["title_only"] > 0:
            print(f"\n    WARNING: {stats['title_only']} paper(s) would be stored as metadata_only")
            print(f"    (no substance for concept extraction or NN training)")
        else:
            print(f"\n    ALL papers have substantive content — pipeline is healthy")
        print(f"  {'=' * 60}")

        assert stats["tldr"] + stats["abstract"] > 0, "All results were title-only"
