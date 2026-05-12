#!/usr/bin/env python3
"""Standalone CLI for textbook ingestion with dry-run preview.

Usage:
    # Full ingest (d2l.ai)
    python -m scripts.ingest_textbook https://d2l.ai

    # Dry run — preview what would be ingested without storing anything
    python -m scripts.ingest_textbook https://d2l.ai --dry-run

    # Inspect a single page
    python -m scripts.ingest_textbook https://d2l.ai --page https://d2l.ai/chapter_linear-networks/linear-regression.html

    # Dry run a single page with full content preview
    python -m scripts.ingest_textbook https://d2l.ai --page https://d2l.ai/chapter_linear-networks/linear-regression.html --dry-run --verbose

    # Custom title and tags
    python -m scripts.ingest_textbook https://d2l.ai --title "Dive into Deep Learning" --tags "deep_learning,ml,textbook"

    # Study chapters immediately after ingesting
    python -m scripts.ingest_textbook https://d2l.ai --study
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_textbook")


def _print_chapter_summary(result) -> None:
    """Print a compact table of chapter results."""
    print(f"\n{'='*80}")
    print(f"  Textbook: {result.title}")
    print(f"  TOC URL:  {result.toc_url}")
    print(f"  Type:     {result.site_type}")
    print(f"{'='*80}")
    print(f"  Discovered: {result.chapters_discovered}")
    print(f"  Ingested:   {result.chapters_ingested}")
    print(f"  Skipped:    {result.chapters_skipped}")
    print(f"  Failed:     {result.chapters_failed}")
    print(f"  Math:       {result.total_math} blocks preserved")
    print(f"  Code:       {result.total_code} blocks preserved")
    print(f"  Chunks:     {result.total_chunks}")
    print(f"{'='*80}\n")

    if result.results:
        print(f"  {'#':>3}  {'Quality':>7}  {'Math':>5}  {'Code':>5}  {'Chunks':>6}  {'Status':<10}  Title")
        print(f"  {'---':>3}  {'-------':>7}  {'-----':>5}  {'-----':>5}  {'------':>6}  {'----------':<10}  -----")
        for i, ch in enumerate(result.results, 1):
            status = "OK" if ch.success else ("SKIP" if ch.skipped else "FAIL")
            reason = f" ({ch.skip_reason})" if ch.skipped else ""
            reason = f" ({ch.error[:30]})" if ch.error and not ch.skipped else reason
            print(f"  {i:>3}  {ch.quality_score:>7.2f}  {ch.math_preserved:>5}  {ch.code_preserved:>5}  {ch.chunk_count:>6}  {status:<10}  {ch.title[:50]}{reason}")
    print()


def _inspect_single_page(page_url: str, site_type: str, verbose: bool) -> None:
    """Fetch, sanitize, and display a single page for inspection."""
    from library.batch_ingest import _fetch_raw_html
    from library.content_sanitizer import detect_site_type, sanitize

    print(f"\nFetching: {page_url}")
    raw_html, err = _fetch_raw_html(page_url)
    if err:
        print(f"  ERROR: {err}")
        return

    if not site_type:
        site_type = detect_site_type(raw_html)
    print(f"  Detected type: {site_type}")
    print(f"  Raw HTML size: {len(raw_html):,} bytes")

    result = sanitize(raw_html, site_type=site_type)
    print(f"\n  Sanitized content:")
    print(f"    Quality score:   {result.quality_score:.3f}")
    print(f"    Math preserved:  {result.math_blocks_preserved}")
    print(f"    Code preserved:  {result.code_blocks_preserved}")
    print(f"    Warnings:        {result.warnings or 'none'}")
    print(f"    Text length:     {len(result.text):,} chars")

    if result.stats:
        print(f"    Stats:           {result.stats}")

    from library.chunks import chunk_text
    chunks = chunk_text(result.text, f"preview_{page_url}")
    print(f"    Chunks:          {len(chunks)}")

    from library.blue_diamonds import GRADUATION_MIN_QUALITY
    bd_eligible = result.quality_score >= GRADUATION_MIN_QUALITY
    print(f"\n  Blue Diamond eligible: {'YES' if bd_eligible else 'NO'} (needs >= {GRADUATION_MIN_QUALITY:.2f})")

    if verbose:
        print(f"\n{'='*80}")
        print("  SANITIZED CONTENT PREVIEW")
        print(f"{'='*80}\n")

        preview_lines = result.text.split("\n")
        for i, line in enumerate(preview_lines[:150], 1):
            print(f"  {i:>4} | {line}")
        if len(preview_lines) > 150:
            print(f"  ... ({len(preview_lines) - 150} more lines)")

        print(f"\n{'='*80}")
        print("  FIRST 3 CHUNKS")
        print(f"{'='*80}")
        for i, chunk in enumerate(chunks[:3]):
            chunk_text_content = chunk.chunk_text if hasattr(chunk, "chunk_text") else str(chunk)
            print(f"\n  --- Chunk {i+1} ({len(chunk_text_content)} chars) ---")
            wrapped = textwrap.fill(chunk_text_content[:500], width=76, initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
            if len(chunk_text_content) > 500:
                print(f"  ... ({len(chunk_text_content) - 500} more chars)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a multi-chapter textbook into the Jarvis library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s https://d2l.ai --dry-run
              %(prog)s https://d2l.ai --page https://d2l.ai/chapter_linear-networks/linear-regression.html --dry-run --verbose
              %(prog)s https://d2l.ai --title "Dive into Deep Learning" --tags "deep_learning,ml" --study
        """),
    )
    parser.add_argument("toc_url", help="Table of contents URL for the textbook")
    parser.add_argument("--title", default="", help="Textbook title (auto-detected if omitted)")
    parser.add_argument("--tags", default="textbook", help="Comma-separated domain tags (default: textbook)")
    parser.add_argument("--dry-run", action="store_true", help="Preview ingestion without storing anything")
    parser.add_argument("--study", action="store_true", help="Run study pipeline on each chapter immediately")
    parser.add_argument("--page", default="", help="Inspect a single page URL instead of full ingest")
    parser.add_argument("--site-type", default="", choices=["", "sphinx_mathjax", "pdf2html", "generic"],
                        help="Force site type (auto-detected if omitted)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full content preview (with --page)")

    args = parser.parse_args()

    if args.page:
        _inspect_single_page(args.page, args.site_type, args.verbose)
        return

    mode = "DRY RUN" if args.dry_run else "LIVE INGEST"
    print(f"\n[{mode}] Starting textbook ingestion: {args.toc_url}\n")

    from library.batch_ingest import ingest_textbook

    result = ingest_textbook(
        toc_url=args.toc_url,
        title=args.title,
        domain_tags=args.tags,
        study_now=args.study,
        dry_run=args.dry_run,
    )

    _print_chapter_summary(result)

    if args.dry_run:
        print("  [DRY RUN] No data was stored. Re-run without --dry-run to ingest.")
    elif result.success:
        print(f"  Successfully ingested {result.chapters_ingested} chapters.")
    else:
        print(f"  Ingestion failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
