#!/usr/bin/env python3
"""Diagnostic script: shows exactly what academic APIs return and what content is fetchable.

Run: python3 brain/tools/research_diagnostic.py "your search query here"

This does NOT modify any Jarvis state. Read-only diagnostic.
"""

import asyncio
import json
import subprocess
import sys
import urllib.request
from textwrap import indent

import aiohttp

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,authors,year,venue,externalIds,citationCount,influentialCitationCount,isOpenAccess,tldr,openAccessPdf"
CROSSREF_BASE = "https://api.crossref.org/works"
CROSSREF_SELECT = "DOI,title,author,published-print,published-online,container-title,is-referenced-by-count,abstract"

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def heading(text: str) -> None:
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")


def subheading(text: str) -> None:
    print(f"\n{CYAN}--- {text} ---{RESET}")


def ok(text: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {text}")


def warn(text: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {text}")


def fail(text: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {text}")


def dim(text: str) -> None:
    print(f"  {DIM}{text}{RESET}")


def content_preview(text: str, label: str = "Content", max_chars: int = 300) -> None:
    if not text:
        warn(f"{label}: (empty)")
        return
    clean = text.strip()
    is_binary = any(c in clean[:200] for c in ['\x00', '\ufffd', '>>stream'])
    has_pdf_markers = any(m in clean[:500] for m in ['%PDF', '>>stream', '/Filter', '/FlateDecode', 'endobj'])
    if is_binary or has_pdf_markers:
        fail(f"{label}: BINARY/PDF GARBAGE detected ({len(clean)} chars)")
        dim(f"First 100 chars: {repr(clean[:100])}")
        return
    printable_ratio = sum(1 for c in clean[:500] if c.isprintable() or c in '\n\r\t') / max(len(clean[:500]), 1)
    if printable_ratio < 0.8:
        fail(f"{label}: Low printable ratio ({printable_ratio:.0%}) - likely garbled ({len(clean)} chars)")
        dim(f"First 100 chars: {repr(clean[:100])}")
        return
    preview = clean[:max_chars]
    if len(clean) > max_chars:
        preview += "..."
    ok(f"{label} ({len(clean)} chars, {printable_ratio:.0%} printable):")
    print(indent(preview, "    "))


def try_fetch_url(url: str, timeout: int = 15) -> dict:
    """Fetch a URL and report what we get back."""
    result = {"url": url, "success": False, "content_type": "", "size": 0, "text": "", "is_pdf": False, "is_html": False}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "JarvisDiagnostic/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            result["content_type"] = ct
            result["final_url"] = resp.url
            raw = resp.read(50_000)  # 50KB cap for diagnostic
            result["size"] = len(raw)
            result["is_pdf"] = "pdf" in ct.lower() or raw[:5] == b"%PDF-"
            result["is_html"] = "html" in ct.lower()

            if result["is_pdf"]:
                result["text"] = f"[PDF binary, {len(raw)} bytes]"
                try:
                    proc = subprocess.run(
                        ["pdftotext", "-", "-"],
                        input=raw, capture_output=True, timeout=10,
                    )
                    if proc.returncode == 0 and len(proc.stdout) > 50:
                        result["pdf_extracted"] = proc.stdout.decode("utf-8", errors="replace")[:2000]
                except FileNotFoundError:
                    result["pdf_extracted"] = "[pdftotext not installed]"
                except Exception as e:
                    result["pdf_extracted"] = f"[extraction failed: {e}]"
            else:
                result["text"] = raw.decode("utf-8", errors="replace")[:5000]

            result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    return result


async def search_s2(query: str) -> list[dict]:
    params = {"query": query, "limit": "5", "fields": S2_FIELDS, "fieldsOfStudy": "Computer Science"}
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{S2_BASE}/paper/search", params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                fail(f"S2 returned HTTP {resp.status}")
                return []
            data = await resp.json()
            return data.get("data", [])


async def search_crossref(query: str) -> list[dict]:
    params = {"query": query, "rows": "5", "select": CROSSREF_SELECT}
    headers = {"User-Agent": "JarvisDiagnostic/1.0 (mailto:research@jarvis.local)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(CROSSREF_BASE, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                fail(f"Crossref returned HTTP {resp.status}")
                return []
            data = await resp.json()
            return data.get("message", {}).get("items", [])


def analyze_s2_paper(paper: dict, idx: int) -> None:
    title = paper.get("title", "?")
    subheading(f"S2 Result #{idx}: {title[:80]}")

    year = paper.get("year", "?")
    venue = paper.get("venue", "(none)")
    citations = paper.get("citationCount", 0)
    is_oa = paper.get("isOpenAccess", False)
    print(f"  Year: {year}  |  Venue: {venue}  |  Citations: {citations}  |  Open Access: {is_oa}")

    ext_ids = paper.get("externalIds", {})
    doi = ext_ids.get("DOI", "")
    arxiv = ext_ids.get("ArXiv", "")
    if doi:
        ok(f"DOI: {doi}  ->  https://doi.org/{doi}")
    elif arxiv:
        ok(f"ArXiv: {arxiv}  ->  https://arxiv.org/abs/{arxiv}")
    else:
        warn("No DOI or ArXiv ID")

    abstract = paper.get("abstract", "")
    tldr_obj = paper.get("tldr") or {}
    tldr = tldr_obj.get("text", "")

    if abstract:
        content_preview(abstract, "Abstract")
    else:
        warn("Abstract: (none returned by API)")

    if tldr:
        content_preview(tldr, "TLDR")
    else:
        dim("TLDR: (none)")

    oa_pdf = paper.get("openAccessPdf") or {}
    pdf_url = oa_pdf.get("url", "")
    if pdf_url:
        ok(f"Open Access PDF URL: {pdf_url}")
        print(f"  {YELLOW}  Fetching PDF to check...{RESET}")
        fetch_result = try_fetch_url(pdf_url)
        if fetch_result["success"]:
            ok(f"PDF fetch: Content-Type={fetch_result['content_type']}, size={fetch_result['size']} bytes")
            if fetch_result.get("is_pdf"):
                ok("Confirmed: actual PDF file")
                extracted = fetch_result.get("pdf_extracted", "")
                if extracted and not extracted.startswith("["):
                    content_preview(extracted, "PDF extracted text")
                else:
                    warn(f"PDF text extraction: {extracted}")
            else:
                warn(f"NOT a PDF! Content-Type: {fetch_result['content_type']}")
                content_preview(fetch_result.get("text", ""), "Fetched content")
        else:
            fail(f"PDF fetch failed: {fetch_result.get('error', '?')}")
    else:
        dim("No open access PDF URL available")

    if doi and not pdf_url:
        doi_url = f"https://doi.org/{doi}"
        print(f"  {YELLOW}  Checking DOI URL redirect...{RESET}")
        fetch_result = try_fetch_url(doi_url)
        if fetch_result["success"]:
            final = fetch_result.get("final_url", doi_url)
            ct = fetch_result["content_type"]
            ok(f"DOI redirects to: {final}")
            ok(f"Content-Type: {ct}, size: {fetch_result['size']} bytes")
            if fetch_result.get("is_pdf"):
                ok("It's a PDF!")
                extracted = fetch_result.get("pdf_extracted", "")
                if extracted and not extracted.startswith("["):
                    content_preview(extracted, "PDF extracted text")
            elif fetch_result.get("is_html"):
                text = fetch_result.get("text", "")
                has_paywall = any(w in text.lower() for w in ["sign in", "subscribe", "purchase", "access denied", "cookie", "consent"])
                if has_paywall:
                    warn("HTML page has paywall/cookie indicators")
                content_preview(text[:500], "HTML page preview")
            else:
                warn(f"Unknown content type: {ct}")
        else:
            fail(f"DOI URL fetch failed: {fetch_result.get('error', '?')}")

    what_jarvis_stores = ""
    depth = ""
    if tldr:
        what_jarvis_stores = tldr[:500]
        depth = "tldr"
    elif abstract and len(abstract) >= 100:
        what_jarvis_stores = abstract[:800]
        depth = "abstract"
    else:
        what_jarvis_stores = title
        depth = "title_only"

    print(f"\n  {BOLD}What Jarvis would store:{RESET}")
    print(f"  content_depth = {depth}")
    print(f"  content ({len(what_jarvis_stores)} chars): {what_jarvis_stores[:200]}{'...' if len(what_jarvis_stores) > 200 else ''}")


def analyze_crossref_item(item: dict, idx: int) -> None:
    titles = item.get("title", ["?"])
    title = titles[0] if titles else "?"
    subheading(f"Crossref Result #{idx}: {title[:80]}")

    doi = item.get("DOI", "")
    container = item.get("container-title", [""])[0] if item.get("container-title") else ""
    citations = item.get("is-referenced-by-count", 0)
    pub = item.get("published-print") or item.get("published-online") or {}
    year = pub.get("date-parts", [[0]])[0][0] if pub.get("date-parts") else "?"
    print(f"  Year: {year}  |  Journal: {container or '(none)'}  |  Citations: {citations}")

    if doi:
        ok(f"DOI: {doi}  ->  https://doi.org/{doi}")
    else:
        warn("No DOI")

    abstract = item.get("abstract", "")
    if abstract:
        import re
        clean = re.sub(r"<[^>]+>", "", abstract)
        content_preview(clean, "Abstract")
    else:
        warn("Abstract: (none returned by Crossref)")


async def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} \"search query\"")
        print(f"Example: python3 {sys.argv[0]} \"self-adaptive systems runtime verification\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    heading(f"Research Diagnostic: \"{query}\"")

    # Semantic Scholar
    heading("SEMANTIC SCHOLAR RESULTS")
    try:
        s2_results = await search_s2(query)
        ok(f"Got {len(s2_results)} results")
        for i, paper in enumerate(s2_results, 1):
            analyze_s2_paper(paper, i)
    except Exception as e:
        fail(f"S2 search failed: {e}")

    # Crossref
    heading("CROSSREF RESULTS")
    try:
        cr_results = await search_crossref(query)
        ok(f"Got {len(cr_results)} results")
        for i, item in enumerate(cr_results, 1):
            analyze_crossref_item(item, i)
    except Exception as e:
        fail(f"Crossref search failed: {e}")

    # Summary
    heading("DIAGNOSTIC SUMMARY")
    total_s2 = len(s2_results) if 's2_results' in dir() else 0
    total_cr = len(cr_results) if 'cr_results' in dir() else 0

    s2_with_abstract = sum(1 for p in (s2_results if 's2_results' in dir() else []) if p.get("abstract"))
    s2_with_tldr = sum(1 for p in (s2_results if 's2_results' in dir() else []) if (p.get("tldr") or {}).get("text"))
    s2_with_pdf = sum(1 for p in (s2_results if 's2_results' in dir() else []) if (p.get("openAccessPdf") or {}).get("url"))
    s2_open = sum(1 for p in (s2_results if 's2_results' in dir() else []) if p.get("isOpenAccess"))

    print(f"  Semantic Scholar: {total_s2} results")
    print(f"    With abstract:         {s2_with_abstract}/{total_s2}")
    print(f"    With TLDR:             {s2_with_tldr}/{total_s2}")
    print(f"    With PDF URL:          {s2_with_pdf}/{total_s2}")
    print(f"    Open access:           {s2_open}/{total_s2}")
    print(f"  Crossref: {total_cr} results")

    cr_with_abstract = sum(1 for i in (cr_results if 'cr_results' in dir() else []) if i.get("abstract"))
    print(f"    With abstract:         {cr_with_abstract}/{total_cr}")

    print(f"\n  {BOLD}What Jarvis currently learns from these:{RESET}")
    s2_title_only = total_s2 - s2_with_abstract - s2_with_tldr + min(s2_with_abstract, s2_with_tldr)
    print(f"    full_text (fetched):   {s2_with_pdf} (if fetch succeeds + pdftotext works)")
    print(f"    abstract:              {s2_with_abstract}")
    print(f"    tldr:                  {s2_with_tldr}")
    print(f"    title_only:            ~{max(0, total_s2 - s2_with_abstract)} (no real content)")

    if s2_with_pdf == 0:
        warn("No PDFs available — Jarvis has nothing to deeply learn from these results")
    if s2_with_abstract < total_s2 // 2:
        warn("Less than half have abstracts — mostly metadata-only results")


if __name__ == "__main__":
    asyncio.run(main())
