"""Content sanitization for textbook and web source ingestion.

Three sanitization strategies based on auto-detected site type:

1. **Sphinx/MathJax** (e.g. d2l.ai): Extracts LaTeX math and code blocks from
   raw HTML BEFORE stripping tags, then re-injects as $...$ / $$...$$ / fenced
   code. Preserves math accuracy for Blue Diamond graduation.

2. **pdf2htmlEX** (e.g. deeplearningbook.org): Best-effort cleanup of
   pixel-positioned glyph HTML. Math is structurally unrecoverable, so quality
   score is capped to prevent Blue Diamond graduation of garbled content.

3. **Generic**: Paragraph re-joining, nav/sidebar removal, whitespace
   normalization. Used for arbitrary URLs.

All functions are pure — no state, no side effects, no external dependencies
beyond stdlib + re.
"""

from __future__ import annotations

import html as html_mod
import re
from dataclasses import dataclass, field


@dataclass
class SanitizedContent:
    text: str
    quality_score: float = 0.0
    math_blocks_preserved: int = 0
    code_blocks_preserved: int = 0
    warnings: list[str] = field(default_factory=list)
    site_type: str = "generic"
    stats: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Site type detection
# ---------------------------------------------------------------------------

def detect_site_type(raw_html: str) -> str:
    """Auto-detect source format from HTML content."""
    lower = raw_html[:5000].lower()
    if "pdf2htmlex" in lower or "pdf2htmlEX" in raw_html[:5000]:
        return "pdf2html"
    if "mathjax" in lower or 'class="math notranslate' in raw_html[:50000]:
        return "sphinx_mathjax"
    return "generic"


# ---------------------------------------------------------------------------
# Sphinx / MathJax sanitizer (d2l.ai path)
# ---------------------------------------------------------------------------

_INLINE_MATH_RE = re.compile(
    r'<span\s+class="math\s+notranslate[^"]*">\s*(\\\(.*?\\\))\s*</span>',
    re.DOTALL,
)

_DISPLAY_MATH_RE = re.compile(
    r'\\\[(.*?)\\\]',
    re.DOTALL,
)

_EQNO_RE = re.compile(
    r'<span\s+class="eqno">\s*\(([^)]+)\)',
)

_CODE_BLOCK_RE = re.compile(
    r'<div\s+class="highlight-python\s+notranslate">\s*<div\s+class="highlight">\s*<pre>(.*?)</pre>',
    re.DOTALL,
)

_PYTORCH_TAB_RE = re.compile(
    r'<div\s+class="mdl-tabs__panel\s+is-active"\s+id="pytorch[^"]*">\s*'
    r'<div\s+class="highlight-python\s+notranslate">\s*<div\s+class="highlight">\s*<pre>(.*?)</pre>',
    re.DOTALL,
)

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|nav|header|footer)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)

_NAV_CRUFT_PATTERNS = [
    re.compile(r"^(Previous|Next|Table of Contents|Show Source|Quick search).*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(navigate_next|Copyright|©).*$", re.MULTILINE),
    re.compile(r"^\s*(pytorch|mxnet|jax|tensorflow)\s*$", re.MULTILINE | re.IGNORECASE),
]


def _strip_html_tags(text: str) -> str:
    """Remove remaining HTML tags and unescape entities."""
    text = _TAG_RE.sub(" ", text)
    text = html_mod.unescape(text)
    return text


def _clean_latex(raw: str) -> str:
    """Clean up LaTeX extracted from HTML: unescape HTML entities."""
    text = html_mod.unescape(raw)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()


def sanitize_sphinx_mathjax(raw_html: str) -> SanitizedContent:
    """Sanitize Sphinx/MathJax HTML, preserving LaTeX math and code blocks."""
    warnings: list[str] = []
    stats: dict[str, int] = {}

    math_store: dict[str, str] = {}
    code_store: dict[str, str] = {}
    eqno_store: dict[str, str] = {}
    math_counter = 0
    code_counter = 0

    working = raw_html

    # -- Extract equation numbers before they get stripped --
    for m in _EQNO_RE.finditer(working):
        eqno_store[m.group(0)] = m.group(1)

    # -- Extract inline math: \(...\) inside <span class="math"> --
    def _replace_inline(m: re.Match) -> str:
        nonlocal math_counter
        latex = m.group(1)
        latex = latex.removeprefix("\\(").removesuffix("\\)")
        latex = _clean_latex(latex)
        key = f"<<MATH_{math_counter}>>"
        math_store[key] = f"${latex}$"
        math_counter += 1
        return f" {key} "

    working = _INLINE_MATH_RE.sub(_replace_inline, working)
    inline_count = math_counter
    stats["inline_math"] = inline_count

    # -- Extract display math: \[...\] --
    def _replace_display(m: re.Match) -> str:
        nonlocal math_counter
        latex = _clean_latex(m.group(1))
        latex = re.sub(r"\\begin\{split\}|\\end\{split\}", "", latex).strip()
        key = f"<<MATH_{math_counter}>>"
        math_store[key] = f"\n\n$${latex}$$\n\n"
        math_counter += 1
        return f"\n{key}\n"

    working = _DISPLAY_MATH_RE.sub(_replace_display, working)
    stats["display_math"] = math_counter - inline_count

    # -- Extract code blocks (prefer PyTorch tabs) --
    def _extract_code(pre_html: str) -> str:
        code = _strip_html_tags(pre_html)
        code = html_mod.unescape(code)
        lines = code.split("\n")
        cleaned = "\n".join(line for line in lines if line.strip())
        return cleaned.strip()

    for m in _PYTORCH_TAB_RE.finditer(working):
        code = _extract_code(m.group(1))
        if code:
            key = f"<<CODE_{code_counter}>>"
            code_store[key] = f"\n\n```python\n{code}\n```\n\n"
            working = working.replace(m.group(0), f"\n{key}\n", 1)
            code_counter += 1

    remaining_code_html = working
    for m in _CODE_BLOCK_RE.finditer(remaining_code_html):
        if any(f"<<CODE_{i}>>" in working[max(0, m.start()-200):m.start()] for i in range(code_counter)):
            continue
        code = _extract_code(m.group(1))
        if code and f"<<CODE_" not in code:
            key = f"<<CODE_{code_counter}>>"
            code_store[key] = f"\n\n```python\n{code}\n```\n\n"
            working = working.replace(m.group(0), f"\n{key}\n", 1)
            code_counter += 1

    stats["code_blocks"] = code_counter

    # -- Strip scripts, styles, nav elements --
    working = _SCRIPT_STYLE_RE.sub(" ", working)

    # -- Strip remaining HTML tags --
    working = _strip_html_tags(working)

    # -- Re-inject preserved content --
    for key, value in math_store.items():
        working = working.replace(key, value)
    for key, value in code_store.items():
        working = working.replace(key, value)

    # -- Remove navigation cruft --
    for pattern in _NAV_CRUFT_PATTERNS:
        working = pattern.sub("", working)

    # -- Paragraph cleanup --
    working = _join_split_paragraphs(working)

    # -- Whitespace normalization --
    working = _TRAILING_WS_RE.sub("", working)
    working = _MULTI_BLANK_RE.sub("\n\n", working)
    working = working.strip()

    # -- Quality scoring --
    quality = _compute_quality_score(working)
    total_math = math_counter

    if total_math == 0 and "math" in raw_html.lower():
        warnings.append("math_detected_in_html_but_none_extracted")

    return SanitizedContent(
        text=working,
        quality_score=quality,
        math_blocks_preserved=total_math,
        code_blocks_preserved=code_counter,
        warnings=warnings,
        site_type="sphinx_mathjax",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# pdf2htmlEX sanitizer (best-effort, math-degraded)
# ---------------------------------------------------------------------------

_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_CHAPTER_HEADER_RE = re.compile(
    r"^\s*CHAPTER\s+\d+\.\s+[A-Z\s]+$", re.MULTILINE,
)
_SINGLE_CHAR_LINE_RE = re.compile(r"^\s*[^\s]{1,2}\s*$", re.MULTILINE)

PDF2HTML_MAX_QUALITY = 0.60


def sanitize_pdf2html(raw_html: str) -> SanitizedContent:
    """Best-effort cleanup for pdf2htmlEX output. Math is structurally lost."""
    warnings = ["math_degraded:pdf2html_source"]
    stats: dict[str, int] = {}

    working = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    working = _strip_html_tags(working)

    page_nums = len(_PAGE_NUM_RE.findall(working))
    working = _PAGE_NUM_RE.sub("", working)
    stats["page_numbers_removed"] = page_nums

    headers = _CHAPTER_HEADER_RE.findall(working)
    if len(headers) > 1:
        first = True
        def _dedup_header(m: re.Match) -> str:
            nonlocal first
            if first:
                first = False
                return m.group(0)
            return ""
        working = _CHAPTER_HEADER_RE.sub(_dedup_header, working)
        stats["headers_deduped"] = len(headers) - 1

    single_before = len(_SINGLE_CHAR_LINE_RE.findall(working))

    working = _join_split_paragraphs(working)
    working = _merge_math_fragments(working)

    single_after = len(_SINGLE_CHAR_LINE_RE.findall(working))
    stats["math_fragments_joined"] = max(0, single_before - single_after)

    working = _TRAILING_WS_RE.sub("", working)
    working = _MULTI_BLANK_RE.sub("\n\n", working)
    working = working.strip()

    quality = min(_compute_quality_score(working), PDF2HTML_MAX_QUALITY)

    return SanitizedContent(
        text=working,
        quality_score=quality,
        math_blocks_preserved=0,
        code_blocks_preserved=0,
        warnings=warnings,
        site_type="pdf2html",
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Generic HTML sanitizer
# ---------------------------------------------------------------------------

def sanitize_general_html(raw_html: str) -> SanitizedContent:
    """Generic cleanup for arbitrary HTML pages."""
    warnings: list[str] = []

    working = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    working = _strip_html_tags(working)

    for pattern in _NAV_CRUFT_PATTERNS:
        working = pattern.sub("", working)

    working = _join_split_paragraphs(working)

    working = _TRAILING_WS_RE.sub("", working)
    working = _MULTI_BLANK_RE.sub("\n\n", working)
    working = working.strip()

    quality = _compute_quality_score(working)

    return SanitizedContent(
        text=working,
        quality_score=quality,
        math_blocks_preserved=0,
        code_blocks_preserved=0,
        warnings=warnings,
        site_type="generic",
    )


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def sanitize(raw_html: str, site_type: str = "") -> SanitizedContent:
    """Auto-detect site type and run appropriate sanitizer."""
    if not site_type:
        site_type = detect_site_type(raw_html)

    if site_type == "sphinx_mathjax":
        return sanitize_sphinx_mathjax(raw_html)
    elif site_type == "pdf2html":
        return sanitize_pdf2html(raw_html)
    else:
        return sanitize_general_html(raw_html)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _join_split_paragraphs(text: str) -> str:
    """Re-join lines that were split mid-sentence."""
    lines = text.split("\n")
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        if not stripped:
            merged.append("")
            i += 1
            continue

        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line:
                break
            if next_line.startswith("$") or next_line.startswith("```"):
                break
            if stripped.endswith((".", "!", "?", ":", ";", '",', '".')):
                break
            if next_line[0].isupper() and not stripped.endswith(","):
                break
            if next_line.startswith(("- ", "* ", "• ")):
                break

            stripped = stripped + " " + next_line
            i += 1

        merged.append(stripped)
        i += 1

    return "\n".join(merged)


_MATH_SYMBOL_RE = re.compile(r"^[∈∀∃→←↔≤≥≠≈∝∞±∓∑∏∫∂∇⊂⊃⊆⊇∩∪∧∨¬⊕⊗=+\-×÷<>]+$")
_SUBSCRIPT_RE = re.compile(r"^[A-Za-z]$")
_INDEX_RE = re.compile(r"^[\d,]+$")


def _merge_math_fragments(text: str) -> str:
    """Best-effort merge of scattered single-char math lines (pdf2htmlEX)."""
    lines = text.split("\n")
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            merged.append("")
            i += 1
            continue

        if len(line) <= 2 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            if _SUBSCRIPT_RE.match(line) and _INDEX_RE.match(next_line):
                merged.append(f"{line}_{{{next_line}}}")
                i += 2
                continue

            if _MATH_SYMBOL_RE.match(line) and next_line:
                if merged and merged[-1]:
                    merged[-1] = merged[-1] + " " + line
                else:
                    merged.append(line)
                i += 1
                continue

        merged.append(lines[i])
        i += 1

    return "\n".join(merged)


def _compute_quality_score(text: str) -> float:
    """Compute content quality score (0.0-1.0)."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0

    long_lines = sum(1 for l in lines if len(l) >= 40)
    long_ratio = long_lines / len(lines) if lines else 0

    total_words = sum(len(l.split()) for l in lines)
    avg_words = total_words / len(lines) if lines else 0
    word_score = min(avg_words / 12.0, 1.0)

    single_char_lines = sum(1 for l in lines if len(l.strip()) <= 2)
    single_ratio = single_char_lines / len(lines) if lines else 0
    single_penalty = max(0, 1.0 - single_ratio * 10)

    math_lines = sum(1 for l in lines if "$" in l)
    math_bonus = min(math_lines / max(len(lines), 1) * 0.5, 0.1)

    score = (long_ratio * 0.35 + word_score * 0.35 + single_penalty * 0.30) + math_bonus
    return min(max(score, 0.0), 1.0)
