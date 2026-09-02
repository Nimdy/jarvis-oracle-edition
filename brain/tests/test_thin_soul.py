"""Thin soul pass — native tool-route polish. No LLM. Fail-closed."""
from __future__ import annotations

import pytest

from personality.thin_soul import thin_soul_native, _strip_markdown_markers


@pytest.fixture(autouse=True)
def _no_soul_log(monkeypatch):
    from personality import thin_soul as ts
    monkeypatch.setattr(ts, "_log_soul_shadow", lambda **_k: None)


def test_strips_markdown_markers():
    out = _strip_markdown_markers("**In summary:** you are fine.")
    assert "**" not in out
    assert "In summary" in out
    assert "*" not in _strip_markdown_markers("What makes you Jarvis?**")


def test_pass_through_plain_status():
    src = "I'm standing by right now, nothing actively processing."
    assert thin_soul_native(src, route="STATUS") == src


def test_strips_markdown_on_native():
    src = "Here's what I remember about that. **Tonya** is family."
    out = thin_soul_native(src, route="MEMORY")
    assert "**" not in out
    assert "Tonya" in out
    assert "family" in out


def test_does_not_invent_numbers():
    src = "My memory cortex has collected 12 training pairs."
    out = thin_soul_native(src, route="STATUS")
    assert "12" in out
    assert "13" not in out


def test_empty_stays_empty():
    assert thin_soul_native("", route="STATUS") == ""
    assert thin_soul_native("   ", route="MEMORY").strip() == ""


def test_unqualified_claim_introduced_is_rejected(monkeypatch):
    from personality import thin_soul as ts

    def _fake_strip(text: str) -> str:
        return text + " I am conscious."

    monkeypatch.setattr(ts, "_strip_markdown_markers", _fake_strip)
    src = "I'm in companion mode."
    assert thin_soul_native(src, route="STATUS") == src


def test_broadcast_sync_applies_thin_soul_before_gate():
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent.joinpath("conversation_handler.py").read_text()
    start = src.find("async def _broadcast_chunk_sync")
    nxt = src.find("\n    async def ", start + 1)
    body = src[start:nxt]
    assert "thin_soul_native" in body
    assert body.find("thin_soul_native") < body.find("_gate_text(text_str)")


def test_llm_send_sentence_does_not_thin_soul():
    """LLM path already logs soul_dims in response.py. Do not double-pass."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent.joinpath("conversation_handler.py").read_text()
    start = src.find("async def _send_sentence")
    nxt = src.find("\n    async def ", start + 1)
    body = src[start:nxt]
    assert "thin_soul_native" not in body
