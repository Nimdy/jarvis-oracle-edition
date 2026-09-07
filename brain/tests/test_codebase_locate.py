"""CODEBASE locate + mouth leash.

The AST index is the eyes. Qwen may only revoice the lookup. No voice→handle_transcription
allowlist. No HRR. Playbook G20/G21 close on the mouth: stats or a ranked hit/abstain,
never an invented subsystem.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from tools.codebase_tool import CodebaseIndex


G20 = "Can you read your own code?"
G21 = "Where is the function that handles my voice?"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tiny_index(tmp_path, monkeypatch):
    monkeypatch.setattr(CodebaseIndex, "_HASH_STORE_PATH", tmp_path / "hashes.json")
    root = tmp_path / "brain"
    root.mkdir()
    _write(
        root / "conversation_handler.py",
        '''
def handle_transcription(text):
    """Mind-path entry for spoken turns and TAP."""
    pass

def _normalize_number_set(raw):
    """Normalize a set of numeric strings so 0.989 and 98.9 match."""
    pass
''',
    )
    _write(root / "autonomy" / "__init__.py", "")
    _write(
        root / "autonomy" / "eval_harness.py",
        '''
class ReplayResult:
    """Result of replaying a single episode against a scoring function."""
    pass

def replay_episodes(episodes, scorer):
    """Replay recorded episodes against a scoring function."""
    pass
''',
    )
    _write(root / "acquisition" / "__init__.py", "")
    _write(
        root / "acquisition" / "job.py",
        '''
class PluginCodeBundle:
    """Bundle of generated plugin code."""
    pass
''',
    )
    _write(root / "cognition" / "__init__.py", "")
    _write(root / "cognition" / "self_view" / "__init__.py", "")
    _write(
        root / "cognition" / "self_view" / "voice_seed.py",
        '''
TEACHER = "native_voice"

def capture_teacher_pair():
    """Record a grounded-to-voiced teacher pair."""
    pass
''',
    )
    _write(
        root / "cognition" / "self_view" / "revoice.py",
        '''
async def revoice_self_view(grounded_text):
    """Speak the grounded self-view in a warm voice."""
    pass
''',
    )
    _write(
        root / "perception" / "__init__.py",
        "",
    )
    _write(
        root / "perception" / "identity_fusion.py",
        '''
def _should_smooth_voice_gap(self):
    """Hold identity across a short voice drop."""
    pass

def check_specialists_empty_where_data_exists():
    """Dashboard probe leftover — must not steal 'Where is X defined'."""
    pass
''',
    )
    _write(
        root / "widgets.py",
        '''
class Widget:
    """The real Widget type."""
    pass

class Decoy:
    """Mentions Widget in a docstring only, scoring function leftover."""
    pass
''',
    )
    idx = CodebaseIndex(root=root)
    idx.build()
    return idx


def test_g21_does_not_pick_scoring_function_docstring(tmp_path, monkeypatch):
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query(G21).lower()
    assert "replayresult" not in out
    assert "eval_harness" not in out
    assert "autonomy" not in out
    assert "scoring function" not in out


def test_g21_does_not_allowlist_handle_transcription(tmp_path, monkeypatch):
    """Plastic retrieve — do not hardwire voice → handle_transcription."""
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query(G21).lower()
    assert "handle_transcription" not in out


def test_g21_abstains_when_object_is_not_a_symbol_name(tmp_path, monkeypatch):
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query(G21).lower()
    assert "don't have a clean location" in out or "do not have a clean location" in out
    assert "voice" in out
    assert "modules" in out and "symbols" in out
    assert "voice_gap" not in out
    assert "identity_fusion" not in out


def test_locate_exact_name(tmp_path, monkeypatch):
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query("Where is handle_transcription?").lower()
    assert "handle_transcription" in out
    assert "conversation_handler" in out


def test_stt_split_snake_case_locates(tmp_path, monkeypatch):
    """Spoken 'handle transcription' is still handle_transcription in the index."""
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query("Jarvis, where is handle transcription?").lower()
    assert "handle_transcription" in out
    assert "conversation_handler" in out
    assert "autonomy" not in out


def test_stt_join_does_not_steal_household_where_is(tmp_path, monkeypatch):
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query("Where is Tonya?").lower()
    assert "handle_transcription" not in out
    assert "autonomy" not in out


def test_name_match_beats_docstring(tmp_path, monkeypatch):
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query("Where is Widget defined?").lower()
    assert "widget" in out
    assert "decoy" not in out
    assert "where_data" not in out
    assert "identity_fusion" not in out


def test_g20_describes_the_index_not_a_keyword_bag(tmp_path, monkeypatch):
    idx = _tiny_index(tmp_path, monkeypatch)
    out = idx.answer_query(G20).lower()
    assert "plugincodebundle" not in out
    assert "index" in out
    assert "modules" in out
    assert "symbols" in out


def test_g21_is_not_system_explanation_regex():
    src = Path(__file__).resolve().parents[1].joinpath("conversation_handler.py").read_text(
        encoding="utf-8",
    )
    start = src.find("_SYSTEM_EXPLANATION_RE = re.compile(")
    assert start != -1
    chunk = src[start:start + 900]
    assert "how do you work" in chunk
    # Locate questions stay CODEBASE retrieve — do not widen this regex onto G21.
    assert G21.lower() not in chunk.lower()
    # The compiled live regex (same text as the handler) must not match G21.
    m = re.search(r'_SYSTEM_EXPLANATION_RE = re.compile\(\s*(r?".*?"|r\'.*?\')', chunk, re.S)
    assert m, chunk[:200]
    # Fall back: G21 is a where-is-function question, not "how do you work".
    assert "where is the function" not in chunk.lower()


class _Client:
    def __init__(self, resp):
        self.resp = resp

    async def chat(self, messages, system_prompt=None, model_override=None):
        return self.resp


def test_revoice_rejects_invented_path():
    from cognition.self_view.revoice import revoice_code_answer

    grounded = "I found handle_transcription — a function in conversation_handler.py at line 12."
    bad = "The voice pipeline lives in autonomy/eval_harness.py inside ReplayResult."
    text, meta = asyncio.run(revoice_code_answer(grounded, _Client(bad)))
    assert text == grounded
    assert meta["used_revoice"] is False
    assert "path" in meta["reason"] or "invented" in meta["reason"]


def test_revoice_rejects_invented_fqn():
    from cognition.self_view.revoice import revoice_code_answer

    grounded = "I don't have a clean location for 'voice'. Name a function or file."
    bad = "That's in the autonomy.voice_pipeline module, obviously."
    text, meta = asyncio.run(revoice_code_answer(grounded, _Client(bad)))
    assert text == grounded
    assert meta["used_revoice"] is False


def test_revoice_faithful_pass():
    from cognition.self_view.revoice import revoice_code_answer

    grounded = "I found handle_transcription — a function in conversation_handler.py at line 12."
    ok = "Your voice lands in handle_transcription — conversation_handler.py, line 12."
    text, meta = asyncio.run(revoice_code_answer(grounded, _Client(ok)))
    assert text == ok
    assert meta["used_revoice"] is True


def test_speak_helper_falls_back_when_qwen_invents():
    from cognition.self_view.revoice import revoice_code_answer

    grounded = "I found handle_transcription — a function in conversation_handler.py at line 12."
    spoken, meta = asyncio.run(
        revoice_code_answer(
            grounded, _Client("It's in the autonomy subsystem with acquisition_orchestrator.")
        )
    )
    assert spoken == grounded
    assert meta["used_revoice"] is False


def test_codebase_handler_does_not_free_narrate():
    src = Path(__file__).resolve().parents[1].joinpath("conversation_handler.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("elif routing.tool == ToolType.CODEBASE:")
    assert idx != -1
    body = src[idx:idx + 4500]
    assert "_revoice_or_ground_codebase" in body
    assert "respond_stream" not in body
