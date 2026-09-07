"""#83 lab session surface lives on existing /learning — not a new v2 page."""
from pathlib import Path

_LEARNING = Path(__file__).resolve().parent.parent / "dashboard" / "static" / "learning.html"


def test_learning_page_lists_unvalidated_learning_and_last_correction() -> None:
    html = _LEARNING.read_text(encoding="utf-8")
    assert "id=\"lab-session\"" in html
    assert "id=\"lab-last\"" in html
    assert "recent_answered" in html
    assert "Jarvis, GOLDEN COMMAND LEARN SKILL" in html
    assert "Jarvis, GOLDEN COMMAND UNVALIDATED LEARNING" in html
    assert "lab.html" not in html
