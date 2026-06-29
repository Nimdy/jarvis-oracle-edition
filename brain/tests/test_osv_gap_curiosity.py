"""OSV Phase D — gap-curiosity invariants (earn-don't-declare + memory-safety).

Locks the guarantees David required: SHADOW/gated at default autonomy, enqueues only when earned,
deduped (each gap researched at most once), bounded, and writes ZERO memories.
"""

from cognition.self_view.gather import architecture_from_registry
from cognition.self_view.synthesizer import SelfViewSynthesizer
from autonomy.osv_gap_curiosity import (
    OSVGapCuriosity, GAP_CURIOSITY_MIN_LEVEL, MAX_PROPOSALS, MAX_SHADOW_LOG,
)


def _model():
    return SelfViewSynthesizer().synthesize(
        {"architecture": architecture_from_registry()}, now=0.0).to_dict()


class _FakeOrch:
    """Records enqueue calls; rejects a repeated question like the real _question_already_queued."""

    def __init__(self):
        self.calls = []
        self._seen = set()

    def enqueue(self, intent):
        self.calls.append(intent.question)
        if intent.question in self._seen:
            return False
        self._seen.add(intent.question)
        return True


def test_proposals_bounded_deduped_and_skip_scoreboard():
    props = OSVGapCuriosity().proposals(_model())
    assert len(props) <= MAX_PROPOSALS
    assert all(not p["area"].startswith("scoreboard") for p in props)  # need reps, not research
    assert len({p["key"] for p in props}) == len(props)               # deduped
    assert props == sorted(props, key=lambda p: -p["priority"])        # architecture-first


def test_gated_at_default_autonomy_is_pure_shadow():
    gc, orch = OSVGapCuriosity(), _FakeOrch()
    summary = gc.feed(orch, autonomy_level=1, model=_model())
    assert summary["gated"] is True
    assert summary["enqueued"] == 0
    assert orch.calls == []  # the gate is upstream of enqueue — zero behavior, zero memory risk


def test_enqueues_when_earned_then_dedups():
    gc, orch = OSVGapCuriosity(), _FakeOrch()
    first = gc.feed(orch, autonomy_level=GAP_CURIOSITY_MIN_LEVEL, model=_model())
    assert first["gated"] is False and first["enqueued"] > 0
    again = gc.feed(orch, autonomy_level=GAP_CURIOSITY_MIN_LEVEL, model=_model())
    assert again["enqueued"] == 0  # each gap researched at most once


def test_shadow_log_is_bounded():
    gc, orch = OSVGapCuriosity(), _FakeOrch()
    for _ in range(MAX_SHADOW_LOG + 15):
        gc.feed(orch, autonomy_level=1, model=_model())
    assert len(gc.shadow_state()["recent"]) <= MAX_SHADOW_LOG


def test_handles_empty_model():
    gc = OSVGapCuriosity()
    assert gc.proposals({}) == []
    assert gc.feed(_FakeOrch(), autonomy_level=5, model={})["enqueued"] == 0
