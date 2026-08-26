"""Lived 2026-08-25: 'that is wrong' did not classify as correction (regex was that's)."""
from autonomy.friction_miner import FrictionMiner


def test_that_is_wrong_is_correction():
    miner = FrictionMiner()
    spoken = (
        "Jarvis that is wrong. I have my thumbs tucked in and I only have four "
        "fingers on each hand. Check again"
    )
    assert miner._classify(spoken, "ten fingers", None) == "correction"
    assert miner._classify("that's wrong", "blue", None) == "correction"
    assert miner._classify("that was incorrect", "ten", None) == "correction"
