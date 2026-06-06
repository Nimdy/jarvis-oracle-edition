"""Fidelity #11: VLM scene-parse must not confabulate negated/partial-word objects.

A raw `obj in description` test credited "no laptop visible" as a present laptop and
"category" as a "cat", feeding phantom detections into the scene tracker + object
memory (corrupting the world model). _object_present_in_description fixes both.
"""
from __future__ import annotations

import pytest

from perception_orchestrator import _object_present_in_description as present


@pytest.mark.parametrize("obj,desc,expected", [
    ("laptop", "there is no laptop visible", False),       # the bug
    ("laptop", "a laptop sits on the desk", True),
    ("laptop", "no people, but a laptop is here", True),   # clause separation
    ("people", "no people, but a laptop is here", False),
    ("phone", "i don't see a phone", False),
    ("phone", "there is a phone and a cup", True),
    ("laptop", "the laptop is missing", False),            # post-absence marker
    ("cat", "a category of items", False),                 # partial-word
    ("cup", "a cupboard in the corner", False),            # partial-word
    ("cup", "a cup of coffee", True),
    ("laptop", "it is not a laptop but a tablet", False),  # 'but' clause
    ("dog", "an empty room", False),
])
def test_object_presence(obj, desc, expected):
    assert present(obj, desc) is expected
