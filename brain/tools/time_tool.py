"""Time tool — returns current datetime."""

from __future__ import annotations

import time


def get_current_time() -> str:
    return time.strftime("It is %I:%M %p on %A, %B %d, %Y.")
