"""Layer 8: Cognitive Quarantine — Shadow + Active-Lite.

Scores suspicious inputs and anomalies across 5 categories.
Shadow layer: score, log, display.
Active-lite layer: pressure-based friction on promotions + soft memory gating.
"""

from epistemic.quarantine.scorer import QuarantineScorer, QuarantineSignal
from epistemic.quarantine.log import QuarantineLog
from epistemic.quarantine.pressure import (
    CATEGORY_POLICY,
    QUARANTINE_SUSPECT_TAG,
    QuarantinePressure,
    PressureState,
    get_quarantine_pressure,
)

__all__ = [
    "QuarantineScorer",
    "QuarantineSignal",
    "QuarantineLog",
    "QuarantinePressure",
    "PressureState",
    "CATEGORY_POLICY",
    "QUARANTINE_SUSPECT_TAG",
    "get_quarantine_pressure",
]
