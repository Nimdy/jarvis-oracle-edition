"""Layer 10: Soul Integrity Index.

Single composite metric summarizing cognitive health across all epistemic layers.
Aggregates: memory coherence, belief contradiction rate, identity separation
accuracy, skill verification success, ethical alignment stability, truth
calibration, belief graph health, quarantine signal rate, autonomy effectiveness,
and reflective audit scores.

When the index drops below a repair threshold, it emits a repair trigger event
that downstream systems can act on (e.g., trigger dream cycle, pause mutations,
increase audit frequency).
"""

from epistemic.soul_integrity.index import SoulIntegrityIndex, IntegrityReport

__all__ = ["SoulIntegrityIndex", "IntegrityReport"]
