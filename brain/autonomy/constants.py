"""Shared constants for the autonomy subsystem.

Defined once here, imported by policy_memory, opportunity_scorer, orchestrator,
and any future module that needs to agree on what "meaningful" means.
"""

MIN_MEANINGFUL_DELTA: float = 0.02
"""Minimum net attribution to count as a 'win' in policy memory.
Below this, the outcome is treated as noise, not signal."""

WARMUP_PERIOD_S: float = 1800.0
"""Seconds after session start during which outcomes are marked as warmup.
Warmup outcomes are stored but excluded from priors, preventing cold-start
noise from poisoning policy memory. 30 minutes is enough for baselines
to stabilize, Ollama models to load, and initial transients to settle."""

MIN_DAYS_FOR_TOD_BASELINE: int = 3
"""Minimum days of continuous runtime before time-of-day metric baselines
influence counterfactual estimation (Phase 2: Temporal Credit).
Requires at least MIN_SAMPLES_PER_BUCKET samples per hour bucket."""
