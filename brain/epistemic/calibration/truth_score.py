"""Truth score: weighted composite of domain calibration scores with maturity tracking.

A truth score from mature data is fundamentally different from one with half the
domains provisional. The TruthScoreReport exposes maturity metadata so Layers 7-10
can weight their trust appropriately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from epistemic.calibration.domain_calibrator import DomainScore, ALL_DOMAINS

TRUTH_SCORE_WEIGHTS: dict[str, float] = {
    # Spatial confidence now contributes directly to composite truth.
    "retrieval":  0.18,
    "reasoning":  0.18,
    "epistemic":  0.18,
    "autonomy":   0.15,
    "prediction": 0.10,
    "confidence": 0.10,
    "salience":   0.03,
    "skill":      0.02,
    "spatial_position": 0.02,
    "spatial_motion": 0.02,
    "spatial_relation": 0.02,
}

PROVISIONAL_THRESHOLD = 4  # more than this many provisional domains -> score is None


@dataclass
class TruthScoreReport:
    truth_score: float | None
    maturity: float
    provisional_count: int
    data_coverage: dict[str, bool]
    domain_scores: dict[str, float]
    domain_provisional: dict[str, bool]
    timestamp: float = 0.0


class TruthScoreCalculator:
    """Produces TruthScoreReport from domain scores."""

    def compute(self, domain_scores: dict[str, DomainScore], timestamp: float = 0.0) -> TruthScoreReport:
        scores: dict[str, float] = {}
        provisional: dict[str, bool] = {}
        coverage: dict[str, bool] = {}

        for domain in ALL_DOMAINS:
            ds = domain_scores.get(domain)
            if ds is None:
                scores[domain] = 0.5
                provisional[domain] = True
                coverage[domain] = False
            else:
                scores[domain] = ds.score
                provisional[domain] = ds.provisional
                coverage[domain] = not ds.provisional

        provisional_count = sum(1 for v in provisional.values() if v)
        maturity = 1.0 - (provisional_count / len(ALL_DOMAINS))

        if provisional_count > PROVISIONAL_THRESHOLD:
            return TruthScoreReport(
                truth_score=None,
                maturity=maturity,
                provisional_count=provisional_count,
                data_coverage=coverage,
                domain_scores=scores,
                domain_provisional=provisional,
                timestamp=timestamp,
            )

        weighted_sum = 0.0
        weight_total = 0.0
        for domain, weight in TRUTH_SCORE_WEIGHTS.items():
            weighted_sum += weight * scores.get(domain, 0.5)
            weight_total += weight

        truth = weighted_sum / weight_total if weight_total > 0 else 0.5

        return TruthScoreReport(
            truth_score=round(max(0.0, min(1.0, truth)), 4),
            maturity=round(maturity, 3),
            provisional_count=provisional_count,
            data_coverage=coverage,
            domain_scores={k: round(v, 4) for k, v in scores.items()},
            domain_provisional=provisional,
            timestamp=timestamp,
        )
