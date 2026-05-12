"""Domain calibrator: computes per-domain calibration scores from raw signals.

Each of the 8 domains has a scoring function mapping subsystem signals to [0.0, 1.0].
When a signal is None (not yet available), the domain uses 0.5 default and is marked provisional.
"""
from __future__ import annotations

from dataclasses import dataclass

from epistemic.calibration.signal_collector import CalibrationSnapshot

ALL_DOMAINS = (
    "retrieval", "autonomy", "prediction", "salience",
    "reasoning", "skill", "epistemic", "confidence",
    "spatial_position", "spatial_motion", "spatial_relation",
)


@dataclass
class DomainScore:
    domain: str
    score: float
    provisional: bool


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _score_retrieval(snap: CalibrationSnapshot) -> DomainScore:
    ref = snap.reference_match_rate
    ranker = snap.ranker_success_rate
    lift = snap.lift
    prov = snap.provenance_weighted_success
    if ref is None and ranker is None:
        return DomainScore("retrieval", 0.5, provisional=True)

    score = 0.0
    weights = 0.0
    if ref is not None:
        score += 0.30 * ref
        weights += 0.30
    if ranker is not None:
        score += 0.25 * ranker
        weights += 0.25
    if lift is not None:
        score += 0.20 * _clamp(lift + 0.5)
        weights += 0.20
    if prov is not None:
        score += 0.25 * prov
        weights += 0.25

    return DomainScore("retrieval", _clamp(score / weights if weights > 0 else 0.5), provisional=False)


def _score_autonomy(snap: CalibrationSnapshot) -> DomainScore:
    imp = snap.improvement_rate
    win = snap.overall_win_rate
    conv = snap.research_conversion_rate
    if imp is None and win is None:
        return DomainScore("autonomy", 0.5, provisional=True)

    score = 0.0
    weights = 0.0
    if imp is not None:
        score += 0.35 * imp
        weights += 0.35
    if win is not None:
        score += 0.35 * win
        weights += 0.35
    if conv is not None:
        score += 0.30 * conv
        weights += 0.30

    return DomainScore("autonomy", _clamp(score / weights if weights > 0 else 0.5), provisional=False)


_BEHAVIORAL_MIN_SAMPLES = 20


def _score_prediction(snap: CalibrationSnapshot) -> DomainScore:
    wm_acc = snap.wm_prediction_accuracy
    beh_acc = snap.prediction_accuracy

    has_wm = wm_acc is not None and snap.wm_prediction_count >= 20
    has_beh = beh_acc is not None

    if not has_wm and not has_beh:
        return DomainScore("prediction", 0.5, provisional=True)

    if has_wm and has_beh:
        score = wm_acc * 0.75 + beh_acc * 0.25
    elif has_wm:
        score = wm_acc
    else:
        score = beh_acc

    return DomainScore("prediction", _clamp(score), provisional=False)


def _score_salience(snap: CalibrationSnapshot) -> DomainScore:
    waste = snap.wasted_rate
    werr = snap.weight_error
    derr = snap.decay_error
    useful = snap.useful_rate
    if waste is None and werr is None and derr is None:
        return DomainScore("salience", 0.5, provisional=True)

    error_score = 0.0
    error_weights = 0.0
    if waste is not None:
        error_score += 0.3 * waste
        error_weights += 0.3
    if werr is not None:
        error_score += 0.35 * werr
        error_weights += 0.35
    if derr is not None:
        error_score += 0.25 * derr
        error_weights += 0.25

    inverted = 1.0 - (error_score / error_weights if error_weights > 0 else 0.5)

    if useful is not None:
        inverted = inverted * 0.85 + useful * 0.15

    return DomainScore("salience", _clamp(inverted), provisional=False)


def _score_reasoning(snap: CalibrationSnapshot) -> DomainScore:
    coh = snap.coherence
    con = snap.consistency
    dep = snap.depth
    if coh is None and con is None and dep is None:
        return DomainScore("reasoning", 0.5, provisional=True)

    score = 0.0
    weights = 0.0
    if coh is not None:
        score += 0.40 * coh
        weights += 0.40
    if con is not None:
        score += 0.35 * con
        weights += 0.35
    if dep is not None:
        score += 0.25 * dep
        weights += 0.25

    normalized = score / weights if weights > 0 else 0.5
    cp = snap.correction_penalty
    if cp is not None and cp > 0:
        penalty = min(0.15, cp * 0.5)
        normalized -= penalty

    return DomainScore("reasoning", _clamp(normalized), provisional=False)


def _score_skill(snap: CalibrationSnapshot) -> DomainScore:
    verified = snap.verified_skill_count
    failures = snap.honesty_failures
    if verified == 0 and failures == 0:
        return DomainScore("skill", 0.5, provisional=True)
    denom = max(1, failures + verified * 10)
    return DomainScore("skill", _clamp(1.0 - (failures / denom)), provisional=False)


def _score_epistemic(snap: CalibrationSnapshot) -> DomainScore:
    debt = snap.contradiction_debt
    resolved = snap.resolved_count
    total = snap.total_beliefs
    near_miss = snap.near_miss_rate
    resolution_rate = resolved / max(1, total)
    base = (1.0 - debt) * 0.45 + resolution_rate * 0.30
    if snap.graph_health_score is not None:
        base += snap.graph_health_score * 0.15
    else:
        base += 0.15
    if near_miss is not None and near_miss > 0:
        base -= min(0.10, near_miss * 0.3)
    else:
        base += 0.10
    return DomainScore(
        "epistemic",
        _clamp(base),
        provisional=(total == 0),
    )


def _score_confidence(snap: CalibrationSnapshot) -> DomainScore:
    if snap.brier_score is None:
        return DomainScore("confidence", 0.5, provisional=True)
    base = 1.0 - snap.brier_score
    worst = snap.worst_route_brier
    worst_n = snap.worst_route_sample_count or 0
    if worst is not None and worst_n >= 10 and worst > snap.brier_score + 0.05:
        base -= 0.15 * (worst - snap.brier_score)
    oc = snap.overconfidence_error
    if oc is not None and oc > 0.05:
        base -= 0.10 * oc
    ece = snap.ece
    if ece is not None and ece > 0.10:
        base -= 0.08 * (ece - 0.10)
    return DomainScore("confidence", _clamp(base), provisional=False)


def _score_spatial_position(snap: CalibrationSnapshot) -> DomainScore:
    if not snap.spatial_calibration_valid and snap.spatial_anchor_count == 0:
        return DomainScore("spatial_position", 0.5, provisional=True)
    score = 0.0
    weights = 0.0
    if snap.spatial_calibration_valid:
        score += 0.40 * 1.0
        weights += 0.40
    else:
        score += 0.40 * 0.2
        weights += 0.40
    if snap.spatial_anchor_count > 0:
        score += 0.30 * min(1.0, snap.spatial_anchor_count / 5)
        weights += 0.30
    if snap.spatial_stable_tracks > 0:
        score += 0.30 * min(1.0, snap.spatial_stable_tracks / 8)
        weights += 0.30
    return DomainScore("spatial_position", _clamp(score / weights if weights > 0 else 0.5), provisional=False)


def _score_spatial_motion(snap: CalibrationSnapshot) -> DomainScore:
    promoted = snap.spatial_promoted_deltas
    rejected = snap.spatial_rejected_promotions
    total = promoted + rejected
    if total == 0:
        return DomainScore("spatial_motion", 0.5, provisional=True)
    accuracy = promoted / total if total > 0 else 0.0
    return DomainScore("spatial_motion", _clamp(accuracy), provisional=(total < 10))


def _score_spatial_relation(snap: CalibrationSnapshot) -> DomainScore:
    contradictions = snap.spatial_contradiction_count
    anchors = snap.spatial_anchor_count
    tracks = snap.spatial_stable_tracks
    if anchors == 0 and tracks == 0:
        return DomainScore("spatial_relation", 0.5, provisional=True)
    health = 1.0
    if contradictions > 0:
        health -= min(0.4, contradictions * 0.1)
    if snap.spatial_anchor_drift_score is not None:
        health -= min(0.3, snap.spatial_anchor_drift_score)
    return DomainScore("spatial_relation", _clamp(health), provisional=(anchors < 2))


_SCORERS = {
    "retrieval": _score_retrieval,
    "autonomy": _score_autonomy,
    "prediction": _score_prediction,
    "salience": _score_salience,
    "reasoning": _score_reasoning,
    "skill": _score_skill,
    "epistemic": _score_epistemic,
    "confidence": _score_confidence,
    "spatial_position": _score_spatial_position,
    "spatial_motion": _score_spatial_motion,
    "spatial_relation": _score_spatial_relation,
}


class DomainCalibrator:
    """Computes per-domain calibration scores from a CalibrationSnapshot."""

    def score_all(self, snap: CalibrationSnapshot) -> dict[str, DomainScore]:
        return {domain: scorer(snap) for domain, scorer in _SCORERS.items()}
