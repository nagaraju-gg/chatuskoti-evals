from __future__ import annotations

from dataclasses import replace
from typing import Literal

from chatuskoti_evals.core.config import DetectorConfig
from chatuskoti_evals.core.models import Resolution, RunMetrics, RunScore

InstrumentState = Literal["partial", "valid", "unreliable", "invalid", "unreliable_invalid"]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (p / 100.0) * (len(s) - 1)
    f = int(k)
    c = k - f
    if f + 1 < len(s):
        return s[f] * (1 - c) + s[f + 1] * c
    return s[f]


def adaptive_detector_config(
    history_scores: list[RunScore],
    base_cfg: DetectorConfig | None = None,
) -> DetectorConfig:
    """Adapt detector thresholds based on history percentiles: adopt_truth at P75, others at P25/P90."""
    if base_cfg is None:
        base_cfg = DetectorConfig()
    if len(history_scores) < 3:
        return base_cfg

    truthnesses = [s.mean.truthness for s in history_scores]
    reliabilities = [s.mean.reliability for s in history_scores]
    validities = [s.mean.validity for s in history_scores]
    spreads = [s.spread for s in history_scores]

    return replace(
        base_cfg,
        adopt_truth_threshold=round(_percentile(truthnesses, 75), 4),
        reliability_threshold=round(_percentile(reliabilities, 25), 4),
        validity_threshold=round(_percentile(validities, 25), 4),
        max_spread=round(_percentile(spreads, 90), 4),
    )


def resolve_vec3(run_score: RunScore, cfg: DetectorConfig) -> Resolution:
    """Map T/R/V to action using staged instrument classification followed by truth logic."""
    t = run_score.mean.truthness

    if cfg.enable_spread_gate and run_score.spread > cfg.max_spread:
        return Resolution("keep_going", f"seed spread {run_score.spread:.3f} exceeds maximum {cfg.max_spread:.3f}")

    if cfg.min_magnitude > 0 and run_score.mag < cfg.min_magnitude:
        return Resolution("keep_going", f"magnitude {run_score.mag:.3f} is below minimum {cfg.min_magnitude:.3f}")

    instrument_state = classify_instrument_state(run_score, cfg)
    if instrument_state == "partial":
        undefined_axes = ", ".join(
            axis
            for axis, status in axis_statuses(run_score).items()
            if status == "undefined"
        )
        return Resolution("keep_going", f"partial Vec3 state; undefined axes: {undefined_axes or 'unknown'}")

    return resolve_truth_condition(t, instrument_state, run_score, cfg)


def classify_instrument_state(run_score: RunScore, cfg: DetectorConfig) -> InstrumentState:
    """First resolver pass: classify instrument health from R/V before applying T logic."""
    statuses = axis_statuses(run_score)
    if statuses.get("reliability") == "undefined" or statuses.get("validity") == "undefined":
        return "partial"

    r = run_score.mean.reliability
    v = run_score.mean.validity
    reliability_low = cfg.enable_reliability and r < cfg.reliability_threshold
    validity_low = cfg.enable_validity and v < cfg.validity_threshold

    if reliability_low and validity_low:
        return "unreliable_invalid"
    if validity_low:
        return "invalid"
    if reliability_low:
        return "unreliable"
    return "valid"


def resolve_truth_condition(
    t: float,
    instrument_state: InstrumentState,
    run_score: RunScore,
    cfg: DetectorConfig,
) -> Resolution:
    """Second resolver pass: apply T-conditional action logic within an instrument state."""
    r = run_score.mean.reliability
    v = run_score.mean.validity

    if instrument_state in {"unreliable", "unreliable_invalid"} and t < -cfg.adopt_truth_threshold:
        return Resolution(
            "rollback",
            f"truthness {t:.3f} is below -{cfg.adopt_truth_threshold:.3f} and reliability {r:.3f} indicates internal damage",
        )
    if instrument_state in {"invalid", "unreliable_invalid"}:
        return Resolution(
            "reframe",
            f"validity {v:.3f} is below threshold {cfg.validity_threshold:.3f}; apparent gain is not decision-ready",
        )
    if instrument_state == "unreliable":
        return Resolution(
            "hold",
            f"reliability {r:.3f} is below threshold {cfg.reliability_threshold:.3f}; result is too unstable to merge",
        )
    if t > cfg.adopt_truth_threshold:
        return Resolution("adopt", f"truthness {t:.3f} exceeds adopt threshold {cfg.adopt_truth_threshold:.3f}")
    return Resolution("reject", f"truthness {t:.3f} does not exceed adopt threshold {cfg.adopt_truth_threshold:.3f}")


def axis_statuses(run_score: RunScore) -> dict[str, str]:
    if run_score.axis_state is None:
        return {"truthness": "measured", "reliability": "measured", "validity": "measured"}
    return run_score.axis_state.status_map()


def resolve_binary(candidate_metrics: list[RunMetrics], baseline_metrics: RunMetrics, cfg: DetectorConfig) -> Resolution:
    """Scalar-only resolver: adopt if mean metric delta exceeds binary_metric_threshold, else reject."""
    mean_metric = sum(item.primary_metric for item in candidate_metrics) / len(candidate_metrics)
    metric_delta = mean_metric - baseline_metrics.primary_metric
    if metric_delta > cfg.binary_metric_threshold:
        return Resolution("adopt", f"mean metric delta {metric_delta:.4f} exceeds binary threshold {cfg.binary_metric_threshold:.4f}")
    return Resolution("reject", f"mean metric delta {metric_delta:.4f} does not exceed binary threshold {cfg.binary_metric_threshold:.4f}")
