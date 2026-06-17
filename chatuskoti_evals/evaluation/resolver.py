from __future__ import annotations

from dataclasses import replace

from chatuskoti_evals.core.config import DetectorConfig
from chatuskoti_evals.core.models import Resolution, RunMetrics, RunScore


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
    """Map T/R/V to action: adopt/hold/reframe/rollback/keep_going based on config thresholds."""
    t = run_score.mean.truthness
    r = run_score.mean.reliability
    v = run_score.mean.validity

    if cfg.enable_spread_gate and run_score.spread > cfg.max_spread:
        return Resolution("keep_going", f"seed spread {run_score.spread:.3f} exceeds maximum {cfg.max_spread:.3f}")
    if cfg.enable_reliability and r < cfg.reliability_threshold and t < -cfg.adopt_truth_threshold:
        return Resolution(
            "rollback",
            f"truthness {t:.3f} is below -{cfg.adopt_truth_threshold:.3f} and reliability {r:.3f} indicates internal damage",
        )
    if cfg.enable_validity and v < cfg.validity_threshold:
        return Resolution(
            "reframe",
            f"validity {v:.3f} is below threshold {cfg.validity_threshold:.3f}; apparent gain is not decision-ready",
        )
    if cfg.enable_reliability and r < cfg.reliability_threshold:
        return Resolution(
            "hold",
            f"reliability {r:.3f} is below threshold {cfg.reliability_threshold:.3f}; result is too unstable to merge",
        )
    if t > cfg.adopt_truth_threshold:
        return Resolution("adopt", f"truthness {t:.3f} exceeds adopt threshold {cfg.adopt_truth_threshold:.3f}")
    return Resolution("reject", f"truthness {t:.3f} does not exceed adopt threshold {cfg.adopt_truth_threshold:.3f}")


def resolve_binary(candidate_metrics: list[RunMetrics], baseline_metrics: RunMetrics, cfg: DetectorConfig) -> Resolution:
    """Scalar-only resolver: adopt if mean metric delta exceeds binary_metric_threshold, else reject."""
    mean_metric = sum(item.primary_metric for item in candidate_metrics) / len(candidate_metrics)
    metric_delta = mean_metric - baseline_metrics.primary_metric
    if metric_delta > cfg.binary_metric_threshold:
        return Resolution("adopt", f"mean metric delta {metric_delta:.4f} exceeds binary threshold {cfg.binary_metric_threshold:.4f}")
    return Resolution("reject", f"mean metric delta {metric_delta:.4f} does not exceed binary threshold {cfg.binary_metric_threshold:.4f}")
