from __future__ import annotations

from chatuskoti_evals.config import DetectorConfig
from chatuskoti_evals.models import Resolution, RunMetrics, RunScore


def resolve_vec3(run_score: RunScore, cfg: DetectorConfig) -> Resolution:
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
    mean_metric = sum(item.primary_metric for item in candidate_metrics) / len(candidate_metrics)
    metric_delta = mean_metric - baseline_metrics.primary_metric
    if metric_delta > cfg.binary_metric_threshold:
        return Resolution("adopt", f"mean metric delta {metric_delta:.4f} exceeds binary threshold {cfg.binary_metric_threshold:.4f}")
    return Resolution("reject", f"mean metric delta {metric_delta:.4f} does not exceed binary threshold {cfg.binary_metric_threshold:.4f}")
