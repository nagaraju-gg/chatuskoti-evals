from __future__ import annotations

import math
from statistics import mean, stdev

from chatuskoti_evals.config import DetectorConfig
from chatuskoti_evals.models import RunMetrics, RunScore, SeedScore, Vec3


def score_run_metrics(
    candidate_metrics: list[RunMetrics],
    baseline_metrics: RunMetrics,
    cfg: DetectorConfig,
) -> tuple[RunScore, list[SeedScore]]:
    if not candidate_metrics:
        raise ValueError("candidate_metrics must not be empty")

    per_seed_scores = [score_single_seed(run, baseline_metrics, cfg) for run in candidate_metrics]

    truthnesses = [seed.vec3.truthness for seed in per_seed_scores]
    reliabilities = [seed.vec3.reliability for seed in per_seed_scores]
    validities = [seed.vec3.validity for seed in per_seed_scores]
    metric_deltas = [float(seed.raw_detectors["metric_delta"]) for seed in per_seed_scores]

    seed_variance_score = 1.0
    if cfg.enable_reliability and len(metric_deltas) > 1:
        seed_variance_score = bounded_inverse(abs(stdev(metric_deltas)), cfg.seed_metric_std_scale)

    mean_reliability = mean(reliabilities)
    if cfg.enable_reliability:
        mean_reliability = clamp(0.65 * mean_reliability + 0.35 * seed_variance_score)

    mean_validity = mean(validities)
    if not cfg.enable_validity:
        mean_validity = 0.75

    mean_vec = Vec3(
        truthness=round(mean(truthnesses), 5),
        reliability=round(mean_reliability, 5),
        validity=round(mean_validity, 5),
    )
    std_vec = Vec3(
        truthness=round(stdev(truthnesses), 5) if len(truthnesses) > 1 else 0.0,
        reliability=round(stdev(reliabilities), 5) if len(reliabilities) > 1 else 0.0,
        validity=round(stdev(validities), 5) if len(validities) > 1 else 0.0,
    )
    mag = round(
        math.sqrt(mean_vec.truthness**2 + mean_vec.reliability**2 + mean_vec.validity**2),
        5,
    )
    spread = round(
        math.sqrt(std_vec.truthness**2 + std_vec.reliability**2 + std_vec.validity**2),
        5,
    )

    fired_signals: list[str] = []
    raw_detectors: dict[str, float] = {}
    axis_components = {"reliability": {}, "validity": {}}
    for seed_score in per_seed_scores:
        for signal in seed_score.fired_signals:
            if signal not in fired_signals:
                fired_signals.append(signal)
        for key, value in seed_score.raw_detectors.items():
            if isinstance(value, (int, float)):
                raw_detectors.setdefault(key, 0.0)
                raw_detectors[key] += float(value)
        for axis_name, component_values in seed_score.axis_components.items():
            for key, value in component_values.items():
                axis_components.setdefault(axis_name, {})
                axis_components[axis_name].setdefault(key, 0.0)
                axis_components[axis_name][key] += float(value)

    raw_detectors = {key: round(value / len(per_seed_scores), 5) for key, value in raw_detectors.items()}
    for axis_name, component_values in axis_components.items():
        axis_components[axis_name] = {
            key: round(value / len(per_seed_scores), 5) for key, value in component_values.items()
        }
    axis_components["reliability"]["seed_variance"] = round(seed_variance_score, 5)
    raw_detectors["metric_delta_std"] = round(stdev(metric_deltas), 5) if len(metric_deltas) > 1 else 0.0
    raw_detectors["seed_variance_score"] = round(seed_variance_score, 5)

    return RunScore(
        mean=mean_vec,
        std=std_vec,
        mag=mag,
        spread=spread,
        fired_signals=fired_signals,
        raw_detectors=raw_detectors,
        axis_components=axis_components,
    ), per_seed_scores


def score_single_seed(
    candidate: RunMetrics,
    baseline: RunMetrics,
    cfg: DetectorConfig,
) -> SeedScore:
    fired_signals: list[str] = []
    raw_detectors: dict[str, float | bool | str] = {}

    metric_delta = candidate.primary_metric - baseline.primary_metric
    truthness = safe_tanh(metric_delta / cfg.truth_delta_scale)

    val_loss_delta = baseline.val_loss - candidate.val_loss
    gap_ratio = ratio(candidate.train_val_gap, baseline.train_val_gap)
    grad_mean_ratio = ratio(candidate.grad_norm_mean, baseline.grad_norm_mean)
    grad_std_ratio = ratio(candidate.grad_norm_std, baseline.grad_norm_std)
    proxy_corr_delta = candidate.proxy_metrics["proxy_metric_corr"] - baseline.proxy_metrics["proxy_metric_corr"]
    efficiency = val_loss_delta / max(candidate.weight_distance, 1e-6)

    raw_detectors.update(
        {
            "metric_delta": metric_delta,
            "val_loss_delta": val_loss_delta,
            "gap_ratio": gap_ratio,
            "grad_mean_ratio": grad_mean_ratio,
            "grad_std_ratio": grad_std_ratio,
            "proxy_corr_delta": proxy_corr_delta,
            "weight_distance": candidate.weight_distance,
            "weight_efficiency": efficiency,
        }
    )

    gap_reliability = score_ratio(gap_ratio, good_max=1.0, warn_threshold=cfg.gap_multiplier, hard_cap=max(2.4, cfg.gap_multiplier * 1.5))
    grad_reliability = score_ratio(
        grad_std_ratio,
        good_max=1.0,
        warn_threshold=cfg.grad_std_multiplier,
        hard_cap=max(2.5, cfg.grad_std_multiplier * 1.5),
    )

    corroborated_grad_damage = (
        grad_mean_ratio > cfg.grad_mean_multiplier
        and (
            grad_std_ratio > cfg.grad_std_multiplier
            or gap_ratio > cfg.gap_multiplier
            or metric_delta <= 0.0
        )
    )
    if corroborated_grad_damage:
        fired_signals.append("exploding_gradients")
        grad_reliability = min(grad_reliability, -1.0)
    if gap_ratio > cfg.gap_multiplier and metric_delta > -0.01:
        fired_signals.append("instability_gap")
    if grad_std_ratio > cfg.grad_std_multiplier:
        fired_signals.append("loss_instability")

    if not all(map(math.isfinite, [candidate.train_loss, candidate.val_loss, candidate.primary_metric])):
        fired_signals.append("nan_loss")
        gap_reliability = -1.0
        grad_reliability = -1.0

    reliability = 0.75
    if cfg.enable_reliability:
        reliability = clamp(0.55 * gap_reliability + 0.45 * grad_reliability)

    comparison_validity = 1.0
    if candidate.eval_hash != baseline.eval_hash:
        fired_signals.append("eval_regime_changed")
        comparison_validity = -1.0
    elif candidate.model_family != baseline.model_family:
        fired_signals.append("model_family_changed")
        comparison_validity = -1.0
    elif candidate.objective_family != baseline.objective_family:
        fired_signals.append("objective_changed")
        comparison_validity = -0.85

    proxy_alignment = score_delta(proxy_corr_delta, negative_threshold=cfg.proxy_corr_drop)
    efficiency_validity = score_floor(efficiency, floor=cfg.weight_efficiency_floor, soft_floor=cfg.weight_efficiency_floor * 1.5)

    if metric_delta >= cfg.adopt_truth_threshold * cfg.truth_delta_scale and grad_std_ratio < cfg.hypercoherence_ratio:
        fired_signals.append("hyper_coherence")
        efficiency_validity = min(efficiency_validity, 0.0)
    if proxy_corr_delta < -cfg.proxy_corr_drop:
        fired_signals.append("proxy_decoupling")

    validity = 0.75
    if cfg.enable_validity:
        validity = clamp(0.45 * comparison_validity + 0.35 * proxy_alignment + 0.20 * efficiency_validity)

    axis_components = {
        "reliability": {
            "gap_health": round(gap_reliability, 5),
            "gradient_health": round(grad_reliability, 5),
        },
        "validity": {
            "comparison_validity": round(comparison_validity, 5),
            "proxy_alignment": round(proxy_alignment, 5),
            "efficiency_validity": round(efficiency_validity, 5),
        },
    }

    return SeedScore(
        vec3=Vec3(round(truthness, 5), round(reliability, 5), round(validity, 5)),
        fired_signals=fired_signals,
        raw_detectors=raw_detectors,
        axis_components=axis_components,
    )


def ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-8:
        return 0.0
    return numerator / denominator


def safe_tanh(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.tanh(value)


def bounded_inverse(value: float, scale: float) -> float:
    if not math.isfinite(value) or scale <= 0:
        return 0.0
    return clamp(1.0 - (value / scale))


def score_ratio(value: float, *, good_max: float, warn_threshold: float, hard_cap: float) -> float:
    if not math.isfinite(value):
        return -1.0
    if value <= good_max:
        return 1.0
    if value <= warn_threshold:
        span = max(warn_threshold - good_max, 1e-6)
        return clamp(1.0 - ((value - good_max) / span))
    if value >= hard_cap:
        return -1.0
    span = max(hard_cap - warn_threshold, 1e-6)
    return clamp(-((value - warn_threshold) / span))


def score_delta(delta: float, *, negative_threshold: float) -> float:
    if not math.isfinite(delta):
        return -1.0
    if delta >= 0:
        return 1.0
    return clamp(1.0 + (delta / max(negative_threshold, 1e-6)) * 2.0)


def score_floor(value: float, *, floor: float, soft_floor: float) -> float:
    if not math.isfinite(value):
        return -1.0
    if value >= soft_floor:
        return 1.0
    if value >= floor:
        span = max(soft_floor - floor, 1e-6)
        return clamp((value - floor) / span)
    span = max(floor, 1e-6)
    return clamp((value / span) - 1.0)


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
