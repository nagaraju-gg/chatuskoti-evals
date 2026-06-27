from __future__ import annotations

import math
from statistics import mean, stdev

from chatuskoti_evals.core.config import DetectorConfig
from chatuskoti_evals.core.models import AxisValue, RunMetrics, RunScore, SeedScore, Vec3, Vec3State


def score_run_metrics(
    candidate_metrics: list[RunMetrics],
    baseline_metrics: RunMetrics,
    cfg: DetectorConfig,
) -> tuple[RunScore, list[SeedScore]]:
    """Score candidate metrics against baseline, producing T/R/V per-seed and aggregated RunScore."""
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

    reliability_axis = axis_from_values(
        name="reliability",
        values=reliabilities,
        enabled=cfg.enable_reliability,
        cfg=cfg,
        run_detector=seed_variance_score,
    )

    validity_axis = axis_from_values(
        name="validity",
        values=validities,
        enabled=cfg.enable_validity,
        cfg=cfg,
    )
    truthness_axis = AxisValue("truthness", round(mean(truthnesses), 5), "measured")

    mean_state = Vec3State(
        truthness=truthness_axis,
        reliability=reliability_axis,
        validity=validity_axis,
    )

    mean_vec = Vec3(
        truthness=truthness_axis.projected(),
        reliability=reliability_axis.projected(),
        validity=validity_axis.projected(),
    )
    std_vec = Vec3(
        truthness=round(stdev(truthnesses), 5) if len(truthnesses) > 1 else 0.0,
        reliability=0.0 if not reliability_axis.is_defined else round(stdev(reliabilities), 5) if len(reliabilities) > 1 else 0.0,
        validity=0.0 if not validity_axis.is_defined else round(stdev(validities), 5) if len(validities) > 1 else 0.0,
    )
    mag = round(
        math.sqrt(
            sum(
                axis.projected() ** 2
                for axis in (mean_state.truthness, mean_state.reliability, mean_state.validity)
                if axis.is_defined
            )
        ),
        5,
    )
    spread = round(
        math.sqrt(std_vec.truthness**2 + std_vec.reliability**2 + std_vec.validity**2),
        5,
    )

    fired_signals: list[str] = []
    raw_detectors: dict[str, float] = {}
    axis_components: dict[str, dict[str, float]] = {"reliability": {}, "validity": {}}
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
        axis_state=mean_state,
    ), per_seed_scores


def score_single_seed(
    candidate: RunMetrics,
    baseline: RunMetrics,
    cfg: DetectorConfig,
) -> SeedScore:
    """Score a single seed: compute truthness, reliability, validity and detect signals."""
    fired_signals: list[str] = []
    raw_detectors: dict[str, float | bool | str] = {}

    metric_delta = candidate.primary_metric - baseline.primary_metric
    truthness = score_truthness(candidate.primary_metric, baseline.primary_metric, cfg)

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

    reliability = cfg.disabled_axis_imputation
    if cfg.enable_reliability:
        reliability = aggregate_components(
            [gap_reliability, grad_reliability],
            cfg,
            weights=[0.55, 0.45],
        )

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

    validity = cfg.disabled_axis_imputation
    if cfg.enable_validity:
        validity = aggregate_components(
            [comparison_validity, proxy_alignment, efficiency_validity],
            cfg,
            weights=[0.45, 0.35, 0.20],
        )

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


def score_truthness(candidate_metric: float, baseline_metric: float, cfg: DetectorConfig) -> float:
    metric_delta = candidate_metric - baseline_metric
    if cfg.truthness_transform == "absolute_delta":
        return safe_tanh(metric_delta / cfg.truth_delta_scale)
    if cfg.truthness_transform == "relative_delta":
        denominator = baseline_metric + cfg.relative_truth_epsilon
        if abs(denominator) < 1e-12:
            return 0.0
        return safe_tanh((metric_delta / denominator) * cfg.relative_truth_scale)
    raise ValueError(f"unknown truthness_transform: {cfg.truthness_transform}")


def aggregate_components(values: list[float], cfg: DetectorConfig, *, weights: list[float] | None = None) -> float:
    if not values:
        return 0.0
    if cfg.axis_aggregation == "weighted":
        if weights is None:
            return clamp(mean(values))
        total_weight = sum(weights)
        if total_weight <= 0:
            return 0.0
        return clamp(sum(value * weight for value, weight in zip(values, weights, strict=False)) / total_weight)
    if cfg.axis_aggregation == "worst_case":
        return clamp(min(values))
    raise ValueError(f"unknown axis_aggregation: {cfg.axis_aggregation}")


def axis_from_values(
    *,
    name: str,
    values: list[float],
    enabled: bool,
    cfg: DetectorConfig,
    run_detector: float | None = None,
) -> AxisValue:
    if not enabled:
        if cfg.disabled_axis_policy == "impute":
            return AxisValue(
                name,
                round(cfg.disabled_axis_imputation, 5),
                "imputed",
                f"axis disabled; imputed {cfg.disabled_axis_imputation:.3f}",
            )
        if cfg.disabled_axis_policy == "undefined":
            return AxisValue(name, None, "undefined", "axis disabled; state is partial")
        raise ValueError(f"unknown disabled_axis_policy: {cfg.disabled_axis_policy}")

    if cfg.axis_aggregation == "weighted":
        value = mean(values)
        if name == "reliability" and run_detector is not None:
            value = clamp(0.65 * value + 0.35 * run_detector)
        return AxisValue(name, round(value, 5), "measured")

    if cfg.axis_aggregation == "worst_case":
        candidates = list(values)
        if name == "reliability" and run_detector is not None:
            candidates.append(run_detector)
        return AxisValue(name, round(min(candidates), 5), "measured")

    raise ValueError(f"unknown axis_aggregation: {cfg.axis_aggregation}")


def ratio(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 when denominator is near zero."""
    if abs(denominator) < 1e-8:
        return 0.0
    return numerator / denominator


def safe_tanh(value: float) -> float:
    """Tanh with NaN/Inf protection, returning 0.0 for non-finite inputs."""
    if not math.isfinite(value):
        return 0.0
    return math.tanh(value)


def bounded_inverse(value: float, scale: float) -> float:
    """Map value/scale to [0,1] — 1 when value is 0, approaching 0 as value exceeds scale."""
    if not math.isfinite(value) or scale <= 0:
        return 0.0
    return clamp(1.0 - (value / scale))


def score_ratio(value: float, *, good_max: float, warn_threshold: float, hard_cap: float) -> float:
    """Score a ratio against thresholds: 1.0 if <= good_max, linear decay to -1.0 at hard_cap."""
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
    """Score a delta: 1.0 if >= 0, linearly decreasing to -1.0 as delta reaches -negative_threshold."""
    if not math.isfinite(delta):
        return -1.0
    if delta >= 0:
        return 1.0
    return clamp(1.0 + (delta / max(negative_threshold, 1e-6)) * 2.0)


def score_floor(value: float, *, floor: float, soft_floor: float) -> float:
    """Score on a floor: 1.0 above soft_floor, linear between floor and soft_floor, negative below floor."""
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
    """Clamp value to [minimum, maximum] range."""
    return max(minimum, min(maximum, value))
