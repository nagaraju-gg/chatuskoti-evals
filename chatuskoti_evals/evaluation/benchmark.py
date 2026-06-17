from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from random import Random
from typing import Any, Protocol

from chatuskoti_evals.core.config import ExperimentConfig, SimulationConfig
from chatuskoti_evals.core.models import ActionSpec, BaselineRecord, RunMetrics, average_run_metrics
from chatuskoti_evals.evaluation.progress import RunProgressContext, RunProgressTracker


class BenchmarkAdapter(Protocol):
    def record_baseline(
        self,
        seeds: list[int],
        *,
        progress: RunProgressTracker | None = None,
        progress_context: RunProgressContext | None = None,
    ) -> BaselineRecord: ...

    def execute(
        self,
        action: ActionSpec,
        seeds: list[int],
        *,
        progress: RunProgressTracker | None = None,
        progress_context: RunProgressContext | None = None,
    ) -> tuple[list[RunMetrics], object]: ...

    def adopt(self, candidate: Any) -> None: ...

    def primary_metric(self) -> float: ...


@dataclass(frozen=True)
class SimulatedState:
    generalization: float = 0.50
    stability: float = 0.72
    proxy_alignment: float = 0.86
    eval_protocol: str = "baseline_v1"
    model_family: str = "resnet18"
    objective_family: str = "cross_entropy"
    scheduler_ready: bool = False
    metric_bias: float = 0.0


@dataclass(frozen=True)
class CandidateState:
    state: SimulatedState
    metric_bonus: float
    weight_distance: float
    tags: tuple[str, ...]


class SimulatedCIFAR100ResNet18Adapter:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.current_state = SimulatedState()

    def record_baseline(
        self,
        seeds: list[int],
        *,
        progress: RunProgressTracker | None = None,
        progress_context: RunProgressContext | None = None,
    ) -> BaselineRecord:
        metrics = self._aggregate_baseline_metrics(self.current_state, seeds)
        baseline_id = sha256(
            f"{metrics.eval_hash}:{metrics.primary_metric:.5f}:{metrics.train_loss:.5f}".encode()
        ).hexdigest()[:12]
        if progress is not None:
            progress.finish_run(count=len(seeds))
        return BaselineRecord(baseline_id=baseline_id, metrics=metrics)

    def execute(
        self,
        action: ActionSpec,
        seeds: list[int],
        *,
        progress: RunProgressTracker | None = None,
        progress_context: RunProgressContext | None = None,
    ) -> tuple[list[RunMetrics], CandidateState]:
        candidate = self._candidate_state(self.current_state, action)
        metrics = [
            self._simulate_metrics(candidate, action, seed=seed, run_prefix=action.name)
            for seed in seeds
        ]
        if progress is not None:
            progress.finish_run(count=len(seeds))
        return metrics, candidate

    def adopt(self, candidate: CandidateState) -> None:
        self.current_state = candidate.state

    def primary_metric(self) -> float:
        state = self.current_state
        return round(self.cfg.baseline_primary_metric + (state.generalization - 0.50) * 0.16, 5)

    def _aggregate_baseline_metrics(self, state: SimulatedState, seeds: list[int]) -> RunMetrics:
        neutral = CandidateState(state=state, metric_bonus=0.0, weight_distance=0.0, tags=tuple())
        per_seed = [self._simulate_metrics(neutral, None, seed, "baseline") for seed in seeds]
        return self._mean_metrics(per_seed, "baseline-mean")

    def _candidate_state(self, state: SimulatedState, action: ActionSpec) -> CandidateState:
        handler = _CANDIDATE_HANDLERS.get(action.name)
        if handler is not None:
            metric_bonus, weight_distance, next_state, tags = handler(state)
            return CandidateState(
                state=next_state,
                metric_bonus=metric_bonus,
                weight_distance=weight_distance,
                tags=tuple(tags),
            )
        return CandidateState(state=state, metric_bonus=0.0, weight_distance=0.10, tags=())

    def _simulate_metrics(
        self,
        candidate: CandidateState,
        action: ActionSpec | None,
        seed: int,
        run_prefix: str,
    ) -> RunMetrics:
        state = candidate.state
        rng = self._rng(run_prefix, seed, state.eval_protocol, state.objective_family, ",".join(candidate.tags))
        noise = rng.uniform(-0.003, 0.003)
        gap_noise = rng.uniform(-0.015, 0.015)
        grad_noise = rng.uniform(-0.12, 0.12)
        proxy_noise = rng.uniform(-0.025, 0.025)

        base_primary = self.cfg.baseline_primary_metric
        primary_metric = (
            base_primary
            + (state.generalization - 0.50) * 0.16
            + state.metric_bias
            + candidate.metric_bonus
            + noise
        )

        train_loss = max(
            0.35,
            self.cfg.baseline_train_loss
            - (state.generalization - 0.50) * 0.85
            - state.metric_bias * 5.0
            - candidate.metric_bonus * 3.5
            + rng.uniform(-0.03, 0.03),
        )
        train_val_gap = max(
            0.02,
            0.18 + (0.82 - state.stability) * 0.50 + gap_noise,
        )
        if "instability_gap" in candidate.tags or "unstable_regularization" in candidate.tags:
            train_val_gap += 0.13
        val_loss = train_loss + train_val_gap

        grad_norm_mean = max(0.3, 2.05 + (0.85 - state.stability) * 5.0 + grad_noise)
        grad_norm_std = max(0.005, 0.11 + (0.82 - state.stability) * 0.22 + rng.uniform(-0.01, 0.01))

        if "hypercoherence" in candidate.tags:
            grad_norm_std = 0.018 + rng.uniform(0.0, 0.005)
            grad_norm_mean = max(0.5, grad_norm_mean - 0.45)
        if "exploding_gradients" in candidate.tags:
            grad_norm_mean += 3.3
            grad_norm_std += 0.20

        proxy_corr = min(0.99, max(0.0, state.proxy_alignment + proxy_noise))
        calibration = min(0.98, max(0.0, state.proxy_alignment - 0.04 + proxy_noise))
        if "proxy_decoupling" in candidate.tags:
            proxy_corr = max(0.05, proxy_corr - 0.42)
            calibration = max(0.03, calibration - 0.30)

        eval_hash = sha256(
            f"{self.cfg.dataset}:{state.model_family}:{state.objective_family}:{state.eval_protocol}".encode()
        ).hexdigest()[:12]

        name = action.name if action else "baseline"
        return RunMetrics(
            run_id=f"{run_prefix}-seed{seed}",
            seed=seed,
            primary_metric=round(primary_metric, 5),
            train_loss=round(train_loss, 5),
            val_loss=round(val_loss, 5),
            train_val_gap=round(train_val_gap, 5),
            grad_norm_mean=round(grad_norm_mean, 5),
            grad_norm_std=round(grad_norm_std, 5),
            weight_distance=round(candidate.weight_distance + rng.uniform(-0.02, 0.02), 5),
            param_count=self.cfg.param_count,
            eval_hash=eval_hash,
            model_family=state.model_family,
            objective_family=state.objective_family,
            proxy_metrics={
                "proxy_metric_corr": round(proxy_corr, 5),
                "calibration": round(calibration, 5),
            },
            detector_inputs={
                "action_name": name,
                "scheduler_ready": state.scheduler_ready,
                "tags": ",".join(candidate.tags),
            },
        )

    _mean_metrics = staticmethod(average_run_metrics)

    @staticmethod
    def _rng(*parts: object) -> Random:
        raw = "|".join(str(part) for part in parts)
        seed = int(sha256(raw.encode("utf-8")).hexdigest()[:16], 16)
        return Random(seed)


CandidateHandler = tuple[float, float, SimulatedState, list[str]]


def _candidate_stochastic_depth_high(state: SimulatedState) -> CandidateHandler:
    metric_bonus = 0.028
    weight_distance = 0.63
    if state.scheduler_ready:
        next_state = replace(
            state,
            generalization=min(0.92, state.generalization + 0.025),
            stability=max(0.0, state.stability - 0.06),
            proxy_alignment=min(1.0, state.proxy_alignment + 0.02),
            metric_bias=state.metric_bias + 0.006,
        )
        return metric_bonus, weight_distance, next_state, ["regularization_applied"]
    next_state = replace(
        state,
        generalization=min(0.92, state.generalization + 0.004),
        stability=max(0.0, state.stability - 0.22),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.32),
        metric_bias=state.metric_bias + 0.018,
    )
    return metric_bonus, weight_distance, next_state, ["hypercoherence", "proxy_decoupling", "unstable_regularization"]


def _candidate_cosine_warmup(state: SimulatedState) -> CandidateHandler:
    return 0.014, 0.22, replace(
        state,
        generalization=min(0.92, state.generalization + 0.018),
        stability=min(0.98, state.stability + 0.18),
        proxy_alignment=min(0.99, state.proxy_alignment + 0.08),
        scheduler_ready=True,
        metric_bias=state.metric_bias + 0.003,
    ), ["optimizer_recovery"]


def _candidate_stochastic_depth_low(state: SimulatedState) -> CandidateHandler:
    weight_distance = 0.34
    if state.scheduler_ready and state.stability >= 0.80:
        return 0.032, weight_distance, replace(
            state,
            generalization=min(0.95, state.generalization + 0.045),
            stability=max(0.0, state.stability - 0.02),
            proxy_alignment=min(1.0, state.proxy_alignment + 0.05),
            metric_bias=state.metric_bias + 0.004,
        ), ["clean_regularization"]
    return 0.008, weight_distance, replace(
        state,
        generalization=min(0.95, state.generalization + 0.010),
        stability=max(0.0, state.stability - 0.06),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.04),
        metric_bias=state.metric_bias + 0.010,
    ), ["fragile_regularization"]


def _candidate_label_smoothing(state: SimulatedState) -> CandidateHandler:
    return 0.015, 0.18, replace(
        state,
        generalization=min(0.95, state.generalization + 0.017),
        stability=min(1.0, state.stability + 0.04),
        proxy_alignment=min(1.0, state.proxy_alignment + 0.03),
        metric_bias=state.metric_bias + 0.002,
    ), ["clean_win"]


def _candidate_mixup(state: SimulatedState) -> CandidateHandler:
    metric_bonus = 0.018 if state.stability >= 0.78 else 0.006
    return metric_bonus, 0.28, replace(
        state,
        generalization=min(0.95, state.generalization + 0.022),
        stability=min(1.0, state.stability + 0.01),
        proxy_alignment=min(1.0, state.proxy_alignment + 0.02),
        metric_bias=state.metric_bias + 0.002,
    ), ["clean_win"]


def _candidate_adamw(state: SimulatedState) -> CandidateHandler:
    return 0.016, 0.26, replace(
        state,
        generalization=min(0.95, state.generalization + 0.020),
        stability=min(1.0, state.stability + 0.05),
        proxy_alignment=min(1.0, state.proxy_alignment + 0.04),
        metric_bias=state.metric_bias + 0.003,
    ), ["clean_win"]


def _candidate_high_lr(state: SimulatedState) -> CandidateHandler:
    return -0.028, 0.48, replace(
        state,
        generalization=max(0.0, state.generalization - 0.035),
        stability=max(0.0, state.stability - 0.30),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.10),
        metric_bias=state.metric_bias - 0.010,
    ), ["exploding_gradients", "broken_failure"]


def _candidate_focal_objective(state: SimulatedState) -> CandidateHandler:
    return 0.010, 0.40, replace(
        state,
        generalization=min(0.95, state.generalization + 0.010),
        objective_family="focal_loss",
        metric_bias=state.metric_bias + 0.006,
    ), ["objective_shift"]


def _candidate_eval_tta(state: SimulatedState) -> CandidateHandler:
    return 0.012, 0.08, replace(
        state,
        eval_protocol="tta_v1",
        metric_bias=state.metric_bias + 0.008,
    ), ["eval_shift"]


def _candidate_dropout_high(state: SimulatedState) -> CandidateHandler:
    return 0.011, 0.31, replace(
        state,
        generalization=min(0.95, state.generalization + 0.006),
        stability=max(0.0, state.stability - 0.18),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.08),
        metric_bias=state.metric_bias + 0.010,
    ), ["pyrrhic", "instability_gap"]


def _candidate_pyrrhic_probe(state: SimulatedState) -> CandidateHandler:
    return 0.028, 0.29, replace(
        state,
        generalization=min(0.95, state.generalization + 0.015),
        stability=max(0.0, state.stability - 0.24),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.06),
        metric_bias=state.metric_bias + 0.014,
    ), ["pyrrhic_probe", "instability_gap"]


def _candidate_metric_gaming_probe(state: SimulatedState) -> CandidateHandler:
    return 0.034, 0.33, replace(
        state,
        generalization=min(0.95, state.generalization + 0.018),
        stability=max(0.0, state.stability - 0.12),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.34),
        metric_bias=state.metric_bias + 0.018,
    ), ["metric_gaming_probe", "hypercoherence", "proxy_decoupling"]


def _candidate_broken_probe(state: SimulatedState) -> CandidateHandler:
    return -0.030, 0.42, replace(
        state,
        generalization=max(0.0, state.generalization - 0.032),
        stability=max(0.0, state.stability - 0.28),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.18),
        metric_bias=state.metric_bias - 0.012,
    ), ["exploding_gradients", "broken_failure"]


def _candidate_pyrrhic_high_variance(state: SimulatedState) -> CandidateHandler:
    return 0.025, 0.30, replace(
        state,
        generalization=min(0.95, state.generalization + 0.012),
        stability=max(0.0, state.stability - 0.20),
        proxy_alignment=min(1.0, state.proxy_alignment - 0.04),
        metric_bias=state.metric_bias + 0.012,
    ), ["pyrrhic_high_variance", "instability_gap"]


def _candidate_pyrrhic_fragile(state: SimulatedState) -> CandidateHandler:
    return 0.022, 0.35, replace(
        state,
        generalization=min(0.95, state.generalization + 0.010),
        stability=max(0.0, state.stability - 0.28),
        proxy_alignment=min(1.0, state.proxy_alignment + 0.02),
        metric_bias=state.metric_bias + 0.010,
    ), ["pyrrhic_fragile", "instability_gap"]


def _candidate_orthogonal_shortcut(state: SimulatedState) -> CandidateHandler:
    return 0.030, 0.25, replace(
        state,
        generalization=min(0.95, state.generalization + 0.020),
        stability=min(1.0, state.stability + 0.05),
        proxy_alignment=max(0.0, state.proxy_alignment - 0.40),
        metric_bias=state.metric_bias + 0.015,
    ), ["orthogonal_shortcut", "proxy_decoupling"]


def _candidate_orthogonal_ref_mismatch(state: SimulatedState) -> CandidateHandler:
    return 0.015, 0.10, replace(
        state,
        generalization=min(0.95, state.generalization + 0.008),
        stability=min(1.0, state.stability + 0.02),
        proxy_alignment=min(1.0, state.proxy_alignment + 0.01),
        eval_protocol="ref_mismatch_v1",
        metric_bias=state.metric_bias + 0.008,
    ), ["orthogonal_ref_mismatch", "eval_shift"]


_CANDIDATE_HANDLERS: dict[str, Callable[[SimulatedState], CandidateHandler]] = {
    "stochastic_depth_high": _candidate_stochastic_depth_high,
    "cosine_warmup": _candidate_cosine_warmup,
    "stochastic_depth_low": _candidate_stochastic_depth_low,
    "label_smoothing": _candidate_label_smoothing,
    "mixup": _candidate_mixup,
    "adamw": _candidate_adamw,
    "high_lr": _candidate_high_lr,
    "focal_objective": _candidate_focal_objective,
    "eval_tta": _candidate_eval_tta,
    "dropout_high": _candidate_dropout_high,
    "pyrrhic_probe": _candidate_pyrrhic_probe,
    "metric_gaming_probe": _candidate_metric_gaming_probe,
    "broken_probe": _candidate_broken_probe,
    "pyrrhic_high_variance": _candidate_pyrrhic_high_variance,
    "pyrrhic_fragile": _candidate_pyrrhic_fragile,
    "orthogonal_shortcut": _candidate_orthogonal_shortcut,
    "orthogonal_ref_mismatch": _candidate_orthogonal_ref_mismatch,
}


def create_benchmark_adapter(cfg: ExperimentConfig) -> BenchmarkAdapter:
    if cfg.backend == "simulator":
        return SimulatedCIFAR100ResNet18Adapter(cfg.simulation)
    raise ValueError(f"unsupported backend: {cfg.backend}")
