from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from random import Random
from typing import Protocol

from chatuskoti_evals.config import ExperimentConfig, SimulationConfig
from chatuskoti_evals.models import ActionSpec, BaselineRecord, RunMetrics
from chatuskoti_evals.progress import RunProgressContext, RunProgressTracker


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

    def adopt(self, candidate: object) -> None: ...

    def canonical_primary_metric(self) -> float: ...


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
            f"{metrics.eval_hash}:{metrics.primary_metric:.5f}:{metrics.train_loss:.5f}".encode("utf-8")
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

    def canonical_primary_metric(self) -> float:
        state = self.current_state
        return round(self.cfg.baseline_primary_metric + (state.generalization - 0.50) * 0.16, 5)

    def _aggregate_baseline_metrics(self, state: SimulatedState, seeds: list[int]) -> RunMetrics:
        neutral = CandidateState(state=state, metric_bonus=0.0, weight_distance=0.0, tags=tuple())
        per_seed = [self._simulate_metrics(neutral, None, seed, "baseline") for seed in seeds]
        return self._mean_metrics(per_seed, "baseline-mean")

    def _candidate_state(self, state: SimulatedState, action: ActionSpec) -> CandidateState:
        tags: list[str] = []
        next_state = state
        metric_bonus = 0.0
        weight_distance = 0.10

        if action.name == "stochastic_depth_high":
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
                tags.extend(["regularization_applied"])
            else:
                next_state = replace(
                    state,
                    generalization=min(0.92, state.generalization + 0.004),
                    stability=max(0.0, state.stability - 0.22),
                    proxy_alignment=max(0.0, state.proxy_alignment - 0.32),
                    metric_bias=state.metric_bias + 0.018,
                )
                tags.extend(["hypercoherence", "proxy_decoupling", "unstable_regularization"])
        elif action.name == "cosine_warmup":
            metric_bonus = 0.014
            weight_distance = 0.22
            next_state = replace(
                state,
                generalization=min(0.92, state.generalization + 0.018),
                stability=min(0.98, state.stability + 0.18),
                proxy_alignment=min(0.99, state.proxy_alignment + 0.08),
                scheduler_ready=True,
                metric_bias=state.metric_bias + 0.003,
            )
            tags.extend(["optimizer_recovery"])
        elif action.name == "stochastic_depth_low":
            weight_distance = 0.34
            if state.scheduler_ready and state.stability >= 0.80:
                metric_bonus = 0.032
                next_state = replace(
                    state,
                    generalization=min(0.95, state.generalization + 0.045),
                    stability=max(0.0, state.stability - 0.02),
                    proxy_alignment=min(1.0, state.proxy_alignment + 0.05),
                    metric_bias=state.metric_bias + 0.004,
                )
                tags.extend(["clean_regularization"])
            else:
                metric_bonus = 0.008
                next_state = replace(
                    state,
                    generalization=min(0.95, state.generalization + 0.010),
                    stability=max(0.0, state.stability - 0.06),
                    proxy_alignment=max(0.0, state.proxy_alignment - 0.04),
                    metric_bias=state.metric_bias + 0.010,
                )
                tags.extend(["fragile_regularization"])
        elif action.name == "label_smoothing":
            metric_bonus = 0.015
            weight_distance = 0.18
            next_state = replace(
                state,
                generalization=min(0.95, state.generalization + 0.017),
                stability=min(1.0, state.stability + 0.04),
                proxy_alignment=min(1.0, state.proxy_alignment + 0.03),
                metric_bias=state.metric_bias + 0.002,
            )
            tags.extend(["clean_win"])
        elif action.name == "mixup":
            metric_bonus = 0.018 if state.stability >= 0.78 else 0.006
            weight_distance = 0.28
            next_state = replace(
                state,
                generalization=min(0.95, state.generalization + 0.022),
                stability=min(1.0, state.stability + 0.01),
                proxy_alignment=min(1.0, state.proxy_alignment + 0.02),
                metric_bias=state.metric_bias + 0.002,
            )
            tags.extend(["clean_win"])
        elif action.name == "adamw":
            metric_bonus = 0.016
            weight_distance = 0.26
            next_state = replace(
                state,
                generalization=min(0.95, state.generalization + 0.020),
                stability=min(1.0, state.stability + 0.05),
                proxy_alignment=min(1.0, state.proxy_alignment + 0.04),
                metric_bias=state.metric_bias + 0.003,
            )
            tags.extend(["clean_win"])
        elif action.name == "high_lr":
            metric_bonus = -0.028
            weight_distance = 0.48
            next_state = replace(
                state,
                generalization=max(0.0, state.generalization - 0.035),
                stability=max(0.0, state.stability - 0.30),
                proxy_alignment=max(0.0, state.proxy_alignment - 0.10),
                metric_bias=state.metric_bias - 0.010,
            )
            tags.extend(["exploding_gradients", "broken_failure"])
        elif action.name == "focal_objective":
            metric_bonus = 0.010
            weight_distance = 0.40
            next_state = replace(
                state,
                generalization=min(0.95, state.generalization + 0.010),
                objective_family="focal_loss",
                metric_bias=state.metric_bias + 0.006,
            )
            tags.extend(["objective_shift"])
        elif action.name == "eval_tta":
            metric_bonus = 0.012
            weight_distance = 0.08
            next_state = replace(
                state,
                eval_protocol="tta_v1",
                metric_bias=state.metric_bias + 0.008,
            )
            tags.extend(["eval_shift"])
        elif action.name == "dropout_high":
            metric_bonus = 0.011
            weight_distance = 0.31
            next_state = replace(
                state,
                generalization=min(0.95, state.generalization + 0.006),
                stability=max(0.0, state.stability - 0.18),
                proxy_alignment=max(0.0, state.proxy_alignment - 0.08),
                metric_bias=state.metric_bias + 0.010,
            )
            tags.extend(["pyrrhic", "instability_gap"])
        elif action.name == "pyrrhic_probe":
            metric_bonus = 0.028
            weight_distance = 0.29
            next_state = replace(
                state,
                generalization=min(0.95, state.generalization + 0.015),
                stability=max(0.0, state.stability - 0.24),
                proxy_alignment=max(0.0, state.proxy_alignment - 0.06),
                metric_bias=state.metric_bias + 0.014,
            )
            tags.extend(["pyrrhic_probe", "instability_gap"])
        elif action.name == "metric_gaming_probe":
            metric_bonus = 0.034
            weight_distance = 0.33
            next_state = replace(
                state,
                generalization=min(0.95, state.generalization + 0.018),
                stability=max(0.0, state.stability - 0.12),
                proxy_alignment=max(0.0, state.proxy_alignment - 0.34),
                metric_bias=state.metric_bias + 0.018,
            )
            tags.extend(["metric_gaming_probe", "hypercoherence", "proxy_decoupling"])
        elif action.name == "broken_probe":
            metric_bonus = -0.030
            weight_distance = 0.42
            next_state = replace(
                state,
                generalization=max(0.0, state.generalization - 0.032),
                stability=max(0.0, state.stability - 0.28),
                proxy_alignment=max(0.0, state.proxy_alignment - 0.18),
                metric_bias=state.metric_bias - 0.012,
            )
            tags.extend(["exploding_gradients", "broken_failure"])

        return CandidateState(
            state=next_state,
            metric_bonus=metric_bonus,
            weight_distance=weight_distance,
            tags=tuple(tags),
        )

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
            f"{self.cfg.dataset}:{state.model_family}:{state.objective_family}:{state.eval_protocol}".encode("utf-8")
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

    @staticmethod
    def _mean_metrics(metrics: list[RunMetrics], run_id: str) -> RunMetrics:
        if not metrics:
            raise ValueError("metrics must not be empty")
        first = metrics[0]
        mean = lambda getter: round(sum(getter(item) for item in metrics) / len(metrics), 5)
        proxy_keys = first.proxy_metrics.keys()
        return RunMetrics(
            run_id=run_id,
            seed=-1,
            primary_metric=mean(lambda item: item.primary_metric),
            train_loss=mean(lambda item: item.train_loss),
            val_loss=mean(lambda item: item.val_loss),
            train_val_gap=mean(lambda item: item.train_val_gap),
            grad_norm_mean=mean(lambda item: item.grad_norm_mean),
            grad_norm_std=mean(lambda item: item.grad_norm_std),
            weight_distance=mean(lambda item: item.weight_distance),
            param_count=first.param_count,
            eval_hash=first.eval_hash,
            model_family=first.model_family,
            objective_family=first.objective_family,
            proxy_metrics={key: mean(lambda item, metric_key=key: item.proxy_metrics[metric_key]) for key in proxy_keys},
            detector_inputs={},
        )

    @staticmethod
    def _rng(*parts: object) -> Random:
        raw = "|".join(str(part) for part in parts)
        seed = int(sha256(raw.encode("utf-8")).hexdigest()[:16], 16)
        return Random(seed)


def create_benchmark_adapter(cfg: ExperimentConfig) -> BenchmarkAdapter:
    if cfg.backend == "simulator":
        return SimulatedCIFAR100ResNet18Adapter(cfg.simulation)
    if cfg.backend == "torch":
        from chatuskoti_evals.torch_backend import TorchCIFAR100ResNet18Adapter

        return TorchCIFAR100ResNet18Adapter(cfg.torch)
    raise ValueError(f"unsupported backend: {cfg.backend}")
