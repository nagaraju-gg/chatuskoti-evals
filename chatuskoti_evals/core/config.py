from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    truthness_transform: str = "absolute_delta"
    truth_delta_scale: float = 0.02
    relative_truth_scale: float = 4.0
    relative_truth_epsilon: float = 1e-8
    axis_aggregation: str = "weighted"
    disabled_axis_policy: str = "impute"
    disabled_axis_imputation: float = 0.75
    max_spread: float = 0.30
    min_magnitude: float = 0.0
    goodhart_threshold: float = 0.75
    structural_distance_threshold: float = 0.60
    adopt_truth_threshold: float = 0.25
    reliability_threshold: float = 0.35
    validity_threshold: float = 0.15
    gap_multiplier: float = 1.6
    grad_mean_multiplier: float = 1.8
    grad_std_multiplier: float = 1.7
    hypercoherence_ratio: float = 0.40
    proxy_corr_drop: float = 0.18
    seed_metric_std_scale: float = 0.02
    weight_efficiency_floor: float = 0.18
    binary_metric_threshold: float = 0.002
    enable_reliability: bool = True
    enable_validity: bool = True
    enable_spread_gate: bool = True


def vec3_prototype_detector_config() -> DetectorConfig:
    """Detector config matching the Vec3 prototype scoring and threshold priors."""
    return DetectorConfig(
        truthness_transform="relative_delta",
        relative_truth_scale=4.0,
        relative_truth_epsilon=1e-8,
        axis_aggregation="worst_case",
        disabled_axis_policy="undefined",
        max_spread=0.35,
        min_magnitude=0.15,
        goodhart_threshold=0.75,
        structural_distance_threshold=0.60,
    )


DETECTOR_PRESETS = {
    "default": DetectorConfig,
    "vec3_prototype": vec3_prototype_detector_config,
}


def detector_config_preset(name: str) -> DetectorConfig:
    try:
        return DETECTOR_PRESETS[name]()
    except KeyError as exc:
        available = ", ".join(sorted(DETECTOR_PRESETS))
        raise ValueError(f"unknown detector preset {name!r}; available presets: {available}") from exc


def load_detector_config(path: Path) -> DetectorConfig:
    """Load a DetectorConfig from a JSON object matching dataclass field names."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("detector config file must contain a JSON object")
    return DetectorConfig(**payload)


@dataclass(frozen=True)
class AblationConfig:
    name: str = "full"
    disable_wisdom: bool = False

    def apply(self, detector: DetectorConfig) -> DetectorConfig:
        normalized = self.normalized_name
        updates: dict[str, bool] = {}
        if normalized == "no_reliability":
            updates["enable_reliability"] = False
        elif normalized == "no_validity":
            updates["enable_validity"] = False
        elif normalized == "no_spread_gate":
            updates["enable_spread_gate"] = False
        elif normalized == "t_only":
            updates["enable_reliability"] = False
            updates["enable_validity"] = False
        elif normalized == "t_r":
            updates["enable_reliability"] = True
            updates["enable_validity"] = False
        elif normalized == "t_v":
            updates["enable_reliability"] = False
            updates["enable_validity"] = True
        elif normalized == "t_r_v":
            updates["enable_reliability"] = True
            updates["enable_validity"] = True
        return DetectorConfig(**{**detector.__dict__, **updates})

    @property
    def wisdom_enabled(self) -> bool:
        return self.normalized_name != "no_wisdom"

    @property
    def normalized_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class LoopConfig:
    controller: str
    max_iterations: int = 4
    n_seeds: int = 3
    width: int = 1
    depth: float = 1.0
    mode: str = "default"


@dataclass(frozen=True)
class SimulationConfig:
    dataset: str = "CIFAR-100"
    model: str = "ResNet-18"
    framework: str = "PyTorch-compatible simulation"
    baseline_train_loss: float = 1.45
    baseline_primary_metric: float = 0.636
    param_count: int = 11_200_000


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path
    keep_raw_seed_metrics: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str = "simulator"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
