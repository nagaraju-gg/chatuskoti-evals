from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    truth_delta_scale: float = 0.02
    max_spread: float = 0.30
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
        aliases = {
            "no_coherence": "no_reliability",
            "no_comparability": "no_validity",
            "no_goodhart": "no_validity",
        }
        return aliases.get(self.name, self.name)


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
class TorchBenchmarkConfig:
    dataset: str = "CIFAR-100"
    model: str = "ResNet-18"
    framework: str = "PyTorch"
    data_dir: Path = Path("data")
    device: str = "auto"
    epochs: int = 3
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 2
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    val_fraction: float = 0.1
    split_seed: int = 13
    download: bool = True
    use_amp: bool = True
    label_smoothing: float = 0.0
    tta_horizontal_flip: bool = True
    log_every_epoch: bool = True


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path
    keep_raw_seed_metrics: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str = "simulator"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    torch: TorchBenchmarkConfig = field(default_factory=TorchBenchmarkConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
