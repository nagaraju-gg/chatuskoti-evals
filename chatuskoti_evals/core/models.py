from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Vec3:
    truthness: float
    reliability: float
    validity: float


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    seed: int
    primary_metric: float
    train_loss: float
    val_loss: float
    train_val_gap: float
    grad_norm_mean: float
    grad_norm_std: float
    weight_distance: float
    param_count: int
    eval_hash: str
    model_family: str
    objective_family: str
    proxy_metrics: dict[str, float]
    detector_inputs: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedScore:
    vec3: Vec3
    fired_signals: list[str]
    raw_detectors: dict[str, float | str | bool]
    axis_components: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RunScore:
    mean: Vec3
    std: Vec3
    mag: float
    spread: float
    fired_signals: list[str]
    raw_detectors: dict[str, float]
    axis_components: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    family: str
    params: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class BaselineRecord:
    baseline_id: str
    metrics: RunMetrics


@dataclass(frozen=True)
class HistoryEntry:
    iteration: int
    timestamp: str
    controller: str
    action_spec: ActionSpec
    baseline_id: str
    run_ids: list[str]
    run_score: RunScore
    resolver_action: str
    resolver_reason: str
    depth: float
    width: int
    accepted_primary_metric: float | None


@dataclass(frozen=True)
class Resolution:
    action: str
    reason: str


@dataclass(frozen=True)
class FailureCaseResult:
    scenario_name: str
    action_spec: ActionSpec
    expected_signals: list[str]
    expected_resolution: str
    narrative: str
    candidate_metric: float
    run_score: RunScore
    resolution: Resolution
    binary_resolution: Resolution
    matched_expectation: bool


@dataclass(frozen=True)
class BundleManifest:
    schema_version: int
    bundle_name: str
    release_label: str
    artifact_kind: str
    generated_at: str
    package_version: str
    git_commit: str
    benchmark: str
    backend: str
    seeds: int
    epochs: int
    controller_mode: str
    ablation: str
    benchmark_spec_id: str
    benchmark_spec: dict[str, Any]
    detector_config: dict[str, Any]
    backend_config: dict[str, Any]
    artifact_paths: dict[str, str]


@dataclass(frozen=True)
class AggregateSummary:
    label: str
    mean_primary_metric: float
    std_primary_metric: float
    mean_truthness: float
    mean_reliability: float
    mean_validity: float
    matched_expectations: int
    total_cases: int


@dataclass(frozen=True)
class CalibrationProfileSummary:
    label: str
    notes: str
    matched_expectations: int
    total_cases: int
    preserved_resolutions: int
    threshold_values: dict[str, float]
    changed_cases: list[str]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(inner) for key, inner in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(inner) for inner in value]
    if isinstance(value, tuple):
        return [to_jsonable(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value
