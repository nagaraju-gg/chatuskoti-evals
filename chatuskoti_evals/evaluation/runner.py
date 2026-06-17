from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatuskoti_evals import __version__
from chatuskoti_evals.core.config import AblationConfig, DetectorConfig, ExperimentConfig, LoopConfig
from chatuskoti_evals.core.models import (
    AggregateSummary,
    BaselineRecord,
    BundleManifest,
    FailureCaseResult,
    HistoryEntry,
    RunMetrics,
    to_jsonable,
)
from chatuskoti_evals.core.wisdom import WisdomStore
from chatuskoti_evals.evaluation.actions import ACTION_INDEX
from chatuskoti_evals.evaluation.benchmark import create_benchmark_adapter
from chatuskoti_evals.evaluation.progress import RunProgressContext, RunProgressTracker
from chatuskoti_evals.evaluation.proposals import ProposalEngine
from chatuskoti_evals.evaluation.reporting import ReportGenerator, aggregate_failure_results
from chatuskoti_evals.evaluation.resolver import resolve_binary, resolve_vec3
from chatuskoti_evals.evaluation.scenarios import get_failure_injection_set
from chatuskoti_evals.evaluation.scoring import score_run_metrics


@dataclass(frozen=True)
class LoopResult:
    controller: str
    initial_baseline: BaselineRecord
    final_baseline: BaselineRecord
    history: list[HistoryEntry]
    accepted_metric: float
    raw_accepted_metric: float
    output_dir: Path


@dataclass(frozen=True)
class FailureCaseExecution:
    scenario_name: str
    action_name: str
    expected_signals: list[str]
    expected_resolution: str
    narrative: str
    candidate_metrics: list[RunMetrics]


def run_single_loop(
    cfg: ExperimentConfig,
    loop_cfg: LoopConfig,
    output_root: Path,
    *,
    progress: RunProgressTracker | None = None,
    seeds: list[int] | None = None,
) -> LoopResult:
    detector_cfg = cfg.ablation.apply(cfg.detector)
    adapter = create_benchmark_adapter(cfg)
    proposal_engine = ProposalEngine()
    report_generator = ReportGenerator(output_root)
    controller_root = output_root / loop_cfg.controller
    controller_root.mkdir(parents=True, exist_ok=True)

    wisdom_path = controller_root / "wisdom_store.json"
    wisdom = WisdomStore.load(wisdom_path) if cfg.ablation.wisdom_enabled else WisdomStore()
    seeds = list(range(loop_cfg.n_seeds)) if seeds is None else seeds
    progress = progress or RunProgressTracker(total_runs=_loop_run_count(loop_cfg))
    run_log_path = _run_log_path()

    initial_baseline = adapter.record_baseline(
        seeds,
        progress=progress,
        progress_context=RunProgressContext(controller=loop_cfg.controller, phase="baseline"),
    )
    current_baseline = initial_baseline
    history: list[HistoryEntry] = []
    per_iteration_metrics: list[list[RunMetrics]] = []
    raw_accepted_metric = current_baseline.metrics.primary_metric

    for iteration in range(1, loop_cfg.max_iterations + 1):
        action = proposal_engine.propose(loop_cfg.controller, history, wisdom, mode=loop_cfg.mode)
        candidate_metrics, candidate_state = adapter.execute(
            action,
            seeds,
            progress=progress,
            progress_context=RunProgressContext(
                controller=loop_cfg.controller,
                phase="iteration",
                iteration=iteration,
                action_name=action.name,
            ),
        )
        compared_baseline_id = current_baseline.baseline_id
        active_wisdom = wisdom if cfg.ablation.wisdom_enabled else WisdomStore()
        run_score, _ = score_run_metrics(candidate_metrics, current_baseline.metrics, detector_cfg)

        if loop_cfg.controller == "vec3":
            resolution = resolve_vec3(run_score, detector_cfg)
        else:
            resolution = resolve_binary(candidate_metrics, current_baseline.metrics, detector_cfg)

        _append_run_log(
            run_log_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            backend=cfg.backend,
            controller=loop_cfg.controller,
            action_name=action.name,
            baseline_id=compared_baseline_id,
            run_score=run_score,
            resolution=resolution,
            ablation=cfg.ablation.normalized_name,
        )

        if resolution.action == "adopt":
            adapter.adopt(candidate_state)
            progress.add_runs(len(seeds))
            current_baseline = adapter.record_baseline(
                seeds,
                progress=progress,
                progress_context=RunProgressContext(
                    controller=loop_cfg.controller,
                    phase="adopted_baseline",
                    iteration=iteration,
                    action_name=action.name,
                ),
            )
            raw_accepted_metric = current_baseline.metrics.primary_metric

        entry = HistoryEntry(
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            controller=loop_cfg.controller,
            action_spec=action,
            baseline_id=compared_baseline_id,
            run_ids=[item.run_id for item in candidate_metrics],
            run_score=run_score,
            resolver_action=resolution.action,
            resolver_reason=resolution.reason,
            depth=loop_cfg.depth,
            width=loop_cfg.width,
            accepted_primary_metric=raw_accepted_metric if resolution.action == "adopt" else None,
        )
        history.append(entry)
        per_iteration_metrics.append(candidate_metrics)
        if cfg.ablation.wisdom_enabled:
            active_wisdom.update(action.family, run_score)
            active_wisdom.save(wisdom_path)
            wisdom = active_wisdom

    accepted_metric = adapter.primary_metric()
    report_dir = report_generator.write_loop_artifacts(
        controller=loop_cfg.controller,
        initial_baseline=initial_baseline,
        final_baseline=current_baseline,
        history=history,
        per_iteration_metrics=per_iteration_metrics,
        wisdom=wisdom,
        accepted_metric=accepted_metric,
    )
    (report_dir / "config.json").write_text(
        json.dumps(
            {
                "loop": to_jsonable(loop_cfg),
                "detector": to_jsonable(detector_cfg),
                "ablation": cfg.ablation.name,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return LoopResult(
        controller=loop_cfg.controller,
        initial_baseline=initial_baseline,
        final_baseline=current_baseline,
        history=history,
        accepted_metric=accepted_metric,
        raw_accepted_metric=raw_accepted_metric,
        output_dir=report_dir,
    )


def run_failure_injection_set(
    output_root: Path,
    cfg: ExperimentConfig | None = None,
    *,
    seeds: int = 1,
) -> list[FailureCaseResult]:
    cfg = cfg or ExperimentConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    detector_cfg = cfg.ablation.apply(cfg.detector)
    report_generator = ReportGenerator(output_root)
    baseline, executions = _collect_failure_case_executions(cfg, seeds)
    results = _score_failure_case_executions(executions, baseline, detector_cfg)
    report_dir = output_root / "failure_injection"
    report_generator.write_failure_injection_report(report_dir, baseline, results)
    _write_manifest(
        output_root / "manifest.json",
        _build_manifest(
            cfg,
            bundle_name=output_root.name,
            artifact_kind="failure_injection",
            seeds=seeds,
            controller_mode="failure_injection",
            ablation=cfg.ablation.name,
            detector_cfg=detector_cfg,
            artifact_paths={
                "summary": "failure_injection/summary.md",
                "summary_json": "failure_injection/aggregate_summary.json",
                "results_json": "failure_injection/failure_results.json",
            },
        ),
    )
    return results


def run_ablation_bundle(
    output_root: Path,
    cfg: ExperimentConfig | None = None,
    *,
    seeds: int = 3,
    ablations: tuple[str, ...] = ("full", "no_reliability", "no_validity", "no_wisdom", "no_spread_gate", "t_only", "t_r", "t_v", "t_r_v"),
) -> list[AggregateSummary]:
    cfg = cfg or ExperimentConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline, executions = _collect_failure_case_executions(cfg, seeds)
    report_generator = ReportGenerator(output_root)
    summaries: list[AggregateSummary] = []

    for ablation_name in ablations:
        ablated_cfg = replace(cfg, ablation=AblationConfig(name=ablation_name))
        detector_cfg = ablated_cfg.ablation.apply(ablated_cfg.detector)
        results = _score_failure_case_executions(executions, baseline, detector_cfg)
        normalized_ablation = ablated_cfg.ablation.normalized_name
        report_generator.write_failure_injection_report(output_root / normalized_ablation / "failure_injection", baseline, results)
        summaries.append(aggregate_failure_results(normalized_ablation, results))

    report_generator.write_ablation_report(output_root, summaries)
    _write_manifest(
        output_root / "manifest.json",
        _build_manifest(
            cfg,
            bundle_name=output_root.name,
            artifact_kind="ablation_bundle",
            seeds=seeds,
            controller_mode="failure_injection",
            ablation="bundle",
            detector_cfg=cfg.detector,
            artifact_paths={
                "summary": "summary.md",
                "summary_json": "summary.json",
                "summary_svg": "ablation_summary.svg",
            },
        ),
    )
    return summaries


def _collect_failure_case_executions(cfg: ExperimentConfig, seeds: int) -> tuple[BaselineRecord, list[FailureCaseExecution]]:
    adapter = create_benchmark_adapter(cfg)
    failure_scenarios = get_failure_injection_set(cfg.backend)
    progress = RunProgressTracker(total_runs=seeds * (len(failure_scenarios) + 1))
    baseline = adapter.record_baseline(
        list(range(seeds)),
        progress=progress,
        progress_context=RunProgressContext(controller="failure_injection", phase="baseline"),
    )
    executions: list[FailureCaseExecution] = []
    for iteration, scenario in enumerate(failure_scenarios, start=1):
        action = ACTION_INDEX[scenario.action_name]
        candidate_metrics, _ = adapter.execute(
            action,
            list(range(seeds)),
            progress=progress,
            progress_context=RunProgressContext(
                controller="failure_injection",
                phase="scenario",
                iteration=iteration,
                action_name=scenario.action_name,
            ),
        )
        executions.append(
            FailureCaseExecution(
                scenario_name=scenario.name,
                action_name=scenario.action_name,
                expected_signals=list(scenario.expected_signals),
                expected_resolution=scenario.expected_resolution,
                narrative=scenario.narrative,
                candidate_metrics=candidate_metrics,
            )
        )
    return baseline, executions


def _score_failure_case_executions(
    executions: list[FailureCaseExecution],
    baseline: BaselineRecord,
    detector_cfg,
    *,
    write_run_log: bool = True,
) -> list[FailureCaseResult]:
    results: list[FailureCaseResult] = []
    for execution in executions:
        action = ACTION_INDEX[execution.action_name]
        run_score, _ = score_run_metrics(execution.candidate_metrics, baseline.metrics, detector_cfg)
        resolution = resolve_vec3(run_score, detector_cfg)
        binary_resolution = resolve_binary(execution.candidate_metrics, baseline.metrics, detector_cfg)
        mean_metric = sum(item.primary_metric for item in execution.candidate_metrics) / len(execution.candidate_metrics)
        matched_signals = all(signal in run_score.fired_signals for signal in execution.expected_signals)
        matched_resolution = resolution.action == execution.expected_resolution
        results.append(
            FailureCaseResult(
                scenario_name=execution.scenario_name,
                action_spec=action,
                expected_signals=list(execution.expected_signals),
                expected_resolution=execution.expected_resolution,
                narrative=execution.narrative,
                candidate_metric=round(mean_metric, 5),
                run_score=run_score,
                resolution=resolution,
                binary_resolution=binary_resolution,
                matched_expectation=matched_signals and matched_resolution,
            )
        )
        if write_run_log:
            _append_run_log(
                _run_log_path(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                backend="failure_injection",
                controller="vec3",
                action_name=execution.action_name,
                baseline_id=baseline.baseline_id,
                run_score=run_score,
                resolution=resolution,
                ablation=getattr(detector_cfg, "ablation_name", "full"),
            )
    return results


def _build_manifest(
    cfg: ExperimentConfig,
    *,
    bundle_name: str,
    artifact_kind: str,
    seeds: int,
    controller_mode: str,
    ablation: str,
    detector_cfg: DetectorConfig,
    artifact_paths: dict[str, str],
) -> BundleManifest:
    backend_config = _backend_config(cfg)
    return BundleManifest(
        schema_version=2,
        bundle_name=bundle_name,
        release_label=f"v{__version__}",
        artifact_kind=artifact_kind,
        generated_at=datetime.now(timezone.utc).isoformat(),
        package_version=__version__,
        git_commit=_git_commit(),
        backend=cfg.backend,
        seeds=seeds,
        epochs=_epoch_count(cfg),
        controller_mode=controller_mode,
        ablation=ablation,
        detector_config=to_jsonable(detector_cfg),
        backend_config=backend_config,
        artifact_paths=artifact_paths,
    )


def _write_manifest(path: Path, manifest: BundleManifest) -> None:
    path.write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")


def _epoch_count(cfg: ExperimentConfig) -> int:
    return 0


def _loop_run_count(loop_cfg: LoopConfig) -> int:
    return loop_cfg.n_seeds * (loop_cfg.max_iterations + 1)


def _backend_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return to_jsonable(cfg.simulation)


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _append_run_log(
    path: Path,
    *,
    timestamp: str,
    backend: str,
    controller: str,
    action_name: str,
    baseline_id: str,
    run_score,
    resolution,
    ablation: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "timestamp": timestamp,
        "backend": backend,
        "controller": controller,
        "action_name": action_name,
        "baseline_id": baseline_id,
        "ablation": ablation,
        "features": run_score.raw_detectors,
        "axis_components": run_score.axis_components,
        "T": run_score.mean.truthness,
        "R": run_score.mean.reliability,
        "V": run_score.mean.validity,
        "spread": run_score.spread,
        "resolver_action": resolution.action,
        "resolver_reason": resolution.reason,
        "fired_signals": run_score.fired_signals,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _run_log_path() -> Path:
    override = os.environ.get("CHATUSKOTI_RUN_LOG_PATH")
    if override:
        return Path(override)
    return Path("_output") / "logs" / "runs.jsonl"
