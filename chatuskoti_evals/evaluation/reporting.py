from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from chatuskoti_evals.core.models import (
    AggregateSummary,
    BaselineRecord,
    FailureCaseResult,
    HistoryEntry,
    RunMetrics,
    to_jsonable,
)
from chatuskoti_evals.core.wisdom import WisdomStore


class ReportGenerator:
    def __init__(self, root: Path):
        self.root = root

    def write_loop_artifacts(
        self,
        controller: str,
        initial_baseline: BaselineRecord,
        final_baseline: BaselineRecord,
        history: list[HistoryEntry],
        per_iteration_metrics: list[list[RunMetrics]],
        wisdom: WisdomStore,
        accepted_metric: float,
    ) -> Path:
        run_dir = self.root / controller
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "history.jsonl").write_text("", encoding="utf-8")
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            for entry in history:
                handle.write(json.dumps(to_jsonable(entry), sort_keys=True) + "\n")

        seed_payload = {
            f"iteration_{index + 1}": [to_jsonable(item) for item in items]
            for index, items in enumerate(per_iteration_metrics)
        }
        (run_dir / "seed_metrics.json").write_text(json.dumps(seed_payload, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "wisdom.json").write_text(json.dumps(wisdom.snapshot(), indent=2, sort_keys=True), encoding="utf-8")

        metric_series = [initial_baseline.metrics.primary_metric]
        accepted_series = [initial_baseline.metrics.primary_metric]
        for metrics, entry in zip(per_iteration_metrics, history):
            metric_series.append(round(sum(item.primary_metric for item in metrics) / len(metrics), 5))
            accepted_series.append(entry.accepted_primary_metric if entry.accepted_primary_metric is not None else accepted_series[-1])
        write_line_chart_svg(run_dir / "metric_trajectory.svg", "Metric Trajectory", {"candidate": metric_series, "accepted": accepted_series})

        action_counts = Counter(entry.resolver_action for entry in history)
        write_bar_chart_svg(run_dir / "action_counts.svg", "Resolver Actions", dict(action_counts))

        octant_counts = Counter(classify_region(entry) for entry in history)
        write_bar_chart_svg(run_dir / "outcome_regions.svg", "Outcome Regions", dict(octant_counts))

        summary_lines = [
            f"# {controller.upper()} Controller Report",
            "",
            f"- Initial baseline metric: `{initial_baseline.metrics.primary_metric:.4f}`",
            f"- Final accepted metric: `{accepted_series[-1]:.4f}`",
            f"- Final baseline metric after adoptions: `{final_baseline.metrics.primary_metric:.4f}`",
            f"- Iterations: `{len(history)}`",
            f"- Fired signals seen: `{', '.join(sorted({signal for entry in history for signal in entry.run_score.fired_signals})) or 'none'}`",
            "",
            "## Iterations",
        ]
        for entry in history:
            summary_lines.extend(
                [
                    f"### Iteration {entry.iteration}: `{entry.action_spec.name}`",
                    f"- Action: `{entry.resolver_action}`",
                    f"- Why: `{entry.resolver_reason}`",
                    f"- TRV: `({entry.run_score.mean.truthness:.3f}, {entry.run_score.mean.reliability:.3f}, {entry.run_score.mean.validity:.3f})`",
                    f"- Reliability components: `{format_components(entry.run_score.axis_components.get('reliability', {}))}`",
                    f"- Validity components: `{format_components(entry.run_score.axis_components.get('validity', {}))}`",
                    f"- Signals: `{', '.join(entry.run_score.fired_signals) or 'none'}`",
                ]
            )
        (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        (run_dir / "aggregate_summary.json").write_text(
            json.dumps(
                {
                    "controller": controller,
                    "initial_baseline_metric": round(initial_baseline.metrics.primary_metric, 5),
                    "final_accepted_metric": round(accepted_metric, 5),
                    "non_adopt_actions": sum(1 for entry in history if entry.resolver_action != "adopt"),
                    "iterations": len(history),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return run_dir

    def write_failure_injection_report(
        self,
        output_dir: Path,
        baseline: BaselineRecord,
        results: list[FailureCaseResult],
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure_results.json").write_text(
            json.dumps([to_jsonable(item) for item in results], indent=2, sort_keys=True),
            encoding="utf-8",
        )

        match_count = sum(1 for item in results if item.matched_expectation)
        action_counts = Counter(item.resolution.action for item in results)
        write_bar_chart_svg(output_dir / "failure_actions.svg", "Failure Set Actions", dict(action_counts))

        summary_lines = [
            "# Failure Case Report",
            "",
            f"- Baseline metric: `{baseline.metrics.primary_metric:.4f}`",
            f"- Cases: `{len(results)}`",
            f"- Expectation matches: `{match_count}/{len(results)}`",
            "",
            "## Cases",
        ]
        for item in results:
            summary_lines.extend(
                [
                    f"### `{item.scenario_name}` via `{item.action_spec.name}`",
                    f"- Narrative: {item.narrative}",
                    f"- Candidate metric: `{item.candidate_metric:.4f}`",
                    f"- Expected resolution: `{item.expected_resolution}`",
                    f"- Actual resolution: `{item.resolution.action}`",
                    f"- Match: `{item.matched_expectation}`",
                    f"- Resolver reason: `{item.resolution.reason}`",
                    f"- Expected signals: `{', '.join(item.expected_signals) or 'none'}`",
                    f"- Actual signals: `{', '.join(item.run_score.fired_signals) or 'none'}`",
                    f"- TRV: `({item.run_score.mean.truthness:.3f}, {item.run_score.mean.reliability:.3f}, {item.run_score.mean.validity:.3f})`",
                    f"- Reliability components: `{format_components(item.run_score.axis_components.get('reliability', {}))}`",
                    f"- Validity components: `{format_components(item.run_score.axis_components.get('validity', {}))}`",
                ]
            )
        path = output_dir / "summary.md"
        path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        aggregate = aggregate_failure_results("full", results)
        (output_dir / "aggregate_summary.json").write_text(
            json.dumps(to_jsonable(aggregate), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def write_ablation_report(self, output_dir: Path, summaries: list[AggregateSummary]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            "# Ablation Report: Representation Sufficiency",
            "",
            "Removing axes tests whether scalar evaluation loses distinctions required for correct decisions.",
            "",
            "| Ablation | Matched | Mean Metric | Mean Truth | Mean Reliability | Mean Validity |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for summary in summaries:
            rows.append(
                f"| `{summary.label}` | `{summary.matched_expectations}/{summary.total_cases}` | "
                f"`{summary.mean_primary_metric:.4f}` | `{summary.mean_truthness:.3f}` | `{summary.mean_reliability:.3f}` | "
                f"`{summary.mean_validity:.3f}` |"
            )
        path = output_dir / "summary.md"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps([to_jsonable(item) for item in summaries], indent=2, sort_keys=True), encoding="utf-8")
        write_bar_chart_svg(
            output_dir / "ablation_summary.svg",
            "Matched Failure Cases by Ablation",
            {summary.label: summary.matched_expectations for summary in summaries},
        )
        return path


def classify_region(entry: HistoryEntry) -> str:
    t = entry.run_score.mean.truthness
    r = entry.run_score.mean.reliability
    v = entry.run_score.mean.validity
    if v < 0 and "eval_regime_changed" in entry.run_score.fired_signals:
        return "invalid_comparison"
    if v < 0:
        return "metric_gaming"
    if t > 0 and r < 0:
        return "pyrrhic"
    if t < 0 and r < 0:
        return "broken"
    if t > 0:
        return "clean_win"
    return "clean_failure"


def aggregate_failure_results(label: str, results: list[FailureCaseResult]) -> AggregateSummary:
    primary_metrics = [item.candidate_metric for item in results]
    truthnesses = [item.run_score.mean.truthness for item in results]
    reliabilities = [item.run_score.mean.reliability for item in results]
    validities = [item.run_score.mean.validity for item in results]
    return AggregateSummary(
        label=label,
        mean_primary_metric=round(mean(primary_metrics), 5),
        std_primary_metric=round(pstdev(primary_metrics) if len(primary_metrics) > 1 else 0.0, 5),
        mean_truthness=round(mean(truthnesses), 5),
        mean_reliability=round(mean(reliabilities), 5),
        mean_validity=round(mean(validities), 5),
        matched_expectations=sum(1 for item in results if item.matched_expectation),
        total_cases=len(results),
    )


def format_components(components: dict[str, float]) -> str:
    if not components:
        return "none"
    return ", ".join(f"{key}={value:.3f}" for key, value in sorted(components.items()))


def escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_line_chart_svg(path: Path, title: str, series: dict[str, list[float]]) -> None:
    width = 720
    height = 320
    padding = 40
    all_values = [value for values in series.values() for value in values]
    min_y = min(all_values) - 0.01
    max_y = max(all_values) + 0.01
    colors = ["#1d4ed8", "#dc2626", "#059669", "#9333ea"]

    def project_x(index: int, total: int) -> float:
        if total <= 1:
            return padding
        return padding + (width - 2 * padding) * index / (total - 1)

    def project_y(value: float) -> float:
        if abs(max_y - min_y) < 1e-8:
            return height / 2
        scale = (value - min_y) / (max_y - min_y)
        return height - padding - scale * (height - 2 * padding)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{padding}" y="24" font-size="18" font-family="Helvetica">{title}</text>',
        f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#333"/>',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#333"/>',
    ]
    for index, (name, values) in enumerate(series.items()):
        points = " ".join(f"{project_x(i, len(values)):.1f},{project_y(value):.1f}" for i, value in enumerate(values))
        color = colors[index % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        lines.append(f'<text x="{width - 180}" y="{32 + index * 18}" font-size="12" font-family="Helvetica" fill="{color}">{name}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bar_chart_svg(path: Path, title: str, values: dict[str, float | int]) -> None:
    width = 720
    height = 320
    padding = 40
    max_value = max(values.values(), default=1)
    bar_width = max(40, int((width - 2 * padding) / max(len(values), 1) * 0.6))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{padding}" y="24" font-size="18" font-family="Helvetica">{title}</text>',
        f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(values.items()):
        x = padding + index * ((width - 2 * padding) / max(len(values), 1))
        usable_height = height - 2 * padding
        bar_height = 0 if max_value == 0 else usable_height * (float(value) / float(max_value))
        y = height - padding - bar_height
        lines.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#2563eb"/>',
                f'<text x="{x:.1f}" y="{height-padding+16}" font-size="11" font-family="Helvetica">{label}</text>',
                f'<text x="{x:.1f}" y="{max(36, y-4):.1f}" font-size="11" font-family="Helvetica">{value}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
