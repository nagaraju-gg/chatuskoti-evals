from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from chatuskoti_evals.models import (
    AggregateSummary,
    BaselineRecord,
    CalibrationProfileSummary,
    FailureCaseResult,
    HistoryEntry,
    RunMetrics,
    to_jsonable,
)
from chatuskoti_evals.wisdom import WisdomStore


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
        final_canonical_metric: float,
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
            f"- Final accepted metric (controller's own eval): `{accepted_series[-1]:.4f}`",
            f"- Final canonical benchmark metric: `{final_canonical_metric:.4f}`",
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
                    "final_canonical_metric": round(final_canonical_metric, 5),
                    "non_adopt_actions": sum(1 for entry in history if entry.resolver_action != "adopt"),
                    "iterations": len(history),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return run_dir

    def write_comparison_report(
        self,
        baseline_metric: float,
        vec3_history: list[HistoryEntry],
        binary_history: list[HistoryEntry],
        vec3_final_metric: float,
        binary_final_metric: float,
        mode: str = "default",
    ) -> Path:
        output = self.root / "comparison.md"
        winner = "vec3" if vec3_final_metric > binary_final_metric else "binary" if binary_final_metric > vec3_final_metric else "tie"
        vec3_rejections = sum(1 for entry in vec3_history if entry.resolver_action in {"reject", "hold", "rollback", "reframe"})
        binary_rejections = sum(1 for entry in binary_history if entry.resolver_action == "reject")
        vec3_signals = sorted({signal for entry in vec3_history for signal in entry.run_score.fired_signals})
        binary_signals = sorted({signal for entry in binary_history for signal in entry.run_score.fired_signals})
        challenge_divergences = describe_challenge_divergences(vec3_history, binary_history) if mode == "challenge" else []

        if mode == "challenge":
            if winner == "vec3":
                verdict_line = "- Verdict: `Vec3` wins on canonical benchmark metric while also preserving TRV validity in challenge mode."
            elif winner == "binary":
                verdict_line = "- Verdict: `binary` is higher on canonical benchmark metric for this challenge run, but that metric must be read together with benchmark-aware invalid merges."
            else:
                verdict_line = "- Verdict: this challenge run is tied on canonical benchmark metric; structural validity is the more important differentiator."
        else:
            verdict_line = {
                "vec3": "- Verdict: `Vec3` currently wins on canonical benchmark metric for this run.",
                "binary": "- Verdict: `binary` currently wins on canonical benchmark metric for this run.",
                "tie": "- Verdict: this run is a tie on canonical benchmark metric.",
            }[winner]

        interpretation_lines = [
            "## Interpretation",
            "",
            verdict_line,
            f"- Vec3 final canonical metric: `{vec3_final_metric:.4f}`",
            f"- Binary final canonical metric: `{binary_final_metric:.4f}`",
            f"- Vec3 non-adopt actions: `{vec3_rejections}`",
            f"- Binary rejects: `{binary_rejections}`",
            f"- Vec3 fired signals: `{', '.join(vec3_signals) or 'none'}`",
            f"- Binary fired signals: `{', '.join(binary_signals) or 'none'}`",
            "",
        ]

        if mode == "challenge":
            interpretation_lines.extend(
                [
                    "## Readout",
                    "",
                    "- This challenge run is designed to test whether the controller distinguishes benchmark-aware bad merges from clean improvements.",
                    "- The main signal here is not final metric alone; it is whether each controller adopts or blocks pyrrhic, metric-gaming, and incomparable cases.",
                    "- Treat this as companion evidence to the canonical failure benchmark rather than as a plain unconstrained leaderboard comparison.",
                ]
            )
        elif winner == "vec3":
            interpretation_lines.extend(
                [
                    "## Readout",
                    "",
                    "- The richer controller logic is helping on the anchored benchmark, which is the behavior we want before scaling up.",
                    "- This run is a candidate for the paper/demo narrative, but it still needs repetition and calibration before publication.",
                ]
            )
        elif winner == "binary":
            interpretation_lines.extend(
                [
                    "## Readout",
                    "",
                    "- The real backend currently does not reproduce the simulator's intended advantage for Vec3.",
                    "- Treat this as calibration feedback, not as publishable evidence for the current thresholds or action semantics.",
                    "- The next step is to tune detector thresholds and intervention semantics on the real backend before making stronger claims.",
                ]
            )
        else:
            interpretation_lines.extend(
                [
                    "## Readout",
                    "",
                    "- Neither controller has a clear advantage yet on the anchored benchmark.",
                    "- This is a signal to tune thresholds and backend action implementations before scaling up.",
                ]
            )

        if mode == "challenge":
            interpretation_lines.extend(
                [
                    "",
                    "## Challenge-specific readout",
                    "",
                    f"- Binary adopted `{len(challenge_divergences)}` benchmark-aware cases that `Vec3` did not adopt."
                    if challenge_divergences
                    else "- Binary did not adopt any benchmark-aware cases that `Vec3` blocked.",
                ]
            )
            interpretation_lines.extend(f"- {item}" for item in challenge_divergences)
            interpretation_lines.extend(
                [
                    "- In `challenge` mode, final metric should be read together with structural validity, not as the only success criterion.",
                    "- A higher binary metric here can reflect merges that the benchmark is explicitly designed to classify as pyrrhic, invalid, or metric-gamed.",
                ]
            )

        body = [
            "# Binary vs Vec3 Comparison",
            "",
            f"- Mode: `{mode}`",
            f"- Baseline metric: `{baseline_metric:.4f}`",
            f"- Vec3 final canonical benchmark metric: `{vec3_final_metric:.4f}`",
            f"- Binary final canonical benchmark metric: `{binary_final_metric:.4f}`",
            "",
            *interpretation_lines,
        ]
        output.write_text("\n".join(body) + "\n", encoding="utf-8")
        (self.root / "comparison_summary.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "baseline_metric": round(baseline_metric, 5),
                    "vec3_final_metric": round(vec3_final_metric, 5),
                    "binary_final_metric": round(binary_final_metric, 5),
                    "challenge_divergence_count": len(challenge_divergences),
                    "challenge_divergences": challenge_divergences,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        write_bar_chart_svg(
            self.root / "controller_comparison.svg",
            "Final Accepted Metric",
            {"baseline": baseline_metric, "binary": binary_final_metric, "vec3": vec3_final_metric},
        )
        if mode == "challenge":
            write_challenge_table_markdown(self.root / "challenge_cases.md", vec3_history, binary_history)
            write_challenge_table_svg(self.root / "challenge_cases.svg", vec3_history, binary_history)
        return output

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
            "# Failure Injection Report",
            "",
            f"- Baseline metric: `{baseline.metrics.primary_metric:.4f}`",
            f"- Cases: `{len(results)}`",
            f"- Expectation matches: `{match_count}/{len(results)}`",
            f"- Binary would adopt: `{sum(1 for item in results if item.binary_resolution.action == 'adopt')}` cases",
            f"- Vec3 would adopt: `{sum(1 for item in results if item.resolution.action == 'adopt')}` cases",
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
                    f"- Binary action: `{item.binary_resolution.action}`",
                    f"- Binary reason: `{item.binary_resolution.reason}`",
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
            "# Failure Benchmark Ablations",
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

    def write_calibration_report(self, output_dir: Path, summaries: list[CalibrationProfileSummary]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            "# Threshold Calibration Sweep",
            "",
            "This report re-resolves the canonical failure benchmark under nearby detector thresholds.",
            "",
            "| Profile | Matched | Preserved vs default | Thresholds | Changed cases |",
            "| --- | --- | --- | --- | --- |",
        ]
        for summary in summaries:
            changed = ", ".join(summary.changed_cases) if summary.changed_cases else "none"
            rows.append(
                f"| `{summary.label}` | `{summary.matched_expectations}/{summary.total_cases}` | "
                f"`{summary.preserved_resolutions}/{summary.total_cases}` | "
                f"`{format_thresholds(summary.threshold_values)}` | {changed} |"
            )
        rows.extend(
            [
                "",
                "## Notes",
                "",
            ]
        )
        for summary in summaries:
            rows.append(f"- `{summary.label}`: {summary.notes}")
        path = output_dir / "summary.md"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(
            json.dumps([to_jsonable(item) for item in summaries], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_bar_chart_svg(
            output_dir / "threshold_sweep.svg",
            "Matched Failure Cases by Threshold Profile",
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


def describe_challenge_divergences(vec3_history: list[HistoryEntry], binary_history: list[HistoryEntry]) -> list[str]:
    divergences: list[str] = []
    for vec3_entry, binary_entry in zip(vec3_history, binary_history):
        if vec3_entry.action_spec.name != binary_entry.action_spec.name:
            continue
        if binary_entry.resolver_action == "adopt" and vec3_entry.resolver_action != "adopt":
            divergences.append(
                f"`{binary_entry.action_spec.name}`: binary `adopt` vs Vec3 `{vec3_entry.resolver_action}` "
                f"({', '.join(vec3_entry.run_score.fired_signals) or 'no explicit signals'})"
            )
    return divergences


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


def write_challenge_table_markdown(path: Path, vec3_history: list[HistoryEntry], binary_history: list[HistoryEntry]) -> None:
    lines = [
        "# Challenge Case Table",
        "",
        "| Action | Binary | Vec3 | Signals | Why It Matters |",
        "| --- | --- | --- | --- | --- |",
    ]
    for vec3_entry, binary_entry in zip(vec3_history, binary_history):
        if vec3_entry.action_spec.name != binary_entry.action_spec.name:
            continue
        lines.append(
            f"| `{vec3_entry.action_spec.name}` | `{binary_entry.resolver_action}` | `{vec3_entry.resolver_action}` | "
            f"`{', '.join(vec3_entry.run_score.fired_signals) or 'none'}` | {challenge_reason(vec3_entry)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_challenge_table_svg(path: Path, vec3_history: list[HistoryEntry], binary_history: list[HistoryEntry]) -> None:
    rows = []
    for vec3_entry, binary_entry in zip(vec3_history, binary_history):
        if vec3_entry.action_spec.name != binary_entry.action_spec.name:
            continue
        rows.append(
            (
                vec3_entry.action_spec.name,
                binary_entry.resolver_action,
                vec3_entry.resolver_action,
                ", ".join(vec3_entry.run_score.fired_signals) or "none",
                challenge_reason(vec3_entry),
            )
        )
    width = 1160
    height = 140 + 74 * max(len(rows), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="30" y="28" font-family="Helvetica" font-size="22" font-weight="700">Challenge Comparison Cases</text>',
        '<text x="30" y="52" font-family="Helvetica" font-size="12" fill="#444">Higher binary metric can come from merging benchmark-aware invalid cases.</text>',
        '<text x="30" y="84" font-family="Helvetica" font-size="12" font-weight="700">Action</text>',
        '<text x="220" y="84" font-family="Helvetica" font-size="12" font-weight="700">Binary</text>',
        '<text x="340" y="84" font-family="Helvetica" font-size="12" font-weight="700">Vec3</text>',
        '<text x="460" y="84" font-family="Helvetica" font-size="12" font-weight="700">Signals</text>',
        '<text x="760" y="84" font-family="Helvetica" font-size="12" font-weight="700">Consequence</text>',
    ]
    for index, row in enumerate(rows):
        y = 112 + 58 * index
        fill = "#f8fafc" if index % 2 == 0 else "#ffffff"
        lines.extend(
            [
                f'<rect x="24" y="{y - 18}" width="1110" height="44" fill="{fill}" stroke="#e5e7eb"/>',
                f'<text x="30" y="{y}" font-family="Helvetica" font-size="12">{row[0]}</text>',
                f'<text x="220" y="{y}" font-family="Helvetica" font-size="12">{row[1]}</text>',
                f'<text x="340" y="{y}" font-family="Helvetica" font-size="12">{row[2]}</text>',
                f'<text x="460" y="{y}" font-family="Helvetica" font-size="12">{escape_xml(row[3])}</text>',
                f'<text x="760" y="{y}" font-family="Helvetica" font-size="12">{escape_xml(row[4])}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def challenge_reason(entry: HistoryEntry) -> str:
    action = entry.resolver_action
    if action == "hold":
        return "Pyrrhic gain: metric improved while internals destabilized."
    if action == "reframe" and {"hyper_coherence", "proxy_decoupling"} & set(entry.run_score.fired_signals):
        return "Metric-gaming risk: validity collapsed despite a top-line gain."
    if action == "reject":
        return "Rejected as a non-improving or unstable change."
    if action == "reframe":
        return "Invalid comparison: the apparent gain is not decision-ready."
    if action == "rollback":
        return "Damaged run: controller should actively revert."
    return "Accepted as a clean change."


def format_components(components: dict[str, float]) -> str:
    if not components:
        return "none"
    return ", ".join(f"{key}={value:.3f}" for key, value in sorted(components.items()))


def format_thresholds(values: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.2f}" for key, value in values.items())


def escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_coupling_angle_svg(
    path: Path,
    coupling_results: list[dict[str, object]],
    tau: float = 0.4,
) -> None:
    width = 720
    height = 320
    pad = 40
    rpad = 120

    if not coupling_results:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="40" y="40" font-family="Helvetica" font-size="14">No coupling data</text></svg>'
        )
        return

    window_ends = [r["window_end"] for r in coupling_results]
    tv_vals = [r["t_v_coupling"] for r in coupling_results]
    tr_vals = [r["t_r_coupling"] for r in coupling_results]
    gw_flags = [r["goodhart_warning"] for r in coupling_results]

    all_vals = tv_vals + tr_vals + [tau, -tau, 0.0]
    min_y = min(all_vals) - 0.15
    max_y = max(all_vals) + 0.15
    if abs(max_y - min_y) < 1e-6:
        max_y = min_y + 1.0

    def px(index: int) -> float:
        n = len(window_ends)
        if n <= 1:
            return pad
        return pad + (width - pad - rpad) * index / (n - 1)

    def py(value: float) -> float:
        return height - pad - (value - min_y) / (max_y - min_y) * (height - 2 * pad)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad}" y="24" font-family="Helvetica" font-size="18" font-weight="700">Coupling Angle Over Window</text>',
        f'<text x="{pad}" y="44" font-family="Helvetica" font-size="12" fill="#666">T-V coupling (solid) crossing -τ triggers Goodhart warning</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-rpad}" y2="{height-pad}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333" stroke-width="1"/>',
    ]

    tau_y = py(-tau)
    lines.append(f'<line x1="{pad}" y1="{tau_y:.1f}" x2="{width-rpad}" y2="{tau_y:.1f}" stroke="#dc2626" stroke-dasharray="6,4" stroke-width="2"/>')
    lines.append(f'<text x="{width-rpad+6}" y="{tau_y+4:.1f}" font-family="Helvetica" font-size="11" fill="#dc2626">-τ = {-tau}</text>')

    zero_y = py(0.0)
    lines.append(f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width-rpad}" y2="{zero_y:.1f}" stroke="#999" stroke-dasharray="3,3" stroke-width="1"/>')

    tv_points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(tv_vals))
    lines.append(f'<polyline fill="none" stroke="#1d4ed8" stroke-width="3" points="{tv_points}"/>')

    tr_points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(tr_vals))
    lines.append(f'<polyline fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="4,3" points="{tr_points}"/>')

    for i, (flag, we) in enumerate(zip(gw_flags, window_ends)):
        if flag:
            cx = px(i)
            cy = py(tv_vals[i])
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#dc2626" stroke="#fff" stroke-width="2"/>')
            lines.append(f'<text x="{cx+8:.1f}" y="{cy+4:.1f}" font-family="Helvetica" font-size="11" fill="#dc2626">W{we}</text>')

    for i, we in enumerate(window_ends):
        x = px(i)
        if i % max(1, len(window_ends) // 6) == 0 or i == len(window_ends) - 1:
            lines.append(f'<text x="{x-6:.1f}" y="{height-pad+16}" font-family="Helvetica" font-size="11" fill="#333">{we}</text>')

    leg_x = width - rpad + 12
    leg_y = pad + 20
    lines.extend([
        f'<rect x="{width-rpad}" y="{pad}" width="{rpad-8}" height="90" rx="6" fill="#f8fafc" stroke="#e5e7eb"/>',
        f'<text x="{leg_x}" y="{leg_y}" font-family="Helvetica" font-size="12" font-weight="700">Legend</text>',
        f'<line x1="{leg_x}" y1="{leg_y+14}" x2="{leg_x+24}" y2="{leg_y+14}" stroke="#1d4ed8" stroke-width="3"/>',
        f'<text x="{leg_x+30}" y="{leg_y+18}" font-family="Helvetica" font-size="11">T-V coupling</text>',
        f'<line x1="{leg_x}" y1="{leg_y+36}" x2="{leg_x+24}" y2="{leg_y+36}" stroke="#059669" stroke-width="2" stroke-dasharray="4,3"/>',
        f'<text x="{leg_x+30}" y="{leg_y+40}" font-family="Helvetica" font-size="11">T-R coupling</text>',
        f'<circle cx="{leg_x+12}" cy="{leg_y+60}" r="6" fill="#dc2626" stroke="#fff" stroke-width="2"/>',
        f'<text x="{leg_x+30}" y="{leg_y+64}" font-family="Helvetica" font-size="11">Warning ⚠</text>',
    ])

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_lead_time_gap_svg(
    path: Path,
    first_warning: int | None,
    precheck_step: int | None,
    lead_time_steps: int | None,
    total_steps: int,
) -> None:
    width = 720
    height = 200
    pad = 40
    rpad = 30

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad}" y="28" font-family="Helvetica" font-size="18" font-weight="700">Lead-Time Gap</text>',
        f'<text x="{pad}" y="48" font-family="Helvetica" font-size="12" fill="#666">Timeline: coupling warning → Goodhart pre-check</text>',
    ]

    if first_warning is None or precheck_step is None:
        lines.extend([
            f'<text x="{width//2}" y="{height//2}" font-family="Helvetica" font-size="14" fill="#999" text-anchor="middle">Insufficient data — warning or pre-check never fired</text>',
            "</svg>",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    max_step = max(total_steps, precheck_step, first_warning) + 1
    if max_step < 2:
        max_step = 2
    effective_max = max(max_step, 10)

    def px(step: int) -> float:
        return pad + (width - pad - rpad) * step / effective_max

    track_y = height // 2 + 10
    line_y = track_y
    arrow_center_x = px((first_warning + precheck_step) / 2)

    lines.append(f'<line x1="{pad}" y1="{line_y}" x2="{width-rpad}" y2="{line_y}" stroke="#d1d5db" stroke-width="3"/>')

    warning_x = px(first_warning)
    precheck_x = px(precheck_step)

    lines.append(f'<circle cx="{warning_x:.1f}" cy="{line_y}" r="10" fill="#f59e0b" stroke="#fff" stroke-width="3"/>')
    lines.append(f'<text x="{warning_x:.1f}" y="{line_y+4}" font-family="Helvetica" font-size="12" font-weight="700" fill="#fff" text-anchor="middle">C</text>')
    lines.append(f'<text x="{warning_x:.1f}" y="{line_y+30}" font-family="Helvetica" font-size="13" font-weight="700" fill="#d97706" text-anchor="middle">Coupling Warning</text>')
    lines.append(f'<text x="{warning_x:.1f}" y="{line_y+46}" font-family="Helvetica" font-size="11" fill="#666" text-anchor="middle">step {first_warning}</text>')

    lines.append(f'<circle cx="{precheck_x:.1f}" cy="{line_y}" r="10" fill="#dc2626" stroke="#fff" stroke-width="3"/>')
    lines.append(f'<text x="{precheck_x:.1f}" y="{line_y+4}" font-family="Helvetica" font-size="12" font-weight="700" fill="#fff" text-anchor="middle">G</text>')
    lines.append(f'<text x="{precheck_x:.1f}" y="{line_y+30}" font-family="Helvetica" font-size="13" font-weight="700" fill="#dc2626" text-anchor="middle">Goodhart Pre-check</text>')
    lines.append(f'<text x="{precheck_x:.1f}" y="{line_y+46}" font-family="Helvetica" font-size="11" fill="#666" text-anchor="middle">step {precheck_step}</text>')

    gap = lead_time_steps if lead_time_steps is not None else 0
    if gap > 0:
        arrow_y = line_y - 32
        lines.append(f'<line x1="{warning_x:.1f}" y1="{arrow_y}" x2="{precheck_x:.1f}" y2="{arrow_y}" stroke="#059669" stroke-width="2" marker-end="url(#arrow)"/>')
        lines.append(f'<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#059669"/></marker></defs>')
        lines.append(f'<text x="{arrow_center_x:.1f}" y="{arrow_y-8}" font-family="Helvetica" font-size="14" font-weight="700" fill="#059669" text-anchor="middle">+{gap} steps lead time</text>')
    elif gap == 0:
        lines.append(f'<text x="{precheck_x:.1f}" y="{line_y-24}" font-family="Helvetica" font-size="14" font-weight="700" fill="#d97706" text-anchor="middle">Simultaneous (0 lead)</text>')
    else:
        lines.append(f'<text x="{arrow_center_x:.1f}" y="{line_y-24}" font-family="Helvetica" font-size="14" fill="#dc2626" text-anchor="middle">Pre-check before warning ({gap} steps)</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")

def write_lead_time_report(
    output_dir: Path,
    lead_result: dict[str, object],
    history_scores: list,
    history_entries: list,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    deltas = lead_result.get("deltas", [])
    coupling_results = lead_result.get("coupling_results", [])
    precheck_step = lead_result.get("precheck_step")
    first_warning = lead_result.get("first_coupling_warning_step")
    lead_time_steps = lead_result.get("lead_time_steps")
    total_steps = lead_result["total_steps"]

    summary_lines = [
        "# Lead-Time Analysis",
        "",
        f"- Total steps observed: `{total_steps}`",
        f"- Coupling window: `{lead_result['window']}`, coupling threshold τ: `{lead_result['tau']}`",
        f"- Goodhart pre-check fired at step: `{precheck_step}`" if precheck_step is not None else "- Goodhart pre-check: `never fired`",
        f"- First coupling warning at window end: `{first_warning}`" if first_warning is not None else "- First coupling warning: `never fired`",
        f"- Lead time (steps): `{lead_time_steps}`" if lead_time_steps is not None else "- Lead time: `N/A`",
        "",
        "## Per-step Deltas",
        "",
        "| Step | ΔT | ΔR | ΔV |",
        "| --- | --- | --- | --- |",
    ]
    for d in deltas:
        summary_lines.append(
            f"| {d['step']} | `{d['delta_T']:.5f}` | `{d['delta_R']:.5f}` | `{d['delta_V']:.5f}` |"
        )

    if coupling_results:
        summary_lines.extend(
            [
                "",
                "## Sliding-Window Coupling",
                "",
                "| Window End | T-V Coupling | T-R Coupling | Goodhart Warning | Pyrrhic Warning |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for r in coupling_results:
            gw = "🚨" if r["goodhart_warning"] else "—"
            pw = "🚨" if r["pyrrhic_warning"] else "—"
            summary_lines.append(
                f"| {r['window_end']} | `{r['t_v_coupling']:.5f}` | `{r['t_r_coupling']:.5f}` | {gw} | {pw} |"
            )

    summary_lines.append("")
    if lead_time_steps is not None and lead_time_steps > 0:
        summary_lines.append(f"**Result**: Coupling warning preceded Goodhart pre-check by **{lead_time_steps} step(s)**.")
        summary_lines.append("This confirms §4.2's prediction: the geometric coupling signal has measurable lead time over snapshot monitoring.")
    elif first_warning is not None and precheck_step is not None and lead_time_steps == 0:
        summary_lines.append("**Result**: Coupling warning and Goodhart pre-check fired simultaneously (lead time = 0).")
    else:
        summary_lines.append("**Result**: No measurable lead time — coupling warning did not precede Goodhart pre-check.")

    svg_series: dict[str, list[float]] = {"T": [], "R": [], "V": []}
    for score in history_scores:
        svg_series["T"].append(score.mean.truthness)
        svg_series["R"].append(score.mean.reliability)
        svg_series["V"].append(score.mean.validity)
    write_line_chart_svg(output_dir / "trv_trajectory.svg", "TRV Trajectory (Progressive Goodhart Approach)", svg_series)

    write_coupling_angle_svg(output_dir / "coupling_angle.svg", coupling_results, tau=lead_result.get("tau", 0.4))

    write_lead_time_gap_svg(
        output_dir / "lead_time_gap.svg",
        first_warning=first_warning,
        precheck_step=precheck_step,
        lead_time_steps=lead_time_steps,
        total_steps=total_steps,
    )

    path = output_dir / "summary.md"
    path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return path


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
