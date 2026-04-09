from __future__ import annotations

import json
import sys
from pathlib import Path

from chatuskoti_evals.config import AblationConfig, DetectorConfig
from chatuskoti_evals.models import RunScore, Vec3
from chatuskoti_evals.reporting import write_bar_chart_svg
from chatuskoti_evals.resolver import resolve_vec3


ABLATIONS = ("full", "no_reliability", "no_validity", "no_wisdom", "no_spread_gate")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python3 scripts/generate_ablation_report.py <failure_results.json> <output_dir>")
        return 1

    source = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(source.read_text(encoding="utf-8"))
    summary_rows = []
    matched_counts: dict[str, int] = {}

    for ablation in ABLATIONS:
        detector = AblationConfig(name=ablation).apply(DetectorConfig())
        row = score_ablation(results, detector, ablation)
        summary_rows.append(row)
        matched_counts[ablation] = row["matched_expectations"]

    write_summary_markdown(output_dir / "summary.md", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    write_bar_chart_svg(output_dir / "ablation_summary.svg", "Matched Failure Cases by Ablation", matched_counts)
    print(f"Wrote ablation report to {output_dir}")
    return 0


def score_ablation(results: list[dict], detector: DetectorConfig, label: str) -> dict[str, object]:
    cases: list[dict[str, str]] = []
    matched = 0
    for item in results:
        run_score = to_run_score(item["run_score"])
        resolution = resolve_vec3(run_score, detector)
        expected_signals = item["expected_signals"]
        actual_signals = item["run_score"]["fired_signals"]
        matched_expectation = all(signal in actual_signals for signal in expected_signals) and resolution.action == item["expected_resolution"]
        if matched_expectation:
            matched += 1
        cases.append(
            {
                "case": item["scenario_name"],
                "expected": item["expected_resolution"],
                "actual": resolution.action,
            }
        )
    return {
        "ablation": label,
        "matched_expectations": matched,
        "total_cases": len(results),
        "cases": cases,
    }


def to_run_score(payload: dict[str, object]) -> RunScore:
    mean_payload = payload["mean"]
    std_payload = payload["std"]
    return RunScore(
        mean=Vec3(
            truthness=float(mean_payload["truthness"]),
            reliability=float(mean_payload["reliability"]),
            validity=float(mean_payload["validity"]),
        ),
        std=Vec3(
            truthness=float(std_payload["truthness"]),
            reliability=float(std_payload["reliability"]),
            validity=float(std_payload["validity"]),
        ),
        mag=float(payload["mag"]),
        spread=float(payload["spread"]),
        fired_signals=list(payload["fired_signals"]),
        raw_detectors={key: float(value) for key, value in payload.get("raw_detectors", {}).items()},
        axis_components={
            axis_name: {key: float(value) for key, value in values.items()}
            for axis_name, values in payload.get("axis_components", {}).items()
        },
    )


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Failure Benchmark Ablations",
        "",
        "This report re-resolves the saved canonical failure benchmark under ablated detector settings.",
        "",
        "| Ablation | Matched | Case outcomes |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        outcomes = ", ".join(f"{case['case']} -> {case['actual']}" for case in row["cases"])
        lines.append(f"| `{row['ablation']}` | `{row['matched_expectations']}/{row['total_cases']}` | {outcomes} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
