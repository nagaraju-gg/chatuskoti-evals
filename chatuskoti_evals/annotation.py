from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnnotationCase:
    case_id: str
    scenario_name: str
    primary_metric_delta: float
    train_loss_delta: float
    val_loss_delta: float
    train_val_gap: float
    grad_norm_mean: float
    grad_norm_std: float
    proxy_metric_corr: float
    raw_metric: float
    baseline_metric: float


@dataclass(frozen=True)
class AnnotationRecord:
    case_id: str
    annotator_id: str
    label: str
    confidence: float
    notes: str


@dataclass
class AnnotationStore:
    records: list[AnnotationRecord] = field(default_factory=list)

    def add(self, record: AnnotationRecord) -> None:
        self.records.append(record)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                [
                    {
                        "case_id": r.case_id,
                        "annotator_id": r.annotator_id,
                        "label": r.label,
                        "confidence": r.confidence,
                        "notes": r.notes,
                    }
                    for r in self.records
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> AnnotationStore:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        store.records = [
            AnnotationRecord(
                case_id=item["case_id"],
                annotator_id=item["annotator_id"],
                label=item["label"],
                confidence=item.get("confidence", 1.0),
                notes=item.get("notes", ""),
            )
            for item in data
        ]
        return store


def extract_cases_to_csv(bundle_path: Path, output_path: Path) -> Path:
    results_path = bundle_path / "canonical_failure" / "failure_injection" / "failure_results.json"
    if not results_path.exists():
        results_path = bundle_path / "failure_injection" / "failure_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Could not find failure_results.json under {bundle_path}")

    data = json.loads(results_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for item in data:
        case_id = item.get("scenario_name", "unknown")
        raw_detectors = item.get("run_score", {}).get("raw_detectors", {})
        baseline_metric = _find_baseline_metric(bundle_path, results_path)
        candidate_metric = item.get("candidate_metric", 0.0)
        rows.append(
            {
                "case_id": case_id,
                "action": item.get("action_spec", {}).get("name", ""),
                "narrative": item.get("narrative", ""),
                "baseline_metric": baseline_metric,
                "candidate_metric": candidate_metric,
                "metric_delta": candidate_metric - baseline_metric,
                "train_loss_delta": raw_detectors.get("metric_delta", ""),
                "val_loss_delta": raw_detectors.get("val_loss_delta", ""),
                "train_val_gap": raw_detectors.get("gap_ratio", ""),
                "grad_norm_mean": raw_detectors.get("grad_mean_ratio", ""),
                "grad_norm_std": raw_detectors.get("grad_std_ratio", ""),
                "proxy_corr_delta": raw_detectors.get("proxy_corr_delta", ""),
                "weight_efficiency": raw_detectors.get("weight_efficiency", ""),
                "label": "",
                "confidence": "",
            }
        )

    # Also try to extract per-iteration cases from comparison/lead-time runs
    for subdir in ["challenge_compare", "vec3", "binary", "lead_time"]:
        history_path = bundle_path / subdir / "history.jsonl"
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if not entry.get("run_score"):
                    continue
                rs = entry["run_score"]
                case_id = f"{subdir}_iter{entry.get('iteration', 0)}"
                rows.append(
                    {
                        "case_id": case_id,
                        "action": entry.get("action_spec", {}).get("name", ""),
                        "narrative": "",
                        "baseline_metric": baseline_metric,
                        "candidate_metric": rs.get("raw_detectors", {}).get("metric_delta", 0),
                        "metric_delta": rs.get("raw_detectors", {}).get("metric_delta", 0),
                        "train_loss_delta": rs.get("raw_detectors", {}).get("val_loss_delta", ""),
                        "val_loss_delta": "",
                        "train_val_gap": rs.get("raw_detectors", {}).get("gap_ratio", ""),
                        "grad_norm_mean": rs.get("raw_detectors", {}).get("grad_mean_ratio", ""),
                        "grad_norm_std": rs.get("raw_detectors", {}).get("grad_std_ratio", ""),
                        "proxy_corr_delta": rs.get("raw_detectors", {}).get("proxy_corr_delta", ""),
                        "weight_efficiency": rs.get("raw_detectors", {}).get("weight_efficiency", ""),
                        "label": "",
                        "confidence": "",
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def _find_baseline_metric(bundle_path: Path, results_path: Path) -> float:
    summary_path = results_path.parent / "aggregate_summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return float(data.get("mean_primary_metric", 0.0))
        except (json.JSONDecodeError, ValueError):
            pass
    return 0.636


def compute_agreement(store: AnnotationStore, vec3_labels: dict[str, str]) -> dict[str, Any]:
    from collections import Counter

    by_case: dict[str, list[AnnotationRecord]] = {}
    for record in store.records:
        by_case.setdefault(record.case_id, []).append(record)

    rater_agreements: list[bool] = []
    vec3_matches: list[bool] = []
    consensus_labels: dict[str, str] = {}

    for case_id, recs in by_case.items():
        labels = Counter(r.label for r in recs)
        most_common = labels.most_common(1)[0][0]
        consensus_labels[case_id] = most_common
        rater_agreements.append(len(set(r.label for r in recs)) == 1)
        if case_id in vec3_labels:
            vec3_matches.append(vec3_labels[case_id] == most_common)

    total = len(by_case)
    return {
        "total_cases": total,
        "full_rater_agreement": sum(rater_agreements),
        "rater_agreement_pct": round(sum(rater_agreements) / max(total, 1) * 100, 1),
        "vec3_matches": sum(vec3_matches),
        "vec3_accuracy_pct": round(sum(vec3_matches) / max(len(vec3_matches), 1) * 100, 1),
        "consensus_labels": consensus_labels,
    }
