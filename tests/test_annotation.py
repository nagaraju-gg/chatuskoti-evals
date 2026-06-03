from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chatuskoti_evals.annotation import AnnotationCase, AnnotationRecord, AnnotationStore, compute_agreement


class AnnotationTests(unittest.TestCase):
    def test_store_roundtrip(self) -> None:
        store = AnnotationStore()
        store.add(AnnotationRecord(case_id="case_1", annotator_id="rater_a", label="pyrrhic", confidence=0.9, notes=""))
        store.add(AnnotationRecord(case_id="case_2", annotator_id="rater_a", label="clean", confidence=1.0, notes=""))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            store.save(path)
            loaded = AnnotationStore.load(path)

        self.assertEqual(len(loaded.records), 2)
        self.assertEqual(loaded.records[0].label, "pyrrhic")
        self.assertEqual(loaded.records[1].annotator_id, "rater_a")

    def test_compute_agreement_full(self) -> None:
        store = AnnotationStore()
        store.add(AnnotationRecord("case_1", "rater_a", "pyrrhic", 1.0, ""))
        store.add(AnnotationRecord("case_1", "rater_b", "pyrrhic", 0.9, ""))
        store.add(AnnotationRecord("case_2", "rater_a", "clean", 1.0, ""))
        store.add(AnnotationRecord("case_2", "rater_b", "clean", 1.0, ""))

        vec3_labels = {"case_1": "hold", "case_2": "adopt"}
        result = compute_agreement(store, vec3_labels)

        self.assertEqual(result["total_cases"], 2)
        self.assertEqual(result["full_rater_agreement"], 2)

    def test_compute_agreement_disagreement(self) -> None:
        store = AnnotationStore()
        store.add(AnnotationRecord("case_1", "rater_a", "pyrrhic", 1.0, ""))
        store.add(AnnotationRecord("case_1", "rater_b", "clean", 0.8, ""))

        result = compute_agreement(store, {})
        self.assertEqual(result["full_rater_agreement"], 0)

    def test_annotation_case_dataclass(self) -> None:
        case = AnnotationCase(
            case_id="test_case",
            scenario_name="pyrrhic",
            primary_metric_delta=0.03,
            train_loss_delta=-0.1,
            val_loss_delta=0.2,
            train_val_gap=0.3,
            grad_norm_mean=2.0,
            grad_norm_std=0.5,
            proxy_metric_corr=0.8,
            raw_metric=0.67,
            baseline_metric=0.64,
        )
        self.assertEqual(case.case_id, "test_case")
        self.assertAlmostEqual(case.primary_metric_delta, 0.03)
        self.assertAlmostEqual(case.raw_metric - case.baseline_metric, 0.03)
