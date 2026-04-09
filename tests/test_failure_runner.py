from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from chatuskoti_evals.config import ExperimentConfig
from chatuskoti_evals.runner import run_ablation_bundle, run_calibration_bundle, run_failure_injection_set


class FailureRunnerTests(unittest.TestCase):
    def test_failure_runner_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "failure_run"
            log_path = Path(tempdir) / "logs" / "runs.jsonl"
            previous = os.environ.get("CHATUSKOTI_RUN_LOG_PATH")
            os.environ["CHATUSKOTI_RUN_LOG_PATH"] = str(log_path)
            try:
                results = run_failure_injection_set(root, ExperimentConfig(), seeds=1)
                self.assertTrue((root / "failure_injection" / "summary.md").exists())
                self.assertTrue((root / "manifest.json").exists())
                self.assertGreaterEqual(len(results), 1)
                self.assertTrue(all(result.binary_resolution.action in {"adopt", "reject"} for result in results))
                self.assertTrue(log_path.exists())
                records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(len(records), len(results))
                self.assertEqual(records[0]["schema_version"], 1)
                self.assertIn("axis_components", records[0])
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["package_version"], "1.2.0")
                self.assertIn("benchmark_spec_id", manifest)
            finally:
                if previous is None:
                    os.environ.pop("CHATUSKOTI_RUN_LOG_PATH", None)
                else:
                    os.environ["CHATUSKOTI_RUN_LOG_PATH"] = previous

    def test_ablation_bundle_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "ablation_run"
            summaries = run_ablation_bundle(root, ExperimentConfig(), seeds=1)
            self.assertEqual(len(summaries), 5)
            self.assertTrue((root / "summary.md").exists())
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "manifest.json").exists())

    def test_calibration_bundle_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "calibration_run"
            summaries = run_calibration_bundle(root, ExperimentConfig(), seeds=1)

            self.assertEqual(len(summaries), 9)
            self.assertTrue((root / "summary.md").exists())
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "threshold_sweep.svg").exists())
            self.assertTrue((root / "default" / "failure_injection" / "summary.md").exists())
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_kind"], "calibration_bundle")
            self.assertEqual(manifest["schema_version"], 2)
