from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from chatuskoti_evals.core.config import ExperimentConfig
from chatuskoti_evals.evaluation.runner import run_failure_injection_set


class RunnerTests(unittest.TestCase):
    def test_failure_injection_produces_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "failure_set"
            log_path = Path(tempdir) / "logs" / "runs.jsonl"
            previous = os.environ.get("CHATUSKOTI_RUN_LOG_PATH")
            os.environ["CHATUSKOTI_RUN_LOG_PATH"] = str(log_path)
            try:
                results = run_failure_injection_set(root, ExperimentConfig(), seeds=1)
                self.assertGreaterEqual(len(results), 1)
                for result in results:
                    self.assertIsNotNone(result.resolution.action)
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["package_version"], "1.3.0")
                records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), len(results))
                self.assertIn("T", records[0])
                self.assertIn("R", records[0])
                self.assertIn("V", records[0])
            finally:
                if previous is None:
                    os.environ.pop("CHATUSKOTI_RUN_LOG_PATH", None)
                else:
                    os.environ["CHATUSKOTI_RUN_LOG_PATH"] = previous

    def test_failure_injection_splits_cases_by_vec3(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "failure_run"
            results = run_failure_injection_set(root, ExperimentConfig(), seeds=1)
            self.assertTrue(all(
                result.resolution.action in {"adopt", "hold", "reframe", "rollback", "keep_going"}
                for result in results
            ))
