from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from chatuskoti_evals.core.config import ExperimentConfig, LoopConfig
from chatuskoti_evals.evaluation.runner import run_ablation_bundle, run_failure_injection_set, run_single_loop


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
                self.assertEqual(manifest["package_version"], "1.3.0")
            finally:
                if previous is None:
                    os.environ.pop("CHATUSKOTI_RUN_LOG_PATH", None)
                else:
                    os.environ["CHATUSKOTI_RUN_LOG_PATH"] = previous

    def test_ablation_bundle_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "ablation_run"
            summaries = run_ablation_bundle(root, ExperimentConfig(), seeds=1)
            self.assertEqual(len(summaries), 9)
            self.assertTrue((root / "summary.md").exists())
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "manifest.json").exists())

    def test_run_single_loop_vec3(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "single_loop"
            loop_cfg = LoopConfig(controller="vec3", max_iterations=1, n_seeds=1)
            result = run_single_loop(ExperimentConfig(), loop_cfg, root)
            self.assertEqual(result.controller, "vec3")
            self.assertEqual(len(result.history), 1)
            self.assertTrue((root / "vec3" / "history.jsonl").exists())
            self.assertTrue((root / "vec3" / "summary.md").exists())

    def test_run_single_loop_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "single_loop_binary"
            loop_cfg = LoopConfig(controller="binary", max_iterations=1, n_seeds=1)
            result = run_single_loop(ExperimentConfig(), loop_cfg, root)
            self.assertEqual(result.controller, "binary")
            self.assertEqual(len(result.history), 1)
