from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from chatuskoti_evals.config import ExperimentConfig
from chatuskoti_evals.runner import run_comparison


class RunnerTests(unittest.TestCase):
    def test_comparison_produces_reports_and_vec3_beats_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "artifacts"
            log_path = Path(tempdir) / "logs" / "runs.jsonl"
            previous = os.environ.get("CHATUSKOTI_RUN_LOG_PATH")
            os.environ["CHATUSKOTI_RUN_LOG_PATH"] = str(log_path)
            try:
                results = run_comparison(root, ExperimentConfig())

                self.assertTrue((root / "vec3" / "summary.md").exists())
                self.assertTrue((root / "binary" / "summary.md").exists())
                self.assertTrue((root / "comparison.md").exists())

                self.assertGreater(results["vec3"].accepted_metric, results["binary"].accepted_metric)
                vec3_signals = {signal for entry in results["vec3"].history for signal in entry.run_score.fired_signals}
                self.assertIn("hyper_coherence", vec3_signals)
                records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(len(records), 8)
                self.assertEqual(records[0]["schema_version"], 1)
                self.assertIn("T", records[0])
                self.assertIn("R", records[0])
                self.assertIn("V", records[0])
                self.assertEqual(manifest["schema_version"], 2)
                self.assertEqual(manifest["package_version"], "1.2.0")
            finally:
                if previous is None:
                    os.environ.pop("CHATUSKOTI_RUN_LOG_PATH", None)
                else:
                    os.environ["CHATUSKOTI_RUN_LOG_PATH"] = previous

    def test_challenge_mode_highlights_structural_divergences(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "challenge_artifacts"
            results = run_comparison(root, ExperimentConfig(), mode="challenge")

            summary = json.loads((root / "comparison_summary.json").read_text(encoding="utf-8"))
            cases = (root / "challenge_cases.md").read_text(encoding="utf-8")

            self.assertEqual(summary["challenge_divergence_count"], 3)
            self.assertGreater(results["binary"].accepted_metric, results["vec3"].accepted_metric)
            self.assertIn("`pyrrhic_probe` | `adopt` | `hold`", cases)
            self.assertIn("`metric_gaming_probe` | `adopt` | `reframe`", cases)
            self.assertIn("`eval_tta` | `adopt` | `reframe`", cases)
