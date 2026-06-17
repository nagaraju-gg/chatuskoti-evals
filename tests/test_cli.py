from __future__ import annotations

import unittest
from pathlib import Path

from chatuskoti_evals.evaluation.runner import run_ablation_bundle, run_failure_injection_set


class CliTests(unittest.TestCase):
    def test_failure_injection_set_runs(self) -> None:
        output = Path("_test_output/failure_set")
        results = run_failure_injection_set(output, seeds=1)
        self.assertEqual(len(results), 4)
        self.assertTrue(output.exists())

    def test_ablation_bundle_runs(self) -> None:
        output = Path("_test_output/ablation")
        summaries = run_ablation_bundle(output, seeds=1)
        self.assertGreater(len(summaries), 0)
        self.assertTrue(output.exists())
