from __future__ import annotations

import unittest

from chatuskoti_evals.core.config import DetectorConfig, ExperimentConfig
from chatuskoti_evals.evaluation.actions import ACTION_INDEX
from chatuskoti_evals.evaluation.benchmark import SimulatedCIFAR100ResNet18Adapter
from chatuskoti_evals.evaluation.resolver import resolve_vec3
from chatuskoti_evals.evaluation.scenarios import FAILURE_INJECTION_SET
from chatuskoti_evals.evaluation.scoring import score_run_metrics


class ScenarioCatalogTests(unittest.TestCase):
    def test_failure_injection_set_matches_expected_detector_outcomes(self) -> None:
        experiment_cfg = ExperimentConfig()
        adapter = SimulatedCIFAR100ResNet18Adapter(experiment_cfg.simulation)
        baseline = adapter.record_baseline([0, 1, 2])
        detector_cfg = DetectorConfig()

        for scenario in FAILURE_INJECTION_SET:
            action = ACTION_INDEX[scenario.action_name]
            candidate_metrics, _ = adapter.execute(action, [0, 1, 2])
            run_score, _ = score_run_metrics(candidate_metrics, baseline.metrics, detector_cfg)
            resolution = resolve_vec3(run_score, detector_cfg)

            for signal in scenario.expected_signals:
                self.assertIn(signal, run_score.fired_signals, msg=scenario.name)
            self.assertEqual(resolution.action, scenario.expected_resolution, msg=scenario.name)
