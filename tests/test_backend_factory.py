from __future__ import annotations

import unittest

from chatuskoti_evals.core.config import ExperimentConfig
from chatuskoti_evals.evaluation.benchmark import SimulatedCIFAR100ResNet18Adapter, create_benchmark_adapter


class BackendFactoryTests(unittest.TestCase):
    def test_simulator_backend_factory(self) -> None:
        adapter = create_benchmark_adapter(ExperimentConfig())
        self.assertIsInstance(adapter, SimulatedCIFAR100ResNet18Adapter)

