from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chatuskoti_evals.core.models import RunScore, Vec3
from chatuskoti_evals.core.wisdom import WisdomStore


class WisdomTests(unittest.TestCase):
    def test_update_predict_and_persist(self) -> None:
        store = WisdomStore()
        score = RunScore(
            mean=Vec3(0.6, 0.5, 0.7),
            std=Vec3(0.0, 0.0, 0.0),
            mag=1.04,
            spread=0.0,
            fired_signals=["clean_win"],
            raw_detectors={},
            axis_components={"reliability": {}, "validity": {}},
        )
        store.update("regularization.stochastic_depth", score)
        store.update("regularization.stochastic_depth", score)

        predicted = store.predict("regularization.stochastic_depth")
        self.assertAlmostEqual(predicted.truthness, 0.6)
        self.assertAlmostEqual(predicted.reliability, 0.5)
        self.assertIn("regularization.stochastic_depth", store.confident_families())

        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "wisdom.json"
            store.save(path)
            loaded = WisdomStore.load(path)
            self.assertEqual(loaded.snapshot()["regularization.stochastic_depth"]["count"], 2)
