from __future__ import annotations

import unittest

from chatuskoti_evals.coupling import (
    compute_coupling_over_history,
    find_goodhart_precheck_step,
    measure_lead_time,
    sliding_window_coupling,
)
from chatuskoti_evals.models import RunScore, Vec3


def make_score(t: float, r: float, v: float, signals: list[str] | None = None) -> RunScore:
    return RunScore(
        mean=Vec3(truthness=t, reliability=r, validity=v),
        std=Vec3(truthness=0.0, reliability=0.0, validity=0.0),
        mag=1.0,
        spread=0.0,
        fired_signals=signals or [],
        raw_detectors={},
        axis_components={},
    )


class CouplingTests(unittest.TestCase):
    def test_compute_deltas_from_history(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.0),
            make_score(0.1, 0.0, 0.0),
            make_score(0.2, -0.1, -0.1),
        ]
        deltas = compute_coupling_over_history(scores)
        self.assertEqual(len(deltas), 2)
        self.assertAlmostEqual(deltas[0]["delta_T"], 0.1)
        self.assertAlmostEqual(deltas[1]["delta_T"], 0.1)
        self.assertAlmostEqual(deltas[1]["delta_V"], -0.1)

    def test_sliding_window_anti_coupling_detects_goodhart(self) -> None:
        all_deltas = [
            {"step": i, "delta_T": 0.02, "delta_R": 0.01, "delta_V": -0.03}
            for i in range(1, 8)
        ]
        results = sliding_window_coupling(all_deltas, window=3, tau=0.3)
        self.assertTrue(len(results) > 0)
        self.assertTrue(any(r["goodhart_warning"] for r in results))

    def test_sliding_window_no_warning_when_coupled_positive(self) -> None:
        all_deltas = [
            {"step": i, "delta_T": 0.02, "delta_R": 0.01, "delta_V": 0.02}
            for i in range(1, 8)
        ]
        results = sliding_window_coupling(all_deltas, window=3, tau=0.3)
        self.assertTrue(all(not r["goodhart_warning"] for r in results))
        self.assertTrue(all(not r["pyrrhic_warning"] for r in results))

    def test_find_goodhart_precheck_step(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.5),
            make_score(0.2, 0.1, 0.4, signals=["proxy_decoupling"]),
            make_score(0.4, 0.2, 0.1, signals=["hyper_coherence", "proxy_decoupling"]),
        ]
        step = find_goodhart_precheck_step(scores)
        self.assertEqual(step, 2)

    def test_find_goodhart_precheck_returns_none(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.5),
            make_score(0.2, 0.1, 0.5),
        ]
        step = find_goodhart_precheck_step(scores)
        self.assertIsNone(step)

    def test_measure_lead_time_positive(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.5),
            make_score(0.02, 0.01, 0.48),
            make_score(0.04, 0.01, 0.45),
            make_score(0.06, 0.0, 0.40),
            make_score(0.08, -0.01, 0.35),
            make_score(0.10, -0.02, 0.28),
            make_score(0.12, -0.03, 0.20),
            make_score(0.14, -0.04, 0.10, signals=["hyper_coherence", "proxy_decoupling"]),
        ]
        result = measure_lead_time(scores, window=3, tau=0.0)
        self.assertTrue(result["goodhart_warning_fired"])
        self.assertTrue(result["goodhart_precheck_fired"])
        self.assertIsNotNone(result["lead_time_steps"])
        self.assertGreaterEqual(result["lead_time_steps"], 0)

    def test_measure_lead_time_no_warning(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.5),
            make_score(0.02, 0.01, 0.52),
            make_score(0.04, 0.01, 0.53),
        ]
        result = measure_lead_time(scores, window=3, tau=0.4)
        self.assertFalse(result["goodhart_warning_fired"])

    def test_short_history_returns_empty(self) -> None:
        scores = [make_score(0.0, 0.0, 0.0)]
        deltas = compute_coupling_over_history(scores)
        self.assertEqual(deltas, [])
        coupling = sliding_window_coupling([], window=5, tau=0.4)
        self.assertEqual(coupling, [])
