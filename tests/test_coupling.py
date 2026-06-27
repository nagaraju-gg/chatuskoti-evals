from __future__ import annotations

import unittest

from chatuskoti_evals.core.coupling import (
    compute_coupling_over_history,
    detect_overlay_diagnostics,
    get_coupling_angle_history,
    sliding_window_coupling,
)
from chatuskoti_evals.core.models import RunScore, Vec3


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

    def test_sliding_window_returns_coupling_values(self) -> None:
        all_deltas = [
            {"step": i, "delta_T": 0.02, "delta_R": 0.01, "delta_V": -0.03}
            for i in range(1, 8)
        ]
        results = sliding_window_coupling(all_deltas, window=3)
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertIn("t_v_coupling", r)
            self.assertIn("t_r_coupling", r)
            self.assertIn("r_v_coupling", r)
            self.assertIn("t_v_angle", r)
            self.assertIn("t_r_angle", r)
            self.assertIn("r_v_angle", r)

    def test_sliding_window_no_negative_coupling_on_aligned_data(self) -> None:
        all_deltas = [
            {"step": i, "delta_T": 0.02, "delta_R": 0.01, "delta_V": 0.02}
            for i in range(1, 8)
        ]
        results = sliding_window_coupling(all_deltas, window=3)
        self.assertTrue(all(r["t_v_coupling"] >= 0 for r in results))
        self.assertTrue(all(r["t_r_coupling"] >= 0 for r in results))

    def test_short_history_returns_empty(self) -> None:
        scores = [make_score(0.0, 0.0, 0.0)]
        deltas = compute_coupling_over_history(scores)
        self.assertEqual(deltas, [])
        coupling = sliding_window_coupling([], window=5)
        self.assertEqual(coupling, [])

    def test_get_coupling_angle_history(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.0),
            make_score(0.1, 0.0, 0.0),
            make_score(0.2, -0.1, -0.1),
        ]
        history = get_coupling_angle_history(scores)
        self.assertEqual(len(history), 2)
        self.assertAlmostEqual(history[0]["tv_product"], 0.0)
        self.assertAlmostEqual(history[0]["tr_product"], 0.0)
        self.assertAlmostEqual(history[1]["tv_product"], -0.01)
        self.assertAlmostEqual(history[1]["tr_product"], -0.01)
        self.assertAlmostEqual(history[1]["rv_product"], 0.01)
        self.assertEqual(history[0]["step"], 1)
        self.assertEqual(history[1]["step"], 2)

    def test_get_coupling_angle_history_short(self) -> None:
        scores = [make_score(0.0, 0.0, 0.0)]
        history = get_coupling_angle_history(scores)
        self.assertEqual(history, [])

    def test_instrument_tradeoff_overlay_uses_paper_window_and_tau(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.0),
            make_score(0.0, 0.1, -0.1),
            make_score(0.0, 0.2, -0.2),
            make_score(0.0, 0.3, -0.3),
            make_score(0.0, 0.4, -0.4),
            make_score(0.0, 0.5, -0.5),
        ]
        diagnostics = detect_overlay_diagnostics(scores)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["diagnostic"], "instrument_tradeoff")
        self.assertEqual(diagnostics[0]["window"], 5)
        self.assertEqual(diagnostics[0]["tau"], 0.4)
        self.assertLessEqual(diagnostics[0]["coupling"], -0.4)

    def test_instrument_tradeoff_overlay_does_not_fire_on_aligned_rv(self) -> None:
        scores = [
            make_score(0.0, 0.0, 0.0),
            make_score(0.0, 0.1, 0.1),
            make_score(0.0, 0.2, 0.2),
            make_score(0.0, 0.3, 0.3),
            make_score(0.0, 0.4, 0.4),
            make_score(0.0, 0.5, 0.5),
        ]
        self.assertEqual(detect_overlay_diagnostics(scores), [])
